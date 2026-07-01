# app.py - TPL Comment Extractor (multi-provider, resilient, token-aware)
#
# Setup:
#   pip install streamlit aiohttp openpyxl pdfplumber
#   (optional OCR) pip install pytesseract pillow  + Tesseract binary
#   streamlit run app.py
#
# Single-file Streamlit app. Upload submittal ZIPs -> reviewer comments are
# extracted by an LLM -> written to the exact TPL_Comments.xlsx format.

import streamlit as st
import zipfile
import io
import pdfplumber
import asyncio
import aiohttp
import json
import re
import time
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime
import tempfile
import os
import gc

try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

st.set_page_config(page_title="TPL Comment Extractor", layout="wide")

# ============================================================================
# CONFIG
# ============================================================================
CONCURRENCY = 2

# --- Request-sizing / retry knobs (safe defaults for free tiers) ------------
CHAR_BUDGET       = 24000   # max chars of PDF text per request (pre-split)
MODEL_CHAR_LIMIT  = 48000   # hard ceiling; above this we split into chunks
MAX_RETRIES       = 4       # attempts per provider before falling back
BACKOFF_BASE      = 2.0     # exponential backoff base (seconds): 2,4,8,...
BACKOFF_CAP       = 30.0    # never wait more than this between retries
REQUEST_TIMEOUT   = 90      # seconds per HTTP request
CHARS_PER_TOKEN   = 4.0     # rough token estimate (≈4 chars/token English)

# ----------------------------------------------------------------------------
# SUPPORTED APIS — add a new LLM service by adding an entry here.
#   url           : the chat/completions endpoint
#   format        : "openai" (Groq, OpenAI, Mistral, DeepSeek, Gemini-compat,
#                   Together...) or "anthropic" (Claude)
#   models        : list of model IDs (first is the dropdown default)
#   extra_headers : (optional) dict of extra headers, e.g. Claude's version
# ----------------------------------------------------------------------------
SUPPORTED_APIS = {
    "groq": {
        "url": "https://api.groq.com/openai/v1/chat/completions",
        "format": "openai",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    },
    "openai": {
        "url": "https://api.openai.com/v1/chat/completions",
        "format": "openai",
        "models": ["gpt-4o", "gpt-4-turbo", "gpt-4o-mini"],
    },
    "claude": {
        "url": "https://api.anthropic.com/v1/messages",
        "format": "anthropic",
        "models": ["claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5"],
        "extra_headers": {"anthropic-version": "2023-06-01"},
    },
    "gemini": {
        "url": "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        "format": "openai",
        "models": ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"],
    },
    "mistral": {
        "url": "https://api.mistral.ai/v1/chat/completions",
        "format": "openai",
        "models": ["mistral-large-latest", "mistral-small-latest"],
    },
    "deepseek": {
        "url": "https://api.deepseek.com/v1/chat/completions",
        "format": "openai",
        "models": ["deepseek-chat"],
    },
    "together": {
        "url": "https://api.together.ai/v1/chat/completions",
        "format": "openai",
        "models": ["meta-llama/Llama-3-70b-chat-hf", "openai/gpt-oss-20b", "mistralai/Mistral-7B-Instruct-v0.1"],
    },
}

# Which HTTP statuses are worth retrying (transient). 429 = rate limit;
# 5xx = server hiccup. 4xx (except 429) are permanent -> don't waste retries.
RETRYABLE_STATUS = {429, 500, 502, 503, 504}


# ============================================================================
# PROVIDER POOL — key-level scheduling
# ============================================================================
# Instead of one key per provider with provider-level retries, we keep a POOL
# of keys. Each key tracks its own stats + cooldown. The scheduler hands out
# the next available key; a key that hits 429 is put on cooldown (not retried
# repeatedly) so the very next request uses a DIFFERENT key. Only when every
# key across every provider is cooling down do we wait for the soonest one.
#
# Scheduling priority (as specified):
#   available Gemini keys -> available Groq keys ->
#   cooled-down Gemini keys -> cooled-down Groq keys
# "available" = no active cooldown. Within a provider, keys are tried in the
# order the user entered them (round-robin so load spreads across a provider).

# Provider priority order for scheduling. Providers not listed here still work
# (they fall in after these, in dict order), keeping the abstraction general.
PROVIDER_PRIORITY = ["gemini", "groq"]


class ApiKey:
    """One API key with its own independent stats and cooldown."""
    def __init__(self, provider, key, label):
        self.provider = provider
        self.key = key
        self.label = label            # e.g. "gemini#1" for analytics
        # per-key stats (requirement 8)
        self.retry_count = 0          # times this key was put on cooldown
        self.cooldown_until = 0.0     # epoch seconds; 0 = available now
        self.last_request_time = 0.0
        self.total_requests = 0
        self.total_failures = 0
        self.total_429 = 0
        self._429_streak = 0          # consecutive 429s -> grows backoff

    def available(self, now):
        return now >= self.cooldown_until

    def cooldown_remaining(self, now):
        return max(0.0, self.cooldown_until - now)

    def mark_429(self, now):
        """Put this key on exponential-backoff cooldown. Doesn't retry it now."""
        self.total_429 += 1
        self.total_failures += 1
        self.retry_count += 1
        self._429_streak += 1
        wait = min(BACKOFF_BASE ** self._429_streak, BACKOFF_CAP)
        self.cooldown_until = now + wait

    def mark_other_failure(self, now):
        """Non-429 failure (5xx/timeout): short cooldown, smaller penalty."""
        self.total_failures += 1
        self.retry_count += 1
        wait = min(BACKOFF_BASE ** 1, BACKOFF_CAP)
        self.cooldown_until = now + wait

    def mark_success(self):
        self._429_streak = 0          # reset backoff growth on success

    def stats_row(self):
        return {
            "key": self.label,
            "provider": self.provider,
            "total_requests": self.total_requests,
            "total_failures": self.total_failures,
            "total_429": self.total_429,
            "retry_count": self.retry_count,
        }


class KeyPool:
    """Holds every configured key and hands out the next one to use.

    provider_key_lists: {provider: [key1, key2, ...]} in the order entered.
    Backward compatible: a single gemini key + single groq key behaves exactly
    like the old primary+fallback flow (gemini used until it 429s, then groq)."""

    def __init__(self, provider_key_lists):
        self.keys = []
        for provider, key_list in provider_key_lists.items():
            for i, k in enumerate(key_list, 1):
                if k and k.strip():
                    self.keys.append(ApiKey(provider, k.strip(), f"{provider}#{i}"))
        # round-robin cursor per provider so we spread load across a provider's
        # keys instead of always hammering its first key.
        self._rr = {}

    def has_any(self):
        return len(self.keys) > 0

    def _provider_rank(self, provider):
        if provider in PROVIDER_PRIORITY:
            return PROVIDER_PRIORITY.index(provider)
        return len(PROVIDER_PRIORITY)  # unlisted providers come after

    def acquire(self, now):
        """Return (api_key, wait_seconds).
        - If a key is available now: (key, 0.0).
        - If all keys are cooling down: (soonest_key, seconds_to_wait) so the
          caller can sleep then use it (requirement 6: retry oldest exhausted).
        - If pool is empty: (None, 0.0)."""
        if not self.keys:
            return None, 0.0

        # 1) Prefer any AVAILABLE key, ordered by provider priority then a
        #    round-robin tiebreak within the pool.
        available = [k for k in self.keys if k.available(now)]
        if available:
            def sort_key(k):
                rr = self._rr.get(k.provider, 0)
                idx = self.keys.index(k)
                return (self._provider_rank(k.provider),
                        (idx - rr) % len(self.keys))
            available.sort(key=sort_key)
            chosen = available[0]
            self._rr[chosen.provider] = self.keys.index(chosen) + 1
            return chosen, 0.0

        # 2) Nothing available -> pick the key whose cooldown expires SOONEST.
        #    Tie-break by provider priority. This implements "retry the oldest
        #    exhausted key after backoff", preferring Gemini then Groq.
        soonest = min(
            self.keys,
            key=lambda k: (k.cooldown_until, self._provider_rank(k.provider))
        )
        return soonest, soonest.cooldown_remaining(now)

    def stats(self):
        return [k.stats_row() for k in self.keys]


# ============================================================================
# ANALYTICS — one row per HTTP request, aggregated into a summary at the end
# ============================================================================
class Analytics:
    """Thread/async-safe-enough for our single-loop use. Records every request."""
    def __init__(self):
        self.rows = []
        self.request_counter = 0

    def next_request_no(self):
        self.request_counter += 1
        return self.request_counter

    def record(self, **row):
        self.rows.append(row)

    def summary(self, total_zips, ok_zips, fail_zips, wall_seconds):
        calls      = len(self.rows)
        in_tokens  = sum(r.get("est_input_tokens", 0)  for r in self.rows)
        out_tokens = sum(r.get("est_output_tokens", 0) for r in self.rows)
        chars_sent = [r.get("chars_sent", 0) for r in self.rows]
        avg_size   = (sum(chars_sent) / len(chars_sent)) if chars_sent else 0
        largest    = max(chars_sent) if chars_sent else 0
        err_429    = sum(1 for r in self.rows if r.get("status") == 429)
        retries    = sum(r.get("retries", 0) for r in self.rows)
        fallbacks  = sum(1 for r in self.rows if r.get("fallback_used"))
        return {
            "Total ZIPs": total_zips,
            "Successful ZIPs": ok_zips,
            "Failed ZIPs": fail_zips,
            "Total API Calls": calls,
            "Total Estimated Input Tokens": in_tokens,
            "Total Estimated Output Tokens": out_tokens,
            "Average Request Size (chars)": round(avg_size),
            "Largest Request (chars)": largest,
            "429 Errors": err_429,
            "Retries": retries,
            "Fallbacks": fallbacks,
            "Total Processing Time (s)": round(wall_seconds, 1),
        }

    def per_key_table(self, key_pool):
        """Per-key analytics (requirement 9). Merges pool stats with request
        rows so each key shows its own request count, failures, 429s, latency."""
        # aggregate latency per key label from the request rows
        by_key = {}
        for r in self.rows:
            lbl = r.get("key_label")
            if lbl is None:
                continue
            d = by_key.setdefault(lbl, {"latency_sum": 0.0, "calls": 0})
            d["latency_sum"] += r.get("latency", 0.0)
            d["calls"] += 1
        table = []
        for ks in key_pool.stats():
            lbl = ks["key"]
            agg = by_key.get(lbl, {"latency_sum": 0.0, "calls": 0})
            avg_lat = (agg["latency_sum"] / agg["calls"]) if agg["calls"] else 0.0
            table.append({
                "Key": lbl,
                "Provider": ks["provider"],
                "Total Requests": ks["total_requests"],
                "Total Failures": ks["total_failures"],
                "Total 429": ks["total_429"],
                "Retry Count": ks["retry_count"],
                "Avg Latency (s)": round(avg_lat, 2),
            })
        return table


def est_tokens(text):
    """Rough token estimate. Good enough for staying under provider limits."""
    return int(len(text) / CHARS_PER_TOKEN) + 1


# ============================================================================
# PDF / ZIP HELPERS
# ============================================================================
def submittal_id_from_name(zip_name):
    """SUB1715-_ Rev No_ 0.zip  ->  SUB1715"""
    m = re.search(r'(SUB\d+)', zip_name, re.I)
    return m.group(1).upper() if m else zip_name.replace('.zip', '')


def extract_pdf_text(zip_bytes):
    """Pull text from the COMMENT-bearing PDF inside the ZIP using pdfplumber.

    Reviewer comments live only in an annotated markup PDF ('*_annotated*') or a
    reviewer 'Response' PDF. The plain datasheet and the submittal lead sheet
    contain NO reviewer comments — reading them makes the LLM invent "comments"
    out of spec-sheet text (false positives). So:
      - If a comment file (annotated/response) exists, read only that.
      - If NONE exists, the submittal has no reviewer comments: return empty.

    Falls back to OCR if a page has images but no extractable text.

    Returns (pages_text, unreadable_pages, had_comment_file).
    """
    pages_text = []
    unreadable_pages = 0
    had_comment_file = False
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            all_pdfs = [f for f in zf.namelist() if f.lower().endswith('.pdf')]
            comment_pdfs = [f for f in all_pdfs
                            if 'annotated' in f.lower() or 'response' in f.lower()]
            if not comment_pdfs:
                return [], 0, False
            had_comment_file = True
            for name in comment_pdfs:
                try:
                    raw = zf.read(name)
                    with pdfplumber.open(io.BytesIO(raw)) as pdf:
                        for page in pdf.pages:
                            t = page.extract_text() or ""
                            has_images = False
                            try:
                                has_images = len(page.images) > 0
                            except Exception:
                                pass

                            # OCR fallback: page has images but no real text
                            if len(t.strip()) < 50 and has_images and OCR_AVAILABLE:
                                try:
                                    img = page.to_image()
                                    ocr_text = pytesseract.image_to_string(img.original)
                                    if len(ocr_text.strip()) > len(t.strip()):
                                        t = ocr_text
                                except Exception:
                                    pass

                            if t.strip():
                                pages_text.append(t)
                            if len(t.strip()) < 50 and has_images:
                                unreadable_pages += 1
                except Exception:
                    pass
    except Exception:
        pass
    return pages_text, unreadable_pages, had_comment_file


# --- Document name extraction ------------------------------------------------
# The Document column needs "{DOC_NO}_{Title}". Different PDFs in the ZIP label
# these differently, so we try several patterns and search EVERY pdf until we
# have both a number and a title (or the best partial we can find).

DOC_NO_RE = re.compile(r'\b(TPL-[A-Z0-9]+-\d+-[A-Z]+-[A-Z]+-\d+)\b', re.I)

# Title labels seen across annotated / response / datasheet / lead sheets.
TITLE_LABEL_RE = re.compile(
    r'(?:Document\s*Title|Doc\.?\s*Title|Title|Subject|Description)\s*[:\-]\s*'
    r'(.+?)'
    r'(?:\s*(?:DOC\s*NO|Doc\.?\s*No|Document\s*No|Rev\b|Revision|Sheet\b|Page\b|$))',
    re.I | re.S
)


def _clean_title(raw):
    t = re.sub(r'\s+', ' ', raw).strip(" :-\t\r\n")
    t = re.sub(r'(\w)-\s+(\w)', r'\1-\2', t)   # rejoin hyphen-split words
    # Drop trailing submittal/project boilerplate that isn't part of the title.
    t = re.split(r'\b(?:Submittal\s*Number|Submittal\s*No|Project\b|Dholera)\b',
                 t, flags=re.I)[0].strip(" :-.")
    return t


def extract_doc_name(pages_text):
    """Build '{DOC_NO}_{Title}' from the first few pages of one PDF."""
    full = "\n".join(pages_text[:3])
    doc_no = None
    m = DOC_NO_RE.search(full)
    if m:
        doc_no = m.group(1)
    title = None
    mt = TITLE_LABEL_RE.search(full)
    if mt:
        cand = _clean_title(mt.group(1))
        # guard against grabbing a whole paragraph
        if 0 < len(cand) <= 120:
            title = cand
    if doc_no and title:
        return f"{doc_no}_{title}"
    return doc_no or title or ""


def extract_doc_name_from_zip(zip_bytes):
    """Search EVERY PDF in the ZIP for the document name.

    Strategy:
      1. Try each PDF; collect the best doc_no and best title found anywhere.
      2. Prefer a result that has BOTH number and title.
      3. Fall back to whichever single piece exists.
    The column only stays blank if NO pdf yields either piece.
    """
    best_combined = ""      # has both no + title
    best_doc_no = ""
    best_title = ""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            all_pdfs = [f for f in zf.namelist() if f.lower().endswith('.pdf')]
            # Order: annotated/response first (usually have the header block),
            # then everything else (datasheet, lead sheet).
            def rank(n):
                nl = n.lower()
                if 'annotated' in nl or 'response' in nl:
                    return 0
                return 1
            for name in sorted(all_pdfs, key=rank):
                try:
                    raw = zf.read(name)
                    with pdfplumber.open(io.BytesIO(raw)) as pdf:
                        pages = []
                        for p in pdf.pages[:3]:
                            t = p.extract_text() or ""
                            if t.strip():
                                pages.append(t)
                        full = "\n".join(pages[:3])

                        if not best_doc_no:
                            m = DOC_NO_RE.search(full)
                            if m:
                                best_doc_no = m.group(1)
                        if not best_title:
                            mt = TITLE_LABEL_RE.search(full)
                            if mt:
                                cand = _clean_title(mt.group(1))
                                if 0 < len(cand) <= 120:
                                    best_title = cand

                        if best_doc_no and best_title:
                            best_combined = f"{best_doc_no}_{best_title}"
                            return best_combined
                except Exception:
                    pass
    except Exception:
        pass
    return best_combined or best_doc_no or best_title or ""


# ============================================================================
# TOKEN REDUCTION — remove repeated boilerplate WITHOUT touching unique content
# ============================================================================
# Reviewer comments are, by definition, the lines that vary between pages.
# Headers / footers / revision blocks / spec boilerplate repeat verbatim on
# many pages. We drop a line only when the SAME normalised line appears on
# multiple pages. Anything unique (i.e. every real comment) is always kept.

def _norm_line(line):
    """Normalise for repetition detection: collapse spaces, strip page numbers."""
    s = re.sub(r'\s+', ' ', line).strip()
    s = re.sub(r'\bpage\s*\d+(\s*of\s*\d+)?\b', '', s, flags=re.I)
    s = re.sub(r'\b\d{1,2}/\d{1,2}/\d{2,4}\b', '', s)   # dates
    return s.lower().strip()


def dedupe_boilerplate(pages_text):
    """Drop lines that repeat across MULTIPLE pages (headers/footers/boilerplate).

    Keeps:
      - every line that appears on only one page (unique content, incl. comments)
      - one copy is NOT kept for repeated lines: repeated == boilerplate here,
        because genuine reviewer comments do not repeat verbatim across pages.
    Safety: if a page would be emptied entirely, keep it as-is (avoid nuking a
    page whose comment happens to resemble a header)."""
    if len(pages_text) < 2:
        return pages_text  # nothing repeats across a single page

    # Count how many pages each normalised line appears on.
    from collections import defaultdict
    page_count = defaultdict(int)
    per_page_lines = []
    for t in pages_text:
        lines = t.split("\n")
        per_page_lines.append(lines)
        seen_here = set()
        for ln in lines:
            n = _norm_line(ln)
            if n and n not in seen_here:
                seen_here.add(n)
                page_count[n] += 1

    # A line is boilerplate if it shows up on >= 2 pages AND is short-ish
    # (real multi-page comment paragraphs are rare; headers/footers are short).
    def is_boiler(nline):
        return page_count.get(nline, 0) >= 2 and len(nline) <= 120

    cleaned_pages = []
    for lines in per_page_lines:
        kept = [ln for ln in lines if not is_boiler(_norm_line(ln))]
        # Safety net: never let dedupe empty a page completely.
        if not any(l.strip() for l in kept):
            kept = lines
        cleaned_pages.append("\n".join(kept))
    return cleaned_pages


# Words/phrases that signal a page carries reviewer comments rather than
# pure spec-sheet boilerplate. Used to prioritise pages if a doc is too long.
COMMENT_SIGNALS = re.compile(
    r'\b(shall|should|to be|required|ensure|provide|verify|comply|complian'
    r'|approved|vendor|deviation|clarification|as per|not acceptable'
    r'|to be furnished|to be considered|consult|review)\b',
    re.I
)


def prioritise_pages(pages_text, char_budget=CHAR_BUDGET):
    """Keep pages in DOCUMENT ORDER. Only drop lowest-signal pages if over budget."""
    total = sum(len(t) for t in pages_text)
    if total <= char_budget:
        return "\n".join(pages_text)
    scored = []
    for i, t in enumerate(pages_text):
        score = len(COMMENT_SIGNALS.findall(t))
        scored.append((score, i, t))
    scored.sort(key=lambda x: (-x[0], x[1]))
    keep_idx, used = set(), 0
    for score, i, t in scored:
        if used + len(t) > char_budget:
            continue
        keep_idx.add(i)
        used += len(t)
    kept = [pages_text[i] for i in range(len(pages_text)) if i in keep_idx]
    return "\n".join(kept)


def split_oversized(text, limit=MODEL_CHAR_LIMIT):
    """If a request is still too big after prioritisation, split on page
    boundaries into <=limit chunks. Returns a list of chunk strings.
    Splitting on '\n' keeps whole lines (never cuts a comment mid-sentence)."""
    if len(text) <= limit:
        return [text]
    chunks, cur, cur_len = [], [], 0
    for line in text.split("\n"):
        add = len(line) + 1
        if cur_len + add > limit and cur:
            chunks.append("\n".join(cur))
            cur, cur_len = [], 0
        cur.append(line)
        cur_len += add
    if cur:
        chunks.append("\n".join(cur))
    return chunks


# ============================================================================
# COMMENT SPLITTER / CLEANERS  (unchanged behaviour)
# ============================================================================
def split_merged_comments(comments_list):
    result = []
    for comment in comments_list:
        parts = re.split(r'(?:(?<=\s)|^)\d{1,2}\)\s+', comment)
        parts = [p.strip(' .') for p in parts if p.strip(' .')]
        if len(parts) > 1:
            result.extend(parts)
        else:
            result.append(comment.strip())
    return result


HEADER_PATTERNS = re.compile(
    r'^(clarification|clarifications required|following points|following items'
    r'|please incorporate|please note the following|comments?:?$)',
    re.I
)


def drop_headers(comments_list):
    out = []
    for c in comments_list:
        if HEADER_PATTERNS.match(c.strip()):
            continue
        out.append(c)
    return out


LEADIN = re.compile(
    r'^(clarifications?\s+required[^:]*:\s*'
    r'|following\s+(?:points|items)[^:]*:\s*'
    r'|please\s+incorporate[^:]*:\s*'
    r'|please\s+note[^:]*:\s*)',
    re.I
)


def strip_leadin(text):
    return LEADIN.sub('', text).strip()


# ============================================================================
# LLM CALL  (prompt wording unchanged)
# ============================================================================
PROMPT_TEMPLATE = """You are extracting reviewer comments from an engineering submittal review.

Below is the text of a submittal PDF. Reviewers have added comments/markups requesting clarifications or changes.

Extract EVERY distinct reviewer comment as its own separate item, in the order they appear in the text.

STRICT RULES:
- Each comment = ONE distinct instruction, requirement, or question.
- If several requirements are written together (numbered 1) 2) 3) OR just run together in a block), SPLIT them into separate items. Never combine two different requirements into one item.
- Do NOT include section headers or lead-ins such as "Clarifications required for the following points", "Following items to be mentioned", "Please incorporate following detail". Skip those; only output the actual items under them.
- Do NOT include document body text, specifications tables, titles, or boilerplate. Only the reviewer's added comments.
- Keep each comment's wording complete and faithful. Do not summarise or shorten.
- Output ONLY valid JSON, no commentary, in exactly this form:
{{"comments": ["first comment", "second comment", "third comment"]}}

SUBMITTAL TEXT:
{body}

JSON:"""


def _headers_and_payload(cfg, api_key, model, prompt):
    """Build (headers, payload) for a provider from its config."""
    fmt = cfg.get("format", "openai")
    headers = {"Content-Type": "application/json"}
    if fmt == "anthropic":
        headers["x-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    for k, v in cfg.get("extra_headers", {}).items():
        headers[k] = v

    if fmt == "anthropic":
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096,
        }
    return headers, payload


def _extract_content(fmt, data):
    """Pull the text content out of a provider's JSON response."""
    if fmt == "anthropic":
        return data["content"][0]["text"]
    return data["choices"][0]["message"]["content"]


async def _single_call(session, cfg, api_key, model, prompt):
    """One HTTP attempt. Returns (content, error, status). Exactly one of
    content/error is non-None. status is the HTTP code (or None on exception)."""
    url = cfg["url"]
    fmt = cfg.get("format", "openai")
    headers, payload = _headers_and_payload(cfg, api_key, model, prompt)
    try:
        async with session.post(url, json=payload, headers=headers,
                                timeout=REQUEST_TIMEOUT) as resp:
            status = resp.status
            if status == 200:
                data = await resp.json()
                try:
                    return _extract_content(fmt, data), None, status
                except (KeyError, IndexError, TypeError) as e:
                    return None, f"bad response shape: {str(e)[:60]}", status
            txt = await resp.text()
            return None, f"HTTP {status}: {txt[:160]}", status
    except asyncio.TimeoutError:
        return None, "timeout", None
    except Exception as e:
        return None, str(e)[:100], None


async def call_with_key_pool(session, prompt, model, primary_provider,
                             key_pool, analytics, sub_id):
    """Send one prompt using the KEY POOL scheduler (replaces provider-level
    retries with key-level scheduling — requirement 10).

    Loop:
      1. Ask the pool for the next key (available first; else soonest-cooldown).
      2. If the chosen key is cooling down, sleep exactly until it's ready
         (this only happens when EVERY key is exhausted — requirement 6/7).
      3. Fire ONE request on that key.
         - 200  -> success, return.
         - 429  -> mark ONLY that key on cooldown, immediately try next key
                   (requirement 3/4). Never hammer the same key.
         - 5xx/timeout -> short cooldown on that key, try next.
         - 4xx (auth/bad) -> that key is broken; long cooldown so we stop using
           it, move on to others.
      4. Give up only after a bounded number of scheduling attempts with no
         success (avoids an infinite loop if all keys are permanently bad).

    The `model` is used when the chosen key's provider == primary_provider;
    otherwise the provider's own default model is used (so a Groq fallback key
    doesn't get sent a Gemini model id).
    """
    if not key_pool.has_any():
        return None, "no API keys configured"

    est_in = est_tokens(prompt)
    # Bound total scheduling attempts: enough to let every key try a few times.
    max_attempts = max(MAX_RETRIES + 1, len(key_pool.keys) * (MAX_RETRIES + 1))
    tried_notes = []

    for _attempt in range(max_attempts):
        now = time.time()
        api_key, wait = key_pool.acquire(now)
        if api_key is None:
            return None, "no API keys configured"

        # Every key is cooling down -> wait for the soonest (requirement 6/7).
        if wait > 0:
            await asyncio.sleep(min(wait, BACKOFF_CAP))
            now = time.time()

        cfg = SUPPORTED_APIS.get(api_key.provider)
        if cfg is None:
            api_key.mark_other_failure(now)
            continue

        # model: primary provider uses the user-picked model; others use default
        use_model = model if api_key.provider == primary_provider else cfg["models"][0]
        is_fallback = api_key.provider != primary_provider

        api_key.total_requests += 1
        api_key.last_request_time = now

        t0 = time.time()
        content, err, status = await _single_call(session, cfg, api_key.key,
                                                  use_model, prompt)
        latency = time.time() - t0

        analytics.record(
            provider=api_key.provider,
            key_label=api_key.label,
            model=use_model,
            request_no=analytics.next_request_no(),
            chars_sent=len(prompt),
            est_input_tokens=est_in,
            est_output_tokens=est_tokens(content) if content else 0,
            latency=round(latency, 2),
            retries=api_key.retry_count,
            status=status,
            fallback_used=is_fallback,
            submittal=sub_id,
        )

        if content is not None:
            api_key.mark_success()
            return content, None

        # Failure handling — per key.
        now = time.time()
        if status == 429:
            api_key.mark_429(now)               # cooldown THIS key only
            tried_notes.append(f"{api_key.label}:429")
            continue                            # immediately try next key
        elif status in RETRYABLE_STATUS or status is None:
            api_key.mark_other_failure(now)     # 5xx / timeout
            tried_notes.append(f"{api_key.label}:{status or 'net'}")
            continue
        else:
            # Permanent (401/400/403): park this key on a long cooldown so the
            # scheduler stops picking it, then move on to other keys.
            api_key.cooldown_until = now + BACKOFF_CAP
            api_key.total_failures += 1
            tried_notes.append(f"{api_key.label}:{status}")
            continue

    return None, f"all keys failed ({'; '.join(tried_notes[:8])})"


def _parse_comments(content):
    """Robustly parse the LLM's JSON into a list of comment strings.
    Handles markdown fences, leading prose, and malformed-but-salvageable JSON."""
    raw_content = content
    content = content.strip()
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    parsed = None
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(content[start:end + 1])
    except Exception:
        parsed = None

    if parsed is None:
        # Salvage a comments array even if the outer JSON is malformed.
        try:
            m = re.search(r'"comments"\s*:\s*\[(.*?)\]', content, re.S)
            if m:
                items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
                parsed = {"comments": [i.encode().decode('unicode_escape') for i in items]}
        except Exception:
            parsed = None

    if parsed is None:
        return None, raw_content

    return parsed.get("comments", []), None


def _clean_comment_list(raw):
    """Apply the existing cleaning pipeline (order preserved)."""
    cleaned = []
    for c in raw:
        if not isinstance(c, str):
            continue
        c = re.sub(r'^\s*\d+\s*[\.\)]\s*', '', c).strip()
        if c:
            cleaned.append(c)
    cleaned = split_merged_comments(cleaned)
    cleaned = drop_headers(cleaned)
    cleaned = [strip_leadin(c) for c in cleaned]
    cleaned = [c for c in cleaned if len(c) >= 3]
    return cleaned


async def _llm_only(session, zip_name, pdf_text, doc_name, unreadable,
                    api_choice, model, key_pool, analytics):
    """Extract comments for one submittal. Splits oversized text, calls the
    key-pool scheduler per chunk, merges results in order."""
    sub_id = submittal_id_from_name(zip_name)
    base = {"submittal": sub_id, "document": doc_name, "unreadable": unreadable}
    if not pdf_text.strip():
        return {**base, "comments": [], "error": "No text in PDF"}

    # Smart request management: keep each request under the model ceiling.
    chunks = split_oversized(pdf_text, MODEL_CHAR_LIMIT)

    all_comments = []
    last_err = None
    for chunk in chunks:
        prompt = PROMPT_TEMPLATE.format(body=chunk)
        content, err = await call_with_key_pool(
            session, prompt, model, api_choice, key_pool, analytics, sub_id
        )
        if err or content is None:
            last_err = err or "no content"
            print(f"DEBUG {sub_id}: {last_err}")
            continue
        parsed, _raw = _parse_comments(content)
        if parsed is None:
            last_err = f"No JSON in response (got: {(_raw or '')[:40]})"
            print(f"DEBUG {sub_id}: {last_err}")
            continue
        all_comments.extend(_clean_comment_list(parsed))

    # If we got comments from at least one chunk, that's a success even if
    # another chunk failed (partial recovery beats total failure).
    if all_comments:
        return {**base, "comments": all_comments, "error": None}
    return {**base, "comments": [], "error": last_err or "no comments extracted"}


async def _process_one(session, zip_name, zip_bytes, sem,
                       api_choice, model, key_pool, analytics):
    """Extract text for ONE zip and call the LLM, under the semaphore so only
    CONCURRENCY PDFs are in memory at once. Frees PDF bytes and gc's between.
    One ZIP failing never stops the batch (exception is caught -> error row)."""
    async with sem:
        try:
            pages, unreadable, had_comment_file = extract_pdf_text(zip_bytes)
            doc_name = extract_doc_name_from_zip(zip_bytes)
            if not had_comment_file:
                # No annotated/response PDF -> genuinely no reviewer comments.
                zip_bytes = None
                res = {"submittal": submittal_id_from_name(zip_name),
                       "document": doc_name, "unreadable": 0,
                       "comments": [], "error": None}
            else:
                zip_bytes = None
                # Token reduction: strip repeated boilerplate BEFORE prioritising.
                pages = dedupe_boilerplate(pages)
                text = prioritise_pages(pages)
                pages = None
                gc.collect()
                res = await _llm_only(session, zip_name, text, doc_name,
                                      unreadable, api_choice, model,
                                      key_pool, analytics)
        except Exception as e:
            res = {"submittal": submittal_id_from_name(zip_name),
                   "document": "", "unreadable": 0,
                   "comments": [], "error": f"Processing error: {str(e)[:60]}"}
        gc.collect()
        return res


async def process_all(zip_files, api_choice, model,
                      key_pool, analytics, progress_cb=None):
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            _process_one(session, zip_name, zip_bytes, sem,
                         api_choice, model, key_pool, analytics)
            for zip_name, zip_bytes in zip_files
        ]
        done = 0
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            done += 1
            if progress_cb:
                progress_cb(done, len(tasks))
    order = {submittal_id_from_name(n): i for i, (n, _) in enumerate(zip_files)}
    results.sort(key=lambda r: order.get(r["submittal"], 999))
    return results


# ============================================================================
# EXCEL (matches TPL_Comments.xlsx exactly)  — unchanged
# ============================================================================
def build_excel(results):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Sheet1"

    header_font  = Font(bold=True, size=11)
    header_fill  = PatternFill("solid", fgColor="D9D9D9")
    center = Alignment(horizontal="center", vertical="top", wrap_text=True)
    left   = Alignment(horizontal="left",   vertical="top", wrap_text=True)
    thin = Side(style="thin")
    border = Border(left=thin, right=thin, top=thin, bottom=thin)
    cnr_fill = PatternFill("solid", fgColor="FFF2CC")

    headers = ["Sr no.", "Submittal", "Document ", "Costumer Comments",
               "Xylem Remarks", "Review Note"]
    for col, h in enumerate(headers, 1):
        c = ws.cell(row=1, column=col, value=h)
        c.font = header_font; c.fill = header_fill
        c.alignment = center; c.border = border

    row = 2
    sr = 1
    for r in results:
        comments = r["comments"]
        has = bool(comments)
        n = max(1, len(comments))

        for k in range(n):
            first = (k == 0)
            c = ws.cell(row=row, column=1, value=(sr if first else None))
            c.alignment = center; c.border = border
            c = ws.cell(row=row, column=2, value=(r["submittal"] if first else None))
            c.alignment = left; c.border = border
            c = ws.cell(row=row, column=3, value=(r.get("document", "") if first else None))
            c.alignment = left; c.border = border
            val = f"{k+1}. {comments[k]}" if has else ""
            c = ws.cell(row=row, column=4, value=val)
            c.alignment = left; c.border = border
            if first and not has:
                c = ws.cell(row=row, column=5, value="Comment not Received")
                c.fill = cnr_fill
            else:
                c = ws.cell(row=row, column=5, value=None)
            c.alignment = left; c.border = border
            if first:
                unread = r.get("unreadable", 0)
                if unread > 0:
                    note = (f"{unread} page(s) could not be read as text "
                            f"(possible scanned comments) — check PDF manually")
                else:
                    note = ""
                cn = ws.cell(row=row, column=6, value=note)
                if unread > 0:
                    cn.fill = cnr_fill
            else:
                cn = ws.cell(row=row, column=6, value=None)
            cn.alignment = left; cn.border = border
            row += 1

        sr += 1

    last = row - 1
    ws.column_dimensions["A"].width = 13
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 50
    ws.column_dimensions["D"].width = 69
    ws.column_dimensions["E"].width = 21
    ws.column_dimensions["F"].width = 30
    ws.freeze_panes = "A2"
    if last >= 1:
        ws.auto_filter.ref = f"A1:F{last}"
    return wb


# ============================================================================
# UI  (unchanged layout; adds optional fallback keys + analytics summary)
# ============================================================================
st.title("TPL Comment Extractor")
st.caption("Upload submittal ZIPs. Comments are extracted by an LLM in parallel, "
           "split into individual items, and written to the TPL Excel format.")

# API selection — in the main page so it's always visible (mobile-friendly)
st.subheader("1. Choose API")
col1, col2 = st.columns(2)
with col1:
    api_choice = st.selectbox("Primary API Service", list(SUPPORTED_APIS.keys()),
                              help="Which provider to prefer first. Its keys are "
                                   "used before falling back to other providers.")
with col2:
    available_models = SUPPORTED_APIS[api_choice]["models"]
    model = st.selectbox("Model", available_models,
                         help="Pick a model, or type a custom one below")

custom_model = st.text_input("Custom model (optional)", value="",
                             help="Override the model above with any model ID")
if custom_model.strip():
    model = custom_model.strip()

# ---------------------------------------------------------------------------
# PROVIDER POOL UI — add/remove multiple keys per provider (Gemini + Groq).
# Keys live in st.session_state so the +/- buttons persist across reruns.
# Backward compatible: enter one Gemini + one Groq key and it behaves as before.
# ---------------------------------------------------------------------------
st.subheader("2. API Key Pool")
st.caption("Add multiple keys per provider. During processing the app rotates "
           "through them: a key that hits a rate limit (429) is put on cooldown "
           "and the next available key is used immediately. Gemini keys are "
           "preferred, then Groq.")

POOL_PROVIDERS = ["gemini", "groq"]

# initialise session state: one empty slot per provider on first load
for prov in POOL_PROVIDERS:
    sk = f"pool_{prov}"
    if sk not in st.session_state:
        st.session_state[sk] = [""]

def _add_key(prov):
    st.session_state[f"pool_{prov}"].append("")

def _remove_key(prov, idx):
    lst = st.session_state[f"pool_{prov}"]
    if 0 <= idx < len(lst):
        lst.pop(idx)
    if not lst:                       # never leave a provider with zero slots
        lst.append("")

for prov in POOL_PROVIDERS:
    sk = f"pool_{prov}"
    with st.expander(f"{prov.upper()} keys "
                     f"({sum(1 for k in st.session_state[sk] if k.strip())} set)",
                     expanded=(prov == api_choice)):
        for idx in range(len(st.session_state[sk])):
            kcol, bcol = st.columns([6, 1])
            with kcol:
                st.session_state[sk][idx] = st.text_input(
                    f"{prov.upper()} Key {idx + 1}",
                    value=st.session_state[sk][idx],
                    type="password",
                    key=f"{sk}_input_{idx}",
                )
            with bcol:
                st.write("")  # vertical spacer to align button with input
                st.button("➖", key=f"{sk}_del_{idx}",
                          help="Remove this key",
                          on_click=_remove_key, args=(prov, idx))
        st.button(f"➕ Add another {prov.upper()} key",
                  key=f"{sk}_add", on_click=_add_key, args=(prov,))

# Collect the configured pool (order preserved, blanks dropped).
provider_key_lists = {
    prov: [k.strip() for k in st.session_state[f"pool_{prov}"] if k.strip()]
    for prov in POOL_PROVIDERS
}
total_keys = sum(len(v) for v in provider_key_lists.values())

if st.button("Clear results & free memory"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    gc.collect()
    st.rerun()

st.divider()
st.subheader("3. Upload & Extract")

if total_keys == 0:
    st.warning("Add at least one Gemini or Groq API key above to begin.")
    st.stop()

# Make sure the primary provider actually has keys; if not, fall back to
# whichever provider does (so the user isn't forced to match the dropdown).
if not provider_key_lists.get(api_choice):
    for prov in POOL_PROVIDERS:
        if provider_key_lists.get(prov):
            api_choice = prov
            model = SUPPORTED_APIS[prov]["models"][0]
            st.info(f"No {api_choice.upper()} keys entered for the selected "
                    f"primary; using {prov.upper()} as primary instead.")
            break

uploaded = st.file_uploader("Upload ZIP files", type="zip",
                            accept_multiple_files=True)

if uploaded:
    st.info(f"{len(uploaded)} ZIP file(s) ready. {total_keys} key(s) in pool.")
    if st.button("Extract Comments"):
        zip_files = [(f.name, f.read()) for f in uploaded]

        # Build the key pool from every configured key.
        key_pool = KeyPool(provider_key_lists)

        analytics = Analytics()
        wall_start = time.time()

        bar = st.progress(0.0)
        status = st.empty()
        def cb(done, total):
            bar.progress(done / total)
            status.text(f"Processed {done}/{total} submittals...")

        status.text(f"Sending to {api_choice.upper()} (pool of {total_keys} keys)...")
        results = asyncio.run(process_all(zip_files, api_choice, model,
                                          key_pool, analytics, cb))
        status.text("Building Excel...")

        wb = build_excel(results)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            data = open(tmp.name, "rb").read()
        os.unlink(tmp.name)

        wall_seconds = time.time() - wall_start

        st.success("Done.")
        st.download_button(
            "Download TPL_Comments.xlsx",
            data=data,
            file_name=f"TPL_Comments_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        total = sum(len(r["comments"]) for r in results)
        errs = [r for r in results if r["error"]]
        ok_zips = len(results) - len(errs)

        st.subheader("Summary")
        st.write(f"Submittals processed: {len(results)}")
        st.write(f"Total comments extracted: {total}")
        st.write(f"Primary API: {api_choice.upper()}  |  Keys in pool: {total_keys}")
        if errs:
            st.warning(f"{len(errs)} submittal(s) had issues:")
            for r in errs:
                st.write(f"- {r['submittal']}: {r['error']}")

        # ---- Request analytics summary ------------------------------------
        summary = analytics.summary(
            total_zips=len(results),
            ok_zips=ok_zips,
            fail_zips=len(errs),
            wall_seconds=wall_seconds,
        )
        st.subheader("Request Analytics")
        st.table(summary)

        # ---- Per-key analytics (requirement 9) ----------------------------
        st.subheader("Per-Key Statistics")
        st.table(analytics.per_key_table(key_pool))

        with st.expander("Preview extracted comments"):
            for r in results:
                st.markdown(f"**{r['submittal']}** — {len(r['comments'])} comment(s)")
                for i, c in enumerate(r["comments"], 1):
                    st.write(f"{i}. {c}")

        gc.collect()
