# app.py - TPL Comment Extractor (Groq + parallel + comment splitting)
#
# Setup:
#   pip install streamlit aiohttp openpyxl pdfplumber
#   Put your key in .streamlit/secrets.toml:  groq_api_key = "gsk_..."
#   streamlit run app.py

import streamlit as st
import zipfile
import io
import pdfplumber
import asyncio
import aiohttp
import json
import re
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from datetime import datetime
import tempfile
import os
import gc

st.set_page_config(page_title="TPL Comment Extractor", layout="wide")

# ============================================================================
# CONFIG
# ============================================================================
CONCURRENCY = 2

# ----------------------------------------------------------------------------
# SUPPORTED APIS — add a new LLM service by adding an entry here.
# No need to touch any function logic below.
#
# Each entry needs:
#   url           : the chat/completions endpoint
#   format        : "openai" (Groq, OpenAI, Mistral, DeepSeek, Gemini-compat,
#                   Together, OpenRouter, local Ollama...) or "anthropic" (Claude)
#   models        : list of model IDs (first one is the default in the dropdown)
#   extra_headers : (optional) dict of extra headers, e.g. Claude's version
#
# MOST modern APIs are "openai" format (OpenAI-compatible). Only Claude uses
# the "anthropic" format. To add a new OpenAI-compatible API, copy a block,
# change url + models, done.
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
    # ---- To add your own, copy this template and fill it in: ----
    # "myapi": {
    #     "url": "https://api.example.com/v1/chat/completions",
    #     "format": "openai",          # or "anthropic"
    #     "models": ["model-id-here"],
    #     # "extra_headers": {"some-header": "value"},   # optional
    # },
}


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
                            if t.strip():
                                pages_text.append(t)
                            if len(t.strip()) < 50:
                                try:
                                    if len(page.images) > 0:
                                        unreadable_pages += 1
                                except Exception:
                                    pass
                except Exception:
                    pass
    except Exception:
        pass
    return pages_text, unreadable_pages, had_comment_file

def extract_doc_name(pages_text):
    """Build '{DOC_NO}_{Title}' from the cover-page text (Document column)."""
    full = "\n".join(pages_text[:3])
    doc_no = None
    m = re.search(r'\b(TPL-[A-Z0-9]+-\d+-[A-Z]+-[A-Z]+-\d+)\b', full, re.I)
    if m:
        doc_no = m.group(1)
    title = None
    mt = re.search(r'(?:Document Title:|Doc\.?\s*Title:)\s*(.+?)(?:DOC\s*NO|Doc\.?\s*No)',
                   full, re.I | re.S)
    if mt:
        title = re.sub(r'\s+', ' ', mt.group(1)).strip()
        title = re.sub(r'(\w)-\s+(\w)', r'\1-\2', title)
    if doc_no and title:
        return f"{doc_no}_{title}"
    return doc_no or title or ""

def extract_doc_name_from_zip(zip_bytes):
    """Get the document name from ANY pdf in the zip (lead sheet / datasheet),
    so the Document column is filled even when there are no comments."""
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            all_pdfs = [f for f in zf.namelist() if f.lower().endswith('.pdf')]
            for name in all_pdfs:
                try:
                    raw = zf.read(name)
                    with pdfplumber.open(io.BytesIO(raw)) as pdf:
                        pages = []
                        for p in pdf.pages[:3]:
                            t = p.extract_text() or ""
                            if t.strip():
                                pages.append(t)
                        dn = extract_doc_name(pages)
                        if dn:
                            return dn
                except Exception:
                    pass
    except Exception:
        pass
    return ""

# Words/phrases that signal a page carries reviewer comments rather than
# pure spec-sheet boilerplate. Used to prioritise pages if a doc is too long.
COMMENT_SIGNALS = re.compile(
    r'\b(shall|should|to be|required|ensure|provide|verify|comply|complian'
    r'|approved|vendor|deviation|clarification|as per|not acceptable'
    r'|to be furnished|to be considered|consult|review)\b',
    re.I
)

def prioritise_pages(pages_text, char_budget=24000):
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

# ============================================================================
# COMMENT SPLITTER
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
# GROQ CALL
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

async def _api_request(session, api_choice, api_key, model, prompt):
    """Call the selected API using its config from SUPPORTED_APIS.
    Returns (content_str, error_str). Exactly one is non-None.

    Reads everything (url, format, headers) from the SUPPORTED_APIS dict, so
    adding a new API is just a new dict entry — no changes needed here."""

    cfg = SUPPORTED_APIS.get(api_choice)
    if cfg is None:
        return None, f"Unknown API: {api_choice}"
    if not api_key:
        return None, f"No API key provided for {api_choice}"

    url = cfg["url"]
    fmt = cfg.get("format", "openai")

    # Build headers
    headers = {"Content-Type": "application/json"}
    if fmt == "anthropic":
        headers["x-api-key"] = api_key
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    # any extra headers from config (e.g. Claude's anthropic-version)
    for k, v in cfg.get("extra_headers", {}).items():
        headers[k] = v

    # Build payload in the right format
    if fmt == "anthropic":
        payload = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": prompt}],
        }
    else:  # openai-compatible
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 4096,
        }

    try:
        async with session.post(url, json=payload, headers=headers, timeout=90) as resp:
            if resp.status == 200:
                data = await resp.json()
                try:
                    if fmt == "anthropic":
                        return data["content"][0]["text"], None
                    else:
                        return data["choices"][0]["message"]["content"], None
                except (KeyError, IndexError, TypeError) as e:
                    return None, f"Unexpected response shape: {str(e)[:60]}"
            if resp.status == 429:
                txt = await resp.text()
                return None, f"429 on {api_choice}: {txt[:200]}"
            txt = await resp.text()
            return None, f"API {resp.status}: {txt[:200]}"
    except Exception as e:
        return None, str(e)[:80]

async def _llm_only(session, zip_name, pdf_text, doc_name, unreadable, api_choice, api_key, model):
    """Call the selected API and clean the result."""
    sub_id = submittal_id_from_name(zip_name)
    base = {"submittal": sub_id, "document": doc_name, "unreadable": unreadable}
    if not pdf_text.strip():
        return {**base, "comments": [], "error": "No text in PDF"}

    prompt = PROMPT_TEMPLATE.format(body=pdf_text)
    content, err = await _api_request(session, api_choice, api_key, model, prompt)
    if err:
        print(f"DEBUG {sub_id}: {err}")
        return {**base, "comments": [], "error": err}

    # Clean common wrappers: markdown fences, "json" label, leading prose
    raw_content = content
    content = content.strip()
    # remove ```json ... ``` or ``` ... ``` fences
    content = re.sub(r'^```(?:json)?\s*', '', content)
    content = re.sub(r'\s*```$', '', content)

    parsed = None
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start != -1 and end != -1 and end > start:
            parsed = json.loads(content[start:end+1])
    except Exception:
        parsed = None

    # Fallback: try to salvage a comments array even if outer JSON is malformed
    if parsed is None:
        try:
            # find "comments": [ ... ]
            m = re.search(r'"comments"\s*:\s*\[(.*?)\]', content, re.S)
            if m:
                items = re.findall(r'"((?:[^"\\]|\\.)*)"', m.group(1))
                parsed = {"comments": [i.encode().decode('unicode_escape') for i in items]}
        except Exception:
            parsed = None

    if parsed is None:
        print(f"DEBUG {sub_id}: No JSON. Raw response: {raw_content[:300]}")
        return {**base, "comments": [], "error": f"No JSON in response (got: {raw_content[:40]})"}

    raw = parsed.get("comments", [])
    print(f"DEBUG {sub_id}: Parsed {len(raw)} comments from {api_choice}")

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

    return {**base, "comments": cleaned, "error": None}

async def _process_one(session, zip_name, zip_bytes, sem, api_choice, api_key, model):
    """Extract text for ONE zip and call the LLM, under the semaphore so only
    CONCURRENCY PDFs are in memory at once. Frees PDF bytes and gc's between."""
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
                text = prioritise_pages(pages)
                pages = None
                gc.collect()
                res = await _llm_only(session, zip_name, text, doc_name, unreadable, api_choice, api_key, model)
        except Exception as e:
            res = {"submittal": submittal_id_from_name(zip_name),
                   "document": "", "unreadable": 0,
                   "comments": [], "error": f"Processing error: {str(e)[:60]}"}
        gc.collect()
        return res

async def process_all(zip_files, api_choice, api_key, model, progress_cb=None):
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [
            _process_one(session, zip_name, zip_bytes, sem, api_choice, api_key, model)
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
# EXCEL (matches TPL_Comments.xlsx exactly)
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
# UI
# ============================================================================
st.title("TPL Comment Extractor")
st.caption("Upload submittal ZIPs. Comments are extracted by an LLM in parallel, "
           "split into individual items, and written to the TPL Excel format.")

# API selection — in the main page so it's always visible (mobile-friendly)
st.subheader("1. Choose API")
col1, col2 = st.columns(2)
with col1:
    api_choice = st.selectbox("API Service", list(SUPPORTED_APIS.keys()),
                              help="Choose which LLM service to use")
with col2:
    available_models = SUPPORTED_APIS[api_choice]["models"]
    model = st.selectbox("Model", available_models,
                         help="Pick a model, or type a custom one below")

custom_model = st.text_input("Custom model (optional)", value="",
                             help="Override the model above with any model ID")
if custom_model.strip():
    model = custom_model.strip()

api_key = st.text_input(f"{api_choice.upper()} API Key", type="password",
                        help=f"Paste your {api_choice.upper()} API key here")

if st.button("Clear results & free memory"):
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    gc.collect()
    st.rerun()

st.divider()
st.subheader("2. Upload & Extract")

if not api_key:
    st.warning(f"Enter your {api_choice.upper()} API key above to begin.")
    st.stop()

uploaded = st.file_uploader("Upload ZIP files", type="zip",
                            accept_multiple_files=True)

if uploaded:
    st.info(f"{len(uploaded)} ZIP file(s) ready.")
    if st.button("Extract Comments"):
        zip_files = [(f.name, f.read()) for f in uploaded]

        bar = st.progress(0.0)
        status = st.empty()
        def cb(done, total):
            bar.progress(done / total)
            status.text(f"Processed {done}/{total} submittals...")

        status.text(f"Sending to {api_choice.upper()}...")
        results = asyncio.run(process_all(zip_files, api_choice, api_key, model, cb))
        status.text("Building Excel...")

        wb = build_excel(results)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp:
            wb.save(tmp.name)
            tmp.seek(0)
            data = open(tmp.name, "rb").read()
        os.unlink(tmp.name)

        st.success("Done.")
        st.download_button(
            "Download TPL_Comments.xlsx",
            data=data,
            file_name=f"TPL_Comments_{datetime.now():%Y%m%d_%H%M%S}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

        total = sum(len(r["comments"]) for r in results)
        errs = [r for r in results if r["error"]]
        st.subheader("Summary")
        st.write(f"Submittals processed: {len(results)}")
        st.write(f"Total comments extracted: {total}")
        st.write(f"API used: {api_choice.upper()}")
        if errs:
            st.warning(f"{len(errs)} submittal(s) had issues:")
            for r in errs:
                st.write(f"- {r['submittal']}: {r['error']}")

        with st.expander("Preview extracted comments"):
            for r in results:
                st.markdown(f"**{r['submittal']}** — {len(r['comments'])} comment(s)")
                for i, c in enumerate(r["comments"], 1):
                    st.write(f"{i}. {c}")

        gc.collect()
