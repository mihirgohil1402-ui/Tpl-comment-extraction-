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

st.set_page_config(page_title="TPL Comment Extractor", layout="wide")

# ============================================================================
# CONFIG
# ============================================================================
GROQ_API_KEY = st.secrets.get("groq_api_key", "")
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL   = "llama-3.3-70b-versatile"

# How many ZIPs to send to Groq at once. Keeps us under rate limits while
# still being fast. 8 concurrent is comfortable on the free tier.
CONCURRENCY = 8

# ============================================================================
# PDF / ZIP HELPERS
# ============================================================================
def submittal_id_from_name(zip_name):
    """SUB1715-_ Rev No_ 0.zip  ->  SUB1715"""
    m = re.search(r'(SUB\d+)', zip_name, re.I)
    return m.group(1).upper() if m else zip_name.replace('.zip', '')

def extract_pdf_text(zip_bytes):
    """Pull text from PDFs inside the ZIP using pdfplumber.

    Reviewer comments live in the *_annotated.pdf as positioned callout text on
    drawing pages, which sit toward the END of the document (after spec tables
    and boilerplate). We collect per-page text, then order pages so the ones
    most likely to contain reviewer comments come FIRST — otherwise a length
    cap would chop off exactly the pages we care about.
    """
    pages_text = []
    try:
        with zipfile.ZipFile(io.BytesIO(zip_bytes), 'r') as zf:
            all_pdfs = [f for f in zf.namelist() if f.lower().endswith('.pdf')]
            annotated = [f for f in all_pdfs if 'annotated' in f.lower()]
            pdfs = annotated if annotated else all_pdfs
            for name in pdfs:
                try:
                    raw = zf.read(name)
                    with pdfplumber.open(io.BytesIO(raw)) as pdf:
                        for page in pdf.pages:
                            t = page.extract_text() or ""
                            if t.strip():
                                pages_text.append(t)
                except Exception:
                    pass
    except Exception:
        pass
    return pages_text

# Words/phrases that signal a page carries reviewer comments rather than
# pure spec-sheet boilerplate. Used to prioritise pages for the LLM.
COMMENT_SIGNALS = re.compile(
    r'\b(shall|should|to be|required|ensure|provide|verify|comply|complian'
    r'|approved|vendor|deviation|clarification|as per|not acceptable'
    r'|to be furnished|to be considered|consult|review)\b',
    re.I
)

def prioritise_pages(pages_text, char_budget=24000):
    """Order pages by how comment-like they are, then pack up to char_budget.
    This keeps comment pages even when the document is long."""
    scored = []
    for i, t in enumerate(pages_text):
        score = len(COMMENT_SIGNALS.findall(t))
        scored.append((score, i, t))
    # comment-rich pages first; preserve original order among equal scores
    scored.sort(key=lambda x: (-x[0], x[1]))
    out, used = [], 0
    for score, i, t in scored:
        if used + len(t) > char_budget:
            t = t[:max(0, char_budget - used)]
        if t:
            out.append(t)
            used += len(t)
        if used >= char_budget:
            break
    return "\n".join(out)

# ============================================================================
# COMMENT SPLITTER (backstop for when the model still merges a numbered list)
# Splits only on clear "N) " list markers. Will NOT break "1.5 sq mm",
# "26 29 23", "433V", "80-100 microns", etc.
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

# Lines that are headers/instructions, not actual review comments.
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

# Remove an inline lead-in that precedes the first real item, e.g.
# "Clarifications required for following points wrt the Jacobs specifications: Approved vendors..."
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

Extract EVERY distinct reviewer comment as its own separate item.

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

async def extract_one(session, zip_name, pdf_text, sem):
    sub_id = submittal_id_from_name(zip_name)
    if not pdf_text.strip():
        return {"submittal": sub_id, "comments": [], "error": "No text in PDF"}

    prompt = PROMPT_TEMPLATE.format(body=pdf_text)
    payload = {
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 3000,
    }
    headers = {"Authorization": f"Bearer {GROQ_API_KEY}",
               "Content-Type": "application/json"}

    async with sem:
        try:
            async with session.post(GROQ_API_URL, json=payload,
                                    headers=headers, timeout=60) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    return {"submittal": sub_id, "comments": [],
                            "error": f"API {resp.status}: {txt[:80]}"}
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
        except Exception as e:
            return {"submittal": sub_id, "comments": [], "error": str(e)[:80]}

    # Parse JSON out of the response
    try:
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            print(f"DEBUG {sub_id}: No JSON found. Response: {content[:200]}")
            return {"submittal": sub_id, "comments": [], "error": "No JSON in response"}
        parsed = json.loads(content[start:end+1])
        raw = parsed.get("comments", [])
        print(f"DEBUG {sub_id}: Parsed {len(raw)} comments from Groq")
    except Exception as e:
        print(f"DEBUG {sub_id}: Parse error: {e}. Response: {content[:200]}")
        return {"submittal": sub_id, "comments": [], "error": f"Parse failed: {str(e)[:40]}"}

    # Clean: strip any leading numbering the model added, then SPLIT merged
    # numbered blocks, THEN drop header/lead-in fragments. Order matters:
    # splitting first turns "Clarifications required: 1) X 2) Y" into a header
    # fragment + the real items, so dropping headers keeps X and Y.
    cleaned = []
    for c in raw:
        if not isinstance(c, str):
            continue
        c = re.sub(r'^\s*\d+\s*[\.\)]\s*', '', c).strip()
        if c:
            cleaned.append(c)
    cleaned = split_merged_comments(cleaned)
    cleaned = drop_headers(cleaned)
    # also strip a leading "header: " lead-in that sits inline before the first item
    cleaned = [strip_leadin(c) for c in cleaned]
    cleaned = [c for c in cleaned if len(c) >= 3]

    return {"submittal": sub_id, "comments": cleaned, "error": None}

async def process_all(zip_files, progress_cb=None):
    sem = asyncio.Semaphore(CONCURRENCY)
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = []
        for zip_name, zip_bytes in zip_files:
            pages = extract_pdf_text(zip_bytes)
            text = prioritise_pages(pages)
            tasks.append(extract_one(session, zip_name, text, sem))
        done = 0
        for coro in asyncio.as_completed(tasks):
            res = await coro
            results.append(res)
            done += 1
            if progress_cb:
                progress_cb(done, len(tasks))
    # keep input order by submittal id
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

    # EXACT header text from the template (note the typo + trailing space)
    headers = ["Sr no.", "Submittal", "Document ", "Costumer Comments", "Xylem Remarks"]
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
        block_start = row

        for k in range(n):
            first = (k == 0)
            # Sr no.
            c = ws.cell(row=row, column=1, value=(sr if first else None))
            c.alignment = center; c.border = border
            # Submittal
            c = ws.cell(row=row, column=2, value=(r["submittal"] if first else None))
            c.alignment = left; c.border = border
            # Document  (LLM path has no doc name; left blank, kept for format parity)
            c = ws.cell(row=row, column=3, value=("" if first else None))
            c.alignment = left; c.border = border
            # Costumer Comments — one per row, numbered
            val = f"{k+1}. {comments[k]}" if has else ""
            c = ws.cell(row=row, column=4, value=val)
            c.alignment = left; c.border = border
            # Xylem Remarks
            if first and not has:
                c = ws.cell(row=row, column=5, value="Comment not Received")
                c.fill = cnr_fill
            else:
                c = ws.cell(row=row, column=5, value=None)
            c.alignment = left; c.border = border
            row += 1

        if n > 1:
            for col in (1, 2, 3, 5):
                ws.merge_cells(start_row=block_start, start_column=col,
                               end_row=row-1, end_column=col)
        sr += 1

    last = row - 1
    ws.column_dimensions["A"].width = 13
    ws.column_dimensions["B"].width = 10
    ws.column_dimensions["C"].width = 50
    ws.column_dimensions["D"].width = 69
    ws.column_dimensions["E"].width = 21
    ws.freeze_panes = "A2"
    if last >= 1:
        ws.auto_filter.ref = f"A1:E{last}"
    return wb

# ============================================================================
# UI
# ============================================================================
st.title("TPL Comment Extractor")
st.caption("Upload submittal ZIPs. Comments are extracted by an LLM in parallel, "
           "split into individual items, and written to the TPL Excel format.")

if not GROQ_API_KEY:
    st.error("Groq API key not set. Add `groq_api_key` to .streamlit/secrets.toml")
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

        status.text("Sending to Groq in parallel...")
        results = asyncio.run(process_all(zip_files, cb))
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
        if errs:
            st.warning(f"{len(errs)} submittal(s) had issues:")
            for r in errs:
                st.write(f"- {r['submittal']}: {r['error']}")

        with st.expander("Preview extracted comments"):
            for r in results:
                st.markdown(f"**{r['submittal']}** — {len(r['comments'])} comment(s)")
                for i, c in enumerate(r["comments"], 1):
                    st.write(f"{i}. {c}")
