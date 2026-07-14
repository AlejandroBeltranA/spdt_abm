#!/usr/bin/env python3
"""
Build a standalone, editable side-by-side review HTML for the WHOLE Paper 1
draft. Parses main.tex section by section (left column = current manuscript)
and aligns the new section drafts on the right (editable). Sections without a
new draft get an editable copy of the original, so the whole paper can be voice-
edited in one place. Figures are baked in, with stale (pre-v5) and Paper-2-bound
figures clearly badged. Edits autosave to localStorage and export to Markdown.

No server: open docs/research/paper draft/too_hot_overleaf_v2/draft_review.html
directly in Chrome (figures load by relative path, keep the file in that folder).

Usage: .venv/bin/python research/applied/scripts/build_review_html.py
"""

from __future__ import annotations

import html
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PAPER = REPO / "docs/research/paper draft/too_hot_overleaf_v2"
MAIN = PAPER / "main.tex"

# New drafts keyed by the main.tex (sub)section title they replace/extend.
DRAFT_FOR_TITLE = {
    "Calibration": "methodology_3_4_draft.md",
    "Validation": "methodology_3_5_validation_draft.md",
}
# New reading order (2026-06-18): study-area lead-in, architecture (+ demand
# equations), data/population, calibration, sensitivity, validation-across-cities.
# Brand-new sections, inserted after a given title (no original text).
INSERT_AFTER = {
    "MABM Architecture": [
        ("§3.3 (cont.) — demand formulation", "methodology_3_3_architecture_draft.md", []),
    ],
    "Calibration": [  # SA now sits straight after calibration, before validation
        ("§3.5 Sensitivity analysis", "methodology_3_5_sensitivity_draft.md", []),
    ],
    "__RESULTS__": [  # inserted right after the \section{Results} heading
        ("§Results.1 In-sample (Newcastle)", "results_1_insample_draft.md", []),
        ("§Results.2–3 Spatial transfer + Reliability layer",
         "results_2_3_transfer_reliability_draft.md", []),
    ],
}
# Present architecture before the data that fills it (main.tex has them the
# other way round); swap these two units after parsing.
SWAP_ORDER = ("MABM Architecture", "Synthetic Population Construction")
# Old Results subsections that move to Paper 2 (badge them).
PAPER2_TITLES = {"Occupancy and Schedule Effects", "Policy Scenario Comparison",
                 "Financial Impact and Spatial Distribution"}
# Both figure_1 (pipeline) and figure_2 (validation) were regenerated for v5
# (2026-06-17), so nothing in the pipeline is stale any more.
STALE_FIGS: set[str] = set()
PAPER2_FIGS = {"figure_3_tradeoffs.png", "figure_4_income_quintiles.png", "figure_5_spatial.png"}

# H2 headings inside a draft .md that are reviewer scaffolding, not paper prose.
# Everything from the first match onward is pulled into a muted "notes" box and
# kept out of the editable draft column and the Markdown export.
META_HEADINGS = (
    "drafting notes", "compared to the prior version", "numbers and their source",
    "citations still to", "open editorial decisions", "open questions", "length",
)

# ── LaTeX → readable HTML ──────────────────────────────────────────────────

def latex_to_html(t: str) -> str:
    t = re.sub(r"\\citep(?:\[[^\]]*\])?\{([^}]*)\}", lambda m: f"<span class='cite'>[{m.group(1)}]</span>", t)
    t = re.sub(r"\\citet(?:\[[^\]]*\])?\{([^}]*)\}", lambda m: f"<span class='cite'>{m.group(1)}</span>", t)
    t = re.sub(r"\\cite\{([^}]*)\}", lambda m: f"<span class='cite'>[{m.group(1)}]</span>", t)
    t = re.sub(r"\\footnote\{.*?\}", "", t, flags=re.S)
    t = re.sub(r"\\(emph|textit)\{([^}]*)\}", r"<em>\2</em>", t)
    t = re.sub(r"\\textbf\{([^}]*)\}", r"<strong>\1</strong>", t)
    t = re.sub(r"\\ref\{[^}]*\}", "[ref]", t)
    t = re.sub(r"\\label\{[^}]*\}", "", t)
    t = re.sub(r"\\textsuperscript\{([^}]*)\}", r"<sup>\1</sup>", t)
    t = t.replace("\\textpm", "±").replace("\\%", "%").replace("\\&", "&").replace("\\,", " ")
    t = t.replace("``", "“").replace("''", "”").replace("~", " ")
    t = t.replace("---", "—").replace("--", "–")
    t = re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", t)   # any leftover \cmd{x} → x
    t = re.sub(r"\\[a-zA-Z]+", "", t)                  # bare \cmd
    return t.strip()


def clean_md(t: str) -> str:
    """Lighter cleanup for the draft .md prose (already mostly plain + some LaTeX).

    Inline math ($...$) is protected so MathJax renders it untouched.
    """
    math_spans: list[str] = []

    def _stash(m):
        math_spans.append(m.group(0))
        return f"\x00M{len(math_spans) - 1}\x00"

    t = re.sub(r"\$[^$]+\$", _stash, t)            # protect inline math
    t = re.sub(r"R\\textsuperscript\{2\}", "R²", t)
    t = re.sub(r"\\textsuperscript\{([^}]*)\}", r"<sup>\1</sup>", t)
    t = t.replace("\\textdegree", "°").replace("\\textpm", "±").replace("\\%", "%")
    t = t.replace("\\pounds", "£").replace("~", " ").replace("\\,", " ").replace("\\&", "&")
    t = re.sub(r"\\textbf\{([^}]*)\}", r"<strong>\1</strong>", t)
    t = re.sub(r"\\(emph|textit)\{([^}]*)\}", r"<em>\2</em>", t)
    t = re.sub(r"\\eqref\{[^}]*\}", "(Eq.)", t)
    t = re.sub(r"\\ref\{(fig|tab):[^}]*\}", lambda m: "[Fig.]" if m.group(1) == "fig" else "[Table]", t)
    t = re.sub(r"\\ref\{[^}]*\}", "(Eq.)", t)
    t = re.sub(r"\\cite\{([^}]*)\}", r"<span class='cite'>[cite: \1]</span>", t)
    t = re.sub(r"\[cite ([^\]]*)\]", r"<span class='cite'>[cite: \1]</span>", t)
    t = re.sub(r"\*\*([^*]*)\*\*", r"<strong>\1</strong>", t)
    t = t.replace("---", "—").replace("--", "–").replace("``", "“").replace("''", "”")
    for i, s in enumerate(math_spans):             # restore inline math
        t = t.replace(f"\x00M{i}\x00", s)
    return t.strip()


def latex_table_to_html(env: str) -> str:
    """Render a LaTeX table environment to an HTML table (cells via clean_md)."""
    cap_m = (re.search(r"\\caption\{(.*)\}\s*\\label", env, re.S)
             or re.search(r"\\caption\{(.*?)\}", env, re.S))
    tab_m = re.search(r"\\begin\{tabular\}\{[^}]*\}(.*?)\\end\{tabular\}", env, re.S)
    if not tab_m:
        return ""
    rows = []
    for line in tab_m.group(1).split(r"\\"):
        line = re.sub(r"\\hline|\\toprule|\\midrule|\\bottomrule", "", line).strip()
        if not line:
            continue
        rows.append([clean_md(c.strip()) for c in line.split("&")])
    if not rows:
        return ""
    head, rest = rows[0], rows[1:]
    thead = "<tr>" + "".join(f"<th>{c}</th>" for c in head) + "</tr>"
    tbody = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rest)
    cap = f"<div class='tabcap'>{clean_md(cap_m.group(1))}</div>" if cap_m else ""
    return f"{cap}<table class='data'><thead>{thead}</thead><tbody>{tbody}</tbody></table>"


# ── Parse main.tex into ordered (sub)section units ─────────────────────────

def parse_main_tex():
    text = MAIN.read_text()
    text = text[text.index("\\section{Introduction}"):]
    text = text.split("\\end{document}")[0]
    lines = text.splitlines()

    units = []
    cur = None
    buf: list[str] = []

    def flush():
        if cur is not None:
            cur["body"] = "\n".join(buf).strip()
            units.append(cur)

    for ln in lines:
        msec = re.match(r"\\section\{(.+?)\}", ln)
        msub = re.match(r"\\subsection\{(.+?)\}", ln)
        if msec or msub:
            flush()
            cur = {"level": "section" if msec else "subsection",
                   "title": (msec or msub).group(1)}
            buf = []
        else:
            if re.match(r"\s*%", ln):   # comment-only line
                continue
            buf.append(ln)
    flush()
    return units


def extract_figs(latex_body: str):
    """Return (body_without_figs, [(file, caption)])."""
    figs = []

    def grab(m):
        env = m.group(0)
        fm = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", env)
        cm = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end)", env, re.S)
        figs.append((fm.group(1) if fm else "", latex_to_html(cm.group(1)) if cm else ""))
        return "\n\n"

    body = re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", grab, latex_body, flags=re.S)
    return body, figs


def split_paras(body: str):
    return [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]


def _md_table_html(chunk: str) -> str | None:
    """Render a GitHub-style markdown table to an HTML <table>, else None."""
    rows = [ln for ln in chunk.splitlines() if ln.strip().startswith("|")]
    if len(rows) < 2 or not re.match(r"^\s*\|[\s:|-]+\|\s*$", rows[1]):
        return None

    def cells(line):
        return [clean_md(c.strip()) for c in line.strip().strip("|").split("|")]

    head = cells(rows[0])
    body = [cells(r) for r in rows[2:]]
    thead = "".join(f"<th>{c}</th>" for c in head)
    tbody = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in body)
    return f"<table class='data'><thead><tr>{thead}</tr></thead><tbody>{tbody}</tbody></table>"


def extract_draft_prose(md_file: str):
    """Split a draft .md into paper content and reviewer notes.

    Returns (items, meta_md) where items is a list of ('head'|'para'|'figure'
    |'table'|'mdtable', payload) for the paper prose, and meta_md is the raw
    markdown of the reviewer-scaffolding sections (kept out of the draft column).
    """
    raw = (PAPER / md_file).read_text()
    # Everything after the FIRST horizontal rule is content; the drafts use more
    # rules as separators, so don't stop at the second one. Paper prose runs up
    # to the first reviewer-notes heading (handled below).
    prose = raw.split("\n---\n", 1)[1] if "\n---\n" in raw else raw
    env_re = re.compile(
        r"\\begin\{(figure|table|equation|equation\*|align|align\*)\}.*?\\end\{\1\}", re.S)
    envs = []

    def stash(m):
        envs.append(m.group(0)); return f"\n\n[[ENV{len(envs)-1}]]\n\n"

    prose = env_re.sub(stash, prose)
    items, meta_chunks = [], []
    in_meta = False
    for chunk in re.split(r"\n\s*\n", prose):
        chunk = chunk.strip()
        if not chunk or chunk == "---":
            continue
        if chunk.startswith("## "):
            head = chunk[3:].strip()
            if any(head.lower().startswith(p) for p in META_HEADINGS):
                in_meta = True
            if in_meta:
                meta_chunks.append(chunk)
                continue
            items.append(("head", head))
            continue
        if in_meta:
            meta_chunks.append(chunk)
            continue
        m = re.match(r"\[\[ENV(\d+)\]\]", chunk)
        if m:
            env = envs[int(m.group(1))]
            if "\\begin{figure}" in env:
                fm = re.search(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", env)
                cm = re.search(r"\\caption\{(.*?)\}\s*(?:\\label|\\end)", env, re.S)
                items.append(("figure", (fm.group(1) if fm else "", clean_md(cm.group(1)) if cm else "")))
            elif "\\begin{table}" in env:
                items.append(("table", env))
            else:                                   # equation / align → MathJax
                items.append(("math", env))
        elif chunk.startswith("#"):
            continue
        elif _md_table_html(chunk):
            items.append(("mdtable", _md_table_html(chunk)))
        else:
            items.append(("para", clean_md(chunk)))
    return items, "\n\n".join(meta_chunks)


# ── HTML assembly ──────────────────────────────────────────────────────────

_pid = 0


def fig_html(fn: str, cap: str) -> str:
    badge = ""
    if fn in STALE_FIGS:
        badge = "<span class='badge stale'>⚠ pre-v5 (Jun 8) — regenerate</span>"
    elif fn in PAPER2_FIGS:
        badge = "<span class='badge p2'>→ Paper 2</span>"
    elif fn:
        badge = "<span class='badge ok'>v5 current</span>"
    return (f"<figure>{badge}<img src='{html.escape(fn)}' alt='{html.escape(fn)}'>"
            f"<figcaption>{cap}</figcaption></figure>")


def editable_para(text_html: str) -> str:
    global _pid
    _pid += 1
    return (f"<div class='prow'><label class='done'><input type='checkbox' data-done='p{_pid}'> done</label>"
            f"<div class='para' contenteditable='true' data-id='p{_pid}'>{text_html}</div></div>")


def draft_column(md_file: str, extra_figs):
    """Return (paper_html, meta_md) for a draft .md."""
    items, meta_md = extract_draft_prose(md_file)
    out = []
    for kind, payload in items:
        if kind == "head":
            out.append(f"<h4>{html.escape(payload)}</h4>")
        elif kind == "para":
            out.append(editable_para(payload))
        elif kind == "figure":
            out.append(fig_html(*payload))
        elif kind == "mdtable":
            out.append(payload)
        elif kind == "table":
            out.append(latex_table_to_html(payload))
        elif kind == "math":
            out.append(f"<div class='eqn'>{html.escape(payload)}</div>")
    for fn in extra_figs:
        out.append(fig_html(fn, ""))
    return "\n".join(out), meta_md


def meta_box(meta_md: str) -> str:
    if not meta_md.strip():
        return ""
    body = clean_md(meta_md)
    body = re.sub(r"^## (.+)$", r"<strong>\1</strong>", body, flags=re.M)
    body = re.sub(r"^- ", "• ", body, flags=re.M)
    body = body.replace("\n", "<br>")
    return (f"<details class='meta'><summary>reviewer notes &amp; open decisions "
            f"(not part of the draft)</summary><div>{body}</div></details>")


def panel(title, level, left_html, right_html, note="", meta_md=""):
    tag = "h2" if level == "section" else "h3"
    note_html = f"<div class='note'>{note}</div>" if note else ""
    return (f"<section class='lvl-{level}'><{tag}>{html.escape(title)}</{tag}>{note_html}"
            f"<div class='cols'><div class='orig'><div class='lbl'>current main.tex</div>{left_html}</div>"
            f"<div class='newcol'><div class='lbl'>draft to edit</div>{right_html}</div></div>"
            f"{meta_box(meta_md)}</section>")


def build():
    units = parse_main_tex()
    # Present architecture before the synthetic population (new reading order).
    titles = [u["title"] for u in units]
    if all(t in titles for t in SWAP_ORDER):
        i, j = titles.index(SWAP_ORDER[0]), titles.index(SWAP_ORDER[1])
        units[i], units[j] = units[j], units[i]
    panels = []
    for u in units:
        title, level = u["title"], u["level"]
        body_no_fig, figs = extract_figs(u["body"])
        left = "".join(f"<p>{latex_to_html(p)}</p>" for p in split_paras(body_no_fig))
        left += "".join(fig_html(f, c) for f, c in figs)
        # Figures kept in this section travel to the new-draft column too, so the
        # right side reads as a figure-complete paper (e.g. fig 1 in Methods).
        right_figs = "".join(fig_html(f, c) for f, c in figs)

        if title in DRAFT_FOR_TITLE:                         # §3.4, §3.5: orig + new draft
            right, meta = draft_column(DRAFT_FOR_TITLE[title], [])
            # the validation draft carries its own figure; calibration has none,
            # so only graft the original figs the draft doesn't already include.
            if not re.search(r"<img", right):
                right += right_figs
            panels.append(panel(title, level, left, right,
                                note="New draft on the right replaces this section.",
                                meta_md=meta))
        elif title in PAPER2_TITLES:                          # cut policy → Paper 2
            panels.append(panel(title, level, left,
                                "<p class='none'>Cut from Paper 1 → Paper 2. Not for editing here.</p>",
                                note="→ Paper 2 (policy material)."))
        else:                                                 # editable copy of original
            right = "".join(editable_para(latex_to_html(p)) for p in split_paras(body_no_fig))
            right += right_figs
            panels.append(panel(title, level, left, right))

        # brand-new sections inserted after this unit
        for ins_title, ins_md, ins_figs in INSERT_AFTER.get(title, []):
            r_html, r_meta = draft_column(ins_md, ins_figs)
            panels.append(panel(ins_title, "subsection",
                                "<p class='none'>New section — no prior text in main.tex.</p>",
                                r_html, note="Brand-new section drafted this round.",
                                meta_md=r_meta))
        if title == "Results":
            for ins_title, ins_md, ins_figs in INSERT_AFTER["__RESULTS__"]:
                r_html, r_meta = draft_column(ins_md, ins_figs)
                panels.append(panel(ins_title, "subsection",
                                    "<p class='none'>New section — replaces cut policy Results.</p>",
                                    r_html, note="Brand-new section drafted this round.",
                                    meta_md=r_meta))

    doc = TEMPLATE.replace("{{BODY}}", "\n".join(panels))
    (PAPER / "draft_review.html").write_text(doc)
    print(f"wrote {PAPER / 'draft_review.html'}  ({_pid} editable paragraphs, {len(units)} main.tex units)")


TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>Paper 1 — full draft review (side by side)</title>
<script>
  window.MathJax = {tex:{tags:'ams',inlineMath:[['$','$']],displayMath:[['\\[','\\]']]},
                    options:{ignoreHtmlClass:'orig|cite'}};
</script>
<script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
<style>
  :root{--green:#2E8B57;}
  body{font:16px/1.6 Georgia,serif;margin:0;color:#1a1a1a;background:#faf9f7;}
  header{position:sticky;top:0;z-index:10;background:#fff;border-bottom:1px solid #ddd;padding:10px 20px;
         display:flex;gap:12px;align-items:center;flex-wrap:wrap;font-family:system-ui,sans-serif;}
  header h1{font-size:15px;margin:0 8px 0 0;}
  header button{font:13px system-ui;padding:6px 12px;border:1px solid #bbb;border-radius:6px;background:#f4f4f4;cursor:pointer;}
  header button:hover{background:#e9e9e9;}
  #prog{font:13px system-ui;color:#555;margin-left:auto;}
  nav{font:12px system-ui;background:#fff;padding:6px 20px;border-bottom:1px solid #eee;line-height:1.9;}
  nav a{color:#357;text-decoration:none;margin-right:12px;} nav a:hover{text-decoration:underline;}
  section{padding:16px 20px;border-bottom:1px solid #eee;}
  section.lvl-section{background:#fcfbf9;}
  h2{font-family:system-ui,sans-serif;font-size:21px;border-left:5px solid var(--green);padding-left:10px;margin:.2em 0;}
  h3{font-family:system-ui,sans-serif;font-size:16px;color:#333;border-left:3px solid #9bbfa9;padding-left:9px;}
  h4{font-family:system-ui,sans-serif;font-size:14px;color:#555;margin:12px 0 5px;}
  .note{font:12px system-ui;color:#777;font-style:italic;margin:2px 0 8px;}
  .cols{display:grid;grid-template-columns:1fr 1fr;gap:20px;align-items:start;}
  .lbl{font:11px system-ui;text-transform:uppercase;letter-spacing:.06em;color:#aaa;margin-bottom:6px;}
  .orig{background:#f3f1ec;border:1px solid #e5e1d8;border-radius:8px;padding:10px 15px;color:#555;font-size:15px;}
  .orig p{margin:.45em 0;} .none{font-style:italic;color:#999;}
  .prow{display:flex;gap:8px;align-items:flex-start;margin:0 0 11px;}
  .done{font:11px system-ui;color:#aaa;white-space:nowrap;padding-top:4px;user-select:none;}
  .para{flex:1;background:#fff;border:1px solid #e0e0e0;border-radius:8px;padding:9px 13px;outline:none;}
  .para:focus{border-color:var(--green);box-shadow:0 0 0 2px rgba(46,139,87,.12);}
  .prow.is-done .para{background:#f1f8f3;border-color:var(--green);}
  .cite{color:#b23;font-family:system-ui;font-size:.8em;}
  figure{margin:14px 0;text-align:center;position:relative;}
  figure img{max-width:100%;border:1px solid #ddd;border-radius:6px;}
  figcaption{font:12px system-ui;color:#666;margin-top:5px;max-width:760px;margin-inline:auto;}
  .badge{display:inline-block;font:11px system-ui;padding:2px 8px;border-radius:10px;margin-bottom:5px;}
  .badge.stale{background:#fbe3e3;color:#a22;} .badge.p2{background:#eee;color:#777;} .badge.ok{background:#e7f3ec;color:#2E8B57;}
  table.data{border-collapse:collapse;margin:12px 0;font:13px system-ui;}
  table.data th,table.data td{border:1px solid #ccc;padding:4px 9px;text-align:right;}
  table.data th:first-child,table.data td:first-child{text-align:left;}
  table.data thead{background:#eef3ef;}
  .hint{font:12px system-ui;color:#888;padding:8px 20px;background:#fff;}
  details.meta{margin:8px 0 2px;font:12px system-ui;color:#777;background:#fbfaf6;border:1px solid #ece8de;border-radius:8px;padding:4px 12px;}
  details.meta summary{cursor:pointer;color:#998;font-style:italic;}
  details.meta > div{padding:8px 2px 4px;line-height:1.55;}
  .eqn{overflow-x:auto;margin:10px 0;padding:4px 0;font-size:15px;}
  .tabcap{font:12px system-ui;color:#666;margin:10px 0 4px;max-width:820px;}
</style></head><body>
<header>
  <h1>Paper 1 — full draft review</h1>
  <button onclick="exportMd()">⬇ Export edited Markdown</button>
  <button onclick="toggleOrig()">⇄ Toggle original</button>
  <button onclick="resetAll()">↺ Reset edits</button>
  <span id="prog"></span>
</header>
<div class="hint"><b>New reading order:</b> study area (lead-in) → model architecture + demand equations → data &amp; synthetic population → calibration → sensitivity → validation across cities. Left = current <code>main.tex</code> (still v4 order, shown per section for reference). Right = draft to edit (autosaves; tick "done" as you go). Equations render via MathJax; reviewer notes sit in the collapsed box under each new section. See <code>PAPER_STRUCTURE_v2.md</code> for the full blueprint. Keep this file in the paper folder so figures load.</div>
<nav id="toc"></nav>
{{BODY}}
<script>
const KEY='paper1_full_review_v1';
function load(){try{return JSON.parse(localStorage.getItem(KEY)||'{}');}catch(e){return {};}}
function save(s){localStorage.setItem(KEY,JSON.stringify(s));}
let store=load();
document.querySelectorAll('.para').forEach(p=>{const id=p.dataset.id;if(store[id]&&store[id].html!=null)p.innerHTML=store[id].html;});
document.querySelectorAll('[data-done]').forEach(c=>{const id=c.dataset.done;if(store[id]&&store[id].done){c.checked=true;c.closest('.prow').classList.add('is-done');}});
document.querySelectorAll('.para').forEach(p=>p.addEventListener('input',()=>{const id=p.dataset.id;store[id]=Object.assign({},store[id],{html:p.innerHTML});save(store);}));
document.querySelectorAll('[data-done]').forEach(c=>c.addEventListener('change',()=>{const id=c.dataset.done;c.closest('.prow').classList.toggle('is-done',c.checked);store[id]=Object.assign({},store[id],{done:c.checked});save(store);prog();}));
// TOC
const toc=document.getElementById('toc');
document.querySelectorAll('section').forEach((s,i)=>{const h=s.querySelector('h2,h3');if(!h)return;const id='sec'+i;s.id=id;const a=document.createElement('a');a.href='#'+id;a.textContent=h.textContent;if(h.tagName==='H3')a.style.marginLeft='14px';toc.appendChild(a);});
function prog(){const t=document.querySelectorAll('.para').length,d=document.querySelectorAll('[data-done]:checked').length;document.getElementById('prog').textContent=`${d}/${t} paragraphs done`;}
function toggleOrig(){document.querySelectorAll('.cols').forEach(c=>{c.style.gridTemplateColumns=c.style.gridTemplateColumns==='1fr'?'1fr 1fr':'1fr';});}
function resetAll(){if(confirm('Discard all edits in this browser?')){localStorage.removeItem(KEY);location.reload();}}
function exportMd(){let md='# Paper 1 — edited draft\n\n';
 document.querySelectorAll('section').forEach(sec=>{const h=sec.querySelector('h2,h3');md+=(h&&h.tagName==='H2'?'## ':'### ')+(h?h.textContent:'')+'\n\n';
  sec.querySelectorAll('.newcol > *').forEach(el=>{if(el.tagName==='H4')md+='**'+el.textContent+'**\n\n';
   else if(el.classList.contains('prow'))md+=el.querySelector('.para').innerText.trim()+'\n\n';
   else if(el.tagName==='FIGURE'){const im=el.querySelector('img');md+=`![fig](${im.getAttribute('src')})\n\n`;}
   else if(el.tagName==='TABLE')md+='[table: five-city transfer]\n\n';});});
 const b=new Blob([md],{type:'text/markdown'});const a=document.createElement('a');a.href=URL.createObjectURL(b);a.download='paper1_edited_draft.md';a.click();}
prog();
</script>
</body></html>
"""

if __name__ == "__main__":
    build()
