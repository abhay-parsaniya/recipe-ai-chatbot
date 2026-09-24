"""Render the real source files as one scrollable, highlighted page."""
import html, sys
from pathlib import Path
from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

ROOT = Path("/home/bacancy/Desktop/recipe-ai-chatbot")
SCR = Path(sys.argv[1])

SHOTS = [
    ("overview", "Architecture", None, None, None,
     """<div class="arch">
       <div class="layer"><b>src/data</b><span>load RecipeNLG · clean · weight the searchable text</span></div>
       <div class="arrow">↓</div>
       <div class="layer"><b>src/search</b><span>TF-IDF · cosine · coverage boost · title bonus · pantry matcher</span></div>
       <div class="arrow">↓</div>
       <div class="layer"><b>src/chatbot</b><span>guardrails · intents · history · response wording</span></div>
       <div class="arrow">↓</div>
       <div class="layer two"><b>src/api</b><span>FastAPI</span><b>app/</b><span>Streamlit</span></div>
       <div class="note">src/llm sits alongside — optional, grounded, degrades to templates</div>
     </div>"""),
    ("preprocess", "Weighting the searchable text",
     "src/data/preprocess.py", 54, 76, None),
    ("engine", "Scoring: cosine × coverage × title bonus",
     "src/search/engine.py", 203, 222, None),
    ("guardrails", "Domain boundary",
     "src/chatbot/guardrails.py", 152, 181, None),
    ("evals", "Measured, not guessed",
     "scripts/eval_search.py", 26, 44, None),
]

fmt = HtmlFormatter(style="github-dark", linenos=False, nowrap=True)
parts = []
for anchor, title, rel, a, b, custom in SHOTS:
    if custom:
        body = custom
        sub = "data → search → chat → interfaces"
    else:
        lines = (ROOT / rel).read_text().splitlines()[a-1:b]
        code = highlight("\n".join(lines), PythonLexer(), fmt)
        numbered = "".join(
            f'<tr><td class="ln">{n}</td><td class="cd">{ln or "&nbsp;"}</td></tr>'
            for n, ln in zip(range(a, b+1), code.split("\n")))
        body = f'<table class="code">{numbered}</table>'
        sub = f"{rel}  ·  lines {a}–{b}"
    parts.append(f'<section id="{anchor}"><h1>{html.escape(title)}</h1>'
                 f'<div class="sub">{html.escape(sub)}</div>{body}</section>')

page = f"""<!doctype html><meta charset="utf-8"><title>Recipe Chatbot — code</title>
<style>
 {fmt.get_style_defs('.highlight')}
 *{{box-sizing:border-box}}
 body{{margin:0;background:#0d1117;color:#c9d1d9;
      font:16px/1.55 "DejaVu Sans Mono",monospace}}
 section{{min-height:100vh;padding:58px 70px}}
 h1{{font-size:38px;margin:0 0 6px;color:#e6edf3}}
 .sub{{color:#8b949e;font-size:19px;margin-bottom:30px}}
 table.code{{border-collapse:collapse;font-size:19.5px;width:100%}}
 td.ln{{color:#484f58;text-align:right;padding-right:22px;width:62px;
        user-select:none}}
 td.cd{{white-space:pre;padding:1px 0}}
 .arch{{margin-top:36px}}
 .layer{{background:#161b22;border:1px solid #30363d;border-radius:10px;
         padding:20px 26px;font-size:22px}}
 .layer b{{color:#58a6ff;margin-right:20px}}
 .layer span{{color:#8b949e;font-size:19px}}
 .layer.two b:nth-of-type(2){{margin-left:44px}}
 .arrow{{color:#484f58;text-align:center;font-size:26px;margin:9px 0}}
 .note{{margin-top:26px;color:#e3b341;font-size:19px}}
</style>{''.join(parts)}"""

out = SCR / "code.html"
out.write_text(page)
print("wrote", out, f"{out.stat().st_size/1024:.0f} KB,", len(SHOTS), "sections")
