"""Build the 2-minute demo video: real code, real output, TTS narration."""
from __future__ import annotations
import json, os, re, subprocess, sys, textwrap
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
import imageio_ffmpeg

ROOT = Path("/home/bacancy/Desktop/recipe-ai-chatbot")
SCR = Path(sys.argv[1])
OUT = SCR / "frames"; OUT.mkdir(exist_ok=True)
AUD = SCR / "audio"; AUD.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

W, H, FPS = 1920, 1080, 10
BG, FG = (13, 17, 23), (201, 209, 217)
DIM, ACC, GRN, YEL, RED = (110,118,129), (88,166,255), (86,211,100), (227,179,65), (248,113,113)
MONO = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
MONOB = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
F   = ImageFont.truetype(MONO, 23)
FB  = ImageFont.truetype(MONOB, 23)
FT  = ImageFont.truetype(MONOB, 34)
FS  = ImageFont.truetype(MONO, 19)
LH  = 31

KEYWORDS = {"def","class","return","if","not","for","in","import","from","and",
            "or","None","True","False","self","elif","else","while","with","as"}


def chrome(d, title, subtitle=""):
    d.rectangle([0, 0, W, 64], fill=(22, 27, 34))
    for i, c in enumerate([(255,95,86),(255,189,46),(39,201,63)]):
        d.ellipse([28+i*26, 25, 42+i*26, 39], fill=c)
    d.text((124, 20), title, font=FB, fill=FG)
    if subtitle:
        d.text((124 + FB.getlength(title) + 24, 22), subtitle, font=F, fill=DIM)
    d.line([0, 64, W, 64], fill=(48, 54, 61), width=2)


def code_frame(path_label, lines, first_line=1, highlight=(), caption=""):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    chrome(d, path_label, "")
    y = 96
    for n, raw in enumerate(lines, first_line):
        if y > H - 150: break
        hot = n in highlight
        if hot:
            d.rectangle([84, y-5, W-60, y+LH-6], fill=(28, 40, 58))
        d.text((22, y), f"{n:>4}", font=FS, fill=(72, 79, 88))
        x = 100
        for tok in re.split(r"(\W)", raw.rstrip("\n")):
            if not tok: continue
            col = FG
            if tok in KEYWORDS: col = (255, 123, 114)
            elif tok.startswith('"') or tok.startswith("'"): col = (165, 214, 255)
            elif raw.lstrip().startswith("#") or raw.lstrip().startswith('"""'): col = (139, 148, 158)
            elif tok.isdigit(): col = (121, 192, 255)
            d.text((x, y), tok, font=FB if hot else F, fill=ACC if hot and tok not in KEYWORDS else col)
            x += F.getlength(tok)
        y += LH
    if caption:
        d.rectangle([0, H-108, W, H], fill=(22, 27, 34))
        d.line([0, H-108, W, H-108], fill=(48,54,61), width=2)
        d.text((44, H-74), caption, font=FT, fill=YEL)
    return img


def term_frame(title, blocks, caption=""):
    """blocks: list of (kind, text) with kind in {you, bot, note}"""
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    chrome(d, title)
    y = 100
    for kind, text in blocks:
        if kind == "you":
            d.text((44, y), "you ▸ ", font=FB, fill=GRN)
            d.text((44 + FB.getlength("you ▸ "), y), text, font=FB, fill=FG)
            y += LH + 8
        elif kind == "note":
            d.text((44, y), text, font=F, fill=YEL); y += LH
        else:
            for ln in text.split("\n"):
                if y > H - 130: break
                col = ACC if re.match(r"^\s*\d+\. ", ln) else FG
                if "[x]" in ln: col = GRN
                if "[ ]" in ln or "missing" in ln: col = RED
                d.text((60, y), ln[:104], font=F, fill=col)
                y += LH
            y += 6
    if caption:
        d.rectangle([0, H-108, W, H], fill=(22, 27, 34))
        d.line([0, H-108, W, H-108], fill=(48,54,61), width=2)
        d.text((44, H-74), caption, font=FT, fill=YEL)
    return img


def title_frame(main, sub, bullets):
    img = Image.new("RGB", (W, H), BG); d = ImageDraw.Draw(img)
    big = ImageFont.truetype(MONOB, 62); mid = ImageFont.truetype(MONO, 30)
    d.text((110, 250), main, font=big, fill=FG)
    d.text((110, 340), sub, font=mid, fill=ACC)
    y = 470
    for b in bullets:
        d.text((130, y), "▸", font=mid, fill=GRN)
        d.text((175, y), b, font=mid, fill=(180, 190, 200)); y += 58
    return img


def src(rel, start, end):
    lines = (ROOT / rel).read_text().splitlines()
    return lines[start-1:end], start


def narrate(idx, text):
    mp3 = AUD / f"{idx:02d}.mp3"
    if not mp3.exists():
        gTTS(text=text, lang="en", tld="co.uk").save(str(mp3))
    out = subprocess.run([FFMPEG, "-i", str(mp3)], capture_output=True, text=True).stderr
    m = re.search(r"Duration: (\d+):(\d+):([\d.]+)", out)
    h, mi, s = m.groups()
    return mp3, int(h)*3600 + int(mi)*60 + float(s)


turns = {t["q"]: t["a"] for t in json.loads((SCR / "turns.json").read_text())}

# ---- segments: (narration, frame builder) --------------------------------
def seg_intro():
    return title_frame("Recipe Recommendation Chatbot",
                       "372,213 recipes · TF-IDF retrieval · no LLM required",
                       ["data  →  search  →  chat  →  API / UI",
                        "351 tests · 93% coverage",
                        "precision@5  0.895   ·   MRR  0.950"])

def seg_preprocess():
    lines, start = src("src/data/preprocess.py", 54, 59)
    body, bstart = src("src/data/preprocess.py", 136, 146)
    return code_frame("src/data/preprocess.py", lines + [""] + body,
                      first_line=start,
                      highlight={58},
                      caption="NER x3  ·  title x2  ·  ingredients x1")

def seg_engine():
    lines, start = src("src/search/engine.py", 203, 222)
    return code_frame("src/search/engine.py  —  search()", lines,
                      first_line=start,
                      highlight={212, 213, 214, 216},
                      caption="cosine  x  coverage  x  title bonus")

def seg_guardrails():
    lines, start = src("src/chatbot/guardrails.py", 163, 181)
    return code_frame("src/chatbot/guardrails.py  —  classify()", lines,
                      first_line=start,
                      highlight={167, 172, 175},
                      caption="food signal wins  ·  unknown questions fail open")

def seg_eval():
    return term_frame("terminal  —  make check", [
        ("you", "pytest -q"),
        ("bot", "351 passed in 126s"),
        ("you", "python -m scripts.eval_search"),
        ("bot", "precision@k      0.895   (relevant results per page)\n"
                "answered         0.950   (cases with >=1 relevant hit)\n"
                "MRR              0.950   (how high the first hit lands)\n"
                "exclusion leaks  0       (must be 0)\n"
                "latency          70 ms median, 88 ms p95"),
    ], caption="every ranking change is measured, not guessed")

Q1 = "what can I make with chicken, tomato and onion?"
def seg_search():
    return term_frame("Recipe Assistant", [("you", Q1), ("bot", turns[Q1])],
                      caption="tells you what each recipe needs BEYOND your query")

def seg_followup():
    return term_frame("Recipe Assistant", [
        ("you", "2"), ("bot", turns["2"]),
        ("you", "how do I make it?"), ("bot", turns["how do I make it?"]),
    ], caption='it knows what "it" refers to')

def seg_guard_demo():
    q = "Who is the president of India?"
    return term_frame("Recipe Assistant", [("you", q), ("bot", turns[q])],
                      caption="out of domain → refuses instead of guessing")

def seg_api():
    api = json.loads((SCR / "api.json").read_text())
    body = json.dumps(api["pantry"], indent=1)
    return term_frame("terminal  —  the same engine over HTTP", [
        ("you", 'curl -s localhost:8000/pantry -d \'{"have":["chicken","onion","tomato","garlic"],"exclude":["cheese"]}\''),
        ("bot", body),
    ], caption="FastAPI · /search /pantry /chat /recipes /health")


def seg_pantry():
    k = "[sidebar] have: chicken, onion, tomato, garlic  |  not: cheese"
    return term_frame("Recipe Assistant  —  what's in your kitchen?",
                      [("you", k), ("bot", turns[k])],
                      caption="ranks by how much of the recipe YOU already have")

SEGMENTS = [
 ("A recipe chatbot over the RecipeNLG dataset. Three hundred and seventy two "
  "thousand recipes, searched locally, with no language model needed.",
  seg_intro),
 ("The data layer builds one searchable field per recipe. Ingredient names "
  "count three times, the title twice. So a recipe is described by what makes "
  "it unusual. Saffron counts. Water doesn't.", seg_preprocess),
 ("Retrieval is T F I D F with cosine similarity, plus two corrections. "
  "A coverage boost, because cosine favours short recipes. And a title bonus: "
  "without it, banana bread returns a banana sandwich.", seg_engine),
 ("Above that sits the chat layer. Rule based intents, conversation memory, "
  "and a guardrail that refuses anything outside cooking.", seg_guardrails),
 ("All of it is measured. Three hundred and fifty one tests, and a retrieval "
  "eval scoring precision at five of point eight nine five.", seg_eval),
 ("Asking in plain English. It strips the filler and answers conversationally. "
  "Notice it tells me what each recipe needs beyond what I asked for.",
  seg_search),
 ("It holds context. I pick one by number, then ask how to make it. It knows "
  "what it refers to, and gives the real method from the dataset.",
  seg_followup),
 ("Out of domain questions are refused, not guessed at. That boundary is "
  "measured on sixty cases.", seg_guard_demo),
 ("And the main feature. I tell it what is in my kitchen. It ranks by how much "
  "of the recipe I already have, ticks off what I have got, and names what is "
  "missing. Cheese is excluded outright.", seg_pantry),
 ("The same engine is exposed over H T T P. FastAPI serves search, pantry, "
  "chat and health, so the Streamlit interface is just one client among "
  "several.", seg_api),
]

if __name__ == "__main__":
    manifest, total = [], 0.0
    for i, (text, builder) in enumerate(SEGMENTS):
        mp3, dur = narrate(i, text)
        img = builder()
        png = OUT / f"seg{i:02d}.png"
        img.save(png)
        manifest.append({"png": str(png), "mp3": str(mp3), "dur": dur})
        total += dur
        print(f"  seg {i}: {dur:5.1f}s  {png.name}")
    (SCR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"\ntotal narration: {total:.1f}s  ({total/60:.2f} min)")
