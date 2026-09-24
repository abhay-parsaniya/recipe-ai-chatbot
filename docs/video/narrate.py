"""Generate the narration track for the LLM-enabled screencast."""
import json
import re
import subprocess
import sys
from pathlib import Path

import imageio_ffmpeg
from gtts import gTTS

SCR = Path(sys.argv[1])
AUD = SCR / "audio2"
AUD.mkdir(exist_ok=True)
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

from segments import SEGMENTS  # noqa: E402

manifest = []
total = 0.0
for i, (anchor, text) in enumerate(SEGMENTS):
    mp3 = AUD / f"{i:02d}.mp3"
    if not mp3.exists():
        gTTS(text=text, lang="en", tld="co.uk").save(str(mp3))
    err = subprocess.run([FFMPEG, "-i", str(mp3)],
                         capture_output=True, text=True).stderr
    h, m, s = re.search(r"Duration: (\d+):(\d+):([\d.]+)", err).groups()
    dur = int(h) * 3600 + int(m) * 60 + float(s)
    manifest.append({"anchor": anchor, "mp3": str(mp3), "dur": dur})
    total += dur
    print(f"  {i:2d} {anchor:<16} {dur:5.1f}s")

(SCR / "manifest2.json").write_text(json.dumps(manifest, indent=1))
code = sum(s["dur"] for s in manifest if not s["anchor"].startswith("demo"))
demo = total - code
print(f"\ncode {code:.1f}s  demo {demo:.1f}s  total {total:.1f}s")
