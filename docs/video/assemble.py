"""Mux narration + frames into the final 2-minute MP4."""
import json, subprocess, sys
from pathlib import Path
import imageio_ffmpeg

SCR = Path(sys.argv[1])
FF = imageio_ffmpeg.get_ffmpeg_exe()
TARGET, GAP = 120.0, 0.30

man = json.loads((SCR / "manifest.json").read_text())
raw_total = sum(s["dur"] for s in man)
tempo = raw_total / (TARGET - GAP * len(man))
tempo = max(1.0, min(tempo, 1.25))          # never speed past intelligibility
print(f"raw narration {raw_total:.1f}s -> tempo {tempo:.3f}")

wavs, durs = [], []
for i, s in enumerate(man):
    wav = SCR / f"a{i:02d}.wav"
    subprocess.run([FF, "-y", "-loglevel", "error", "-i", s["mp3"],
                    "-filter:a", f"atempo={tempo:.4f}",
                    "-ar", "44100", "-ac", "2", str(wav)], check=True)
    sil = SCR / f"s{i:02d}.wav"
    subprocess.run([FF, "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", f"anullsrc=r=44100:cl=stereo", "-t", str(GAP),
                    str(sil)], check=True)
    wavs += [wav, sil]
    durs.append(s["dur"] / tempo + GAP)

# audio track
alist = SCR / "audio_list.txt"
alist.write_text("".join(f"file '{w}'\n" for w in wavs))
voice = SCR / "voice.wav"
subprocess.run([FF, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0",
                "-i", str(alist), "-c", "copy", str(voice)], check=True)

# video track: hold each frame for its narration
vlist = SCR / "video_list.txt"
lines = []
for s, d in zip(man, durs):
    lines.append(f"file '{s['png']}'\nduration {d:.3f}\n")
lines.append(f"file '{man[-1]['png']}'\n")      # concat demuxer needs a repeat
vlist.write_text("".join(lines))

out = SCR / "recipe-chatbot-demo.mp4"
subprocess.run([FF, "-y", "-loglevel", "error",
                "-f", "concat", "-safe", "0", "-i", str(vlist),
                "-i", str(voice),
                "-vf", "fps=25,format=yuv420p,scale=1920:1080",
                "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                "-c:a", "aac", "-b:a", "160k", "-shortest",
                str(out)], check=True)

probe = subprocess.run([FF, "-i", str(out)], capture_output=True, text=True).stderr
print([l.strip() for l in probe.splitlines() if "Duration" in l or "Stream" in l])
print("written:", out, f"{out.stat().st_size/1e6:.1f} MB")
