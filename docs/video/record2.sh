#!/usr/bin/env bash
set -u
SCR="/tmp/claude-1000/-home-bacancy-Desktop-recipe-ai-chatbot/c605e30b-dce3-4864-85ab-2fa6f7539338/scratchpad"
FF="$("$SCR/vidvenv/bin/python" -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"
rm -f "$SCR/screen2_raw.mp4"
"$FF" -y -loglevel error -f x11grab -framerate 25 -video_size 1920x1080 \
      -i :99.0 -t 130 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
      "$SCR/screen2_raw.mp4" &
FFPID=$!
sleep 1.0
cd "$SCR" && "$SCR/vidvenv/bin/python" "$SCR/choreograph2.py" "$SCR"
wait "$FFPID" 2>/dev/null
ls -lh "$SCR/screen2_raw.mp4" | awk '{print "raw: "$5}'
