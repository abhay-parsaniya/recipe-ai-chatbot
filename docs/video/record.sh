#!/usr/bin/env bash
# Capture the virtual display while the choreography drives the real UI.
set -u
SCR="/tmp/claude-1000/-home-bacancy-Desktop-recipe-ai-chatbot/c605e30b-dce3-4864-85ab-2fa6f7539338/scratchpad"
FF="$("$SCR/vidvenv/bin/python" -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')"

rm -f "$SCR/screen_raw.mp4"

# Recorder runs for a little longer than the run; we trim to the audio later.
"$FF" -y -loglevel error -f x11grab -framerate 25 -video_size 1920x1080 \
      -i :99.0 -t 123 -c:v libx264 -preset veryfast -crf 20 -pix_fmt yuv420p \
      "$SCR/screen_raw.mp4" &
FFPID=$!

sleep 1.0                      # known lead-in, trimmed off during the mux
echo "LEAD_IN=1.0"

"$SCR/vidvenv/bin/python" "$SCR/choreograph.py" "$SCR"
RC=$?

wait "$FFPID" 2>/dev/null
echo "choreography rc=$RC"
ls -lh "$SCR/screen_raw.mp4" | awk '{print "raw capture: "$5}'
