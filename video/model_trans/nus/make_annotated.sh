#!/bin/bash
set -e
cd "/Users/zhanqianwu/Documents/工作/工作文档/Giga/Giga_world_1/github_page/video/model_trans/nus"

SRC="combined_hstack_synced.mp4"
OUT="combined_annotated.mp4"
FONT="/System/Library/Fonts/Helvetica.ttc"

ffmpeg -y -i "$SRC" \
  -filter_complex "[0:v]drawbox=x=10:y=10:w=520:h=130:color=black@0.6:t=fill,\
drawtext=fontfile=${FONT}:text='▶  FORWARD':fontsize=38:fontcolor=0x33ff77:x=28:y=24:enable='lt(mod(t\,46)\,23)',\
drawtext=fontfile=${FONT}:text='◀  BACKWARD':fontsize=38:fontcolor=0xff5577:x=28:y=24:enable='gte(mod(t\,46)\,23)',\
drawtext=fontfile=${FONT}:text='TIME\: %{pts\:hms}    FRAME\: %{eif\\:n\\:d}':fontsize=22:fontcolor=white:x=28:y=88[v]" \
  -map "[v]" -an \
  -c:v libx264 -preset fast -crf 18 -movflags +faststart \
  "$OUT" 2>&1 | tail -5

echo "--- DONE ---"
ls -la "$OUT"