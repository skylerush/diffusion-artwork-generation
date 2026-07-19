#!/usr/bin/env bash
# Show all project metrics tables.
#   From a terminal:   ./metrics.sh
#   Double-clicking also works: the window PAUSES at the end instead of closing,
#   and a copy of the output is always saved to  experiments/metrics_latest.txt
cd "$(dirname "$0")"
PYTHONIOENCODING=utf-8 ./.venv/Scripts/python.exe src/common/show_metrics.py | tee experiments/metrics_latest.txt
echo
echo "(saved a copy to experiments/metrics_latest.txt)"
# Pause only when attached to a real keyboard (double-click / interactive terminal),
# so piping or scripted use never blocks:
if [ -t 0 ]; then
    read -r -p "Press Enter to close..."
fi
