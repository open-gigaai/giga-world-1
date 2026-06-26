pkill -9 -f 0_get_aesthetic.py
pkill -9 -f 1_get_motion_amplitude.py
pkill -9 -f 2_get_motion_smoothness.py
pkill -9 -f 3_get_semantic.py
pkill -9 -f 4_get_naturalness.py

pkill -9 -f 0_get_drifting_aesthetic.py
pkill -9 -f 1_get_drifting_motion_smoothness.py
pkill -9 -f 2_get_drifting_semantic.py
pkill -9 -f 3_get_drifting_naturalness.py

pkill -9 -f run_metrics.sh
pkill -9 -f run_metrics_ddp.sh