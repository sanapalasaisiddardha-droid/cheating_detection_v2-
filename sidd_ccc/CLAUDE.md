# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the pipeline (outputs go to outputs/test1, test2, etc.)
python run_pipeline.py --video path/to/exam.mp4

# Run with custom config
python run_pipeline.py --video exam.mp4 --config my_config.yaml

# Install dependencies
pip install -r requirements.txt
```

## Architecture

This is a 7-stage computer vision pipeline for detecting cheating in exam CCTV footage. Each stage is a separate module in `core/`:

```
Video → Preprocessing → Detection → Pose → Spatial → Features → Classifier → Alerts
         (Stage 1)      (Stage 2)   (3)     (4)        (5)         (6)        (7)
```

**Data Flow:**
1. `preprocessing.py` - Generator yields frames at configured FPS with CLAHE enhancement
2. `detector.py` - YOLOv8 + BoT-SORT returns tracked person detections with persistent IDs
3. `pose_estimator.py` - YOLOv8-Pose extracts 17 COCO keypoints; `HeadPoseEstimator` uses cv2.solvePnP (SQPNP flag) with 5 facial keypoints to compute yaw/pitch/roll
4. `spatial.py` - After warmup period, assigns seats via median positions, clusters into rows with DBSCAN (eps=35), identifies teacher by movement, builds neighbor graph
5. `features.py` - Extracts per-student features (head direction, hand position) and pairwise features (mutual gaze, whispering posture, paper sharing)
6. `classifier.py` - Sliding window analysis (10s windows, 2s stride) with weighted scoring; multi-criteria filter requires sustained behavior across multiple windows
7. `alert_generator.py` - Generates tiered alerts, saves annotated evidence frames with bounding boxes

**Key Design Decisions:**
- All thresholds live in `config/settings.yaml` - no magic numbers in code
- Frame processing uses generator pattern to avoid loading entire video into memory
- Track IDs must be checked for None (first few frames before tracking stabilizes)
- Pose-to-detection matching uses IoU > 0.3 threshold
- Seat labels use format `R{row}_C{col}` based on DBSCAN clustering of Y-coordinates

## Configuration

`config/settings.yaml` controls all parameters. Key tuning knobs for false positive reduction:
- `classifier.min_suspicious_windows` - Increase to require more evidence
- `classifier.min_confidence` - Increase threshold (0.5-0.7 typical)
- `classifier.require_mutual` - Set true to require both students showing behavior
- `spatial.seat_warmup_frames` - Must be high enough to accumulate `min_track_length` observations

## Critical Implementation Notes

- OpenCV solvePnP requires `SOLVEPNP_SQPNP` flag (not ITERATIVE) because we only have 5 facial keypoints
- Image coordinates: Y increases downward, so "elevated wrist" means wrist_y < shoulder_y
- Always convert YOLO tensors to CPU numpy: `.cpu().numpy()`
- Pair IDs must be sorted tuples: `tuple(sorted([id_a, id_b]))` for consistency
