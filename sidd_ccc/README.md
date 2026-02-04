# Exam Cheating Detection Pipeline

7-stage computer vision pipeline that detects cheating behavior in exam CCTV footage.

## Quick Start

```bash
pip install -r requirements.txt
python run_pipeline.py --video path/to/exam.mp4
```

## Pipeline Stages

1. **Preprocessing** - Frame extraction at 5fps, brightness normalization
2. **Detection & Tracking** - YOLOv8m + BoT-SORT persistent tracking
3. **Pose & Head Estimation** - YOLOv8-Pose + solvePnP head orientation
4. **Spatial Analysis** - Seat assignment, neighbor graph, teacher identification
5. **Behavioral Features** - Head direction, mutual gaze, whispering posture, paper sharing
6. **Temporal Classification** - Sliding window scoring + 5-criteria AND-filter
7. **Alert Generation** - Confidence tiers, annotated video, evidence frames, PDF report

## Outputs

- `outputs/annotated_video.mp4` - Video with bounding boxes and head pose arrows
- `outputs/report.pdf` - Human-readable report with evidence
- `outputs/evidence/` - Annotated screenshots of suspicious moments
- `outputs/summary.json` - Machine-readable results
- `outputs/detection_log.jsonl` - Per-frame audit trail

## Configuration

All parameters in `config/settings.yaml`. Key tuning knobs:

- `classifier.min_suspicious_windows` - Raise to reduce false positives
- `classifier.min_confidence` - Raise for stricter filtering
- `head_pose.yaw_looking_threshold` - Lower catches more subtle looks
- `classifier.weights.*` - Adjust behavior importance

## Usage Examples

```bash
# Basic usage
python run_pipeline.py --video exam_recording.mp4

# Custom config and output directory
python run_pipeline.py --video exam.mp4 --config my_config.yaml --output results/

# Process with specific settings
python run_pipeline.py --video classroom.mp4 --output ./analysis_output
```

## Requirements

- Python 3.10+
- CUDA-capable GPU recommended (will fall back to CPU)
- ~4GB disk space for models (auto-downloaded on first run)

## Project Structure

```
exam_cheating_detection/
├── config/
│   └── settings.yaml       # All tunable parameters
├── core/
│   ├── preprocessing.py    # Stage 1: Frame extraction
│   ├── detector.py         # Stage 2: Person detection & tracking
│   ├── pose_estimator.py   # Stage 3: Body & head pose
│   ├── spatial.py          # Stage 4: Seat assignment & neighbors
│   ├── features.py         # Stage 5: Behavioral features
│   ├── classifier.py       # Stage 6: Temporal classification
│   └── alert_generator.py  # Stage 7: Output generation
├── utils/
│   ├── visualization.py    # Drawing functions
│   └── report_generator.py # PDF report generation
├── run_pipeline.py         # Main entry point
└── requirements.txt        # Dependencies
```
