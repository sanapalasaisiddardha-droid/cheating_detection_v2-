"""
Exam Cheating Detection Pipeline - Main Entry Point

Usage:
  python run_pipeline.py --video path/to/exam.mp4 [--config config/settings.yaml] [--output outputs/]
"""

import argparse
import os
import sys
import time
import json

import yaml
import cv2
import numpy as np

from core.preprocessing import VideoPreprocessor
from core.detector import PersonDetector
from core.pose_estimator import PoseEstimator, HeadPoseEstimator
from core.spatial import SpatialAnalyzer
from core.features import BehaviorFeatureExtractor
from core.classifier import TemporalClassifier
from core.alert_generator import AlertGenerator
from utils.visualization import draw_detection, draw_head_pose_arrow, draw_info_overlay
from utils.report_generator import ReportGenerator


def load_config(path: str) -> dict:
    """Load configuration from YAML file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required keys
    required_keys = ['pipeline', 'preprocessing', 'detection', 'tracking',
                     'pose', 'head_pose', 'spatial', 'features', 'classifier', 'alerts']
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Missing required config section: {key}")

    return config


def compute_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute Intersection over Union of two [x1,y1,x2,y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0


def match_pose_to_detection(detections: list, pose_results: list) -> dict:
    """Match pose estimations to tracked detections using IoU overlap."""
    matched = {}

    for det in detections:
        best_iou = 0
        best_pose = None
        det_box = det["bbox"]

        for pose in pose_results:
            iou = compute_iou(det_box, pose["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_pose = pose

        if best_iou > 0.3 and best_pose is not None:
            matched[det["track_id"]] = best_pose

    return matched


def get_next_test_folder(base_output="outputs"):
    """Find the next available test folder number."""
    os.makedirs(base_output, exist_ok=True)
    existing = [d for d in os.listdir(base_output) if d.startswith("test") and os.path.isdir(os.path.join(base_output, d))]

    # Extract numbers from existing test folders
    numbers = []
    for d in existing:
        try:
            num = int(d.replace("test", ""))
            numbers.append(num)
        except ValueError:
            pass

    next_num = max(numbers) + 1 if numbers else 1
    return os.path.join(base_output, f"test{next_num}")


def main():
    parser = argparse.ArgumentParser(description="Exam Cheating Detection Pipeline")
    parser.add_argument("--video", required=True, help="Path to exam video file")
    parser.add_argument("--config", default="config/settings.yaml", help="Path to config")
    parser.add_argument("--output", default="outputs_v2", help="Base output directory")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)

    # Create auto-numbered test folder inside outputs
    output_dir = get_next_test_folder(args.output)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output folder: {output_dir}")

    print("=" * 60)
    print("  EXAM CHEATING DETECTION PIPELINE")
    print("=" * 60)

    # --- Initialize all stages ---
    print("\n[INIT] Loading models...")
    preprocessor = VideoPreprocessor(args.video, config)
    detector = PersonDetector(config)
    pose_estimator = PoseEstimator(config)
    head_pose_estimator = HeadPoseEstimator(config)
    spatial = SpatialAnalyzer(config)
    feature_extractor = BehaviorFeatureExtractor(config)
    classifier = TemporalClassifier(config)
    alert_gen = AlertGenerator(config, spatial)

    video_info = preprocessor.get_info()
    print(f"[INIT] Video: {video_info['duration']:.1f}s, "
          f"{video_info['fps']:.1f}fps, {video_info['width']}x{video_info['height']}")

    # --- Setup video writer for annotated output ---
    if config["alerts"]["save_annotated_video"]:
        annotated_path = os.path.join(output_dir, "annotated_video.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            annotated_path,
            fourcc,
            config["pipeline"]["sample_fps"],
            (config["preprocessing"]["resize_width"],
             config["preprocessing"]["resize_height"])
        )
    else:
        writer = None

    # --- Storage for evidence frames ---
    frame_buffer = {}  # frame_idx -> frame image (store every Nth for evidence)
    all_frame_data = []  # per-frame log

    # ===================================================
    # MAIN PROCESSING LOOP
    # ===================================================
    print("\n[PROCESSING] Running pipeline...")
    frame_count = 0
    t_start = time.time()

    for frame_idx, timestamp, frame in preprocessor.process():
        frame_count += 1

        # --- Stage 2: Detection & Tracking ---
        detections = detector.detect_and_track(frame)
        if not detections:
            continue

        # --- Stage 3: Pose Estimation ---
        pose_results = pose_estimator.estimate(frame)
        pose_map = match_pose_to_detection(detections, pose_results)

        # Compute head pose for each detection that has pose
        head_poses = {}
        for track_id, pose_data in pose_map.items():
            hp = head_pose_estimator.estimate_head_pose(
                pose_data["keypoints"], frame.shape
            )
            if hp is not None:
                head_poses[track_id] = hp

        # --- Stage 4: Spatial Update ---
        spatial.update(detections, frame_idx)

        # --- Stage 5: Feature Extraction ---
        student_features = {}
        for det in detections:
            tid = det["track_id"]
            if spatial.is_teacher(tid):
                continue
            seat = spatial.seats.get(tid)
            if seat is None:
                continue
            pose_data = pose_map.get(tid)
            hp = head_poses.get(tid)
            if pose_data is None or hp is None:
                continue

            sf = feature_extractor.extract_student_features(
                det, pose_data, hp, seat
            )
            student_features[tid] = sf

        # Pairwise features for all neighbor pairs
        pair_features_list = []
        processed_pairs = set()
        for tid, sf in student_features.items():
            for neighbor_id in spatial.get_neighbors(tid):
                pair_key = tuple(sorted([tid, neighbor_id]))
                if pair_key in processed_pairs:
                    continue
                processed_pairs.add(pair_key)

                if neighbor_id not in student_features:
                    continue

                pf = feature_extractor.extract_pair_features(
                    sf, student_features[neighbor_id],
                    spatial.seats[tid], spatial.seats[neighbor_id]
                )

                # Attach teacher proximity info
                teacher_pos = spatial.get_teacher_position(detections)
                if teacher_pos is not None:
                    d_to_pair = min(
                        np.linalg.norm(np.array(teacher_pos) - np.array(spatial.seats[tid])),
                        np.linalg.norm(np.array(teacher_pos) - np.array(spatial.seats[neighbor_id]))
                    )
                    pf["teacher_distance"] = d_to_pair
                else:
                    pf["teacher_distance"] = float('inf')

                pair_features_list.append(pf)

        # --- Stage 6: Feed to classifier ---
        classifier.add_frame(pair_features_list, frame_idx, timestamp)

        # --- Annotate frame ---
        vis_frame = frame.copy()
        for det in detections:
            tid = det["track_id"]
            color = (0, 255, 0)  # green default
            label = spatial.get_seat_label(tid)
            if spatial.is_teacher(tid):
                color = (255, 200, 0)  # cyan for teacher
                label = "TEACHER"
            draw_detection(vis_frame, det["bbox"], tid, color, label)

            # Draw head pose arrow
            if tid in head_poses and tid in pose_map:
                nose = pose_map[tid]["keypoints"][0][:2]
                if nose[0] > 0 and nose[1] > 0:
                    hp = head_poses[tid]
                    arrow_color = (0, 0, 255) if hp["is_looking_sideways"] else (0, 255, 0)
                    draw_head_pose_arrow(vis_frame, nose, hp["direction_vector"],
                                         color=arrow_color)

        draw_info_overlay(vis_frame, timestamp, frame_idx, [])

        if writer is not None:
            writer.write(vis_frame)

        # Store frame and detection data for evidence (every 5th processed frame)
        if frame_count % 5 == 0:
            frame_buffer[frame_idx] = {
                "frame": frame.copy(),
                "detections": detections,
                "timestamp": timestamp,
                "head_poses": head_poses,
                "pose_map": pose_map,
            }

        # Log
        all_frame_data.append({
            "frame_idx": frame_idx,
            "timestamp": timestamp,
            "num_detections": len(detections),
            "num_students": len(student_features),
            "num_pairs": len(pair_features_list),
        })

        # Progress
        if frame_count % 50 == 0:
            elapsed = time.time() - t_start
            fps_actual = frame_count / elapsed
            print(f"  Frame {frame_count} | {timestamp:.1f}s | {fps_actual:.1f} fps")

    if writer is not None:
        writer.release()

    elapsed = time.time() - t_start
    print(f"\n[PROCESSING] Done. {frame_count} frames in {elapsed:.1f}s "
          f"({frame_count/elapsed:.1f} fps)")

    # ===================================================
    # CLASSIFICATION
    # ===================================================
    print("\n[CLASSIFY] Running temporal analysis...")
    classifications = classifier.classify_all()

    cheating_pairs = {k: v for k, v in classifications.items() if v["is_cheating"]}
    print(f"[CLASSIFY] Found {len(cheating_pairs)} cheating pair(s)")

    for pair_id, result in cheating_pairs.items():
        labels = (spatial.get_seat_label(pair_id[0]),
                  spatial.get_seat_label(pair_id[1]))
        print(f"  -> {labels[0]} & {labels[1]}: "
              f"confidence={result['confidence']:.2f}, "
              f"windows={result['num_suspicious_windows']}")

    # ===================================================
    # GENERATE OUTPUTS
    # ===================================================
    print("\n[OUTPUT] Generating alerts and reports...")

    alerts = alert_gen.generate_alerts(classifications, frame_buffer)

    # Save evidence frames
    if config["alerts"]["save_evidence_frames"]:
        evidence_dir = os.path.join(output_dir, "evidence")
        alert_gen.save_evidence_frames(alerts, frame_buffer, evidence_dir)

    # Save JSON log
    log_path = os.path.join(output_dir, "detection_log.jsonl")
    alert_gen.save_json_log(all_frame_data, log_path)

    # Save PDF report
    if config["alerts"]["save_report"]:
        report_path = os.path.join(output_dir, "report.pdf")
        report_gen = ReportGenerator(config)
        report_gen.generate(alerts, video_info, spatial, report_path)
        print(f"[OUTPUT] Report saved: {report_path}")

    # Save summary JSON
    summary = {
        "video": args.video,
        "duration": video_info["duration"],
        "frames_processed": frame_count,
        "processing_time": elapsed,
        "total_students": len(spatial.seats),
        "teacher_identified": spatial.teacher_id is not None,
        "cheating_pairs_found": len(cheating_pairs),
        "alerts": [
            {
                "seats": a["seat_labels"],
                "tier": a["tier"],
                "confidence": a["confidence"],
                "timestamps": a["timestamps"],
                "behaviors": a["behaviors"],
            }
            for a in alerts
        ],
    }
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"[OUTPUT] Summary saved: {summary_path}")

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE - {len(alerts)} alert(s) generated")
    print(f"  Outputs in: {output_dir}/")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
