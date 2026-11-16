#!/usr/bin/env python3
"""
Ball Possession Classification Pipeline - Official Entry Point

This script provides the main entry point for the Ball Possession Classification pipeline.
It integrates all modules (EnhancedClassifier, data loaders) into a unified workflow.

Usage:
    python main.py --data-dir <path> [--output <path>]

    --help: Show usage information

    --output: Path to save predictions CSV (default: results/<video name>.csv)
    --data-dir: Path to tactical data directory (default: ./data/tactical)

Author: AI Agent
Date: 2026-11-15
"""

import argparse
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_classifier import EnhancedPossessionClassifier

RESULTS_DIR = Path(__file__).resolve().parent / "results"
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def default_output_path(data_dir: Path) -> Path:
    """Return results/<video name>.csv for a data folder.

    The video name is the stem of the only video file in the folder. Without
    exactly one video, it is the tracking file name without the "_atd"
    suffix (BAR-ATH_tactical_atd.txt -> BAR-ATH_tactical), which follows the
    same naming convention as the video.
    """
    data_dir = Path(data_dir)
    videos = sorted(p for p in data_dir.iterdir()
                    if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS)
    if len(videos) == 1:
        name = videos[0].stem
    else:
        tracking = sorted(data_dir.glob("*_atd.txt"))
        name = tracking[0].name[:-len("_atd.txt")] if len(tracking) == 1 else "predictions"
    return RESULTS_DIR / f"{name or 'predictions'}.csv"


class BallPossessionClassifier:
    """
    Main classifier class that orchestrates the complete ball possession classification pipeline.
    
    This class integrates all modules (EnhancedClassifier, data loaders) into a unified workflow.
    It provides a clean interface for running the complete pipeline from data loading to prediction generation.
    """
    
    def __init__(self):
        """Initialize the pipeline. Everything runs on the CPU."""
        self.classifier = EnhancedPossessionClassifier()
    
    def run_pipeline(self, data_dir: Path, output_path: Path) -> None:
        """
        Run the complete classification pipeline.
        
        Args:
            data_dir: Path to tactical data directory
            output_path: Path to save predictions CSV
        """
        print("=" * 60)
        print("BALL POSSESSION CLASSIFICATION PIPELINE")
        print("=" * 60)
        print(f"Data Directory: {data_dir}")
        print(f"Output Path: {output_path}")
        print("=" * 60)
        
        # Load tactical data
        print("\n[1/4] Loading tactical data...")
        video_frames, ball_positions, player_positions = self.classifier.load_tactical_data(data_dir)
        print(f"Loaded {len(video_frames)} video frames")
        print(f"Ball positions tracked: {sum(pos is not None for pos in ball_positions.values())} frames")
        
        # Load team assignments from ATD XML (already done in load_tactical_data)
        print("\n[2/4] Team assignments loaded...")
        print(f"Team assignments loaded for {len(self.classifier.player_teams)} players")
        
        # Run classification
        print("\n[3/4] Running classification...")
        predictions = self.classifier.classify_frames(video_frames, ball_positions, player_positions)
        stats = self.classifier.get_statistics()
        print(f"Dead-ball stretches: {stats['dead_ball_stretches']}  "
              f"(Smart Tagging restarts available: {stats['restart_tags']})")
        print(f"Classification complete. Generated {len(predictions)} predictions")
        
        # Generate submission CSV
        print("\n[4/4] Generating submission CSV...")
        self.generate_submission(predictions, output_path)
        print(f"Submission saved to: {output_path}")
        
        print("\n" + "=" * 60)
        print("PIPELINE COMPLETE")
        print("=" * 60)
    
    def generate_submission(self, predictions: pd.DataFrame, output_path: Path) -> pd.DataFrame:
        """
        Generate submission CSV file.
        
        Args:
            predictions: DataFrame with frame and prediction columns from classify_frames
            output_path: Path to save the submission CSV
            
        Returns:
            DataFrame containing the submission data
        """
        # Generate submission-ready DataFrame from classifier
        submission_df = self.classifier.generate_submission(predictions)
        
        # Save to CSV, creating the output folder if needed
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        # "\n" on every OS, so the same input gives a byte-identical file everywhere.
        submission_df.to_csv(output_path, index=False, lineterminator="\n")
        
        return submission_df


def main():
    """Main entry point for the Ball Possession Classification pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Ball Possession Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Classify a match folder -> results/<video name>.csv
    python main.py --data-dir /path/to/match

    # Choose the output file yourself
    python main.py --data-dir /path/to/match --output my_predictions.csv

The pipeline will:
    1. Load the tracking (*_atd.txt + *_atd.xml) and, if present, Smart Tagging
    2. Load team assignments from the tracking metadata
    3. Find dead-ball stretches and decide possession between them
    4. Write a CSV with frame and label columns

Output format (one row per tracking frame, frames from 0):
    frame,label
    0,X
    1,H
    ...
        """
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save predictions CSV (default: results/<video name>.csv)"
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/tactical",
        help="Path to tactical data directory (default: ./data/tactical)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not Path(args.data_dir).exists():
        print(f"Error: Data directory '{args.data_dir}' does not exist.")
        sys.exit(1)
    
    output_path = Path(args.output) if args.output else default_output_path(Path(args.data_dir))
    
    # Create classifier and run pipeline
    classifier = BallPossessionClassifier()
    classifier.run_pipeline(Path(args.data_dir), output_path)


if __name__ == "__main__":
    main()
