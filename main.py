#!/usr/bin/env python3
"""
Ball Possession Classification Pipeline - Official Entry Point

This script provides the main entry point for the Ball Possession Classification pipeline.
It integrates all modules (EnhancedClassifier, data loaders) into a unified workflow.

Usage:
    python main.py --device cuda/cpu --output <path> --data-dir <path>
    
    --help: Show usage information
    
    --device: Device to run inference on (cuda or cpu, default: cuda)
    --output: Path to save predictions CSV (default: predictions.csv)
    --data-dir: Path to tactical data directory (default: ./data/tactical)

Author: AI Agent
Date: 2026-11-15
"""

import argparse
import os
import sys
from pathlib import Path
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from src.enhanced_classifier import EnhancedPossessionClassifier


def discover_files_in_directory(data_dir: Path) -> dict:
    """
    Discover and identify tactical data files in a directory.
    
    Args:
        data_dir: Directory to search for tactical data files
        
    Returns:
        Dictionary mapping file type to discovered file paths
    """
    files = {
        'atd_tracking': None,
        'smart_tagging': None,
        'atd_xml': None,
        'homography': None,
        'ground_truth': None
    }
    
    # Patterns to match different dataset naming conventions
    patterns = {
        'atd_tracking': [
            'BAR-ATH_tactical_atd.txt',
            'M4-ATH_tactical_atd.txt', 
            'RIFLE-ATH_tactical_atd.txt',
            'tactical_atd.txt',
            'tracking_data.txt'
        ],
        'smart_tagging': [
            'BAR-ATH_tactical_smart_tagging.xml',
            'M4-ATH_tactical_smart_tagging.xml',
            'RIFLE-ATH_tactical_smart_tagging.xml',
            'smart_tagging.xml',
            'events.xml'
        ],
        'atd_xml': [
            'BAR-ATH_tactical_atd.xml',
            'M4-ATH_tactical_atd.xml',
            'RIFLE-ATH_tactical_atd.xml',
            'atd.xml',
            'tracking.xml'
        ],
        'homography': [
            'BAR-ATH_tactical_homography.csv',
            'M4-ATH_tactical_homography.csv',
            'RIFLE-ATH_tactical_homography.csv',
            'homography.csv',
            'camera_calibration.csv'
        ],
        'ground_truth': [
            'ground_truth_public.csv',
            'ground_truth.csv',
            'gt_public.csv',
            'labels.csv'
        ]
    }
    
    # Walk through directory and subdirectories
    for root, dirs, filenames in os.walk(data_dir):
        for filename in filenames:
            # Check each pattern category
            for file_type, file_patterns in patterns.items():
                if filename in file_patterns:
                    file_path = Path(root) / filename
                    if files[file_type] is None:
                        files[file_type] = file_path
                    else:
                        print(f"Warning: Multiple {file_type} files found. Using first: {files[file_type]}")
    
    return files


class BallPossessionClassifier:
    """
    Main classifier class that orchestrates the complete ball possession classification pipeline.
    
    This class integrates all modules (EnhancedClassifier, data loaders) into a unified workflow.
    It provides a clean interface for running the complete pipeline from data loading to prediction generation.
    """
    
    def __init__(self, device: str = "cpu"):
        """
        Initialize the BallPossessionClassifier.
        
        Args:
            device: Device to run inference on (cuda or cpu)
        """
        self.device = device
        # EnhancedPossessionClassifier doesn't use device parameter directly
        # It handles device internally when needed
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
        print(f"Device: {self.device}")
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
        predictions = self.classifier.classify_frames(video_frames, ball_positions, player_positions, smart_tags=self.classifier.smart_tags, homographies=self.classifier.homographies)
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
        
        # Save to CSV
        submission_df.to_csv(output_path, index=False)
        
        return submission_df


def main():
    """Main entry point for the Ball Possession Classification pipeline."""
    
    parser = argparse.ArgumentParser(
        description="Ball Possession Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default settings (CUDA device)
    python main.py
    
    # Run with CPU device
    python main.py --device cpu
    
    # Specify output path
    python main.py --output my_predictions.csv
    
    # Specify custom data directory
    python main.py --data-dir /path/to/data/tactical
    
    # Full command
    python main.py --device cuda --output predictions.csv --data-dir ./data/tactical

The pipeline will:
    1. Load tactical video frames and ATD tracking data
    2. Load team assignments from ATD XML
    3. Run ball possession classification using EnhancedClassifier
    4. Generate submission CSV with frame and label columns
    5. Save predictions to the specified output path

Expected output format:
    frame,label
    11185,H
    11186,A
    ...
        """
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to run inference on (cuda or cpu, default: cuda)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.csv",
        help="Path to save predictions CSV (default: predictions.csv)"
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
    
    output_path = Path(args.output)
    
    # Create classifier and run pipeline
    classifier = BallPossessionClassifier(device=args.device)
    classifier.run_pipeline(Path(args.data_dir), output_path)


if __name__ == "__main__":
    main()
