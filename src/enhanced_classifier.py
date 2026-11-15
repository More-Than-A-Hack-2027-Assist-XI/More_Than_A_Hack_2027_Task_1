"""
Enhanced Ball Possession Classifier for More Than A Hack 2027
Incorporates Smart Tagging and homography data for improved accuracy.
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')


class EnhancedPossessionClassifier:
    """
    Enhanced classifier that combines:
    1. Ball tracking data (nearest player to ball)
    2. Smart Tagging (team possession indicators)
    3. Homography data (field position context)
    """
    
    def __init__(self, 
                 ball_tracking_window: int = 5,
                 smart_tag_weight: float = 0.3,
                 homography_weight: float = 0.2,
                 possession_threshold: float = 0.6):
        """
        Initialize classifier with configurable parameters.
        
        Args:
            ball_tracking_window: Frames to consider for majority vote
            smart_tag_weight: Weight for Smart Tagging evidence (0-1)
            homography_weight: Weight for homography-based context (0-1)
            possession_threshold: Confidence threshold for possession claims
        """
        self.ball_tracking_window = ball_tracking_window
        self.smart_tag_weight = smart_tag_weight
        self.homography_weight = homography_weight
        self.possession_threshold = possession_threshold
        
        # Smart tagging team mapping (H=Home, A=Away, D=Dead)
        self.team_mapping = {
            'H': 'H',  # Home team
            'A': 'A',  # Away team  
            'D': 'D',  # Dead ball
            'N/A': None,  # Unknown/missing
            '': None  # Empty string
        }
        
        # Field dimensions (standard soccer pitch)
        self.pitch_length = 105.3  # meters
        self.pitch_width = 68.0    # meters
        
        # Statistics tracking
        self.stats = {
            'total_frames': 0,
            'ball_tracked': 0,
            'smart_tag_available': 0,
            'homography_available': 0,
            'predictions': {'H': 0, 'A': 0, 'D': 0, 'X': 0}
        }
    
    def load_tactical_data(self, data_dir: Path) -> Tuple[List[int], Dict[int, Tuple[float, float, float]], Dict[int, Dict[str, Dict[str, Tuple[float, float, float]]]]]:
        """
        Load all tactical data files from directory.
        
        Args:
            data_dir: Directory containing tactical data files
            
        Returns:
            Tuple of (video_frames list, ball_positions dict, player_positions dict)
        """
        print("[1/4] Loading tactical data...")
        
        # Find tactical data files
        atd_file = None
        smart_tagging_file = None
        atd_xml_file = None
        homography_file = None
        
        for file in data_dir.iterdir():
            if file.name.startswith('BAR-ATH_tactical_atd.txt'):
                atd_file = file
            elif file.name.startswith('BAR-ATH_tactical_smart_tagging.xml'):
                smart_tagging_file = file
            elif file.name.startswith('BAR-ATH_tactical_atd.xml'):
                atd_xml_file = file
            elif file.name.startswith('BAR-ATH_tactical_homography.csv'):
                homography_file = file
        
        if not atd_file:
            raise FileNotFoundError(f"Could not find BAR-ATH_tactical_atd.txt in {data_dir}")
        
        # Load ATD tracking data
        ball_positions, player_positions = self.load_atd_tracking(str(atd_file))
        
        # Load Smart Tagging data if available
        smart_tagging_data = None
        if smart_tagging_file and smart_tagging_file.exists():
            smart_tagging_data = self.load_smart_tagging(str(smart_tagging_file))
        
        # Load ATD team assignments if available
        self.player_teams = {}
        atd_team_assignments = None
        if atd_xml_file and atd_xml_file.exists():
            atd_team_assignments = self.load_atd_team_assignments(str(atd_xml_file))
            self.player_teams = atd_team_assignments
        
        # Load homography data if available
        homography_data = None
        if homography_file and homography_file.exists():
            homography_data = self.load_homography(str(homography_file))
        
        # Extract video frames from ATD data
        video_frames = list(range(len(ball_positions)))
        
        # Store frames as instance attribute for later use in predict_frame()
        self.frames = video_frames
        
        return video_frames, ball_positions, player_positions
    
    def load_atd_team_assignments(self, xml_path: str) -> Dict[int, str]:
        """
        Parse ATD XML and extract static player-to-team assignments.
        Team assignments are defined once at file level in the <Players> section,
        not per frame in Sessions.
        
        Args:
            xml_path: Path to ATD.xml file
            
        Returns:
            Dictionary mapping player_id -> team (HOME/AWAY)
        """
        player_teams = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Get team IDs from Teams section (static, not per-frame)
            teams = root.find('.//Teams')
            team_ids = {}
            if teams is not None:
                for team in teams.findall('Team'):
                    team_id = team.get('id')
                    if team_id is None:
                        continue
                    try:
                        team_name = team.find('Name').text
                        # HOME=0 for Barcelona, AWAY=1 for Atletic Club
                        if 'Barcelona' in team_name or 'MET_user_team_o4Hossvb8M' in team_id:
                            team_ids[team_id] = 'HOME'
                        elif 'Atletic Club' in team_name or 'MET_user_team_x7C4FFX2HF' in team_id:
                            team_ids[team_id] = 'AWAY'
                    except (AttributeError, TypeError):
                        continue
            
            # Get static player definitions from Players section at file level
            players_section = root.find('.//Players')
            if players_section is not None:
                for player in players_section.findall('Player'):
                    try:
                        player_id_str = player.get('id')
                        if player_id_str is None:
                            continue
                        player_id = int(player_id_str)
                        
                        player_team_id = player.get('teamId')
                        if player_team_id is None:
                            continue
                        
                        if player_team_id in team_ids:
                            player_teams[player_id] = team_ids[player_team_id]
                    except (ValueError, AttributeError):
                        # Skip malformed player entries
                        continue
        
        except Exception as e:
            print(f"Error loading ATD team assignments: {e}")
        
        return player_teams

    def load_smart_tagging(self, xml_path: str) -> Dict[int, str]:
        """
        Parse Smart Tagging XML and extract team possession data.
        
        Args:
            xml_path: Path to SMART_TAGGING.xml file
            
        Returns:
            Dictionary mapping frame number to team (H/A/D) or None
        """
        smart_tags = {}
        
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for event in root.findall('.//event'):
                frame_num = int(event.get('frame'))
                
                # Get team from Smart Tagging data
                team_elem = event.find('.//team')
                if team_elem is not None:
                    team = team_elem.text.strip()
                    smart_tags[frame_num] = self.team_mapping.get(team, None)
                    
        except Exception as e:
            print(f"Warning: Error parsing Smart Tagging XML: {e}")
        
        return smart_tags
    
    def load_homography(self, csv_path: str) -> Dict[int, Tuple[float, float]]:
        """
        Load homography matrices for field position context.
        
        Args:
            csv_path: Path to HOMOGRAPHY.csv file
            
        Returns:
            Dictionary mapping frame number to homography matrix
        """
        homographies = {}
        
        try:
            df = pd.read_csv(csv_path)
            
            # Group by frame and get homography matrices
            for frame_num, group in df.groupby('frame'):
                if len(group) > 0:
                    # Extract homography matrix (typically 8 values for 3x3 matrix)
                    h_vals = group['homography'].values.flatten()
                    if len(h_vals) >= 9:
                        homographies[frame_num] = tuple(h_vals[:9])
                        
        except Exception as e:
            print(f"Warning: Error loading homography data: {e}")
        
        return homographies
    
    def load_atd_tracking(self, txt_path: str) -> Tuple[Dict[int, Tuple[float, float, float]], Dict[int, Dict[str, Dict[str, Tuple[float, float, float]]]]]:
        """
        Load ATD tracking data with 3D joint coordinates.
        
        Args:
            txt_path: Path to ATD tracking .txt file
            
        Returns:
            Tuple of (ball_positions_dict, player_positions_dict)
            - ball_positions_dict: frame_num -> (x, y, z) for ball position (joint 0)
            - player_positions_dict: frame_num -> {player_id -> {'x', 'y', 'z'}}
        """
        ball_positions = {}
        player_positions = {}
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            current_frame = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Parse format: "frame_num:joint1_x,joint1_y,joint1_z;joint2_x,joint2_y,joint2_z;..."
                if ':' in line:
                    frame_part, joints_part = line.split(':', 1)
                    try:
                        current_frame = int(frame_part.strip())
                    except ValueError:
                        continue
                    
                    # Initialize for this frame
                    ball_positions[current_frame] = None
                    player_positions[current_frame] = {}
                    
                    # Split joints by semicolon
                    joint_sets = joints_part.split(';')
                    joint_idx = 0
                    
                    for joint_set in joint_sets:
                        joint_set = joint_set.strip()
                        if not joint_set:
                            continue
                        
                        coords = joint_set.split(',')
                        if len(coords) >= 3:
                            try:
                                x = float(coords[0])
                                y = float(coords[1])
                                z = float(coords[2])
                                
                                # Check if valid (not all NaN)
                                if not (np.isnan(x) and np.isnan(y) and np.isnan(z)):
                                    if joint_idx == 0:
                                        # Joint 0 is ball position
                                        ball_positions[current_frame] = (x, y, z)
                                    else:
                                        # Remaining joints are player positions
                                        player_id = f"p{joint_idx}"
                                        player_positions[current_frame][player_id] = {
                                            'x': x,
                                            'y': y,
                                            'z': z
                                        }
                            except (ValueError, IndexError):
                                pass
                        
                        joint_idx += 1
                    
                    continue
                
                # Fallback for old format (e.g., "frame 1")
                parts = line.split()
                if len(parts) >= 2 and parts[0].lower() == 'frame':
                    current_frame = int(parts[1])
                    player_positions[current_frame] = {}
                    continue
                
                # Player data (e.g., "player 1 x y z ...")
                if len(parts) >= 5 and parts[0].lower() == 'player':
                    try:
                        player_id = int(parts[1])
                        x = float(parts[2])
                        y = float(parts[3])
                        z = float(parts[4])
                        
                        # Store normalized coordinates (0-1 range)
                        player_positions[player_id] = {
                            'x': x,
                            'y': y,
                            'z': z,
                            'normalized_x': x / self.pitch_length,
                            'normalized_y': y / self.pitch_width
                        }
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"Warning: Error loading ATD tracking data: {e}")
        
        return ball_positions, player_positions
    
    def load_ball_tracking(self, txt_path: str) -> Dict[int, Tuple[float, float]]:
        """
        Load ball tracking data if available.
        
        Args:
            txt_path: Path to ATD tracking file (may contain ball data)
            
        Returns:
            Dictionary mapping frame number to ball position or None
        """
        ball_positions = {}
        
        try:
            with open(txt_path, 'r') as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                
                # Check for ball data (e.g., "ball x y z")
                if len(parts) >= 4 and parts[0].lower() == 'ball':
                    try:
                        x = float(parts[1])
                        y = float(parts[2])
                        z = float(parts[3])
                        
                        ball_positions[int(parts[0])] = {
                            'x': x,
                            'y': y,
                            'z': z,
                            'normalized_x': x / self.pitch_length,
                            'normalized_y': y / self.pitch_width
                        }
                    except (ValueError, IndexError):
                        continue
                        
        except Exception as e:
            print(f"Warning: Error loading ball tracking data: {e}")
        
        return ball_positions
    
    def get_nearest_player_to_ball(
        self, 
        frame_num: int, 
        ball_pos: Dict[str, float], 
        player_positions: Dict[int, Dict[str, float]]
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find the nearest player to the ball position.
        
        Args:
            frame_num: Frame number
            ball_pos: Ball position dictionary
            player_positions: Player positions dictionary
            
        Returns:
            Tuple of (player_id, team_assignment) or (None, None) if no players available
        """
        if not player_positions:
            return None, None
        
        min_distance = float('inf')
        nearest_player = None
        nearest_team = None
        
        for player_id, pos in player_positions.items():
            try:
                # Calculate Euclidean distance (2D on normalized field)
                dist = np.sqrt(
                    (pos['normalized_x'] - ball_pos['normalized_x'])**2 +
                    (pos['normalized_y'] - ball_pos['normalized_y'])**2
                )
                
                if dist < min_distance:
                    min_distance = dist
                    nearest_player = player_id
                    nearest_team = self.player_teams.get(player_id) if self.player_teams else None
                    
            except (KeyError, TypeError):
                continue
        
        return nearest_player, nearest_team
    
    def get_ball_owner_from_smart_tag(
        self, 
        frame_num: int, 
        smart_tags: Dict[int, str]
    ) -> Optional[str]:
        """
        Get ball owner from Smart Tagging data.
        
        Args:
            frame_num: Frame number
            smart_tags: Smart tagging dictionary
            
        Returns:
            Team (H/A) or None if not available
        """
        if frame_num in smart_tags:
            return smart_tags[frame_num]
        return None
    
    def get_ball_owner_from_homography(
        self, 
        frame_num: int, 
        homographies: Dict[int, Tuple[float, float]],
        ball_pos: Optional[Dict[str, float]] = None
    ) -> Optional[str]:
        """
        Infer ball owner from homography and field position.
        
        Args:
            frame_num: Frame number
            homographies: Homography dictionary
            ball_pos: Ball position (optional)
            
        Returns:
            Team (H/A) or None if cannot determine
        """
        # This is a simplified heuristic - in practice would use more complex logic
        # For now, return None as homography alone isn't sufficient
        return None
    
    def classify_frame(
        self, 
        frame_num: int,
        ball_pos: Optional[Dict[str, float]],
        player_positions: Dict[int, Dict[str, float]],
        smart_tag: Optional[str],
        homography: Optional[Tuple[float, float]] = None
    ) -> Tuple[str, float]:
        """
        Classify possession for a single frame using all available data sources.
        
        Args:
            frame_num: Frame number
            ball_pos: Ball position (may be None)
            player_positions: Player positions
            smart_tag: Smart tagging team indicator
            homography: Homography data
            
        Returns:
            Tuple of (predicted_team, confidence)
        """
        # Update statistics
        self.stats['total_frames'] += 1
        
        if ball_pos is not None:
            self.stats['ball_tracked'] += 1
        if smart_tag is not None:
            self.stats['smart_tag_available'] += 1
        if homography is not None:
            self.stats['homography_available'] += 1
        
        # DEBUG: Print first 5 frames to trace execution
        if frame_num < 5:
            print(f"Frame {frame_num}: ball_pos={ball_pos}, player_positions keys={list(player_positions.keys())[:5] if player_positions else []}, smart_tag={smart_tag}")
        
        # Initialize vote counters
        votes = {'H': 0, 'A': 0, 'D': 0, 'X': 0}
        confidence = 0.0
        
        # Source 1: Ball tracking + nearest player (highest weight)
        if ball_pos is not None and player_positions:
            nearest_player, nearest_team = self.get_nearest_player_to_ball(
                frame_num, ball_pos, player_positions
            )
            
            # DEBUG: Print first frame's nearest player result
            if frame_num == 0:
                print(f"Frame {frame_num}: nearest_player={nearest_player}, nearest_team={nearest_team}")
            
            # If we have a nearest player with team assignment, use it for prediction
            if nearest_player is not None and nearest_team is not None:
                votes[nearest_team] += 1
                confidence += 0.6
            else:
                # No team assignment available - mark as uncertain
                votes['X'] += 1
                confidence += 0.4
        
        # Source 2: Smart Tagging (medium weight)
        if smart_tag is not None and smart_tag in ['H', 'A']:
            votes[smart_tag] += 1
            confidence += self.smart_tag_weight * 0.5
        
        # Source 3: Homography context (lower weight)
        if homography is not None:
            # In practice, this would use field position analysis
            pass
        
        # Determine prediction
        total_votes = sum(votes.values())
        
        if total_votes == 0:
            return 'X', 0.0
        
        # Find highest vote count
        max_votes = max(votes.values())
        predicted_team = max(votes, key=votes.get)
        
        # Calculate confidence based on vote proportion and evidence quality
        vote_confidence = max_votes / total_votes
        final_confidence = min(1.0, confidence + vote_confidence * 0.3)
        
        # Apply threshold
        if final_confidence >= self.possession_threshold:
            return predicted_team, final_confidence
        
        # Below threshold - mark as uncertain
        return 'X', final_confidence
    
    def classify_frames(
        self,
        video_frames: List[int],
        ball_positions_dict: Dict[int, Tuple[float, float, float]],
        player_positions_dict: Dict[int, Dict[str, Dict[str, Tuple[float, float, float]]]],
        smart_tags: Dict[int, str] = None,
        homographies: Dict[int, Tuple[float, float]] = None
    ) -> pd.DataFrame:
        """
        Classify possession for all frames.
        
        Args:
            video_frames: List of frame numbers
            ball_positions_dict: Ball tracking positions per frame
            player_positions_dict: Player positions per frame
            smart_tags: Smart tagging team indicators (optional)
            homographies: Homography matrices (optional)
            
        Returns:
            DataFrame with frame numbers and predictions
        """
        predictions = []
        homography = homographies or {}
        
        # Process frames in order
        all_frames = sorted(video_frames)
        
        for frame_num in all_frames:
            ball_pos = ball_positions_dict.get(frame_num)
            player_positions = player_positions_dict.get(frame_num, {})
            smart_tag = smart_tags.get(frame_num) if smart_tags else None
            homography_mat = homography.get(frame_num) if homography else None
            
            pred, conf = self.classify_frame(
                frame_num, ball_pos, player_positions, smart_tag, homography_mat
            )
            
            predictions.append({
                'frame': frame_num,
                'prediction': pred,
                'confidence': conf
            })
        
        # Update prediction statistics
        for pred in predictions:
            self.stats['predictions'][pred['prediction']] += 1
        
        return pd.DataFrame(predictions)
    
    def generate_submission(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate submission-ready DataFrame.
        
        Args:
            predictions_df: DataFrame with frame predictions
            
        Returns:
            Submission-ready DataFrame with columns: frame, label
        """
        # Sort by frame number
        submission = predictions_df.sort_values('frame').copy()
        
        # Rename prediction column to label
        submission = submission.rename(columns={'prediction': 'label'})
        
        # Ensure integer frame numbers (handle NaN/missing values gracefully)
        submission['frame'] = pd.to_numeric(submission['frame'], errors='coerce').fillna(-1).astype(int)
        
        return submission
    
    def get_statistics(self) -> Dict:
        """Get classifier statistics."""
        total = self.stats['total_frames']
        return {
            'total_frames': total,
            'ball_tracked': self.stats['ball_tracked'],
            'ball_tracking_rate': self.stats['ball_tracked'] / total if total > 0 else 0,
            'smart_tag_available': self.stats['smart_tag_available'],
            'homography_available': self.stats['homography_available'],
            'prediction_distribution': {
                k: v / total if total > 0 else 0 
                for k, v in self.stats['predictions'].items()
            }
        }


def main():
    """Main execution function."""
    print("Enhanced Ball Possession Classifier")
    print("=" * 50)
    
    # Initialize classifier
    classifier = EnhancedPossessionClassifier(
        ball_tracking_window=5,
        smart_tag_weight=0.3,
        homography_weight=0.2,
        possession_threshold=0.6
    )
    
    # Define data paths using absolute paths from script location
    script_root = Path(__file__).parent.parent.resolve()
    
    data_dir = script_root / "data" / "tactical"
    
    atd_path = data_dir / "BAR-ATH_tactical_atd.txt"
    smart_tag_path = data_dir / "BAR-ATH_tactical_smart_tagging.xml"
    homography_path = data_dir / "BAR-ATH_tactical_homography.csv"
    
    print(f"\nLoading data from: {data_dir}")
    print(f"  ATD Tracking: {atd_path.exists()}")
    print(f"  Smart Tagging: {smart_tag_path.exists()}")
    print(f"  Homography: {homography_path.exists()}")
    
    # Load data
    print("\nLoading ATD tracking data...")
    ball_positions_dict, player_positions_dict = classifier.load_atd_tracking(str(atd_path))
    print(f"  Loaded {len(ball_positions_dict)} frames with ball positions, {len(player_positions_dict)} frames with player positions")
    
    # Convert ball_positions_dict to the format expected by classify_frames
    # ball_positions_dict: frame_num -> (x, y, z) tuple or None
    ball_positions = {}
    for frame_num, coords in ball_positions_dict.items():
        if coords is None:
            continue  # Skip frames without ball tracking data
        x, y, z = coords
        ball_positions[frame_num] = {
            'x': x,
            'y': y,
            'z': z,
            'normalized_x': x / classifier.pitch_length,
            'normalized_y': y / classifier.pitch_width
        }
    print(f"  Ball tracked on {len(ball_positions)} frames")
    
    print("Loading Smart Tagging data...")
    smart_tags = classifier.load_smart_tagging(str(smart_tag_path))
    print(f"  Smart tags available for {len(smart_tags)} frames")
    
    print("Loading homography data...")
    homographies = classifier.load_homography(str(homography_path))
    print(f"  Homography matrices: {len(homographies)} frames")
    
    # Classify frames
    print("\nClassifying frames...")
    predictions_df = classifier.classify_frames(
        list(ball_positions_dict.keys()),  # video_frames
        ball_positions_dict,                # ball_positions_dict
        player_positions_dict,              # player_positions_dict
        smart_tags,                         # smart_tags
        homographies                        # homographies
    )
    
    # Generate submission
    print("Generating submission...")
    submission = classifier.generate_submission(predictions_df)
    
    # Save results
    output_dir = script_root / "results"
    output_dir.mkdir(exist_ok=True)
    
    submission_path = output_dir / "enhanced_classifier_submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission saved to: {submission_path}")
    
    # Print statistics
    stats = classifier.get_statistics()
    print("\nClassifier Statistics:")
    print(f"  Total frames: {stats['total_frames']:,}")
    print(f"  Ball tracking rate: {stats['ball_tracking_rate']*100:.1f}%")
    print(f"  Smart tag availability: {stats['smart_tag_available']:,} frames")
    print(f"\nPrediction distribution:")
    for label, rate in stats['prediction_distribution'].items():
        print(f"  {label}: {rate*100:.1f}% ({int(stats['total_frames']*rate):,} frames)")
    
    return submission


if __name__ == "__main__":
    main()
