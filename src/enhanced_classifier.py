"""
Recovered tracking-only possession classifier for More Than A Hack 2027.
Optional Smart Tagging and homography loaders are legacy, inactive code.
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
    Tracking-only nearest-player recovery using the official EPTS format.
    Missing ball -> D follows the raw baseline heuristic, not detected dead play.
    Legacy tuning arguments are retained for API compatibility; this recovery
    does not apply temporal smoothing or optional evidence fusion.
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
    
    def load_tactical_data(self, data_dir):
        """Load tracking on the zero-based submission timeline."""
        data_dir = Path(data_dir)
        candidates = sorted(data_dir.glob('*_atd.txt'))
        if len(candidates) == 0:
            raise FileNotFoundError(f'No *_atd.txt found in {data_dir}')
        if len(candidates) > 1:
            raise ValueError(f'Ambiguous *_atd.txt: {candidates}')
        raw = candidates[0]
        meta = raw.with_suffix('.xml')
        if not meta.exists():
            raise FileNotFoundError(f'Missing metadata: {meta}')
        self.player_teams = self.load_atd_team_assignments(meta)
        self.track_ids = [int(p.get('id')) for p in ET.parse(meta).getroot().iter('Player')]
        ball, players = self.load_atd_tracking(raw)
        self.frames = sorted(ball)
        self.smart_tags = {}
        self.homographies = {}
        print('Tracking-only recovery: Smart Tagging and homography are not used.')
        return self.frames, ball, players
    
    def load_atd_team_assignments(self, xml_path):
        """Use XML track IDs and the EPTS home/away team ordering."""
        root = ET.parse(xml_path).getroot()
        teams = list(root.iter('Team'))
        if len(teams) != 2:
            raise ValueError('Expected exactly two teams in EPTS metadata')
        labels = {teams[0].get('id'): 'H', teams[1].get('id'): 'A'}
        return {int(p.get('id')): labels[p.get('teamId')] for p in root.iter('Player')}

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
    
    def load_atd_tracking(self, txt_path):
        """Read frame:players:ball; each object is normalized x,y,speed."""
        raw = Path(txt_path)
        meta = raw.with_suffix('.xml')
        ids = [int(p.get('id')) for p in ET.parse(meta).getroot().iter('Player')]
        ball, players = {}, {}
        def position(group):
            values = group.split(',')
            if len(values) != 3:
                raise ValueError('Expected x,y,speed object group')
            nx, ny, speed = map(float, values)
            if not np.isfinite(nx) or not np.isfinite(ny):
                return None
            return {'normalized_x': nx, 'normalized_y': ny,
                    'x': nx * self.pitch_length, 'y': ny * self.pitch_width,
                    'speed': speed}
        with raw.open() as handle:
            for line in handle:
                frame_text, player_text, ball_text = line.strip().split(':')
                frame = int(frame_text) - 1
                if frame < 0 or frame in ball:
                    raise ValueError('Invalid or duplicate tracking frame')
                groups = player_text.split(';')
                if len(groups) != len(ids):
                    raise ValueError('Track slot count differs from XML metadata')
                ball[frame] = position(ball_text)
                players[frame] = {}
                for pid, group in zip(ids, groups):
                    pos = position(group)
                    if pos is not None:
                        players[frame][pid] = pos
        return ball, players
    
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
    
    def get_nearest_player_to_ball(self, frame_num, ball_pos, player_positions):
        """Compare both axes in metres and use canonical integer track IDs."""
        if ball_pos is None or not player_positions:
            return None, None
        best_id, best_distance = None, float('inf')
        for pid, pos in player_positions.items():
            dx = (pos['normalized_x'] - ball_pos['normalized_x']) * self.pitch_length
            dy = (pos['normalized_y'] - ball_pos['normalized_y']) * self.pitch_width
            distance = dx * dx + dy * dy
            if np.isfinite(distance) and distance < best_distance:
                best_id, best_distance = pid, distance
        return best_id, getattr(self, 'player_teams', {}).get(best_id)
    
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
    
    def classify_frame(self, frame_num, ball_pos, player_positions, smart_tag, homography=None):
        """Tracking-only recovery; missing ball -> D is a baseline heuristic."""
        self.stats['total_frames'] += 1
        if ball_pos is None:
            return 'D', 0.0
        self.stats['ball_tracked'] += 1
        player, team = self.get_nearest_player_to_ball(frame_num, ball_pos, player_positions)
        if player is None or team not in ('H', 'A'):
            return 'X', 0.0
        # Evidence score, not a calibrated probability.
        return team, 0.9
    
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
    
    def generate_submission(self, predictions_df):
        """The official format contains exactly frame,label."""
        submission = predictions_df.rename(columns={'prediction': 'label'})[['frame', 'label']].copy()
        submission['frame'] = pd.to_numeric(submission['frame'], errors='raise').astype(int)
        if submission['frame'].duplicated().any():
            raise ValueError('Duplicate submission frames')
        if not submission['label'].isin(['H', 'A', 'D', 'X']).all():
            raise ValueError('Invalid submission labels')
        return submission.sort_values('frame').reset_index(drop=True)
    
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
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from main import main as pipeline_main
    return pipeline_main()


if __name__ == "__main__":
    main()
