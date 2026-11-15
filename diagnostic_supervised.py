"""Verify recovered tracking contracts against independent raw EPTS parsing."""
from pathlib import Path
import sys
import math
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from src.enhanced_classifier import EnhancedPossessionClassifier

c = EnhancedPossessionClassifier()
frames, ball, players = c.load_tactical_data(ROOT / 'data/tactical')
assert frames == list(range(199319)), 'Incorrect video timeline'
assert set(c.player_teams.values()) == {'H', 'A'}
assert len(c.player_teams) == 46
raw = ROOT / 'data/tactical/BAR-ATH_tactical_atd.txt'
meta = ET.parse(raw.with_suffix('.xml')).getroot()
track_ids = [int(p.get('id')) for p in meta.iter('Player')]
sample_frames, missing = [], None
valid_count = 0
with raw.open() as handle:
    for line in handle:
        frame_text, groups, raw_ball = line.strip().split(':')
        frame = int(frame_text) - 1
        bx, by = map(float, raw_ball.split(',')[:2])
        valid = math.isfinite(bx) and math.isfinite(by)
        assert (ball[frame] is not None) == valid
        if not valid:
            if missing is None:
                missing = frame
            continue
        valid_count += 1
        assert ball[frame]['normalized_x'] == bx
        assert ball[frame]['normalized_y'] == by
        # Sample separated points throughout the match, independent reference distances.
        if players[frame] and (not sample_frames or frame - sample_frames[-1] > 40000):
            distances = []
            for pid, group in zip(track_ids, groups.split(';')):
                px, py = map(float, group.split(',')[:2])
                if math.isfinite(px) and math.isfinite(py):
                    distances.append((((px-bx)*c.pitch_length)**2 + ((py-by)*c.pitch_width)**2, pid))
            expected = min(distances, key=lambda item: item[0])[1]
            pid, team = c.get_nearest_player_to_ball(frame, ball[frame], players[frame])
            assert pid == expected
            result = c.classify_frame(frame, ball[frame], players[frame], None)
            assert result[0] == c.player_teams[expected]
            print('Sample', frame, 'ball', (bx, by), 'nearest', pid, 'team', team, 'result', result)
            sample_frames.append(frame)
assert valid_count == 113488
assert c.classify_frame(missing, None, players[missing], None)[0] == 'D'
assert c.classify_frame(sample_frames[0], ball[sample_frames[0]], {}, None)[0] == 'X'
submission = c.generate_submission(c.classify_frames(sample_frames, ball, players))
assert list(submission.columns) == ['frame', 'label']
assert len(submission) == len(sample_frames)
print('PASS: 199319 zero-based frames; 46 tracks; valid ball 113488; missing ball 85831; independent nearest-player checks; D/X fallbacks; exact submission schema.')
