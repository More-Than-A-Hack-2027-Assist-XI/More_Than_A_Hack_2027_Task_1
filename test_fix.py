import sys
sys.path.insert(0, 'src')

from enhanced_classifier import EnhancedPossessionClassifier
from pathlib import Path

# Test the fix with a small sample
classifier = EnhancedPossessionClassifier()
data_dir = Path('data/tactical')
classifier.load_tactical_data(data_dir)

print(f'Frames loaded: {len(classifier.frames)}')
print(f'Players detected: {len(classifier.players)}')
print(f'Player teams sample: {dict(list(classifier.player_teams.items())[:5])}')

# Test classification on a small sample
test_frames = [100, 200, 300, 400, 500]
predictions = classifier.classify_ball_possession(test_frames)
print(f'\nSample predictions for frames {test_frames}:')
for frame, pred in zip(test_frames, predictions):
    print(f'  Frame {frame}: {pred}')

# Count distribution
from collections import Counter
dist = Counter(predictions)
print(f'\nPrediction distribution: {dict(dist)}')
