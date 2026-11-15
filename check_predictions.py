import pandas as pd

# Check current predictions distribution
submission = pd.read_csv('results/predictions.csv')
print("Current prediction distribution:")
print(submission['label'].value_counts())
print(f"\nTotal frames: {len(submission)}")
print(f"X percentage: {(submission['label'] == 'X').sum() / len(submission) * 100:.2f}%")
