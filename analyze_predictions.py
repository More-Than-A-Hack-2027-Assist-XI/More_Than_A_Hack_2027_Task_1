import pandas as pd

# Load predictions
preds = pd.read_csv('results/predictions.csv')
print(f"Total frames: {len(preds)}")
print(f"\nLabel distribution:")
print(preds['label'].value_counts())
print(f"\nX predictions: {(preds['label'] == 'X').sum()} ({100*(preds['label'] == 'X').sum()/len(preds):.2f}%)")
print(f"D predictions: {(preds['label'] == 'D').sum()} ({100*(preds['label'] == 'D').sum()/len(preds):.2f}%)")
print(f"A predictions: {(preds['label'] == 'A').sum()} ({100*(preds['label'] == 'A').sum()/len(preds):.2f}%)")

# Check frame ranges with D predictions
d_frames = preds[preds['label'] == 'D']['frame'].values
print(f"\nFirst 10 D prediction frames: {d_frames[:10]}")
print(f"Last 10 D prediction frames: {d_frames[-10:]}")

# Check if any D predictions are in ground truth range (frames 11185-18684)
gt_start, gt_end = 11185, 18684
in_gt_range = d_frames[(d_frames >= gt_start) & (d_frames <= gt_end)]
print(f"\nD predictions in GT range ({gt_start}-{gt_end}): {len(in_gt_range)}")
if len(in_gt_range) > 0:
    print(f"Frames: {sorted(in_gt_range)[:20]}...")
