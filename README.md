# Ball Possession Classifier

Reads the player and ball tracking of one football match and decides, **frame by frame, which
team is closest to the ball**. The result is a `frame,label` CSV.

It is rule-based: no training, no labelled data, no GPU. It works on any match whose tracking is
in the format below, whatever the file and folder names are.

A full description of the program, including every error case, is in
[Project_Report.pdf](Project_Report.pdf).

---

## Quick start

Requirements: Python 3.11 or newer, with `numpy` and `pandas`.

```bash
# Install the dependencies
python -m pip install -r requirements.txt

# Classify one match folder -> results/<video name>.csv
python main.py --device cpu --data-dir <folder>

# Or choose the output file yourself
python main.py --device cpu --data-dir <folder> --output <file.csv>
```

---

## What is in this project

| File | What it does |
|---|---|
| `main.py` | Command-line entry point. Finds the input files, loads them, classifies every frame and writes the CSV. |
| `src/enhanced_classifier.py` | The engine: tracking parser, team mapping, decision rule and CSV checks. Can also be imported as a library. |
| `requirements.txt` | Python dependencies (`numpy`, `pandas`). |
| `Project_Report.pdf` | Detailed capability report. |

---

## Input

The data folder must contain **exactly one** `*_atd.txt` file and, next to it, an `.xml` file
with the same name stem (for example `anything_atd.txt` + `anything_atd.xml`). Only the top level
of the folder is searched. Other files in the folder are ignored.

**Tracking file (`*_atd.txt`)**: one line per frame, `frame:players:ball`.

- `players` is a `;`-separated list of player slots, in the same order and number as the players
  in the XML.
- Each player and the ball are written as `x,y,speed`, with `x` and `y` as normalised pitch
  coordinates. `NaN` means the object is not visible.
- Frame numbers start at 1. Lines may come in any order and may have gaps, but must not repeat.

**Metadata file (`*_atd.xml`)**: exactly two `<Team id="…">` elements, where the first becomes
**H** (home) and the second becomes **A** (away). It also needs one `<Player id="…" teamId="…">`
per tracking slot, in slot order.

---

## How a frame is labelled

Every frame is decided on its own:

| Situation in the frame | Label |
|---|---|
| Ball and at least one player visible | **H** or **A**: the team of the player nearest to the ball |
| Ball not visible | **D** |
| Ball visible, no player visible | **X** |

- Distances are in metres, using a fixed 105.3 m × 68.0 m pitch.
- If two players are equally close, the one listed first in the XML wins.
- Output frame = input frame − 1, so the CSV is indexed from 0, like video frames.

## Output

Without `--output`, the CSV is written to `results/<video name>.csv` in the project folder. The
video name is the name of the video file in the data folder, if there is exactly one. Otherwise
it is the tracking file name without `_atd`: for example, `match_tactical_atd.txt` gives
`results/match_tactical.csv`. Missing output folders are created automatically. `results/` is
ignored by git.

The file has exactly the columns `frame,label`, one row per tracking line, sorted by frame. Labels
are only `H`, `A`, `D` or `X`. Before writing, the program checks for integer frames, repeated
frames and invalid labels. Gaps in the input are not filled.

Bad input stops the program with an explicit error instead of producing a wrong file: a missing
or ambiguous tracking file, missing metadata, a team count other than two, a wrong slot count,
malformed `x,y,speed` groups, and repeated or invalid frame numbers.

---

## Using it as a library

```python
from src.enhanced_classifier import EnhancedPossessionClassifier

c = EnhancedPossessionClassifier()
frames, ball, players = c.load_tactical_data("path/to/folder")
pred = c.classify_frames(frames, ball, players)   # frame, prediction, confidence
csv = c.generate_submission(pred)                 # checked frame,label table
c.get_statistics()                                # label shares, ball-tracking rate
```

`c.player_teams` maps each player id to H/A. The `confidence` column is a fixed score, not a
probability, and is not written to the CSV.

---

## Limitations

- **No real dead-ball detection.** `D` only means the ball has no tracked position. When the
  ball is out of play but still tracked, the program still outputs H or A.
- **No time logic.** There is no smoothing and no memory between frames, so the label can switch
  teams from one frame to the next.
- **Nearest is not possession.** The rule finds the team closest to the ball, not the team that
  controls it.
- **Tracking only.** Event tagging, homography, broadcast video and ball speed are not used.
  Older loaders for event tagging and homography remain in the code but are never called.
- `--device` is accepted but has no effect; everything runs on the CPU.
- One match per run, and the whole tracking file is held in memory.

---

## Background

The project was built for the *More than a Hack 2027 — Phase 1: Ball Possession* challenge. The
challenge kit and the earlier development files are no longer in the working tree. They remain
in git history; for example, the challenge's original README can be shown with
`git show 59da388:README.md`.

The match data is licensed separately for the competition and is not included in this repository.
