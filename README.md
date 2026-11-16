# Ball Possession Classifier

Reads the player and ball tracking of one football match and decides, **frame by frame, which
team is closest to the ball**. The result is a `frame,label` CSV. A separate video player shows
those labels on top of the match video so you can check them by eye.

It is rule-based: no training, no labelled data, no GPU. It works on any match whose tracking is
in the format below, whatever the file and folder names are.

A full description of the program, including every error case, is in
[Project_Report.pdf](Project_Report.pdf).

---

## Quick start

Requirements: Python 3 with `numpy` and `pandas`. The video player also needs `opencv-python`
and Tkinter (included with standard Python on Windows).

```bash
# Classify one match folder
python main.py --device cpu --data-dir <folder> --output <file.csv>

# Watch the labels on the video
python video_player.py <video.mp4> <file.csv> [--max-width 1280]

# Run the regression tests
python -m unittest test_generic_inputs -v
```

The output folder must already exist.

---

## What is in this project

| File | What it does |
|---|---|
| `main.py` | Command-line entry point. Finds the input files, loads them, classifies every frame and writes the CSV. |
| `src/enhanced_classifier.py` | The engine: tracking parser, team mapping, decision rule and CSV checks. Can also be imported as a library. |
| `video_player.py` | Desktop player that shows each frame's label over the video, with seeking and frame stepping. |
| `test_generic_inputs.py` | 8 regression tests on small synthetic inputs. |
| `diagnostic_supervised.py` | Self-check against an independent re-parse of a real tracking file. Written for one specific data file. |
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

A CSV with exactly the columns `frame,label`, one row per tracking line, sorted by frame. Labels
are only `H`, `A`, `D` or `X`. Before writing, the program checks for integer frames, repeated
frames and invalid labels. Gaps in the input are not filled.

Bad input stops the program with an explicit error instead of producing a wrong file: a missing
or ambiguous tracking file, missing metadata, a team count other than two, a wrong slot count,
malformed `x,y,speed` groups, and repeated or invalid frame numbers.

---

## Video player

```bash
python video_player.py <video> <labels.csv>
```

- Shows the current label, its name and the frame number over the video. Frames without a CSV
  row show `?`.
- The timeline bar can be clicked or dragged to jump anywhere. The time readout shows
  `hh:mm:ss.mmm`.
- It plays at the video's frame rate without skipping frames. The window is resizable and keeps
  the aspect ratio.
- It warns when the video and the CSV have different frame counts.
- It only displays. The video is not changed or exported. The interface text is in Turkish.

| Key | Action |
|---|---|
| Space | Play / pause |
| Left / Right arrow | Jump 5 seconds |
| A / D | Pause and step one frame |
| Esc | Close |

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
official challenge documents are kept unchanged in this folder: [CHALLENGE.md](CHALLENGE.md),
[DATA.md](DATA.md), [EVALUATION.md](EVALUATION.md), [SUBMISSION.md](SUBMISSION.md),
[RULES.md](RULES.md), [FAQ.md](FAQ.md) and [OVERVIEW.md](OVERVIEW.md). The official evaluator
and starter code are in `evaluation/` and `starter/`. The challenge's original README is in git
history: `git show 59da388:README.md`.

The match data is licensed separately for the competition and is not included in this repository.
