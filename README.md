# Ball Possession Classifier

Reads the player and ball tracking of one football match and decides, **for every frame, which
team is in possession or whether the ball is out of play**. The result is a `frame,label` CSV.

It is rule-based: no training, no labelled data, no GPU. Dead balls are found from the ball leaving
the pitch, from restarts (Smart Tagging set pieces, or a placed ball being kicked) and from the
players' tempo. Possession only changes hands when the other team gains *controlled* possession.
It works on any match in the input format below, whatever the file and folder names are.

A full description of the program, including every rule and constant, is in
[Project_Report.pdf](Project_Report.pdf).

---

## Quick start

Requirements: Python 3.11 or newer.

```bash
# Install the dependencies
python -m pip install -r requirements.txt

# Classify one match folder -> results/<video name>.csv
python main.py --data-dir <folder>

# Or choose the output file yourself
python main.py --data-dir <folder> --output <file.csv>
```

The same input always gives a byte-identical output file (LF line endings on every OS).

---

## What is in this project

| File | What it does |
|---|---|
| `main.py` | Command-line entry point. Finds the input files, loads them, classifies every frame and writes the CSV. |
| `src/enhanced_classifier.py` | The engine: tracking and Smart Tagging parsers, dead-ball detection, possession logic and CSV checks. Can also be imported as a library. |
| `requirements.txt` | Pinned Python dependencies (`numpy`, `pandas`). |
| `Project_Report.pdf` | Detailed technical report. |

---

## Input

The data folder must contain **exactly one** `*_atd.txt` file and, next to it, an `.xml` file
with the same name stem (for example `match_atd.txt` + `match_atd.xml`). Only the top level of the
folder is searched.

**Tracking file (`*_atd.txt`)**: one line per frame, `frame:players:ball`.

- `players` is a `;`-separated list of player slots, in the same order and number as the players
  in the XML.
- Each player and the ball are written as `x,y,speed`, with `x` and `y` as normalised pitch
  coordinates. `NaN` means the object is not visible.
- Frame numbers start at 1. Lines may come in any order and may have gaps, but must not repeat.

**Metadata file (`*_atd.xml`)**: exactly two `<Team id="…">` elements, where the first becomes
**H** (home) and the second becomes **A** (away). It also needs one `<Player id="…" teamId="…">`
per tracking slot, in slot order. `<FrameRate>` is read if present (default 25).

**Smart Tagging (optional)**: `<prefix>_smart_tagging.xml`, where `<prefix>` is the tracking file
name without `_atd.txt`. If there is no such file, the only `*_smart_tagging.xml` in the folder
is used. Only `SET PIECES` instances are read: their type (corner kick, free kick, goal kick,
kick-off, throw-in, penalty) and their start time, which marks a restart. Without the file,
restarts are detected from tracking alone.

---

## How a frame is labelled

The match is processed as a whole, in two steps.

### 1. Dead ball (`D`)

| Situation | Dead ball from | Until |
|---|---|---|
| The ball goes more than 0.5 m past a line | the moment it crosses the line | the restart: the first of a kick or throw back into the pitch (ball faster than 8 m/s), a tagged set piece, or the players back at playing tempo for 2 s. Never earlier than 2 s, never later than 120 s. |
| A restart without the ball leaving the pitch: a tagged set piece, or a placed ball (still for 2 s) that is kicked | the moment the players stopped playing (median player speed below 2 m/s, searched up to 30 s back; at least 2 s) | the restart |

### 2. Possession (`H` / `A`) between dead balls

- A team has **close control** in a frame when its player is the nearest to the ball and within
  1.5 m of it.
- Possession passes to the other team only after it has **1 s of close control** without the first
  team getting the ball back. The change is dated back to the start of that run, so it has no
  delay. Blips shorter than 0.12 s count as tracking noise.
- Touches, deflections, balls in flight and loose balls do not change possession: it stays with
  the last team that had it.
- When the ball is not tracked, the team in possession keeps it.
- A stretch of play with evidence but no 1 s of control goes to the team with the most
  close-control frames. A stretch with no player near the ball at all is `X`.

All constants are parameters of `EnhancedPossessionClassifier` and are set from their football or
physical meaning, not fitted to any labels.

## Output

Without `--output`, the CSV is written to `results/<video name>.csv` in the project folder. The
video name is the name of the video file in the data folder, if there is exactly one. Otherwise
it is the tracking file name without `_atd`: for example, `match_tactical_atd.txt` gives
`results/match_tactical.csv`. Missing output folders are created automatically. `results/` is
ignored by git.

The file has exactly the columns `frame,label`, one row per tracking line, sorted by frame. Labels
are only `H`, `A`, `D` or `X`. Output frame = input frame − 1, so the CSV is indexed from 0, like
video frames.

Bad input stops the program with an explicit error instead of producing a wrong file: a missing
or ambiguous tracking file, missing metadata, a team count other than two, a wrong slot count,
malformed `x,y,speed` groups, and repeated or invalid frame numbers.

---

## Using it as a library

```python
from src.enhanced_classifier import EnhancedPossessionClassifier

c = EnhancedPossessionClassifier()                # every constant can be passed here
frames, ball, players = c.load_tactical_data("path/to/folder")
pred = c.classify_frames(frames, ball, players)   # frame, prediction, source
csv = c.generate_submission(pred)                 # checked frame,label table
c.get_statistics()                                # label shares, dead-ball stretches
c.dead_intervals                                  # [(start, end, reason), ...]
```

The `source` column says why each frame got its label (`dead:out`, `dead:restart`, `control`,
`weak` or `none`). It is not written to the CSV.

---

## Limitations

- **Thresholds are reasoned, not learned.** They follow from football and physics and from
  whole-match tracking statistics; they have not been optimised.
- **Stoppages inside the pitch** are only found when a restart is visible: a tagged set piece or a
  placed ball being kicked. An injury stoppage restarted by a quick pass is missed.
- **The start of such a stoppage** is estimated from the players slowing down, so slow build-up
  play just before a restart can be counted as dead time (at most 30 s).
- **A ball held perfectly still for 2 s and then kicked** is read as a set piece, even in open
  play.
- **Nearest is not control.** In crowded duels the nearest player may not be the one who has the
  ball.
- The restarting team from Smart Tagging, the homography and the broadcast video are not used.
- Everything runs on the CPU. One match per run, and the whole tracking file is held in memory. Pitch size is fixed at 105.3 × 68.0 m.

---

## Background

The project was built for the *More than a Hack 2027 — Phase 1: Ball Possession* challenge. No
labelled data of any kind was used to build or tune it. The challenge kit and the earlier
development files are no longer in the working tree but remain in git history; for example, the
challenge's original README can be shown with `git show 59da388:README.md`.

The match data is licensed separately for the competition and is not included in this repository.
