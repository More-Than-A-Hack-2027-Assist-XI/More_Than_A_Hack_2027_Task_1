# Ball Possession Classifier

Reads the player and ball tracking of one football match and labels **every frame**:

| Label | Meaning |
|---|---|
| `H` | the home team has the ball, ball in play |
| `A` | the away team has the ball, ball in play |
| `D` | dead ball: the ball is not in play |
| `X` | the camera is not showing the game, or there is no evidence for either team |

The result is a `frame,label` CSV with one row per frame. The program is rule-based: it needs no
training, no labelled data and no GPU, and the same input always gives a byte-identical file.
The full technical description is in [Project_Report.pdf](Project_Report.pdf).

---

## Quick start

Requirements: Python 3.11 or newer.

```bash
python -m pip install -r requirements.txt     # numpy and pandas, pinned

python main.py --data-dir <match folder>                        # -> results/<match name>.csv
python main.py --data-dir <match folder> --output <file.csv>    # choose the file yourself
```

| Option | Meaning | Default |
|---|---|---|
| `--data-dir` | folder with the match's tracking files | `./data/tactical` |
| `--output` | where to write the CSV | `results/<match name>.csv` |

---

## What `main.py` does

`main.py` runs the whole pipeline in four stages and reports each one on the console.

1. **Load the data.** It finds the tracking pair (`*_atd.txt` + `*_atd.xml`) in the data folder
   and, if present, the Smart Tagging file. Then it reads every frame's player and ball positions.
2. **Assign teams.** Every tracking slot is mapped to `H` or `A` from the metadata.
3. **Classify.** First the dead-ball stretches are found, then possession inside each stretch of
   play (see [How the algorithm works](#how-the-algorithm-works)).
4. **Write the CSV.** The labels are checked (integer frames, no duplicates, only H/A/D/X) and
   written with LF line endings. A missing output folder is created.

A real run on a full match (199,319 tracking frames, about 2 h 15 min) takes about 15 seconds on
a CPU:

```text
============================================================
BALL POSSESSION CLASSIFICATION PIPELINE
============================================================
Data Directory: ..\more-than-a-hack-2027\data\tactical
Output Path: ...\results\BAR-ATH_tactical.csv
============================================================

[1/4] Loading tactical data...
Smart Tagging: BAR-ATH_tactical_smart_tagging.xml (59 restarts)
Loaded 199319 frames
Ball positions tracked: 113488 frames

[2/4] Team assignments loaded...
Team assignments loaded for 46 players

[3/4] Running classification...
Dead-ball stretches: 116  (Smart Tagging restarts available: 59)
Classification complete. Generated 199319 predictions

[4/4] Generating submission CSV...
Submission saved to: ...\results\BAR-ATH_tactical.csv

============================================================
PIPELINE COMPLETE
============================================================
```

### What the output looks like

The CSV has one row per tracking frame, numbered from 0. Two excerpts from
the run above. The first is a change of possession from the home to the away team, the second the
ball going out of play:

```text
frame,label          frame,label
13669,H              17454,A
13670,H              17455,A
13671,A              17456,D
13672,A              17457,D
```

On that match the program produced:

| Result | Value |
|---|---|
| Frames labelled H / A / D / X | 37.8% / 28.7% / 24.0% / 9.4% |
| Dead-ball stretches | 116: 81 after the ball left the pitch, 35 before restarts inside it |
| Dead-ball stretch length | median 5.6 s, mean 16.5 s |
| Possessions | 240, mean length 22 s |

About four out of five `X` frames are before kick-off, at half-time and after the final whistle,
when nobody plays the ball.

Without `--output`, the file is named after the tracking file without `_atd`
(`match_tactical_atd.txt` → `results/match_tactical.csv`). `results/` is ignored by git.

---

## How the algorithm works

The challenge defines possession as follows. While the ball is in play, one team has it. It
changes hands only when the other team gains **controlled** possession. A loose or contested ball
stays with the last team that had it. A dead ball belongs to nobody.

A frame-by-frame rule ("the team of the nearest player") cannot follow this definition. It flips
whenever the nearest player changes, and it cannot tell an untracked ball from a dead one. The
program therefore looks at the **whole match at once**, in two steps.

```text
tracking + metadata ─┐
                     ├─► signals per frame ─► 1. dead-ball stretches ─► 2. possession in ─► frame,label
Smart Tagging (opt.) ┘                                                  each stretch of play     CSV
```

### Signals computed for every frame

| Signal | How it is computed |
|---|---|
| ball position | normalised coordinates × pitch size (105.3 × 68.0 m) |
| ball speed | distance between the positions 2 frames before and 2 frames after, divided by 0.16 s; the speed in the file is empty |
| ball out of the pitch | the ball is more than 0.5 m beyond a touchline or goal line |
| kick | the ball is on the pitch and moving faster than 8 m/s, faster than anyone can carry it |
| ball standing still | it moved less than 0.5 m during the last second |
| player tempo | the median speed of all tracked players, averaged over 1 s; at 2 m/s or more the players are playing, below it they are walking. Frames with fewer than 3 players tracked are left out |
| close control | the nearest player to the ball is within 1.5 m **and** the ball moves at dribbling pace (8 m/s or less); that player's team has close control in this frame. A ball flying past a player is not under control |
| kicked away by | the nearest player is within 1.5 m and the ball leaves at 8 m/s or more: that player's team has just played the ball |
| no view of the game | fewer than 3 players are tracked for at least 1 s: a close-up, a replay, the crowd or the VAR room. Any view of play shows more players, and shorter drops are tracking flicker |
| tagged restart | the start of a Smart Tagging `SET PIECES` instance (corner kick, free kick, goal kick, kick-off, throw-in, penalty) |

### Step 1: dead-ball stretches (`D`)

A stoppage has a start (the ball leaves the pitch or the referee stops play) and an end (the
restart). The program looks for both.

**A. The ball leaves the pitch** (throw-ins, goal kicks, corners, goals)

- **Start:** the first frame the ball is beyond the line, traced back from the first frame it is
  0.5 m out.
- **End:** the restart is the earliest of three events:
  1. a kick back into play, dated back to the frame where the ball started moving;
  2. a tagged restart;
  3. the players back at playing tempo for 2 s.
- The restart is searched no earlier than 2 s after the ball went out, since no restart is quicker.
  If none is found within 120 s, the stretch ends there.

**B. A restart without the ball leaving the pitch** (fouls, offside, kick-off after a goal,
penalties)

- **End (the restart):** either a tagged restart, or a placed ball, meaning a ball standing still
  for 2 s that is then kicked within 0.5 s. Restarts closer than 3 s together count as one. In that
  case the frame-exact placed-ball kick is preferred over the tag, which is only accurate to about
  a second.
- **Start (the whistle):** searched backwards from the restart, at the point where the players
  stopped playing (player tempo below 2 m/s), at most 30 s back and at least 2 s before the
  restart.
- A restart within 3 s of a stretch already found in A belongs to that stretch and adds nothing.

Overlapping stretches are merged. Every frame inside a stretch is `D`.

### Frames where the camera is not showing the game (`X`)

A broadcast feed cuts to close-ups, replays, the crowd or the VAR room, and the tracking then sees
only one or two people. Nothing can be said about possession there. Frames with no view of the
game are therefore `X`, and the team in possession is **not** carried across them. After the cut,
possession is decided afresh. The only exception is a cut inside a dead-ball stretch, such as a
replay shown during a stoppage: there the label stays `D`, because the ball is dead.

### Step 2: possession in each stretch of play (`H` / `A`)

Between two dead-ball stretches or cuts away from the game, the ball is in play, so exactly one
team has it.

1. **Close-control evidence.** Each frame where one team has close control is evidence for that
   team. There is no evidence when nobody is within 1.5 m or the ball is moving fast. That covers
   a pass or clearance in flight, a deflection and an untracked ball.
2. **Runs.** The evidence frames are grouped into runs of the same team. A run of fewer than 3
   frames (0.12 s) is tracking noise and is dropped, so a run is not broken by such a blip.
3. **Control.** A run is control in either of two cases. Passes between teammates keep a run
   going, because a run belongs to the team, not to one player.
   - **The team keeps the ball:** at least 1 s of close control (dribbling, shielding).
   - **A player receives the ball and plays it:** at least 0.2 s of close control, then the ball
     is kicked away by the same team within 0.3 s. This is the quick trap-and-pass or
     first-touch pass. A deflection cannot qualify, because the ball never slows down at the
     player.
4. **Labelling.** The first team with control owns the stretch from its first frame. Possession
   passes to the other team only when that team has control, and the change is **dated back to
   the first frame of that run**. The label therefore changes when the interception happened, not
   one second later.
5. **Everything else keeps the current team:** touches, deflections, aerial duels, loose balls and
   frames without a tracked ball.
6. **Edge cases.** A stretch with evidence but no 1 s of control goes to the team with the most
   evidence frames. A stretch with no evidence at all is `X`.

### Worked example

| Time | On the pitch | Evidence | Label |
|---|---|---|---|
| 0.0–4.0 s | a home player dribbles | home, close control | `H` |
| 4.0–4.1 s | a pass by the home team brushes an away defender's leg | ball fast: no evidence | `H` |
| 4.1–7.0 s | the pass rolls loose, nobody near the ball | none | `H` |
| 7.0–7.4 s | an away player runs onto it, takes it in his stride and passes forward | away, 0.4 s at his feet, then kicked away by him: control | `A` from 7.0 s |
| 7.4–9.0 s | the away pass in the air, then an away teammate keeps the ball | away | `A` |
| 9.0 s | the ball crosses the touchline | ball out | `D` from 9.0 s |
| 9.0–15.0 s | the throw-in is prepared | — | `D` |
| 15.0 s | the throw-in is taken, ball faster than 8 m/s | kick: restart | play from 15.0 s |
| 15.0 s → | the team that takes control after the throw-in | control | that team from 15.0 s |

### Parameters

All constants are arguments of `EnhancedPossessionClassifier`. Each was chosen from its football or
physical meaning, checked against whole-match tracking statistics (how fast the ball moves, how
often it leaves the pitch or stands still). None was fitted to labels.

| Parameter | Value | Meaning |
|---|---|---|
| `control_radius_m` | 1.5 m | about one stride, plus tracking noise |
| `min_control_s` | 1.0 s | close control that counts as keeping the ball |
| `control_max_speed_mps` | 8 m/s | a ball under control moves at dribbling pace at most |
| `min_played_s` | 0.2 s | close control before a pass that counts as receiving the ball |
| `played_window_s` | 0.3 s | the pass must follow within this time to count as playing it |
| `noise_run_s` | 0.12 s | shorter runs are tracking noise |
| `out_margin_m` | 0.5 m | the ball must be this far out (absorbs ball-position noise) |
| `kick_speed_mps` | 8 m/s | a kicked or thrown ball |
| `still_move_m` | 0.5 m | movement in 1 s below which the ball is standing still |
| `placed_ball_s` | 2.0 s | a ball still this long has been placed for a restart |
| `min_dead_s` / `max_dead_s` | 2 s / 120 s | shortest stoppage; longest searched after the ball goes out |
| `stoppage_lookback_s` | 30 s | how far back the start of a stoppage is searched |
| `play_tempo_mps` | 2.0 m/s | median player speed of play rather than walking |
| `resume_hold_s` | 2.0 s | how long that tempo must hold for play to count as resumed |
| `min_players_in_view` | 3 | fewer tracked players means the camera is not showing the game |
| `no_view_min_s` | 1.0 s | how long that must last; shorter drops are tracking flicker |

---

## Input files

The data folder must contain **exactly one** `*_atd.txt` file and, next to it, an `.xml` file
with the same name stem. File and folder names are free; only the top level of the folder is
searched.

- **Tracking (`*_atd.txt`)**, EPTS format. Each line is `frame:players:ball`. Player slots are
  separated by `;`, in the same order and number as in the metadata. Each object is `x,y,speed`
  with normalised coordinates, and `NaN` means not visible. Frame numbers start at 1 and must not
  repeat.
- **Metadata (`*_atd.xml`).** Exactly two `<Team>` elements: the first is `H`, the second `A`.
  There is one `<Player id teamId>` per slot. `<FrameRate>` is read (default 25).
- **Smart Tagging (optional).** `<prefix>_smart_tagging.xml` (prefix = tracking file name without
  `_atd.txt`), or else the only `*_smart_tagging.xml` in the folder. Only `SET PIECES` restarts
  are read. Without the file, restarts come from tracking alone.

Bad input stops the program with an explicit error instead of a wrong file. That covers a missing
or ambiguous tracking file, missing metadata or attributes, a team count other than two, a wrong
slot count, malformed `x,y,speed` groups, and repeated or invalid frame numbers.

---

## Using it as a library

```python
from src.enhanced_classifier import EnhancedPossessionClassifier

c = EnhancedPossessionClassifier()                # any parameter above can be passed here
frames, ball, players = c.load_tactical_data("path/to/folder")
pred = c.classify_frames(frames, ball, players)   # columns: frame, prediction, source
csv = c.generate_submission(pred)                 # checked frame,label table
c.get_statistics()                                # label shares, number of dead-ball stretches
c.dead_intervals                                  # [(start, end, reason), ...]
```

The `source` column explains each label:

| Source | Meaning |
|---|---|
| `dead:out` | dead ball after the ball left the pitch |
| `dead:restart` | dead ball before a restart inside the pitch |
| `control` | possession from close control |
| `weak` | stretch without 1 s of control, given to the team with the most evidence |
| `no view` | the camera is not showing the game (`X`) |
| `none` | no evidence (`X`) |

---

## Limitations

- **The thresholds are reasoned, not learned.** They have not been optimised.
- **Stoppages inside the pitch** are only found when a restart is visible: a tagged set piece or
  a placed ball being kicked. An injury stoppage restarted by a quick pass is missed.
- **The start of such a stoppage** is estimated from the players slowing down. Slow build-up play
  just before a restart can therefore be counted as dead time (at most 30 s).
- **A ball held perfectly still for 2 s and then kicked** is read as a set piece, even in open
  play.
- **Nearest is not control.** In crowded duels the nearest player may not be the one with the
  ball.
- **A tackle can count as control** when the ball was slow at the tackler's feet for 0.2 s before
  he played it away.
- The algorithm uses tracking data and Smart Tagging only. The videos are never read, and the
  homography and the restarting team from Smart Tagging are not used.
- The program runs on the CPU, one match per run, and holds the whole tracking file in memory.

---

## Development and data use

Built for the *More than a Hack 2027, Phase 1: Ball Possession* challenge. No model was trained
and no labelled data was used to fit anything. The challenge's public 5-minute ground truth was
looked at to understand what the labels mean, as the challenge invites. The finished program was
scored on it once as a correctness check. No parameter was chosen from that score.

The match data is licensed separately and is not included in this repository. The challenge kit
and earlier development files remain in git history (for example
`git show 59da388:README.md`).
