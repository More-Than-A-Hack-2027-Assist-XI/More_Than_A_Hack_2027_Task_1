"""
Ball possession classifier for EPTS tracking data.

Every frame gets one label:
    H / A  home / away team in possession, ball in play
    D      ball not in play (dead ball)
    X      no evidence for any team in this stretch of play

The decision is made over the whole match, not frame by frame:

1. Dead ball (D) is built from explicit restart logic:
   - the ball leaves the pitch -> dead until the restart is taken
     (a kick or throw back into the pitch, a tagged set piece, or the players
     resuming play);
   - a restart without a preceding exit (free kick, kick-off, penalty):
     tagged set pieces from Smart Tagging, or a placed ball (standing still)
     that is then kicked. The stoppage before it starts where the players
     stopped playing.
2. Possession (H/A) between dead-ball stretches only changes hands when the
   other team gains *controlled* possession: one of its players must be close
   to the ball for a minimum accumulated time. Touches, deflections and balls
   in flight do not change possession. A change is dated back to the start of
   the controlling run, so there is no detection delay.
3. When the ball is not tracked during play, the team in possession keeps it.

All constants have a physical or football meaning (see __init__) and were set
from that meaning and from whole-match tracking statistics, not fitted to any
labels.
"""

import xml.etree.ElementTree as ET
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LABELS = ("H", "A", "D", "X")


def _attr(element: ET.Element, name: str) -> str:
    """Required XML attribute, with a clear error when it is missing."""
    value = element.get(name)
    if value is None:
        raise ValueError(f'<{element.tag}> element without required "{name}" attribute')
    return value
RESTART_TYPES = ("corner kick", "free kick", "goal kick", "kick off", "throw-in",
                 "throw in", "penalty")


class EnhancedPossessionClassifier:
    """Rule-based possession classifier with dead-ball and controlled-possession logic."""

    def __init__(self,
                 control_radius_m: float = 1.5,
                 min_control_s: float = 1.0,
                 noise_run_s: float = 0.12,
                 out_margin_m: float = 0.5,
                 kick_speed_mps: float = 8.0,
                 still_move_m: float = 0.5,
                 placed_ball_s: float = 2.0,
                 min_dead_s: float = 2.0,
                 max_dead_s: float = 120.0,
                 stoppage_lookback_s: float = 30.0,
                 play_tempo_mps: float = 2.0,
                 resume_hold_s: float = 2.0):
        """
        Args:
            control_radius_m: a player this close to the ball can be playing it
                (about one stride, plus tracking noise).
            min_control_s: accumulated close-control time a team needs before
                possession passes to it. The rules call a reception controlled
                when the player keeps the ball; 1 s is the conservative end.
            noise_run_s: runs of a team near the ball shorter than this (3 frames)
                are treated as tracking noise.
            out_margin_m: the ball counts as out of the pitch when it is this far
                beyond a line (absorbs position noise of the tracked ball).
            kick_speed_mps: ball speed that means it was kicked or thrown; a ball
                carried by a walking player stays well below it.
            still_move_m: the ball is standing still when it moves less than this
                in one second.
            placed_ball_s: a ball standing still this long is placed for a restart.
            min_dead_s: shortest possible stoppage; no restart is taken sooner.
            max_dead_s: longest stoppage searched after the ball leaves the pitch.
            stoppage_lookback_s: how far back a stoppage before a restart is searched.
            play_tempo_mps: median player speed above which the players are playing
                rather than walking through a stoppage.
            resume_hold_s: how long that tempo must hold to count as play resumed.
        """
        self.control_radius_m = control_radius_m
        self.min_control_s = min_control_s
        self.noise_run_s = noise_run_s
        self.out_margin_m = out_margin_m
        self.kick_speed_mps = kick_speed_mps
        self.still_move_m = still_move_m
        self.placed_ball_s = placed_ball_s
        self.min_dead_s = min_dead_s
        self.max_dead_s = max_dead_s
        self.stoppage_lookback_s = stoppage_lookback_s
        self.play_tempo_mps = play_tempo_mps
        self.resume_hold_s = resume_hold_s

        # Pitch size in metres (normalised coordinates are multiplied by these).
        self.pitch_length = 105.3
        self.pitch_width = 68.0
        self.fps = 25.0

        self.player_teams: Dict[int, str] = {}
        self.track_ids: List[int] = []
        self.restart_tags: List[Tuple[int, str]] = []
        self.dead_intervals: List[Tuple[int, int, str]] = []
        self.stats = {'total_frames': 0, 'ball_tracked': 0,
                      'predictions': {k: 0 for k in LABELS}}

    # ------------------------------------------------------------------ loading
    def load_tactical_data(self, data_dir):
        """Load tracking (and Smart Tagging, if present) on the zero-based video timeline."""
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
        root = ET.parse(meta).getroot()
        rate = root.findtext('.//FrameRate')
        if rate:
            self.fps = float(rate)
        self.player_teams = self.load_atd_team_assignments(meta)
        self.track_ids = [int(_attr(p, 'id')) for p in root.iter('Player')]
        ball, players = self.load_atd_tracking(raw)
        self.frames = sorted(ball)

        tagging = self.find_smart_tagging(data_dir, raw)
        self.restart_tags = self.load_restart_tags(tagging) if tagging else []
        if tagging:
            print(f'Smart Tagging: {tagging.name} ({len(self.restart_tags)} restarts)')
        else:
            print('Smart Tagging: not found, restarts are detected from tracking only')
        return self.frames, ball, players

    def load_atd_team_assignments(self, xml_path):
        """Use XML track IDs and the EPTS home/away team ordering."""
        root = ET.parse(xml_path).getroot()
        teams = list(root.iter('Team'))
        if len(teams) != 2:
            raise ValueError('Expected exactly two teams in EPTS metadata')
        labels = {_attr(teams[0], 'id'): 'H', _attr(teams[1], 'id'): 'A'}
        return {int(_attr(p, 'id')): labels[_attr(p, 'teamId')] for p in root.iter('Player')}

    def load_atd_tracking(self, txt_path):
        """Read frame:players:ball; each object is normalized x,y,speed."""
        raw = Path(txt_path)
        meta = raw.with_suffix('.xml')
        ids = [int(_attr(p, 'id')) for p in ET.parse(meta).getroot().iter('Player')]
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

    @staticmethod
    def find_smart_tagging(data_dir: Path, raw: Path) -> Optional[Path]:
        """Smart Tagging file of the same recording: <prefix>_smart_tagging.xml."""
        prefix = raw.name[:-len('_atd.txt')]
        same = data_dir / f'{prefix}_smart_tagging.xml'
        if same.exists():
            return same
        others = sorted(data_dir.glob('*_smart_tagging.xml'))
        return others[0] if len(others) == 1 else None

    def load_restart_tags(self, xml_path) -> List[Tuple[int, str]]:
        """Set-piece instances as (video frame of the restart, restart type)."""
        restarts = []
        for inst in ET.parse(xml_path).getroot().iter('instance'):
            code = (inst.findtext('code') or '').strip().lower()
            labels = {(l.findtext('group') or '').strip().lower(): (l.findtext('text') or '').strip()
                      for l in inst.findall('label')}
            kind = labels.get('type', '').lower()
            if code == 'set pieces' and any(kind.startswith(t) for t in RESTART_TYPES):
                start = inst.findtext('start')
                if start is not None:
                    restarts.append((int(round(float(start) * self.fps)), kind))
        return sorted(restarts)

    # ---------------------------------------------------------- frame helpers
    def get_nearest_player_to_ball(self, frame_num, ball_pos, player_positions):
        """Nearest tracked player to the ball (id, team), distances in metres."""
        if ball_pos is None or not player_positions:
            return None, None
        best_id, best_distance = None, float('inf')
        for pid, pos in player_positions.items():
            dx = (pos['normalized_x'] - ball_pos['normalized_x']) * self.pitch_length
            dy = (pos['normalized_y'] - ball_pos['normalized_y']) * self.pitch_width
            distance = dx * dx + dy * dy
            if np.isfinite(distance) and distance < best_distance:
                best_id, best_distance = pid, distance
        if best_id is None:
            return None, None
        return best_id, self.player_teams.get(best_id)

    def _arrays(self, frames, ball, players):
        """Dense per-frame arrays (metres) indexed by video frame."""
        n = max(frames) + 1
        slots = {pid: i for i, pid in enumerate(self.track_ids)}
        bx = np.full(n, np.nan); by = np.full(n, np.nan)
        px = np.full((n, len(slots)), np.nan, dtype=np.float32)
        py = np.full_like(px, np.nan); ps = np.full_like(px, np.nan)
        for f in frames:
            b = ball.get(f)
            if b is not None:
                bx[f] = b['normalized_x'] * self.pitch_length
                by[f] = b['normalized_y'] * self.pitch_width
            for pid, p in players.get(f, {}).items():
                i = slots.get(pid)
                if i is not None:
                    px[f, i] = p['normalized_x'] * self.pitch_length
                    py[f, i] = p['normalized_y'] * self.pitch_width
                    ps[f, i] = p['speed']
        team = np.array([self.player_teams.get(pid, '') for pid in self.track_ids])
        return n, bx, by, px, py, ps, team

    # ------------------------------------------------------------ classifier
    def classify_frames(self, video_frames, ball_positions_dict, player_positions_dict, **_ignored):
        """Label every frame. Returns DataFrame(frame, prediction, source)."""
        frames = sorted(video_frames)
        if not frames:
            return pd.DataFrame(columns=['frame', 'prediction', 'source'])
        n, bx, by, px, py, ps, team = self._arrays(frames, ball_positions_dict, player_positions_dict)
        fps = self.fps
        sec = lambda s: max(1, int(round(s * fps)))
        L, W = self.pitch_length, self.pitch_width

        vis = np.isfinite(bx)
        # Ball speed from positions, central difference over 0.16 s.
        speed = np.full(n, np.nan)
        speed[2:-2] = np.hypot(bx[4:] - bx[:-4], by[4:] - by[:-4]) / (4 / fps)
        # Ball standing still: moved less than still_move_m during the last second.
        w = sec(1.0)
        still = np.zeros(n, bool)
        still[w:] = np.hypot(bx[w:] - bx[:-w], by[w:] - by[:-w]) < self.still_move_m
        beyond = np.maximum.reduce([-bx, bx - L, -by, by - W])       # metres past a line
        outside = vis & (beyond > 0)
        out = vis & (beyond > self.out_margin_m)
        inside = vis & (beyond <= self.out_margin_m)
        kick = inside & (speed >= self.kick_speed_mps)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', category=RuntimeWarning)
            tempo = np.nanmedian(ps, axis=1)
        tempo = pd.Series(tempo).rolling(sec(1.0), min_periods=1).mean().to_numpy()
        playing = tempo >= self.play_tempo_mps
        hold = sec(self.resume_hold_s)
        resumed = pd.Series(playing.astype(float)).rolling(hold).min().to_numpy() == 1  # end of a hold window

        def first(mask, lo, hi):
            lo, hi = max(lo, 0), min(hi, n)
            idx = np.flatnonzero(mask[lo:hi])
            return lo + idx[0] if idx.size else None

        dead: List[Tuple[int, int, str]] = []
        min_dead, max_dead = sec(self.min_dead_s), sec(self.max_dead_s)
        tags = [f for f, _ in self.restart_tags]

        # 1) Ball leaves the pitch -> dead until the restart.
        o = 0
        while True:
            o = first(out, o, n)
            if o is None:
                break
            start = o
            while start > 0 and outside[start - 1]:
                start -= 1
            lo, hi = o + min_dead, o + max_dead
            ends = []
            k = first(kick, lo, hi)
            if k is not None:
                while k > lo and inside[k - 1] and speed[k - 1] > 1.0:
                    k -= 1          # back to the moment the ball started moving
                ends.append(k)
            ends += [t for t in tags if lo <= t < hi][:1]
            r = first(resumed, lo + hold, hi)
            if r is not None:
                ends.append(r - hold + 1)
            end = min(ends) if ends else min(hi, n)
            dead.append((start, end, 'out'))
            o = max(end, o + 1)

        # 2) Restarts without a preceding exit: tagged set pieces and placed balls.
        restarts = [(t, 'tag') for t in tags]
        placed = sec(self.placed_ball_s)
        run = 0
        for f in range(n):
            run = run + 1 if (vis[f] and still[f]) else 0
            if run >= placed and f + 1 < n and not still[f + 1]:
                k = first(kick, f + 1, f + 1 + sec(0.5))
                if k is not None:
                    restarts.append((f + 1, 'placed'))
        restarts.sort()
        merged: List[Tuple[int, str]] = []
        for f, kind in restarts:                       # one restart per 3 s
            if merged and f - merged[-1][0] <= sec(3.0):
                if kind == 'placed':                   # tracking is frame-exact, tags ~1 s
                    merged[-1] = (f, 'tag+placed' if merged[-1][1] != 'placed' else 'placed')
                continue
            merged.append((f, kind))

        def covered(f):
            return any(a - sec(3.0) <= f <= b + sec(3.0) for a, b, _ in dead)

        lookback = sec(self.stoppage_lookback_s)
        for r, kind in merged:
            if r >= n or covered(r):
                continue
            s = r - min_dead
            limit = max(0, r - lookback)
            while s > limit and not playing[s - 1]:
                s -= 1
            dead.append((s, r, 'restart'))

        dead.sort()
        flat: List[Tuple[int, int, str]] = []
        for a, b, why in dead:                         # merge overlaps
            if flat and a <= flat[-1][1]:
                pa, pb, pw = flat[-1]
                flat[-1] = (pa, max(pb, b), pw)
            else:
                flat.append((a, b, why))
        self.dead_intervals = flat

        label = np.full(n, 'X', dtype='<U1')
        source = np.full(n, 'none', dtype=object)
        for a, b, why in flat:
            label[a:b] = 'D'
            source[a:b] = f'dead:{why}'

        # 3) Possession between dead stretches.
        valid = np.isfinite(px) & np.isfinite(bx)[:, None]
        d2 = np.where(valid, (px - bx[:, None]) ** 2 + (py - by[:, None]) ** 2, np.inf)
        nearest = d2.argmin(axis=1)
        near_d = np.sqrt(d2[np.arange(n), nearest])
        cand = np.where(near_d <= self.control_radius_m, team[nearest], '')

        bounds, prev = [], 0
        for a, b, _ in flat:
            if a > prev:
                bounds.append((prev, a))
            prev = max(prev, b)
        if prev < n:
            bounds.append((prev, n))
        need, noise = sec(self.min_control_s), sec(self.noise_run_s)
        for a, b in bounds:
            self._label_segment(a, b, cand, label, source, need, noise)

        frame_set = np.array(frames)
        result = pd.DataFrame({'frame': frame_set,
                               'prediction': label[frame_set],
                               'source': source[frame_set]})
        self.stats['total_frames'] = len(frames)
        self.stats['ball_tracked'] = int(vis[frame_set].sum())
        self.stats['predictions'] = result.prediction.value_counts().reindex(list(LABELS), fill_value=0).to_dict()
        return result

    @staticmethod
    def _label_segment(a, b, cand, label, source, need, noise):
        """Possession within one stretch of play [a, b)."""
        idx = [f for f in range(a, b) if cand[f] in ('H', 'A')]
        if not idx:
            return                                     # no evidence: stays X
        runs = []                                      # [team, first frame, count]
        for f in idx:
            if runs and runs[-1][0] == cand[f]:
                runs[-1][2] += 1
            else:
                runs.append([cand[f], f, 1])
        runs = [r for r in runs if r[2] >= noise]      # drop tracking-noise blips
        merged = []
        for r in runs:
            if merged and merged[-1][0] == r[0]:
                merged[-1][2] += r[2]
            else:
                merged.append(list(r))
        control = [r for r in merged if r[2] >= need]
        if not control:
            totals = {t: sum(r[2] for r in merged if r[0] == t) for t in ('H', 'A')}
            best = max(totals, key=lambda t: totals[t]) if merged else cand[idx[0]]
            label[a:b] = best
            source[a:b] = 'weak'
            return
        current, start = control[0][0], a
        for team, first, _ in control[1:]:
            if team != current:
                label[start:first] = current
                source[start:first] = 'control'
                current, start = team, first
        label[start:b] = current
        source[start:b] = 'control'

    # --------------------------------------------------------------- output
    def generate_submission(self, predictions_df):
        """The official format contains exactly frame,label."""
        submission = predictions_df.rename(columns={'prediction': 'label'})[['frame', 'label']].copy()
        submission['frame'] = pd.to_numeric(submission['frame'], errors='raise').astype(int)
        if submission['frame'].duplicated().any():
            raise ValueError('Duplicate submission frames')
        if not submission['label'].isin(list(LABELS)).all():
            raise ValueError('Invalid submission labels')
        return submission.sort_values('frame').reset_index(drop=True)

    def get_statistics(self) -> Dict:
        """Label shares, ball-tracking rate and number of dead-ball stretches."""
        total = self.stats['total_frames']
        return {
            'total_frames': total,
            'ball_tracked': self.stats['ball_tracked'],
            'ball_tracking_rate': self.stats['ball_tracked'] / total if total else 0,
            'dead_ball_stretches': len(self.dead_intervals),
            'restart_tags': len(self.restart_tags),
            'prediction_distribution': {k: v / total if total else 0
                                        for k, v in self.stats['predictions'].items()},
        }


def main():
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from main import main as pipeline_main
    return pipeline_main()


if __name__ == "__main__":
    main()
