"""Play a video with frame-indexed possession labels from a frame,label CSV.

Usage: python video_player.py VIDEO CSV
Dependency: python -m pip install opencv-python
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path


LABEL_NAMES = {
    "H": "Ev sahibi",
    "A": "Deplasman",
    "D": "Oyun durdu / top oyun disinda",
    "X": "Degerlendirme disi",
}
COLORS = {
    "H": (100, 240, 100),
    "A": (80, 190, 255),
    "D": (100, 100, 255),
    "X": (200, 200, 200),
}


def load_labels(path: Path) -> dict[int, str]:
    labels: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not {"frame", "label"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV basligi 'frame,label' sutunlarini icermeli.")
        for row in reader:
            try:
                frame = int(row["frame"])
                label = row["label"].strip().upper()
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"CSV satir {reader.line_num}: gecersiz veri.") from exc
            if frame < 0 or label not in LABEL_NAMES:
                raise ValueError(f"CSV satir {reader.line_num}: gecersiz kare veya etiket.")
            if frame in labels:
                raise ValueError(f"CSV icinde tekrarlanan kare: {frame}")
            labels[frame] = label
    if not labels:
        raise ValueError("CSV bos.")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--max-width", type=int, default=1280,
                        help="Gosterim genisligi ust siniri (varsayilan: 1280).")
    args = parser.parse_args()
    if args.max_width < 320:
        parser.error("--max-width en az 320 olmali.")
    try:
        labels = load_labels(args.csv)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    try:
        import cv2
    except ImportError:
        parser.exit(1, "OpenCV gerekli: python -m pip install opencv-python\n")

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        cap.release()
        parser.error(f"Video acilamadi: {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not math.isfinite(fps) or fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        parser.error("Videonun kare sayisi okunamadi.")
    import tkinter as tk
    from tkinter import ttk

    root = tk.Tk()
    root.title(f"Possession Player — {args.video.name}")
    root.configure(bg="#141820")
    root.geometry(f"{min(args.max_width, root.winfo_screenwidth() - 100)}x750")
    root.minsize(640, 420)
    canvas = tk.Canvas(root, bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)
    controls = ttk.Frame(root, padding=12)
    controls.pack(fill="x")
    position = tk.DoubleVar(value=0)
    time_text = tk.StringVar()
    status = tk.StringVar()
    playing = True
    dragging = False
    current = 0
    image = None
    photo = None
    deadline = time.perf_counter()

    def timestamp(seconds):
        milliseconds = round(seconds * 1000)
        hours, rest = divmod(milliseconds, 3600000)
        minutes, rest = divmod(rest, 60000)
        seconds, ms = divmod(rest, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"

    def update_time(frame):
        time_text.set(f"{timestamp(frame / fps)} / {timestamp(total / fps)}")

    def render(event=None):
        nonlocal photo
        if image is None:
            return
        width, height = max(1, canvas.winfo_width()), max(1, canvas.winfo_height())
        ratio = min(width / image.shape[1], height / image.shape[0])
        view = cv2.resize(image, (max(1, round(image.shape[1] * ratio)),
                                  max(1, round(image.shape[0] * ratio))))
        rgb = cv2.cvtColor(view, cv2.COLOR_BGR2RGB)
        # Native Tk PPM image support avoids an additional imaging dependency.
        data = f"P6\n{rgb.shape[1]} {rgb.shape[0]}\n255\n".encode() + rgb.tobytes()
        photo = tk.PhotoImage(data=data, format="PPM")
        canvas.delete("all")
        canvas.create_image(width // 2, height // 2, image=photo)
        label = labels.get(current, "?")
        names = {"H": "Ev sahibi", "A": "Deplasman",
                 "D": "Oyun durmuş / top oyun dışında", "X": "Değerlendirme dışı"}
        title = f"{label} ({names.get(label, 'CSV kaydı yok')})"
        canvas.create_rectangle(10, 10, min(width - 10, 560), 80, fill="#141820", outline="")
        canvas.create_text(24, 28, text=title, fill="white", anchor="w",
                           font=("Segoe UI", 17, "bold"))
        canvas.create_text(24, 60, text=f"Kare: {current}", fill="#cbd5e1",
                           anchor="w", font=("Segoe UI", 11))

    def read_frame():
        nonlocal image
        ok, decoded = cap.read()
        if not ok:
            status.set("Video karesi okunamadı.")
            return False
        image = decoded
        position.set(current)
        update_time(current)
        render()
        return True

    def seek(frame):
        nonlocal current, deadline
        target = max(0, min(total - 1, int(frame)))
        if cap.set(cv2.CAP_PROP_POS_FRAMES, target):
            current = target
            read_frame()
            deadline = time.perf_counter() + 1 / fps
        else:
            status.set("Bu videoda konuma atlama desteklenmiyor.")

    def toggle(event=None):
        nonlocal playing, deadline
        if current == total - 1 and not playing:
            seek(0)
        playing = not playing
        play_button.configure(text="Duraklat" if playing else "Oynat")
        deadline = time.perf_counter() + 1 / fps
        return "break"

    def jump(frames, pause=False):
        nonlocal playing
        if pause:
            playing = False
            play_button.configure(text="Oynat")
        seek(current + frames)
        return "break"

    def begin_drag(event):
        nonlocal dragging
        dragging = True
        position.set(frame_at_pointer(event))
        seek(position.get())
        # Suppress ttk's default trough click, which only moves by one step.
        return "break"

    def frame_at_pointer(event):
        start_x = timeline.coords(0)[0]
        end_x = timeline.coords(total - 1)[0]
        local_x = event.x_root - timeline.winfo_rootx()
        fraction = (local_x - start_x) / max(1, end_x - start_x)
        return round(max(0.0, min(1.0, fraction)) * (total - 1))

    def move_drag(event):
        if dragging:
            position.set(frame_at_pointer(event))
            update_time(int(position.get()))
        return "break"

    def preview(value):
        if dragging:
            update_time(int(float(value)))

    def end_drag(event):
        nonlocal dragging
        if dragging:
            position.set(frame_at_pointer(event))
            seek(position.get())
            dragging = False

    timeline = ttk.Scale(controls, from_=0, to=total - 1, variable=position,
                         command=preview, takefocus=False)
    timeline.pack(fill="x", pady=(0, 10))
    timeline.bind("<ButtonPress-1>", begin_drag)
    timeline.bind("<B1-Motion>", move_drag)
    root.bind("<ButtonRelease-1>", end_drag, add="+")
    row = ttk.Frame(controls)
    row.pack(fill="x")
    ttk.Button(row, text="−5 sn", command=lambda: jump(-round(5 * fps)), takefocus=False).pack(side="left")
    play_button = ttk.Button(row, text="Duraklat", command=toggle, takefocus=False)
    play_button.pack(side="left", padx=6)
    ttk.Button(row, text="+5 sn", command=lambda: jump(round(5 * fps)), takefocus=False).pack(side="left")
    ttk.Button(row, text="Önceki kare", command=lambda: jump(-1, True), takefocus=False).pack(side="left", padx=6)
    ttk.Button(row, text="Sonraki kare", command=lambda: jump(1, True), takefocus=False).pack(side="left")
    ttk.Label(row, textvariable=time_text).pack(side="right")
    ttk.Label(controls, text="Space: oynat/duraklat    ←/→: 5 saniye    A/D: kare    Esc: çıkış").pack(anchor="w", pady=(10, 0))
    ttk.Label(controls, textvariable=status).pack(anchor="w")
    if len(labels) != total:
        status.set(f"Uyarı: Video {total} kare, CSV {len(labels)} satır. Eksik etiketler ? gösterilir.")
    canvas.bind("<Configure>", render)
    canvas.bind("<Button-1>", lambda event: canvas.focus_set())
    root.bind("<space>", toggle)
    root.bind("<Left>", lambda event: jump(-round(5 * fps)))
    root.bind("<Right>", lambda event: jump(round(5 * fps)))
    root.bind("a", lambda event: jump(-1, True))
    root.bind("d", lambda event: jump(1, True))
    root.bind("<Escape>", lambda event: root.destroy())

    def tick():
        nonlocal current, playing, deadline
        if playing and not dragging and time.perf_counter() >= deadline:
            if current < total - 1:
                current += 1
                if not read_frame():
                    playing = False
                # Keep all frames visible; avoid accumulating processing-time drift.
                deadline = max(deadline + 1 / fps, time.perf_counter())
            else:
                playing = False
            if not playing:
                play_button.configure(text="Oynat")
        root.after(5, tick)

    try:
        root.update_idletasks()
        read_frame()
        deadline = time.perf_counter() + 1 / fps
        root.after(5, tick)
        root.mainloop()
    finally:
        cap.release()


if __name__ == "__main__":
    main()
