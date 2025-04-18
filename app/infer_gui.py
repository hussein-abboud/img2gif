#!filepath: app/infer_gui.py

import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional

import imageio
import torch
from PIL import Image, ImageTk, ImageSequence
from torchvision import transforms
import tkinter as tk
from tkinter import filedialog, messagebox

import inspect
import logging

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.model import models_regression_sequence
from src.utils.project_paths import MODEL_DIR, GROUND_TRUTH_DIR, IMG_SIZE

DISPLAY_SCALE = 4
DISPLAY_FPS = 10
MAX_FRAMES = 14
CONTAINER_SIZE = IMG_SIZE * DISPLAY_SCALE

BG = "#1e1e1e"
SIDEBAR_BG = "#252526"
FG = "#ffffff"
BTN_BG = "#3d7bfd"
BTN_HOVER = "#2e61d2"
CARD_BG = "#2d2d30"
BORDER_CLR = "#3c3c41"


def discover_models() -> Dict[str, Path]:
    model_classes = {
        name: cls for name, cls in inspect.getmembers(models_regression_sequence, inspect.isclass)
        if issubclass(cls, torch.nn.Module)
    }
    return {
        pt.stem: pt for pt in MODEL_DIR.glob("*.pt")
        if pt.stem in model_classes
    }


class InferenceApp(tk.Frame):
    def __init__(self, master: tk.Tk):
        super().__init__(master, bg=BG)
        master.title("GIF Predictor – Model Comparator")
        master.configure(bg=BG)
        master.geometry("1200x700")

        self.model_files = discover_models()
        self.selected_models: Dict[str, bool] = {k: False for k in self.model_files}
        self.model_instances: Dict[str, torch.nn.Module] = {}
        self.pred_sequences: Dict[str, torch.Tensor] = {}
        self.gt_sequence: Optional[List[Image.Image]] = None
        self.image_path: Optional[Path] = None
        self.scrub_active: bool = False

        self._build_ui()
        self._refresh_model_sidebar()

    def _build_ui(self) -> None:
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self.pack(fill="both", expand=True)

        self.sidebar = tk.Frame(self, padx=10, pady=10, bg=SIDEBAR_BG)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        tk.Label(self.sidebar, text="Models", font=("Segoe UI", 12, "bold"), fg=FG, bg=SIDEBAR_BG).pack(anchor="w")
        self.model_frame = tk.Frame(self.sidebar, bg=SIDEBAR_BG)
        self.model_frame.pack(fill="x", pady=(5, 15))

        def make_btn(text, cmd):
            return tk.Button(self.sidebar, text=text, width=18, command=cmd,
                             bg=BTN_BG, fg=FG, relief="flat", bd=0,
                             activebackground=BTN_HOVER, activeforeground=FG)

        make_btn("Load Image", self.load_image).pack(pady=2)
        make_btn("Load Ground‑Truth", self.load_ground_truth).pack(pady=2)
        make_btn("Refresh / Run", self.run_inference).pack(pady=10)

        self.slider = tk.Scale(self.sidebar, from_=0, to=MAX_FRAMES - 1, orient="horizontal",
                               label="Scrub Frame", command=self.update_display,
                               bg=SIDEBAR_BG, fg=FG, troughcolor=BORDER_CLR, highlightthickness=0)
        self.slider.pack(fill="x")

        self.display_frame = tk.Frame(self, padx=10, pady=10, bg=BG)
        self.display_frame.grid(row=0, column=1, sticky="nsew")
        self.display_frame.rowconfigure(0, weight=1)
        self.display_frame.columnconfigure(0, weight=1)

        self.canvas_container = tk.Frame(self.display_frame, bg=BG)
        self.canvas_container.pack(expand=True, fill="both")

    def _refresh_model_sidebar(self) -> None:
        for child in self.model_frame.winfo_children():
            child.destroy()
        for name in sorted(self.model_files.keys()):
            var = tk.BooleanVar(value=self.selected_models.get(name, False))
            def _toggle(n=name, v=var):
                self.selected_models[n] = v.get()
            tk.Checkbutton(self.model_frame, text=name, variable=var, command=_toggle,
                           fg=FG, bg=SIDEBAR_BG, activebackground=SIDEBAR_BG,
                           selectcolor=SIDEBAR_BG, highlightthickness=0).pack(anchor="w")

    def load_image(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PNG image", "*.png")])
        if path:
            self.image_path = Path(path)
            logging.info(f"Loaded input image: {path}")
            self.pred_sequences.clear()
            self.gt_sequence = None
            self._clear_display()

    def load_ground_truth(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("GIF", "*.gif")])
        if not path:
            return
        try:
            frames: List[Image.Image] = []
            with Image.open(path) as gif:
                for frm in ImageSequence.Iterator(gif):
                    frm = frm.convert("RGB").resize((CONTAINER_SIZE, CONTAINER_SIZE), Image.NEAREST)
                    frames.append(frm.copy())
            frames = frames[1:]  # drop frame 0
            while len(frames) < MAX_FRAMES:
                frames.append(frames[-1].copy())
            self.gt_sequence = frames[:MAX_FRAMES]
            logging.info(f"Ground-truth loaded: {len(frames)} frames (frame_1–14 aligned)")
            self._render_all()
        except Exception as e:
            messagebox.showerror("Ground‑Truth Error", f"Cannot load GIF: {e}")

    def run_inference(self) -> None:
        self.scrub_active = False
        if self.image_path is None:
            messagebox.showwarning("Input Missing", "Please load an image first.")
            return
        selected = [m for m, v in self.selected_models.items() if v]
        if not selected:
            messagebox.showwarning("Model Selection", "Select at least one model.")
            return

        def _worker(model_name: str):
            try:
                if model_name not in self.model_instances:
                    cls = getattr(models_regression_sequence, model_name)
                    state = torch.load(self.model_files[model_name], map_location="cpu")
                    net = cls()
                    net.load_state_dict(state)
                    net.eval()
                    self.model_instances[model_name] = net
                net = self.model_instances[model_name]

                preprocess = transforms.Compose([
                    transforms.Resize((IMG_SIZE, IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
                ])
                img = Image.open(self.image_path).convert("RGB")
                tensor = preprocess(img).unsqueeze(0)
                with torch.no_grad():
                    out = net(tensor)
                    if isinstance(out, tuple):
                        out = out[0]
                    if out.dim() == 5 and out.size(0) == 1:
                        out = out.squeeze(0)
                self.pred_sequences[model_name] = out
                logging.info(f"Inference done: {model_name}")
            except Exception as exc:
                messagebox.showerror("Inference Error", f"{model_name}: {exc}")

        threads: List[threading.Thread] = []
        for m in selected:
            t = threading.Thread(target=_worker, args=(m,))
            t.start()
            threads.append(t)
        self.after(100, lambda: self._wait_threads_then_render(threads))

    def _wait_threads_then_render(self, threads: List[threading.Thread]) -> None:
        if any(t.is_alive() for t in threads):
            self.after(100, lambda: self._wait_threads_then_render(threads))
        else:
            self._clear_display()
            self._render_all()

    def _clear_display(self) -> None:
        for child in self.canvas_container.winfo_children():
            child.destroy()

    def _render_all(self) -> None:
        self._clear_display()
        col, row = 0, 0

        if self.gt_sequence:
            frames = self.gt_sequence[int(self.slider.get()):]
            frames = frames[:MAX_FRAMES] + [frames[-1]] * max(0, MAX_FRAMES - len(frames))
            self._add_gif_container("Ground Truth", frames, row, col)
            col += 1

        for name in sorted(self.selected_models.keys()):
            if not self.selected_models[name] or name not in self.pred_sequences:
                continue
            if col >= 3:
                row += 1
                col = 0
            frames = self._tensor_to_frames(self.pred_sequences[name])
            self._add_gif_container(name, frames, row, col)
            col += 1

    def _add_gif_container(self, title: str, frames: List[Image.Image], r: int, c: int) -> None:
        container = tk.Frame(self.canvas_container, bd=1, relief="solid",
                             padx=5, pady=5, bg=CARD_BG, highlightbackground=BORDER_CLR, highlightthickness=1)
        container.grid(row=r, column=c, padx=10, pady=10)
        tk.Label(container, text=title, font=("Segoe UI", 10, "bold"), bg=CARD_BG, fg=FG).pack()
        lbl = tk.Label(container, bg=CARD_BG)
        lbl.pack()
        tk.Label(container, text=f"{len(frames)} frames", bg=CARD_BG, fg=FG).pack()
        lbl.frames = [ImageTk.PhotoImage(f) for f in frames]
        lbl.idx = 0

        def _anim(label=lbl):
            if not hasattr(label, "frames"):
                return
            if not str(label) in label.tk.call("winfo", "children", label.master):
                return  # <- safeguard: skip if label is destroyed
            try:
                label.configure(image=label.frames[label.idx])
                if not self.scrub_active:
                    label.idx = (label.idx + 1) % len(label.frames)
                    self.after(int(1000 / DISPLAY_FPS), _anim)
            except tk.TclError:
                pass  # <- label may have been destroyed before next frame

        _anim()

    def _tensor_to_frames(self, seq: torch.Tensor) -> List[Image.Image]:
        unnorm = transforms.Normalize(mean=[-1, -1, -1], std=[2, 2, 2])
        frames = []
        start = int(self.slider.get())
        end = min(seq.size(0), start + MAX_FRAMES)
        for i in range(start, end):
            t = seq[i].squeeze()
            t = unnorm(t).clamp(0, 1)
            img = transforms.ToPILImage()(t)
            img = img.resize((CONTAINER_SIZE, CONTAINER_SIZE), Image.NEAREST)
            frames.append(img)
        if len(frames) == 1:
            frames.append(frames[0])
        return frames

    def update_display(self, _val) -> None:
        self.scrub_active = True
        if self.pred_sequences or self.gt_sequence:
            self._render_all()


if __name__ == "__main__":
    root = tk.Tk()
    root.tk_setPalette(background=BG, foreground=FG, activeBackground=BTN_BG, activeForeground=FG)
    app = InferenceApp(root)
    root.mainloop()
