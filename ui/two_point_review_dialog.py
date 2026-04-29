import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk


class TwoPointReviewDialog(ctk.CTkToplevel):
    def __init__(self, master, frame_rgb, review_meta):
        super().__init__(master)
        self.title("Two-Point Target Review")
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        window_w = min(max(1080, screen_w - 70), 1280)
        window_h = min(max(820, screen_h - 80), 940)
        self.geometry(f"{window_w}x{window_h}")
        self.minsize(1040, 780)
        self.configure(fg_color="#090b0f")
        self.transient(master)
        self.grab_set()

        self.review_meta = review_meta
        self.original_image = Image.fromarray(frame_rgb)
        self.image_width, self.image_height = self.original_image.size
        self.primary_point = None
        self.reference_point = None
        self.active_slot = "primary"
        self.result = {"action": "cancel", "points": None}

        self.max_image_width = min(840, int(window_w * 0.68))
        self.max_image_height = max(430, min(650, window_h - 170))
        self.display_size = self._compute_display_size()
        self.scale_x = self.display_size[0] / self.image_width
        self.scale_y = self.display_size[1] / self.image_height

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.focus_force()

    def _compute_display_size(self):
        width, height = self.original_image.size
        scale = min(self.max_image_width / width, self.max_image_height / height, 1.0)
        return (int(width * scale), int(height * scale))

    def _build_ui(self):
        shell = ctk.CTkFrame(self, fg_color="#090b0f")
        shell.pack(fill="both", expand=True, padx=18, pady=16)

        ctk.CTkLabel(
            shell,
            text="Pick Two Tracking Points",
            font=("Consolas", 20, "bold"),
            text_color="#edf3f8",
        ).pack(anchor="w", pady=(0, 4))

        ctk.CTkLabel(
            shell,
            text="Primary = first stroke endpoint. Reference = second stroke endpoint. Pick the two body/contact points whose distance should drive the stroke.",
            font=("Consolas", 11),
            text_color="#8fa0af",
        ).pack(anchor="w", pady=(0, 10))

        content = ctk.CTkFrame(shell, fg_color="#090b0f")
        content.pack(fill="both", expand=True)

        image_frame = ctk.CTkFrame(content, fg_color="#111720", corner_radius=8)
        image_frame.pack(side="left", fill="both", expand=True, padx=(0, 12))

        self.canvas = tk.Canvas(
            image_frame,
            width=self.display_size[0],
            height=self.display_size[1],
            bg="#000000",
            highlightthickness=0,
        )
        self.canvas.pack(expand=True, padx=12, pady=12)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        resized = self.original_image.resize(self.display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        controls = ctk.CTkFrame(content, fg_color="#111720", corner_radius=8, width=360)
        controls.pack(side="right", fill="y")
        controls.pack_propagate(False)

        self.primary_btn = ctk.CTkButton(
            controls,
            text="SET PRIMARY",
            command=lambda: self._set_active_slot("primary"),
            fg_color="#b8831f",
            hover_color="#d99a28",
            font=("Consolas", 12, "bold"),
            height=34,
        )
        self.primary_btn.pack(fill="x", padx=12, pady=(12, 8))

        self.reference_btn = ctk.CTkButton(
            controls,
            text="SET REFERENCE",
            command=lambda: self._set_active_slot("reference"),
            fg_color="#1f6fb8",
            hover_color="#2687d9",
            font=("Consolas", 12, "bold"),
            height=34,
        )
        self.reference_btn.pack(fill="x", padx=12, pady=(0, 10))

        self.info_label = ctk.CTkLabel(
            controls,
            text="",
            font=("Consolas", 11),
            text_color="#edf3f8",
            justify="left",
            wraplength=320,
        )
        self.info_label.pack(anchor="w", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            controls,
            text=(
                "How to choose points:\n"
                "1. Primary: pelvis/groin/contact point of one target.\n"
                "2. Reference: pelvis/groin/contact point of the other target.\n"
                "The generated stroke is based on the distance between these two points.\n\n"
                "Automatic guesses are disabled for now. Manual two-point selection is safer than a wrong automatic start."
            ),
            font=("Consolas", 10),
            text_color="#8fa0af",
            justify="left",
            wraplength=320,
        ).pack(anchor="w", padx=12, pady=(0, 12))

        action_frame = ctk.CTkFrame(controls, fg_color="#111720")
        action_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))

        ctk.CTkButton(
            action_frame,
            text="START TWO-POINT",
            command=self._submit,
            fg_color="#008866",
            hover_color="#00aa7f",
            font=("Consolas", 12, "bold"),
            height=36,
        ).pack(fill="x", pady=(0, 8))

        ctk.CTkButton(
            action_frame,
            text="CANCEL",
            command=self._cancel,
            fg_color="#883333",
            hover_color="#aa4444",
            font=("Consolas", 12, "bold"),
            height=34,
        ).pack(fill="x")

        self._update_info()
        self._redraw_overlay()

    def _set_active_slot(self, slot):
        self.active_slot = slot
        self._update_info()

    def _to_canvas(self, point):
        return (point[0] * self.scale_x, point[1] * self.scale_y)

    def _draw_cross(self, point, color, label):
        x, y = self._to_canvas(point)
        self.canvas.create_oval(x - 8, y - 8, x + 8, y + 8, outline=color, width=3, tags="overlay")
        self.canvas.create_line(x - 14, y, x + 14, y, fill=color, width=2, tags="overlay")
        self.canvas.create_line(x, y - 14, x, y + 14, fill=color, width=2, tags="overlay")
        self.canvas.create_text(x + 16, y - 14, text=label, fill=color, font=("Consolas", 11, "bold"), tags="overlay")

    def _redraw_overlay(self):
        self.canvas.delete("overlay")
        if self.primary_point is not None and self.reference_point is not None:
            px, py = self._to_canvas(self.primary_point)
            rx, ry = self._to_canvas(self.reference_point)
            self.canvas.create_line(px, py, rx, ry, fill="#ffffff", width=1, dash=(4, 4), tags="overlay")
        if self.primary_point is not None:
            self._draw_cross(self.primary_point, "#ffb84d", "PRIMARY")
        if self.reference_point is not None:
            self._draw_cross(self.reference_point, "#42d9c8", "REFERENCE")

    def _update_info(self):
        primary = "none" if self.primary_point is None else f"({self.primary_point[0]}, {self.primary_point[1]})"
        reference = "none" if self.reference_point is None else f"({self.reference_point[0]}, {self.reference_point[1]})"
        self.info_label.configure(text=f"Active slot: {self.active_slot.upper()}\nPrimary: {primary}\nReference: {reference}")

    def _on_canvas_click(self, event):
        x = int(event.x / self.scale_x)
        y = int(event.y / self.scale_y)
        point = (
            max(0, min(self.image_width - 1, x)),
            max(0, min(self.image_height - 1, y)),
        )
        if self.active_slot == "primary":
            self.primary_point = point
            self.active_slot = "reference"
        else:
            self.reference_point = point
            self.active_slot = "primary"
        self._update_info()
        self._redraw_overlay()

    def _submit(self):
        if self.primary_point is None or self.reference_point is None:
            return
        self.result = {
            "action": "submit",
            "points": {
                "primary": self.primary_point,
                "reference": self.reference_point,
            },
        }
        self.destroy()

    def _cancel(self):
        self.result = {"action": "cancel", "points": None}
        self.destroy()
