import tkinter as tk

import customtkinter as ctk
from PIL import Image, ImageTk


class ReannotateDialog(ctk.CTkToplevel):
    def __init__(self, master, frame_rgb, request_meta, on_submit, on_bridge, on_abort):
        super().__init__(master)
        self.title("Target Review")
        self.geometry("980x760")
        self.minsize(860, 680)
        self.configure(fg_color="#101010")
        self.transient(master)
        self.grab_set()

        self.on_submit = on_submit
        self.on_bridge = on_bridge
        self.on_abort = on_abort
        self.request_id = request_meta["request_id"]
        self.original_image = Image.fromarray(frame_rgb)
        self.image_width, self.image_height = self.original_image.size
        self.current_point = tuple(request_meta.get("current_point") or ()) or None
        self.candidate_points = [tuple(point) for point in request_meta.get("candidate_points", [])]
        self.selected_point = self.current_point or (self.candidate_points[0] if self.candidate_points else None)
        self.display_size = self._compute_display_size()
        self.scale_x = self.display_size[0] / self.image_width
        self.scale_y = self.display_size[1] / self.image_height

        self._build_ui(request_meta)
        self.protocol("WM_DELETE_WINDOW", self._bridge)
        self.focus_force()

    def _compute_display_size(self):
        max_width = 900
        max_height = 560
        width, height = self.original_image.size
        scale = min(max_width / width, max_height / height, 1.0)
        return (int(width * scale), int(height * scale))

    def _build_ui(self, request_meta):
        title = ctk.CTkLabel(
            self,
            text="Target Review Required",
            font=("Consolas", 20, "bold"),
            text_color="#00ffff",
        )
        title.pack(pady=(14, 6))

        subtitle = ctk.CTkLabel(
            self,
            text=(
                "Tracking confidence dropped. Pick a new target point, choose a suggested point, "
                "or keep the previous stroke bridge."
            ),
            font=("Consolas", 11),
            text_color="#d0d0d0",
        )
        subtitle.pack(pady=(0, 8))

        info = ctk.CTkLabel(
            self,
            text=f"Time: {request_meta.get('at_ms', 0)} ms | Request: {self.request_id}",
            font=("Consolas", 11),
            text_color="#a0ffa0",
        )
        info.pack(pady=(0, 10))

        image_frame = ctk.CTkFrame(self, fg_color="#141414")
        image_frame.pack(fill="both", expand=True, padx=18, pady=8)

        self.canvas = tk.Canvas(
            image_frame,
            width=self.display_size[0],
            height=self.display_size[1],
            bg="#000000",
            highlightthickness=0,
        )
        self.canvas.pack(padx=12, pady=12)
        self.canvas.bind("<Button-1>", self._on_canvas_click)

        resized = self.original_image.resize(self.display_size, Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.canvas.create_image(0, 0, anchor="nw", image=self.photo)

        controls = ctk.CTkFrame(self, fg_color="#101010")
        controls.pack(fill="x", padx=18, pady=(0, 10))

        candidate_frame = ctk.CTkFrame(controls, fg_color="#151515")
        candidate_frame.pack(fill="x", pady=(0, 10))

        ctk.CTkLabel(
            candidate_frame,
            text="Suggested Points",
            font=("Consolas", 12, "bold"),
            text_color="#ffd166",
        ).pack(anchor="w", padx=10, pady=(8, 4))

        if self.candidate_points:
            for index, point in enumerate(self.candidate_points[:6], start=1):
                button = ctk.CTkButton(
                    candidate_frame,
                    text=f"Candidate {index}: ({point[0]}, {point[1]})",
                    command=lambda p=point: self._select_point(p),
                    fg_color="#223344",
                    hover_color="#335577",
                    font=("Consolas", 11),
                )
                button.pack(fill="x", padx=10, pady=4)
        else:
            ctk.CTkLabel(
                candidate_frame,
                text="No automatic candidates were available. Click the image to choose a point.",
                font=("Consolas", 11),
                text_color="#cccccc",
            ).pack(anchor="w", padx=10, pady=(4, 10))

        self.selected_label = ctk.CTkLabel(
            controls,
            text=self._selected_text(),
            font=("Consolas", 12),
            text_color="#ffffff",
        )
        self.selected_label.pack(anchor="w", pady=(0, 10))

        action_frame = ctk.CTkFrame(controls, fg_color="#101010")
        action_frame.pack(fill="x")

        submit_btn = ctk.CTkButton(
            action_frame,
            text="Resume With Selected Point",
            command=self._submit,
            fg_color="#008866",
            hover_color="#00aa7f",
            font=("Consolas", 12, "bold"),
        )
        submit_btn.pack(side="left", padx=(0, 10))

        bridge_btn = ctk.CTkButton(
            action_frame,
            text="Keep Previous Flow",
            command=self._bridge,
            fg_color="#7a5c00",
            hover_color="#9a7400",
            font=("Consolas", 12, "bold"),
        )
        bridge_btn.pack(side="left", padx=(0, 10))

        abort_btn = ctk.CTkButton(
            action_frame,
            text="Abort Video",
            command=self._abort,
            fg_color="#883333",
            hover_color="#aa4444",
            font=("Consolas", 12, "bold"),
        )
        abort_btn.pack(side="left")

        self._redraw_overlay()

    def _selected_text(self):
        if self.selected_point is None:
            return "Selected Point: none"
        return f"Selected Point: ({int(self.selected_point[0])}, {int(self.selected_point[1])})"

    def _to_canvas(self, point):
        return (point[0] * self.scale_x, point[1] * self.scale_y)

    def _draw_cross(self, point, color, radius=8, width=2):
        x, y = self._to_canvas(point)
        self.canvas.create_line(x - radius, y, x + radius, y, fill=color, width=width, tags="overlay")
        self.canvas.create_line(x, y - radius, x, y + radius, fill=color, width=width, tags="overlay")

    def _redraw_overlay(self):
        self.canvas.delete("overlay")

        for index, point in enumerate(self.candidate_points[:6], start=1):
            x, y = self._to_canvas(point)
            self.canvas.create_oval(x - 7, y - 7, x + 7, y + 7, outline="#ffd166", width=2, tags="overlay")
            self.canvas.create_text(x + 12, y - 10, text=str(index), fill="#ffd166", font=("Consolas", 10, "bold"), tags="overlay")

        if self.current_point is not None:
            self._draw_cross(self.current_point, "#00ffff", radius=10, width=2)

        if self.selected_point is not None:
            self._draw_cross(self.selected_point, "#ff00ff", radius=12, width=3)

    def _select_point(self, point):
        self.selected_point = (int(point[0]), int(point[1]))
        self.selected_label.configure(text=self._selected_text())
        self._redraw_overlay()

    def _on_canvas_click(self, event):
        x = int(event.x / self.scale_x)
        y = int(event.y / self.scale_y)
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))
        self._select_point((x, y))

    def _submit(self):
        if self.selected_point is None:
            return
        self.on_submit(self.request_id, self.selected_point)
        self.destroy()

    def _bridge(self):
        self.on_bridge(self.request_id)
        self.destroy()

    def _abort(self):
        self.on_abort(self.request_id)
        self.destroy()
