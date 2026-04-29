import tkinter as tk
from threading import Thread

import customtkinter as ctk
from PIL import Image, ImageTk


class TargetReviewDialog(ctk.CTkToplevel):
    def __init__(self, master, frame_rgb, review_meta, on_validate=None):
        super().__init__(master)
        self.title("Initial Target Review")
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        window_w = min(max(1080, screen_w - 70), 1280)
        window_h = min(max(860, screen_h - 70), 980)
        self.geometry(f"{window_w}x{window_h}")
        self.minsize(1040, 820)
        self.configure(fg_color="#090b0f")
        self.transient(master)
        self.grab_set()

        self.review_meta = review_meta
        self.on_validate = on_validate
        self.original_image = Image.fromarray(frame_rgb)
        self.image_width, self.image_height = self.original_image.size
        self.auto_point = tuple(review_meta.get("auto_point") or ())
        self.user_seed_point = tuple(review_meta.get("user_seed_point") or ()) or None
        self.candidate_points = [tuple(point) for point in review_meta.get("candidate_points", [])]
        self.selected_point = self.user_seed_point or self.auto_point or (self.candidate_points[0] if self.candidate_points else None)
        self.validation_point = None
        self.validation_result = None
        self.validation_running = False
        self.validation_close_requested = False
        self.result = {"action": "cancel", "point": None}
        self.max_image_width = min(820, int(window_w * 0.66))
        self.max_image_height = max(420, min(620, window_h - 190))
        self.display_size = self._compute_display_size()
        self.scale_x = self.display_size[0] / self.image_width
        self.scale_y = self.display_size[1] / self.image_height

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._cancel)
        self.focus_force()
        self.after(120, self._auto_validate_initial)

    def _compute_display_size(self):
        max_width = self.max_image_width
        max_height = self.max_image_height
        width, height = self.original_image.size
        scale = min(max_width / width, max_height / height, 1.0)
        return (int(width * scale), int(height * scale))

    def _build_ui(self):
        shell = ctk.CTkFrame(self, fg_color="#090b0f")
        shell.pack(fill="both", expand=True, padx=18, pady=16)

        title = ctk.CTkLabel(
            shell,
            text="Pick The Initial Tracking Target",
            font=("Consolas", 20, "bold"),
            text_color="#edf3f8",
        )
        title.pack(anchor="w", pady=(0, 4))

        subtitle = ctk.CTkLabel(
            shell,
            text=(
                "Select the pelvis, groin, or the most reliable stroke proxy before analysis starts. "
                "You can choose a suggestion or click directly on the frame."
            ),
            font=("Consolas", 11),
            text_color="#8fa0af",
        )
        subtitle.pack(anchor="w", pady=(0, 8))

        info = ctk.CTkLabel(
            shell,
            text=(
                f"Frame: {self.review_meta.get('frame_index', 0)} | "
                f"Heatmap Peak: {self.review_meta.get('heatmap_peak', 0.0):.2f}"
            ),
            font=("Consolas", 11),
            text_color="#42d9c8",
        )
        info.pack(anchor="w", pady=(0, 10))

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

        candidate_frame = ctk.CTkFrame(controls, fg_color="#151d28", corner_radius=6)
        candidate_frame.pack(fill="x", padx=10, pady=(10, 8))

        ctk.CTkLabel(
            candidate_frame,
            text="Suggested Starting Points",
            font=("Consolas", 12, "bold"),
            text_color="#ffb84d",
        ).pack(anchor="w", padx=10, pady=(8, 4))

        if self.candidate_points:
            grid = ctk.CTkFrame(candidate_frame, fg_color="#151d28")
            grid.pack(fill="x", padx=8, pady=(0, 8))
            for index, point in enumerate(self.candidate_points[:6], start=1):
                button = ctk.CTkButton(
                    grid,
                    text=f"{index}. ({point[0]}, {point[1]})",
                    command=lambda p=point: self._select_point(p),
                    fg_color="#1b2635",
                    hover_color="#2c3a4d",
                    font=("Consolas", 10),
                    height=30,
                    corner_radius=5,
                )
                button.grid(row=(index - 1) // 2, column=(index - 1) % 2, sticky="ew", padx=4, pady=4)
            grid.grid_columnconfigure(0, weight=1)
            grid.grid_columnconfigure(1, weight=1)
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
        self.selected_label.pack(anchor="w", padx=12, pady=(0, 8))

        self.validation_label = ctk.CTkLabel(
            controls,
            text="Validation: not run",
            font=("Consolas", 12),
            text_color="#bbbbbb",
        )
        self.validation_label.pack(anchor="w", padx=12, pady=(0, 4))

        self.validation_text = ctk.CTkTextbox(
            controls,
            height=104,
            fg_color="#151d28",
            border_width=1,
            border_color="#2c3a4d",
            font=("Consolas", 10),
            activate_scrollbars=True,
        )
        self.validation_text.pack(fill="x", padx=10, pady=(0, 8))
        self.validation_text.insert("1.0", "Validation has not been run yet.\n")
        self.validation_text.configure(state="disabled")

        hint_label = ctk.CTkLabel(
            controls,
            text="Cyan = auto proposal, magenta = selected point, green = previously chosen seed",
            font=("Consolas", 10),
            text_color="#bbbbbb",
            wraplength=320,
            justify="left",
        )
        hint_label.pack(anchor="w", padx=12, pady=(0, 8))

        action_frame = ctk.CTkFrame(controls, fg_color="#111720")
        action_frame.pack(fill="x", side="bottom", padx=10, pady=(0, 10))

        self.submit_btn = ctk.CTkButton(
            action_frame,
            text="Start Selected",
            command=self._submit_selected,
            fg_color="#008866",
            hover_color="#00aa7f",
            font=("Consolas", 12, "bold"),
            height=34,
            corner_radius=5,
        )
        self.submit_btn.grid(row=0, column=0, sticky="ew", padx=(0, 5), pady=(0, 6))

        self.auto_btn = ctk.CTkButton(
            action_frame,
            text="Use Auto",
            command=self._use_auto_point,
            fg_color="#005f88",
            hover_color="#0077aa",
            font=("Consolas", 12, "bold"),
            height=34,
            corner_radius=5,
        )
        self.auto_btn.grid(row=0, column=1, sticky="ew", padx=(5, 0), pady=(0, 6))

        self.validate_btn = ctk.CTkButton(
            action_frame,
            text="Validate",
            command=self._start_validation,
            fg_color="#2c3a4d",
            hover_color="#3a4d63",
            font=("Consolas", 12, "bold"),
            height=34,
            corner_radius=5,
        )
        self.validate_btn.grid(row=1, column=0, sticky="ew", padx=(0, 5))

        self.cancel_btn = ctk.CTkButton(
            action_frame,
            text="Cancel",
            command=self._cancel,
            fg_color="#883333",
            hover_color="#aa4444",
            font=("Consolas", 12, "bold"),
            height=34,
            corner_radius=5,
        )
        self.cancel_btn.grid(row=1, column=1, sticky="ew", padx=(5, 0))

        self.validation_progress = ctk.CTkProgressBar(
            action_frame,
            height=8,
            progress_color="#42d9c8",
            fg_color="#151d28",
        )
        self.validation_progress.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.validation_progress.set(0.0)
        action_frame.grid_columnconfigure(0, weight=1)
        action_frame.grid_columnconfigure(1, weight=1)

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

        if self.auto_point:
            self._draw_cross(self.auto_point, "#00ffff", radius=10, width=2)

        if self.user_seed_point is not None:
            self._draw_cross(self.user_seed_point, "#66ff66", radius=10, width=2)

        if self.selected_point is not None:
            self._draw_cross(self.selected_point, "#ff00ff", radius=12, width=3)

    def _select_point(self, point):
        self.selected_point = (int(point[0]), int(point[1]))
        self.selected_label.configure(text=self._selected_text())
        if self.validation_point != self.selected_point:
            self.validation_result = None
            self.validation_point = None
            self._set_validation_text("Validation: selection changed", "Run validation to check if this point stays trackable.\n")
        self._redraw_overlay()

    def _on_canvas_click(self, event):
        x = int(event.x / self.scale_x)
        y = int(event.y / self.scale_y)
        x = max(0, min(self.image_width - 1, x))
        y = max(0, min(self.image_height - 1, y))
        self._select_point((x, y))

    def _submit_selected(self):
        if self.validation_running:
            self._set_validation_text(
                "Validation: running",
                "Please wait for validation to finish before starting analysis.\n",
                color="#ffd166",
            )
            return
        if self.selected_point is None:
            return
        self.result = {
            "action": "submit",
            "point": self.selected_point,
            "validation": self.validation_result,
        }
        self.destroy()

    def _use_auto_point(self):
        if self.validation_running:
            self._set_validation_text(
                "Validation: running",
                "Please wait for validation to finish before starting analysis.\n",
                color="#ffd166",
            )
            return
        point = self.auto_point or self.selected_point
        if point is None:
            return
        self.result = {
            "action": "auto",
            "point": point,
            "validation": self.validation_result if self.validation_point == point else None,
        }
        self.destroy()

    def _cancel(self):
        if self.validation_running:
            self.validation_close_requested = True
            self._set_validation_text(
                "Validation: running",
                "Validation is still running. The dialog will stay open until it finishes safely.\n",
                color="#ffd166",
            )
            return
        self.result = {"action": "cancel", "point": None}
        self.destroy()

    def _auto_validate_initial(self):
        if self.selected_point is not None and self.on_validate is not None:
            self._start_validation()

    def _set_validation_text(self, label_text, body_text, color="#bbbbbb"):
        self.validation_label.configure(text=label_text, text_color=color)
        self.validation_text.configure(state="normal")
        self.validation_text.delete("1.0", "end")
        self.validation_text.insert("1.0", body_text)
        self.validation_text.configure(state="disabled")

    def _start_validation(self):
        if self.validation_running or self.selected_point is None or self.on_validate is None:
            return

        self.validation_running = True
        self.validation_close_requested = False
        self._set_action_buttons_state("disabled")
        self.validate_btn.configure(state="disabled", text="Validating...")
        self.validation_progress.configure(mode="indeterminate")
        self.validation_progress.start()
        self._set_validation_text(
            "Validation: running",
            "Checking the selected point across sampled motion windows...\n",
            color="#ffd166",
        )

        point = self.selected_point

        def _worker():
            try:
                result = self.on_validate(point)
                self._safe_finish_validation(point, result, None)
            except Exception as exc:
                self._safe_finish_validation(point, None, str(exc))

        Thread(target=_worker, daemon=True).start()

    def _set_action_buttons_state(self, state):
        for button in (self.submit_btn, self.auto_btn, self.cancel_btn):
            button.configure(state=state)

    def _safe_finish_validation(self, point, result, error):
        try:
            if self.winfo_exists():
                self.after(0, lambda: self._finish_validation(point, result, error))
        except tk.TclError:
            pass

    def _finish_validation(self, point, result, error):
        if not self.winfo_exists():
            return
        self.validation_running = False
        self.validation_progress.stop()
        self.validation_progress.configure(mode="determinate")
        self.validation_progress.set(1.0 if error is None else 0.0)
        self._set_action_buttons_state("normal")
        self.validate_btn.configure(state="normal", text="Validate Selected Point")

        if error is not None:
            self._set_validation_text(
                "Validation: failed",
                f"{error}\n",
                color="#ff8888",
            )
            if self.validation_close_requested:
                self.result = {"action": "cancel", "point": None}
                self.destroy()
            return

        self.validation_point = point
        self.validation_result = result

        level = result.level
        if level == "HIGH":
            color = "#66ff99"
        elif level == "MEDIUM":
            color = "#ffd166"
        else:
            color = "#ff8888"

        lines = [
            f"Score: {result.score:.2f} ({level})",
            f"Survivability: {result.survivability_ratio * 100:.1f}%",
            f"Mean features: {result.mean_feature_count:.1f}",
            f"Retention: {result.mean_retention_ratio * 100:.1f}%",
            f"Motion span: {result.motion_span_px:.1f}px",
            f"Reacquire count: {result.reacquire_count}",
            f"Lost frames: {result.lost_frame_count}",
            "",
            "Sample checks:",
        ]

        for sample in result.sample_summaries:
            lines.append(
                f"F{sample['frame']}: {sample['status']} | feat={sample['feature_count']} "
                f"| keep={sample['retention_ratio']:.2f} | pt=({sample['point'][0]}, {sample['point'][1]})"
            )

        self._set_validation_text(
            f"Validation: {level}",
            "\n".join(lines) + "\n",
            color=color,
        )
        if self.validation_close_requested:
            self.result = {"action": "cancel", "point": None}
            self.destroy()
