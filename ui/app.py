import json
import os
import threading
from collections import Counter

import customtkinter as ctk
import cv2
from PIL import Image, ImageTk
from tkinter import filedialog, simpledialog
from tkinterdnd2 import DND_FILES, TkinterDnD

from core_ai.processor import VideoProcessor
from monitoring.resource_monitor import ResourceMonitorFrame
from tracking import TargetPointProposer
from ui.reannotate_dialog import ReannotateDialog
from ui.target_review_dialog import TargetReviewDialog

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


PALETTE = {
    "bg": "#090b0f",
    "panel": "#111720",
    "panel_soft": "#151d28",
    "panel_lift": "#1b2635",
    "line": "#2c3a4d",
    "text": "#edf3f8",
    "muted": "#8fa0af",
    "accent": "#42d9c8",
    "accent_2": "#ffb84d",
    "danger": "#ff5c7a",
    "ok": "#5ee38a",
}


class App(ctk.CTk, TkinterDnD.DnDWrapper):
    def __init__(self):
        super().__init__()

        venv_base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".venv"))
        tkdnd_path = os.path.join(venv_base, "Lib", "site-packages", "tkinterdnd2", "tkdnd", "win-x64")
        self.tk.call("lappend", "auto_path", tkdnd_path)

        try:
            self.TkdndVersion = self.tk.call("package", "require", "tkdnd")
        except Exception as e:
            print(f"Warning: Could not load tkdnd: {e}")
            self.TkdndVersion = None

        self.title("VIDEO-TO-FUNSCRIPT V2")
        self.geometry("1280x820")
        self.minsize(1060, 720)
        self.configure(fg_color=PALETTE["bg"])

        self.sidebar = ctk.CTkFrame(self, width=300, corner_radius=0, fg_color=PALETTE["panel"])
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        self.brand_label = ctk.CTkLabel(
            self.sidebar,
            text="EROSCRIPT MAKER",
            font=("Consolas", 20, "bold"),
            text_color=PALETTE["text"],
        )
        self.brand_label.pack(anchor="w", padx=18, pady=(18, 2))

        self.brand_subtitle = ctk.CTkLabel(
            self.sidebar,
            text="video analysis queue",
            font=("Consolas", 11),
            text_color=PALETTE["muted"],
        )
        self.brand_subtitle.pack(anchor="w", padx=18, pady=(0, 18))

        self.queue_label = ctk.CTkLabel(
            self.sidebar,
            text="PROCESSING QUEUE",
            font=("Consolas", 14, "bold"),
            text_color=PALETTE["accent"],
        )
        self.queue_label.pack(anchor="w", padx=18, pady=(0, 8))

        self.queue_list = ctk.CTkScrollableFrame(
            self.sidebar,
            width=260,
            height=560,
            fg_color=PALETTE["panel_soft"],
            border_width=1,
            border_color=PALETTE["line"],
        )
        self.queue_list.pack(fill="both", expand=True, pady=(0, 12), padx=14)
        self.queue_list.bind("<Button-1>", self.open_video_file_dialog)

        self.queue_hint = ctk.CTkLabel(
            self.sidebar,
            text="Drop videos here or click the queue to browse. Each file gets its own target review.",
            font=("Consolas", 10),
            text_color=PALETTE["muted"],
            wraplength=260,
            justify="left",
        )
        self.queue_hint.pack(anchor="w", padx=18, pady=(0, 16))

        self.main_frame = ctk.CTkFrame(self, corner_radius=0, fg_color=PALETTE["bg"])
        self.main_frame.pack(side="left", fill="both", expand=True, padx=14, pady=14)

        self.preview_frame = ctk.CTkFrame(self.main_frame, height=410, fg_color=PALETTE["panel"], corner_radius=8)
        self.preview_frame.pack(fill="both", expand=True, pady=(0, 10))
        self.preview_canvas = ctk.CTkCanvas(self.preview_frame, bg="#05070a", highlightthickness=0)
        self.preview_canvas.pack(fill="both", expand=True)
        self.preview_canvas.unbind("<Button-1>")

        self.result_frame = ctk.CTkFrame(self.main_frame, height=214, fg_color=PALETTE["panel"], corner_radius=8)
        self.result_frame.pack(fill="x", pady=(0, 10))
        self.result_frame.pack_propagate(False)

        self.result_header = ctk.CTkFrame(self.result_frame, fg_color=PALETTE["panel"], height=28)
        self.result_header.pack(fill="x", padx=10, pady=(8, 0))
        self.result_title = ctk.CTkLabel(
            self.result_header,
            text="FINAL FUNSCRIPT",
            font=("Consolas", 12, "bold"),
            text_color=PALETTE["text"],
        )
        self.result_title.pack(side="left")
        self.result_meta_label = ctk.CTkLabel(
            self.result_header,
            text="waiting for completed analysis",
            font=("Consolas", 10),
            text_color=PALETTE["muted"],
        )
        self.result_meta_label.pack(side="right")

        self.result_canvas = ctk.CTkCanvas(self.result_frame, height=86, bg=PALETTE["panel_soft"], highlightthickness=0)
        self.result_canvas.pack(fill="both", expand=True, padx=10, pady=(4, 6))
        self.result_canvas.bind("<Configure>", lambda _event: self._draw_result_graph())

        self.result_controls = ctk.CTkFrame(self.result_frame, fg_color=PALETTE["panel"], height=30)
        self.result_controls.pack(fill="x", padx=10, pady=(0, 8))
        self.snap_peaks_btn = ctk.CTkButton(
            self.result_controls,
            text="Snap Peaks",
            command=self._snap_result_peaks,
            fg_color=PALETTE["panel_lift"],
            hover_color="#2c3a4d",
            font=("Consolas", 10, "bold"),
            height=26,
            corner_radius=5,
        )
        self.snap_peaks_btn.pack(side="left", padx=(0, 6))
        self.snap_all_btn = ctk.CTkButton(
            self.result_controls,
            text="Snap All",
            command=self._snap_result_all_values,
            fg_color=PALETTE["panel_lift"],
            hover_color="#2c3a4d",
            font=("Consolas", 10, "bold"),
            height=26,
            width=78,
            corner_radius=5,
        )
        self.snap_all_btn.pack(side="left", padx=(0, 6))
        self.reaction_btn = ctk.CTkButton(
            self.result_controls,
            text="React",
            command=self._preview_reaction_bounces,
            fg_color=PALETTE["panel_lift"],
            hover_color="#2c3a4d",
            font=("Consolas", 10, "bold"),
            height=26,
            width=68,
            corner_radius=5,
        )
        self.reaction_btn.pack(side="left", padx=(0, 6))
        self.normalize_range_btn = ctk.CTkButton(
            self.result_controls,
            text="Normalize 0-100",
            command=self._normalize_result_range,
            fg_color=PALETTE["panel_lift"],
            hover_color="#2c3a4d",
            font=("Consolas", 10, "bold"),
            height=26,
            corner_radius=5,
        )
        self.normalize_range_btn.pack(side="left", padx=(0, 6))
        self.custom_range_btn = ctk.CTkButton(
            self.result_controls,
            text="Custom Range",
            command=self._custom_result_range,
            fg_color=PALETTE["panel_lift"],
            hover_color="#2c3a4d",
            font=("Consolas", 10, "bold"),
            height=26,
            corner_radius=5,
        )
        self.custom_range_btn.pack(side="left", padx=(0, 10))
        self.save_result_btn = ctk.CTkButton(
            self.result_controls,
            text="Save",
            command=self._save_result_changes,
            fg_color=PALETTE["ok"],
            hover_color="#49bd70",
            text_color="#071012",
            font=("Consolas", 10, "bold"),
            height=26,
            width=64,
            corner_radius=5,
        )
        self.save_result_btn.pack(side="left", padx=(0, 6))
        self.discard_result_btn = ctk.CTkButton(
            self.result_controls,
            text="Discard",
            command=self._discard_result_changes,
            fg_color=PALETTE["danger"],
            hover_color="#d94d67",
            font=("Consolas", 10, "bold"),
            height=26,
            width=78,
            corner_radius=5,
        )
        self.discard_result_btn.pack(side="left")

        self.reaction_settings = ctk.CTkFrame(self.result_frame, fg_color=PALETTE["panel"], height=34)
        self.reaction_settings.pack(fill="x", padx=10, pady=(0, 8))
        self.reaction_settings.pack_propagate(False)
        self.reaction_label = ctk.CTkLabel(
            self.reaction_settings,
            text="Reaction: strength 20",
            font=("Consolas", 10, "bold"),
            text_color=PALETTE["muted"],
        )
        self.reaction_label.pack(side="left", padx=(0, 8))
        self.reaction_strength_slider = ctk.CTkSlider(
            self.reaction_settings,
            from_=1,
            to=45,
            number_of_steps=44,
            width=130,
            command=self._set_reaction_strength,
            progress_color=PALETTE["accent"],
            button_color=PALETTE["accent"],
            button_hover_color="#35b9ac",
        )
        self.reaction_strength_slider.set(20)
        self.reaction_strength_slider.pack(side="left", padx=(0, 10))
        self.reaction_bounce_selector = ctk.CTkSegmentedButton(
            self.reaction_settings,
            values=["1x", "2x", "3x"],
            command=self._set_reaction_bounces,
            height=24,
            selected_color=PALETTE["accent"],
            selected_hover_color="#35b9ac",
            unselected_color=PALETTE["panel_lift"],
            unselected_hover_color="#2c3a4d",
            text_color=PALETTE["text"],
        )
        self.reaction_bounce_selector.set("2x")
        self.reaction_bounce_selector.pack(side="left", padx=(0, 10))
        self.reaction_density_selector = ctk.CTkSegmentedButton(
            self.reaction_settings,
            values=["Tight", "Normal", "Loose"],
            command=self._set_reaction_density,
            height=24,
            selected_color=PALETTE["accent_2"],
            selected_hover_color="#d79c40",
            unselected_color=PALETTE["panel_lift"],
            unselected_hover_color="#2c3a4d",
            text_color=PALETTE["text"],
        )
        self.reaction_density_selector.set("Normal")
        self.reaction_density_selector.pack(side="left", padx=(0, 10))
        self.reaction_span_selector = ctk.CTkSegmentedButton(
            self.reaction_settings,
            values=["Micro", "Quick", "Short", "Normal"],
            command=self._set_reaction_span,
            height=24,
            selected_color=PALETTE["ok"],
            selected_hover_color="#49bd70",
            unselected_color=PALETTE["panel_lift"],
            unselected_hover_color="#2c3a4d",
            text_color=PALETTE["text"],
        )
        self.reaction_span_selector.set("Short")
        self.reaction_span_selector.pack(side="left")

        self.bottom_frame = ctk.CTkFrame(self.main_frame, height=190, fg_color=PALETTE["panel"], corner_radius=8)
        self.bottom_frame.pack(fill="x")
        self.bottom_frame.pack_propagate(False)

        self.control_panel = ctk.CTkFrame(self.bottom_frame, fg_color=PALETTE["panel"])
        self.control_panel.pack(side="left", fill="both", expand=True, padx=12, pady=12)

        self.run_btn = ctk.CTkButton(
            self.control_panel,
            text="RUN ANALYSIS",
            command=self.start_analysis,
            fg_color=PALETTE["accent"],
            hover_color="#35b9ac",
            text_color="#071012",
            font=("Consolas", 12, "bold"),
            corner_radius=6,
        )
        self.run_btn.pack(anchor="w", pady=(0, 10))

        self.status_label = ctk.CTkLabel(
            self.control_panel,
            text="STATUS: IDLE",
            text_color=PALETTE["text"],
            font=("Consolas", 10),
        )
        self.status_label.pack(anchor="w", pady=(0, 8))

        self.ai_model_label = ctk.CTkLabel(
            self.control_panel,
            text="AI MODELS: SAM3 idle | DINOv3 idle",
            text_color=PALETTE["accent_2"],
            font=("Consolas", 10),
        )
        self.ai_model_label.pack(anchor="w", pady=(0, 6))

        self.ai_load_progress = ctk.CTkProgressBar(self.control_panel, width=270, progress_color=PALETTE["accent"])
        self.ai_load_progress.pack(anchor="w", pady=(0, 12))
        self.ai_load_progress.set(0.0)

        self.stroke_panel = ctk.CTkFrame(self.control_panel, fg_color=PALETTE["panel_lift"], corner_radius=6)
        self.stroke_panel.pack(anchor="w", fill="x", pady=(0, 4))
        self.stroke_label = ctk.CTkLabel(
            self.stroke_panel,
            text="LIVE STROKE",
            font=("Consolas", 10, "bold"),
            text_color=PALETTE["muted"],
        )
        self.stroke_label.pack(anchor="w", padx=10, pady=(8, 0))
        self.stroke_canvas = ctk.CTkCanvas(self.stroke_panel, height=42, bg=PALETTE["panel_lift"], highlightthickness=0)
        self.stroke_canvas.pack(fill="x", padx=10, pady=(2, 10))
        self.stroke_canvas.bind("<Configure>", lambda _event: self._draw_stroke_gauge())

        self.monitor = ResourceMonitorFrame(self.bottom_frame, width=360)
        self.monitor.pack(side="right", fill="y", padx=12, pady=12)

        self.processor = None
        self.target_point_proposer = TargetPointProposer()
        self.active_target_review_dialog = None
        self.active_reannotate_dialog = None
        self.pending_reannotate_request_id = None
        self.video_queue = []
        self.current_job_index = None
        self.is_processing_queue = False
        self.photo = None
        self.preview_image_bounds = None
        self.prompt_point = None
        self.show_prompt_overlay = True
        self.model_states = {"sam3": "idle", "dinov3": "idle"}
        self.current_stroke_pos = None
        self.latest_funscript_actions = []
        self.latest_funscript_data = None
        self.latest_funscript_output_path = None
        self.saved_funscript_actions = []
        self.modified_action_indices = set()
        self.pending_funscript_changes = False
        self.reaction_strength = 20
        self.reaction_bounces = 2
        self.reaction_density = "Normal"
        self.reaction_span = "Short"
        self.reaction_base_actions = []
        self.reaction_preview_active = False
        self._register_queue_drop_target()

    def _register_queue_drop_target(self):
        if self.TkdndVersion is None:
            return

        targets = [self.queue_list, self.queue_label, self.queue_hint]
        for attr in ("_parent_canvas", "_scrollbar"):
            widget = getattr(self.queue_list, attr, None)
            if widget is not None:
                targets.append(widget)

        for widget in targets:
            try:
                widget.drop_target_register(DND_FILES)
                widget.dnd_bind("<<Drop>>", self.handle_drop)
            except Exception as exc:
                print(f"Warning: queue drop target unavailable for {widget}: {exc}")
            try:
                widget.bind("<Button-1>", self.open_video_file_dialog)
            except Exception:
                pass

    @staticmethod
    def _is_video_file(file_path):
        return file_path.lower().endswith((".mp4", ".avi", ".mkv", ".mov", ".webm"))

    def handle_drop(self, event):
        files = self.tk.splitlist(event.data)
        added = 0
        for file_path in files:
            if self._is_video_file(file_path):
                self.add_to_queue(file_path)
                added += 1

        if added:
            self.status_label.configure(text=f"STATUS: ADDED {added} VIDEO(S) TO QUEUE")
        else:
            self.status_label.configure(text="STATUS: DROP VIDEO FILES ONLY")

    def open_video_file_dialog(self, _event=None):
        file_paths = filedialog.askopenfilenames(
            title="Select videos to add",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mkv *.mov *.webm"),
                ("All files", "*.*"),
            ],
        )
        added = 0
        for file_path in file_paths:
            if self._is_video_file(file_path):
                self.add_to_queue(file_path)
                added += 1

        if added:
            self.status_label.configure(text=f"STATUS: ADDED {added} VIDEO(S) TO QUEUE")

    def add_to_queue(self, file_path):
        file_path = os.path.abspath(file_path)
        if any(item["path"] == file_path for item in self.video_queue):
            return

        filename = os.path.basename(file_path)
        item_frame = ctk.CTkFrame(self.queue_list, fg_color=PALETTE["panel_lift"], corner_radius=6)
        item_frame.pack(fill="x", pady=4, padx=6)

        lbl = ctk.CTkLabel(
            item_frame,
            text=filename,
            font=("Consolas", 10, "bold"),
            text_color=PALETTE["text"],
            anchor="w",
        )
        lbl.pack(side="left", fill="x", expand=True, padx=8, pady=7)

        status = ctk.CTkLabel(item_frame, text="QUEUED", font=("Consolas", 8, "bold"), text_color=PALETTE["accent"])
        status.pack(side="right", padx=8)

        self.video_queue.append(
            {
                "path": file_path,
                "frame": item_frame,
                "label": lbl,
                "status": status,
                "state": "queued",
            }
        )

    def handle_click(self, event):
        self.status_label.configure(text="STATUS: USE INITIAL TARGET REVIEW TO SET TARGET")

    def start_analysis(self):
        if self.is_processing_queue:
            return

        if not self.video_queue:
            for file_name in sorted(os.listdir("Video")):
                if file_name.endswith((".mp4", ".avi", ".mkv")):
                    self.add_to_queue(os.path.join("Video", file_name))

        if not self.video_queue:
            self.status_label.configure(text="STATUS: NO VIDEOS FOUND")
            return

        self.is_processing_queue = True
        self.run_btn.configure(state="disabled")
        self._process_next_queue_item()

    def _process_next_queue_item(self):
        next_index = None
        for index, item in enumerate(self.video_queue):
            if item["state"] == "queued":
                next_index = index
                break

        if next_index is None:
            self.is_processing_queue = False
            self.current_job_index = None
            self.run_btn.configure(state="normal")
            self.status_label.configure(text="STATUS: QUEUE COMPLETE")
            return

        self.current_job_index = next_index
        job = self.video_queue[next_index]
        video_path = job["path"]
        self._update_queue_item(next_index, "review", "REVIEW", PALETTE["accent_2"])
        self.prompt_point = None
        self.status_label.configure(text="STATUS: PREPARING TARGET REVIEW...")

        try:
            review = self.target_point_proposer.prepare(video_path, user_seed_ratio=getattr(self, "prompt_point", None))
            self._display_review_frame(review.frame_bgr)

            review_result = self._open_target_review_dialog(video_path, review)
            if review_result is None or review_result.get("action") == "cancel":
                self._update_queue_item(next_index, "skipped", "SKIPPED", PALETTE["muted"])
                self.status_label.configure(text=f"STATUS: SKIPPED {os.path.basename(video_path)}")
                self.after(150, self._process_next_queue_item)
                return

            selected_point = review_result.get("point")
            if selected_point is None:
                self._update_queue_item(next_index, "skipped", "SKIPPED", PALETTE["muted"])
                self.status_label.configure(text=f"STATUS: NO TARGET FOR {os.path.basename(video_path)}")
                self.after(150, self._process_next_queue_item)
                return

            self._set_prompt_from_absolute_point(
                selected_point,
                review.width,
                review.height,
                status_text="STATUS: INITIAL TARGET CONFIRMED",
            )

            validation_result = review_result.get("validation")
            if validation_result is not None:
                self.status_label.configure(
                    text=(
                        f"STATUS: TARGET {validation_result.level} "
                        f"({validation_result.score:.2f}) CONFIRMED"
                    )
                )

            if validation_result is not None:
                self.status_label.configure(
                    text=(
                        f"STATUS: INITIALIZING AI ({validation_result.level} TARGET {validation_result.score:.2f})"
                    )
                )
            else:
                self.status_label.configure(text="STATUS: INITIALIZING AI...")
            self.show_prompt_overlay = False
            prompt = getattr(self, "prompt_point", None)
            self.pending_reannotate_request_id = None
            self.current_stroke_pos = None
            self.latest_funscript_actions = []
            self.latest_funscript_data = None
            self.latest_funscript_output_path = None
            self.saved_funscript_actions = []
            self.modified_action_indices = set()
            self.pending_funscript_changes = False
            self.reaction_base_actions = []
            self.reaction_preview_active = False
            self._draw_stroke_gauge()
            self._draw_result_graph()
            self.ai_load_progress.set(0.0)
            self._set_model_states("loading", "loading", text="AI MODELS: SAM3 loading | DINOv3 waiting")
            self._update_queue_item(next_index, "processing", "RUNNING", PALETTE["ok"])
            threading.Thread(target=self._run_process, args=(video_path, prompt), daemon=True).start()
        except Exception as e:
            self._update_queue_item(next_index, "failed", "FAILED", PALETTE["danger"])
            self.status_label.configure(text=f"ERROR: {str(e)[:40]}...")
            print(f"Error initializing AI: {e}")
            self.after(250, self._process_next_queue_item)

    def _run_process(self, video_path, prompt):
        try:
            self.processor = VideoProcessor(
                callback=self.update_progress,
                prompt_point=prompt,
                status_callback=self._handle_processor_status,
            )
            data = self.processor.process_video(video_path)
            if data:
                output_path = os.path.splitext(video_path)[0] + ".funscript"
                with open(output_path, "w", encoding="utf-8") as file_handle:
                    json.dump(data, file_handle)
                self.after(
                    0,
                    lambda d=data, path=output_path: self._finish_current_job(
                        success=True,
                        status_text=f"STATUS: SAVED TO {os.path.basename(path)}",
                        data=d,
                        output_path=path,
                    ),
                )
            else:
                self.after(
                    0,
                    lambda: self._finish_current_job(
                        success=False,
                        status_text="STATUS: STOPPED/ERROR",
                        data=None,
                    ),
                )
        except Exception as e:
            err_msg = str(e)
            print(f"Error during process: {err_msg}")
            self.after(
                0,
                lambda msg=err_msg: self._finish_current_job(
                    success=False,
                    status_text=f"ERROR: {msg[:40]}",
                    data=None,
                ),
            )

    def _finish_current_job(self, success, status_text, data=None, output_path=None):
        if data and data.get("actions"):
            normalized_actions = self._copy_actions(data["actions"])
            self.latest_funscript_data = dict(data)
            self.latest_funscript_data["actions"] = normalized_actions
            self.latest_funscript_actions = self._copy_actions(normalized_actions)
            self.saved_funscript_actions = self._copy_actions(normalized_actions)
            self.modified_action_indices = set()
            self.pending_funscript_changes = False
            self.reaction_base_actions = []
            self.reaction_preview_active = False
            self.latest_funscript_output_path = output_path
            self._draw_result_graph()
        if data is not None:
            if self.latest_funscript_data is None:
                self.latest_funscript_data = data
            if output_path is not None:
                self.latest_funscript_output_path = output_path

        if self.current_job_index is not None:
            if success:
                self._update_queue_item(self.current_job_index, "done", "DONE", PALETTE["ok"])
            else:
                self._update_queue_item(self.current_job_index, "failed", "FAILED", PALETTE["danger"])

        self.status_label.configure(text=status_text)
        self.processor = None
        self.pending_reannotate_request_id = None
        if self.is_processing_queue:
            self.after(350, self._process_next_queue_item)
        else:
            self.run_btn.configure(state="normal")

    def update_progress(self, current, total, frame, meta=None):
        self.show_prompt_overlay = False
        frame_bgr = frame.copy()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._set_preview_image(Image.fromarray(frame_rgb))

        percent = (current / total) * 100 if total else 0
        status_text = f"STATUS: PROCESSING ({percent:.1f}%)"

        if meta and meta.get("event") == "manual_reannotation":
            status_text = "STATUS: WAITING FOR TARGET REVIEW"
            self.after(0, lambda f=frame_bgr, m=meta: self._open_reannotate_dialog(f, m))
        elif meta and "stroke_pos" in meta:
            self.current_stroke_pos = int(meta["stroke_pos"])
            self.after(0, self._draw_stroke_gauge)

        self.after(0, lambda text=status_text: self.status_label.configure(text=text))

    def _update_queue_item(self, index, state, text, color):
        if index is None or index < 0 or index >= len(self.video_queue):
            return

        item = self.video_queue[index]
        item["state"] = state
        item["status"].configure(text=text, text_color=color)

        if state == "processing":
            item["frame"].configure(fg_color="#203321")
        elif state == "review":
            item["frame"].configure(fg_color="#332a1d")
        elif state == "done":
            item["frame"].configure(fg_color="#172820")
        elif state == "failed":
            item["frame"].configure(fg_color="#341c25")
        elif state == "skipped":
            item["frame"].configure(fg_color="#20242b")
        else:
            item["frame"].configure(fg_color=PALETTE["panel_lift"])

    def _open_target_review_dialog(self, video_path, review):
        if self.active_target_review_dialog is not None:
            try:
                self.active_target_review_dialog.destroy()
            except Exception:
                pass

        frame_rgb = cv2.cvtColor(review.frame_bgr, cv2.COLOR_BGR2RGB)
        review_meta = {
            "video_path": video_path,
            "frame_index": review.frame_index,
            "heatmap_peak": review.heatmap_peak,
            "auto_point": review.auto_point,
            "user_seed_point": review.user_seed_point,
            "candidate_points": review.candidate_points,
        }

        dialog = TargetReviewDialog(
            self,
            frame_rgb=frame_rgb,
            review_meta=review_meta,
            on_validate=lambda point: self._validate_initial_target_point(video_path, review, point),
        )
        self.active_target_review_dialog = dialog
        self.wait_window(dialog)
        self.active_target_review_dialog = None
        return dialog.result

    def _validate_initial_target_point(self, video_path, review, point):
        return self.target_point_proposer.validate_point(
            video_path=video_path,
            point=point,
            start_frame_index=review.frame_index,
        )

    def _open_reannotate_dialog(self, frame_bgr, meta):
        if self.processor is None:
            return

        request_id = meta["request_id"]
        if self.pending_reannotate_request_id == request_id and self.active_reannotate_dialog is not None:
            return

        if self.active_reannotate_dialog is not None:
            try:
                self.active_reannotate_dialog.destroy()
            except Exception:
                pass

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.pending_reannotate_request_id = request_id
        self.active_reannotate_dialog = ReannotateDialog(
            self,
            frame_rgb=frame_rgb,
            request_meta=meta,
            on_submit=self._submit_reannotation,
            on_bridge=self._bridge_reannotation,
            on_abort=self._abort_reannotation,
        )

    def _submit_reannotation(self, request_id, point):
        if self.processor is None or request_id != self.pending_reannotate_request_id:
            return
        self.processor.submit_reannotation(point=point, action="resume")
        self.pending_reannotate_request_id = None
        self.active_reannotate_dialog = None
        self.status_label.configure(text="STATUS: RESUMING WITH USER TARGET")

    def _bridge_reannotation(self, request_id):
        if self.processor is None or request_id != self.pending_reannotate_request_id:
            return
        self.processor.submit_reannotation(point=None, action="bridge")
        self.pending_reannotate_request_id = None
        self.active_reannotate_dialog = None
        self.status_label.configure(text="STATUS: CONTINUING WITH PREVIOUS FLOW")

    def _abort_reannotation(self, request_id):
        if self.processor is None or request_id != self.pending_reannotate_request_id:
            return
        self.processor.submit_reannotation(point=None, action="abort")
        self.pending_reannotate_request_id = None
        self.active_reannotate_dialog = None
        self.status_label.configure(text="STATUS: ABORT REQUESTED")

    def _display_review_frame(self, frame_bgr):
        self.show_prompt_overlay = True
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self._set_preview_image(Image.fromarray(frame_rgb))

    def _fit_preview_image(self, image):
        canvas_width = max(1, self.preview_canvas.winfo_width())
        canvas_height = max(1, self.preview_canvas.winfo_height())
        img_w, img_h = image.size

        if canvas_width <= 1 or canvas_height <= 1 or img_w <= 0 or img_h <= 0:
            self.preview_image_bounds = (0, 0, img_w, img_h)
            return image

        scale = min(canvas_width / img_w, canvas_height / img_h)
        display_w = max(1, int(img_w * scale))
        display_h = max(1, int(img_h * scale))
        offset_x = (canvas_width - display_w) // 2
        offset_y = (canvas_height - display_h) // 2
        self.preview_image_bounds = (offset_x, offset_y, display_w, display_h)
        return image.resize((display_w, display_h), Image.Resampling.LANCZOS)

    def _set_preview_image(self, image):
        fitted = self._fit_preview_image(image)
        self.photo = ImageTk.PhotoImage(image=fitted)
        self.after(0, self._draw_preview)

    def _set_prompt_point(self, point_ratio, status_text=None):
        self.prompt_point = (
            max(0.0, min(1.0, float(point_ratio[0]))),
            max(0.0, min(1.0, float(point_ratio[1]))),
        )
        self._draw_prompt_marker()

        if status_text is not None:
            self.status_label.configure(
                text=(
                    f"{status_text} "
                    f"({int(self.prompt_point[0] * 100)}%, {int(self.prompt_point[1] * 100)}%)"
                )
            )

    def _set_prompt_from_absolute_point(self, point, width, height, status_text=None):
        if width <= 0 or height <= 0:
            return
        self._set_prompt_point((point[0] / width, point[1] / height), status_text=status_text)

    def _draw_prompt_marker(self):
        self.preview_canvas.delete("marker")
        if self.prompt_point is None or not self.show_prompt_overlay:
            return

        canvas_width = self.preview_canvas.winfo_width()
        canvas_height = self.preview_canvas.winfo_height()
        if self.preview_image_bounds is not None:
            offset_x, offset_y, image_width, image_height = self.preview_image_bounds
            x = offset_x + (self.prompt_point[0] * image_width)
            y = offset_y + (self.prompt_point[1] * image_height)
        else:
            x = self.prompt_point[0] * canvas_width
            y = self.prompt_point[1] * canvas_height
        self.preview_canvas.create_oval(
            x - 5,
            y - 5,
            x + 5,
            y + 5,
            fill="#ff00ff",
            outline="white",
            tags="marker",
        )

    def _draw_preview(self):
        self.preview_canvas.delete("preview_image")
        canvas_width = max(1, self.preview_canvas.winfo_width())
        canvas_height = max(1, self.preview_canvas.winfo_height())
        self.preview_canvas.create_rectangle(0, 0, canvas_width, canvas_height, fill="#05070a", outline="", tags="preview_image")
        if self.photo is not None:
            x = 0
            y = 0
            if self.preview_image_bounds is not None:
                x, y, _, _ = self.preview_image_bounds
            self.preview_canvas.create_image(x, y, anchor="nw", image=self.photo, tags="preview_image")
        self._draw_prompt_marker()

    def _draw_stroke_gauge(self):
        if not hasattr(self, "stroke_canvas"):
            return

        canvas = self.stroke_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        pad = 12
        track_x0 = pad
        track_x1 = max(track_x0 + 1, width - pad)
        center_y = height // 2

        canvas.create_line(track_x0, center_y, track_x1, center_y, fill=PALETTE["line"], width=8, capstyle="round")
        for ratio in (0.0, 0.5, 1.0):
            x = track_x0 + ((track_x1 - track_x0) * ratio)
            canvas.create_line(x, center_y - 9, x, center_y + 9, fill="#3f4d5e", width=1)

        if self.current_stroke_pos is None:
            canvas.create_text(track_x0, center_y, anchor="w", text="waiting", fill=PALETTE["muted"], font=("Consolas", 10))
            return

        pos = max(0, min(100, int(self.current_stroke_pos)))
        x = track_x0 + ((track_x1 - track_x0) * (pos / 100.0))
        fill_color = PALETTE["accent"] if pos >= 50 else PALETTE["accent_2"]
        canvas.create_line(track_x0, center_y, x, center_y, fill=fill_color, width=8, capstyle="round")
        canvas.create_oval(x - 8, center_y - 8, x + 8, center_y + 8, fill=fill_color, outline="#ffffff", width=1)
        canvas.create_text(track_x0, center_y - 16, anchor="w", text="BOTTOM", fill=PALETTE["muted"], font=("Consolas", 8))
        canvas.create_text(track_x1, center_y - 16, anchor="e", text="TOP", fill=PALETTE["muted"], font=("Consolas", 8))

    @staticmethod
    def _clamp_pos(value):
        return max(0, min(100, int(round(value))))

    def _copy_actions(self, actions):
        copied = []
        for action in actions or []:
            if "at" not in action or "pos" not in action:
                continue
            copied.append({"at": int(action["at"]), "pos": self._clamp_pos(action["pos"])})
        return copied

    def _find_modified_action_indices(self, actions):
        modified = set()
        saved_counts = Counter(
            (int(action["at"]), int(action["pos"]))
            for action in self.saved_funscript_actions or []
        )
        for index, action in enumerate(actions):
            key = (int(action["at"]), int(action["pos"]))
            if saved_counts[key] > 0:
                saved_counts[key] -= 1
            else:
                modified.add(index)
        return modified

    def _apply_result_actions(self, actions, label):
        if not actions:
            self.status_label.configure(text="STATUS: NO RESULT TO MODIFY")
            return

        normalized_actions = self._copy_actions(actions)
        data = dict(self.latest_funscript_data or {})
        data["actions"] = normalized_actions
        self.latest_funscript_data = data
        self.latest_funscript_actions = normalized_actions
        self.modified_action_indices = self._find_modified_action_indices(normalized_actions)
        self.pending_funscript_changes = bool(self.modified_action_indices)
        if not label.startswith("REACTION"):
            self.reaction_base_actions = []
            self.reaction_preview_active = False
        self._draw_result_graph()
        if self.pending_funscript_changes:
            self.status_label.configure(text=f"STATUS: {label} PREVIEW - SAVE TO WRITE FILE")
        else:
            self.status_label.configure(text=f"STATUS: {label} MADE NO CHANGES")

    def _save_result_changes(self):
        if not self.latest_funscript_actions:
            self.status_label.configure(text="STATUS: NO RESULT TO SAVE")
            return

        if not self.pending_funscript_changes:
            self.status_label.configure(text="STATUS: NO UNSAVED FUNSCRIPT CHANGES")
            return

        output_path = self.latest_funscript_output_path
        if not output_path:
            self.status_label.configure(text="STATUS: NO FUNSCRIPT OUTPUT PATH")
            return

        try:
            with open(output_path, "w", encoding="utf-8") as file_handle:
                json.dump(self.latest_funscript_data, file_handle)
            self.saved_funscript_actions = self._copy_actions(self.latest_funscript_actions)
            self.modified_action_indices = set()
            self.pending_funscript_changes = False
            self.reaction_base_actions = []
            self.reaction_preview_active = False
            self._draw_result_graph()
            self.status_label.configure(text=f"STATUS: SAVED {os.path.basename(output_path)}")
        except Exception as exc:
            self.status_label.configure(text=f"STATUS: SAVE FAILED: {str(exc)[:32]}")

    def _discard_result_changes(self):
        if not self.pending_funscript_changes:
            self.status_label.configure(text="STATUS: NO UNSAVED CHANGES TO DISCARD")
            return

        restored_actions = self._copy_actions(self.saved_funscript_actions)
        data = dict(self.latest_funscript_data or {})
        data["actions"] = restored_actions
        self.latest_funscript_data = data
        self.latest_funscript_actions = restored_actions
        self.modified_action_indices = set()
        self.pending_funscript_changes = False
        self.reaction_base_actions = []
        self.reaction_preview_active = False
        self._draw_result_graph()
        self.status_label.configure(text="STATUS: DISCARDED FUNSCRIPT PREVIEW CHANGES")

    def _snap_result_peaks(self):
        actions = self.latest_funscript_actions or []
        if len(actions) < 2:
            self.status_label.configure(text="STATUS: NEED AT LEAST 2 ACTIONS")
            return

        snapped = []
        for index, action in enumerate(actions):
            pos = int(action["pos"])
            if len(actions) == 2:
                target = 100 if pos >= actions[1 - index]["pos"] else 0
            elif index == 0:
                target = 0 if pos <= actions[index + 1]["pos"] else 100
            elif index == len(actions) - 1:
                target = 0 if pos <= actions[index - 1]["pos"] else 100
            else:
                prev_pos = int(actions[index - 1]["pos"])
                next_pos = int(actions[index + 1]["pos"])
                if pos >= prev_pos and pos >= next_pos:
                    target = 100
                elif pos <= prev_pos and pos <= next_pos:
                    target = 0
                else:
                    target = pos
            snapped.append({"at": int(action["at"]), "pos": target})

        self._apply_result_actions(snapped, "SNAP PEAKS")

    def _snap_result_all_values(self):
        actions = self.latest_funscript_actions or []
        if len(actions) < 2:
            self.status_label.configure(text="STATUS: NEED AT LEAST 2 ACTIONS")
            return

        positions = [int(action["pos"]) for action in actions]
        low = min(positions)
        high = max(positions)
        if high <= low:
            self.status_label.configure(text="STATUS: CANNOT SNAP FLAT RESULT")
            return

        threshold = low + ((high - low) / 2.0)
        snapped = [
            {
                "at": int(action["at"]),
                "pos": 100 if int(action["pos"]) >= threshold else 0,
            }
            for action in actions
        ]
        self._apply_result_actions(snapped, "SNAP ALL")

    def _set_reaction_strength(self, value):
        self.reaction_strength = max(1, min(45, int(round(float(value)))))
        self.reaction_label.configure(text=f"Reaction: strength {self.reaction_strength}")
        if self.reaction_preview_active:
            self._preview_reaction_bounces()

    def _set_reaction_bounces(self, value):
        self.reaction_bounces = int(str(value).replace("x", ""))
        if self.reaction_preview_active:
            self._preview_reaction_bounces()

    def _set_reaction_density(self, value):
        self.reaction_density = value
        if self.reaction_preview_active:
            self._preview_reaction_bounces()

    def _set_reaction_span(self, value):
        self.reaction_span = value
        if self.reaction_preview_active:
            self._preview_reaction_bounces()

    def _reaction_spacing_ms(self):
        return {
            "Tight": 70,
            "Normal": 110,
            "Loose": 160,
        }.get(self.reaction_density, 110)

    def _reaction_span_ratio(self):
        return {
            "Micro": 0.14,
            "Quick": 0.22,
            "Short": 0.35,
            "Normal": 0.50,
        }.get(self.reaction_span, 0.35)

    def _reaction_effective_gap_ms(self):
        spacing_ms = self._reaction_spacing_ms()
        return {
            "Micro": min(22, spacing_ms),
            "Quick": min(35, spacing_ms),
            "Short": min(55, spacing_ms),
            "Normal": spacing_ms,
        }.get(self.reaction_span, spacing_ms)

    def _build_reaction_actions(self, actions):
        reserve_gap_ms = self._reaction_spacing_ms()
        point_gap_ms = self._reaction_effective_gap_ms()
        span_ratio = self._reaction_span_ratio()
        strength = max(1, min(45, int(self.reaction_strength)))
        max_bounces = max(1, min(3, int(self.reaction_bounces)))
        min_swing = max(8, int(strength * 0.5))
        reacted = []
        inserted_count = 0

        for index, action in enumerate(actions):
            current = {"at": int(action["at"]), "pos": int(action["pos"])}
            reacted.append(current)

            if index <= 0 or index >= len(actions) - 1:
                continue

            prev_pos = int(actions[index - 1]["pos"])
            next_pos = int(actions[index + 1]["pos"])
            current_pos = current["pos"]
            current_at = current["at"]
            next_at = int(actions[index + 1]["at"])
            available_ms = next_at - current_at

            # Reaction is only meaningful after the stroke hits the lower endpoint.
            is_bottom_hit = (
                current_pos <= prev_pos
                and current_pos <= next_pos
                and current_pos <= 25
                and max(prev_pos, next_pos) >= 70
            )
            has_enough_swing = (
                abs(prev_pos - current_pos) >= min_swing
                and abs(next_pos - current_pos) >= min_swing
            )
            if not (is_bottom_hit and has_enough_swing):
                continue

            bounce_count = max_bounces
            while bounce_count > 0 and available_ms < point_gap_ms * ((bounce_count * 2) + 1):
                bounce_count -= 1
            if bounce_count <= 0:
                continue

            max_window_ms = max(0, available_ms - min(reserve_gap_ms, available_ms * 0.45))
            reaction_window_ms = min(max_window_ms, available_ms * span_ratio)
            while bounce_count > 0 and reaction_window_ms < point_gap_ms * (bounce_count * 2):
                bounce_count -= 1
            if bounce_count <= 0:
                continue

            step_ms = reaction_window_ms / (bounce_count * 2)
            for bounce_index in range(1, bounce_count + 1):
                amplitude = strength * (bounce_index / bounce_count)
                bounce_pos = self._clamp_pos(current_pos + amplitude)
                bounce_at = int(round(current_at + (step_ms * ((bounce_index - 1) * 2 + 1))))
                return_at = int(round(current_at + (step_ms * (bounce_index * 2))))
                reacted.append({"at": bounce_at, "pos": bounce_pos})
                reacted.append({"at": return_at, "pos": current_pos})
                inserted_count += 2

        reacted.sort(key=lambda item: item["at"])
        return reacted, inserted_count

    def _preview_reaction_bounces(self):
        if not self.reaction_base_actions:
            self.reaction_base_actions = self._copy_actions(self.latest_funscript_actions)

        if len(self.reaction_base_actions) < 3:
            self.status_label.configure(text="STATUS: NEED AT LEAST 3 ACTIONS")
            return

        reacted, inserted_count = self._build_reaction_actions(self.reaction_base_actions)
        if inserted_count == 0:
            self.status_label.configure(text="STATUS: NO SAFE BOTTOM GAP FOR REACTION")
            return

        self.reaction_preview_active = True
        self._apply_result_actions(reacted, f"REACTION +{inserted_count}")

    def _normalize_result_range(self):
        self._remap_result_range(0, 100, label="NORMALIZE 0-100")

    def _custom_result_range(self):
        value = simpledialog.askstring(
            "Custom Range",
            "Enter low,high values (example: 5,95)",
            parent=self,
        )
        if not value:
            return

        try:
            low_text, high_text = value.replace(" ", "").split(",", 1)
            low = self._clamp_pos(float(low_text))
            high = self._clamp_pos(float(high_text))
        except Exception:
            self.status_label.configure(text="STATUS: CUSTOM RANGE FORMAT IS low,high")
            return

        if high <= low:
            self.status_label.configure(text="STATUS: CUSTOM RANGE HIGH MUST BE GREATER THAN LOW")
            return

        self._remap_result_range(low, high, label=f"CUSTOM RANGE {low}-{high}")

    def _remap_result_range(self, target_low, target_high, label):
        actions = self.latest_funscript_actions or []
        if len(actions) < 2:
            self.status_label.configure(text="STATUS: NEED AT LEAST 2 ACTIONS")
            return

        positions = [int(action["pos"]) for action in actions]
        source_low = min(positions)
        source_high = max(positions)
        if source_high <= source_low:
            self.status_label.configure(text="STATUS: CANNOT REMAP FLAT RESULT")
            return

        remapped = []
        for action in actions:
            ratio = (int(action["pos"]) - source_low) / max(1, source_high - source_low)
            pos = target_low + (ratio * (target_high - target_low))
            remapped.append({"at": int(action["at"]), "pos": self._clamp_pos(pos)})

        self._apply_result_actions(remapped, label)

    def _draw_result_graph(self):
        if not hasattr(self, "result_canvas"):
            return

        canvas = self.result_canvas
        canvas.delete("all")
        width = max(1, canvas.winfo_width())
        height = max(1, canvas.winfo_height())
        actions = self.latest_funscript_actions or []
        x_pad = 36
        right_pad = 12
        y_pad = 10

        canvas.create_rectangle(0, 0, width, height, fill=PALETTE["panel_soft"], outline="")
        for level in (100, 75, 50, 25, 0):
            y = y_pad + ((100 - level) / 100.0 * max(1, height - (y_pad * 2)))
            line_color = "#2d3c50" if level in (0, 50, 100) else "#243142"
            canvas.create_line(x_pad - 4, y, width - right_pad, y, fill=line_color, dash=(4, 4))
            canvas.create_text(
                x_pad - 9,
                y,
                text=str(level),
                fill=PALETTE["muted"],
                font=("Consolas", 8),
                anchor="e",
            )

        if len(actions) < 2:
            canvas.create_text(
                width // 2,
                height // 2,
                text="Final stroke graph will appear here",
                fill=PALETTE["muted"],
                font=("Consolas", 10),
            )
            self.result_meta_label.configure(text="waiting for completed analysis")
            return

        start_at = actions[0]["at"]
        end_at = max(start_at + 1, actions[-1]["at"])
        coords = []
        for action in actions:
            x = x_pad + ((action["at"] - start_at) / (end_at - start_at) * max(1, width - x_pad - right_pad))
            y = y_pad + ((100 - action["pos"]) / 100.0 * max(1, height - (y_pad * 2)))
            coords.extend([x, y])

        if len(coords) >= 4:
            canvas.create_line(*coords, fill=PALETTE["accent"], width=2, smooth=True)

        for index, action in enumerate(actions):
            x = x_pad + ((action["at"] - start_at) / (end_at - start_at) * max(1, width - x_pad - right_pad))
            y = y_pad + ((100 - action["pos"]) / 100.0 * max(1, height - (y_pad * 2)))
            color = PALETTE["accent_2"] if action["pos"] <= 15 or action["pos"] >= 85 else PALETTE["accent"]
            if index in self.modified_action_indices:
                canvas.create_oval(x - 5, y - 5, x + 5, y + 5, fill="", outline=PALETTE["danger"], width=2)
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="#ffffff", width=1)
            else:
                canvas.create_oval(x - 2, y - 2, x + 2, y + 2, fill=color, outline="")

        duration_s = (end_at - start_at) / 1000.0
        positions = [action["pos"] for action in actions]
        unsaved_text = (
            f" | unsaved {len(self.modified_action_indices)}"
            if self.pending_funscript_changes
            else ""
        )
        self.result_meta_label.configure(
            text=f"{len(actions)} actions | {duration_s:.1f}s | range {min(positions)}-{max(positions)}{unsaved_text}"
        )

    def _handle_processor_status(self, meta):
        self.after(0, lambda m=meta: self._apply_processor_status(m))

    def _apply_processor_status(self, meta):
        progress = max(0.0, min(1.0, float(meta.get("progress", 0.0))))
        self.ai_load_progress.set(progress)

        stage = meta.get("stage")
        message = meta.get("message", "")
        sam3_state = meta.get("sam3_state", self.model_states["sam3"])
        dinov3_state = meta.get("dinov3_state", self.model_states["dinov3"])
        self._set_model_states(sam3_state, dinov3_state)

        if stage in {"init", "sam3_loading", "sam3_ready", "sam3_failed", "dinov3_loading", "dinov3_ready", "dinov3_failed", "ready"}:
            self.status_label.configure(text=f"STATUS: {message.upper()}")

    def _set_model_states(self, sam3_state, dinov3_state, text=None):
        self.model_states["sam3"] = sam3_state
        self.model_states["dinov3"] = dinov3_state

        if text is None:
            text = f"AI MODELS: SAM3 {sam3_state} | DINOv3 {dinov3_state}"
        self.ai_model_label.configure(text=text)


if __name__ == "__main__":
    app = App()
    app.mainloop()
