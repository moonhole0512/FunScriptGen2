import threading
import time
import tkinter as tk

import GPUtil
import psutil


PALETTE = {
    "panel": "#111720",
    "panel_soft": "#151d28",
    "line": "#2c3a4d",
    "text": "#edf3f8",
    "muted": "#8fa0af",
    "cpu": "#42d9c8",
    "gpu": "#ffb84d",
    "vram": "#5ee38a",
}


class ResourceGraph(tk.Canvas):
    def __init__(self, master, width=330, height=118, bg=PALETTE["panel_soft"], on_metrics=None, **kwargs):
        super().__init__(master, width=width, height=height, bg=bg, highlightthickness=0, **kwargs)
        self.width = width
        self.height = height
        self.data_points = {"cpu": [], "gpu": [], "vram": []}
        self.max_points = 50
        self.colors = {"cpu": PALETTE["cpu"], "gpu": PALETTE["gpu"], "vram": PALETTE["vram"]}
        self.on_metrics = on_metrics
        self._running = True
        self.bind("<Configure>", self._on_resize)
        
        self._draw_grid()
        self.update_thread = threading.Thread(target=self._update_loop, daemon=True)
        self.update_thread.start()

    def destroy(self):
        self._running = False
        super().destroy()

    def _on_resize(self, event):
        self.width = max(1, int(event.width))
        self.height = max(1, int(event.height))
        self._redraw()

    def _draw_grid(self):
        self.delete("grid")
        height = max(1, int(self.height))
        width = max(1, int(self.width))
        for i in range(0, height + 1, 30):
            self.create_line(0, i, width, i, fill=PALETTE["line"], dash=(4, 4), tags="grid")
        for i in range(0, width + 1, 55):
            self.create_line(i, 0, i, height, fill="#223044", tags="grid")

    def _update_loop(self):
        while self._running:
            cpu = psutil.cpu_percent()
            gpu_usage = 0
            vram_usage = 0
            
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu_usage = gpus[0].load * 100
                    vram_usage = gpus[0].memoryUtil * 100
            except:
                pass
            
            self._add_data("cpu", cpu)
            self._add_data("gpu", gpu_usage)
            self._add_data("vram", vram_usage)
            
            try:
                self.after(0, self._redraw)
                if self.on_metrics is not None:
                    metrics = {
                        "cpu": cpu,
                        "gpu": gpu_usage,
                        "vram": vram_usage,
                    }
                    self.after(0, lambda m=metrics: self.on_metrics(m))
            except tk.TclError:
                self._running = False
                break
            time.sleep(1)

    def _add_data(self, key, value):
        self.data_points[key].append(value)
        if len(self.data_points[key]) > self.max_points:
            self.data_points[key].pop(0)

    def _redraw(self):
        if not self.winfo_exists():
            return
        self.delete("plot")
        self._draw_grid()
        width = max(1, int(self.width))
        height = max(1, int(self.height))
        for key, points in self.data_points.items():
            if len(points) < 2:
                continue
            
            coords = []
            visible_points = points[-self.max_points :]
            x_step = width / max(1, len(visible_points) - 1)
            for i, p in enumerate(visible_points):
                x = i * x_step
                y = height - (p / 100 * height)
                coords.append(x)
                coords.append(y)
            
            self.create_line(*coords, fill=self.colors[key], width=2, tags="plot", smooth=True)

class ResourceMonitorFrame(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.configure(bg=PALETTE["panel"])
        
        self.title = tk.Label(
            self,
            text="RESOURCE MONITOR",
            fg=PALETTE["text"],
            bg=PALETTE["panel"],
            font=("Consolas", 12, "bold"),
        )
        self.title.pack(anchor="w", padx=10, pady=(4, 2))
        
        self.graph = ResourceGraph(self, on_metrics=self._update_metrics)
        self.graph.pack(padx=10, pady=(2, 7), fill="both", expand=True)
        
        self.legend = tk.Frame(self, bg=PALETTE["panel"])
        self.legend.pack(fill="x", padx=8, pady=(0, 6))

        self.metric_labels = {}
        self._add_legend("CPU", "cpu", PALETTE["cpu"])
        self._add_legend("GPU", "gpu", PALETTE["gpu"])
        self._add_legend("VRAM", "vram", PALETTE["vram"])

    def _add_legend(self, label, key, color):
        f = tk.Frame(self.legend, bg=PALETTE["panel_soft"], highlightthickness=1, highlightbackground=PALETTE["line"])
        f.pack(side="left", expand=True, fill="x", padx=3)
        tk.Frame(f, width=8, height=8, bg=color).pack(side="left", padx=(8, 5), pady=7)
        tk.Label(f, text=label, fg=PALETTE["muted"], bg=PALETTE["panel_soft"], font=("Consolas", 9, "bold")).pack(side="left")
        value_label = tk.Label(f, text="0%", fg=color, bg=PALETTE["panel_soft"], font=("Consolas", 9, "bold"))
        value_label.pack(side="right", padx=(4, 8))
        self.metric_labels[key] = value_label

    def _update_metrics(self, metrics):
        for key, label in self.metric_labels.items():
            value = float(metrics.get(key, 0.0))
            label.configure(text=f"{value:5.1f}%")
