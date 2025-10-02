import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
from typing import Optional
from model_manager import get_model, LoggingMixin
from utils import OOP_EXPLANATION, MODEL_INFOS

MODEL_CHOICES = [
    "Text-to-Image (SD-Turbo)",
    "Image Classification (ViT-Base-16)",
]

class App(tk.Tk, LoggingMixin):   # multiple inheritance
    def __init__(self):
        super().__init__()
        self.title("HIT137 – Hugging Face OOP GUI")
        self.geometry("980x700")
        self.minsize(920, 620)

        # State
        self.current_model_label: Optional[str] = None
        self.current_model = None
        self.current_image_path: Optional[str] = None
        self.current_image_preview: Optional[ImageTk.PhotoImage] = None
        self.generated_image_pil: Optional[Image.Image] = None
        self.generated_image_preview: Optional[ImageTk.PhotoImage] = None

        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        # Top controls
        top = ttk.Frame(self, padding=10)
        top.pack(side=tk.TOP, fill=tk.X)

        ttk.Label(top, text="Select Model:", font=("Segoe UI", 10, "bold")).pack(side=tk.LEFT)
        self.model_combo = ttk.Combobox(top, values=MODEL_CHOICES, state="readonly", width=35)
        self.model_combo.current(0)
        self.model_combo.pack(side=tk.LEFT, padx=8)
        ttk.Button(top, text="Load Model", command=self.on_load_model).pack(side=tk.LEFT, padx=4)

        ttk.Separator(self, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=4)

        # Middle: input + output
        mid = ttk.Frame(self, padding=10)
        mid.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Left: input area
        left = ttk.LabelFrame(mid, text="Input", padding=10)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.text_input = tk.Text(left, height=8)
        self.text_input.insert("1.0", "Write a short prompt (used for Text-to-Image)…")
        self.text_input.pack(fill=tk.X)

        btns = ttk.Frame(left)
        btns.pack(fill=tk.X, pady=6)
        ttk.Button(btns, text="Choose Image…", command=self.on_choose_image).pack(side=tk.LEFT)
        ttk.Button(btns, text="Run", command=self.on_run).pack(side=tk.LEFT, padx=6)
        ttk.Button(btns, text="Save Image…", command=self.on_save_image).pack(side=tk.LEFT)

        self.image_canvas = tk.Label(left, relief=tk.SUNKEN, width=44, height=14)
        self.image_canvas.pack(pady=6)

        # Right: output area
        right = ttk.LabelFrame(mid, text="Output", padding=10)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.output_text = tk.Text(right, height=18)
        self.output_text.pack(fill=tk.BOTH, expand=True)

        # Bottom: model info + OOP explanation
        bottom = ttk.Frame(self, padding=10)
        bottom.pack(side=tk.BOTTOM, fill=tk.BOTH)

        self.info_box = tk.Text(bottom, width=50, height=6)
        self.info_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 8))

        self.oop_box = tk.Text(bottom, width=50, height=6)
        self.oop_box.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.oop_box.insert("1.0", OOP_EXPLANATION)
        self.oop_box.configure(state="disabled")

        self.update_model_info(MODEL_CHOICES[0])

    # ---------------- Handlers ----------------
    def update_model_info(self, label: str):
        self.info_box.configure(state="normal")
        self.info_box.delete("1.0", tk.END)
        self.info_box.insert("1.0", f"Model: {label}\n{MODEL_INFOS.get(label, '')}")
        self.info_box.configure(state="disabled")

    def on_load_model(self):
        sel = self.model_combo.get()
        self.update_model_info(sel)
        try:
            self.current_model_label = sel
            self.current_model = get_model(sel)
            self.log(f"Loading {sel} …")
            self.current_model.load()
            messagebox.showinfo("Model", f"{sel} loaded successfully.")
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def on_choose_image(self):
        path = filedialog.askopenfilename(
            title="Select an image (for classification)",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")]
        )
        if not path:
            return
        self.current_image_path = path
        img = Image.open(path).convert("RGB")
        img.thumbnail((360, 360))
        self.current_image_preview = ImageTk.PhotoImage(img)
        self.image_canvas.configure(image=self.current_image_preview)
        self.generated_image_pil = None
        self.generated_image_preview = None

    def on_run(self):
        if not self.current_model:
            messagebox.showwarning("Model", "Please load a model first.")
            return

        label = self.current_model_label
        self.output_text.delete("1.0", tk.END)

        try:
            if label.startswith("Text-to-Image"):
                prompt = self.text_input.get("1.0", tk.END).strip()
                image = self.current_model.run(prompt)  # PIL.Image
                self.generated_image_pil = image.copy()
                preview = image.copy()
                preview.thumbnail((360, 360))
                self.generated_image_preview = ImageTk.PhotoImage(preview)
                self.image_canvas.configure(image=self.generated_image_preview)
                self.output_text.insert(tk.END, "Image generated successfully. Use 'Save Image…' to export.")

            elif label.startswith("Image Classification"):
                if not self.current_image_path:
                    messagebox.showinfo("Input", "Choose an image first.")
                    return
                out = self.current_model.run(self.current_image_path)
                self.output_text.insert(tk.END, f"Label: {out['label']}\nScore: {out['score']:.4f}")

            else:
                self.output_text.insert(tk.END, "Unsupported selection.")

        except Exception as e:
            messagebox.showerror("Run Error", str(e))

    def on_save_image(self):
        if not self.generated_image_pil:
            messagebox.showinfo("Save", "No generated image to save.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png"), ("JPEG image", "*.jpg;*.jpeg")]
        )
        if not path:
            return
        try:
            self.generated_image_pil.save(path)
            messagebox.showinfo("Saved", f"Saved to: {path}")
        except Exception as e:
            messagebox.showerror("Save Error", str(e))