import time
from functools import wraps

# ---------- Decorator (timing) ----------
def timed(label=""):
    def deco(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            t0 = time.time()
            out = func(*args, **kwargs)
            dt = (time.time() - t0) * 1000
            print(f"[TIMER] {label or func.__name__}: {dt:.1f} ms")
            return out
        return wrapper
    return deco


# ---------- Explanations to show in GUI ----------
OOP_EXPLANATION = """
• Encapsulation: Each model class hides its pipeline inside a private attribute (__pipe).
• Polymorphism: All model classes implement the same run(input_data) interface differently.
• Inheritance: ModelBase is the parent; specific models inherit and override load/run.
• Multiple inheritance: The App class inherits from Tk and LoggingMixin.
• Decorators: @timed measures how long the model call takes.
• Method overriding: load() and run() are overridden in child classes.
"""

MODEL_INFOS = {
    "Text-to-Image (SD-Turbo)":
        "Category: Text → Image. Stable Diffusion Turbo for fast 512×512 image synthesis. Use short prompts; 1–4 steps recommended.",
    "Image Classification (ViT-Base-16)":
        "Category: Image → Label. Vision Transformer model to predict the main object."
}