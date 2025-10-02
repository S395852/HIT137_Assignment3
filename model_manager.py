from abc import ABC, abstractmethod
from typing import Any, Dict
from transformers import pipeline
from utils import timed

# ---------- Mixin for simple logging (used in multiple inheritance) ----------
class LoggingMixin:
    def log(self, msg: str):
        print(f"[APP] {msg}")

# ---------- Base class ----------
class ModelBase(ABC):
    def __init__(self, task: str, model_name: str):
        self._task = task
        self._model_name = model_name
        self.__pipe = None  # encapsulated (private)

    @timed("load")
    def load(self):
        # Default loader uses Hugging Face Transformers pipeline
        if self.__pipe is None:
            self.__pipe = pipeline(self._task, model=self._model_name)
        return self.__pipe

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        ...

# ---------- Concrete models ----------
class TextToImageModel(ModelBase):
    def __init__(self):
        # Using a compact / fast model id for demo purposes
        super().__init__("text-to-image", "stabilityai/sd-turbo")

    @timed("load")
    def load(self):
        # Override to use Diffusers instead of Transformers
        if self._ModelBase__pipe is None:  # access the private slot via name-mangling
            from diffusers import AutoPipelineForText2Image
            import torch
            pipe = AutoPipelineForText2Image.from_pretrained(self._model_name, torch_dtype=torch.float32)
            pipe = pipe.to("cpu")  # CPU-friendly default; switch to 'cuda' if you have a GPU
            self._ModelBase__pipe = pipe
        return self._ModelBase__pipe

    @timed("text-to-image")
    def run(self, input_data: str):
        pipe = self.load()
        prompt = input_data or "a cute koala reading a book, digital art"
        # sd-turbo is optimized for very few steps (1–4). guidance_scale=0 for speed.
        result = pipe(prompt, num_inference_steps=2, guidance_scale=0.0)
        return result.images[0]  # PIL.Image

class ImageClassificationModel(ModelBase):
    def __init__(self):
        super().__init__("image-classification", "google/vit-base-patch16-224")

    @timed("image-classification")
    def run(self, input_data: Any) -> Dict[str, float]:
        pipe = self.load()
        preds = pipe(input_data)  # list of dicts
        best = max(preds, key=lambda d: d["score"])
        return {"label": best["label"], "score": float(best["score"])}

# ---------- Factory ----------
def get_model(label: str) -> ModelBase:
    if label == "Text-to-Image (SD-Turbo)":
        return TextToImageModel()
    elif label == "Image Classification (ViT-Base-16)":
        return ImageClassificationModel()
    else:
        raise ValueError("Unknown model selection")