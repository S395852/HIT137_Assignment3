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
        self.__pipe = None  # encapsulated

    @timed("load")
    def load(self):
        if self.__pipe is None:
            self.__pipe = pipeline(self._task, model=self._model_name)
        return self.__pipe

    @abstractmethod
    def run(self, input_data: Any) -> Any:
        ...

# ---------- Concrete models ----------
class TextGenerationModel(ModelBase):
    def __init__(self):
        super().__init__("text-generation", "distilgpt2")

    @timed("text-generation")
    def run(self, input_data: str) -> str:
        pipe = self.load()
        prompt = input_data or "Hello from HIT137!"
        out = pipe(prompt, max_new_tokens=40, do_sample=True, temperature=0.9)[0]["generated_text"]
        return out

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
    if label == "Text Generation (distilgpt2)":
        return TextGenerationModel()
    elif label == "Image Classification (ViT-Base-16)":
        return ImageClassificationModel()
    else:
        raise ValueError("Unknown model selection")
