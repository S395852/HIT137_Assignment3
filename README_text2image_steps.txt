HIT137 A3 — Text-to-Image Upgrade (Windows / VS Code)

1) Create / reset your virtual environment (PowerShell):
   - Close VS Code terminals if open.
   - If you see a ".venv" folder in your project, delete it (optional, to start clean).
   - python -m venv .venv
   - .\.venv\Scripts\Activate
   - python -m pip install --upgrade pip

2) Install dependencies:
   - pip install -r requirements.txt
   - Install PyTorch (choose ONE):
     * CPU-only:  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
     * NVIDIA GPU (CUDA 12.1): pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

3) Replace your files with the ones in this folder:
   - utils.py, model_manager.py, gui.py, main.py

4) Run the app:
   - python main.py

5) How it works now:
   - "Text-to-Image (SD-Turbo)" uses Diffusers to generate an image from your prompt (left text box).
   - "Image Classification (ViT-Base-16)" still lets you choose an image and shows the top label + score.
   - Use "Save Image…" to export a generated image.

Notes:
- sd-turbo is optimized for very few steps. We set num_inference_steps=2 and guidance_scale=0.0 for speed.
- On CPU, generation will be slower, but should still complete for 512×512.
- If you have trouble downloading models, check your internet connection or try again later.