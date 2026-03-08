# Sign Language Detection with DETR

Train a DETR model for sign-language symbol detection from webcam-captured data.

## End-to-End Execution Steps
1. Install dependencies:
```bash
pip install uv
```
```bash
uv sync
```
2. Confirm your classes and split config in `src/config.json`:
- `dataset_root`
- `train_split`
- `split_seed`
- `classes`
- `colors`
3. Collect data:
```bash
uv run src/utils/collect_images.py --total-images 150 --camera-id 0
```
4. Create checkpoints folder:
```bash
mkdir checkpoints
```
5. Train model:
```bash
uv run src/train.py
```
6. Test with saved checkpoint (update path in `src/test.py` first):
```bash
uv run src/test.py
```
7. Run realtime inference (update path in `src/realtime.py` first):
```bash
uv run src/realtime.py
```

## Capture Features
- Supports one-hand and two-hand signs.
- Uses full hand landmarks (including fingers), not palm-only, when MediaPipe is available.
- Writes one YOLO label line per detected hand for the current class.
- Captures only when detection is stable and image sharpness is above threshold.
- Stops automatically when all class targets are completed.
- Saves into unified dataset structure:
  - `{dataset_root}/images`
  - `{dataset_root}/labels`

Train/test folders are not required. Split is created in-memory using `train_split` and `split_seed`.

## Capture Controls
- `n`: next class
- `p`: previous class
- `space`: manual save for current class
- `q`: quit early

## Capture Command Variants
Per-class target:
```bash
uv run src/utils/collect_images.py --images-per-class 150 --camera-id 0
```

Fixed total target:
```bash
uv run src/utils/collect_images.py --total-images 150 --camera-id 0
```

## Notes
- If `mp.solutions` is missing in your MediaPipe build, the script falls back to OpenCV skin-based hand detection.
- More data per class generally improves recognition quality.

## Reference
- [DETR Colab walkthrough](https://colab.research.google.com/github/facebookresearch/detr/blob/colab/notebooks/detr_demo.ipynb)
