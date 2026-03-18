from ultralytics import YOLOE
from sentence_transformers import SentenceTransformer, util
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yoloe-26n-seg.pt"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

ymodel = YOLOE(str(MODEL_PATH))
model  = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
]

print("Pre-computing COCO embeddings...")
object_embeddings = model.encode(COCO_CLASSES, convert_to_tensor=True,
                                 show_progress_bar=False)
print("Ready.\n")

# ── Rank all 80 classes by prompt similarity ──────────────────────────────────
def get_ranked_classes(prompt: str) -> list[str]:
    prompt_emb    = model.encode(prompt, convert_to_tensor=True,
                                 show_progress_bar=False)
    cosine_scores = util.cos_sim(prompt_emb, object_embeddings)[0]
    ranked_idx    = torch.argsort(cosine_scores, descending=True)
    return [COCO_CLASSES[i] for i in ranked_idx]

# ── Batch search with early exit ──────────────────────────────────────────────
def find_best(image_path: str, prompt: str,
              batch_size: int = 10) -> dict | None:

    ranked_classes = get_ranked_classes(prompt)
    n_batches      = (len(ranked_classes) + batch_size - 1) // batch_size

    print(f"Prompt:   '{prompt}'")
    print(f"Image:    '{image_path}'")
    print(f"Strategy: batches of {batch_size} · max {n_batches} re-params\n")

    for batch_idx, batch_start in enumerate(range(0, len(ranked_classes),
                                                   batch_size)):
        batch = ranked_classes[batch_start : batch_start + batch_size]
        print(f"Batch {batch_idx+1:02d}: {batch}")

        # ── Single re-param + single inference for this batch ─────────────────
        ymodel.set_classes(batch, ymodel.get_text_pe(batch))
        results = ymodel.predict(image_path, conf=0.25, verbose=False)

        # ── Collect detections ────────────────────────────────────────────────
        batch_detections = []
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            names = results[0].names
            for box in results[0].boxes:
                label    = names[int(box.cls)]
                conf     = float(box.conf)
                rank_idx = ranked_classes.index(label)
                score    = (1 / (rank_idx + 1)) * conf
                bbox     = box.xyxy[0].cpu().numpy().astype(int)
                batch_detections.append({
                    "label":    label,
                    "conf":     conf,
                    "rank_idx": rank_idx,
                    "score":    score,
                    "bbox":     bbox,
                })
                print(f"         found: {label:20s}  "
                      f"rank:{rank_idx+1:02d}  conf:{conf:.3f}  score:{score:.4f}")

        # ── Early exit ────────────────────────────────────────────────────────
        if batch_detections:
            best = max(batch_detections, key=lambda d: d["score"])
            skipped = n_batches - (batch_idx + 1)

            print(f"\n  Stopped at batch {batch_idx+1}/{n_batches} "
                  f"— skipped {skipped} remaining batches")
            print(f"\n{'='*50}")
            print(f"Best: {best['label'].upper()}")
            print(f"  Score: {best['score']:.4f}  "
                  f"Conf:  {best['conf']:.3f}  "
                  f"Rank:  #{best['rank_idx']+1}/80")
            print(f"  BBox:  {best['bbox']}")
            print(f"{'='*50}\n")
            return best

        print(f"         no detections\n")

    print("No detections found in any batch.")
    return None

# ── Draw result ───────────────────────────────────────────────────────────────
def draw_and_show(image_path: str, best: dict, prompt: str):
    import cv2
    frame = cv2.imread(image_path)
    x1, y1, x2, y2 = best["bbox"]
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
    tag = f"{best['label']}  {best['score']:.2f}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    cv2.rectangle(frame, (x1, y1-th-12), (x1+tw+8, y1), (0, 255, 0), -1)
    cv2.putText(frame, tag, (x1+4, y1-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)
    cv2.putText(frame, f"Query: {prompt}",
                (10, frame.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image_stem = Path(image_path).stem
    safe_prompt = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in prompt
    ).strip("_") or "query"
    out = OUTPUT_DIR / f"{image_stem}_result-{safe_prompt}.jpg"
    cv2.imwrite(str(out), frame)
    print(f"Saved: {out}")
    try:
        from PIL import Image
        Image.open(str(out)).show()
    except Exception:
        pass

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = IMAGE_DIR / "chair.jpeg"
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image file not found: {IMAGE_PATH}\n"
            f"Place test images under: {IMAGE_DIR}"
        )

    tests = [
        (str(IMAGE_PATH), "Something to eat"),
        (str(IMAGE_PATH), "Place where you can sit"),
        (str(IMAGE_PATH), "Appliance to heat up food"),
    ]

    for image_path, prompt in tests:
        best = find_best(image_path, prompt, batch_size=10)
        if best:
            draw_and_show(image_path, best, prompt)
        print()