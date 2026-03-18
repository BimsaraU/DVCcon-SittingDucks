from ultralytics import YOLOE
from sentence_transformers import SentenceTransformer, util
import torch
import time
import psutil
import os
import gc
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "yoloe-26n-seg.pt"
IMAGE_DIR = BASE_DIR / "images"
OUTPUT_DIR = BASE_DIR / "outputs"

PROC = psutil.Process(os.getpid())

def snapshot() -> dict:
    mem = PROC.memory_info()
    return {
        "rss_mb":  mem.rss / 1024**2,
        "cpu_pct": PROC.cpu_percent(interval=0.05),
    }

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

# ── Rank ──────────────────────────────────────────────────────────────────────
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
    print(f"  {'Batch':>5}  {'Reparam':>9}  {'Infer':>7}  "
          f"{'RAM':>7}  {'CPU%':>5}  {'Dets':>4}  Status")
    print(f"  {'─'*60}")

    t_wall_start = time.perf_counter()
    gc.collect()

    for batch_idx, batch_start in enumerate(range(0, len(ranked_classes),
                                                   batch_size)):
        batch = ranked_classes[batch_start : batch_start + batch_size]

        # ── Re-param ──────────────────────────────────────────────────────────
        s0 = snapshot()
        t0 = time.perf_counter()
        ymodel.set_classes(batch, ymodel.get_text_pe(batch))
        reparam_ms = (time.perf_counter() - t0) * 1000

        # ── Inference ─────────────────────────────────────────────────────────
        t1      = time.perf_counter()
        results = ymodel.predict(image_path, conf=0.25, verbose=False)
        infer_ms = (time.perf_counter() - t1) * 1000
        s1 = snapshot()

        # ── Collect detections from this batch ────────────────────────────────
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

        n_dets = len(batch_detections)
        status = "FOUND — stopping" if n_dets > 0 else "no match"

        print(f"  {batch_idx+1:>5}  {reparam_ms:>8.1f}ms"
              f"  {infer_ms:>6.1f}ms"
              f"  {s1['rss_mb']:>6.1f}MB"
              f"  {s1['cpu_pct']:>5.1f}"
              f"  {n_dets:>4}  {status}")

        # ── Early exit — found something in this batch ────────────────────────
        if batch_detections:
            best      = max(batch_detections, key=lambda d: d["score"])
            wall_ms   = (time.perf_counter() - t_wall_start) * 1000
            batches_run = batch_idx + 1
            batches_skipped = n_batches - batches_run

            print(f"\n  Early exit at batch {batches_run}/{n_batches} "
                  f"— skipped {batches_skipped} batches "
                  f"({batches_skipped * batch_size} classes unscanned)")
            print(f"  Wall time: {wall_ms:.1f}ms  "
                  f"({batches_run} re-params instead of {n_batches})")

            print(f"\n{'='*50}")
            print(f"Best: {best['label'].upper()}")
            print(f"  Score: {best['score']:.4f}  "
                  f"Conf: {best['conf']:.3f}  "
                  f"Rank: #{best['rank_idx']+1}/80")
            print(f"  BBox: {best['bbox']}")
            print(f"{'='*50}\n")

            # Print what was in this batch so you can see what competed
            if len(batch_detections) > 1:
                print(f"  All detections in winning batch:")
                for d in sorted(batch_detections,
                                key=lambda x: x["score"], reverse=True):
                    marker = " <-- best" if d is best else ""
                    print(f"    {d['label']:20s}  "
                          f"score:{d['score']:.4f}  "
                          f"conf:{d['conf']:.3f}{marker}")
                print()

            return best

    # ── No detections in any batch ────────────────────────────────────────────
    wall_ms = (time.perf_counter() - t_wall_start) * 1000
    print(f"\n  No detections found after {n_batches} batches "
          f"({wall_ms:.1f}ms)")
    return None

# ── Draw ──────────────────────────────────────────────────────────────────────
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
    out = OUTPUT_DIR / f"{image_stem}_result-{prompt}.jpg"
    cv2.imwrite(str(out), frame)
    print(f"Saved: {out}")
    try:
        from PIL import Image
        Image.open(str(out)).show()
    except Exception:
        pass

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    IMAGE_PATH = IMAGE_DIR / "000000004177.jpg"
    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Image file not found: {IMAGE_PATH}\n"
            f"Place test images under: {IMAGE_DIR}"
        )

    tests = [
        (str(IMAGE_PATH), "Something to drink tea from"),
        (str(IMAGE_PATH), "Appliance to heat up food")
    ]

    for image_path, prompt in tests:
        best = find_best(image_path, prompt, batch_size=10)
        if best:
            draw_and_show(image_path, best, prompt)
        print()
    termin = input("Press 0 to exit..")
    while termin != "0":
        termin = input("Press 0 to exit..")