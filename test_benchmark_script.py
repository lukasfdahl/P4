"""
benchmark.py  —  run this file directly to benchmark all four models.

Edit the CONFIG block below, then just press Run.

Ground truth is loaded from a sidecar .npz with the same stem as the video
(the format download.py produces).  If none exists, metrics are 0 but
timing still works fine.
"""

import os, time, json, random
import numpy as np
import torch
import cv_reader
import ultralytics

from torch.utils.data import DataLoader, Dataset

from model         import ObjectDetector
from faster_rcnn   import FasterRCNNDetector
from data_helpers  import decode_mv, decode_residuals
from eval_framwork import BoundingBox, Prediction, evaluate
from helpers       import load_checkpoint


# config
VIDEOS = [
    "videos/clip1.mp4",
    "videos/clip2.mp4",
]

OUR_MODEL_CKPT       = "checkpoints/best.pt"
RCNN_SCRATCH_CKPT    = None                     # None → random weights
RCNN_PRETRAINED_CKPT = None                     # None → COCO weights
YOLO_CKPT            = "yolo11n.pt"            # downloads automatically if missing

CONFIG_YAML          = "config.yaml"
OUTPUT_JSON          = "benchmark_results.json"

# Model / data settings — should match your training config
NUM_CLASSES    = 23
NUM_QUERIES    = 10
CLIP_LENGTH    = 5
FRAME_H        = 64
FRAME_W        = 64
BASE_MV_SCALE  = 16
CONF_THRESHOLD = 0.1
BATCH_SIZE     = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Timer
class Timer:
    def __init__(self):
        self._t      = {}
        self._totals = {}
        self._counts = {}

    def start(self, stage):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self._t[stage] = time.perf_counter()

    def stop(self, stage):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - self._t.pop(stage)
        self._totals[stage] = self._totals.get(stage, 0.0) + elapsed
        self._counts[stage] = self._counts.get(stage, 0)   + 1
        return elapsed

    def total(self, s): return self._totals.get(s, 0.0)
    def mean(self, s):
        n = self._counts.get(s, 0)
        return self._totals.get(s, 0.0) / n if n else 0.0


# Video decode

class VideoArrays:
    def __init__(self, mv, res, frame_types):
        self.motion_vectors = mv
        self.residuals      = res
        self.frame_types    = frame_types
        self.n_frames       = len(frame_types)


def decode_video(path):
    t0  = time.perf_counter()
    raw = cv_reader.read(path, width=FRAME_W, height=FRAME_H)
    elapsed = time.perf_counter() - t0
    return VideoArrays(
        raw["motion_vectors"],
        raw["residuals"],
        [str(ft) for ft in raw["frame_types"]],
    ), elapsed


# Sliding-window dataset

class ClipDataset(Dataset):
    def __init__(self, arrays):
        self.arrays = arrays
        self.h_tok  = FRAME_H // BASE_MV_SCALE
        self.w_tok  = FRAME_W // BASE_MV_SCALE
        self.starts = list(range(0, arrays.n_frames - CLIP_LENGTH + 1))

    def __len__(self): return len(self.starts)

    def __getitem__(self, idx):
        s  = self.starts[idx]
        sl = slice(s, s + CLIP_LENGTH)
        ft = self.arrays.frame_types[sl]
        return {
            "start":          s,
            "iframe_mask":    torch.tensor([f == "I" for f in ft], dtype=torch.bool),
            "motion_vectors": decode_mv(self.arrays.motion_vectors[sl], self.h_tok, self.w_tok),
            "residuals":      decode_residuals(self.arrays.residuals[sl], FRAME_H, FRAME_W),
        }


def collate(batch):
    return {
        "starts":         [b["start"] for b in batch],
        "iframe_mask":    torch.stack([b["iframe_mask"]    for b in batch]),
        "motion_vectors": torch.stack([b["motion_vectors"] for b in batch]),
        "residuals":      torch.stack([b["residuals"]      for b in batch]),
    }


# Ground truth loader

def load_gt(video_path, n_frames):
    npz = os.path.splitext(video_path)[0] + ".npz"
    if not os.path.exists(npz):
        return [[] for _ in range(n_frames)], False
    data = np.load(npz)
    boxes, classes = data["boxes"], data["true_class"]
    gt = []
    for i in range(n_frames):
        cls = int(classes[i]) if i < len(classes) else -1
        if cls != -1 and i < len(boxes):
            b = boxes[i]
            gt.append([BoundingBox(float(b[0]), float(b[1]), float(b[2]), float(b[3]), cls)])
        else:
            gt.append([])
    return gt, True


#  Inference runners 
@torch.no_grad()
def run_compressed(model, arrays):
    timer       = Timer()
    n           = arrays.n_frames
    best_conf   = [-1.0] * n
    frame_preds = [[] for _ in range(n)]

    loader = DataLoader(ClipDataset(arrays), batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=0, collate_fn=collate)

    for batch in loader:
        timer.start("preprocess")
        mv   = batch["motion_vectors"].to(DEVICE, non_blocking=True)
        res  = batch["residuals"].to(DEVICE,      non_blocking=True)
        mask = batch["iframe_mask"].to(DEVICE,    non_blocking=True)
        timer.stop("preprocess")

        timer.start("inference")
        boxes_out, logits_out = model(mv, res, mask)
        timer.stop("inference")

        timer.start("postprocess")
        B, T, Q, _ = boxes_out.shape
        probs = torch.softmax(logits_out, dim=-1)[..., :NUM_CLASSES]
        confs_t, cls_t = probs.max(dim=-1)

        b_np = torch.stack([
            torch.minimum(boxes_out[..., 0], boxes_out[..., 1]),
            torch.maximum(boxes_out[..., 0], boxes_out[..., 1]),
            torch.minimum(boxes_out[..., 2], boxes_out[..., 3]),
            torch.maximum(boxes_out[..., 2], boxes_out[..., 3]),
        ], dim=-1).cpu().numpy()
        c_np = confs_t.cpu().numpy()
        k_np = cls_t.cpu().numpy()

        for b_i, clip_start in enumerate(batch["starts"]):
            for t in range(T):
                fi = clip_start + t
                if fi >= n: continue
                for q in range(Q):
                    c = float(c_np[b_i, t, q])
                    if c < CONF_THRESHOLD or c <= best_conf[fi]: continue
                    best_conf[fi] = c
                    frame_preds[fi] = [Prediction(
                        float(b_np[b_i,t,q,0]), float(b_np[b_i,t,q,1]),
                        float(b_np[b_i,t,q,2]), float(b_np[b_i,t,q,3]),
                        int(k_np[b_i,t,q]), c,
                    )]
        timer.stop("postprocess")

    return frame_preds, timer


def run_yolo(model, arrays):
    timer       = Timer()
    frame_preds = []

    for i in range(arrays.n_frames):
        timer.start("preprocess")
        frame = arrays.residuals[i]          # uint8 [H, W, 3] — raw pixels, no MV
        timer.stop("preprocess")

        timer.start("inference")
        results = model.predict(frame, verbose=False, conf=CONF_THRESHOLD)
        timer.stop("inference")

        timer.start("postprocess")
        preds = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0: continue
            xyyn   = r.boxes.xyxyn.cpu().numpy()   # normalised xmin,ymin,xmax,ymax
            confs  = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            for j in range(len(confs)):
                xmin, ymin, xmax, ymax = xyyn[j]
                # convert to our [xmin, xmax, ymin, ymax] format
                preds.append(Prediction(xmin, xmax, ymin, ymax, labels[j], float(confs[j])))
        frame_preds.append(preds)
        timer.stop("postprocess")

    return frame_preds, timer


# Model loading

def load_our_model():
    cfg = {}
    if os.path.exists(CONFIG_YAML):
        import yaml
        with open(CONFIG_YAML) as f:
            cfg = yaml.safe_load(f).get("model", {})
    m = ObjectDetector(
        num_classes        = cfg.get("num_classes",        NUM_CLASSES),
        scales             = cfg.get("scales",             [8, 16, 32]),
        base_mv_scale      = cfg.get("base_mv_scale",      BASE_MV_SCALE),
        clip_length        = cfg.get("clip_length",        CLIP_LENGTH),
        expected_h_tokens  = cfg.get("expected_h_tokens",  FRAME_H // 8),
        expected_w_tokens  = cfg.get("expected_w_tokens",  FRAME_W // 8),
        hidden_dim         = cfg.get("hidden_dim",         256),
        num_heads          = cfg.get("num_heads",          8),
        num_encoder_layers = cfg.get("num_encoder_layers", 4),
        num_decoder_layers = cfg.get("num_decoder_layers", 4),
        num_queries        = cfg.get("num_queries",        NUM_QUERIES),
    ).to(DEVICE)
    if OUR_MODEL_CKPT and os.path.exists(OUR_MODEL_CKPT):
        load_checkpoint(OUR_MODEL_CKPT, m)
    return m.eval()


def load_rcnn(pretrained):
    ckpt = RCNN_PRETRAINED_CKPT if pretrained else RCNN_SCRATCH_CKPT
    m = FasterRCNNDetector(NUM_CLASSES, NUM_QUERIES, pretrained=pretrained).to(DEVICE)
    if ckpt and os.path.exists(ckpt):
        load_checkpoint(ckpt, m)
    return m.eval()


def load_yolo():
    return ultralytics.YOLO(YOLO_CKPT or "yolo11n.pt")


# Console output
def print_timing(name, timer, n_frames):
    total_s = sum(timer._totals.values())
    print(f"\n    [{name}]")
    for stage in ("preprocess", "inference", "postprocess"):
        t = timer.total(stage)
        print(f"      {stage:<14} {t:6.3f}s   ({1000*t/n_frames:6.2f} ms/frame)")
    print(f"      {'TOTAL':<14} {total_s:6.3f}s   ({1000*total_s/n_frames:6.2f} ms/frame)"
          f"  [{n_frames/total_s:.1f} fps]")


# Main

def main():
    print(f"\n[bench] Device: {DEVICE}")
    print("[bench] Loading models...")

    models = [
        ("ObjectDetector",        load_our_model(),  run_compressed),
        ("FasterRCNN_scratch",    load_rcnn(False),  run_compressed),
        ("FasterRCNN_pretrained", load_rcnn(True),   run_compressed),
        ("YOLO",                  load_yolo(),       run_yolo),
    ]

    all_results = {}

    for video_path in VIDEOS:
        print(f"\n{'='*60}")
        print(f"[bench] {video_path}")
        print(f"{'='*60}")

        print("  Decoding video...")
        arrays, decode_s = decode_video(video_path)
        print(f"  {arrays.n_frames} frames  |  decode: {decode_s:.3f}s "
              f"({1000*decode_s/max(arrays.n_frames,1):.2f} ms/frame)")

        gt, has_gt = load_gt(video_path, arrays.n_frames)
        print(f"  GT: {'loaded' if has_gt else 'not found — metrics will be 0'}")

        # Shuffle model order each run — reduces warm-up and thermal bias
        order = list(range(len(models)))
        random.shuffle(order)
        print(f"\n  Run order: {' → '.join(models[i][0] for i in order)}")

        video_entry = {"n_frames": arrays.n_frames, "decode_s": decode_s, "has_gt": has_gt}

        for i in order:
            name, model, runner = models[i]
            preds, timer = runner(model, arrays)
            print_timing(name, timer, arrays.n_frames)

            lat = timer.total("inference") / max(arrays.n_frames, 1)
            m   = evaluate(preds, gt, latency=lat)
            print(f"      metrics  acc={m.accuracy:.3f}  mAP50={m.mAP_50:.3f}"
                  f"  IoU={m.iou:.3f}  prec={m.weighted_precision:.3f}")

            video_entry[name] = {
                "timing": {s: {"total_s": timer.total(s), "mean_ms": timer.mean(s)*1000}
                           for s in ("preprocess", "inference", "postprocess")},
                "metrics": {
                    "accuracy": m.accuracy, "iou": m.iou,
                    "mAP_50": m.mAP_50, "mAP_95": m.mAP_95,
                    "weighted_precision": m.weighted_precision,
                    "latency_per_frame": m.latency,
                },
            }

        all_results[os.path.basename(video_path)] = video_entry

    # Summary across videos
    if len(VIDEOS) > 1:
        print(f"\n{'='*60}")
        print("[bench] SUMMARY (mean across all videos)")
        print(f"  {'Model':<28} {'Acc':>5} {'mAP50':>6} {'IoU':>5} {'ms/frame':>9}")
        print(f"  {'-'*53}")
        for name, _, _ in models:
            rows = [r[name] for r in all_results.values() if name in r]
            if not rows: continue
            n    = len(rows)
            acc  = sum(r["metrics"]["accuracy"]          for r in rows) / n
            mp   = sum(r["metrics"]["mAP_50"]            for r in rows) / n
            iou  = sum(r["metrics"]["iou"]               for r in rows) / n
            lat  = sum(r["metrics"]["latency_per_frame"] for r in rows) / n
            print(f"  {name:<28} {acc:>5.3f} {mp:>6.3f} {iou:>5.3f} {lat*1000:>9.2f}")

    with open(OUTPUT_JSON, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[bench] Results saved → {OUTPUT_JSON}")


if __name__ == "__main__":
    main()