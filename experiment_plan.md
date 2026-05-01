# Experiment Plan — Compressed Domain Object Detector

All phases use best settings from all previous phases.
★ = primary result for report

---

## Phase 0 — Baseline

**Goal:** confirm full pipeline works end-to-end, establish numbers to beat  
**Data:** 500 videos per class (2500 total), all 5 classes  
**Fixed settings:** MV + residuals, 64×64, scales [8,16,32], 300 epochs, lr 4e-4, wd 1e-4, warmup 10

| ID | Name | epochs | batch | lr | wd | Notes |
|----|------|--------|-------|----|----|-------|
| P0 ★ | Baseline | 300 | 256 | 4e-4 | 1e-4 | reference point for all later phases |

**Config changes from default:**
```yaml
max_files_per_class: 500
epochs:        300
batch_size:    256
lr:            4.0e-4
weight_decay:  1e-4
warmup_epochs: 10
use_motionvectors: true
use_residuals:     true
scales:        [8, 16, 32]
frame_h:       64
frame_w:       64
```

---

## Phase 1 — Hyperparameter Search (batch & LR)

**Goal:** find best batch size and learning rate cheaply before heavy runs  
**Data:** 100 videos per class (500 total) — fast iteration  
**Fixed settings:** MV + residuals, 64×64, scales [8,16,32], 100 epochs, wd 1e-4

| ID | batch | lr | Notes |
|----|-------|----|-------|
| P1-A | 256 | 4e-4 | same as P0 small data — reference |
| P1-B | 64 | 1e-4 | DETR paper original settings |
| P1-C | 512 | 8e-4 | lr scaled linearly with batch |
| P1-D | 1024 | 1.6e-3 | max VRAM — lr scaled linearly |
| P1-E | 256 | 1e-4 | lower lr, same batch as reference |

Linear scaling rule: `lr = 1e-4 * (batch / 64)`

**Config changes per run (everything else fixed):**
```yaml
max_files_per_class: 100
epochs:     100
warmup_epochs: 5
batch_size: 256   # change per run
lr:         4e-4  # change per run
```

---

## Phase 2 — Architecture Search

**Goal:** find best model size before drawing conclusions about inputs  
**Data:** 500 videos per class (2500 total)  
**Fixed settings:** best hp from phase 1, MV + residuals, 64×64, scales [8,16,32], 300 epochs

| ID | hidden_dim | enc layers | dec layers | params (approx) | Notes |
|----|-----------|------------|------------|-----------------|-------|
| P2-A | 256 | 4 | 4 | ~8M | baseline architecture |
| P2-B | 128 | 4 | 4 | ~2M | smaller — faster, less expressive |
| P2-C | 512 | 4 | 4 | ~30M | larger hidden — more capacity |
| P2-D | 256 | 2 | 2 | ~4M | shallower transformer |
| P2-E | 256 | 6 | 6 | ~12M | deeper transformer |

**Config changes per run:**
```yaml
hidden_dim:         256   # change per run
num_encoder_layers: 4     # change per run
num_decoder_layers: 4     # change per run
```

---

## Phase 3 — Input Ablation ★ (main contribution)

**Goal:** prove detection is possible in compressed domain — core paper result  
**Data:** 500 videos per class (2500 total)  
**Fixed settings:** best hp from phase 1, best architecture from phase 2, 300 epochs

| ID | Motion vectors | Residuals | Notes |
|----|---------------|-----------|-------|
| P3-A | ✓ | ✓ | full input — re-use P0 checkpoint if architecture unchanged |
| P3-B ★ | ✓ | ✗ | MV only — compressed domain, no RGB decoding ever |
| P3-C | ✗ | ✓ | residual only — closest to standard RGB baseline |

P3-B is the headline result: if detection works with zero RGB decoding, the compressed domain thesis is proven.

**Config changes per run:**
```yaml
use_motionvectors: true   # false for P3-C
use_residuals:     true   # false for P3-B
```

---

## Phase 4 — Resolution & Scale

**Goal:** spatial resolution impact on localization quality  
**Data:** 500 videos per class (2500 total)  
**Fixed settings:** best hp from phase 1, best architecture from phase 2, best inputs from phase 3, 300 epochs

| ID | frame_h/w | scales | tokens/frame | Notes |
|----|-----------|--------|--------------|-------|
| P4-A | 64×64 | [8, 16, 32] | 64 | baseline — re-use P0 |
| P4-B | 64×64 | [4, 8, 16] | 256 | finer scales, same resolution |
| P4-C | 128×128 | [8, 16, 32] | 256 | higher res, same scales |
| P4-D | 128×128 | [4, 8, 16] | 1024 | both — most spatial detail |
| P4-E | 64×64 | [16, 32, 64] | 16 | coarser — lower bound check |

Note: P4-D may need `batch_size` halved if OOM at 128×128.

**Config changes per run:**
```yaml
frame_h:  64    # 128 for P4-C and P4-D
frame_w:  64    # 128 for P4-C and P4-D
scales:   [8, 16, 32]   # change per run
# if OOM at 128x128: halve batch_size and adjust lr proportionally
```

---

## Summary — Recommended order

| Phase | Focus | Data | Runs | Purpose |
|-------|-------|------|------|---------|
| P0 | Baseline | 2500 | 1 | confirm pipeline |
| P1 | Batch + LR | 500 | 5 | cheap tuning |
| P2 | Architecture | 2500 | 5 | best model before ablation |
| P3 ★ | Input ablation | 2500 | 3 | main paper result |
| P4 | Resolution | 2500 | 5 | spatial analysis |

**Total runs: 19**  
**Key metric to report:** mAP50, IoU, val loss across all phases. For P3, also report whether MV-only achieves non-zero mAP50 — that is the thesis.
