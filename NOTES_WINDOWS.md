# GVHMR — Windows 11 + Python 3.11 + RTX 5070 (cu128) install notes

## What differs from upstream requirements.txt

| Upstream pin | What was installed instead | Why |
|---|---|---|
| torch 2.3.0+cu121 / torchvision 0.18.0+cu121 | torch 2.11.0+cu128, torchvision 0.26.0+cu128 (`https://download.pytorch.org/whl/cu128`) | RTX 5070 needs cu128; cu121 wheels are ancient |
| pytorch3d 0.7.6 linux wheel | `pytorch3d-0.7.9+d9839a9pt2.11.0cu128-cp311-cp311-win_amd64.whl` from https://github.com/MiroPsota/torch_packages_builder/releases | official wheel is linux-only; this community build has the compiled `_C` ext for pt2.11.0+cu128/cp311/win |
| numpy==1.23.5 | numpy 2.4.6 (unpinned) | torch 2.11 / pycolmap 4.x wheels are numpy-2 ABI |
| ultralytics==8.2.42 | ultralytics 8.4.194 (unpinned) | torch>=2.6 defaults `torch.load(weights_only=True)`; old ultralytics can't load yolov8x.pt |
| cython_bbox, lapx | — skipped — | not imported anywhere in `hmr4d/` or `tools/` (ultralytics vendors its own lap) |
| chumpy | chumpy 0.70 via `pip install chumpy --no-build-isolation` + patch below | SMPL_*.pkl are pickled chumpy.Ch objects |
| (missing) | yacs | imported by `hmr4d/network/hmr2/hmr2.py` but absent from requirements.txt |
| DPVO / third-party | — skipped — | SimpleVO is the default (`--use_dpvo` not passed) |

## chumpy patches applied in `.venv/Lib/site-packages/chumpy/`

1. `ch.py`: `inspect.getargspec(` → `inspect.getfullargspec(` (getargspec removed in py3.11)
2. `__init__.py` line 11: `from numpy import bool, int, float, complex, object, unicode, str, nan, inf` →
   ```python
   from numpy import bool_ as bool, nan, inf
   import builtins as _builtins
   int, float, complex, object, unicode, str = _builtins.int, _builtins.float, _builtins.complex, _builtins.object, _builtins.str, _builtins.str
   ```

## Source patch (repo)

`hmr4d/utils/preproc/relpose/solver_two_view.py` `PycolmapRansacTwoViewGeometrySolver.solve()`:
upstream crashes when `estimate_calibrated_two_view_geometry` returns `None`, or returns a
`TwoViewGeometry` with `cam2_from_cam1=None` (config=WATERMARK/PLANAR/degenerate — ~1/3 of pairs
on the bundled tennis.mp4). Patched to fall back to an identity delta (no measured camera
motion → cam_angvel ≈ 0, the model's static-cam mode).

## Checkpoints (inputs/checkpoints/)

Downloaded via `gdown --folder https://drive.google.com/drive/folders/1eebJ13FUEXrKBawHpJroW0sNSxLjh9xD`
- gvhmr/gvhmr_siga24_release.ckpt (156M) ✓
- hmr2/epoch=10-step=25000.ckpt (2.6G) ✓
- vitpose/vitpose-h-multi-coco.pth (2.4G) ✓
- yolo/yolov8x.pt (131M) ✓
- dpvo/dpvo.pth — intentionally skipped
- body_models/smplx/*.npz + body_models/smpl/*.pkl — copied from lora3d

## Verified (CPU-only, no GPU work queued)

- `import hmr4d`; `from hmr4d.utils.preproc import Tracker, Extractor, VitPoseExtractor, SimpleVO`
- hydra `compose(config_name="demo")` + `instantiate(cfg.model)` → DemoPL (40.85M params) and
  `load_pretrained_model('inputs/checkpoints/gvhmr/gvhmr_siga24_release.ckpt')` — clean load
- `load_hmr2()` builds + loads the 2.6G hmr2 ckpt on CPU (670M params)
- `Tracker()` → ultralytics loads `yolov8x.pt`
- `make_smplx("supermotion")` → SMPLX fwd (10475 verts); `make_smplx("smpl")` → SMPL fwd (6890 verts)
- `SimpleVO.compute()` on `docs/example_video/tennis.mp4` → (312,4,4) trajectory, end-to-end

## Gotchas when actually running

- `OMP: Error #135: Cannot set thread affinity mask` can appear under MSYS/bash on this machine.
  Workaround: prefix the command with `KMP_AFFINITY=disabled` (env var), or run from cmd/PowerShell.
- Almost every pipeline stage calls `.cuda()` — the RTX 5070 (sm_120, cu128) is required at runtime.

## Run the demo

```bash
cd /c/Users/jrdev/Documents/GitHub/GVHMR
KMP_AFFINITY=disabled .venv/Scripts/python.exe tools/demo/demo.py --video docs/example_video/tennis.mp4 --verbose
# static camera videos: add -s (skips VO entirely)
# results: outputs/demo/<video_name>/{hmr4d_results.pt, 1_incam.mp4, 2_global.mp4, *_incam_global_horiz.mp4}
```
