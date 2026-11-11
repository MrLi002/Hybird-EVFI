# Third-party source attribution

This research implementation includes or adapts code from the following fixed revisions.
It does not assign a new blanket license to upstream code.

| Source | Revision | Files and use |
| --- | --- | --- |
| [Sudadaaaa/OF-AF](https://github.com/Sudadaaaa/OF-AF) | `f7791a6aad64cec8af37f0fb5d90002ace1a1a87` | `models/encoder.py`, `models/BiFEB.py`, ResBlocks, BiW structure and parameter names; unchanged `models/softsplat_cuda.py`; original reference modules under `tests/reference_ofaf/` |
| [intelpro/CBMNet](https://github.com/intelpro/CBMNet) | `248405c313ce29f87ad42873df46792c76f55d1a` | `models/cbm_submodules.py`, extracted components in `models/cbm_modules.py`, refinement design and second-order smoothness formula |
| [yuhan0802/DSER-pp](https://github.com/yuhan0802/DSER-pp) | `786bd6d0dbf08eac1484f78044952d24c08d262b` | BASR patch retrieval design, independently adapted with separate masks, safe empty-key handling and complete residual output gating |

The source document records OF-AF as having no top-level license, CBMNet as research/educational use, and DSER-pp as Apache-2.0. Preserve upstream attribution and inspect the relevant authors' terms before redistributing derived code. This package is supplied for the requested research implementation; no upstream pretrained weights or datasets are bundled.

The CUDA softsplat implementation retains its source header and code. Standard PyTorch, einops, NumPy, Pillow, scikit-image and LPIPS dependencies retain their own licenses.

