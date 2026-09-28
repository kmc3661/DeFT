# Third-party notices

- The internal `vispruner_qwen_ov` compatibility backend and benchmark loaders
  were extracted from the VisPruner-based research tree distributed under
  Apache License 2.0. The license text is retained in `LICENSE`. Public upstream:
  https://github.com/anonymous-4869/VisPruner (URL recorded in the source README). The extracted research implementation
  includes subsequent local modifications; it is not claimed to be upstream
  VisPruner or its original algorithm.
- `_vendor/lmms_qwen3_vl.py` and `_vendor/lmms_llava_onevision1_5.py` are copied
  lmms-eval model wrappers. The MIT license and original copyright notice are
  retained in `licenses/lmms-eval-LICENSE`. Upstream:
  https://github.com/EvolvingLMMs-Lab/lmms-eval .
- Hugging Face Transformers, PyTorch, Qwen-VL utilities, FlashAttention and
  pycocoevalcap are external dependencies, not relicensed by this package.
- Checkpoint-provided OV code and all model weights/datasets remain under their
  respective upstream licenses and are not included in the archive.

`docs/source_manifest.json` records hashes of the copied source files before
packaging changes. The principal packaging change to the model backend replaces
host-specific model-wrapper imports with the included wrappers. Release archives
contain their own final-file hash manifest. Required upstream attribution is retained. Third-party weights and datasets
are subject to their own licenses.
