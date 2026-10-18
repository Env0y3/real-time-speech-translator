# Optional SenseVoice CPU setup

The GUI imports without FunASR. Actual SenseVoice recognition needs the optional
packages and model weights below. GUI/unit-test success alone does not prove
microphone capture, recognition, translation, or speech playback works.

## Validated platform and installation

Validated in a separate clean virtual environment on Debian 13 x86_64,
CPython 3.12.14. These exact CPU-wheel pins target Linux/Windows x86_64;
macOS/ARM and other Python versions were not validated. Use the official PyTorch
platform selector for those platforms, and adapt the `+cpu` requirements rather
than blindly installing the Linux instructions.

From the project root, with Python 3.12:

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install --no-cache-dir torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install --no-cache-dir -r requirements-models.txt
# Add the GUI only if needed:
python -m pip install --no-cache-dir -r requirements-gui.txt
python -m pip check
python -c "import torch, torchaudio, numpy; from funasr import AutoModel; print(torch.__version__, torchaudio.__version__, numpy.__version__, torch.cuda.is_available())"
```

Expected CPU versions are `2.6.0+cpu`, `2.6.0+cpu`, `2.2.6`, and `False`.
The first install must use the official **CPU index**, not ordinary PyPI.
The `+cpu` pins intentionally fail instead of silently selecting CUDA builds if
the CPU install was skipped. Torch and TorchAudio must have matching versions.
The earlier Torch 2.5.1 CPU attempt imported AutoModel but current Transformers
reported its PyTorch backend disabled (requires >=2.6); do not use that pair.

These pins are a tested top-level dependency set, not a full transitive lockfile.
`pip check` catches declared dependency conflicts but cannot replace the real
model checks below. The core plus model environment occupied about 1.7 GB before adding the GUI
during validation. Reserve several GB for environment, temporary downloads and weights.
The official model checkpoint is approximately 936 MB; CPU execution still needs
additional RAM. No CUDA toolkit, paid API, GPU, or access token is required for
this ASR setup. Translation providers have their own separate configuration.

## Model preparation

FunASR defaults to downloading `iic/SenseVoiceSmall` from ModelScope on the first
load. Use a persistent, writable location with adequate disk space:

```sh
export MODELSCOPE_CACHE="$HOME/.cache/speech-translator/modelscope"
export MODELSCOPE_HOME="$HOME/.cache/speech-translator/modelscope-sdk"
python -c "from asr.sensevoice_asr import load_sensevoice_model; load_sensevoice_model(); print('MODEL_INIT_OK')"
```

In restricted/container environments set both variables to writable directories.
`MODELSCOPE_CACHE` alone was insufficient: ModelScope tried to create SDK files
under read-only `~/.modelscope`. Setting `MODELSCOPE_HOME` fixed that observed
failure. The app uses CPU and does not enable remote model code.

Network access is needed for initial download, and the upstream model identifier
is mutable. For reproducible deployments, retain and version the validated model
snapshot; do not assume `master` will always have identical bytes. Keep cache paths
stable between setup and GUI launch. Launch the app from the same activated
virtual environment. Do not commit model weights or environment directories.

## Validation performed

- Clean CPU dependency install and `from funasr import AutoModel`: passed.
- Core `requirements.txt` and GUI `requirements-gui.txt` installed into the same
  model environment: passed; `pip check`: no broken requirements (PySide6 6.12.0).
- Real official ModelScope download (20 files) and CPU model construction: passed,
  checkpoint reported `All keys matched successfully`. First load including
  download took approximately 50 seconds in this environment, not a latency guarantee.
- Real generation on one second of synthetic float32 silence: completed and
  returned `嗯。` after metadata. This is a silence hallucination, not an accuracy
  test; the application's voice activity gating is still important.
- Real project wrapper (`load_sensevoice_model` + `recognize_sensevoice_sync`)
  on the downloaded official `example/zh.mp3` (5.616 seconds): passed. Output was
  `开放时间早上9点至下午5点。`; metadata tags were removed. Cached model
  construction took 16.4 seconds and sample inference 0.48 seconds with two CPU
  threads. This is a single prerecorded sample, not an ASR accuracy benchmark.
- No microphone hardware, desktop audio routing, live translation API, or speaker
  end-to-end test was performed by this dependency validation.

## Official references

- [PyTorch CPU install commands and matched versions](https://pytorch.org/get-started/previous-versions/)
- [FunASR source and setup](https://github.com/modelscope/FunASR)
- [SenseVoice official source](https://github.com/FunAudioLLM/SenseVoice)
- [Official model files and model-license](https://huggingface.co/FunAudioLLM/SenseVoiceSmall/tree/main)

Review the applicable model license before redistribution or deployment. The
repository and model license are not necessarily the same.
