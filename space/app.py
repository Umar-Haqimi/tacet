"""Tacet remote separator: vocals/instrumental separation on ZeroGPU.

One API endpoint, `separate`, takes the audio bytes Tacet already has (webm/opus,
mp4/aac, anything ffmpeg reads) and returns two Opus files: vocals and instrumental.

Pipeline:
  1. CPU: ffmpeg decodes the upload to 44.1 kHz stereo float32 WAV.
  2. GPU: audio-separator runs a BS/Mel-Band RoFormer model on it. Float input keeps
     float output.
  3. CPU: ffmpeg encodes both stems to Opus for a small download.

audio-separator rescales any stem whose peak passes 1.0, independently per stem, which
would shift the stems' levels against each other and against the original. So the mix
goes in 6 dB down (no stem gets near 1.0) and the stems come back up 6 dB on encode.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import gradio as gr
import spaces

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "model_bs_roformer_ep_317_sdr_12.9755.ckpt")
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/audio-separator-models")
OPUS_BITRATE = os.environ.get("OPUS_BITRATE", "192k")
MAX_SECONDS = float(os.environ.get("MAX_SECONDS", "900"))  # refuse anything longer than 15 min
# GPU seconds budget: fixed model-load overhead plus a per-audio-second cost, capped.
GPU_BASE_SECONDS = float(os.environ.get("GPU_BASE_SECONDS", "20"))
GPU_SECONDS_PER_AUDIO_SECOND = float(os.environ.get("GPU_SECONDS_PER_AUDIO_SECOND", "0.15"))
GPU_MAX_SECONDS = float(os.environ.get("GPU_MAX_SECONDS", "180"))
MOCK = os.environ.get("TACET_MOCK") == "1"  # local testing without a model or GPU

SAMPLE_RATE = 44_100
HEADROOM = 0.5  # -6 dB going in, undone on the way out

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("tacet-space")


# -- ffmpeg helpers ------------------------------------------------------------


def run(cmd: list[str]) -> str:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise gr.Error(f"{cmd[0]} failed: {result.stderr.strip()[-400:]}")
    return result.stdout


def probe_seconds(path: str) -> float:
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", path])
    try:
        return float(json.loads(out)["format"]["duration"])
    except (KeyError, ValueError, TypeError):
        return 0.0


def decode_to_float_wav(src: str, dst: str) -> None:
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-vn",
         "-ac", "2", "-ar", str(SAMPLE_RATE), "-af", f"volume={HEADROOM}", "-c:a", "pcm_f32le", dst])


def encode_opus(src: str, dst: str) -> None:
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src, "-af", f"volume={1 / HEADROOM}",
         "-c:a", "libopus", "-b:a", OPUS_BITRATE, "-vbr", "on", "-application", "audio", dst])


# -- Model -------------------------------------------------------------------------


def prefetch_model() -> None:
    """Download weights on CPU at startup so GPU time is never spent downloading."""
    if MOCK:
        return
    from audio_separator.separator import Separator

    Separator(model_file_dir=MODEL_DIR, info_only=True).download_model_and_data(MODEL_FILENAME)


def gpu_duration(wav_path: str, seconds: float, workdir: str) -> int:
    return int(min(GPU_MAX_SECONDS, GPU_BASE_SECONDS + seconds * GPU_SECONDS_PER_AUDIO_SECOND))


@spaces.GPU(duration=gpu_duration)
def separate_on_gpu(wav_path: str, seconds: float, workdir: str) -> tuple[str, str]:
    if MOCK:
        return mock_separate(wav_path, workdir)

    # Built inside the GPU context so audio-separator sees CUDA when it picks a device.
    from audio_separator.separator import Separator

    started = time.monotonic()
    separator = Separator(
        model_file_dir=MODEL_DIR,
        output_dir=workdir,
        output_format="WAV",
        use_soundfile=True,  # keeps the float32 subtype of the input
        normalization_threshold=1.0,  # only touches peaks above 1.0, which the headroom avoids
        amplification_threshold=0.0,
        use_autocast=True,
        log_level=logging.WARNING,
    )
    separator.load_model(model_filename=MODEL_FILENAME)
    loaded = time.monotonic()
    outputs = separator.separate(wav_path, custom_output_names={"Vocals": "vocals", "Instrumental": "instrumental"})
    log.info("model load %.1fs, separation %.1fs for %.0fs of audio",
             loaded - started, time.monotonic() - loaded, seconds)

    found = {"vocals": None, "instrumental": None}
    for name in outputs:
        path = Path(name) if os.path.isabs(name) else Path(workdir) / name
        stem = path.stem.lower()
        if stem in found:
            found[stem] = str(path)
    if not found["vocals"] or not found["instrumental"]:
        raise gr.Error(f"Model did not produce both stems (got {outputs}). Is it a vocals model?")
    return found["vocals"], found["instrumental"]


def mock_separate(wav_path: str, workdir: str) -> tuple[str, str]:
    vocals = os.path.join(workdir, "vocals.wav")
    instrumental = os.path.join(workdir, "instrumental.wav")
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", wav_path, "-af", "volume=0.3", "-c:a", "pcm_f32le", vocals])
    run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", wav_path, "-af", "volume=0.7", "-c:a", "pcm_f32le", instrumental])
    return vocals, instrumental


# -- Endpoint ------------------------------------------------------------------------


def separate(audio_file: str | None) -> tuple[str, str]:
    if not audio_file:
        raise gr.Error("No audio was uploaded.")

    workdir = tempfile.mkdtemp(prefix="tacet-")
    try:
        wav_path = os.path.join(workdir, "mix.wav")
        decode_to_float_wav(audio_file, wav_path)
        seconds = probe_seconds(wav_path)
        if seconds <= 0:
            raise gr.Error("Could not read any audio from the upload.")
        if seconds > MAX_SECONDS:
            raise gr.Error(f"Audio is {seconds:.0f}s long, the limit is {MAX_SECONDS:.0f}s.")

        vocals_wav, instrumental_wav = separate_on_gpu(wav_path, seconds, workdir)

        out_dir = tempfile.mkdtemp(prefix="tacet-out-")
        vocals = os.path.join(out_dir, "vocals.opus")
        instrumental = os.path.join(out_dir, "instrumental.opus")
        encode_opus(vocals_wav, vocals)
        encode_opus(instrumental_wav, instrumental)
        return vocals, instrumental
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


with gr.Blocks(title="Tacet remote separator") as demo:
    gr.Markdown(
        "# Tacet remote separator\n"
        f"Vocal/instrumental separation with `{MODEL_FILENAME}` on ZeroGPU. "
        "Tacet calls the `separate` API endpoint; this page is for testing by hand."
    )
    with gr.Row():
        audio_in = gr.File(label="Song", file_types=["audio", "video"])
        with gr.Column():
            vocals_out = gr.File(label="Vocals (Opus)")
            instrumental_out = gr.File(label="Instrumental (Opus)")
    gr.Button("Separate", variant="primary").click(
        separate, inputs=audio_in, outputs=[vocals_out, instrumental_out], api_name="separate"
    )

prefetch_model()

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch()
