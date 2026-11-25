"""Tacet remote separator: vocals/instrumental separation on ZeroGPU.

One API endpoint, `separate`, takes the audio bytes Tacet already has (webm/opus,
mp4/aac, anything ffmpeg reads) and returns two Opus files: vocals and instrumental.

Pipeline:
  1. CPU: ffmpeg decodes the upload to 44.1 kHz stereo float32 WAV.
  2. GPU: audio-separator runs a BS/Mel-Band RoFormer model on it. Float input keeps
     float output.
  3. CPU: ffmpeg encodes both stems to Opus for a small download.
  4. CPU: with an HF_TOKEN secret set, both stems are committed to a private dataset
     repo and the result carries their huggingface.co links. The *.hf.space proxy can
     be very slow to download from (tens of KB/s), while the Hub CDN is not.

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
import uuid
from pathlib import Path

import gradio as gr
import spaces

MODEL_FILENAME = os.environ.get("MODEL_FILENAME", "mel_band_roformer_instrumental_instv8n_gabox.ckpt")
MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/audio-separator-models")
OPUS_BITRATE = os.environ.get("OPUS_BITRATE", "96k")  # Tacet stores stems at 96k anyway
HF_TOKEN = os.environ.get("HF_TOKEN", "")
STEMS_REPO = os.environ.get("STEMS_REPO") or f"{os.environ.get('SPACE_AUTHOR_NAME', 'me')}/tacet-stems"
STEMS_TTL_SECONDS = float(os.environ.get("STEMS_TTL_HOURS", "24")) * 3600
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


# -- Stems on the Hub -------------------------------------------------------------------


def expired_job_folders(names: list[str], now: float) -> list[str]:
    """Job folders are named <unix time>-<random>; return the ones past their TTL."""
    expired = []
    for name in names:
        stamp = name.split("-", 1)[0]
        if stamp.isdigit() and now - int(stamp) > STEMS_TTL_SECONDS:
            expired.append(name)
    return expired


def publish_stems(vocals: str, instrumental: str, api=None) -> dict | None:
    """Commit both stems (and drop expired ones) in one commit; return their links."""
    if not HF_TOKEN and api is None:
        return None
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi

    api = api or HfApi(token=HF_TOKEN)
    now = time.time()
    job = f"{int(now)}-{uuid.uuid4().hex[:10]}"
    operations = [
        CommitOperationAdd(path_in_repo=f"stems/{job}/vocals.ogg", path_or_fileobj=vocals),
        CommitOperationAdd(path_in_repo=f"stems/{job}/instrumental.ogg", path_or_fileobj=instrumental),
    ]
    try:
        folders = [
            entry.path.split("/", 1)[1]
            for entry in api.list_repo_tree(STEMS_REPO, repo_type="dataset", path_in_repo="stems")
            if entry.path.count("/") == 1 and not hasattr(entry, "size")  # folders only
        ]
        for name in expired_job_folders(folders, now):
            operations.append(CommitOperationDelete(path_in_repo=f"stems/{name}/", is_folder=True))
    except Exception as error:  # first job: no stems/ folder yet
        log.info("not cleaning up old stems: %s", error)

    commit = api.create_commit(
        STEMS_REPO, repo_type="dataset", operations=operations, commit_message=f"stems {job}"
    )
    base = f"https://huggingface.co/datasets/{STEMS_REPO}/resolve/{commit.oid}/stems/{job}"
    return {"vocals": f"{base}/vocals.ogg", "instrumental": f"{base}/instrumental.ogg"}


# -- Endpoint ------------------------------------------------------------------------


def separate(audio_file: str | None) -> tuple[str, str, dict | None]:
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
        vocals = os.path.join(out_dir, "vocals.ogg")
        instrumental = os.path.join(out_dir, "instrumental.ogg")
        encode_opus(vocals_wav, vocals)
        encode_opus(instrumental_wav, instrumental)

        links = None
        try:
            started = time.monotonic()
            links = publish_stems(vocals, instrumental)
            if links:
                log.info("stems published to %s in %.1fs", STEMS_REPO, time.monotonic() - started)
        except Exception as error:  # the files are still returned, just over the slow route
            log.warning("could not publish stems to %s: %s", STEMS_REPO, error)
        return vocals, instrumental, links
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
            links_out = gr.JSON(label="Hub links")
    gr.Button("Separate", variant="primary").click(
        separate, inputs=audio_in, outputs=[vocals_out, instrumental_out, links_out], api_name="separate"
    )

prefetch_model()

if __name__ == "__main__":
    # SSR (a Node proxy in front of Python) is no use to an API-only backend.
    demo.queue(default_concurrency_limit=2).launch(ssr_mode=False)
