---
title: Tacet Remote Separator
emoji: 🎤
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 6.32.0
python_version: "3.12"
app_file: app.py
pinned: false
license: agpl-3.0
short_description: RoFormer vocal separation backend for Tacet
---

# Tacet remote separator

A ZeroGPU backend for [Tacet](https://github.com/better-lyrics/tacet). It runs a
RoFormer vocal model through [audio-separator](https://github.com/nomadkaraoke/python-audio-separator)
and returns vocals and instrumental as Opus.

## Setup

1. Create a **Gradio** Space, pick **ZeroGPU** hardware, make it **private**.
2. Push `app.py`, `requirements.txt`, `packages.txt` and this README.
3. In Tacet's popup, set the backend to the Space URL (`https://<user>-<space>.hf.space`)
   and paste a **read** token from https://huggingface.co/settings/tokens.

## Settings (Space variables, all optional)

| Variable | Default | What it does |
|---|---|---|
| `MODEL_FILENAME` | `mel_band_roformer_instrumental_instv8n_gabox.ckpt` | Any 2-stem vocals model audio-separator knows |
| `OPUS_BITRATE` | `96k` | Bitrate of the returned stems (Tacet stores 96k) |
| `HF_TOKEN` (secret) | none | Write token for `STEMS_REPO`. With it, stems go back over the Hub CDN instead of the slow `*.hf.space` proxy |
| `STEMS_REPO` | `<you>/tacet-stems` | Private dataset the stems are committed to; folders older than `STEMS_TTL_HOURS` (24) are deleted |
| `MAX_SECONDS` | `900` | Longest track accepted |
| `GPU_BASE_SECONDS` / `GPU_SECONDS_PER_AUDIO_SECOND` / `GPU_MAX_SECONDS` | `20` / `0.15` / `180` | GPU duration requested per call. Lower is better queue priority; too low and long songs time out |

## API

`POST /gradio_api/call/separate` with one file input, returns `[vocals, instrumental]`.

Local test without a GPU or model: `TACET_MOCK=1 python app.py`.
