import "fake-indexeddb/auto";
import { IDBFactory } from "fake-indexeddb";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { bytesToBase64 } from "../src/relay/base64.js";
import type { RemoteBackend } from "../src/remote/remote-settings.js";
import type { CaptureChunkMessage, ModelChoice, TrackPipelineOutboundMessage } from "./protocol2.js";
import type { SeparationHost } from "./separation-host.js";

// Node has no OfflineAudioContext or AudioEncoder: stand in for decoding and encoding.
const FRAMES = 8;
vi.mock("../src/separation/audio-codec.js", () => ({
  decodeFileToFloat32: vi.fn(async (blob: Blob) => {
    const marker = await blob.text();
    const frames = marker === "SHORT" ? FRAMES - 3 : FRAMES;
    return {
      channels: [new Float32Array(frames).fill(0.5), new Float32Array(frames).fill(0.5)],
      sampleRate: 44_100,
      numFrames: frames,
    };
  }),
}));
vi.mock("../src/cache/opus-codec.js", () => ({
  encodePcmToOpus: vi.fn(async (channels: Float32Array[]) => new Blob([`opus:${channels[0].length}`])),
}));
const separateRemotely = vi.fn();
vi.mock("../src/remote/gradio-separator.js", () => ({
  separateRemotely: (...args: unknown[]) => separateRemotely(...args),
}));

const { TrackPipeline } = await import("./track-pipeline.js");

const posted: TrackPipelineOutboundMessage[] = [];
const TEST_MODEL: ModelChoice = { modelUrl: "https://models.example.com/htdemucs.onnx", modelSha256: "a".repeat(64) };
const BACKEND: RemoteBackend = { origin: "https://a-b.hf.space", token: "hf_x", fallbackToLocal: true };

let localInits = 0;
function fakeSeparationHost(): SeparationHost {
  return {
    cancel(): void {},
    async init(): Promise<void> {
      localInits++;
    },
    async process(): Promise<void> {},
    dispose(): void {},
  } as unknown as SeparationHost;
}

function captureChunk(videoId: string): CaptureChunkMessage {
  return {
    type: "blk-capture-chunk",
    videoId,
    mimeType: 'audio/webm; codecs="opus"',
    index: 0,
    total: 1,
    data: bytesToBase64(new Uint8Array([9, 8, 7, videoId.length])),
  };
}

function pipelineWith(backend: RemoteBackend | null) {
  return new TrackPipeline(
    fakeSeparationHost(),
    () => 250 * 1024 * 1024,
    () => TEST_MODEL,
    () => backend
  );
}

async function until(predicate: () => boolean): Promise<void> {
  for (let i = 0; i < 200 && !predicate(); i++) await new Promise(resolve => setTimeout(resolve, 5));
  expect(predicate()).toBe(true);
}

const doneFor = (videoId: string) => () => posted.some(m => m.type === "blk-track-done" && m.videoId === videoId);

beforeEach(() => {
  posted.length = 0;
  localInits = 0;
  separateRemotely.mockReset();
  indexedDB = new IDBFactory();
  globalThis.chrome = {
    runtime: {
      sendMessage: async (message: TrackPipelineOutboundMessage) => {
        posted.push(message);
        return undefined;
      },
    },
  } as unknown as typeof chrome;
});

describe("TrackPipeline with a remote backend", () => {
  it("separates on the Space and never loads the local model", async () => {
    separateRemotely.mockResolvedValue({ vocals: new Blob(["V"]), instrumental: new Blob(["I"]) });
    pipelineWith(BACKEND).handleCaptureChunk(captureChunk("remote1"));
    await until(doneFor("remote1"));

    expect(localInits).toBe(0);
    const [opts] = separateRemotely.mock.calls[0];
    expect(opts).toMatchObject({ origin: BACKEND.origin, token: "hf_x", fileName: "remote1.webm" });
    expect(posted.some(m => m.type === "blk-track-stage" && m.stage === "downloading-model")).toBe(false);
  });

  it("fits stems that come back short to the original length", async () => {
    separateRemotely.mockResolvedValue({ vocals: new Blob(["SHORT"]), instrumental: new Blob(["I"]) });
    pipelineWith(BACKEND).handleCaptureChunk(captureChunk("remote2"));
    await until(doneFor("remote2"));

    const vocalsChunk = posted.find(m => m.type === "blk-stem-chunk" && m.stem === "vocals");
    expect(vocalsChunk && atob((vocalsChunk as { data: string }).data)).toBe(`opus:${FRAMES}`);
  });

  it("falls back to this device when the Space fails", async () => {
    separateRemotely.mockRejectedValue(new Error("You have exceeded your GPU quota"));
    pipelineWith(BACKEND).handleCaptureChunk(captureChunk("remote3"));
    await until(doneFor("remote3"));

    expect(localInits).toBe(1);
  });

  it("reports the failure instead when fallback is off", async () => {
    separateRemotely.mockRejectedValue(new Error("HTTP 503"));
    pipelineWith({ ...BACKEND, fallbackToLocal: false }).handleCaptureChunk(captureChunk("remote4"));
    await until(() => posted.some(m => m.type === "blk-track-error"));

    expect(localInits).toBe(0);
    expect(posted.find(m => m.type === "blk-track-error")).toMatchObject({
      code: "remote-failed",
      message: "HTTP 503",
    });
  });

  it("aborts the request when the listener moves to another track", async () => {
    let seenSignal: AbortSignal | undefined;
    separateRemotely.mockImplementation(
      (opts: { signal: AbortSignal }) =>
        new Promise((_resolve, reject) => {
          seenSignal = opts.signal;
          opts.signal.addEventListener("abort", () => reject(new DOMException("Aborted", "AbortError")));
        })
    );
    const pipeline = pipelineWith(BACKEND);
    pipeline.handleCaptureChunk(captureChunk("remote5"));
    await until(() => seenSignal !== undefined);
    pipeline.cancelActive();

    expect(seenSignal?.aborted).toBe(true);
    await new Promise(resolve => setTimeout(resolve, 20));
    expect(localInits).toBe(0);
    expect(posted.some(m => m.type === "blk-track-error")).toBe(false);
  });

  it("stays local when no backend is set", async () => {
    pipelineWith(null).handleCaptureChunk(captureChunk("local1"));
    await until(doneFor("local1"));
    expect(separateRemotely).not.toHaveBeenCalled();
    expect(localInits).toBe(1);
  });
});
