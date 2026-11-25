import { describe, expect, it } from "vitest";
import {
  type FetchLike,
  RemoteSeparationError,
  parseSseEvents,
  readStemRefs,
  separateRemotely,
} from "./gradio-separator";

const ORIGIN = "https://a-b.hf.space";

const COMPLETE = `event: complete\ndata: [{"path": "/tmp/gradio/x/vocals.opus", "url": "http://10.0.0.1:7860/gradio_api/file=/tmp/gradio/x/vocals.opus"}, {"path": "/tmp/gradio/y/instrumental.opus"}]\n\n`;

interface Seen {
  url: string;
  init?: RequestInit;
}

function fakeFetch(stream: string, overrides: Record<string, Response> = {}): { fetchImpl: FetchLike; seen: Seen[] } {
  const seen: Seen[] = [];
  const fetchImpl: FetchLike = async (url, init) => {
    seen.push({ url, init });
    if (overrides[url]) return overrides[url];
    if (url.endsWith("/gradio_api/upload")) return Response.json(["/tmp/gradio/up/song.webm"]);
    if (url.endsWith("/gradio_api/call/separate")) return Response.json({ event_id: "ev1" });
    if (url.endsWith("/gradio_api/call/separate/ev1")) return new Response(stream);
    if (url.includes("/gradio_api/file=")) return new Response(url.includes("vocals") ? "V" : "I");
    return new Response("nope", { status: 404 });
  };
  return { fetchImpl, seen };
}

function run(fetchImpl: FetchLike) {
  return separateRemotely({
    origin: ORIGIN,
    token: "hf_secret",
    audio: new Blob([new Uint8Array([1, 2, 3])], { type: "audio/webm" }),
    fileName: "abc.webm",
    fetchImpl,
  });
}

describe("parseSseEvents", () => {
  it("splits named events and skips nothing that carries data", () => {
    const events = parseSseEvents("event: heartbeat\ndata: null\n\nevent: complete\ndata: [1]\n\n");
    expect(events).toEqual([
      { event: "heartbeat", data: "null" },
      { event: "complete", data: "[1]" },
    ]);
  });

  it("copes with CRLF line endings", () => {
    expect(parseSseEvents("event: error\r\ndata: null\r\n\r\n")).toEqual([{ event: "error", data: "null" }]);
  });
});

describe("readStemRefs", () => {
  it("names a quota error as one", () => {
    const events = parseSseEvents(
      'event: error\ndata: {"error": "You have exceeded your GPU quota (60s requested vs. 12s left)."}\n\n'
    );
    expect(() => readStemRefs(events)).toThrow(expect.objectContaining({ code: "quota" }));
  });

  it("passes the Space's own error message along", () => {
    const events = parseSseEvents('event: error\ndata: {"error": "Audio is 1200s long, the limit is 900s."}\n\n');
    expect(() => readStemRefs(events)).toThrow("Audio is 1200s long");
  });

  it("refuses a stream that ends without a result", () => {
    expect(() => readStemRefs(parseSseEvents("event: heartbeat\ndata: null\n\n"))).toThrow(RemoteSeparationError);
  });
});

describe("separateRemotely", () => {
  it("uploads, calls, waits, and downloads both stems", async () => {
    const { fetchImpl, seen } = fakeFetch(`event: heartbeat\ndata: null\n\n${COMPLETE}`);
    const stems = await run(fetchImpl);

    expect(await stems.vocals.text()).toBe("V");
    expect(await stems.instrumental.text()).toBe("I");
    const call = seen.find(s => s.url === `${ORIGIN}/gradio_api/call/separate`);
    expect(JSON.parse(String(call?.init?.body))).toEqual({
      data: [{ path: "/tmp/gradio/up/song.webm", orig_name: "abc.webm", meta: { _type: "gradio.FileData" } }],
    });
  });

  it("sends the token on every request", async () => {
    const { fetchImpl, seen } = fakeFetch(COMPLETE);
    await run(fetchImpl);
    expect(seen).toHaveLength(5);
    for (const { init } of seen) {
      expect((init?.headers as Record<string, string>).Authorization).toBe("Bearer hf_secret");
    }
  });

  it("downloads from the Space origin, not the internal url in the result", async () => {
    const { fetchImpl, seen } = fakeFetch(COMPLETE);
    await run(fetchImpl);
    const downloads = seen.filter(s => s.url.includes("/gradio_api/file=")).map(s => s.url);
    expect(downloads).toEqual([
      `${ORIGIN}/gradio_api/file=/tmp/gradio/x/vocals.opus`,
      `${ORIGIN}/gradio_api/file=/tmp/gradio/y/instrumental.opus`,
    ]);
  });

  it("explains a refused request", async () => {
    const { fetchImpl } = fakeFetch(COMPLETE, {
      [`${ORIGIN}/gradio_api/upload`]: new Response("Unauthorized", { status: 401 }),
    });
    await expect(run(fetchImpl)).rejects.toThrow(/HTTP 401 \(check the Space address/);
  });

  it("stops when aborted", async () => {
    const controller = new AbortController();
    controller.abort();
    const fetchImpl: FetchLike = async (_url, init) => {
      init?.signal?.throwIfAborted();
      return new Response("");
    };
    await expect(
      separateRemotely({
        origin: ORIGIN,
        token: "",
        audio: new Blob([]),
        fileName: "a.webm",
        signal: controller.signal,
        fetchImpl,
      })
    ).rejects.toMatchObject({ name: "AbortError" });
  });
});

describe("separateRemotely with Hub links", () => {
  const HUB = "https://huggingface.co/datasets/a/tacet-stems/resolve/abc123/stems/1-x";
  const WITH_LINKS = `event: complete\ndata: [{"path": "/tmp/vocals.ogg"}, {"path": "/tmp/instrumental.ogg"}, {"vocals": "${HUB}/vocals.ogg", "instrumental": "${HUB}/instrumental.ogg"}]\n\n`;

  it("downloads from the Hub instead of the Space", async () => {
    const { fetchImpl, seen } = fakeFetch(WITH_LINKS, {
      [`${HUB}/vocals.ogg`]: new Response("HV"),
      [`${HUB}/instrumental.ogg`]: new Response("HI"),
    });
    const stems = await run(fetchImpl);

    expect(await stems.vocals.text()).toBe("HV");
    expect(await stems.instrumental.text()).toBe("HI");
    expect(seen.some(s => s.url.includes("/gradio_api/file="))).toBe(false);
    const hubCall = seen.find(s => s.url.startsWith(HUB));
    expect((hubCall?.init?.headers as Record<string, string>).Authorization).toBe("Bearer hf_secret");
  });

  it("falls back to the Space when the Hub download fails", async () => {
    const { fetchImpl, seen } = fakeFetch(WITH_LINKS, {
      [`${HUB}/vocals.ogg`]: new Response("gone", { status: 404 }),
      [`${HUB}/instrumental.ogg`]: new Response("HI"),
    });
    const stems = await run(fetchImpl);

    expect(await stems.vocals.text()).toBe("V");
    expect(seen.filter(s => s.url.includes("/gradio_api/file="))).toHaveLength(2);
  });

  it("ignores links that do not point at huggingface.co", async () => {
    const evil = `event: complete\ndata: [{"path": "/tmp/v.ogg"}, {"path": "/tmp/i.ogg"}, {"vocals": "https://evil.example/v", "instrumental": "https://evil.example/i"}]\n\n`;
    const { fetchImpl, seen } = fakeFetch(evil);
    await run(fetchImpl);
    expect(seen.some(s => s.url.startsWith("https://evil.example"))).toBe(false);
  });
});
