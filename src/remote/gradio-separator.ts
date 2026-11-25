// -- Remote separation over a Gradio Space's HTTP API ---------------------------------
//
// Speaks the plain HTTP protocol (no @gradio/client: the offscreen code is not
// bundled, so it cannot import npm packages):
//   1. POST {origin}/gradio_api/upload              multipart "files"  -> ["<server path>"]
//   2. POST {origin}/gradio_api/call/separate       {data:[FileData]}  -> {event_id}
//   3. GET  {origin}/gradio_api/call/separate/{id}  server-sent events -> complete | error
//   4. GET  each returned stem: from the Hub CDN when the Space published them there
//      (result[2] = {vocals, instrumental} links), else {origin}/gradio_api/file=<path>.
//      The *.hf.space proxy can crawl at tens of KB/s; huggingface.co does not.
// Every request carries the Hugging Face token, which is what lets a private Space
// answer and what bills ZeroGPU time to the account's own quota.

const API_NAME = "separate";

type RemoteErrorCode = "http" | "quota" | "app-error" | "bad-response";

class RemoteSeparationError extends Error {
  constructor(
    readonly code: RemoteErrorCode,
    message: string
  ) {
    super(message);
    this.name = "RemoteSeparationError";
  }
}

interface SseEvent {
  event: string;
  data: string;
}

function parseSseEvents(text: string): SseEvent[] {
  const events: SseEvent[] = [];
  for (const block of text.replace(/\r\n/g, "\n").split(/\n\n+/)) {
    let event = "message";
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith("event:")) event = line.slice(6).trim();
      else if (line.startsWith("data:")) data.push(line.slice(5).replace(/^ /, ""));
    }
    if (data.length > 0 || event !== "message") events.push({ event, data: data.join("\n") });
  }
  return events;
}

function looksLikeQuotaMessage(message: string): boolean {
  return /quota/i.test(message);
}

function errorFromEventData(data: string): RemoteSeparationError {
  let message = "the Space reported an error";
  try {
    const parsed: unknown = JSON.parse(data);
    if (typeof parsed === "object" && parsed !== null) {
      const text = (parsed as { error?: unknown }).error;
      if (typeof text === "string" && text !== "") message = text;
    } else if (typeof parsed === "string" && parsed !== "") {
      message = parsed;
    }
  } catch {
    if (data.trim() !== "" && data.trim() !== "null") message = data.trim();
  }
  return new RemoteSeparationError(looksLikeQuotaMessage(message) ? "quota" : "app-error", message);
}

interface FileRef {
  path: string;
  url?: string | null;
}

function isFileRef(value: unknown): value is FileRef {
  return typeof value === "object" && value !== null && typeof (value as { path?: unknown }).path === "string";
}

interface HubLinks {
  vocals: string;
  instrumental: string;
}

const HUB_LINK_PREFIX = "https://huggingface.co/";

function readHubLinks(value: unknown): HubLinks | null {
  if (typeof value !== "object" || value === null) return null;
  const { vocals, instrumental } = value as { vocals?: unknown; instrumental?: unknown };
  if (typeof vocals !== "string" || typeof instrumental !== "string") return null;
  if (!vocals.startsWith(HUB_LINK_PREFIX) || !instrumental.startsWith(HUB_LINK_PREFIX)) return null;
  return { vocals, instrumental };
}

interface StemRefs {
  vocals: FileRef;
  instrumental: FileRef;
  hub: HubLinks | null;
}

function readStemRefs(events: SseEvent[]): StemRefs {
  for (const { event, data } of events) {
    if (event === "error") throw errorFromEventData(data);
    if (event !== "complete") continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(data);
    } catch {
      throw new RemoteSeparationError("bad-response", "the result was not JSON");
    }
    if (!Array.isArray(parsed) || parsed.length < 2 || !isFileRef(parsed[0]) || !isFileRef(parsed[1])) {
      throw new RemoteSeparationError("bad-response", `expected two files, got ${data.slice(0, 200)}`);
    }
    return { vocals: parsed[0], instrumental: parsed[1], hub: readHubLinks(parsed[2]) };
  }
  throw new RemoteSeparationError("bad-response", "the event stream ended without a result");
}

function fileUrl(origin: string, ref: FileRef): string {
  // Built from the path rather than trusting ref.url, whose host can be the Space's
  // internal address when it sits behind the Hugging Face proxy.
  return `${origin}/gradio_api/file=${ref.path}`;
}

type FetchLike = (input: string, init?: RequestInit) => Promise<Response>;

interface RemoteSeparateOptions {
  origin: string;
  token: string;
  audio: Blob;
  fileName: string;
  signal?: AbortSignal;
  fetchImpl?: FetchLike;
  onPhase?: (phase: "uploading" | "queued" | "downloading") => void;
}

interface RemoteStems {
  vocals: Blob;
  instrumental: Blob;
}

async function checkedFetch(fetchImpl: FetchLike, url: string, init: RequestInit, what: string): Promise<Response> {
  const response = await fetchImpl(url, init);
  if (response.ok) return response;
  const body = (await response.text().catch(() => "")).slice(0, 300);
  const code: RemoteErrorCode = looksLikeQuotaMessage(body) ? "quota" : "http";
  const hint =
    response.status === 401 || response.status === 403 || response.status === 404
      ? " (check the Space address and that the token can read it)"
      : "";
  throw new RemoteSeparationError(code, `${what} failed with HTTP ${response.status}${hint}: ${body}`);
}

async function separateRemotely(opts: RemoteSeparateOptions): Promise<RemoteStems> {
  const fetchImpl = opts.fetchImpl ?? ((input, init) => fetch(input, init));
  const auth: Record<string, string> = opts.token ? { Authorization: `Bearer ${opts.token}` } : {};
  const base = { signal: opts.signal };

  opts.onPhase?.("uploading");
  const form = new FormData();
  form.append("files", opts.audio, opts.fileName);
  const uploadResponse = await checkedFetch(
    fetchImpl,
    `${opts.origin}/gradio_api/upload`,
    { ...base, method: "POST", headers: auth, body: form },
    "upload"
  );
  const uploaded: unknown = await uploadResponse.json();
  if (!Array.isArray(uploaded) || typeof uploaded[0] !== "string") {
    throw new RemoteSeparationError("bad-response", `upload returned ${JSON.stringify(uploaded).slice(0, 200)}`);
  }

  const callResponse = await checkedFetch(
    fetchImpl,
    `${opts.origin}/gradio_api/call/${API_NAME}`,
    {
      ...base,
      method: "POST",
      headers: { ...auth, "Content-Type": "application/json" },
      body: JSON.stringify({
        data: [{ path: uploaded[0], orig_name: opts.fileName, meta: { _type: "gradio.FileData" } }],
      }),
    },
    "starting the job"
  );
  const call: unknown = await callResponse.json();
  const eventId = typeof call === "object" && call !== null ? (call as { event_id?: unknown }).event_id : undefined;
  if (typeof eventId !== "string") {
    throw new RemoteSeparationError("bad-response", `no event id in ${JSON.stringify(call).slice(0, 200)}`);
  }

  opts.onPhase?.("queued");
  const streamResponse = await checkedFetch(
    fetchImpl,
    `${opts.origin}/gradio_api/call/${API_NAME}/${eventId}`,
    { ...base, headers: auth },
    "waiting for the result"
  );
  const refs = readStemRefs(parseSseEvents(await streamResponse.text()));

  opts.onPhase?.("downloading");
  const download = async (urls: [string, string]): Promise<RemoteStems> => {
    const [vocals, instrumental] = await Promise.all(
      urls.map(async url => (await checkedFetch(fetchImpl, url, { ...base, headers: auth }, "download")).blob())
    );
    return { vocals, instrumental };
  };

  if (refs.hub) {
    try {
      return await download([refs.hub.vocals, refs.hub.instrumental]);
    } catch (error) {
      if (opts.signal?.aborted) throw error;
      // The Space still serves the same files, just slowly.
    }
  }
  return download([fileUrl(opts.origin, refs.vocals), fileUrl(opts.origin, refs.instrumental)]);
}

export { RemoteSeparationError, parseSseEvents, readStemRefs, separateRemotely };
export type { RemoteErrorCode, RemoteSeparateOptions, RemoteStems, SseEvent, FetchLike, HubLinks };
