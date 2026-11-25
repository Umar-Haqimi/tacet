// -- Remote separation backend settings -------------------------------------------
//
// Kept apart from the synced settings on purpose: the token lives in
// chrome.storage.local so it never leaves this machine through Chrome sync.

const REMOTE_STORAGE_KEY = "blk-remote";

interface RemoteSettings {
  enabled: boolean;
  spaceUrl: string;
  token: string;
  fallbackToLocal: boolean;
}

const DEFAULT_REMOTE_SETTINGS: RemoteSettings = {
  enabled: false,
  spaceUrl: "",
  token: "",
  fallbackToLocal: true,
};

function sanitizeRemoteSettings(raw: unknown): RemoteSettings {
  const record = typeof raw === "object" && raw !== null ? (raw as Record<string, unknown>) : {};
  return {
    enabled: typeof record.enabled === "boolean" ? record.enabled : DEFAULT_REMOTE_SETTINGS.enabled,
    spaceUrl: typeof record.spaceUrl === "string" ? record.spaceUrl.trim() : DEFAULT_REMOTE_SETTINGS.spaceUrl,
    token: typeof record.token === "string" ? record.token.trim() : DEFAULT_REMOTE_SETTINGS.token,
    fallbackToLocal:
      typeof record.fallbackToLocal === "boolean" ? record.fallbackToLocal : DEFAULT_REMOTE_SETTINGS.fallbackToLocal,
  };
}

// -- Space address ------------------------------------------------------------------
//
// Accepts what people paste: "owner/name", the huggingface.co/spaces page, or the
// direct *.hf.space origin. Returns the *.hf.space origin, or null when unusable.

const HF_SPACE_HOST_SUFFIX = ".hf.space";

function spaceSubdomain(owner: string, name: string): string {
  return `${owner}-${name}`.toLowerCase().replace(/[._]/g, "-");
}

function resolveSpaceOrigin(input: string): string | null {
  const value = input.trim();
  if (value === "") return null;

  const shorthand = /^([\w.-]+)\/([\w.-]+)$/.exec(value);
  if (shorthand) return `https://${spaceSubdomain(shorthand[1], shorthand[2])}${HF_SPACE_HOST_SUFFIX}`;

  let url: URL;
  try {
    url = new URL(value);
  } catch {
    return null;
  }
  if (url.protocol !== "https:") return null;

  if (url.hostname.endsWith(HF_SPACE_HOST_SUFFIX)) return url.origin;

  if (url.hostname === "huggingface.co") {
    const page = /^\/spaces\/([\w.-]+)\/([\w.-]+)\/?/.exec(url.pathname);
    if (page) return `https://${spaceSubdomain(page[1], page[2])}${HF_SPACE_HOST_SUFFIX}`;
  }
  return null;
}

interface RemoteBackend {
  origin: string;
  token: string;
  fallbackToLocal: boolean;
}

function remoteBackendFrom(settings: RemoteSettings | null | undefined): RemoteBackend | null {
  if (!settings?.enabled) return null;
  const origin = resolveSpaceOrigin(settings.spaceUrl);
  if (!origin) return null;
  return { origin, token: settings.token, fallbackToLocal: settings.fallbackToLocal };
}

export { REMOTE_STORAGE_KEY, DEFAULT_REMOTE_SETTINGS, sanitizeRemoteSettings, resolveSpaceOrigin, remoteBackendFrom };
export type { RemoteSettings, RemoteBackend };
