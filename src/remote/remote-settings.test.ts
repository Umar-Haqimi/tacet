import { describe, expect, it } from "vitest";
import {
  DEFAULT_REMOTE_SETTINGS,
  remoteBackendFrom,
  resolveSpaceOrigin,
  sanitizeRemoteSettings,
} from "./remote-settings";

describe("resolveSpaceOrigin", () => {
  it("turns owner/name shorthand into the hf.space origin", () => {
    expect(resolveSpaceOrigin("Umar-Haqimi/tacet-separator")).toBe("https://umar-haqimi-tacet-separator.hf.space");
  });

  it("reads the huggingface.co Space page", () => {
    expect(resolveSpaceOrigin("https://huggingface.co/spaces/someone/My_Space.v2")).toBe(
      "https://someone-my-space-v2.hf.space"
    );
  });

  it("keeps a direct hf.space address, dropping any path", () => {
    expect(resolveSpaceOrigin("https://someone-thing.hf.space/?view=api")).toBe("https://someone-thing.hf.space");
  });

  it("refuses anything that is not a Space", () => {
    expect(resolveSpaceOrigin("")).toBeNull();
    expect(resolveSpaceOrigin("https://example.com/spaces/a/b")).toBeNull();
    expect(resolveSpaceOrigin("http://someone-thing.hf.space")).toBeNull();
    expect(resolveSpaceOrigin("not a url")).toBeNull();
  });
});

describe("sanitizeRemoteSettings", () => {
  it("falls back to the defaults for junk", () => {
    expect(sanitizeRemoteSettings(null)).toEqual(DEFAULT_REMOTE_SETTINGS);
    expect(sanitizeRemoteSettings({ enabled: "yes", token: 4 })).toEqual(DEFAULT_REMOTE_SETTINGS);
  });

  it("trims the address and token", () => {
    const settings = sanitizeRemoteSettings({ enabled: true, spaceUrl: "  a/b ", token: " hf_x\n" });
    expect(settings.spaceUrl).toBe("a/b");
    expect(settings.token).toBe("hf_x");
  });
});

describe("remoteBackendFrom", () => {
  it("is null while switched off or without a usable address", () => {
    expect(remoteBackendFrom(undefined)).toBeNull();
    expect(remoteBackendFrom({ ...DEFAULT_REMOTE_SETTINGS, spaceUrl: "a/b" })).toBeNull();
    expect(remoteBackendFrom({ ...DEFAULT_REMOTE_SETTINGS, enabled: true, spaceUrl: "nope" })).toBeNull();
  });

  it("resolves the origin when switched on", () => {
    expect(remoteBackendFrom({ enabled: true, spaceUrl: "a/b", token: "hf_x", fallbackToLocal: false })).toEqual({
      origin: "https://a-b.hf.space",
      token: "hf_x",
      fallbackToLocal: false,
    });
  });
});
