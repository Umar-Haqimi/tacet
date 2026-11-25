import { describe, expect, it } from "vitest";
import { fitToFrames } from "./fit-frames";

describe("fitToFrames", () => {
  it("trims a long stem", () => {
    const [left] = fitToFrames([new Float32Array([1, 2, 3, 4])], 2, 1);
    expect(Array.from(left)).toEqual([1, 2]);
  });

  it("pads a short stem with silence", () => {
    const [left] = fitToFrames([new Float32Array([1, 2])], 4, 1);
    expect(Array.from(left)).toEqual([1, 2, 0, 0]);
  });

  it("reuses an exact-length channel as is", () => {
    const channel = new Float32Array(3);
    expect(fitToFrames([channel], 3, 1)[0]).toBe(channel);
  });

  it("duplicates a mono stem into the channels asked for", () => {
    const fitted = fitToFrames([new Float32Array([5, 6])], 2, 2);
    expect(fitted.map(c => Array.from(c))).toEqual([
      [5, 6],
      [5, 6],
    ]);
  });
});
