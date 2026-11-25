// -- Stem length alignment ---------------------------------------------------------
//
// Remote stems are decoded from Opus, so they can come back a few samples longer or
// shorter than the locally decoded original. The rest of the pipeline expects every
// stem to be exactly the original's length: trim the tail or pad it with silence.

function fitToFrames(channels: readonly Float32Array[], targetFrames: number, channelCount: number): Float32Array[] {
  const fitted: Float32Array[] = [];
  for (let c = 0; c < channelCount; c++) {
    const source = channels[c] ?? channels[0];
    if (source && source.length === targetFrames) {
      fitted.push(source);
      continue;
    }
    const out = new Float32Array(targetFrames);
    if (source) out.set(source.subarray(0, Math.min(source.length, targetFrames)));
    fitted.push(out);
  }
  return fitted;
}

export { fitToFrames };
