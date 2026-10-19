// Voice mode's microphone, in the audio thread: mixed to mono and posted as 20 ms of 16 kHz int16 at a time, what the
// speech service reads. A context at 16 kHz (voice.ts asks for one; the browser resamples) passes samples through;
// at another rate (Firefox gives a microphone only at the device's) each output sample is the mean of the input samples
// its span covers, a box filter that keeps speech's band and drops most of what would alias.
class Capture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.step = sampleRate / 16000;
    this.span = 0;
    this.sum = 0;
    this.count = 0;
    this.out = new Int16Array(320);
    this.n = 0;
  }

  process(inputs) {
    const chans = inputs[0];
    if (!chans || !chans.length) return true;
    for (let i = 0; i < chans[0].length; i++) {
      let x = 0;
      for (const c of chans) x += c[i];
      x /= chans.length;
      if (this.step === 1) { this.push(x); continue; }
      this.sum += x;
      this.count++;
      if (++this.span >= this.step) {
        this.span -= this.step;
        this.push(this.sum / this.count);
        this.sum = 0;
        this.count = 0;
      }
    }
    return true;
  }

  push(x) {
    this.out[this.n++] = Math.max(-32768, Math.min(32767, Math.round(x * 32768)));
    if (this.n < this.out.length) return;
    const b = this.out.buffer;
    this.port.postMessage(b, [b]);
    this.out = new Int16Array(320);
    this.n = 0;
  }
}

registerProcessor("chrysalis-capture", Capture);
