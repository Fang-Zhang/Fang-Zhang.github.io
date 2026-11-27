---
title: "Audio Models, Explained (1/7): One Second of Speech Is 16,000 Numbers, the Standard Front End Turns It Into 100 Frames of 80 Values, and a Spectrogram Matched to 0.6 dB Can Still Give a Waveform at -3 dB SNR"
date: 2027-07-13 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [audio, speech, spectrogram, mel, signal processing, series]
description: "Part 1 of 7. How sound becomes data: sampling and aliasing, the short-time Fourier transform, the Mel filterbank and what it discards, measured on 3,000 spoken-digit recordings, plus a map of the audio tasks this series covers."
---

## Why start with the signal

Every model in this series, a speech recogniser, a speech synthesiser, a spoken-dialogue system, a music generator, begins by deciding what a sound is as a data structure. Text arrives already discrete; sound does not. A microphone produces a pressure that varies continuously, and the first decisions (how often to measure it, what to compute from the measurements, what to throw away) shape everything built on top. This part makes those decisions concrete and measures their consequences on real recordings, so the later parts can refer to them without re-deriving them.

The series has seven parts. Each covers the state of practice in the field, the principle behind it, and a small reproduction that runs on a CPU, with a Limits section stating what the small version cannot show.

## What the sources say

The most widely used speech recogniser front end is specified in one sentence of the [Whisper paper](https://arxiv.org/abs/2212.04356): "All audio is re-sampled to 16,000 Hz, and an 80-channel log-magnitude Mel spectrogram representation is computed on 25-millisecond windows with a stride of 10 milliseconds." The same paper trains on 680,000 hours of multilingual supervision and breaks audio into 30-second segments. Three choices are packed into that sentence, a sampling rate, a window and stride, and a frequency scale, and the experiments below take them apart one at a time.

The rest of the series is organised around the tasks that sit on top of this representation. Each is described here in the words of its source:

- **Compression into tokens.** [EnCodec](https://arxiv.org/abs/2210.13438) is "a state-of-the-art real-time, high-fidelity, audio codec leveraging neural networks", a streaming encoder-decoder "with quantized latent space", evaluated on 24 kHz monophonic and 48 kHz stereophonic audio. Part 2.
- **Recognition.** Whisper, above, maps audio to text. Part 3.
- **Synthesis and voice cloning.** [VALL-E](https://arxiv.org/abs/2301.02111) trains "a neural codec language model" on discrete codes from a codec and regards TTS "as a conditional language modeling task rather than continuous signal regression"; it synthesises personalised speech "with only a 3-second enrolled recording of an unseen speaker". Part 4.
- **Spoken dialogue.** [Moshi](https://arxiv.org/abs/2410.00037) is described as "a speech-text foundation model and full-duplex spoken dialogue framework", contrasted with pipelines of voice activity detection, speech recognition, textual dialogue and text-to-speech whose complexity "induces a latency of several seconds". Part 5.
- **Audio and music generation.** [AudioLM](https://arxiv.org/abs/2209.03143) "maps the input audio to a sequence of discrete tokens and casts audio generation as a language modeling task"; [MusicGen](https://arxiv.org/abs/2306.05284) is "a single Language Model (LM) that operates over several streams of compressed discrete music representation, i.e., tokens". Part 6.
- Part 7 covers how these systems are evaluated and what they put at risk.

The thread is the one visible in the quotations: raw waveforms are too long to model directly, so the field converts them into something shorter, either a spectrogram that a network reads or discrete tokens that a language model predicts. This part covers the first of those conversions.

## Experiments

All data comes from the [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset): 3,000 recordings, six speakers, the digits 0 to 9, each a mono 16-bit file at 8 kHz of 0.29 to 0.99 s. The spoken digit "seven" used in the figure is 5,131 samples, or 0.641 s. All code is in the last section and was run as shown.

![Three panels. Left: spectrum of a 3 kHz tone; sampled at 8 kHz it appears at 3,000 Hz, sampled at 4 kHz it appears at 1,000 Hz. Middle: an 80-band log-Mel spectrogram of a spoken digit seven, 10,262 samples at 16 kHz reduced to 65 frames, with two bright vowel regions around 0.2 and 0.4 s. Right: against window length from 5 to 200 ms, the smear of a click grows from 1.2 to 50 ms while the dip between two tones 100 Hz apart goes from none at 5 and 10 ms to 7 dB at 25 ms and fully split from 50 ms.](/assets/pic/audio-1-sound-as-data.png)

**1. Sampling.** A 3 kHz sine sampled at 8,000 samples per second shows its strongest component at 3,000 Hz. Sampled at 4,000 samples per second, whose limit is half the rate, 2 kHz, the strongest component appears at 1,000 Hz. Once the samples are taken, nothing in them says which tone it was. On real speech the same effect is smaller but measurable: halving the rate of a clip by keeping every second sample, with no low-pass filter first, changed the signal by 5.9 percent of its energy (-12.3 dB) compared with a properly filtered resample. This is the content of the [sampling theorem](https://en.wikipedia.org/wiki/Nyquist%E2%80%93Shannon_sampling_theorem): a rate r can represent frequencies below r / 2 only, and anything above must be removed before sampling, not after.

**2. Size.** One second at 16 kHz and 16 bits is 16,000 steps and 32,000 bytes. The Whisper front end turns it into 100 frames of 80 values: 8,000 values, 32,000 bytes in float32, so the bytes are the same and the number of steps is 160 times smaller. A 30-second window is 3,000 frames. The saving is in sequence length, which is what an attention layer pays for quadratically, not in storage. The spoken "seven" went from 10,262 samples (after resampling to 16 kHz) to 65 frames.

**3. Window length.** The short-time Fourier transform cuts the signal into overlapping windows and takes a spectrum of each. A window of N samples gives frequency bins spaced sr / N apart and smears an event inside it over a fraction of its own length (about a quarter in this test). The test signal at 16 kHz had two steady tones at 1,000 and 1,100 Hz and one click at 0.5 s:

| window | bin spacing | two tones | click smeared over |
|---|---|---|---|
| 5 ms | 200 Hz | one peak | 1.2 ms |
| 10 ms | 100 Hz | one peak | 2.5 ms |
| 25 ms | 40 Hz | two peaks, 7.0 dB dip | 6.3 ms |
| 50 ms | 20 Hz | two peaks, fully split | 12.5 ms |
| 100 ms | 10 Hz | two peaks, fully split | 25.0 ms |
| 200 ms | 5 Hz | two peaks, fully split | 50.0 ms |

The click widens in step with the window while the tones only separate once the window is long enough, so no single window is sharp in both. The 25 ms window of the Whisper front end is the first row in which two tones 100 Hz apart separate, and it smears a click by about 6 ms; the 10 ms stride then samples the time axis more finely than the window resolves it.

**4. The Mel scale.** A Mel filterbank groups the 201 bins of a 400-point transform into 80 overlapping triangular bands whose width grows with frequency. A hand-written implementation of the scale used by [librosa](https://librosa.org/doc/0.11.0/generated/librosa.filters.mel.html) matched the library's matrix to 1.3e-9. The 80 band centres run from 37 Hz (first five: 37, 74, 112, 149, 186) to 7,699 Hz; the lowest band is 74 Hz wide and the highest 591 Hz. Resolution is spent where speech has most structure and saved elsewhere, and the log applied afterwards compresses the very large range of energies.

**5. What the representation discards.** Griffin and Lim's [algorithm](https://dub.ucsd.edu/CATbox/Reader/GriffinLimMSTFT.pdf) estimates a signal from a spectrogram magnitude by iterating between the signal and its spectrogram. Fifty clips (digits 0 to 4) were converted to the 16 kHz front end, the Mel bands were mapped back to a 201-bin magnitude, and 60 Griffin-Lim iterations produced a waveform. Its spectrogram was compared with the original on the bins within 50 dB of the clip's peak:

| input to Griffin-Lim | mean error on the 201-bin spectrogram | 0 to 1 kHz | 1 to 4 kHz | waveform SNR |
|---|---|---|---|---|
| exact magnitude | 0.63 dB | 0.78 dB | 0.56 dB | -3.1 dB |
| 80 Mel bands | 10.07 dB | 7.85 dB | 11.23 dB | -2.6 dB |
| 40 Mel bands | 10.51 dB | 8.76 dB | 11.42 dB | -2.7 dB |
| 20 Mel bands | 10.99 dB | 9.47 dB | 11.77 dB | -2.8 dB |

Two separate losses show up. Going through Mel bands raises the error from 0.63 dB to about 10 dB, and most of that was already present at 80 bands: halving the bands twice added less than 1 dB. The second loss is the phase. The waveform that reproduced the exact magnitude to 0.63 dB had a signal-to-noise ratio of -3.1 dB against the original, meaning it is not close to the original sample by sample. As controls, inverting the exact complex transform gave 312 dB (numerically perfect) and inverting the exact magnitude with the phase set to zero gave -0.5 dB.

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **The sampling rate is a hard ceiling set before any model sees the data.** The 3 kHz tone became a 1 kHz tone and could not be recovered, so resampling to a fixed rate (16 kHz in Whisper's case) must filter first. Every later model inherits the band limit: nothing above 8 kHz exists for a 16 kHz system.
2. **A spectrogram is a change of sequence length more than of size.** The same 32,000 bytes per second arrive as 100 steps instead of 16,000, which is what makes attention and convolution stacks affordable, and what makes the window and stride the first hyperparameters of any audio model.
3. **Window length is a trade between two kinds of precision, and the standard choice sits in the middle.** At 25 ms two tones 100 Hz apart separate with a 7 dB dip and a click spreads over 6 ms. Shorter windows lose pitch detail, longer ones blur onsets.
4. **The Mel front end keeps the shape of the spectrum and discards fine structure.** The error that remained after Mel compression changed little between 80, 40 and 20 bands in this test, which says that most of the loss is inside bands, not from their count. The test does not isolate how much of the 10 dB comes from the representation and how much from Griffin-Lim's inability to invert an inconsistent spectrogram; it measures the pair.
5. **Phase is why waveforms are hard to generate.** A spectrogram can be matched closely while the waveform differs completely at the sample level, so a signal-to-noise ratio against the reference is the wrong score for generated audio, and a model that predicts spectrograms needs a separate component to produce a waveform. That component is the subject of Part 2, where audio is compressed into tokens that a decoder turns back into sound.

## Limits

- The recordings are single spoken digits from six speakers, 0.3 to 1 s long. They are not continuous speech, music or noisy audio, and the numbers say nothing about those.
- The recordings were captured at 8 kHz and resampled to 16 kHz for the Whisper-style front end, so nothing exists above 4 kHz: roughly the top 18 of the 80 Mel bands are empty in the figure. In real 16 kHz speech those bands carry content that this test does not exercise.
- The window experiment uses synthetic tones and a click, which gives clean readings; the dips of 277 dB and above for 50 ms and longer are numerical floors, reported as "fully split".
- The 50 clips in experiment 5 are digits 0 to 4 only (the first files in name order), the inversion uses 60 Griffin-Lim iterations, and the errors depend on that algorithm; a neural vocoder would invert a Mel spectrogram better. The score covers only bins within 50 dB of each clip's peak.
- Nothing about the sound quality was judged by ear; all comparisons are numerical.
- The tasks in the source list are described from their abstracts; their internals are covered in the parts that follow.

## Code and output

```python
import json, glob, wave, numpy as np, scipy.signal as ss, librosa
R = "free-spoken-digit-dataset-master/recordings/"
def load(p):
    w = wave.open(p); sr = w.getframerate(); x = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float64) / 32768; return x, sr, w.getsampwidth(), w.getnchannels()
files = sorted(glob.glob(R + "*.wav")); spk = sorted({f.split("_")[-2] for f in files}); x0, sr0, sw, ch = load(R + "7_george_0.wav"); lens = [len(load(f)[0]) / 8000 for f in files[:300:3]]
print(f"dataset: {len(files)} recordings, speakers {spk}, sampling rate {sr0} Hz, {sw*8}-bit, {ch} channel; example 7_george_0: {len(x0)} samples = {len(x0)/sr0:.3f} s; durations over 100 clips: min {min(lens):.2f} max {max(lens):.2f} s")
print("\n1. Sampling: a 3 kHz tone, 1 s")
for fs in (8000, 4000):
    t = np.arange(fs) / fs; s = np.sin(2 * np.pi * 3000 * t); sp = np.abs(np.fft.rfft(s)); print(f"  sampled at {fs} Hz (Nyquist {fs//2} Hz): strongest component at {np.argmax(sp)} Hz")
ref = ss.resample_poly(x0, 1, 2); naive = x0[::2]; n = min(len(ref), len(naive)); ref = ref[:n]; naive = naive[:n]
err = np.sum((naive - ref) ** 2) / np.sum(ref ** 2); print(f"  speech clip 8 kHz -> 4 kHz: dropping every second sample without a low-pass filter changes the signal by {10*np.log10(err):.1f} dB relative to the filtered version (error energy {err*100:.1f}% of signal energy)")
print("\n2. Sizes for 1 s of audio")
print(f"  waveform 16 kHz, 16-bit: {16000*2:,} bytes, 16,000 steps; log-Mel 80 bands, 25 ms window, 10 ms stride: {100*80:,} values = {100*80*4:,} bytes in float32, 100 steps; 30 s window: {3000} frames")
print("\n3. Window length against resolution (16 kHz; two steady tones at 1000 and 1100 Hz plus a click at 0.5 s)")
sr = 16000; t = np.arange(sr) / sr; sig = np.sin(2 * np.pi * 1000 * t) + np.sin(2 * np.pi * 1100 * t); sig[8000] += 40.0; trade = []
for ms in (5, 10, 25, 50, 100, 200):
    N = int(sr * ms / 1000); hop = max(1, N // 8); S = np.abs(librosa.stft(sig, n_fft=N * 8, win_length=N, hop_length=hop, window="hann", center=True)) ** 2; f = librosa.fft_frequencies(sr=sr, n_fft=N * 8); tt = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
    mid = S[:, (tt > 0.2) & (tt < 0.4)].mean(1); band = (f >= 850) & (f <= 1250); sub = mid[band]; pk = [i for i in range(1, len(sub) - 1) if sub[i] > sub[i - 1] and sub[i] >= sub[i + 1] and sub[i] > 0.05 * sub.max()]
    if len(pk) >= 2:
        pk = sorted(pk, key=lambda i: -sub[i])[:2]; lo, hi = sorted(pk); depth = 10 * np.log10(min(sub[lo], sub[hi]) / sub[lo:hi + 1].min())
    else: depth = 0.0
    bandt = S[(f > 3000) & (f < 5000)].sum(0); hm = bandt.max() / 2; above = tt[bandt >= hm]; width = (above.max() - above.min()) * 1000
    trade.append({"ms": ms, "bin_hz": sr / N, "valley_db": depth, "click_ms": width, "peaks": len(pk)}); print(f"  window {ms:>3} ms: bin spacing {sr/N:6.1f} Hz; peaks found near 1 kHz: {len(pk)}; dip between the two tones {depth:5.1f} dB; click smeared over {width:6.1f} ms (half-maximum width)")
print("\n4. Mel filterbank")
def hz2mel(f): f = np.asarray(f, float); lin = f / (200 / 3); logstep = np.log(6.4) / 27; return np.where(f >= 1000, 15 + np.log(np.maximum(f, 1e-9) / 1000) / logstep, lin)
def mel2hz(m): m = np.asarray(m, float); logstep = np.log(6.4) / 27; return np.where(m >= 15, 1000 * np.exp(logstep * (m - 15)), m * (200 / 3))
def mel_fb(sr, n_fft, n_mels):
    fftf = np.linspace(0, sr / 2, n_fft // 2 + 1); pts = mel2hz(np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)); fd = np.diff(pts); ramps = pts[:, None] - fftf[None, :]
    lower = -ramps[:-2] / fd[:-1, None]; upper = ramps[2:] / fd[1:, None]; W = np.maximum(0, np.minimum(lower, upper)); return W * (2.0 / (pts[2:n_mels + 2] - pts[:n_mels]))[:, None]
mine = mel_fb(16000, 400, 80); lib = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80); print(f"  hand-written filterbank (80 x 201) against librosa: max abs difference {np.abs(mine-lib).max():.1e}")
edges = mel2hz(np.linspace(0, hz2mel(8000), 82)); print(f"  80 band centres (Hz): first 5 {np.round(edges[1:6]).astype(int).tolist()}, last 5 {np.round(edges[76:81]).astype(int).tolist()}; bandwidth of the lowest band {edges[2]-edges[0]:.0f} Hz, of the highest {edges[81]-edges[79]:.0f} Hz")
print("\n5. What the log-Mel spectrogram throws away (50 clips resampled to 16 kHz; Griffin-Lim 60 iterations; only bins within 50 dB of each clip's peak are scored)")
L = lambda a: 10 * np.log10(np.maximum(a, 1e-12)); res = {}
def clipset():
    for f in files[:1500:30]:
        x, s_, _, _ = load(f); yield librosa.resample(x, orig_sr=s_, target_sr=16000)
def score(x, Pinv):
    y = librosa.griffinlim(np.sqrt(Pinv), n_iter=60, n_fft=400, hop_length=160, length=len(x), rng=np.random.default_rng(0)); P2 = np.abs(librosa.stft(y, n_fft=400, hop_length=160)) ** 2; return y, P2
for nm in (None, 80, 40, 20):
    d_lin = []; snr = []; zl = {z: [] for z in ((0, 1000), (1000, 4000))}; fq = librosa.fft_frequencies(sr=16000, n_fft=400)
    for x in clipset():
        P = np.abs(librosa.stft(x, n_fft=400, hop_length=160)) ** 2
        Pinv = P if nm is None else librosa.feature.inverse.mel_to_stft(librosa.feature.melspectrogram(S=P, sr=16000, n_mels=nm), sr=16000, n_fft=400, power=2.0)
        y, P2 = score(x, Pinv); k = P > P.max() * 1e-5
        for z in zl: m = ((fq >= z[0]) & (fq < z[1]))[:, None] & k; zl[z].append(np.mean(np.abs(L(P[m]) - L(P2[m]))))
        d_lin.append(np.mean(np.abs(L(P[k]) - L(P2[k])))); snr.append(10 * np.log10(np.sum(x ** 2) / np.sum((x - y) ** 2)))
    res[str(nm)] = (float(np.mean(d_lin)), float(np.mean(snr))); print(f"  {'exact magnitude (no Mel)' if nm is None else str(nm)+' Mel bands':>24}: mean |dB error| on the 201-bin spectrogram {np.mean(d_lin):5.2f} dB, waveform SNR {np.mean(snr):6.1f} dB; 0 to 1 kHz {np.mean(zl[(0,1000)]):5.2f} dB, 1 to 4 kHz {np.mean(zl[(1000,4000)]):5.2f} dB")
x, s_, _, _ = load(files[0]); x = librosa.resample(x, orig_sr=s_, target_sr=16000); S = librosa.stft(x, n_fft=400, hop_length=160); y = librosa.istft(S, hop_length=160, length=len(x)); print(f"  control, exact complex STFT inverted: waveform SNR {10*np.log10(np.sum(x**2)/np.sum((x-y)**2)):.0f} dB; the same with the phase set to zero: SNR {10*np.log10(np.sum(x**2)/np.sum((x-librosa.istft(np.abs(S).astype(complex), hop_length=160, length=len(x)))**2)):.1f} dB")
json.dump({"trade": trade, "res": res}, open("h1.json", "w"))
```

```text
  N = int(sr * ms / 1000); hop = max(1, N // 8); S = np.abs(librosa.stft(sig, n_fft=N * 8, win_length=N, hop_length=hop, window="hann", center=True)) ** 2; f = librosa.fft_frequencies(sr=sr, n_fft=N * 8); tt = librosa.frames_to_time(np.arange(S.shape[1]), sr=sr, hop_length=hop)
dataset: 3000 recordings, speakers ['george', 'jackson', 'lucas', 'nicolas', 'theo', 'yweweler'], sampling rate 8000 Hz, 16-bit, 1 channel; example 7_george_0: 5131 samples = 0.641 s; durations over 100 clips: min 0.29 max 0.99 s

1. Sampling: a 3 kHz tone, 1 s
  sampled at 8000 Hz (Nyquist 4000 Hz): strongest component at 3000 Hz
  sampled at 4000 Hz (Nyquist 2000 Hz): strongest component at 1000 Hz
  speech clip 8 kHz -> 4 kHz: dropping every second sample without a low-pass filter changes the signal by -12.3 dB relative to the filtered version (error energy 5.9% of signal energy)

2. Sizes for 1 s of audio
  waveform 16 kHz, 16-bit: 32,000 bytes, 16,000 steps; log-Mel 80 bands, 25 ms window, 10 ms stride: 8,000 values = 32,000 bytes in float32, 100 steps; 30 s window: 3000 frames

3. Window length against resolution (16 kHz; two steady tones at 1000 and 1100 Hz plus a click at 0.5 s)
  window   5 ms: bin spacing  200.0 Hz; peaks found near 1 kHz: 1; dip between the two tones   0.0 dB; click smeared over    1.2 ms (half-maximum width)
  window  10 ms: bin spacing  100.0 Hz; peaks found near 1 kHz: 1; dip between the two tones   0.0 dB; click smeared over    2.5 ms (half-maximum width)
  window  25 ms: bin spacing   40.0 Hz; peaks found near 1 kHz: 2; dip between the two tones   7.0 dB; click smeared over    6.3 ms (half-maximum width)
  window  50 ms: bin spacing   20.0 Hz; peaks found near 1 kHz: 2; dip between the two tones 277.4 dB; click smeared over   12.5 ms (half-maximum width)
  window 100 ms: bin spacing   10.0 Hz; peaks found near 1 kHz: 2; dip between the two tones 290.4 dB; click smeared over   25.0 ms (half-maximum width)
  window 200 ms: bin spacing    5.0 Hz; peaks found near 1 kHz: 2; dip between the two tones 304.2 dB; click smeared over   50.0 ms (half-maximum width)

4. Mel filterbank
  hand-written filterbank (80 x 201) against librosa: max abs difference 1.3e-09
  80 band centres (Hz): first 5 [37, 74, 112, 149, 186], last 5 [6602, 6861, 7129, 7409, 7699]; bandwidth of the lowest band 74 Hz, of the highest 591 Hz

5. What the log-Mel spectrogram throws away (50 clips resampled to 16 kHz; Griffin-Lim 60 iterations; only bins within 50 dB of each clip's peak are scored)
  exact magnitude (no Mel): mean |dB error| on the 201-bin spectrogram  0.63 dB, waveform SNR   -3.1 dB; 0 to 1 kHz  0.78 dB, 1 to 4 kHz  0.56 dB
              80 Mel bands: mean |dB error| on the 201-bin spectrogram 10.07 dB, waveform SNR   -2.6 dB; 0 to 1 kHz  7.85 dB, 1 to 4 kHz 11.23 dB
              40 Mel bands: mean |dB error| on the 201-bin spectrogram 10.51 dB, waveform SNR   -2.7 dB; 0 to 1 kHz  8.76 dB, 1 to 4 kHz 11.42 dB
              20 Mel bands: mean |dB error| on the 201-bin spectrogram 10.99 dB, waveform SNR   -2.8 dB; 0 to 1 kHz  9.47 dB, 1 to 4 kHz 11.77 dB
  control, exact complex STFT inverted: waveform SNR 312 dB; the same with the phase set to zero: SNR -0.5 dB
```

## References

- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356), 2022.
- [High Fidelity Neural Audio Compression (EnCodec)](https://arxiv.org/abs/2210.13438), 2022.
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111), 2023.
- [AudioLM: a Language Modeling Approach to Audio Generation](https://arxiv.org/abs/2209.03143), 2022.
- [Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284), 2023.
- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037), 2024.
- Griffin and Lim, [Signal Estimation from Modified Short-Time Fourier Transform](https://dub.ucsd.edu/CATbox/Reader/GriffinLimMSTFT.pdf), IEEE Transactions on Acoustics, Speech, and Signal Processing, 1984.
- [Free Spoken Digit Dataset](https://github.com/Jakobovski/free-spoken-digit-dataset).
