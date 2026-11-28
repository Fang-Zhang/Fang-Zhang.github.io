---
title: "Audio Models, Explained (7/7): At 1.4 dB SI-SDR Plain PCM Kept 30 Percent Digit Error While Small Codecs at 0 to 4 dB Reached 62 to 84 Percent, so Reconstruction Metrics Cannot Rank Systems Across Families"
date: 2027-07-19 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [audio, evaluation, speech recognition, speaker verification, watermarking, voice cloning, series]
description: "Part 7 of 7. Why WER, SI-SDR, MOS and speaker similarity answer different questions, what the cloning and watermarking literature says, and a measured test of whether a reconstruction metric predicts intelligibility and speaker identity."
---

## One number per question

Every audio system in this series was scored by some number, and each number answers a different question. Word (here digit) error rate asks whether the content survived. A signal-level ratio such as SI-SDR asks how close the waveform is to the reference. Speaker similarity asks whether the voice is the intended one. A mean opinion score asks whether listeners like it. A latency budget asks whether it is fast enough to talk to. A system can win on one and lose on another, and the choice of metric is often the choice of which failure goes unnoticed. This part states what each metric is, summarises what the literature says about cloning risk and watermarking, and then measures with the models of Parts 2 to 4 how well one cheap metric, SI-SDR, predicts two others.

## What the sources say

- **SI-SDR (signal fidelity).** [SDR - half-baked or well done?](https://arxiv.org/abs/1811.02508) argues that the signal-to-distortion ratio of the BSS_eval toolkit "has generally been improperly used and abused", and notes that "hundreds of papers" relied on it, "often arguing that differences on the order of 0.1 dB proved the effectiveness of a method". It proposes the scale-invariant variant, SI-SDR, which this series has used since Part 2. It measures waveform agreement with a reference, not what a listener or a recogniser extracts from the audio.
- **Naturalness and speaker similarity (listener-based).** [VALL-E](https://arxiv.org/abs/2301.02111) reports that it "significantly outperforms the state-of-the-art zero-shot TTS system in terms of speech naturalness and speaker similarity" and synthesises personalised speech "with only a 3-second enrolled recording of an unseen speaker". Naturalness ratings come from listeners; the standard reference for running such tests is ITU-T [P.800](https://www.itu.int/rec/T-REC-P.800-199608-I), "Methods for subjective determination of transmission quality". Listening tests are expensive, so learned predictors exist: the [UTMOS](https://arxiv.org/abs/2204.02152) system was built for the VoiceMOS Challenge 2022, which asked entrants "to predict the MOS values of speech samples", with a main track for in-domain prediction and an out-of-domain track "for which there is less labeled data from different listening tests". A predicted score is a model of one set of listeners, not a replacement for a new test.
- **Spoofing and misuse.** [ASVspoof 2019](https://arxiv.org/abs/1911.01601) states that speaker verification systems "are vulnerable to replay, speech synthesis, and voice conversion attacks". Its database covers logical-access attacks "generated with the latest speech synthesis and voice conversion technologies" and replay attacks, and it introduces the tandem detection cost function, "which reflects the impact of spoofing and countermeasures on the reliability of a fixed ASV system". The point for evaluation: a detector is judged together with the verifier it protects, not alone.
- **Watermarking (proactive provenance).** [WavMark](https://arxiv.org/abs/2308.12770) frames cloning risk as "voice fraud and speaker impersonation" and encodes "up to 32 bits of watermark within a mere 1-second audio snippet"; with 10 to 20 second hosts it reports "an average Bit Error Rate (BER) of 0.48% across ten common attacks". [AudioSeal](https://arxiv.org/abs/2401.17264) is "the first audio watermarking technique designed specifically for localized detection of AI-generated speech", detecting "up to the sample level" with a detector up to "two orders of magnitude faster" than earlier models. Both report robustness against a fixed list of manipulations; a watermark claim is only as strong as the attack list it was tested against.
- **Generation metrics.** [MusicGen](https://arxiv.org/abs/2306.05284), covered in Part 6, reports Fréchet audio distance, KL divergence, a CLAP score and human ratings together, because no one of them tracks the others reliably.

## Experiment: does SI-SDR predict intelligibility and speaker identity?

**Question.** If a reconstruction is better by SI-SDR, is it also read more correctly by a recogniser and assigned more often to the right speaker? Part 2 reported SI-SDR for its codecs; this test applies the recognisers of Part 3 and the speaker classifier of Part 4 to the same reconstructions.

**Audio.** The 500 clips of the held-out voice (digits, 8 kHz, one second) are split by index: the 250 odd-indexed clips are evaluated; the 250 even-indexed clips only build the target speaker's centroid for the classifier.

**Degradations (17 conditions plus the clean control).**
- The three codecs of Part 2 (dropout-trained, fixed 8 codebooks, fixed 2 codebooks), each decoding from 1, 2, 4 and 8 codebooks. One codebook costs 400 bit/s (8 bits, 50 frames per second), so these span 0.4 to 3.2 kbit/s.
- Uniform PCM at 2, 3, 4, 6 and 8 bits per sample at 8 kHz, which is 16 to 64 kbit/s.

**Measurements.**
- SI-SDR of the reconstruction against the original one-second clip (as in Part 2).
- Digit error rate (edit distance over reference digits) of three recognisers on 200 four-digit sequences built from the degraded clips: the CTC and attention models of Part 3 and whisper-tiny with its text parsed into digits. The same clip choices and silent gaps (40 to 150 ms) are used in every condition, giving 800 reference digits per condition. The Part 3 models were trained on clean speech of five voices, never on codec or quantised audio.
- The percentage of clips that the speaker classifier of Part 4 assigns to the target speaker among six candidates (chance 16.7 percent). Centroids for the five training voices come from their real clips and the target's from the even-indexed held-out clips.

![Three panels with SI-SDR in dB on the horizontal axis and one measured quantity on the vertical axis, one marker per condition, colour by family. Left: CTC digit error rate; the uniform PCM points fall steeply from 76 percent at minus 3.5 dB to 14 percent at 32 dB, while all codec points sit between 61 and 92 percent for SI-SDR between minus 3 and 4 dB. Middle: whisper-tiny digit error rate; codec points are at 96 to 104 percent, PCM points fall from 80 to 13.5 percent. Right: share of clips assigned to the target speaker; codec points between 72 and 93 percent, PCM points at 0 to 2 percent for SI-SDR up to 7 dB, 49 percent at 19.6 dB and 83 percent at 32 dB. The dashed line marks the clean control in each panel.](/assets/pic/audio-7-metrics.png)

**1. Clean control.** On the real clips the digit error rate is 15.1 percent (CTC), 11.9 percent (attention) and 12.3 percent (whisper-tiny), and 84.4 percent of clips are assigned to the target speaker.

**2. Within a family, SI-SDR orders the error.** For uniform PCM and for the fixed 8-codebook codec, error falls as SI-SDR rises for all three recognisers (rank correlation between SI-SDR and digit error of -1.0 on five and four points). PCM goes from 76.1 percent CTC error at -3.5 dB through 30.1 percent at 1.4 dB to 14.4 percent at 31.9 dB.

**3. Across families, it does not.** At 1.4 dB SI-SDR, 3-bit PCM has 30.1 percent CTC error (26.8 percent attention, 56.0 percent whisper-tiny). The 8-codebook codec reaches 3.7 dB with 61.6 percent CTC error (55.1 and 95.9 percent), and the dropout-trained codec at 2.1 dB has 83.3 percent (78.8 and 97.4 percent). Over all 17 conditions the rank correlation between SI-SDR and digit error is -0.71 (CTC), -0.68 (attention) and -0.65 (whisper-tiny): the metric carries information, but a codec at 2 to 4 dB and a quantiser at 1 to 2 dB are not in the same intelligibility class. Whisper-tiny returns 96 to 104 percent error on every codec output, so extra or wrong digits dominate; the test does not separate the causes.

**4. Speaker assignment follows neither metric.** PCM at 2, 3 and 4 bits gives 0, 0 and 2 percent of clips assigned to the target, and 6-bit PCM at 19.6 dB gives 49 percent, against 84.4 percent for the clean clips: the classifier, trained on clean audio, rejects quantisation noise. The codecs give 72 to 93 percent, and in 9 of the 12 codec conditions the reconstruction is assigned to the target speaker more often than the real clips are. The lowest-quality outputs (one codebook, -2.4 to -2.9 dB) give 90 to 93 percent. For the fixed 8-codebook codec the share falls from 92.8 to 72.0 percent as codebooks are added and SI-SDR rises, while digit error also falls. Part 4 found the same pattern in another form: synthetic audio with 0.0 percent digit error and a high speaker-acceptance rate that real recordings did not reach, which is consistent with a smoothed signal sitting near a class centre rather than a better voice (the cause was not tested here).

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **Choose the metric by the failure that matters.** For a recogniser-facing pipeline (dictation, a voice agent), content error on the actual recogniser is the measurement; SI-SDR predicted it only inside one degradation family. Check which family a comparison crosses before trusting a dB difference. This is the point of the SI-SDR paper's warning about 0.1 dB claims, in a different setting: the unit is comparable only when the error type is the same.
2. **A speaker classifier is an instrument with its own blind spots.** It rejected noisy but intelligible audio and accepted blurry audio. Any speaker-similarity number for a cloned voice should be reported with a control for real recordings, as Part 4 did, and with the verifier named, as ASVspoof 2019's tandem cost does for spoofing detectors.
3. **Probability and reconstruction metrics do not stand in for listening.** Part 2's SI-SDR, Part 6's bits per frame and this part's digit error each moved differently. Naturalness needs listeners (P.800 describes how), and predicted MOS needs a check on the target domain, which is the lesson of the out-of-domain track of the VoiceMOS Challenge.
4. **Watermark and cloning claims carry their threat model.** The two papers above report results on fixed attack lists, with 10 to 20 second hosts in WavMark's case; short clips, re-encoding through a codec like the ones in this series, or an attacker who knows the scheme are different tests. No watermark was implemented here, so nothing in this part measures whether a mark survives the codecs of Part 2.

## What a small team can reproduce

Reproduced here on 32 CPU cores with no GPU, with public data (the Free Spoken Digit Dataset, six speakers) and a run of about 20 minutes for this part:
- content error, signal fidelity and a speaker-classifier score for degraded audio; codec bitrate ladders; latency budgets (Part 5); token-model comparisons (Part 6).

Not reproduced, and out of reach without more resources:
- mean opinion scores, which need recruited listeners and a controlled protocol;
- zero-shot voice cloning at VALL-E's scale (60 thousand hours of training speech), so cloning-quality numbers in this series are about small closed-set models only;
- a watermark evaluation, which needs an implementation of the scheme and an attack suite;
- spoofing countermeasure evaluation on ASVspoof-style data with a real verifier.

## Limits

- One seed, one held-out speaker, 250 clips and 800 reference digits per condition. Differences of a few points between conditions are within noise; the 2x gap in point 3 is not.
- The codecs and the recognisers are small models trained on the same six-voice digit dataset, with the codecs weak (Part 2: 2 to 4 dB SI-SDR at best). The recognisers never saw codec or quantised audio, so the digit error mixes the codec's information loss with the recogniser's mismatch. A recogniser trained on degraded audio would score all conditions lower and could change the order across families.
- Whisper-tiny at 8 kHz on one-spoken-digit clips is outside its training conditions; its error on codec outputs says that the output is not read as the digits, not why. The whisper numbers use the same 4-digit sequences as the other two recognisers.
- SI-SDR is computed on the full one-second clip including its zero padding, as in Part 2; recognition uses the clip trimmed to its original length.
- The speaker score is nearest-centroid assignment among six speakers with a classifier trained on five voices. It is not a standard verification metric (no equal error rate, no threshold sweep) and not a human similarity rating.
- No listening test was run, so no MOS is reported; no watermark was built or attacked; no voice cloning of a real person was done, and every clip is from a public dataset.
- Results from other papers cited above come from different data, models and listeners and are not comparable with these runs.

## Code and output

```python
import json, os, sys, numpy as np, torch, torch.nn as nn, torch.nn.functional as F, librosa
from asr import *
from tts import *
torch.manual_seed(0); rng = np.random.default_rng(0); C = load_clips(); M = clip_mels(C); spks = sorted(C); train_spk = [s for s in spks if s != HELD]; A, B, Ct = split_held(M[HELD])
class SpkNet2(nn.Module):
    def __init__(s, n):
        super().__init__(); s.c = nn.Sequential(nn.Conv1d(NMEL, 128, 5, padding=2), nn.GELU(), nn.Conv1d(128, 128, 5, padding=2), nn.GELU(), nn.Conv1d(128, 128, 5, padding=2), nn.GELU()); s.p = nn.Linear(256, 64); s.o = nn.Linear(64, n)
    def emb(s, x, T):
        h = s.c(x); m = (torch.arange(x.shape[2])[None] < torch.tensor(T)[:, None]).float()[:, None]; n = m.sum(2); mu = (h * m).sum(2) / n; sd = (((h - mu[:, :, None]) ** 2 * m).sum(2) / n + 1e-5).sqrt(); return s.p(torch.cat([mu, sd], 1))
    def forward(s, x, T): return s.o(F.gelu(s.emb(x, T)))
def pad(ms):
    T = [m.shape[1] for m in ms]; x = torch.full((len(ms), NMEL, max(T)), SIL)
    for i, m in enumerate(ms): x[i, :, :m.shape[1]] = m
    return x, T

net = SpkNet2(len(train_spk)); net.load_state_dict(torch.load('spk.pt')); net.eval()
```

```python
# H7 (4-digit sequences, DER): does a reconstruction metric (SI-SDR) predict intelligibility (digit error) and speaker identity? Held-out speaker, single digits.
import json, re, glob, wave, numpy as np, torch, librosa
import asr as A, codec as K
from asr import CTCModel, AttModel, feats, ctc_decode, edit, SR, HELD
from transformers import WhisperForConditionalGeneration, WhisperProcessor
torch.manual_seed(0); torch.set_num_threads(16)
X, spk = K.load_all(); files = sorted(glob.glob(K.R + "*.wav")); dig = np.array([int(f.split("/")[-1].split("_")[0]) for f in files])
lens = np.array([min(8000, wave.open(f).getnframes()) for f in files])
te = np.where(spk == HELD)[0]; ev = te[1::2]; cal = te[0::2]
print(f"held-out clips {len(te)}; eval on {len(ev)} (odd indices), speaker centroid from {len(cal)} (even indices)", flush=True)
ctc = CTCModel(); ctc.load_state_dict(torch.load("asr_ctc.pt")); ctc.eval(); att = AttModel(); att.load_state_dict(torch.load("asr_att.pt")); att.eval()
wp = WhisperProcessor.from_pretrained("whisper-tiny"); wm = WhisperForConditionalGeneration.from_pretrained("whisper-tiny").eval()
WORDS = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9}
def parse(t): return [int(m) if m.isdigit() else WORDS[m] for m in re.findall(r"[0-9]|zero|oh|one|two|three|four|five|six|seven|eight|nine", t.lower())]
def whisper(waves):
    out = []
    for i in range(0, len(waves), 25):
        w16 = [librosa.resample(w, orig_sr=SR, target_sr=16000) for w in waves[i:i + 25]]; inp = wp(w16, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad(): ids = wm.generate(inp.input_features, language="en", task="transcribe", max_new_tokens=60)
        out += wp.batch_decode(ids, skip_special_tokens=True)
    return out
import tts_eval_spk as S   # speaker classifier from H4, trained on the 5 non-held-out speakers
def spk_embed(waves):
    ms = [feats([w])[0][0] for w in waves]; out = []
    with torch.no_grad():
        for i in range(0, len(ms), 50): x, T = S.pad(ms[i:i + 50]); out.append(S.net.emb(x, T))
    return torch.cat(out)
real = lambda idx: [X[i, 0, :lens[i]].numpy() for i in idx]
names = list(S.train_spk) + [HELD]; cent = {}
for s in S.train_spk: cent[s] = spk_embed(real(np.where(spk == s)[0][::2])).mean(0)
cent[HELD] = spk_embed(real(cal)).mean(0); CM = torch.stack([cent[s] for s in names]); ti = names.index(HELD)
rng = np.random.default_rng(7); NSEQ = 200; SEQ = [rng.integers(len(ev), size=4) for _ in range(NSEQ)]; GAPS = [[int(rng.uniform(0.04, 0.15) * SR) for _ in range(3)] for _ in range(NSEQ)]
def build(waves):  # 4-digit sequences from the (degraded) clips; same clip choices and gaps for every condition
    out = []
    for ix, gp in zip(SEQ, GAPS):
        parts = []
        for j, i in enumerate(ix):
            parts.append(waves[i]);
            if j < 3: parts.append(np.zeros(gp[j], np.float32))
        out.append(np.concatenate(parts))
    return out
def evaluate(waves, ref, sdr):
    r = {"si_sdr": sdr}; sq = build(waves); refs = [[ref[i] for i in ix] for ix in SEQ]; N = sum(len(x) for x in refs)
    P = {"ctc": [], "att": []}
    for i in range(0, NSEQ, 50):
        m, ln = feats(sq[i:i + 50])
        with torch.no_grad(): P["ctc"] += ctc_decode(ctc(m, ln), ln); P["att"] += att.greedy(m, ln, maxlen=14)
    P["whisper"] = [parse(t) for t in whisper(sq)]
    for k, p in P.items(): r[k] = 100 * sum(edit(a, b) for a, b in zip(refs, p)) / N   # digit error rate on 4-digit sequences
    r["spk_target"] = 100 * (torch.cdist(spk_embed(waves), CM).argmin(1) == ti).float().mean().item()
    return r
ref = dig[ev].tolist(); res = {}; x = X[ev]
def trim(y): return [y[i, 0, :lens[ev[i]]].detach().numpy() for i in range(len(ev))]
res["clean"] = evaluate(real(ev), ref, float("inf")); print("clean", res["clean"], flush=True)
for k in ("drop", "f8", "f2"):
    m = K.Codec(); m.load_state_dict(torch.load(f"codec_{k}.pt")); m.eval()
    for nq in (1, 2, 4, 8):
        with torch.no_grad(): y, _, _, _ = m(x, torch.full((len(x),), nq))
        res[f"{k}/n={nq}"] = evaluate(trim(y), ref, K.sisdr(x[:, 0], y[:, 0]).mean().item()); print(f"{k}/n={nq}", res[f"{k}/n={nq}"], flush=True)
for b in (2, 3, 4, 6, 8):
    L = 2 ** b; xq = torch.round((x.clamp(-1, 1 - 1e-6) + 1) / 2 * L - 0.5).clamp(0, L - 1); xq = (xq + 0.5) / L * 2 - 1
    res[f"pcm/{b}bit"] = evaluate(trim(xq), ref, K.sisdr(x[:, 0], xq[:, 0]).mean().item()); print(f"pcm/{b}bit", res[f"pcm/{b}bit"], flush=True)
json.dump(res, open("h7.json", "w"), indent=1)
from scipy.stats import spearmanr
ks = [k for k in res if k != "clean"]; sd = [res[k]["si_sdr"] for k in ks]
for m in ("ctc", "att", "whisper", "spk_target"): print(f"Spearman(SI-SDR, {m}) = {spearmanr(sd, [res[k][m] for k in ks])[0]:.2f} over {len(ks)} conditions")
```

```text
held-out clips 500; eval on 250 (odd indices), speaker centroid from 250 (even indices)
clean {'si_sdr': inf, 'ctc': 15.125, 'att': 11.875, 'whisper': 12.25, 'spk_target': 84.3999981880188}
drop/n=1 {'si_sdr': -2.759681224822998, 'ctc': 91.875, 'att': 92.25, 'whisper': 103.75, 'spk_target': 90.39999842643738}
drop/n=2 {'si_sdr': -0.033768124878406525, 'ctc': 87.625, 'att': 84.875, 'whisper': 100.5, 'spk_target': 92.40000247955322}
drop/n=4 {'si_sdr': 1.5783486366271973, 'ctc': 84.25, 'att': 81.125, 'whisper': 101.125, 'spk_target': 92.00000166893005}
drop/n=8 {'si_sdr': 2.0518879890441895, 'ctc': 83.25, 'att': 78.75, 'whisper': 97.375, 'spk_target': 91.60000085830688}
f8/n=1 {'si_sdr': -2.909991502761841, 'ctc': 75.125, 'att': 67.25, 'whisper': 99.625, 'spk_target': 92.79999732971191}
f8/n=2 {'si_sdr': 0.13864625990390778, 'ctc': 69.0, 'att': 62.375, 'whisper': 99.0, 'spk_target': 81.99999928474426}
f8/n=4 {'si_sdr': 2.447279453277588, 'ctc': 63.0, 'att': 56.625, 'whisper': 98.75, 'spk_target': 76.39999985694885}
f8/n=8 {'si_sdr': 3.7060492038726807, 'ctc': 61.625, 'att': 55.125, 'whisper': 95.875, 'spk_target': 72.00000286102295}
f2/n=1 {'si_sdr': -2.382927656173706, 'ctc': 77.75, 'att': 71.75, 'whisper': 99.75, 'spk_target': 91.20000004768372}
f2/n=2 {'si_sdr': 0.09804383665323257, 'ctc': 71.375, 'att': 68.125, 'whisper': 99.0, 'spk_target': 85.6000006198883}
f2/n=4 {'si_sdr': 0.11475750058889389, 'ctc': 71.875, 'att': 68.0, 'whisper': 99.0, 'spk_target': 86.00000143051147}
f2/n=8 {'si_sdr': 0.14196564257144928, 'ctc': 71.875, 'att': 68.125, 'whisper': 99.125, 'spk_target': 86.00000143051147}
pcm/2bit {'si_sdr': -3.452634572982788, 'ctc': 76.125, 'att': 76.25, 'whisper': 80.0, 'spk_target': 0.0}
pcm/3bit {'si_sdr': 1.417630910873413, 'ctc': 30.125, 'att': 26.75, 'whisper': 56.0, 'spk_target': 0.0}
pcm/4bit {'si_sdr': 7.27191686630249, 'ctc': 16.625, 'att': 12.25, 'whisper': 31.375, 'spk_target': 1.9999999552965164}
pcm/6bit {'si_sdr': 19.583845138549805, 'ctc': 15.125, 'att': 10.0, 'whisper': 22.5, 'spk_target': 49.2000013589859}
pcm/8bit {'si_sdr': 31.865367889404297, 'ctc': 14.375, 'att': 12.0, 'whisper': 13.5, 'spk_target': 82.8000009059906}
Spearman(SI-SDR, ctc) = -0.71 over 17 conditions
Spearman(SI-SDR, att) = -0.68 over 17 conditions
Spearman(SI-SDR, whisper) = -0.65 over 17 conditions
Spearman(SI-SDR, spk_target) = -0.38 over 17 conditions
```
