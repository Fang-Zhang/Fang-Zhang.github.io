---
title: "Audio Models, Explained (5/7): A Speech Cascade Spent 1.4 Seconds on Turn-End Wait, Recognition and the First Token of a Reply, Before Any Speech Synthesis, Against 230 Milliseconds Between Human Turns"
date: 2027-07-17 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [audio, spoken dialogue, latency, moshi, whisper, series]
description: "Part 5 of 7. Why voice assistants answer slowly and what full-duplex models change: a measured latency budget of turn-end detection, recognition and a small language model on CPU, set against the figures reported for Moshi."
---

## Two ways to hold a spoken conversation

A spoken dialogue system can be a chain or a single model. The chain, called a cascade here, runs voice activity detection, speech recognition (Part 3), a text language model and speech synthesis (Part 4) one after another. The single model reads and writes audio tokens (Part 2) directly.

[Moshi](https://arxiv.org/abs/2410.00037) (Kyutai, 2024) is the reference design for the second kind, and its paper argues against the first in three points. Pipelines of "voice activity detection, speech recognition, textual dialogue and text-to-speech" have "a latency of several seconds between interactions", where natural conversation shows "response times of a few hundred milliseconds"; text as the intermediate modality loses "emotion or non-speech sounds"; and they "rely on a segmentation into speaker turns, which does not take into account overlapping speech, interruptions and interjections".

## What the source says

- **Two streams.** Moshi models its own speech and the user's "in parallel", as two audio token streams, so it listens and speaks at once and has no explicit turn segmentation. Fine-tuning on the Fisher corpus (2,000 hours of telephone conversations with each side on a separate channel) provides the separate streams, and that fine-tuning step is what gives it full-duplex behaviour.
- **Codec and frame size.** Its codec, Mimi, is causal; "both its initial frame size and overall stride correspond to 80ms" (12.5 Hz), so audio can be encoded and decoded as it arrives.
- **Inner Monologue.** The model predicts time-aligned text tokens before the audio tokens of each frame, which the paper reports "significantly improves the linguistic quality of generated speech" and which also gives streaming recognition and synthesis from the same model.
- **Backbone.** Helium, a 7-billion-parameter text model pre-trained on 2.1 trillion tokens, with a smaller Depth Transformer for the audio tokens within a frame.
- **Latency.** The paper reports "a theoretical latency of 160ms, 200ms in practice", against "the 230 ms average in natural conversations measured over 10 languages" (citing [Stivers et al., 2009](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/)). The 160 ms is the 80 ms frame plus an acoustic delay of one frame in the fine-tuned model.

## Experiments

A cascade was assembled from measured parts on this machine (32 CPU cores, no GPU, 16 threads used), and its latency measured as the time between the speaker stopping and the first output, in three parts.

**1. Turn-end wait.** A cascade cannot answer until it decides the speaker has finished, which means waiting for a stretch of silence (the hangover). If the hangover is shorter than a pause inside the speaker's own sentence, the system cuts in early. On 1,000 randomly chosen utterances of [LibriSpeech](https://www.openslr.org/12/) dev-clean (read audiobook speech, median 5.9 s, range 1.5 to 32.1 s), an energy rule (frames more than 30 or 40 dB below the utterance peak count as silence) gives the longest pause inside each utterance. The longest internal pause had a median of 320 ms and a 90th percentile of 710 ms at 30 dB, and 160 and 550 ms at 40 dB. The share of utterances that a given hangover would cut before their end:

| hangover | 100 ms | 200 ms | 300 ms | 400 ms | 500 ms | 600 ms | 800 ms | 1,000 ms |
|---|---|---|---|---|---|---|---|---|
| 30 dB rule | 77.0% | 62.7% | 53.0% | 40.5% | 27.7% | 17.9% | 6.2% | 1.9% |
| 40 dB rule | 62.0% | 45.0% | 31.8% | 20.6% | 13.0% | 7.1% | 2.7% | 0.7% |

**2. Recognition.** Whisper tiny (37.8 M parameters) and base (72.6 M), run on whole clips after they end, with greedy decoding, median of four clips per length bucket:

| clip length | tiny: encoder / total | base: encoder / total |
|---|---|---|
| 1 to 3 s | 91 / 213 ms | 207 / 388 ms |
| 3 to 6 s | 93 / 350 ms | 210 / 509 ms |
| 6 to 10 s | 85 / 574 ms | 205 / 787 ms |
| 10 to 15 s | 86 / 584 ms | 201 / 1,003 ms |

The encoder time does not depend on the clip length: Whisper pads every input to a 30-second window ("Whisper models are trained on 30-second audio chunks and cannot consume longer audio inputs at once", [Whisper paper](https://arxiv.org/abs/2212.04356)).

**3. Reply generation.** [Qwen2.5-0.5B-Instruct](https://arxiv.org/abs/2412.15115) (494 M parameters, 32-bit floats, a plain greedy loop in the transformers library), five short questions, prompts of 31 to 38 tokens after the chat template, instruction to answer in short spoken sentences. First token after 452 to 583 ms (median 487 ms), then 6.6 to 8.0 tokens per second; the first sentence ended at token 15 to 26 (median 22), 2.3 to 3.9 s after the prompt arrived (median 3.2 s).

**Budget.** With a 600 ms hangover (which cuts 7.1 percent of utterances under the 40 dB rule and 17.9 percent under the 30 dB rule), Whisper tiny on a 3 to 6 s clip (350 ms) and the median language model figures:

- first text token: 600 + 350 + 487 = **1,436 ms**;
- first complete sentence: 600 + 350 + 3,214 = **4,164 ms**;
- speech synthesis: not measured, so both figures are lower bounds for a voice reply.

![Left: share of 1,000 read-speech utterances that would be cut off before they end against the silence needed to end a turn, from 77 and 62 percent at 100 ms to 2 and 1 percent at 1,000 ms for two silence thresholds. Middle: recognition time after the clip ends for Whisper tiny and base against clip length, with a constant encoder floor of about 90 and 205 ms. Right: latency budget bars: a cascade needs 1,436 ms to the first text token and 4,164 ms to the first full sentence before any speech synthesis, against 230 ms for human turn gaps and 160 to 200 ms for Moshi.](/assets/pic/audio-5-dialogue.png)

## Engineering reading

These points are this series' reading of the measurements and the sources.

1. **The turn-end wait cannot be tuned away.** At 200 ms, a number near the human average, a silence rule cut 45 to 63 percent of these read utterances; keeping early cut-offs under 10 percent took 600 to 800 ms. A system that waits for silence responds at least a few hundred milliseconds late; answering at human speed requires starting before the other party is finished, which a cascade's segmentation into turns does not allow and a model that listens while it speaks can in principle do.
2. **The recogniser has a floor.** A fixed 30-second window puts 90 ms (tiny) to 205 ms (base) under every utterance on this CPU, however short. Streaming recognisers exist for this reason; the numbers above are the non-streaming case.
3. **The language model dominates once it is on a CPU, and the unit that matters is the first speakable chunk.** The first token arrived after about half a second, but a first sentence took 3.2 s at 8 tokens per second. Cascades hide this by handing the synthesiser the first clause instead of the first sentence; no part of that pipelining was tested here.
4. **Serial sums overstate real pipelines and understate them elsewhere.** Real systems overlap stages (a streaming recogniser works during the hangover), but they also add network hops and a synthesis stage that this budget leaves out.
5. **End-to-end moves the cost, not just the sum.** Moshi's figures come from one 7B model run on hardware that is not this CPU; the comparison in the figure shows magnitudes, not a like-for-like benchmark, and the model gives up the modularity (swap the recogniser, swap the voice) that the cascade keeps.

## Limits

- Speech synthesis was not measured, no GPU was used, and the language model was a 0.5B model in an unoptimized loop; faster runtimes and quantization would shorten the language model rows.
- The turn-end wait uses an energy rule on read audiobook speech, not a trained voice activity detector on conversation; pauses and turn gaps in spontaneous talk differ, and no overlap, interruption or backchannel was tested.
- Recognition used four clips per bucket and the language model five prompts, on one machine; no recognition accuracy or answer quality was evaluated.
- Prompts were short single questions; real dialogues carry history, which lengthens the first-token time.
- Moshi was not run; its 160 ms, 200 ms and the 230 ms human average are the paper's figures, measured differently and on other hardware.

## Code and output

```python
import glob, json, numpy as np, soundfile as sf, librosa
fs = sorted(glob.glob("LibriSpeech/dev-clean/*/*/*.flac")); print("files", len(fs), flush=True)
rng = np.random.default_rng(0); pick = [fs[i] for i in rng.choice(len(fs), 1000, replace=False)]
Ts = list(range(100, 1100, 100)); out = {}
durs = []; res = {30: [], 40: []}
for f in pick:
    x, sr = sf.read(f); assert sr == 16000; durs.append(len(x) / sr)
    for tdb in (30, 40):
        iv = librosa.effects.split(x, top_db=tdb, frame_length=400, hop_length=160); gaps = [(iv[i + 1][0] - iv[i][1]) / sr * 1000 for i in range(len(iv) - 1)]
        res[tdb].append(max(gaps) if gaps else 0.0)
print(f"utterances 1000, duration median {np.median(durs):.1f}s, range {min(durs):.1f}-{max(durs):.1f}s")
for tdb in (30, 40):
    g = np.array(res[tdb]); print(f"top_db={tdb}: longest internal pause median {np.median(g):.0f} ms, p90 {np.percentile(g,90):.0f} ms, max {g.max():.0f} ms")
    print("  hangover ms -> utterances cut before their end: " + ", ".join(f"{T}:{(g >= T).mean()*100:.1f}%" for T in Ts))
    out[tdb] = {str(T): float((g >= T).mean() * 100) for T in Ts}
json.dump(dict(curve=out, durs_median=float(np.median(durs))), open("lat_endpoint.json", "w"))
```

```python
import glob, time, json, numpy as np, soundfile as sf, torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
torch.set_num_threads(16)
fs = sorted(glob.glob("LibriSpeech/dev-clean/*/*/*.flac")); rng = np.random.default_rng(1); pick = [fs[i] for i in rng.choice(len(fs), 300, replace=False)]
cl = []
for f in pick:
    x, sr = sf.read(f); cl.append((len(x) / sr, f, x.astype(np.float32)))
sel = []
for lo, hi in ((1, 3), (3, 6), (6, 10), (10, 15)):
    sel += [c for c in cl if lo <= c[0] < hi][:4]
res = {}
for name in ("whisper-tiny", "whisper-base"):
    p = WhisperProcessor.from_pretrained(name); m = WhisperForConditionalGeneration.from_pretrained(name).eval(); rows = []
    inp = p(sel[0][2], sampling_rate=16000, return_tensors="pt")
    with torch.no_grad(): m.generate(inp.input_features, language="en", task="transcribe", max_new_tokens=40)
    for d, f, x in sel:
        inp = p(x, sampling_rate=16000, return_tensors="pt"); t0 = time.time()
        with torch.no_grad(): enc = m.model.encoder(inp.input_features)
        te = time.time() - t0; t0 = time.time()
        with torch.no_grad(): ids = m.generate(inp.input_features, language="en", task="transcribe", max_new_tokens=120)
        tt = time.time() - t0; rows.append((d, te, tt, p.batch_decode(ids, skip_special_tokens=True)[0]))
    res[name] = [(d, te, tt) for d, te, tt, _ in rows]
    print(name, "params", sum(q.numel() for q in m.parameters()))
    for lo, hi in ((1, 3), (3, 6), (6, 10), (10, 15)):
        r = [x for x in rows if lo <= x[0] < hi]; print(f"  {lo}-{hi} s clips (n={len(r)}): encoder {np.median([x[1] for x in r])*1000:.0f} ms, full transcription {np.median([x[2] for x in r])*1000:.0f} ms")
    print("  sample:", rows[5][0].__round__(1), "s ->", repr(rows[5][3][:100]))
json.dump(res, open("lat_asr.json", "w"))
```

```python
import time, json, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_num_threads(16)
tok = AutoTokenizer.from_pretrained("qwen05"); m = AutoModelForCausalLM.from_pretrained("qwen05", dtype=torch.float32).eval(); print("params", sum(q.numel() for q in m.parameters()))
Q = ["How do I boil an egg so the yolk stays soft?", "What is the difference between weather and climate?", "Why is the sky blue?", "How long does it take to walk ten kilometres?", "Can you explain what a prime number is?"]
rows = []
def run(q, n=60):
    msgs = [{"role": "system", "content": "You are a voice assistant. Answer in short spoken sentences."}, {"role": "user", "content": q}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)["input_ids"]; t0 = time.time(); ts = []; txt = ""; first_sent = None
    with torch.no_grad():
        o = m(ids, use_cache=True); pk = o.past_key_values; nxt = o.logits[0, -1].argmax().item(); ts.append(time.time() - t0)
        for _ in range(n):
            txt += tok.decode([nxt])
            if first_sent is None and any(c in tok.decode([nxt]) for c in ".!?"): first_sent = (len(ts), ts[-1])
            o = m(torch.tensor([[nxt]]), past_key_values=pk, use_cache=True); pk = o.past_key_values; nxt = o.logits[0, -1].argmax().item(); ts.append(time.time() - t0)
            if nxt == tok.eos_token_id or nxt == tok.convert_tokens_to_ids("<|im_end|>"): break
    return ids.shape[1], ts, first_sent, txt
run(Q[0], 5)
for q in Q:
    n, ts, fs, txt = run(q); dec = (len(ts) - 1) / (ts[-1] - ts[0]); rows.append(dict(prompt=n, ttft=ts[0], tps=dec, first_tokens=fs[0] if fs else None, first_sent=fs[1] if fs else None)); print(f"prompt {n} tok: first token {ts[0]*1000:.0f} ms, decode {dec:.1f} tok/s, first sentence ends at token {rows[-1]['first_tokens']} after {rows[-1]['first_sent']*1000 if fs else float('nan'):.0f} ms | {txt[:70]!r}", flush=True)
for k in ("ttft", "tps", "first_sent", "first_tokens"): print(k, "median", np.median([r[k] for r in rows]))
json.dump(rows, open("lat_llm.json", "w"))
```

```python
import json, numpy as np
ep = json.load(open("lat_endpoint.json")); asr = json.load(open("lat_asr.json")); llm = json.load(open("lat_llm.json"))
H = 600; a = np.median([tt for d, te, tt in asr["whisper-tiny"] if 3 <= d < 6]) * 1000; ab = np.median([tt for d, te, tt in asr["whisper-base"] if 3 <= d < 6]) * 1000
ft = np.median([r["ttft"] for r in llm]) * 1000; fs = np.median([r["first_sent"] for r in llm]) * 1000
print(f"hangover {H} ms (utterances cut early: {ep['curve']['40'][str(H)]:.1f}% at top_db 40, {ep['curve']['30'][str(H)]:.1f}% at top_db 30)")
print(f"ASR whisper-tiny, 3-6 s clip: {a:.0f} ms; whisper-base: {ab:.0f} ms")
print(f"LLM first token {ft:.0f} ms; first sentence boundary {fs:.0f} ms (median {np.median([r['first_tokens'] for r in llm]):.0f} tokens)")
print(f"cascade, first text token : {H}+{a:.0f}+{ft:.0f} = {H+a+ft:.0f} ms   (TTS not measured)")
print(f"cascade, first sentence   : {H}+{a:.0f}+{fs:.0f} = {H+a+fs:.0f} ms   (TTS not measured)")
print(f"reference: human average 230 ms; Moshi 160 ms theoretical (80 ms frame + 80 ms acoustic delay), 200 ms in practice")
json.dump(dict(H=H, a=a, ab=ab, ft=ft, fs=fs), open("lat_budget.json", "w"))
```

```text
# --- lat_endpoint.txt ---
files 2703
utterances 1000, duration median 5.9s, range 1.5-32.1s
top_db=30: longest internal pause median 320 ms, p90 710 ms, max 1630 ms
  hangover ms -> utterances cut before their end: 100:77.0%, 200:62.7%, 300:53.0%, 400:40.5%, 500:27.7%, 600:17.9%, 700:10.7%, 800:6.2%, 900:3.3%, 1000:1.9%
top_db=40: longest internal pause median 160 ms, p90 550 ms, max 1430 ms
  hangover ms -> utterances cut before their end: 100:62.0%, 200:45.0%, 300:31.8%, 400:20.6%, 500:13.0%, 600:7.1%, 700:4.2%, 800:2.7%, 900:1.6%, 1000:0.7%
# --- lat_asr.txt ---
whisper-tiny params 37760640
  1-3 s clips (n=4): encoder 91 ms, full transcription 213 ms
  3-6 s clips (n=4): encoder 93 ms, full transcription 350 ms
  6-10 s clips (n=4): encoder 85 ms, full transcription 574 ms
  10-15 s clips (n=4): encoder 86 ms, full transcription 584 ms
  sample: 5.2 s -> ' If they have cast off the vulgar herd, they have not cast off iPadia.'
whisper-base params 72593920
  1-3 s clips (n=4): encoder 207 ms, full transcription 388 ms
  3-6 s clips (n=4): encoder 210 ms, full transcription 509 ms
  6-10 s clips (n=4): encoder 205 ms, full transcription 787 ms
  10-15 s clips (n=4): encoder 201 ms, full transcription 1003 ms
  sample: 5.2 s -> ' If they have cast off the vulgar herd, they have not cast off a patia.'
# --- lat_llm.txt ---
params 494032768
prompt 38 tok: first token 583 ms, decode 7.9 tok/s, first sentence ends at token 22 after 3214 ms | 'To boil an egg so the yolk stays soft, use a small amount of water and'
prompt 34 tok: first token 487 ms, decode 8.0 tok/s, first sentence ends at token 26 after 3618 ms | 'Weather refers to the current state of the atmosphere, while climate i'
prompt 31 tok: first token 471 ms, decode 8.0 tok/s, first sentence ends at token 19 after 2706 ms | 'The sky is blue because of the scattering of sunlight by tiny particle'
prompt 35 tok: first token 544 ms, decode 7.9 tok/s, first sentence ends at token 15 after 2305 ms | 'It takes about 20-30 minutes to walk ten kilometers.'
prompt 34 tok: first token 452 ms, decode 6.6 tok/s, first sentence ends at token 24 after 3940 ms | 'A prime number is a natural number greater than 1 that has no positive'
ttft median 0.4868602752685547
tps median 7.945497158089915
first_sent median 3.2141175270080566
first_tokens median 22.0
# --- lat_budget.txt ---
hangover 600 ms (utterances cut early: 7.1% at top_db 40, 17.9% at top_db 30)
ASR whisper-tiny, 3-6 s clip: 350 ms; whisper-base: 509 ms
LLM first token 487 ms; first sentence boundary 3214 ms (median 22 tokens)
cascade, first text token : 600+350+487 = 1436 ms   (TTS not measured)
cascade, first sentence   : 600+350+3214 = 4164 ms   (TTS not measured)
reference: human average 230 ms; Moshi 160 ms theoretical (80 ms frame + 80 ms acoustic delay), 200 ms in practice
```

## References

- A. Défossez et al., [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037), 2024.
- T. Stivers et al., [Universals and cultural variation in turn-taking in conversation](https://pmc.ncbi.nlm.nih.gov/articles/PMC2705608/), PNAS, 2009.
- A. Radford et al., [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356), 2022.
- Qwen Team, [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115), 2024.
- V. Panayotov et al., [LibriSpeech corpus](https://www.openslr.org/12/), 2015.
