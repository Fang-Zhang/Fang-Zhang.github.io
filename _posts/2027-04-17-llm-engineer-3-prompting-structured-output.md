---
title: "The LLM Engineer Path, Part 3: Prompting and Structured Output"
date: 2027-04-17 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, prompt engineering, structured output, JSON schema, constrained decoding, python]
description: "Part 3 of a 14-part series for developers moving into LLM engineering. The parts of a prompt, three levels of getting machine-readable output, and a measured experiment: a small model writes JSON records free-form and under a finite-state constraint, with the result that constraining removes broken syntax but not wrong content."
---

Applications need output they can parse. This post covers how a prompt is built, three ways to get structured output, and one measurement that shows what each of them can and cannot guarantee.

## What a prompt is made of

A prompt is text the model continues. Anthropic's [prompt engineering overview](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview) starts by assuming you have "a clear definition of the success criteria for your use case", "some ways to empirically test against those criteria" and "a first draft prompt", and says to establish the first two before tuning wording. Part 8 returns to the testing; here, a prompt has four parts:

1. **Instructions.** What to do, for whom, in what style.
2. **Input.** The text to work on, clearly separated (for example inside tags or triple quotes), so the model can tell your instructions from the data.
3. **Examples.** The GPT-3 paper, [Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165), describes tasks "specified purely via text interaction with the model", with "few-shot demonstrations". Show 2 to 5 input-output pairs in the exact format you want back.
4. **Output format.** The fields, types and allowed values you want.

```python
def build_prompt(text, examples):
    shots = "\n".join(f"Text: {t}\nJSON: {j}" for t, j in examples)
    return (
        "Extract the person's name and age from the text. "
        'Reply with JSON only, like {"name": str, "age": int}.\n\n'
        f"{shots}\n\nText: {text}\nJSON:"
    )

print(build_prompt("Ben is 41.", [("Anna, aged 34, said hi.", '{"name": "anna", "age": 34}')]))
```

## Three levels of structured output

1. **Ask in the prompt, then parse and retry.** Works often, fails unpredictably, and you pay for the retries.
2. **Provider schema modes.** OpenAI's [Structured Outputs](https://platform.openai.com/docs/guides/structured-outputs) "constrains completed, non-refusal model output to your supplied JSON Schema". The same page lists what stays your job: "Your application still needs to handle refusals, incomplete responses, and API errors, and validate any business rules that the schema does not express." Other providers and open-source servers offer similar features with different limits; read the documentation of the one you use.
3. **Constrained decoding.** The paper [Efficient Guided Generation for Large Language Models](https://arxiv.org/abs/2307.09702) recasts generation "in terms of transitions between the states of a finite-state machine", building "an index over a language model's vocabulary" so that structure is guaranteed, with little overhead. In plain terms: before sampling each token, set the score of every token the format forbids to minus infinity (so softmax gives it 0%), then sample from what is left.

Level 3 can be built in a few lines. The rest of this post does so.

## The experiment

A small character-level Transformer (2 layers, trained on CPU) learns to write records like `{"name": "anna", "age": 34, "city": "oslo"}`. The training data has a hidden rule: each name always goes with one city (anna with oslo, ben with paris, and so on). We sample 300 records at three points in training, in two ways:

- **free**: sample each character from the model's chances.
- **constrained**: a finite-state template says which characters are allowed next (fixed keys, letters for names, up to two digits for age), and the others get minus infinity before sampling.

Each output is then checked: does it parse as JSON, does it have exactly the keys `name`, `age`, `city` with an integer age from 0 to 99, and does the city match the name?

```python
def sample(model, constrained, g):
    out = ""
    for _ in range(70):
        ctx = torch.tensor([[idx["\n"]] + [idx[c] for c in out]])[:, -T:]
        with torch.no_grad(): logits = model(ctx)[0, -1]
        if constrained:
            ok = allowed(out)
            if not ok: break
            mask = torch.full((V,), float("-inf")); mask[[idx[c] for c in ok if c in idx]] = 0; logits = logits + mask   # forbid the rest
        c = chars[torch.multinomial(F.softmax(logits, -1), 1, generator=g).item()]
        if c == "\n": break
        out += c
        if out.endswith("}"): break
    return out

def check(s):
    try: d = json.loads(s)
    except Exception: return "invalid json"
    if not (isinstance(d, dict) and list(d) == ["name", "age", "city"] and isinstance(d["age"], int) and 0 <= d["age"] <= 99): return "wrong shape"
    return "right city" if people.get(d["name"]) == d["city"] else "valid, wrong city"

def check(s):
    try: d = json.loads(s)
    except Exception: return "invalid json"
    if not (isinstance(d, dict) and list(d) == ["name", "age", "city"] and isinstance(d["age"], int) and 0 <= d["age"] <= 99): return "wrong shape"
    return "right city" if people.get(d["name"]) == d["city"] else "valid, wrong city"
```

(`allowed(out)` is a function of about 20 lines that returns the set of legal next characters for the text so far. It is the finite-state machine from the paper, written by hand for one fixed template.)

Real output (300 samples each):

```
  100 steps  free         invalid json 11%, wrong shape 16%, valid but wrong city 57%, right city 16%
  100 steps  constrained  valid but wrong city 74%, right city 26%
  300 steps  free         invalid json 1%, wrong shape 1%, valid but wrong city 4%, right city 94%
  300 steps  constrained  valid but wrong city 6%, right city 94%
 1500 steps  free         valid but wrong city 1%, right city 98%
 1500 steps  constrained  valid but wrong city 1%, right city 99%
```

![Stacked bars for six runs: at 100 training steps, free sampling gives 27 percent broken records (11 invalid JSON, 16 wrong shape), 57 percent valid but wrong city and 16 percent right; constrained sampling gives 0 percent broken, 74 percent wrong city and 26 percent right. At 300 and 1,500 steps both methods are above 94 percent right]({{ site.baseurl }}/assets/pic/llmeng-structured.png){: width="900" }
*Percentages of 300 generated records per bar.*

What the numbers say:

- **Constraining removes malformed output.** At 100 steps, free sampling breaks 27% of records (11% not JSON, 16% wrong keys or types). Constrained sampling breaks 0%, at every stage.
- **Constraining does not make the content right.** At 100 steps the constrained model is 100% valid and 74% wrong: city does not match name. Valid syntax and correct meaning are different properties, which is the point of the OpenAI warning about business rules.
- **Training fixes content; constraints fix syntax.** By 300 steps the model has learned the name-city rule, and both methods are about 94% right. The constraint then matters little for this toy template. For a larger schema, longer outputs and a weaker model it matters more.
- **A model that is wrong 74% of the time can look fine if you only check that the JSON parses.** Check the content.

## The failure to reproduce

Look at the 100-step constrained row again. Every one of those 300 records parses, so a pipeline that stops at `json.loads` would report 100% success. Add one more check, "is the city the one that goes with this name?", and success falls to 26%. Write that second check whenever there is a rule the schema cannot express (a city that belongs to a name, an end date after a start date, an amount that matches a total).

## Try it

1. Change `people` so two names share a city. Which check from `check` becomes harder to pass, and why?
2. Widen the age constraint to allow three digits. Run the 100-step case. What does `check` report?
3. Write the allowed-character function for a different template, for example `{"id": 3 digits, "ok": true or false}`.

## Limits

- The model is tiny and the template is fixed. Real schema constraints work on tokens, not characters, and can interact with how the tokenizer splits text; the Outlines paper covers the index that handles this.
- Provider structured-output features differ in supported schema features and in how they report refusals. The OpenAI page quoted here describes OpenAI's own implementation.
- This post tests output format, not prompt quality. Comparing two prompts needs a test set, which Part 8 covers.
