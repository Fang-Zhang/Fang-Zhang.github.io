---
title: "The LLM Engineer Path, Part 7: Agents and Tool Use"
date: 2027-04-21 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [LLM engineer, agents, tool use, function calling, ReAct, safety, python]
description: "Part 7 of a 14-part series for developers moving into LLM engineering. What a tool-calling agent is, the loop around it, and five failures a harness must handle: malformed output, unknown tools, unsafe arguments, endless loops, and the quadratic growth of what each step re-sends. Tested with scripted stand-ins for the model."
---

A model can only produce text. A **tool-using agent** is a loop that lets that text trigger actions in your code, such as a search, a database query or a calculation, and feeds the results back. Most of the engineering is not in the model but in the loop around it. This post builds that loop and tests it against five kinds of bad model behaviour.

## What an agent is

Anthropic's [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents) draws a line: "Workflows are systems where LLMs and tools are orchestrated through predefined code paths. Agents, on the other hand, are systems where LLMs dynamically direct their own processes and tool usage, maintaining control over how they accomplish tasks." A workflow is code with LLM calls inside it; an agent is an LLM calling your code. Choose a workflow when the steps are known in advance: it is cheaper to run and easier to test.

The agent idea has a research origin. [ReAct](https://arxiv.org/abs/2210.03629) has the model produce "reasoning traces and task-specific actions in an interleaved manner": reasoning traces "help the model induce, track, and update action plans", while actions "allow it to interface with external sources, such as knowledge bases or environments".

## How tool calling works

The model never runs anything. OpenAI's [function calling guide](https://platform.openai.com/docs/guides/function-calling) describes a tool call as "a special kind of response" in which the model, after looking at the prompt, "determines that in order to follow the instructions in the prompt, it needs to call one of the tools we made available to it". The response carries a tool name and JSON-encoded arguments, and then: "When the model calls a function, you must execute it and return the result." Your code is the one executing, so your code is where validation, permissions and limits live.

One cost fact from the same guide: "callable function definitions count against the model's context limit and are billed as input tokens". Every tool you describe is paid for on every request.

## The loop

![A flow diagram: the model replies with JSON; the harness parses it, checks the tool name and argument types, checks for a repeated call and the step budget, runs the tool inside try/except, appends the result or an error to the transcript, and sends the whole transcript back to the model. On the right, a chart of cumulative words sent: a naive loop reaches 11,150 words in 50 steps while a hardened loop stops at 246 words]({{ site.baseurl }}/assets/pic/llmeng-agent-loop.png){: width="900" }
*Left: the harness. Right: measured cost of a model that never stops calling the same tool.*

The two functions below are the entire loop, once naive and once hardened. The task is "What is the cost of 3 apples and 2 pears with 10% tax?", with a price-lookup tool and a calculator.

```python
def lookup_price(item): return PRICES[item]
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}
def safe_calc(expr):                                    # only numbers and + - * / ( )
    def ev(n):
        if isinstance(n, ast.Expression): return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)): return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in OPS: return OPS[type(n.op)](ev(n.left), ev(n.right))
        raise ValueError("only numbers and + - * / are allowed")
    return round(ev(ast.parse(expr, mode="eval")), 2)
TOOLS = {"lookup_price": {"fn": lookup_price, "args": {"item": str}}, "calculator": {"fn": safe_calc, "args": {"expr": str}}}
```

```python
def naive(policy, cap=50):
    msgs = [{"role": "user", "content": TASK}]; sent = 0; obs = []
    for step in range(1, cap + 1):
        sent += words(msgs); out = json.loads(policy(msgs))                         # crashes on bad JSON
        if "final" in out: return dict(outcome="answered " + out["final"], steps=step, sent=sent, obs=obs)
        t = TOOLS[out["tool"]]                                                       # crashes on unknown tool
        res = eval(out["args"]["expr"]) if out["tool"] == "calculator" else t["fn"](**out["args"])   # eval runs ANY python
        obs.append(str(res)); msgs += [{"role": "assistant", "content": json.dumps(out)}, {"role": "tool", "content": str(res)}]
    return dict(outcome="hit the cap of %d steps, no answer" % cap, steps=cap, sent=sent, obs=obs)

def hardened(policy, max_steps=8, max_repeat=2):
    msgs = [{"role": "user", "content": TASK}]; sent = 0; seen = {}
    for step in range(1, max_steps + 1):
        sent += words(msgs); raw = policy(msgs)
        msgs.append({"role": "assistant", "content": raw})
        try: out = json.loads(raw)
        except json.JSONDecodeError as e: msgs.append({"role": "tool", "content": f"ERROR: reply was not valid JSON ({e.msg}). Reply with one JSON object."}); continue
        if "final" in out: return dict(outcome="answered " + out["final"], steps=step, sent=sent)
        spec = TOOLS.get(out.get("tool"))
        if spec is None: msgs.append({"role": "tool", "content": f"ERROR: unknown tool {out.get('tool')!r}. Tools: {sorted(TOOLS)}"}); continue
        args = out.get("args", {})
        if set(args) != set(spec["args"]) or not all(isinstance(args[k], t) for k, t in spec["args"].items()): msgs.append({"role": "tool", "content": f"ERROR: arguments must be {spec['args']}"}); continue
        key = raw; seen[key] = seen.get(key, 0) + 1
        if seen[key] > max_repeat: return dict(outcome="stopped: same call repeated %d times" % seen[key], steps=step, sent=sent)
        try: res = spec["fn"](**args)
        except Exception as e: res = f"ERROR: {type(e).__name__}: {e}"
        msgs.append({"role": "tool", "content": str(res)})
    return dict(outcome="stopped: step budget of %d used" % max_steps, steps=max_steps, sent=sent)
```

No model is available offline here, so five scripted functions stand in for it: one behaves well (look up apple, look up pear, calculate, answer), and four misbehave in a specific way. The harness is real; the "models" only produce the JSON a real one sometimes would. Real output:

```
good              naive     answered 3.41                                steps  4  words sent 340
good              hardened  answered 3.41                                steps  4  words sent 340
loops forever     naive     hit the cap of 50 steps, no answer           steps 50  words sent 11150
loops forever     hardened  stopped: same call repeated 3 times          steps  3  words sent 246
malformed JSON    naive     CRASH JSONDecodeError: Illegal trailing comma before end of object: line
malformed JSON    hardened  answered 3.41                                steps  5  words sent 508
unknown tool      naive     CRASH KeyError: 'get_price'
unknown tool      hardened  answered 3.41                                steps  5  words sent 464
unsafe expression naive     answered 3.41                                steps  4  words sent 340
unsafe expression hardened  answered 3.41                                steps  5  words sent 480
naive harness, unsafe expression, first tool result: /path/to/the/script/directory
hardened harness, same expression, tool result    : ERROR: ValueError - only numbers and + - * / are allowed
```

## What each row teaches

- **Malformed JSON** (a trailing comma, which models do produce): the naive loop crashes. The hardened loop returns the parser's message to the model as an observation, and the model recovers on its next step. Errors are input, not exceptions.
- **Unknown tool**: same pattern. The error message lists the valid tool names, which is what lets the model correct itself.
- **Unsafe expression**: the naive calculator passes the model's text to `eval`. The scripted model asked for `__import__('os').getcwd()`, and the naive harness executed it and returned the directory of the script. It was harmless here and would be arbitrary code execution with any other payload. Anything the model writes is untrusted input, even if you wrote the prompt. The hardened `safe_calc` parses the expression into a tree and accepts only numbers and the four arithmetic operators. Part 10 returns to this.
- **Endless loop**: the naive version stopped only because the test had a 50-step cap. The hardened version counted identical calls and stopped on the third.
- **Cost growth.** Each step re-sends the whole transcript. With a base prompt of 76 words and 6 words added per step, words sent after n steps is 76n + 3n(n - 1): 246 for 3 steps and 11,150 for 50. It grows with the square of the number of steps, which is why a budget is a cost control and not just a safety feature. (Words stand in for tokens here; real tokenisers produce more tokens than words. Caching, covered in Part 9, softens but does not remove the effect.)

## The failure to reproduce

Run the `loops forever` row with `naive` and watch the "words sent" counter: 11,150 words for one question, 33 times the cost of the successful run, with no answer. A real agent loop will find a way to do this eventually: a search that returns nothing, a tool that keeps timing out, a model that re-reads the same file. Put three limits in every agent: a maximum number of steps, a detector for repeated identical calls, and a maximum budget in tokens or money.

## Try it

1. Add a third tool, `delete_item(item)`, and try to give the model access to it. What should the harness require before running it? (A confirmation, a permission list, or both.)
2. The repeat detector compares the raw reply. A model that changes one character each time defeats it. Write a detector that compares tool name and arguments instead, and add a limit on total tool errors.
3. Replace the scripted `good` policy with a real model if you can run one, and print which row it behaves like. Real models are less predictable than these scripts; this post did not run one.

## Limits

- The models are scripted, so the test shows the harness's behaviour, not how often real models fail in each way. That frequency depends on the model and has to be measured (Part 8).
- Only one task, two tools and no concurrency. Production agents add memory, multiple tool calls per turn and long-running jobs.
- Argument validation here checks Python types. A schema validator (JSON Schema) is the usual next step, and validation of meaning (is this file one the agent may delete?) always stays your code's job.
