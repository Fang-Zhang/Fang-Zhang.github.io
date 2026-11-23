---
title: "From Zero to ML Engineer, Part 10: Serving a Model Over HTTP"
date: 2027-05-08 06:00:00 +1300
categories: [Tech, AI/ML]
tags: [machine learning, beginner, deployment, fastapi, model serving, ML engineer]
description: "Part 10 of a 12-part series for complete beginners. Wrap a trained model in a small web service with FastAPI, reject malformed input, check that the service agrees with the model it wraps, and measure why batching matters. Includes a test that starts the server and calls it."
---

## What you will do, and what you get

You will put the tumour model behind a web address, so that any program, in any language, can send it 30 numbers and get a probability back. At the end you will have a 20-line service, a test script that starts it and talks to it over real HTTP, and four checks with measured results: it answers real input, it rejects malformed input before the model sees it, it agrees with the in-process model to four decimals, and a batch request scores 100 rows in 3 ms where 100 single requests take 113 ms.

![Three panels. Left: a diagram of a client sending a JSON request to a service that checks the input shape, then calls the model and returns the probability; a wrong shape goes to a 422 rejection. Middle: bar chart, 113 ms for 100 single requests against 3 ms for one batch request. Right: four test requests: real numbers give 200, three numbers give 422, text gives 422, and 30 values multiplied by 1000 give 200, marked with an exclamation mark]({{ site.baseurl }}/assets/pic/mlpath-10-service.png){: width="900" }

## How: the steps

1. **Train once and save the whole pipeline** (scaler plus model) to a file, as in Part 9.
2. **Describe the input** as a typed schema: here, a list of exactly 30 numbers.
3. **Write the endpoints:** `GET /health` (is it alive, which model version), `POST /predict` (one row) and `POST /predict_batch` (many rows).
4. **Load the model once, at startup**, not on every request.
5. **Test it the way a client would:** start the server, send real HTTP requests, check the answers, stop the server.

## Why it works: the principle

**Why a service at all.** A saved model is a file that only Python can load, and loading one is only safe from a trusted source (Part 9). A service turns it into a stable contract: send JSON, receive JSON. The app that needs a prediction does not need your libraries, your versions or your model file. When you retrain, you redeploy one thing, and the response carries a `model_version` field so that anyone can tell which model produced an answer.

**FastAPI and pydantic do the boring parts.** [FastAPI](https://fastapi.tiangolo.com/tutorial/first-steps/) turns a Python function into an HTTP endpoint with a decorator; [its request-body guide](https://fastapi.tiangolo.com/tutorial/body/) shows how a class derived from a [pydantic](https://docs.pydantic.dev/latest/) model describes the expected JSON, so that bad input is rejected automatically with status 422 and an explanation, and the endpoint function only ever sees valid data. [Uvicorn](https://uvicorn.dev/) is the server that runs it. Once the server is running, FastAPI also generates a browsable page of your endpoints at `/docs`.

**Load once.** Reading the model file takes far longer than a prediction. The `joblib.load` line sits at the top of the module, so it runs once when the server starts and every request reuses the model in memory.

**Why the service must be tested against the model.** Wrapping a model adds places for mistakes: columns in the wrong order, a missing scaling step, a different rounding. The test compares 50 rows scored through HTTP with the same rows scored in-process. The largest difference is 0.00005, the effect of rounding the output to four decimals. If that number were 0.3, something in the wrapper would be wrong. Saving the whole pipeline rather than the bare model, as in Part 9, is what keeps scaling from being forgotten.

**Batching.** Every HTTP request carries fixed costs: opening a connection, parsing JSON, validating, formatting a reply. Scoring 100 rows with 100 requests took 113 ms (1.1 ms each) on this machine; one request with 100 rows took 3 ms. The model itself is not the slow part. Timings vary by machine and by run, so read the ratio rather than the exact values; the batch endpoint is capped at 1,000 rows so one client cannot send an unbounded request.

**What validation does not do.** The service rejects 3 numbers instead of 30 (422), and 30 text values (422). But when all 30 real values are multiplied by 1000, the shape is perfect and the service answers (malignant, probability 1.0), although no tumour has those measurements. Shape and type checks are not sense checks. In practice you add range checks from the training data (the minimum and maximum of each column), and you log inputs, which is the starting point for Part 12's question of what to watch once a model is live.

## Example, worked through

The test script prints:

- **Health:** status 200, model version `lr-2027-05`.
- **One real row**, the first malignant row of the test set: 200, probability 1.0, label malignant. The truth is malignant.
- **Bad input:** 3 numbers: 422. 30 text values: 422. Neither reaches the model.
- **Absurd input:** all values times 1000: 200, probability 1.0. The model extrapolates a straight-line score far beyond anything it saw.
- **Agreement:** the service and the in-process model differ by at most 0.00005 on 50 rows.
- **Batch:** 113 ms for 100 single calls against 3 ms for one batch call.

## Implementation

Three files in one folder. You need `pip install fastapi "uvicorn[standard]" scikit-learn joblib`. Run `python train_save.py` once, then `python check.py`, which starts the server itself, calls it, and stops it. To use the service by hand instead, run `python -m uvicorn app:app --port 8000` and open `/docs` on that port in a browser.

**train_save.py**

```python
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = load_breast_cancer(return_X_y=True)
y = 1 - y                                           # 1 = malignant
Xa, Xt, ya, yt = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000)).fit(Xa, ya)
joblib.dump(model, "model.joblib")
print(f"saved model.joblib; test accuracy {100 * model.score(Xt, yt):.1f}%; expects {X.shape[1]} numbers per tumour")
```

**app.py**

```python
import joblib, numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_VERSION = "lr-2027-05"
model = joblib.load("model.joblib")                  # loaded once, when the server starts
N_FEATURES = 30
app = FastAPI(title="Tumour classifier")

class Tumour(BaseModel):
    features: list[float] = Field(min_length=N_FEATURES, max_length=N_FEATURES)

class Batch(BaseModel):
    rows: list[Tumour] = Field(min_length=1, max_length=1000)

@app.get("/health")
def health(): return {"status": "ok", "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(t: Tumour):
    p = float(model.predict_proba(np.array([t.features]))[0, 1])
    return {"malignant_probability": round(p, 4), "label": "malignant" if p >= 0.5 else "benign", "model_version": MODEL_VERSION}

@app.post("/predict_batch")
def predict_batch(b: Batch):
    p = model.predict_proba(np.array([r.features for r in b.rows]))[:, 1]
    return {"malignant_probability": [round(float(x), 4) for x in p], "model_version": MODEL_VERSION}
```

**check.py**

```python
import http.client, json, subprocess, sys, time
import joblib, numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

PORT = 8765
def call(path, payload=None):
    conn = http.client.HTTPConnection("127.0.0.1", PORT, timeout=10)
    conn.request("GET" if payload is None else "POST", path, body=None if payload is None else json.dumps(payload), headers={"Content-Type": "application/json"})
    r = conn.getresponse(); out = (r.status, json.loads(r.read())); conn.close(); return out

server = subprocess.Popen([sys.executable, "-m", "uvicorn", "app:app", "--port", str(PORT), "--log-level", "warning"])
try:
    for _ in range(50):                                          # wait until the server answers
        try: print("1. GET /health ->", call("/health")); break
        except Exception: time.sleep(0.2)

    X, y = load_breast_cancer(return_X_y=True); y = 1 - y
    _, Xt, _, yt = train_test_split(X, y, test_size=0.3, stratify=y, random_state=0)
    i = int(np.argmax(yt == 1)); row = Xt[i].tolist()               # the first malignant test row
    print("2. POST /predict, one real test row ->", call("/predict", {"features": row}), "| truth:", "malignant" if yt[i] else "benign")

    print("3. POST /predict with 3 numbers instead of 30 ->", call("/predict", {"features": [1.0, 2.0, 3.0]})[0], "(rejected before it reaches the model)")
    print("   POST /predict with a text value ->", call("/predict", {"features": ["a"] * 30})[0])
    absurd = [v * 1000 for v in row]                              # the right shape, but physically absurd values
    print("   POST /predict with all 30 values multiplied by 1000 ->", call("/predict", {"features": absurd}), "(the shape is valid, so it answers)")

    model = joblib.load("model.joblib")                          # the service must agree with the model it wraps
    direct = model.predict_proba(Xt[:50])[:, 1]
    served = np.array([call("/predict", {"features": r.tolist()})[1]["malignant_probability"] for r in Xt[:50]])
    print(f"4. largest difference between the service and the model in-process on 50 rows: {np.abs(direct - served).max():.5f} (rounding to 4 decimals)")

    t0 = time.perf_counter(); [call("/predict", {"features": r.tolist()}) for r in Xt[:100]]; one_by_one = time.perf_counter() - t0
    t0 = time.perf_counter(); call("/predict_batch", {"rows": [{"features": r.tolist()} for r in Xt[:100]]}); batch = time.perf_counter() - t0
    print(f"5. 100 rows: 100 single requests {1000 * one_by_one:.0f} ms ({10 * one_by_one:.1f} ms each), one batch request {1000 * batch:.0f} ms")
finally:
    server.terminate(); server.wait()
```

Output of `python check.py`:

```
1. GET /health -> (200, {'status': 'ok', 'model_version': 'lr-2027-05'})
2. POST /predict, one real test row -> (200, {'malignant_probability': 1.0, 'label': 'malignant', 'model_version': 'lr-2027-05'}) | truth: malignant
3. POST /predict with 3 numbers instead of 30 -> 422 (rejected before it reaches the model)
   POST /predict with a text value -> 422
   POST /predict with all 30 values multiplied by 1000 -> (200, {'malignant_probability': 1.0, 'label': 'malignant', 'model_version': 'lr-2027-05'}) (the shape is valid, so it answers)
4. largest difference between the service and the model in-process on 50 rows: 0.00005 (rounding to 4 decimals)
5. 100 rows: 100 single requests 113 ms (1.1 ms each), one batch request 3 ms
```

## Try it

1. Add a `GET /model-info` endpoint that returns the 30 feature names and the model version. Who would call it?
2. Add a range check to `Tumour` that rejects values outside twice the training minimum and maximum. Does the 1000x request still get an answer?
3. Start the server with two worker processes (`--workers 2`) and send the 100 single requests from two threads. How does the total time change?
