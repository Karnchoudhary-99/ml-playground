# In-Browser ML Playground

Train tiny machine-learning models right in the browser — no server, no
build step, no dependencies. Plain JavaScript modules shared between Node
(for tests) and the browser (for the interactive demo page).

## Roadmap

- [x] Project scaffold (package.json, linear-regression smoke module, tests, CI, README)
- [x] Linear regression + k-NN trainers with live charts
- [x] Interactive demo page (draw data, train, see the decision boundary)
- [x] Model import/export + dataset presets
- [ ] Self-review checklist, then public

## Quick start

```sh
npm test              # 33 tests, dependency-free
node examples/demo.mjs  # tour of every model with metrics
node examples/export.mjs  # train, export to JSON, reload, compare predictions
```

Interactive demo (ES modules need http, not `file://`):

```sh
python3 -m http.server
# open http://localhost:8000
```

Pick a dataset and a model, hit **Train**, and watch the decision
boundary (or regression curve) appear. Click the chart to add your own
point — the boundary retrains live.

## Models

| Model | Module | Notes |
| --- | --- | --- |
| Linear regression | `src/linear.js` | Closed-form least squares |
| k-NN classifier / regressor | `src/knn.js` | Euclidean/Manhattan, distance-weighted votes |
| Logistic regression | `src/logistic.js` | Batch gradient descent, L2 — standardise features first (`src/scaling.js`) |
| Datasets | `src/datasets.js` | Deterministic seeded presets: two blobs, moons, spirals, rings, XOR quadrants, noisy line |
| Model I/O | `src/serialization.js` | Versioned JSON export/import for every model (`ml-playground-model` v1); classifiers carry their scaler, so a reloaded model predicts identically |
| Metrics | `src/metrics.js` | Accuracy, MSE, R², confusion matrix |

## Model import/export

Every trained model can be saved as a JSON file — from the demo page
(**Export model**) or from Node (`serializeLinear` / `serializeLogistic` /
`serializeKnn` in `src/serialization.js`) — and loaded back later
(**Import model**, or `deserializeModel` / `modelFromString`). Loading is
strict: unknown formats, unsupported versions, missing fields and
non-finite numbers are rejected rather than silently mis-loaded. k-NN
models store their training points; logistic models store weights, bias
and hyperparameters; classifiers trained on scaled features store the
scaler, so imported predictions match the original exactly (verified by
`node examples/export.mjs`: 0 disagreements on 100 training points).

## Baseline (from `node examples/demo.mjs`, seeds fixed)

| Dataset | Model | Train | Test |
| --- | --- | --- | --- |
| Two blobs | k-NN (k=5) | 100.0% acc | 100.0% acc |
| Two blobs | Logistic regression | 98.9% acc | 100.0% acc |
| Moons | k-NN (k=5) | 100.0% acc | 96.7% acc |
| Moons | Logistic regression | 87.8% acc | 83.3% acc |
| Noisy line | Linear regression | — | MSE 0.450, R² 0.976 |
| Noisy line | k-NN regressor (k=3) | — | MSE 0.046, R² 0.997 |

The moons row is the teaching point: a linear model cannot solve
non-linear data, while k-NN can.

## License

MIT — see [LICENSE](LICENSE).
