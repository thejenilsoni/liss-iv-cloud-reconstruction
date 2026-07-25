# Contributing

## Development setup

```bash
cp .env.example .env
make install
make dev
```

Before submitting a change:

```bash
make lint
make test
make build
```

## Commit style

Use short imperative messages that describe one coherent change:

```text
Add temporal context encoder
Calibrate cloud probability threshold
Preserve GeoTIFF transform on export
```

Keep code, tests, and documentation for a feature together where practical.
Do not commit datasets, satellite scenes, checkpoints, credentials, or
generated output.

## Scientific changes

Changes to masking, reconstruction, or metrics should include:

- a stated hypothesis;
- the geographic validation split used;
- before-and-after paired metrics;
- important failure cases;
- an ablation when multiple mechanisms changed.

Reference-free metrics on real cloudy scenes are not a substitute for
evaluation against clear targets.

