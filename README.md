# Q-Med

**Q-Med** is a research and educational decision-support prototype that evaluates **when** quantum chemistry simulation is worth running for a small molecular system.

> **Disclaimer:** Q-Med is a scientific-computing and educational research prototype, **not** a clinical or drug-approval tool.

## Repository Layout

```text
q-med/
  app.py
  qmed/
    __init__.py
    molecules.py
    qee.py
    active_space.py
    vqe_runner.py
    classical.py
    benchmark.py
    storage.py
    config.py
  data/
    benchmarks.json
  tests/
    test_week1.py
  notebooks/
    h2_curve.ipynb
  requirements.txt
  LICENSE
  README.md
```

## Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -v
```

## License

MIT License. See [LICENSE](LICENSE).
