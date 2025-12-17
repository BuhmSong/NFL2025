# NFL2025 — Super Bowl Winner Prediction Dashboard

Interactive dashboard to estimate who is most likely to win the Super Bowl, using an Elo-style model and Monte Carlo playoff simulation.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
~/.local/bin/streamlit run app.py
```

## Data source

This app mirrors the current playoff seeds shown on `https://www.nfl.com/standings/playoff-picture`.

Under the hood, the data is loaded from NFL’s authenticated endpoints, so you must paste your own **Authorization token** (`Bearer ...`) into the sidebar to fetch the latest playoff picture.
