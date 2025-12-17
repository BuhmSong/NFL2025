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

This app mirrors the current playoff seeds shown on `https://www.espn.com/nfl/standings/_/view/playoff`.

Under the hood, it uses ESPN’s public standings endpoint:

- `https://site.web.api.espn.com/apis/v2/sports/football/nfl/standings?view=playoff`
