# NFL2025 — Super Bowl Winner Prediction Dashboard

Interactive dashboard to estimate who is most likely to win the Super Bowl, using an Elo-style model and Monte Carlo playoff simulation.

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input data format

The app expects **14 rows** (7 AFC + 7 NFC playoff teams) with these columns:

- `team`: team name (unique)
- `conference`: `AFC` or `NFC`
- `seed`: 1..7 within each conference (if you enable auto-seeding, this can be any number)
- `rating`: Elo-like rating (higher = better; 1500 ~ average)

Use `data_sample_playoffs.csv` as a template.
