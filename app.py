from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st


REQUIRED_COLS = ["team", "conference", "seed", "rating"]
CONFERENCES = ["AFC", "NFC"]


@dataclass(frozen=True)
class ModelParams:
    k: float = 400.0
    home_field_elo: float = 55.0


def elo_win_prob(rating_a: float, rating_b: float, *, params: ModelParams, home_advantage_a: float = 0.0) -> float:
    """Elo-style win probability for team A."""
    diff = (rating_a + home_advantage_a) - rating_b
    return 1.0 / (1.0 + 10 ** (-diff / params.k))


def normalize_input_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Expected: {REQUIRED_COLS}")

    df = df[REQUIRED_COLS].copy()
    df["team"] = df["team"].astype(str).str.strip()
    df["conference"] = df["conference"].astype(str).str.strip().str.upper()

    # seed/rating coercion
    df["seed"] = pd.to_numeric(df["seed"], errors="coerce").astype("Int64")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if df["team"].isna().any() or (df["team"].str.len() == 0).any():
        raise ValueError("Column 'team' contains empty values.")

    bad_conf = sorted(set(df.loc[~df["conference"].isin(CONFERENCES), "conference"].unique()))
    if bad_conf:
        raise ValueError(f"Column 'conference' must be one of {CONFERENCES}. Bad values: {bad_conf}")

    if df["rating"].isna().any():
        raise ValueError("Column 'rating' must be numeric (no blanks).")

    return df


def auto_seed_by_rating(df: pd.DataFrame) -> pd.DataFrame:
    """Assign seeds 1..7 within each conference by descending rating."""
    out = df.copy()
    out["seed"] = (
        out.sort_values(["conference", "rating"], ascending=[True, False])
        .groupby("conference")
        .cumcount()
        .add(1)
        .astype(int)
    )
    return out


def validate_playoff_field(df: pd.DataFrame) -> None:
    # Ensure exactly 7 per conference, seeds 1..7 unique.
    counts = df["conference"].value_counts().to_dict()
    for conf in CONFERENCES:
        if counts.get(conf, 0) != 7:
            raise ValueError(f"Need exactly 7 teams for {conf}; got {counts.get(conf, 0)}.")

        seeds = df.loc[df["conference"] == conf, "seed"]
        if seeds.isna().any():
            raise ValueError(f"Seeds for {conf} contain blanks; enable auto-seeding or provide 1..7.")

        seeds_list = seeds.astype(int).tolist()
        if sorted(seeds_list) != list(range(1, 8)):
            raise ValueError(f"{conf} seeds must be exactly 1..7 (each once). Got: {sorted(seeds_list)}")

    if df["team"].duplicated().any():
        dups = sorted(df.loc[df["team"].duplicated(), "team"].unique().tolist())
        raise ValueError(f"Duplicate team names found: {dups}")


def simulate_game(team_a: str, team_b: str, ratings: Dict[str, float], *, p: float, rng: np.random.Generator) -> str:
    # p is probability team_a wins
    return team_a if rng.random() < p else team_b


def simulate_conference(df_conf: pd.DataFrame, *, params: ModelParams, rng: np.random.Generator) -> Tuple[str, List[Tuple[str, str, str]]]:
    """Return (champion, list of (round, team_a, team_b) winners encoded in third?)."""
    # map seed->team
    by_seed = (
        df_conf.set_index("seed")[["team", "rating"]]
        .sort_index()
        .to_dict(orient="index")
    )
    ratings = {v["team"]: float(v["rating"]) for v in by_seed.values()}

    def wp(home: str, away: str) -> float:
        return elo_win_prob(ratings[home], ratings[away], params=params, home_advantage_a=params.home_field_elo)

    # Wild card: 2v7, 3v6, 4v5 (higher seed home)
    wc = [
        (2, 7),
        (3, 6),
        (4, 5),
    ]
    wc_winners: List[int] = []
    for h, a in wc:
        home = by_seed[h]["team"]
        away = by_seed[a]["team"]
        winner = simulate_game(home, away, ratings, p=wp(home, away), rng=rng)
        winner_seed = h if winner == home else a
        wc_winners.append(winner_seed)

    # Divisional: seed 1 vs lowest remaining; remaining two play; higher seed home
    remaining = sorted([1] + wc_winners)
    lowest = max(remaining)
    s1_opponent = lowest
    other = sorted([s for s in remaining if s not in {1, s1_opponent}])

    div_pairs = [(1, s1_opponent), (other[0], other[1])]
    div_winners: List[int] = []
    for h, a in div_pairs:
        home = by_seed[h]["team"]
        away = by_seed[a]["team"]
        winner = simulate_game(home, away, ratings, p=wp(home, away), rng=rng)
        winner_seed = h if winner == home else a
        div_winners.append(winner_seed)

    # Conference championship: higher seed home
    h, a = sorted(div_winners)
    home = by_seed[h]["team"]
    away = by_seed[a]["team"]
    winner = simulate_game(home, away, ratings, p=wp(home, away), rng=rng)
    return winner, []


def simulate_playoffs(df: pd.DataFrame, *, params: ModelParams, n_sims: int, rng_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)

    ratings = dict(zip(df["team"].tolist(), df["rating"].astype(float).tolist()))
    by_conf = {c: df.loc[df["conference"] == c].copy() for c in CONFERENCES}

    sb_wins: Dict[str, int] = {t: 0 for t in df["team"].tolist()}
    conf_wins: Dict[str, int] = {t: 0 for t in df["team"].tolist()}

    for _ in range(n_sims):
        afc_champ, _ = simulate_conference(by_conf["AFC"], params=params, rng=rng)
        nfc_champ, _ = simulate_conference(by_conf["NFC"], params=params, rng=rng)
        conf_wins[afc_champ] += 1
        conf_wins[nfc_champ] += 1

        p_afc = elo_win_prob(ratings[afc_champ], ratings[nfc_champ], params=params, home_advantage_a=0.0)
        sb_champ = afc_champ if rng.random() < p_afc else nfc_champ
        sb_wins[sb_champ] += 1

    out = df[["team", "conference", "seed", "rating"]].copy()
    out["p_conf"] = out["team"].map(lambda t: conf_wins.get(t, 0) / n_sims)
    out["p_super_bowl"] = out["team"].map(lambda t: sb_wins.get(t, 0) / n_sims)
    out = out.sort_values(["p_super_bowl", "p_conf", "rating"], ascending=[False, False, False]).reset_index(drop=True)
    return out


def chart_probs(df_probs: pd.DataFrame, col: str, title: str) -> alt.Chart:
    chart_df = df_probs[["team", "conference", col]].copy()
    chart_df[col] = (chart_df[col] * 100.0).round(2)
    return (
        alt.Chart(chart_df)
        .mark_bar()
        .encode(
            x=alt.X(f"{col}:Q", title="Probability (%)"),
            y=alt.Y("team:N", sort="-x", title=""),
            color=alt.Color("conference:N"),
            tooltip=["team:N", "conference:N", alt.Tooltip(f"{col}:Q", format=".2f")],
        )
        .properties(title=title, height=min(560, 22 * len(chart_df) + 40))
    )


def main() -> None:
    st.set_page_config(page_title="Super Bowl Winner Prediction Dashboard", layout="wide")
    st.title("Super Bowl Winner Prediction Dashboard")

    with st.sidebar:
        st.header("Data")
        source = st.radio(
            "Team ratings source",
            ["Use sample playoff field", "Upload CSV", "Manual entry"],
            index=0,
        )

        auto_seed = st.checkbox("Auto-seed 1..7 by rating within each conference", value=True)
        st.divider()

        st.header("Model")
        home_field_elo = st.slider("Home-field advantage (Elo points)", 0, 120, 55, 5)
        k = st.slider("Elo scale (higher = less sensitive)", 200, 600, 400, 25)

        st.divider()
        st.header("Simulation")
        n_sims = st.slider("Monte Carlo simulations", 500, 50000, 20000, 500)
        rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=2025, step=1)

    params = ModelParams(k=float(k), home_field_elo=float(home_field_elo))

    if source == "Use sample playoff field":
        raw = pd.read_csv("data_sample_playoffs.csv")
    elif source == "Upload CSV":
        up = st.file_uploader("Upload a CSV with columns: team, conference, seed, rating", type=["csv"])
        if up is None:
            st.info("Upload a CSV to get started.")
            return
        raw = pd.read_csv(up)
    else:
        st.caption("Enter 14 playoff teams (7 AFC, 7 NFC). Ratings are on an Elo-like scale (1500 = average).")
        starter = pd.read_csv("data_sample_playoffs.csv")
        raw = st.data_editor(starter, num_rows="fixed", use_container_width=True)

    try:
        df = normalize_input_df(pd.DataFrame(raw))
        if auto_seed:
            df = auto_seed_by_rating(df)
        validate_playoff_field(df)
    except Exception as e:
        st.error(str(e))
        st.stop()

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.subheader("Playoff field")
        st.dataframe(df.sort_values(["conference", "seed"]), use_container_width=True, hide_index=True)

        st.subheader("Matchup win probability")
        teams = df["team"].tolist()
        a, b, loc = st.columns([1, 1, 1])
        with a:
            team_a = st.selectbox("Team A", teams, index=0)
        with b:
            team_b = st.selectbox("Team B", [t for t in teams if t != team_a], index=0)
        with loc:
            venue = st.selectbox("Venue", ["Neutral", "Team A home", "Team B home"], index=0)

        ra = float(df.loc[df["team"] == team_a, "rating"].iloc[0])
        rb = float(df.loc[df["team"] == team_b, "rating"].iloc[0])
        hfa_a = params.home_field_elo if venue == "Team A home" else 0.0
        hfa_a = -params.home_field_elo if venue == "Team B home" else hfa_a
        p_a = elo_win_prob(ra, rb, params=params, home_advantage_a=hfa_a)
        st.metric("P(Team A wins)", f"{p_a*100:.1f}%")

    with right:
        st.subheader("Super Bowl win probabilities")
        probs = simulate_playoffs(df, params=params, n_sims=int(n_sims), rng_seed=int(rng_seed))
        winner = probs.iloc[0]
        st.success(f"Prediction (by simulation): **{winner['team']}** ({winner['p_super_bowl']*100:.1f}% to win Super Bowl)")

        st.altair_chart(chart_probs(probs, "p_super_bowl", "Chance to win the Super Bowl"), use_container_width=True)
        st.altair_chart(chart_probs(probs, "p_conf", "Chance to win conference"), use_container_width=True)

        st.subheader("Full table")
        show = probs.copy()
        show["p_conf"] = (show["p_conf"] * 100.0).round(2)
        show["p_super_bowl"] = (show["p_super_bowl"] * 100.0).round(2)
        show = show.rename(columns={"p_conf": "p_conf_%", "p_super_bowl": "p_super_bowl_%"})
        st.dataframe(show, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
