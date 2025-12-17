from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, TypedDict

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st


REQUIRED_COLS = ["team", "conference", "seed", "rating"]
CONFERENCES = ["AFC", "NFC"]
ESPN_PLAYOFF_STANDINGS_URL = "https://www.espn.com/nfl/standings/_/view/playoff"
ESPN_STANDINGS_API = "https://site.web.api.espn.com/apis/v2/sports/football/nfl/standings"


@dataclass(frozen=True)
class ModelParams:
    k: float = 400.0
    home_field_elo: float = 55.0


@dataclass(frozen=True)
class RatingParams:
    """
    Build an Elo-like rating *from ESPN playoff-standings data*.

    Notes:
    - This only uses what ESPN returns in the public standings endpoint (record + point differential + streak).
    """

    base_elo: float = 1500.0
    elo_scale_from_win_pct: float = 400.0
    point_diff_elo_per_point_per_game: float = 7.0
    streak_elo_per_game: float = 8.0
    seed_baseline_step: float = 18.0


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


def _default_rating_from_seed(seed: int, *, rp: RatingParams) -> float:
    return rp.base_elo + (8 - int(seed)) * rp.seed_baseline_step


@st.cache_data(ttl=300, show_spinner=False)
def fetch_espn_playoff_standings() -> Any:
    """
    Public ESPN endpoint that powers the playoff standings view.
    """
    r = requests.get(ESPN_STANDINGS_API, params={"view": "playoff"}, timeout=20)
    r.raise_for_status()
    return r.json()


def _stats_to_map(stats: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for s in stats:
        name = str(s.get("name") or "").strip()
        if name:
            out[name] = s
    return out


class EspnTeamRow(TypedDict):
    team: str
    conference: str
    seed: int
    rating: float


def _rating_from_espn_stats(seed: int, stats: List[Dict[str, Any]], *, rp: RatingParams) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a team rating from ESPN standings stats.
    Returns (rating, debug_dict).
    """
    stats_map = _stats_to_map(stats)
    wins = float(stats_map.get("wins", {}).get("value", 0.0))
    losses = float(stats_map.get("losses", {}).get("value", 0.0))
    ties = float(stats_map.get("ties", {}).get("value", 0.0))
    win_pct = float(stats_map.get("winPercent", {}).get("value", 0.0))
    point_diff = float(stats_map.get("pointDifferential", {}).get("value", 0.0))
    streak = float(stats_map.get("streak", {}).get("value", 0.0))

    games = wins + losses + ties
    rating = _default_rating_from_seed(seed, rp=rp)
    debug: Dict[str, Any] = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_pct": win_pct,
        "point_diff": point_diff,
        "streak": streak,
    }

    # Win% to Elo-like rating (logit transform on win pct)
    wp = float(np.clip(win_pct if win_pct else (wins + 0.5 * ties) / games if games > 0 else 0.5, 0.05, 0.95))
    elo_from_wp = rp.base_elo + rp.elo_scale_from_win_pct * math.log10(wp / (1.0 - wp))
    rating = elo_from_wp
    debug["elo_from_win_pct"] = elo_from_wp

    # Point differential adjustment (per-game)
    if games > 0:
        pd_per_game = point_diff / games
        pd_adj = pd_per_game * rp.point_diff_elo_per_point_per_game
        rating += pd_adj
        debug["point_diff_per_game"] = pd_per_game
        debug["elo_point_diff_adj"] = pd_adj

    # Recent form proxy: streak (positive = win streak)
    streak_adj = streak * rp.streak_elo_per_game
    rating += streak_adj
    debug["elo_streak_adj"] = streak_adj

    return float(rating), debug


def build_playoff_field_from_espn(*, rp: RatingParams) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    payload = fetch_espn_playoff_standings()
    # Structure: top-level has children for conferences.
    children = payload.get("children", [])
    if not isinstance(children, list) or not children:
        raise ValueError("Unexpected ESPN standings payload: missing 'children'.")

    rows: List[EspnTeamRow] = []
    debug_by_team: Dict[str, Any] = {}

    for conf_obj in children:
        conf_name = str(conf_obj.get("abbreviation") or conf_obj.get("name") or "").strip().upper()
        if conf_name not in CONFERENCES:
            continue
        standings = conf_obj.get("standings", {})
        entries = standings.get("entries", [])
        if not isinstance(entries, list) or not entries:
            raise ValueError(f"Unexpected ESPN standings payload: missing entries for {conf_name}.")

        for entry in entries:
            team = entry.get("team", {}) or {}
            team_name = str(team.get("displayName") or team.get("name") or "").strip()
            if not team_name:
                continue
            stats = entry.get("stats", []) or []
            stats_map = _stats_to_map(stats)
            seed_val = stats_map.get("playoffSeed", {}).get("value")
            try:
                seed = int(seed_val)
            except Exception:
                continue
            if not (1 <= seed <= 7):
                continue

            rating, dbg = _rating_from_espn_stats(seed, stats, rp=rp)
            rows.append({"team": team_name, "conference": conf_name, "seed": seed, "rating": rating})
            debug_by_team[team_name] = {"conference": conf_name, "seed": seed, **dbg}

    df = normalize_input_df(pd.DataFrame(rows))
    validate_playoff_field(df)
    return df.sort_values(["conference", "seed"]).reset_index(drop=True), debug_by_team


def main() -> None:
    st.set_page_config(page_title="Super Bowl Winner Prediction Dashboard", layout="wide")
    st.title("Super Bowl Winner Prediction Dashboard")
    st.caption(f"Data source: `{ESPN_PLAYOFF_STANDINGS_URL}` (via ESPN public standings endpoint)")

    with st.sidebar:
        st.header("Data")
        st.caption("This app mirrors ESPN’s current playoff seeds and updates automatically.")
        refresh = st.button("Refresh from ESPN now", use_container_width=True)
        if refresh:
            fetch_espn_playoff_standings.clear()
            st.cache_data.clear()
        st.divider()

        st.header("Model")
        home_field_elo = st.slider("Home-field advantage (Elo points)", 0, 120, 55, 5)
        k = st.slider("Elo scale (higher = less sensitive)", 200, 600, 400, 25)

        st.subheader("Rating inputs (from ESPN data)")
        base_elo = st.slider("Base Elo", 1300, 1700, 1500, 10)
        elo_scale_from_wp = st.slider("Win% → Elo scale", 200, 700, 400, 25)
        pd_weight = st.slider("Point diff weight (Elo per point/game)", 0.0, 15.0, 7.0, 0.5)
        streak_weight = st.slider("Streak weight (Elo per streak game)", 0.0, 25.0, 8.0, 0.5)
        seed_step = st.slider("Seed baseline step (fallback)", 0.0, 40.0, 18.0, 1.0)
        st.caption("Injuries/point spread aren’t available from this ESPN endpoint; those require additional data sources.")

        st.divider()
        st.header("Simulation")
        n_sims = st.slider("Monte Carlo simulations", 500, 50000, 20000, 500)
        rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=2025, step=1)

    params = ModelParams(k=float(k), home_field_elo=float(home_field_elo))
    rp = RatingParams(
        base_elo=float(base_elo),
        elo_scale_from_win_pct=float(elo_scale_from_wp),
        point_diff_elo_per_point_per_game=float(pd_weight),
        streak_elo_per_game=float(streak_weight),
        seed_baseline_step=float(seed_step),
    )

    try:
        df, rating_debug = build_playoff_field_from_espn(rp=rp)
    except Exception as e:
        st.error(str(e))
        st.stop()

    # Streamlined UI
    top = st.container()
    tabs = st.tabs(["Overview", "Playoff picture", "Matchups", "Debug"])

    with top:
        probs = simulate_playoffs(df, params=params, n_sims=int(n_sims), rng_seed=int(rng_seed))
        winner = probs.iloc[0]
        c1, c2, c3 = st.columns([1.2, 0.9, 0.9])
        with c1:
            st.metric("Predicted Super Bowl winner", str(winner["team"]), f"{winner['p_super_bowl']*100:.1f}%")
        with c2:
            st.metric("Simulations", f"{int(n_sims):,}")
        with c3:
            st.metric("Home-field Elo", f"{params.home_field_elo:.0f}")

    with tabs[0]:
        left, right = st.columns([1, 1], gap="large")
        with left:
            st.subheader("Super Bowl win probabilities")
            st.altair_chart(chart_probs(probs, "p_super_bowl", "Chance to win the Super Bowl"), use_container_width=True)
        with right:
            st.subheader("Conference win probabilities")
            st.altair_chart(chart_probs(probs, "p_conf", "Chance to win conference"), use_container_width=True)

        st.subheader("Full probabilities table")
        show = probs.copy()
        show["p_conf"] = (show["p_conf"] * 100.0).round(2)
        show["p_super_bowl"] = (show["p_super_bowl"] * 100.0).round(2)
        show = show.rename(columns={"p_conf": "p_conf_%", "p_super_bowl": "p_super_bowl_%"})
        st.dataframe(show, use_container_width=True, hide_index=True)

    with tabs[1]:
        st.subheader("Current playoff seeds (from ESPN)")
        afc = df.loc[df["conference"] == "AFC"].sort_values("seed")
        nfc = df.loc[df["conference"] == "NFC"].sort_values("seed")
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown("**AFC**")
            st.dataframe(afc[["seed", "team", "rating"]], use_container_width=True, hide_index=True)
        with c2:
            st.markdown("**NFC**")
            st.dataframe(nfc[["seed", "team", "rating"]], use_container_width=True, hide_index=True)

        st.download_button(
            "Download current field as CSV",
            data=df[REQUIRED_COLS].to_csv(index=False).encode("utf-8"),
            file_name="playoff_field_from_espn.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with tabs[2]:
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
        st.caption("Home/away is modeled via the Home-field Elo slider.")

    with tabs[3]:
        st.subheader("Rating debug (what we extracted from NFL)")
        dbg_df = (
            pd.DataFrame.from_dict(rating_debug, orient="index")
            .reset_index()
            .rename(columns={"index": "team"})
            .sort_values(["conference", "seed"])
        )
        st.dataframe(dbg_df, use_container_width=True, hide_index=True)
        st.caption("If wins/points fields are missing here, NFL likely changed the payload or restricted fields.")


if __name__ == "__main__":
    main()
