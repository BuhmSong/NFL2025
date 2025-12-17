from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st


REQUIRED_COLS = ["team", "conference", "seed", "rating"]
CONFERENCES = ["AFC", "NFC"]
PLAYOFF_PICTURE_URL = "https://www.nfl.com/standings/playoff-picture"
NFL_API_BASE = "https://api.nfl.com/football/v2"


@dataclass(frozen=True)
class ModelParams:
    k: float = 400.0
    home_field_elo: float = 55.0


@dataclass(frozen=True)
class RatingParams:
    """
    Build an Elo-like rating *from NFL playoff-picture data*.

    Notes:
    - The NFL playoff-picture endpoints we use require authentication; the app can only be as "automatic"
      as the available data allows.
    - If some stats aren't present in the payload, we fall back to a seed-based baseline.
    """

    base_elo: float = 1500.0
    elo_scale_from_win_pct: float = 400.0
    point_diff_elo_per_point_per_game: float = 7.0
    recent_form_elo_scale: float = 140.0
    seed_baseline_step: float = 18.0


@dataclass(frozen=True)
class InjuryParams:
    enabled: bool = True
    qb_multiplier: float = 3.0
    status_elo: Dict[str, float] = None  # filled in __post_init__ pattern below

    def __post_init__(self) -> None:  # type: ignore[override]
        if self.status_elo is None:
            object.__setattr__(
                self,
                "status_elo",
                {
                    "out": -7.0,
                    "doubtful": -4.0,
                    "questionable": -2.0,
                    "ir": -5.0,
                    "inactive": -6.0,
                },
            )


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


def _team_name_from_obj(obj: Any) -> Optional[str]:
    if isinstance(obj, str):
        s = obj.strip()
        return s or None
    if not isinstance(obj, dict):
        return None

    for k in ["displayName", "fullName", "name", "teamName", "team_name", "clubName", "club_name"]:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()

    # Some payloads use nested structures like {"team": {"displayName": ...}}
    for k in ["team", "club", "franchise"]:
        v = obj.get(k)
        name = _team_name_from_obj(v)
        if name:
            return name
    return None


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nfl_playoff_picture_payload(*, conference: str, auth_token: str) -> Any:
    conf = conference.upper()
    if conf not in CONFERENCES:
        raise ValueError(f"conference must be one of {CONFERENCES}")

    url = f"{NFL_API_BASE}/standings/playoffpicture/{conf.lower()}"
    headers = {"Authorization": auth_token.strip()}
    r = requests.get(url, headers=headers, timeout=20)
    if r.status_code == 401:
        raise ValueError(
            "NFL playoff-picture endpoint returned 401 (unauthorized). "
            "This data is not publicly accessible without an Authorization token."
        )
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=300, show_spinner=False)
def fetch_nfl_injuries_week(*, auth_token: str, season: int, season_type: str, week: int) -> Any:
    """
    Fetch weekly injury report.
    Parameters are inferred from the playoff-picture payload where possible.
    """
    url = f"{NFL_API_BASE}/injuries/report/week"
    headers = {"Authorization": auth_token.strip()}
    params = {"season": int(season), "seasonType": str(season_type), "week": int(week)}
    r = requests.get(url, headers=headers, params=params, timeout=20)
    if r.status_code == 401:
        raise ValueError("NFL injuries endpoint returned 401 (unauthorized).")
    r.raise_for_status()
    return r.json()


def _flatten_scalars(obj: Any) -> Iterable[Tuple[str, Any]]:
    """Yield (lower_key, scalar_value) for nested dict/list structures."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            lk = str(k).strip().lower()
            if isinstance(v, (dict, list)):
                yield from _flatten_scalars(v)
            else:
                yield lk, v
    elif isinstance(obj, list):
        for v in obj:
            yield from _flatten_scalars(v)


def _pick_number(flat: Dict[str, List[Any]], *, keys_exact: List[str], keys_contains: List[str] | None = None) -> Optional[float]:
    def as_num(x: Any) -> Optional[float]:
        if isinstance(x, (int, float)):
            return float(x)
        try:
            return float(str(x))
        except Exception:
            return None

    for k in keys_exact:
        lk = k.lower()
        for v in flat.get(lk, []):
            n = as_num(v)
            if n is not None:
                return n

    if keys_contains:
        for lk, values in flat.items():
            if any(part.lower() in lk for part in keys_contains):
                for v in values:
                    n = as_num(v)
                    if n is not None:
                        return n
    return None


def _pick_str(flat: Dict[str, List[Any]], *, keys_exact: List[str], keys_contains: List[str] | None = None) -> Optional[str]:
    for k in keys_exact:
        lk = k.lower()
        for v in flat.get(lk, []):
            if isinstance(v, str) and v.strip():
                return v.strip()
    if keys_contains:
        for lk, values in flat.items():
            if any(part.lower() in lk for part in keys_contains):
                for v in values:
                    if isinstance(v, str) and v.strip():
                        return v.strip()
    return None


def _collect_seed_candidate_dicts(payload: Any) -> List[Dict[str, Any]]:
    candidates: List[Dict[str, Any]] = []

    def seed_from_dict(d: Dict[str, Any]) -> Optional[int]:
        for sk in ["seed", "seednumber", "seed_num", "seedrank", "seed_rank"]:
            if sk in {k.lower(): k for k in d.keys()}:
                # fall through to generic scan below
                break
        for k, v in d.items():
            lk = str(k).strip().lower()
            if lk in {"seed", "seednumber", "seed_num", "seedrank", "seed_rank"}:
                if isinstance(v, (int, float)) and int(v) == v:
                    iv = int(v)
                    if 1 <= iv <= 7:
                        return iv
                try:
                    iv = int(str(v))
                    if 1 <= iv <= 7:
                        return iv
                except Exception:
                    pass
        return None

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            seed = seed_from_dict(x)
            if seed is not None:
                name = _team_name_from_obj(x)
                if name:
                    candidates.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(payload)
    return candidates


def _best_seed_entries(payload: Any) -> Dict[int, Dict[str, Any]]:
    """Pick the 'best' dict for each seed (prefer those with record-like fields)."""
    cand = _collect_seed_candidate_dicts(payload)
    by_seed: Dict[int, List[Dict[str, Any]]] = {}

    def seed_from_dict(d: Dict[str, Any]) -> Optional[int]:
        for k, v in d.items():
            lk = str(k).strip().lower()
            if lk in {"seed", "seednumber", "seed_num", "seedrank", "seed_rank"}:
                try:
                    iv = int(str(v))
                except Exception:
                    return None
                return iv if 1 <= iv <= 7 else None
        return None

    def score(d: Dict[str, Any]) -> int:
        flat: Dict[str, List[Any]] = {}
        for k, v in _flatten_scalars(d):
            flat.setdefault(k, []).append(v)
        s = 0
        # reward record-ish fields
        for k in ["wins", "losses", "ties", "winpercentage", "percentage", "pointsfor", "pointsagainst", "pointdifferential"]:
            if k in flat:
                s += 2
        # reward explicit team identifiers
        for k in ["abbr", "abbreviation", "teamid", "gsisid", "id"]:
            if k in flat:
                s += 1
        return s

    for d in cand:
        seed = seed_from_dict(d)
        if seed is None:
            continue
        by_seed.setdefault(seed, []).append(d)

    best: Dict[int, Dict[str, Any]] = {}
    for seed, ds in by_seed.items():
        best[seed] = sorted(ds, key=score, reverse=True)[0]
    return best


def _rating_from_seed_entry(seed: int, entry: Dict[str, Any], *, rp: RatingParams) -> Tuple[float, Dict[str, Any]]:
    """
    Compute a team rating from whatever record-ish fields exist in the seed entry.
    Returns (rating, debug_dict).
    """
    flat: Dict[str, List[Any]] = {}
    for k, v in _flatten_scalars(entry):
        flat.setdefault(k, []).append(v)

    wins = _pick_number(flat, keys_exact=["wins"], keys_contains=["wins"])
    losses = _pick_number(flat, keys_exact=["losses"], keys_contains=["losses"])
    ties = _pick_number(flat, keys_exact=["ties"], keys_contains=["ties"])
    points_for = _pick_number(flat, keys_exact=["pointsfor"], keys_contains=["pointsfor", "pf"])
    points_against = _pick_number(flat, keys_exact=["pointsagainst"], keys_contains=["pointsagainst", "pa"])
    games = None

    if wins is not None and losses is not None:
        games = wins + losses + (ties or 0.0)

    # Baseline: seed-based
    rating = _default_rating_from_seed(seed, rp=rp)
    debug: Dict[str, Any] = {
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "points_for": points_for,
        "points_against": points_against,
    }

    # Win% to Elo-like rating (logit transform on win pct)
    if wins is not None and losses is not None:
        denom = wins + losses + (ties or 0.0)
        if denom > 0:
            wp = (wins + 0.5 * (ties or 0.0)) / denom
            wp = float(np.clip(wp, 0.05, 0.95))
            elo_from_wp = rp.base_elo + rp.elo_scale_from_win_pct * math.log10(wp / (1.0 - wp))
            rating = elo_from_wp
            debug["win_pct"] = wp
            debug["elo_from_win_pct"] = elo_from_wp

    # Point differential adjustment, if available
    if points_for is not None and points_against is not None and games is not None and games > 0:
        pd_per_game = (points_for - points_against) / games
        pd_adj = pd_per_game * rp.point_diff_elo_per_point_per_game
        rating = rating + pd_adj
        debug["point_diff_per_game"] = pd_per_game
        debug["elo_point_diff_adj"] = pd_adj

    # Recent form: try to find last5 wins/losses if present
    last5_w = _pick_number(flat, keys_exact=["last5wins"], keys_contains=["last5", "last_five", "lastfive"])
    last5_l = _pick_number(flat, keys_exact=["last5losses"], keys_contains=["last5loss", "lastfive"])
    if last5_w is not None and last5_l is not None:
        last5_games = last5_w + last5_l
        if last5_games > 0 and wins is not None and losses is not None:
            last5_wp = last5_w / last5_games
            overall_wp = (wins + 0.5 * (ties or 0.0)) / (wins + losses + (ties or 0.0))
            form_adj = (last5_wp - overall_wp) * rp.recent_form_elo_scale
            rating = rating + form_adj
            debug["last5_win_pct"] = last5_wp
            debug["elo_recent_form_adj"] = form_adj

    return float(rating), debug


def _compute_injury_adjustments(inj_payload: Any, *, ip: InjuryParams) -> Dict[str, float]:
    """
    Heuristic injury impact model from NFL injury payload.
    Returns {team_name: elo_adjustment} (negative numbers mean worse).
    """
    if not ip.enabled:
        return {}

    adj: Dict[str, float] = {}

    def norm_status(s: str) -> str:
        s = s.strip().lower()
        # common normalizations
        if s in {"questionable", "ques"}:
            return "questionable"
        if s in {"doubtful", "doub"}:
            return "doubtful"
        if s in {"out"}:
            return "out"
        if s in {"inactive"}:
            return "inactive"
        if "injured reserve" in s or s == "ir":
            return "ir"
        return s

    def walk(x: Any) -> None:
        if isinstance(x, dict):
            # Identify a likely player-injury record
            flat: Dict[str, List[Any]] = {}
            for k, v in _flatten_scalars(x):
                flat.setdefault(k, []).append(v)

            status = _pick_str(flat, keys_exact=["gameStatus", "gamestatus", "status"], keys_contains=["status"])
            pos = _pick_str(flat, keys_exact=["position", "pos"], keys_contains=["position"])
            if status:
                st_norm = norm_status(status)
                if st_norm in ip.status_elo:
                    team = _team_name_from_obj(x.get("team")) or _team_name_from_obj(x.get("club")) or _team_name_from_obj(x)
                    if team:
                        delta = float(ip.status_elo[st_norm])
                        if pos and pos.strip().upper() == "QB":
                            delta *= float(ip.qb_multiplier)
                        adj[team] = adj.get(team, 0.0) + delta

            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)

    walk(inj_payload)
    return adj


def _infer_season_week_from_payload(payload: Any) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    flat: Dict[str, List[Any]] = {}
    for k, v in _flatten_scalars(payload):
        flat.setdefault(k, []).append(v)
    season = _pick_number(flat, keys_exact=["season"], keys_contains=["season"])
    week = _pick_number(flat, keys_exact=["week"], keys_contains=["week"])
    season_type = _pick_str(flat, keys_exact=["seasonType", "seasontype"], keys_contains=["seasontype"])
    # Normalize seasonType into common NFL strings
    if season_type:
        st = season_type.strip().upper()
        if st in {"REG", "POST", "PRE"}:
            season_type = st
    return (int(season) if season is not None else None), season_type, (int(week) if week is not None else None)


def build_playoff_field_from_nfl(
    *,
    auth_token: str,
    rp: RatingParams,
    ip: InjuryParams,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    payloads = {
        "AFC": fetch_nfl_playoff_picture_payload(conference="AFC", auth_token=auth_token),
        "NFC": fetch_nfl_playoff_picture_payload(conference="NFC", auth_token=auth_token),
    }

    rows: List[Dict[str, Any]] = []
    debug_by_team: Dict[str, Any] = {}

    # Try to infer season/week from the payload and apply injury adjustments.
    season, season_type, week = _infer_season_week_from_payload(payloads["AFC"])
    injury_adj_by_team: Dict[str, float] = {}
    injury_meta: Dict[str, Any] = {"season": season, "seasonType": season_type, "week": week}
    if ip.enabled and season is not None and season_type is not None and week is not None:
        try:
            inj_payload = fetch_nfl_injuries_week(auth_token=auth_token, season=season, season_type=season_type, week=week)
            injury_adj_by_team = _compute_injury_adjustments(inj_payload, ip=ip)
            injury_meta["injuries_loaded"] = True
        except Exception:
            injury_meta["injuries_loaded"] = False
    else:
        injury_meta["injuries_loaded"] = False

    for conf, payload in payloads.items():
        best = _best_seed_entries(payload)
        if sorted(best.keys()) != list(range(1, 8)):
            raise ValueError(
                f"Could not extract a full 1..7 seed list for {conf} (found: {sorted(best.keys())}). "
                "NFL may have changed their payload format."
            )
        for seed in range(1, 8):
            entry = best[seed]
            team = _team_name_from_obj(entry)
            if not team:
                raise ValueError(f"Could not extract team name for {conf} seed {seed}.")
            rating, dbg = _rating_from_seed_entry(seed, entry, rp=rp)
            inj_adj = float(injury_adj_by_team.get(team, 0.0))
            rating = float(rating + inj_adj)
            rows.append({"team": team, "conference": conf, "seed": seed, "rating": rating})
            debug_by_team[team] = {"conference": conf, "seed": seed, "injury_elo_adj": inj_adj, **dbg, **injury_meta}

    df = normalize_input_df(pd.DataFrame(rows))
    validate_playoff_field(df)
    return df.sort_values(["conference", "seed"]).reset_index(drop=True), debug_by_team


def main() -> None:
    st.set_page_config(page_title="Super Bowl Winner Prediction Dashboard", layout="wide")
    st.title("Super Bowl Winner Prediction Dashboard")
    st.caption(f"Playoff picture source: `{PLAYOFF_PICTURE_URL}`")

    with st.sidebar:
        st.header("Data")
        st.caption("This app mirrors the current playoff seeds from NFL and re-runs the simulation automatically.")
        st.caption("Because NFL’s underlying data endpoints require auth, you must provide your own token.")
        nfl_token = st.text_input("NFL Authorization token", type="password", placeholder="Bearer <your_jwt_here>")
        refresh = st.button("Refresh from NFL now", use_container_width=True)
        if refresh:
            fetch_nfl_playoff_picture_payload.clear()
            fetch_nfl_injuries_week.clear()
            st.cache_data.clear()
        st.divider()

        st.header("Model")
        home_field_elo = st.slider("Home-field advantage (Elo points)", 0, 120, 55, 5)
        k = st.slider("Elo scale (higher = less sensitive)", 200, 600, 400, 25)

        st.subheader("Rating inputs (from NFL data)")
        base_elo = st.slider("Base Elo", 1300, 1700, 1500, 10)
        elo_scale_from_wp = st.slider("Win% → Elo scale", 200, 700, 400, 25)
        pd_weight = st.slider("Point diff weight (Elo per point/game)", 0.0, 15.0, 7.0, 0.5)
        form_weight = st.slider("Recent form weight", 0.0, 250.0, 140.0, 10.0)
        seed_step = st.slider("Seed baseline step (fallback)", 0.0, 40.0, 18.0, 1.0)

        st.subheader("Injuries (from NFL)")
        injuries_on = st.toggle("Apply injury adjustments", value=True)
        qb_mult = st.slider("QB injury multiplier", 1.0, 5.0, 3.0, 0.5)

        st.divider()
        st.header("Simulation")
        n_sims = st.slider("Monte Carlo simulations", 500, 50000, 20000, 500)
        rng_seed = st.number_input("Random seed", min_value=0, max_value=10_000_000, value=2025, step=1)

    params = ModelParams(k=float(k), home_field_elo=float(home_field_elo))
    rp = RatingParams(
        base_elo=float(base_elo),
        elo_scale_from_win_pct=float(elo_scale_from_wp),
        point_diff_elo_per_point_per_game=float(pd_weight),
        recent_form_elo_scale=float(form_weight),
        seed_baseline_step=float(seed_step),
    )
    ip = InjuryParams(enabled=bool(injuries_on), qb_multiplier=float(qb_mult))

    if not nfl_token:
        st.info("Enter your NFL `Authorization: Bearer ...` token in the sidebar to load the current playoff picture.")
        st.stop()

    try:
        df, rating_debug = build_playoff_field_from_nfl(auth_token=nfl_token, rp=rp, ip=ip)
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
        st.subheader("Current playoff seeds (from NFL)")
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
            file_name="playoff_field_from_nfl.csv",
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
        st.caption("Home/away is modeled via the Home-field Elo slider. (Injury + point-spread integrations require additional data sources.)")

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
