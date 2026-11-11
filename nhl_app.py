"""
NHL Game Simulator  (nhl_app.py)
================================
Simulates any NHL game thousands of times using how the two teams have
actually played in their recent games (real NHL play-by-play data):

  * shot rates when leading / tied / trailing (real score effects)
  * power play and penalty kill shot rates, penalties taken
  * where each team's shots come from, and what it allows
  * shooter finishing talent and the goalie's save percentage
  * real end-of-game behaviour: pulled goalies, empty-net goals,
    3-on-3 overtime and shootouts (or playoff sudden-death overtime)

Run it (in PowerShell, NOT the Python >>> prompt):
    python -m pip install -U streamlit plotly pandas numpy requests
    python -m streamlit run nhl_app.py

Data comes from the NHL's free public API (api-web.nhle.com), no key needed.
"""

import base64
import datetime as dt
import io
import json
import math
import re
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import streamlit.components.v1 as components

API = "https://api-web.nhle.com/v1"

st.set_page_config(page_title="NHL Game Simulator", page_icon="🏒", layout="wide")

# --------------------------------------------------------------------------
# Model constants (league-average priors; real team data pulls away from these)
# --------------------------------------------------------------------------
STATES = ("lead", "tied", "trail", "pp", "sh")
MIRROR = {"lead": "trail", "tied": "tied", "trail": "lead", "pp": "sh", "sh": "pp"}
LG_EV = 28.5                      # even-strength shots on goal per 60 (one team)
SCORE_MULT = {"lead": 0.90, "tied": 1.00, "trail": 1.12}   # score effects
LG_RATE = {"lead": LG_EV * 0.90, "tied": LG_EV, "trail": LG_EV * 1.12,
           "pp": 54.0, "sh": 8.0}
SHRINK_MIN = {"lead": 80, "tied": 80, "trail": 80, "pp": 40, "sh": 60}
LG_PEN60 = 3.0                    # minor penalties per team per 60
LG_SH_PCT = 0.095                 # league shooting % on shots on goal
LG_SV = 0.903                     # league save %
PP_GOAL_MULT = 1.30               # power-play shots are more dangerous
HOME_EDGE = 0.035                 # home teams win ~54% in the NHL
OT3_SHOT, OT3_GOAL = 1.6, 1.35    # 3-on-3 overtime is wide open
PULL_DOWN1, PULL_DOWN2 = 150, 210 # seconds left when the goalie gets pulled
PULL_SHOT, PULL_GOAL = 1.7, 1.2   # 6-on-5 attack boost
EN_RATE = 0.0045                  # empty-net goals per second against a pulled goalie
DT = 5                            # simulation time step (seconds)

# Shot-on-goal scoring rate by distance from the net (prior; refit from real shots)
DIST_EDGES = np.array([0, 10, 20, 30, 40, 60, 999.0])
DIST_PRIOR = np.array([0.20, 0.14, 0.09, 0.06, 0.035, 0.015])

DIVISION_OF = {
    **dict.fromkeys(["ANA", "CGY", "EDM", "LAK", "SEA", "SJS", "VAN", "VGK"], "Pacific"),
    **dict.fromkeys(["CHI", "COL", "DAL", "MIN", "NSH", "STL", "UTA", "WPG"], "Central"),
    **dict.fromkeys(["BOS", "BUF", "DET", "FLA", "MTL", "OTT", "TBL", "TOR"], "Atlantic"),
    **dict.fromkeys(["CAR", "CBJ", "NJD", "NYI", "NYR", "PHI", "PIT", "WSH"], "Metropolitan"),
}
CONF_DIVS = {"Western": ("Central", "Pacific"), "Eastern": ("Atlantic", "Metropolitan")}
CONF_OF = {d: c for c, ds in CONF_DIVS.items() for d in ds}

TEAM_COLORS = {
    "ANA": "#F47A38", "BOS": "#FFB81C", "BUF": "#003087", "CGY": "#C8102E",
    "CAR": "#CC0000", "CHI": "#CF0A2C", "COL": "#6F263D", "CBJ": "#002654",
    "DAL": "#006847", "DET": "#CE1126", "EDM": "#FF4C00", "FLA": "#C8102E",
    "LAK": "#111111", "MIN": "#154734", "MTL": "#AF1E2D", "NSH": "#FFB81C",
    "NJD": "#CE1126", "NYI": "#00539B", "NYR": "#0038A8", "OTT": "#C52032",
    "PHI": "#F74902", "PIT": "#FCB514", "SJS": "#006D75", "SEA": "#355464",
    "STL": "#002F87", "TBL": "#002868", "TOR": "#00205B", "UTA": "#6CACE4",
    "VAN": "#00843D", "VGK": "#B4975A", "WSH": "#C8102E", "WPG": "#041E42",
}


def team_colors(home, away):
    """Home/away colours, swapping the away colour if the two are too close."""
    def rgb(h):
        h = h.lstrip("#")
        return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)])
    ch, ca = TEAM_COLORS.get(home, "#1f77b4"), TEAM_COLORS.get(away, "#d62728")
    if np.abs(rgb(ch) - rgb(ca)).sum() < 100:
        ca = "#8A8D8F" if np.abs(rgb(ch) - rgb("#8A8D8F")).sum() > 150 else "#E07A00"
    return ch, ca


# --------------------------------------------------------------------------
# Data fetching
# --------------------------------------------------------------------------
@st.cache_resource
def _session():
    s = requests.Session()
    s.headers["User-Agent"] = "nhl-sim-app/1.0"
    return s


@st.cache_resource
def _pbp_store():
    """Finished games never change, so keep their play-by-play for the whole session."""
    return {}


def get_json(path, session=None):
    s = session or _session()
    r = s.get(f"{API}/{path}", timeout=25)
    r.raise_for_status()
    return r.json()


@st.cache_data(ttl=1800, show_spinner=False)
def fetch(path):
    return get_json(path)


def season_id(day=None):
    day = day or dt.date.today()
    y = day.year if day.month >= 8 else day.year - 1
    return f"{y}{y + 1}"


def prev_season_id():
    s = season_id()
    y = int(s[:4]) - 1
    return f"{y}{y + 1}"


@st.cache_data(ttl=1800, show_spinner=False)
def load_standings():
    data = fetch("standings/now")
    rows = []
    for t in data.get("standings", []):
        ab = t["teamAbbrev"]["default"]
        rows.append(dict(
            team=ab,
            name=(t.get("teamName") or {}).get("default", ab),
            gp=t.get("gamesPlayed", 0) or 0,
            w=t.get("wins", 0), l=t.get("losses", 0), otl=t.get("otLosses", 0),
            pts=t.get("points", 0), gf=t.get("goalFor", 0), ga=t.get("goalAgainst", 0),
            l10=f'{t.get("l10Wins", 0)}-{t.get("l10Losses", 0)}-{t.get("l10OtLosses", 0)}',
            streak=f'{t.get("streakCode", "")}{t.get("streakCount", "")}',
            rw=t.get("regulationWins", t.get("wins", 0)) or 0,
            div=(t.get("divisionName") if t.get("divisionName") in CONF_OF else DIVISION_OF.get(ab, "")),
        ))
    df = pd.DataFrame(rows).sort_values("team").reset_index(drop=True)
    df["conf"] = df["div"].map(CONF_OF).fillna("")
    return df


def _secs(mmss):
    try:
        m, s = str(mmss).split(":")
        return int(m) * 60 + int(s)
    except ValueError:
        return 0


def slim_pbp(js):
    """Keep only what the model needs from a play-by-play feed."""
    plays = []
    for p in js.get("plays", []):
        pdsc = p.get("periodDescriptor") or {}
        if pdsc.get("periodType") == "SO":
            continue
        per = int(pdsc.get("number", 1) or 1)
        d = p.get("details") or {}
        plays.append((
            (per - 1) * 1200 + _secs(p.get("timeInPeriod", "00:00")),
            per,
            p.get("typeDescKey", ""),
            d.get("eventOwnerTeamId"),
            d.get("xCoord"), d.get("yCoord"),
            d.get("shootingPlayerId") or d.get("scoringPlayerId"),
            str(p.get("situationCode") or "1551"),
            d.get("typeCode"),
            d.get("goalieInNetId"),
        ))
    plays.sort(key=lambda r: r[0])
    home, away = js.get("homeTeam") or {}, js.get("awayTeam") or {}
    return dict(
        id=js.get("id"), date=js.get("gameDate", ""),
        home=home.get("abbrev"), away=away.get("abbrev"),
        home_id=home.get("id"), away_id=away.get("id"),
        home_score=home.get("score"), away_score=away.get("score"),
        plays=plays,
        roster=[(r.get("playerId"), r.get("teamId")) for r in js.get("rosterSpots", [])],
    )


def fetch_pbps(ids):
    store, sess = _pbp_store(), _session()
    missing = [i for i in ids if i not in store]

    def one(gid):
        try:
            return gid, slim_pbp(get_json(f"gamecenter/{gid}/play-by-play", sess))
        except Exception:
            return gid, None

    if missing:
        with ThreadPoolExecutor(max_workers=8) as ex:
            for gid, g in ex.map(one, missing):
                if g is not None:
                    store[gid] = g
    return [store[i] for i in ids if i in store]


def recent_game_ids(team, n):
    """Last n finished regular-season games (topped up from last season early on)."""
    def finished(path):
        try:
            games = fetch(path).get("games", [])
        except Exception:
            return []
        games = [g for g in games
                 if g.get("gameType") == 2 and g.get("gameState") in ("OFF", "FINAL")]
        return sorted(games, key=lambda g: g.get("gameDate", ""))

    games = finished(f"club-schedule-season/{team}/now")
    if len(games) < n:
        games = finished(f"club-schedule-season/{team}/{prev_season_id()}") + games
    return [g["id"] for g in games[-n:]]


def classify(sit, is_home, diff):
    """Game state from one team's point of view, using the NHL situation code
    (away goalie, away skaters, home skaters, home goalie)."""
    s = str(sit).zfill(4)
    try:
        ag, a_sk, h_sk, hg = (int(c) for c in s[:4])
    except ValueError:
        return "tied"
    own_g, own_s, opp_g, opp_s = (hg, h_sk, ag, a_sk) if is_home else (ag, a_sk, hg, h_sk)
    if own_g == 0 or opp_g == 0:
        return "en"
    if own_s > opp_s:
        return "pp"
    if own_s < opp_s:
        return "sh"
    return "lead" if diff > 0 else ("tied" if diff == 0 else "trail")


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def build_profile(team, n_recent):
    """Everything the simulator knows about one team, from real data."""
    ids = recent_game_ids(team, n_recent)
    games = fetch_pbps(ids)

    toi = dict.fromkeys(STATES, 0.0)
    sf, sa = dict.fromkeys(STATES, 0), dict.fromkeys(STATES, 0)
    shots_for, shots_ag = [], []
    gf_per, ga_per = np.zeros(4), np.zeros(4)
    sf_per, sa_per = np.zeros(4), np.zeros(4)
    ppg = ppga = pens = drawn = fo = fow = hits = 0
    starters, recent_games, lineups = [], [], []
    team_id = None

    for g in games:
        is_home = g["home"] == team
        tid = g["home_id"] if is_home else g["away_id"]
        team_id = tid
        us = them = 0
        prev_t, prev_state, starter = 0, None, None
        for (t, per, typ, owner, x, y, _shooter, sit, tcode, goalie) in g["plays"]:
            if prev_state in toi and per <= 3:
                toi[prev_state] += max(0, t - prev_t)
            prev_t = t
            state = classify(sit, is_home, us - them) if per <= 3 else "ot"
            if typ in ("shot-on-goal", "goal"):
                ours = owner == tid
                if not ours and starter is None and goalie:
                    starter = goalie
                if state in toi:
                    (sf if ours else sa)[state] += 1
                (sf_per if ours else sa_per)[min(per, 4) - 1] += 1
                if x is not None and y is not None and state not in ("en", "ot"):
                    nx, ny = (x, y) if x >= 0 else (-x, -y)
                    rec = (nx, ny, math.hypot(89 - nx, ny), typ == "goal")
                    (shots_for if ours else shots_ag).append(rec)
                if typ == "goal":
                    if ours:
                        us += 1
                        gf_per[min(per, 4) - 1] += 1
                        ppg += state == "pp"
                    else:
                        them += 1
                        ga_per[min(per, 4) - 1] += 1
                        ppga += state == "sh"
            elif typ == "penalty" and tcode in ("MIN", "BEN", "MAJ"):
                if owner == tid:
                    pens += 1
                else:
                    drawn += 1
            elif typ == "faceoff":
                fo += 1
                fow += owner == tid
            elif typ == "hit" and owner == tid:
                hits += 1
            prev_state = state
        starters.append(starter)
        lineups.append({pid for pid, t_ in g["roster"] if t_ == tid})
        opp = g["away"] if is_home else g["home"]
        mine, theirs = (g["home_score"], g["away_score"]) if is_home else (g["away_score"], g["home_score"])
        if mine is None:
            mine, theirs = us, them
        recent_games.append(dict(date=g["date"], opp=("vs " if is_home else "@ ") + str(opp),
                                 result=f"{'W' if mine > theirs else 'L'} {mine}-{theirs}"))

    # Shrunk shot rates (per 60) for and against in every state
    r_for, r_ag = {}, {}
    for s in STATES:
        mins = toi[s] / 60
        k = SHRINK_MIN[s]
        r_for[s] = 60 * (sf[s] + LG_RATE[s] / 60 * k) / (mins + k)
        r_ag[s] = 60 * (sa[s] + LG_RATE[MIRROR[s]] / 60 * k) / (mins + k)
    reg_min = sum(toi.values()) / 60 + 1e-9
    pen60 = 60 * (pens + LG_PEN60 / 60 * 300) / (reg_min + 300)

    # Season player stats for this club
    try:
        cs = fetch(f"club-stats/{team}/now")
    except Exception:
        cs = {}
    skaters = []
    for s in cs.get("skaters", []):
        gp = s.get("gamesPlayed", 0) or 0
        if gp <= 0:
            continue
        skaters.append(dict(
            id=s.get("playerId"),
            name=f'{(s.get("firstName") or {}).get("default", "")} {(s.get("lastName") or {}).get("default", "")}'.strip(),
            pos=s.get("positionCode", "F"), gp=gp,
            g=s.get("goals", 0) or 0, a=s.get("assists", 0) or 0,
            shots=s.get("shots", 0) or 0, ppg=s.get("powerPlayGoals", 0) or 0,
            toi=float(s.get("avgTimeOnIcePerGame", 0) or 0),
            pim=s.get("penaltyMinutes", 0) or 0,
        ))
    goalies = []
    for gk in cs.get("goalies", []):
        sa_ = gk.get("shotsAgainst", 0) or 0
        svp = gk.get("savePercentage", 0) or 0
        goalies.append(dict(
            id=gk.get("playerId"),
            name=f'{(gk.get("firstName") or {}).get("default", "")} {(gk.get("lastName") or {}).get("default", "")}'.strip(),
            gp=gk.get("gamesPlayed", 0) or 0, gs=gk.get("gamesStarted", 0) or 0,
            sa=sa_, saves=gk.get("saves", round(sa_ * svp)) or 0,
            svp=svp, gaa=gk.get("goalsAgainstAverage", 0) or 0,
        ))

    n = max(len(games), 1)
    return dict(
        team=team, team_id=team_id, n_games=len(games), recent_games=recent_games,
        toi=toi, sf=sf, sa=sa, r_for=r_for, r_ag=r_ag, pen60=pen60,
        shots_for=shots_for, shots_ag=shots_ag,
        gf_per=(gf_per / n).tolist(), ga_per=(ga_per / n).tolist(),
        sf_per=(sf_per / n).tolist(), sa_per=(sa_per / n).tolist(),
        pp_pct=ppg / drawn if drawn else None, pk_pct=1 - ppga / pens if pens else None,
        pens_pg=pens / n, fo_pct=fow / fo if fo else None, hits_pg=hits / n,
        sog_pg=sum(sf_per) / n, soga_pg=sum(sa_per) / n,
        gf_pg=sum(gf_per) / n, ga_pg=sum(ga_per) / n,
        skaters=skaters, goalies=goalies, starters=starters,
        recent_lineup=set().union(*lineups[-2:]) if lineups else set(),
    )


def default_lineup(prof):
    """Probable starting goalie and players not in the recent lineup."""
    gids = {g["id"] for g in prof["goalies"]}
    recent = [s for s in prof["starters"][-5:] if s in gids]
    if recent:
        c = Counter(recent)
        top = max(c.values())
        goalie = next(s for s in reversed(recent) if c[s] == top)
    elif prof["goalies"]:
        goalie = max(prof["goalies"], key=lambda g: (g["gs"], g["gp"]))["id"]
    else:
        goalie = None
    lineup = prof["recent_lineup"]
    out = tuple(sorted(s["id"] for s in prof["skaters"] if lineup and s["id"] not in lineup))
    return goalie, out


# --------------------------------------------------------------------------
# Building the two sides for the simulation
# --------------------------------------------------------------------------
def dist_model(dists, goals):
    b = np.clip(np.digitize(dists, DIST_EDGES) - 1, 0, 5)
    n = np.bincount(b, minlength=6)
    g = np.bincount(b, weights=goals.astype(float), minlength=6)
    return (g + DIST_PRIOR * 150) / (n + 150)


def _pool(shots, model):
    if len(shots) < 20:   # not enough real data yet: generic slot-heavy pool
        rng = np.random.default_rng(7)
        x = np.clip(rng.normal(70, 12, 300), 30, 88)
        y = np.clip(rng.normal(0, 13, 300), -38, 38)
        d = np.hypot(89 - x, y)
    else:
        arr = np.array(shots, dtype=float)
        x, y, d = arr[:, 0], arr[:, 1], arr[:, 2]
    b = np.clip(np.digitize(d, DIST_EDGES) - 1, 0, 5)
    return x, y, model[b]


def _dressed(prof, out_ids):
    sk = [s for s in prof["skaters"] if s["id"] not in set(out_ids)]
    fw = sorted([s for s in sk if s["pos"] != "D"], key=lambda s: -s["toi"])[:12]
    dm = sorted([s for s in sk if s["pos"] == "D"], key=lambda s: -s["toi"])[:6]
    return fw + dm


def _gk_factor(gk):
    """How many goals this goalie allows relative to a league-average goalie (shrunk)."""
    sv = (gk["saves"] + LG_SV * 600) / (gk["sa"] + 600)
    return float(np.clip((1 - sv) / (1 - LG_SV), 0.7, 1.4))


def build_side(prof, opp, goalie_id, out_ids, model):
    dressed = _dressed(prof, out_ids)
    if not dressed:
        dressed = [dict(id=-1, name=f"{prof['team']} skater", pos="F", gp=1, g=0, a=0, shots=2, ppg=0, toi=900)]
    gp = np.array([max(s["gp"], 1) for s in dressed], float)
    shots = np.array([s["shots"] for s in dressed], float)
    spg = shots / gp + 0.3
    toi = np.array([s["toi"] for s in dressed], float) + 1
    w_ev = spg / spg.sum()
    w_pp = spg * (toi / toi.mean()) ** 2
    w_pp /= w_pp.sum()
    w_ast = np.array([s["a"] for s in dressed], float) / gp + 0.05
    w_ast /= w_ast.sum()
    w_pen = np.array([s.get("pim", 0) for s in dressed], float) / gp + 0.3
    w_pen /= w_pen.sum()
    goals = np.array([s["g"] for s in dressed], float)
    fin = np.clip((goals + LG_SH_PCT * 120) / (shots + 120) / LG_SH_PCT, 0.6, 1.6)

    # Lineup strength vs the lineup the team has actually been using
    base = _dressed(prof, default_lineup(prof)[1])
    q0 = sum(s["shots"] / max(s["gp"], 1) for s in base) or 1
    q1 = sum(s["shots"] / max(s["gp"], 1) for s in dressed)
    off = float(np.clip(1 + 0.5 * (q1 / q0 - 1), 0.85, 1.15)) if base else 1.0

    gk = next((g for g in prof["goalies"] if g["id"] == goalie_id), None)
    if gk:
        gk_factor = _gk_factor(gk)
        gk_name = gk["name"]
    else:
        gk_factor, gk_name = 1.0, "Unknown goalie"

    rf, ra = prof["r_for"], opp["r_ag"]
    r_ev = np.array([rf[s] * ra[MIRROR[s]] / LG_RATE[s] for s in ("lead", "tied", "trail")])
    r_pp = rf["pp"] * ra["sh"] / LG_RATE["pp"]
    r_sh = rf["sh"] * ra["pp"] / LG_RATE["sh"]

    fx, fy, fp = _pool(prof["shots_for"], model)
    ax, ay, ap = _pool(prof["shots_ag"], model)
    return dict(
        team=prof["team"], names=[s["name"] for s in dressed], pos=[s["pos"] for s in dressed],
        ids=[s["id"] for s in dressed],
        w_ev=w_ev, w_pen=w_pen, w_pp=w_pp, w_ast=w_ast, fin=fin,
        r_ev=r_ev * off / 3600, r_pp=r_pp * off / 3600, r_sh=r_sh * off / 3600,
        pen_rate=prof["pen60"] / 3600, gk_factor=gk_factor, gk_name=gk_name, off=off,
        loc_for=(fx, fy, fp), loc_ag=(ax, ay, ap),
    )


# --------------------------------------------------------------------------
# The simulation engine (all games run side by side with numpy)
# --------------------------------------------------------------------------
def simulate(home, away, n, playoff=False, record=400, seed=None):
    rng = np.random.default_rng(seed)
    T = [home, away]
    P = [len(home["names"]), len(away["names"])]
    goals = np.zeros((n, 2), np.int32)
    sog = np.zeros((n, 2), np.int32)
    ppg = np.zeros((n, 2), np.int32)
    ppo = np.zeros((n, 2), np.int32)
    eng = np.zeros((n, 2), np.int32)
    so_goal = np.zeros((n, 2), np.int32)
    pen = np.zeros((n, 2), np.int32)          # seconds left that team is shorthanded
    g_mat = [np.zeros((n, p), np.int16) for p in P]
    a_mat = [np.zeros((n, p), np.int16) for p in P]
    s_mat = [np.zeros((n, p), np.int16) for p in P]
    goals_by_per = np.zeros((2, 4))
    sog_by_min = np.zeros((2, 66))
    events = []                                # detailed log for the first `record` games

    def assists(team, scorers):
        k, pn, w = len(scorers), P[team], T[team]["w_ast"]
        a1 = np.full(k, -1)
        a2 = np.full(k, -1)
        if pn < 2:
            return a1, a2
        a1 = rng.choice(pn, k, p=w)
        for _ in range(8):
            bad = a1 == scorers
            if not bad.any():
                break
            a1[bad] = rng.choice(pn, bad.sum(), p=w)
        a1[(a1 == scorers) | (rng.random(k) < 0.07)] = -1
        a2 = rng.choice(pn, k, p=w)
        for _ in range(8):
            bad = (a2 == scorers) | (a2 == a1)
            if not bad.any():
                break
            a2[bad] = rng.choice(pn, bad.sum(), p=w)
        a2[(a2 == scorers) | (a2 == a1) | (a1 < 0) | (rng.random(k) < 0.22)] = -1
        return a1, a2

    def log(sims, t, team, kind, players, a1, a2, xs, ys, strength):
        for j in np.nonzero(sims < record)[0]:
            s = sims[j]
            events.append((int(s), int(t), team, kind, int(players[j]), int(a1[j]), int(a2[j]),
                           float(xs[j]), float(ys[j]), int(goals[s, 0]), int(goals[s, 1]),
                           strength[j] if isinstance(strength, np.ndarray) else strength))

    def score(team, sims, scorers, is_pp, xs, ys, t, strength, empty_net=False):
        opp = 1 - team
        goals[sims, team] += 1
        np.add.at(g_mat[team], (sims, scorers), 1)
        a1, a2 = assists(team, scorers)
        for a in (a1, a2):
            ok = a >= 0
            np.add.at(a_mat[team], (sims[ok], a[ok]), 1)
        ppg[sims[is_pp], team] += 1
        pen[sims[is_pp], opp] = 0                      # power-play goal ends the minor
        if empty_net:
            eng[sims, team] += 1
        goals_by_per[team, min(t // 1200, 3)] += len(sims)
        log(sims, t, team, "goal", scorers, a1, a2, xs, ys, strength)

    def step(t, active, mode):
        idx = np.nonzero(active)[0]
        k = idx.size
        if k == 0:
            return
        g0, g1 = goals[idx, 0].copy(), goals[idx, 1].copy()
        shorthanded = pen[idx] > 0
        pulled = np.zeros((k, 2), bool)
        if mode == "reg" and t >= 2400:
            rem, d = 3600 - t, g0 - g1
            pulled[:, 0] = ((d == -1) & (rem <= PULL_DOWN1)) | ((d == -2) & (rem <= PULL_DOWN2))
            pulled[:, 1] = ((d == 1) & (rem <= PULL_DOWN1)) | ((d == 2) & (rem <= PULL_DOWN2))
        for team in (0, 1):
            opp, tm, om = 1 - team, T[team], T[1 - team]
            d = (g0 - g1) if team == 0 else (g1 - g0)
            own_sh, opp_sh = shorthanded[:, team], shorthanded[:, opp]
            pp = opp_sh & ~own_sh
            sh = own_sh & ~opp_sh
            si = np.where(d > 0, 0, np.where(d == 0, 1, 2))
            if mode == "reg":
                rate = np.where(pp, tm["r_pp"], np.where(sh, tm["r_sh"], tm["r_ev"][si]))
                gmult = np.where(pp, PP_GOAL_MULT, 1.0)
            elif mode == "ot3":
                rate = np.full(k, tm["r_ev"][1] * OT3_SHOT)
                gmult = np.full(k, OT3_GOAL)
            else:
                rate = np.full(k, tm["r_ev"][1])
                gmult = np.ones(k)
            own_pull, opp_pull = pulled[:, team], pulled[:, opp]
            rate = np.where(own_pull, rate * PULL_SHOT, rate)
            gmult = np.where(own_pull, gmult * PULL_GOAL, gmult)
            u = rng.random(k)
            shot = (u < rate * DT) & ~opp_pull
            en = opp_pull & (u < EN_RATE * DT)
            strength = np.where(pp, "PP", np.where(sh, "SH", np.where(own_pull, "6v5", "EV")))
            if mode == "ot3":
                strength = np.full(k, "3v3")

            j = np.nonzero(shot)[0]
            if j.size:
                sims, kk = idx[j], j.size
                use_for = rng.random(kk) < 0.5
                fx, fy, fp = tm["loc_for"]
                ax, ay, ap = om["loc_ag"]
                lf, la = rng.integers(0, len(fx), kk), rng.integers(0, len(ax), kk)
                xs = np.where(use_for, fx[lf], ax[la]) + rng.normal(0, 1.5, kk)
                ys = np.where(use_for, fy[lf], ay[la]) + rng.normal(0, 1.5, kk)
                base = np.where(use_for, fp[lf], ap[la])
                is_pp = pp[j]
                shooter = np.where(is_pp, rng.choice(P[team], kk, p=tm["w_pp"]),
                                   rng.choice(P[team], kk, p=tm["w_ev"]))
                p = base * gmult[j] * tm["fin"][shooter] * om["gk_factor"]
                goal = rng.random(kk) < np.clip(p, 0, 0.7)
                sog[sims, team] += 1
                np.add.at(s_mat[team], (sims, shooter), 1)
                sog_by_min[team, min(t // 60, 65)] += kk
                miss = ~goal
                none = np.full(kk, -1)
                log(sims[miss], t, team, "shot", shooter[miss], none[miss], none[miss],
                    xs[miss], ys[miss], strength[j][miss])
                if goal.any():
                    score(team, sims[goal], shooter[goal], is_pp[goal], xs[goal], ys[goal],
                          t, strength[j][goal])
            j = np.nonzero(en)[0]
            if j.size:
                sims, kk = idx[j], j.size
                scorer = rng.choice(P[team], kk, p=tm["w_ev"])
                sog[sims, team] += 1
                np.add.at(s_mat[team], (sims, scorer), 1)
                score(team, sims, scorer, np.zeros(kk, bool), rng.uniform(-20, 40, kk),
                      rng.uniform(-30, 30, kk), t, "EN", empty_net=True)

            if mode == "reg" and t < 3600 - 30:
                j = np.nonzero(rng.random(k) < tm["pen_rate"] * DT)[0]
                if j.size:
                    sims = idx[j]
                    new = pen[sims, team] == 0
                    ppo[sims[new], opp] += 1
                    pen[sims, team] = np.minimum(pen[sims, team] + 120, 240)
                    pl = rng.choice(P[team], j.size, p=tm["w_pen"])
                    none = np.full(j.size, -1)
                    log(sims, t, team, "pen", pl, none, none, np.zeros(j.size), np.zeros(j.size), "")
        pen[idx] = np.maximum(pen[idx] - DT, 0)

    everyone = np.ones(n, bool)
    for s in range(3600 // DT):
        step(s * DT, everyone, "reg")
    pen[:] = 0
    tied_after_reg = goals[:, 0] == goals[:, 1]

    if not playoff:
        for s in range(300 // DT):
            step(3600 + s * DT, goals[:, 0] == goals[:, 1], "ot3")
        still = goals[:, 0] == goals[:, 1]
        p_home = float(np.clip(0.5 + 0.15 * (away["gk_factor"] - home["gk_factor"]), 0.35, 0.65))
        home_wins = rng.random(n) < p_home
        for team, mask in ((0, still & home_wins), (1, still & ~home_wins)):
            goals[mask, team] += 1
            so_goal[mask, team] += 1
            sims = np.nonzero(mask)[0]
            for s in sims[sims < record]:
                events.append((int(s), 3900, team, "so", -1, -1, -1, 0.0, 0.0,
                               int(goals[s, 0]), int(goals[s, 1]), "SO"))
        so_dec = still
    else:
        for per in range(8):
            for s in range(1200 // DT):
                step(3600 + per * 1200 + s * DT, goals[:, 0] == goals[:, 1], "ot5")
        still = goals[:, 0] == goals[:, 1]
        coin = rng.random(n) < 0.5
        goals[still & coin, 0] += 1
        goals[still & ~coin, 1] += 1
        so_dec = np.zeros(n, bool)

    return dict(goals=goals, sog=sog, ppg=ppg, ppo=ppo, eng=eng, so_goal=so_goal,
                ot_dec=tied_after_reg & ~so_dec, so_dec=so_dec, reg_dec=~tied_after_reg,
                g_mat=g_mat, a_mat=a_mat, s_mat=s_mat, goals_by_per=goals_by_per / n,
                sog_by_min=sog_by_min / n, events=events, n=n, record=min(record, n))


# --------------------------------------------------------------------------
# Fast team-level engine (used for the season, playoffs and win probability).
# Same rules as the detailed engine, but each simulated row can be a different
# matchup and start from any clock time and score. Only goals are tracked.
# --------------------------------------------------------------------------
LG3 = np.array([LG_RATE["lead"], LG_RATE["tied"], LG_RATE["trail"]])


def team_vector(prof, model):
    """One team's real-data profile boiled down to the numbers the fast engine needs."""
    _, out = default_lineup(prof)
    dressed = _dressed(prof, out)
    if dressed:
        gp = np.array([max(s["gp"], 1) for s in dressed], float)
        shots = np.array([s["shots"] for s in dressed], float)
        goals = np.array([s["g"] for s in dressed], float)
        toi = np.array([s["toi"] for s in dressed], float) + 1
        spg = shots / gp + 0.3
        w_ev = spg / spg.sum()
        w_pp = spg * (toi / toi.mean()) ** 2
        w_pp /= w_pp.sum()
        fin = np.clip((goals + LG_SH_PCT * 120) / (shots + 120) / LG_SH_PCT, 0.6, 1.6)
        fin_ev, fin_pp = float(w_ev @ fin), float(w_pp @ fin)
    else:
        fin_ev = fin_pp = 1.0
    gks = prof["goalies"]
    if gks:   # season-long: goalies split starts the way they have been
        w = np.array([g["gs"] + 0.5 for g in gks], float)
        gk = float(np.dot(w, [_gk_factor(g) for g in gks]) / w.sum())
    else:
        gk = 1.0
    return dict(F=np.array([prof["r_for"][s] for s in STATES]), Ag=np.array([prof["r_ag"][s] for s in STATES]),
                pen=prof["pen60"] / 3600, qf=float(_pool(prof["shots_for"], model)[2].mean()),
                qa=float(_pool(prof["shots_ag"], model)[2].mean()), fin=fin_ev, fin_pp=fin_pp, gk=gk)


def stack(vecs, idx):
    return {k: np.array([v[k] for v in vecs])[idx] for k in vecs[0]}


def pair_sides(H, A, edge=HOME_EDGE):
    """Per-row rates for home team H vs away team A (both dicts of stacked arrays)."""
    def one(o, d, mult):
        qb = 0.5 * o["qf"] + 0.5 * d["qa"]
        return dict(
            r_ev=o["F"][:, :3] * d["Ag"][:, [2, 1, 0]] / LG3 * mult / 3600,
            r_pp=o["F"][:, 3] * d["Ag"][:, 4] / LG_RATE["pp"] * mult / 3600,
            r_sh=o["F"][:, 4] * d["Ag"][:, 3] / LG_RATE["sh"] * mult / 3600,
            q=qb * o["fin"] * d["gk"], q_pp=qb * o["fin_pp"] * d["gk"] * PP_GOAL_MULT,
            pen=o["pen"], gk=o["gk"])
    return [one(H, A, 1 + edge), one(A, H, 1 - edge)]


def sides_from_main(H, A, rows):
    """Fast-engine rows built from the detailed engine's two sides (same lineups/goalies)."""
    def one(o, d):
        qb = 0.5 * o["loc_for"][2].mean() + 0.5 * d["loc_ag"][2].mean()
        one_row = dict(r_ev=o["r_ev"][None, :], r_pp=np.array([o["r_pp"]]), r_sh=np.array([o["r_sh"]]),
                       q=np.array([qb * (o["w_ev"] @ o["fin"]) * d["gk_factor"]]),
                       q_pp=np.array([qb * (o["w_pp"] @ o["fin"]) * d["gk_factor"] * PP_GOAL_MULT]),
                       pen=np.array([o["pen_rate"]]), gk=np.array([o["gk_factor"]]))
        return {k: np.repeat(v, rows, axis=0) for k, v in one_row.items()}
    return [one(H, A), one(A, H)]


def fast_reg(S, start_t=None, start_g=None, seed=None):
    """Play regulation for every row; returns final goals (rows, 2)."""
    rng = np.random.default_rng(seed)
    n = len(S[0]["q"])
    st_t = np.zeros(n, int) if start_t is None else np.asarray(start_t, int)
    order = np.argsort(st_t, kind="stable")
    st_s = st_t[order]
    Ss = [{k: v[order] for k, v in side.items()} for side in S]
    goals = (np.zeros((n, 2), np.int32) if start_g is None
             else np.asarray(start_g, np.int32)[order].copy())
    pen = np.zeros((n, 2), np.int32)
    ar = np.arange(n)
    for t in range(0, 3600, DT):
        k = int(np.searchsorted(st_s, t, side="right"))
        if k == 0:
            continue
        g, pn = goals[:k], pen[:k]
        d0 = g[:, 0] - g[:, 1]
        shd = pn > 0
        pulls = None
        if t >= 2400:
            rem = 3600 - t
            pulls = (((d0 == -1) & (rem <= PULL_DOWN1)) | ((d0 == -2) & (rem <= PULL_DOWN2)),
                     ((d0 == 1) & (rem <= PULL_DOWN1)) | ((d0 == 2) & (rem <= PULL_DOWN2)))
        for team in (0, 1):
            opp, tm = 1 - team, Ss[team]
            d = d0 if team == 0 else -d0
            pp = shd[:, opp] & ~shd[:, team]
            sh = shd[:, team] & ~shd[:, opp]
            si = np.where(d > 0, 0, np.where(d == 0, 1, 2))
            rate = np.where(pp, tm["r_pp"][:k], np.where(sh, tm["r_sh"][:k], tm["r_ev"][:k][ar[:k], si]))
            q = np.where(pp, tm["q_pp"][:k], tm["q"][:k])
            if pulls is not None:
                own, other = pulls[team], pulls[opp]
                rate = np.where(own, rate * PULL_SHOT, rate)
                q = np.where(own, q * PULL_GOAL, q)
                pg = np.where(other, EN_RATE * DT, rate * DT * np.minimum(q, 0.7))
            else:
                pg = rate * DT * np.minimum(q, 0.7)
            sc = rng.random(k) < pg
            g[sc, team] += 1
            pn[sc & pp, opp] = 0
            if t < 3570:
                take = rng.random(k) < tm["pen"][:k] * DT
                pn[take, team] = np.minimum(pn[take, team] + 120, 240)
        np.maximum(pn - DT, 0, out=pn)
    inv = np.empty(n, int)
    inv[order] = np.arange(n)
    return goals[inv]


def ot_home_prob(S, playoff, rem=300.0):
    """Chance the home team wins a game that is tied going into (the rest of) overtime."""
    lam = []
    for tm in S:
        base = tm["r_ev"][:, 1] * np.minimum(tm["q"], 0.7)
        lam.append(base if playoff else base * OT3_SHOT * OT3_GOAL)
    tot = lam[0] + lam[1] + 1e-12
    share = lam[0] / tot
    if playoff:
        return share
    p_goal = 1 - np.exp(-tot * rem)
    p_so = np.clip(0.5 + 0.15 * (S[1]["gk"] - S[0]["gk"]), 0.35, 0.65)
    return share * p_goal + (1 - p_goal) * p_so


def win_prob_curve(H, A, feat, playoff, reps=1500, seed=5):
    """Home win probability through the featured game, re-simulated from each moment."""
    ev = feat["events"]
    goals_ev = [e for e in ev if e[3] == "goal"]
    end = max([e[1] for e in ev] + [3600])
    cps = set(range(0, 3600, 60)) | {e[1] + DT for e in goals_ev if e[1] + DT < 3600}
    if end > 3600:
        cps |= set(range(3600, end, 30))
    cps = sorted(cps)

    def score_at(t):
        last = (0, 0)
        for e in goals_ev:
            if e[1] < t:
                last = (e[9], e[10])
        return last

    scores = [score_at(t) for t in cps]
    reg = [i for i, t in enumerate(cps) if t < 3600]
    probs = np.zeros(len(cps))
    S = sides_from_main(H, A, len(reg) * reps)
    g = fast_reg(S, np.repeat([cps[i] for i in reg], reps),
                 np.repeat(np.array([scores[i] for i in reg]).reshape(-1, 2), reps, axis=0), seed=seed)
    d = g[:, 0] - g[:, 1]
    p = (d > 0) + (d == 0) * ot_home_prob(S, playoff)
    probs[reg] = p.reshape(len(reg), reps).mean(1)
    S1 = sides_from_main(H, A, 1)
    for i, t in enumerate(cps):
        if t >= 3600:
            hs, as_ = scores[i]
            if hs != as_:
                probs[i] = float(hs > as_)
            else:
                probs[i] = float(ot_home_prob(S1, playoff, rem=max(3900 - t, 1))[0])
    final = feat["final"]
    times = [t / 60 for t in cps] + [end / 60 + 0.5]
    return dict(t=times, home=list(probs) + [1.0 if final[0] > final[1] else 0.0])


def series_outcomes(p_home, p_away):
    """Best-of-7 (2-2-1-1-1). The 'hi' team has home ice; p_home = its win chance
    at home, p_away = on the road. Returns {('hi'|'lo', games): probability}."""
    venue = (1, 1, 0, 0, 1, 0, 1)
    out = {(w, n): 0.0 for w in ("hi", "lo") for n in (4, 5, 6, 7)}

    def rec(w, l, pr):
        g = w + l
        if w == 4:
            out[("hi", g)] += pr
            return
        if l == 4:
            out[("lo", g)] += pr
            return
        p = p_home if venue[g] else p_away
        rec(w + 1, l, pr * p)
        rec(w, l + 1, pr * (1 - p))

    rec(0, 0, 1.0)
    return out


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def league_model(n_recent, reps=300):
    """Win/OT/loss chances for every possible home-away pairing of the 32 teams."""
    teams = load_standings()["team"].tolist()
    ids = sorted({i for t in teams for i in recent_game_ids(t, n_recent)})
    fetch_pbps(ids)
    profs = {t: build_profile(t, n_recent) for t in teams}
    allshots = [s for p in profs.values() for s in p["shots_for"]]
    if allshots:
        arr = np.array(allshots, dtype=float)
        model = dist_model(arr[:, 2], arr[:, 3])
    else:
        model = DIST_PRIOR.copy()
    vecs = [team_vector(profs[t], model) for t in teams]
    N = len(teams)
    hi, ai = np.array([(i, j) for i in range(N) for j in range(N) if i != j]).T
    S = pair_sides(stack(vecs, np.repeat(hi, reps)), stack(vecs, np.repeat(ai, reps)))
    g = fast_reg(S, seed=11)
    d = g[:, 0] - g[:, 1]
    tie = d == 0
    ph_reg, ph_po = ot_home_prob(S, False), ot_home_prob(S, True)

    def agg(x):
        return np.asarray(x, float).reshape(-1, reps).mean(1)

    R = np.zeros((N, N, 4))       # home reg win, home OT/SO win, away OT/SO win, away reg win
    R[hi, ai, 0] = agg(d > 0)
    R[hi, ai, 1] = agg(tie * ph_reg)
    R[hi, ai, 2] = agg(tie * (1 - ph_reg))
    R[hi, ai, 3] = agg(d < 0)
    W = np.zeros((N, N))          # playoff game: P(home team wins)
    W[hi, ai] = agg((d > 0) + tie * ph_po)
    SP = np.zeros((N, N))         # P(row team wins a series with home ice)
    for i in range(N):
        for j in range(N):
            if i != j:
                o = series_outcomes(W[i, j], 1 - W[j, i])
                SP[i, j] = sum(v for (w, _), v in o.items() if w == "hi")
    return dict(teams=teams, R=R, W=W, SP=SP)


@st.cache_data(ttl=3600, show_spinner=False)
def remaining_games():
    games = {}
    for t in load_standings()["team"]:
        try:
            gs = fetch(f"club-schedule-season/{t}/now").get("games", [])
        except Exception:
            continue
        for g in gs:
            if g.get("gameType") == 2 and g.get("gameState") not in ("OFF", "FINAL"):
                games[g["id"]] = (g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"], g.get("gameDate", ""))
    return sorted(games.values(), key=lambda x: x[2])


def make_bracket(key, div_members):
    """NHL format: top 3 per division + 2 wild cards per conference.
    Returns {conf: [side1, side2]}, each side = [(div winner, wild card), (2nd, 3rd)]."""
    out = {}
    for conf, (d1, d2) in CONF_DIVS.items():
        tops = {d: sorted(div_members[d], key=lambda t: -key[t])[:3] for d in (d1, d2)}
        rest = sorted([t for d in (d1, d2) for t in div_members[d] if t not in tops[d]], key=lambda t: -key[t])
        if len(rest) < 2 or any(len(tops[d]) < 3 for d in (d1, d2)):
            continue
        best = d1 if key[tops[d1][0]] >= key[tops[d2][0]] else d2
        other = d2 if best == d1 else d1
        out[conf] = [[(tops[best][0], rest[1]), (tops[best][1], tops[best][2])],
                     [(tops[other][0], rest[0]), (tops[other][1], tops[other][2])]]
    return out


def play_bracket(br, key, SP, rng, counts):
    """Play out a bracket once; counts[team, round] += 1 for rounds reached
    (0 playoffs, 1 second round, 2 conference final, 3 final, 4 Cup)."""
    def series(a, b):
        hi, lo = (a, b) if key[a] >= key[b] else (b, a)
        return hi if rng.random() < SP[hi, lo] else lo

    champs = []
    for sides in br.values():
        side_w = []
        for side in sides:
            r1 = [series(*m) for m in side]
            for m in side:
                counts[m[0], 0] += 1
                counts[m[1], 0] += 1
            for t in r1:
                counts[t, 1] += 1
            side_w.append(series(*r1))
        for t in side_w:
            counts[t, 2] += 1
        cw = series(*side_w)
        counts[cw, 3] += 1
        champs.append(cw)
    if len(champs) == 2:
        counts[series(*champs), 4] += 1


@st.cache_data(ttl=3600, show_spinner=False, max_entries=6)
def season_sim(n_recent, n_seasons, seed=None):
    lm = league_model(n_recent)
    teams, SP = lm["teams"], lm["SP"]
    N, idx = len(teams), {t: i for i, t in enumerate(teams)}
    stn = load_standings().set_index("team").loc[teams]
    rng = np.random.default_rng(seed)
    rem = [(idx[h], idx[a]) for h, a, _ in remaining_games() if h in idx and a in idx]
    pts0, rw0 = stn["pts"].to_numpy(float), stn["rw"].to_numpy(float)
    pts = np.tile(pts0, (n_seasons, 1))
    rw = np.tile(rw0, (n_seasons, 1))
    if rem:
        h, a = np.array(rem).T
        G = len(h)
        cum = lm["R"][h, a].cumsum(1)[:, :3]
        Hm = np.zeros((G, N), np.float32)
        Hm[np.arange(G), h] = 1
        Am = np.zeros((G, N), np.float32)
        Am[np.arange(G), a] = 1
        HP, AP = np.array([2, 2, 1, 0], np.float32), np.array([0, 1, 2, 2], np.float32)
        HR, AR = np.array([1, 0, 0, 0], np.float32), np.array([0, 0, 0, 1], np.float32)
        for s0 in range(0, n_seasons, 1000):
            s1 = min(s0 + 1000, n_seasons)
            u = rng.random((s1 - s0, G))
            oc = (u[:, :, None] > cum[None]).sum(2)
            pts[s0:s1] += HP[oc] @ Hm + AP[oc] @ Am
            rw[s0:s1] += HR[oc] @ Hm + AR[oc] @ Am
    key = pts + rw * 1e-3 + rng.random(pts.shape) * 1e-6
    divs = stn["div"].tolist()
    div_members = {d: [i for i in range(N) if divs[i] == d] for d in CONF_OF}
    div_win = np.zeros(N)
    for d, mem in div_members.items():
        if mem:
            w = np.array(mem)[key[:, mem].argmax(1)]
            div_win += np.bincount(w, minlength=N)
    pres = np.bincount(key.argmax(1), minlength=N)
    counts = np.zeros((N, 5))
    for s in range(n_seasons):
        play_bracket(make_bracket(key[s], div_members), key[s], SP, rng, counts)
    df = pd.DataFrame({
        "Team": teams, "Div": divs, "GP": stn["gp"].to_numpy(), "PTS now": pts0.astype(int),
        "Proj PTS": pts.mean(0), "Low (10%)": np.percentile(pts, 10, axis=0),
        "High (90%)": np.percentile(pts, 90, axis=0),
        "Playoffs %": counts[:, 0] / n_seasons * 100, "Division %": div_win / n_seasons * 100,
        "Presidents' %": pres / n_seasons * 100, "Conf final %": counts[:, 2] / n_seasons * 100,
        "Final %": counts[:, 3] / n_seasons * 100, "Cup %": counts[:, 4] / n_seasons * 100,
    })
    return dict(table=df.sort_values("Proj PTS", ascending=False).reset_index(drop=True),
                n_games=len(rem), n_seasons=n_seasons)


@st.cache_data(ttl=1800, show_spinner=False)
def today_bracket(n_recent, n=20000):
    """Bracket from the current standings, played out n times."""
    lm = league_model(n_recent)
    teams, SP = lm["teams"], lm["SP"]
    N = len(teams)
    stn = load_standings().set_index("team").loc[teams]
    key = stn["pts"].to_numpy(float) + stn["rw"].to_numpy(float) * 1e-3 - stn["gp"].to_numpy(float) * 1e-6
    divs = stn["div"].tolist()
    br = make_bracket(key, {d: [i for i in range(N) if divs[i] == d] for d in CONF_OF})
    counts = np.zeros((N, 5))
    rng = np.random.default_rng(1)
    if br:
        for _ in range(n):
            play_bracket(br, key, SP, rng, counts)
    return dict(teams=teams, SP=SP, key=key, br=br, counts=counts, n=n)


@st.cache_data(ttl=3 * 3600, show_spinner=False)
def pair_series(a, b, n_recent, reps=8000):
    """Per-game playoff win chances for a vs b in both buildings (a = row 0 of output)."""
    pa, pb = build_profile(a, n_recent), build_profile(b, n_recent)
    shots = pa["shots_for"] + pb["shots_for"] + pa["shots_ag"] + pb["shots_ag"]
    model = dist_model(np.array(shots, float)[:, 2], np.array(shots, float)[:, 3]) if shots else DIST_PRIOR.copy()
    va, vb = team_vector(pa, model), team_vector(pb, model)
    out = []
    for home, away in ((va, vb), (vb, va)):
        S = pair_sides(stack([home], np.zeros(reps, int)), stack([away], np.zeros(reps, int)))
        g = fast_reg(S, seed=3)
        d = g[:, 0] - g[:, 1]
        out.append(float(((d > 0) + (d == 0) * ot_home_prob(S, True)).mean()))
    return out[0], 1 - out[1]      # a wins at home, a wins on the road


# --------------------------------------------------------------------------
# Running a matchup + summarising
# --------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=40)
def run_matchup(home, away, h_goalie, a_goalie, h_out, a_out, n, playoff, n_recent, seed=None):
    hp, ap = build_profile(home, n_recent), build_profile(away, n_recent)
    shots = hp["shots_for"] + hp["shots_ag"] + ap["shots_for"] + ap["shots_ag"]
    if shots:
        arr = np.array(shots, dtype=float)
        model = dist_model(arr[:, 2], arr[:, 3])
    else:
        model = DIST_PRIOR.copy()
    H = build_side(hp, ap, h_goalie, h_out, model)
    A = build_side(ap, hp, a_goalie, a_out, model)
    for side, mult in ((H, 1 + HOME_EDGE), (A, 1 - HOME_EDGE)):
        for key in ("r_ev", "r_pp", "r_sh"):
            side[key] = side[key] * mult
    r = simulate(H, A, n, playoff=playoff, seed=seed)
    res = summarize(r, H, A)
    res["winprob"] = win_prob_curve(H, A, res["featured"], playoff)
    res["playoff"] = playoff
    return res


def summarize(r, H, A):
    g, n = r["goals"], r["n"]
    hw = g[:, 0] > g[:, 1]
    scores = Counter(map(tuple, g.tolist()))
    top = scores.most_common(12)

    def players(team, side):
        gm, am, sm = r["g_mat"][team], r["a_mat"][team], r["s_mat"][team]
        pts = gm + am
        df = pd.DataFrame(dict(
            Player=side["names"], Pos=side["pos"],
            G=gm.mean(0), A=am.mean(0), PTS=pts.mean(0), SOG=sm.mean(0),
            **{"Goal %": (gm > 0).mean(0) * 100, "Point %": (pts > 0).mean(0) * 100},
        ))
        return df.sort_values("PTS", ascending=False).reset_index(drop=True)

    def goalie(team, side):
        opp = 1 - team
        sa = r["sog"][:, opp] - r["eng"][:, opp]
        ga = g[:, opp] - r["eng"][:, opp] - r["so_goal"][:, opp]
        won = hw if team == 0 else ~hw
        return dict(Goalie=side["gk_name"], Saves=(sa - ga).mean(), GA=ga.mean(),
                    **{"SV%": (sa - ga).sum() / max(sa.sum(), 1),
                       "Shutout %": ((g[:, opp] - r["so_goal"][:, opp]) == 0).mean() * 100,
                       "Win %": won.mean() * 100})

    # Featured game: one of the recorded games with the most common final score
    modal = top[0][0]
    rec = r["record"]
    cands = [i for i in range(rec) if tuple(g[i]) == modal]
    if not cands:
        cands = [int(np.argmin(np.abs(g[:rec, 0] - modal[0]) + np.abs(g[:rec, 1] - modal[1])))]
    feat = cands[0]
    ev = sorted([e for e in r["events"] if e[0] == feat], key=lambda e: (e[1], e[3] != "pen"))

    return dict(
        home=H["team"], away=A["team"], n=n,
        home_win=hw.mean(), away_win=1 - hw.mean(),
        reg=r["reg_dec"].mean(), ot=r["ot_dec"].mean(), so=r["so_dec"].mean(),
        home_reg_win=(hw & r["reg_dec"]).mean(), away_reg_win=(~hw & r["reg_dec"]).mean(),
        goals_mean=g.mean(0), sog_mean=r["sog"].mean(0), ppg_mean=r["ppg"].mean(0),
        ppo_mean=r["ppo"].mean(0), eng_mean=r["eng"].mean(0),
        home_cover=((g[:, 0] - g[:, 1]) >= 2).mean(), away_cover=((g[:, 1] - g[:, 0]) >= 2).mean(),
        total=g.sum(1).astype(np.int16), margin=(g[:, 0] - g[:, 1]).astype(np.int16), top_scores=[(s, c / n) for s, c in top],
        goal_dist=[np.bincount(np.minimum(g[:, i], 9), minlength=10) / n for i in (0, 1)],
        players=[players(0, H), players(1, A)], goalies=[goalie(0, H), goalie(1, A)],
        goals_by_per=r["goals_by_per"], sog_by_min=r["sog_by_min"],
        featured=dict(events=ev, final=tuple(int(v) for v in g[feat]),
                      ot=bool(r["ot_dec"][feat]), so=bool(r["so_dec"][feat]),
                      names=[H["names"], A["names"]]),
        sides=dict(home=dict(off=H["off"], gk=H["gk_name"], gkf=H["gk_factor"]),
                   away=dict(off=A["off"], gk=A["gk_name"], gkf=A["gk_factor"])),
    )


# --------------------------------------------------------------------------
# Charts
# --------------------------------------------------------------------------
def clock(t):
    per = t // 1200 + 1
    rem = 1200 - t % 1200
    if per >= 4:
        label = "OT" if per == 4 else f"{per - 3}OT"
    else:
        label = f"P{per}"
    return f"{label} {rem // 60:02d}:{rem % 60:02d}"


def rink_shapes():
    red, blue, line = "#d62728", "#1f4e9c", dict(width=2)
    corner = ("M -72,-42.5 L 72,-42.5 Q 100,-42.5 100,-14.5 L 100,14.5 Q 100,42.5 72,42.5 "
              "L -72,42.5 Q -100,42.5 -100,14.5 L -100,-14.5 Q -100,-42.5 -72,-42.5 Z")
    shapes = [dict(type="path", path=corner, line=dict(color="#333", width=3),
                   fillcolor="#f7fbff", layer="below")]
    shapes.append(dict(type="line", x0=0, x1=0, y0=-42.5, y1=42.5, line=dict(color=red, width=4), layer="below"))
    for x in (-25, 25):
        shapes.append(dict(type="line", x0=x, x1=x, y0=-42.5, y1=42.5, line=dict(color=blue, width=5), layer="below"))
    for x in (-89, 89):
        shapes.append(dict(type="line", x0=x, x1=x, y0=-36.7, y1=36.7, line=dict(color=red, **line), layer="below"))
        sgn = 1 if x > 0 else -1
        shapes.append(dict(type="rect", x0=89 * sgn, x1=92.5 * sgn, y0=-3, y1=3,
                           line=dict(color="#333", width=1.5), fillcolor="#ddd", layer="below"))
        shapes.append(dict(type="path", layer="below", fillcolor="rgba(70,130,220,0.25)",
                           line=dict(color=red, width=1),
                           path=f"M {89 * sgn},-4 L {(89 - 4.5) * sgn},-4 Q {(89 - 6.5) * sgn},0 {(89 - 4.5) * sgn},4 L {89 * sgn},4 Z"))
    shapes.append(dict(type="circle", x0=-15, x1=15, y0=-15, y1=15, line=dict(color=blue, **line), layer="below"))
    for x in (-69, 69):
        for y in (-22, 22):
            shapes.append(dict(type="circle", x0=x - 15, x1=x + 15, y0=y - 15, y1=y + 15,
                               line=dict(color=red, **line), layer="below"))
    for x in (-20, 20):
        for y in (-22, 22):
            shapes.append(dict(type="circle", x0=x - 1, x1=x + 1, y0=y - 1, y1=y + 1,
                               line=dict(color=red), fillcolor=red, layer="below"))
    return shapes


def rink_layout(height=520):
    return dict(
        shapes=rink_shapes(), height=height, plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(range=[-102, 102], visible=False, fixedrange=True),
        yaxis=dict(range=[-45, 45], visible=False, scaleanchor="x", fixedrange=True),
        margin=dict(l=10, r=10, t=60, b=10), legend=dict(orientation="h", y=-0.02),
    )


def event_text(e, names, home, away):
    _, t, team, kind, pl, a1, a2, *_rest = e
    strength = e[11]
    ab = home if team == 0 else away
    nm = names[team]
    who = nm[pl] if 0 <= pl < len(nm) else ""
    if kind == "goal":
        ast = ", ".join(nm[a] for a in (a1, a2) if 0 <= a < len(nm)) or "unassisted"
        tag = f" ({strength})" if strength not in ("EV", "", "3v3") else (" (3v3 OT)" if strength == "3v3" else "")
        return f"🚨 {ab} GOAL{tag}: {who} ({ast})"
    if kind == "shot":
        return f"{ab} shot on goal: {who}"
    if kind == "pen":
        return f"⛔ {ab} penalty: {who} (2 min)"
    if kind == "so":
        return f"🥅 {ab} wins the shootout"
    return kind


ANIM_TEMPLATE = r"""<style>
  body { margin: 0; font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #fff; color: #222; }
  #app { max-width: 1160px; margin: 0 auto; }
  #board { display: flex; align-items: stretch; justify-content: center; gap: 0; margin: 4px 0 6px 0; }
  .team { display: flex; align-items: center; gap: 14px; padding: 6px 18px; color: #fff; min-width: 150px; }
  #awayBox { border-radius: 10px 0 0 10px; justify-content: flex-end; }
  #homeBox { border-radius: 0 10px 10px 0; justify-content: flex-start; }
  .ab { font-size: 22px; font-weight: 700; letter-spacing: 1px; }
  .sc { font-size: 34px; font-weight: 800; min-width: 28px; text-align: center; transition: transform .25s; }
  .sc.pop { transform: scale(1.6); }
  .mid { background: #111; color: #fff; padding: 4px 18px; text-align: center; min-width: 170px; }
  #clock { font-size: 22px; font-weight: 700; font-variant-numeric: tabular-nums; }
  #sub { font-size: 12px; opacity: .85; }
  #pp { text-align: center; font-size: 13px; font-weight: 600; height: 18px; color: #b00020; }
  #wp { display: flex; height: 16px; border-radius: 8px; overflow: hidden; margin: 2px auto 4px auto; max-width: 520px; position: relative; font-size: 11px; }
  #wpA, #wpH { height: 100%; transition: width .4s; }
  #wpTxt { position: absolute; width: 100%; text-align: center; color: #fff; font-weight: 700; line-height: 16px; text-shadow: 0 0 3px #000; }
  canvas { display: block; margin: 0 auto; }
  #controls { display: flex; flex-wrap: wrap; align-items: center; gap: 6px; margin: 8px 4px 4px 4px; }
  button { border: 1px solid #ccc; background: #fafafa; border-radius: 8px; padding: 6px 12px; font-size: 14px; cursor: pointer; }
  button:hover { background: #eee; }
  button.on { background: #222; color: #fff; border-color: #222; }
  #play { min-width: 92px; font-weight: 700; }
  #scrub { flex: 1 1 220px; min-width: 160px; }
  #tlabel { font-size: 13px; color: #555; font-variant-numeric: tabular-nums; min-width: 70px; }
  #ticker { font-size: 13px; color: #333; margin: 4px 6px; line-height: 1.55; min-height: 82px; }
  #ticker .g { font-weight: 700; }
</style>
<div id="app">
  <div id="board">
    <div class="team" id="awayBox"><span class="ab" id="awayAb"></span><span class="sc" id="awaySc">0</span></div>
    <div class="mid"><div id="clock">P1 20:00</div><div id="sub">SOG 0 – 0</div></div>
    <div class="team" id="homeBox"><span class="sc" id="homeSc">0</span><span class="ab" id="homeAb"></span></div>
  </div>
  <div id="wp"><div id="wpA"></div><div id="wpH"></div><span id="wpTxt"></span></div>
  <div id="pp"></div>
  <canvas id="rink"></canvas>
  <div id="controls">
    <button id="restart" title="Restart">⏮</button>
    <button id="play">▶ Play</button>
    <span id="speeds"></span>
    <button id="nextgoal">⏭ Next goal</button>
    <input type="range" id="scrub" min="0" step="1" value="0">
    <span id="tlabel"></span>
  </div>
  <div id="ticker"></div>
</div>
<script>
const D = __DATA__;
const HC = D.hc, AC = D.ac;
const SPEEDS = [["1×", 5], ["2×", 10], ["5×", 25], ["10×", 50]];   // game seconds per real second
const CELEBRATE_MS = 2800;

// ---------- canvas & coordinates (rink feet: x -100..100, y -42.5..42.5) ----------
const cv = document.getElementById("rink"), ctx = cv.getContext("2d");
let W = 800, H = 350, S = 4;
function resize() {
  W = Math.max(300, Math.min(document.body.clientWidth - 8, 1150));
  S = W / 204; H = Math.round(89 * S);
  const dpr = window.devicePixelRatio || 1;
  cv.width = W * dpr; cv.height = H * dpr; cv.style.width = W + "px"; cv.style.height = H + "px";
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener("resize", resize); resize();
const X = x => (x + 102) * S, Y = y => (44.5 - y) * S;

// ---------- deterministic randomness so scrubbing always shows the same game ----------
function mulberry32(a) { return function () { a |= 0; a = a + 0x6D2B79F5 | 0; let t = Math.imul(a ^ a >>> 15, 1 | a);
  t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t; return ((t ^ t >>> 14) >>> 0) / 4294967296; }; }
const rnd = mulberry32(20261111), R = (a, b) => a + (b - a) * rnd();
const DIR = team => team === 0 ? 1 : -1;        // home attacks right (+x), away attacks left

const ev = D.events.slice().sort((a, b) => a.t - b.t);
const goals = ev.filter(e => e.kind === "goal");
const shots = ev.filter(e => e.kind === "shot" || e.kind === "goal");
const pens = ev.filter(e => e.kind === "pen");
const so = ev.find(e => e.kind === "so");

// ---------- build the puck's path through the whole game ----------
const K = [{ t: 0, x: 0, y: 0, own: -1, type: "faceoff" }];
let pos = { t: 0, x: 0, y: 0 };
function push(t, x, y, own, type) {
  if (t <= pos.t) t = pos.t + 0.25;
  x = Math.max(-97, Math.min(97, x)); y = Math.max(-40, Math.min(40, y));
  K.push({ t, x, y, own, type: type || "move" }); pos = { t, x, y };
}
function faceoffDot(attTeam) { return [DIR(attTeam) * 69, rnd() < 0.5 ? 22 : -22]; }
function wander(tEnd) {            // back-and-forth play between the key moments
  let own = rnd() < 0.5 ? 0 : 1;
  while (tEnd - pos.t > 6) {
    if (rnd() < 0.35) own = 1 - own;          // turnover
    const d = DIR(own);
    const nx = Math.max(-82, Math.min(82, pos.x + d * R(12, 42)));
    push(pos.t + R(2.5, 5.5), nx, R(-35, 35), own);
  }
}
const marks = [1200, 2400, 3600].filter(p => p < D.end).map(p => ({ t: p, kind: "period" }));
const items = ev.filter(e => e.kind !== "so").concat(marks)
  .sort((a, b) => a.t - b.t || (a.kind === "period" ? -1 : 1));
for (const it of items) {
  if (it.kind === "period") { wander(it.t - 2); push(Math.max(it.t, pos.t + 0.5), 0, 0, -1, "faceoff"); continue; }
  if (it.kind === "pen") {
    wander(it.t);
    const [fx, fy] = faceoffDot(1 - it.team);             // faceoff in the penalized team's end
    push(it.t + 3, fx, fy, -1, "faceoff"); continue;
  }
  const d = DIR(it.team), gap = it.t - pos.t;
  if (gap > 3) {                    // carry the puck in and work it around before the shot
    const build = Math.min(12, gap - 1);
    wander(it.t - build);
    push(it.t - build * 0.8, d * R(27, 40), R(-30, 30), it.team);
    if (build > 6) push(it.t - build * 0.45, d * R(60, 84), R(-36, 36), it.team);
    if (build > 9) push(it.t - build * 0.2, d * R(45, 62), R(-25, 25), it.team);
  }
  push(Math.max(it.t, pos.t + 0.4), it.x, it.y, it.team, "shot");
  it.launch = pos.t;
  const ny = R(-2.4, 2.4);
  if (it.kind === "goal") {
    push(pos.t + 0.9, d * 90.6, ny * 0.6, it.team, "goal");
    it.arr = pos.t;
    push(pos.t + 4, d * 90.6, ny * 0.6, it.team, "rest");
    push(pos.t + 1.5, 0, 0, -1, "faceoff");
  } else {
    push(pos.t + 0.7, d * 85.5, ny, it.team, "save");
    it.arr = pos.t;
    if (rnd() < 0.55) { const [fx, fy] = faceoffDot(it.team); push(pos.t + 3.5, fx, fy, -1, "faceoff"); }
    else push(pos.t + 1.4, d * R(66, 80), R(-18, 18), it.team);   // rebound / keeps cycling
  }
}
wander(D.end); push(Math.max(D.end, pos.t + 0.5), pos.x, pos.y, -1, "end");

function puckAt(t) {
  if (t <= K[0].t) return { x: K[0].x, y: K[0].y, own: -1 };
  let lo = 0, hi = K.length - 1;
  if (t >= K[hi].t) return { x: K[hi].x, y: K[hi].y, own: -1 };
  while (hi - lo > 1) { const m = (lo + hi) >> 1; if (K[m].t <= t) lo = m; else hi = m; }
  const a = K[lo], b = K[hi];
  let u = (t - a.t) / (b.t - a.t);
  if (b.type !== "goal" && b.type !== "save") u = u * u * (3 - 2 * u);    // ease skating moves, shots fly straight
  return { x: a.x + (b.x - a.x) * u, y: a.y + (b.y - a.y) * u, own: b.type === "faceoff" ? -1 : b.own };
}

// ---------- drawing ----------
function rinkPath() {
  const r = 28 * S; ctx.beginPath(); ctx.moveTo(X(-72), Y(42.5)); ctx.lineTo(X(72), Y(42.5));
  ctx.arcTo(X(100), Y(42.5), X(100), Y(14.5), r); ctx.lineTo(X(100), Y(-14.5));
  ctx.arcTo(X(100), Y(-42.5), X(72), Y(-42.5), r); ctx.lineTo(X(-72), Y(-42.5));
  ctx.arcTo(X(-100), Y(-42.5), X(-100), Y(-14.5), r); ctx.lineTo(X(-100), Y(14.5));
  ctx.arcTo(X(-100), Y(42.5), X(-72), Y(42.5), r); ctx.closePath();
}
function circle(x, y, r, stroke, lw, fill) {
  ctx.beginPath(); ctx.arc(X(x), Y(y), r * S, 0, Math.PI * 2);
  if (fill) { ctx.fillStyle = fill; ctx.fill(); }
  if (stroke) { ctx.strokeStyle = stroke; ctx.lineWidth = lw; ctx.stroke(); }
}
function vline(x, color, w) { ctx.fillStyle = color; ctx.fillRect(X(x) - w * S / 2, Y(42.5), w * S, 85 * S); }
function star(x, y, r, fill, stroke) {
  ctx.beginPath();
  for (let i = 0; i < 10; i++) {
    const a = -Math.PI / 2 + i * Math.PI / 5, rr = i % 2 ? r * 0.45 : r;
    ctx.lineTo(X(x) + Math.cos(a) * rr, Y(y) + Math.sin(a) * rr);
  }
  ctx.closePath(); ctx.fillStyle = fill; ctx.fill(); ctx.strokeStyle = stroke; ctx.lineWidth = 1; ctx.stroke();
}
function alpha(hex, a) {
  const h = hex.replace("#", ""); const n = parseInt(h, 16);
  return `rgba(${n >> 16 & 255},${n >> 8 & 255},${n & 255},${a})`;
}
function drawRink(T) {
  ctx.clearRect(0, 0, W, H);
  ctx.save(); rinkPath(); ctx.fillStyle = "#f5f9fc"; ctx.fill(); ctx.clip();
  ctx.font = `700 ${Math.round(9 * S)}px sans-serif`; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillStyle = alpha(HC, 0.07); ctx.fillText(D.home, X(0), Y(0));
  vline(0, "#d62728", 1); vline(25, "#1f4e9c", 1); vline(-25, "#1f4e9c", 1);
  vline(89, "#d62728", 0.25); vline(-89, "#d62728", 0.25);
  circle(0, 0, 15, "#1f4e9c", 2); circle(0, 0, 0.6, null, 0, "#1f4e9c");
  for (const x of [-69, 69]) for (const y of [-22, 22]) { circle(x, y, 15, "#d62728", 2); circle(x, y, 1, null, 0, "#d62728"); }
  for (const x of [-20, 20]) for (const y of [-22, 22]) circle(x, y, 1, null, 0, "#d62728");
  // creases
  ctx.fillStyle = "rgba(90,150,230,0.30)"; ctx.strokeStyle = "#d62728"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.arc(X(89), Y(0), 6 * S, Math.PI / 2, Math.PI * 1.5); ctx.closePath(); ctx.fill(); ctx.stroke();
  ctx.beginPath(); ctx.arc(X(-89), Y(0), 6 * S, -Math.PI / 2, Math.PI / 2); ctx.closePath(); ctx.fill(); ctx.stroke();
  // goal lights (red glow behind the net after a goal)
  for (const g of goals) {
    const on = (celebrate && celebrate.g === g) || (T >= g.arr && T < g.arr + 4);
    if (on) {
      const d = DIR(g.team), pulse = 0.55 + 0.45 * Math.sin(performance.now() / 90);
      const grd = ctx.createRadialGradient(X(d * 92), Y(0), 1, X(d * 92), Y(0), 16 * S);
      grd.addColorStop(0, `rgba(255,30,30,${0.75 * pulse})`); grd.addColorStop(1, "rgba(255,30,30,0)");
      ctx.fillStyle = grd; ctx.fillRect(X(d * 92) - 16 * S, Y(0) - 16 * S, 32 * S, 32 * S);
    }
  }
  ctx.restore();
  rinkPath(); ctx.strokeStyle = "#333"; ctx.lineWidth = 3; ctx.stroke();
  // nets
  for (const d of [1, -1]) {
    ctx.fillStyle = "#e2e2e2"; ctx.strokeStyle = "#555"; ctx.lineWidth = 1.5;
    const x0 = d > 0 ? X(89) : X(-92.5);
    ctx.fillRect(x0, Y(3), 3.5 * S, 6 * S); ctx.strokeRect(x0, Y(3), 3.5 * S, 6 * S);
  }
  // goalies (away goalie guards the right net, home goalie the left); pulled for empty-net goals
  for (const [team, x] of [[1, 86.5], [0, -86.5]]) {
    const pulled = goals.some(g => g.team !== team && g.str === "EN" && T > g.t - 20 && T < g.arr + 4);
    if (pulled) continue;
    const col = team === 0 ? HC : AC;
    ctx.fillStyle = col; ctx.strokeStyle = "#111"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.ellipse(X(x), Y(0), 1.6 * S, 3.2 * S, 0, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
  }
}
function drawMarks(T) {
  for (const s of shots) {
    if (s.t > T) break;
    const col = s.team === 0 ? HC : AC;
    if (s.kind === "goal") { if (T >= s.arr) star(s.x, s.y, Math.max(7, 2.6 * S), col, "#000"); }
    else circle(s.x, s.y, 1.25, alpha(col, 0.6), 1.6);
  }
  // shot trails
  for (const s of shots) {
    if (s.launch === undefined || T < s.launch || T > s.launch + 3) continue;
    const p = puckAt(Math.min(T, s.arr)), a = 1 - (T - s.launch) / 3;
    ctx.strokeStyle = alpha(s.team === 0 ? HC : AC, 0.7 * a); ctx.lineWidth = 3;
    ctx.setLineDash([6, 4]); ctx.beginPath(); ctx.moveTo(X(s.x), Y(s.y)); ctx.lineTo(X(p.x), Y(p.y)); ctx.stroke(); ctx.setLineDash([]);
  }
}
function drawPuck(T) {
  for (let k = 5; k >= 1; k--) {             // motion blur trail
    const p = puckAt(T - k * 0.03 * speed);
    ctx.fillStyle = `rgba(0,0,0,${0.10 * (6 - k) / 5})`;
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), Math.max(3, 1.1 * S), 0, Math.PI * 2); ctx.fill();
  }
  const p = puckAt(T);
  if (p.own >= 0) {
    ctx.strokeStyle = alpha(p.own === 0 ? HC : AC, 0.9); ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), Math.max(7, 2.6 * S), 0, Math.PI * 2); ctx.stroke();
  }
  ctx.fillStyle = "#000"; ctx.beginPath(); ctx.arc(X(p.x), Y(p.y), Math.max(3.5, 1.2 * S), 0, Math.PI * 2); ctx.fill();
}

// ---------- celebration ----------
let celebrate = null, confetti = [];
function startCelebration(g, now) {
  celebrate = { g, start: now, until: now + CELEBRATE_MS };
  const col = g.team === 0 ? HC : AC, d = DIR(g.team);
  confetti = [];
  for (let i = 0; i < 90; i++) confetti.push({ x: X(d * 88), y: Y(0), vx: -d * R(60, 420), vy: R(-380, 120),
    c: i % 3 === 0 ? "#ffffff" : (i % 3 === 1 ? col : "#ffd400"), s: R(3, 7), r: R(0, 6) });
  const el = document.getElementById(g.team === 0 ? "homeSc" : "awaySc");
  el.classList.add("pop"); setTimeout(() => el.classList.remove("pop"), 450);
}
function drawCelebration(now, dt) {
  if (!celebrate) return;
  const g = celebrate.g, col = g.team === 0 ? HC : AC, k = (now - celebrate.start) / CELEBRATE_MS;
  for (const c of confetti) {
    c.vy += 520 * dt; c.x += c.vx * dt; c.y += c.vy * dt; c.r += dt * 8;
    ctx.save(); ctx.translate(c.x, c.y); ctx.rotate(c.r); ctx.fillStyle = c.c; ctx.fillRect(-c.s / 2, -c.s / 4, c.s, c.s / 2); ctx.restore();
  }
  const bandH = Math.max(70, H * 0.3), y0 = H / 2 - bandH / 2;
  const fade = Math.min(1, k * 6, (1 - k) * 5);
  ctx.globalAlpha = Math.max(0, fade) * 0.92; ctx.fillStyle = col; ctx.fillRect(0, y0, W, bandH);
  ctx.globalAlpha = Math.max(0, fade);
  const scale = 1 + 0.08 * Math.sin(now / 80);
  ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.font = `900 ${Math.round(bandH * 0.42 * scale)}px sans-serif`;
  ctx.shadowColor = "rgba(0,0,0,.5)"; ctx.shadowBlur = 8;
  const tag = { PP: "POWER-PLAY GOAL!", SH: "SHORTHANDED GOAL!", EN: "EMPTY-NET GOAL!", "3v3": "OVERTIME WINNER!", "6v5": "GOAL! (6-on-5)" }[g.str] || "GOAL!";
  ctx.fillText(`${g.team === 0 ? D.home : D.away} ${tag}`, W / 2, y0 + bandH * 0.36);
  ctx.shadowBlur = 0;
  ctx.font = `600 ${Math.round(bandH * 0.17)}px sans-serif`;
  ctx.fillText(`${g.name}${g.ast.length ? "  (" + g.ast.join(", ") + ")" : "  (unassisted)"}   ·   ${D.away} ${g.as_} – ${g.hs} ${D.home}`,
    W / 2, y0 + bandH * 0.76);
  ctx.globalAlpha = 1;
}
function drawOverlay(T) {
  if (!playing && T === 0) {
    ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.fillRect(0, H / 2 - 30, W, 60);
    ctx.fillStyle = "#fff"; ctx.font = "700 22px sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.fillText("Press ▶ Play to drop the puck", W / 2, H / 2);
  }
  if (T >= D.end && !celebrate) {
    const fin = D.final, extra = D.so ? " (SO)" : D.ot ? " (OT)" : "";
    ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(0, H / 2 - 42, W, 84);
    ctx.fillStyle = "#fff"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    ctx.font = "800 30px sans-serif"; ctx.fillText(`FINAL${extra}:  ${D.away} ${fin[1]} – ${fin[0]} ${D.home}`, W / 2, H / 2 - 10);
    ctx.font = "500 15px sans-serif";
    const w = fin[0] > fin[1] ? D.home : D.away;
    ctx.fillText(so ? `${w} wins the shootout` : `${w} wins`, W / 2, H / 2 + 22);
  }
}

// ---------- scoreboard, power play, win probability, ticker ----------
const $ = id => document.getElementById(id);
$("awayAb").textContent = D.away; $("homeAb").textContent = D.home;
$("awayBox").style.background = AC; $("homeBox").style.background = HC;
$("wpA").style.background = AC; $("wpH").style.background = HC;
function clockStr(T) {
  if (T >= D.end) return D.so ? "FINAL/SO" : D.ot ? "FINAL/OT" : "FINAL";
  if (T >= 3600) {
    if (!D.playoff) { const r = Math.max(0, 3900 - T); return `OT ${Math.floor(r / 60)}:${String(Math.floor(r % 60)).padStart(2, "0")}`; }
    const p = Math.floor((T - 3600) / 1200) + 1, r = 1200 - (T - 3600) % 1200;
    return `${p === 1 ? "" : p}OT ${Math.floor(r / 60)}:${String(Math.floor(r % 60)).padStart(2, "0")}`;
  }
  const p = Math.min(3, Math.floor(T / 1200) + 1), r = p * 1200 - T;
  return `P${p} ${Math.floor(r / 60)}:${String(Math.floor(r % 60)).padStart(2, "0")}`;
}
function evClock(t) { return t >= 3900 && !D.playoff ? "SO" : clockStr(Math.min(t, D.end - 0.01)); }
function scoreAt(T) { let s = [0, 0]; for (const g of goals) if (T >= g.arr) s = [g.hs, g.as_]; if (T >= D.end) s = D.final; return s; }
function wpAt(T) {
  const m = (T + 6) / 60; let p = D.wp.home[0];   // +6s: the curve is re-simulated just after each goal
  for (let i = 0; i < D.wp.t.length; i++) if (D.wp.t[i] <= m) p = D.wp.home[i];
  if (T >= D.end) p = D.wp.home[D.wp.home.length - 1];
  return p;
}
function evText(e) {
  const ab = e.team === 0 ? D.home : D.away;
  if (e.kind === "goal") return `<span class="g">🚨 ${ab} GOAL${e.str && e.str !== "EV" ? " (" + e.str + ")" : ""}: ${e.name}</span>` +
    (e.ast.length ? ` — assists: ${e.ast.join(", ")}` : " — unassisted");
  if (e.kind === "shot") return `${ab} shot: ${e.name} — saved`;
  if (e.kind === "pen") return `⛔ ${ab} penalty: ${e.name} (2 min)`;
  if (e.kind === "so") return `🥅 ${ab} wins the shootout`;
  return "";
}
let lastBoard = "";
function updateBoard(T) {
  const s = scoreAt(T);
  const sog = [0, 0]; for (const x of shots) if (x.t <= T) sog[x.team]++;
  let pp = "";
  for (const p of pens) {
    if (T < p.t || T >= p.t + 120) continue;
    const ended = goals.some(g => g.team !== p.team && g.str === "PP" && g.t > p.t && g.arr <= T);
    if (ended) continue;
    const left = p.t + 120 - T, ab = p.team === 0 ? D.away : D.home;
    pp = `⚡ ${ab} POWER PLAY ${Math.floor(left / 60)}:${String(Math.floor(left % 60)).padStart(2, "0")}`;
  }
  const wp = wpAt(T);
  const key = [s, sog, pp, Math.round(wp * 100), clockStr(T)].join("|");
  if (key === lastBoard) return;
  lastBoard = key;
  $("homeSc").textContent = s[0]; $("awaySc").textContent = s[1];
  $("clock").textContent = clockStr(T);
  $("sub").textContent = `SOG  ${D.away} ${sog[1]} – ${sog[0]} ${D.home}`;
  $("pp").textContent = pp;
  $("wpA").style.width = (100 - wp * 100) + "%"; $("wpH").style.width = (wp * 100) + "%";
  $("wpTxt").textContent = `Win probability  ${D.away} ${Math.round(100 - wp * 100)}%  –  ${Math.round(wp * 100)}% ${D.home}`;
  const past = ev.filter(e => (e.kind === "goal" ? T >= e.arr : e.t <= T) && (e.kind !== "so" || T >= D.end)).slice(-5).reverse();
  $("ticker").innerHTML = past.map(e => `<div><b>${evClock(e.t)}</b> &nbsp; ${evText(e)}</div>`).join("") ||
    "<div style='color:#888'>Play-by-play will appear here.</div>";
  $("tlabel").textContent = clockStr(T);
}

// ---------- playback ----------
let T = 0, playing = false, speed = 10, lastNow = null, dragging = false;
const scrub = $("scrub"); scrub.max = Math.ceil(D.end);
const playBtn = $("play");
function setPlaying(v) { playing = v; playBtn.textContent = v ? "⏸ Pause" : (T >= D.end ? "↻ Replay" : "▶ Play"); }
playBtn.onclick = () => { if (T >= D.end) { T = 0; celebrate = null; } setPlaying(!playing); };
$("restart").onclick = () => { T = 0; celebrate = null; lastBoard = ""; setPlaying(true); };
$("nextgoal").onclick = () => {
  const g = goals.find(g => g.t - 8 > T + 0.5);   // each press moves on to the following goal
  celebrate = null; lastBoard = "";
  if (g) { T = Math.max(0, g.t - 8); setPlaying(true); } else { T = D.end; setPlaying(false); }
};
const sp = $("speeds");
for (const [lbl, v] of SPEEDS) {
  const b = document.createElement("button"); b.textContent = lbl; b.dataset.v = v;
  if (v === speed) b.classList.add("on");
  b.onclick = () => { speed = v; for (const x of sp.children) x.classList.toggle("on", +x.dataset.v === v); };
  sp.appendChild(b);
}
scrub.addEventListener("pointerdown", () => dragging = true);
window.addEventListener("pointerup", () => dragging = false);
scrub.addEventListener("input", () => { T = +scrub.value; celebrate = null; lastBoard = ""; if (T >= D.end) setPlaying(false); });

function frame(now) {
  const dt = lastNow === null ? 0 : Math.min(0.1, (now - lastNow) / 1000);
  lastNow = now;
  if (playing && !dragging) {
    if (celebrate) { if (now >= celebrate.until) celebrate = null; }
    else {
      const nT = Math.min(T + dt * speed, D.end);
      const g = goals.find(g => g.arr > T && g.arr <= nT);
      if (g) { T = g.arr; startCelebration(g, now); }
      else T = nT;
      if (T >= D.end) setPlaying(false);
    }
  }
  if (!dragging) scrub.value = T;
  drawRink(T); drawMarks(T); drawPuck(T); drawCelebration(now, dt); drawOverlay(T);
  updateBoard(T);
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);
</script>
"""


def animation_html(res):
    """Self-contained HTML/JS player: moving puck, goal celebrations, speed controls."""
    f = res["featured"]
    home, away = res["home"], res["away"]
    ch, ca = team_colors(home, away)
    events = []
    for (_sim, t, team, kind, pl, a1, a2, x, y, hs, as_, strength) in f["events"]:
        nm = f["names"][team]
        rx, ry = (x, y) if team == 0 else (-x, -y)
        events.append(dict(
            t=int(t), team=int(team), kind=kind,
            name=nm[pl] if 0 <= pl < len(nm) else "",
            ast=[nm[a] for a in (a1, a2) if 0 <= a < len(nm)],
            x=round(float(np.clip(rx, -97, 97)), 1), y=round(float(np.clip(ry, -40, 40)), 1),
            hs=int(hs), as_=int(as_), str=str(strength)))
    end = max([e["t"] for e in events] + [3600])
    data = dict(home=home, away=away, hc=ch, ac=ca, events=events, end=end,
                final=[int(v) for v in f["final"]], ot=bool(f["ot"]), so=bool(f["so"]),
                playoff=bool(res.get("playoff")),
                wp=dict(t=[float(v) for v in res["winprob"]["t"]], home=[float(v) for v in res["winprob"]["home"]]))
    return ANIM_TEMPLATE.replace("__DATA__", json.dumps(data))


def flow_figure(res):
    """Real-style game flow chart: cumulative shots, goals, and power plays."""
    f = res["featured"]
    home, away = res["home"], res["away"]
    ch, ca = team_colors(home, away)
    end = max([e[1] for e in f["events"]] + [3600])
    fig = go.Figure()
    for team, col, ab in ((0, ch, home), (1, ca, away)):
        sh = [e for e in f["events"] if e[2] == team and e[3] in ("shot", "goal")]
        xs = [0] + [e[1] / 60 for e in sh] + [end / 60]
        ys = list(range(len(sh) + 1)) + [len(sh)]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(shape="hv", color=col, width=3),
                                 name=f"{ab} shots on goal"))
        gl = [(e, i + 1) for i, e in enumerate(sh) if e[3] == "goal"]
        fig.add_trace(go.Scatter(x=[e[1] / 60 for e, _ in gl], y=[c for _, c in gl], mode="markers",
                                 marker=dict(symbol="star", size=16, color=col, line=dict(width=1, color="black")),
                                 name=f"{ab} goals",
                                 hovertext=[f"{clock(e[1])} {event_text(e, f['names'], home, away)}" for e, _ in gl],
                                 hoverinfo="text"))
    for e in f["events"]:
        if e[3] == "pen":
            pp_team = 1 - e[2]
            ends = [g[1] for g in f["events"] if g[3] == "goal" and g[2] == pp_team and e[1] < g[1] <= e[1] + 120]
            stop = min(ends) if ends else e[1] + 120
            fig.add_vrect(x0=e[1] / 60, x1=stop / 60, fillcolor=ch if pp_team == 0 else ca,
                          opacity=0.12, line_width=0)
    for m in (20, 40, 60):
        fig.add_vline(x=m, line=dict(color="#999", dash="dot"))
    fig.update_layout(height=380, xaxis_title="Game minute (shaded = power play)",
                      yaxis_title="Cumulative shots on goal", margin=dict(t=30, b=40),
                      legend=dict(orientation="h", y=1.12))
    return fig


def winprob_figure(res):
    wp, f = res["winprob"], res["featured"]
    home, away = res["home"], res["away"]
    ch, ca = team_colors(home, away)
    t = wp["t"]
    ph = np.array(wp["home"]) * 100
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ph, mode="lines", line=dict(color=ch, width=3),
                             name=f"{home} win %", hovertemplate="%{y:.0f}%<extra>" + home + "</extra>"))
    fig.add_trace(go.Scatter(x=t, y=100 - ph, mode="lines", line=dict(color=ca, width=3),
                             name=f"{away} win %", hovertemplate="%{y:.0f}%<extra>" + away + "</extra>"))
    for e in f["events"]:
        if e[3] == "goal":
            i = max([k for k, tt in enumerate(t) if tt * 60 <= e[1] + DT] or [0])
            y = ph[i] if e[2] == 0 else 100 - ph[i]
            fig.add_trace(go.Scatter(x=[e[1] / 60], y=[y], mode="markers", showlegend=False,
                                     marker=dict(symbol="star", size=15, color=ch if e[2] == 0 else ca,
                                                 line=dict(width=1, color="black")),
                                     hovertext=f"{clock(e[1])} {event_text(e, f['names'], home, away)}",
                                     hoverinfo="text"))
    fig.add_hline(y=50, line=dict(color="#999", dash="dot"))
    for m in (20, 40, 60):
        fig.add_vline(x=m, line=dict(color="#ccc", dash="dot"))
    fig.update_layout(height=340, yaxis=dict(range=[0, 100], title="Win probability %"),
                      xaxis_title="Game minute", margin=dict(t=30, b=40), legend=dict(orientation="h", y=1.12))
    return fig


def american(p):
    p = min(max(p, 0.001), 0.999)
    return f"{-100 * p / (1 - p):.0f}" if p >= 0.5 else f"+{100 * (1 - p) / p:.0f}"


def implied(price):
    return 100 / (price + 100) if price > 0 else -price / (-price + 100)


def payout(price):
    return price if price > 0 else 10000 / -price


def shot_map(prof, which, color):
    shots = prof["shots_for"] if which == "for" else prof["shots_ag"]
    fig = go.Figure()
    if shots:
        arr = np.array(shots, dtype=float)
        fig.add_trace(go.Histogram2dContour(
            x=arr[:, 0], y=arr[:, 1], colorscale=[[0, "rgba(255,255,255,0)"], [1, color]],
            showscale=False, ncontours=12, contours=dict(coloring="fill", showlines=False),
            xbins=dict(start=0, end=100, size=5), ybins=dict(start=-42.5, end=42.5, size=5), opacity=0.8,
            hoverinfo="skip"))
        gl = arr[arr[:, 3] == 1]
        fig.add_trace(go.Scatter(x=gl[:, 0], y=gl[:, 1], mode="markers", name="goals",
                                 marker=dict(symbol="star", size=10, color="black")))
    lay = rink_layout(330)
    lay["xaxis"]["range"] = [-2, 102]
    lay["margin"] = dict(l=5, r=5, t=10, b=5)
    lay["showlegend"] = False
    fig.update_layout(**lay)
    return fig


def pct(x):
    return "–" if x is None else f"{100 * x:.1f}%"


# --------------------------------------------------------------------------
# Sportsbook odds (the-odds-api.com)
# --------------------------------------------------------------------------
@st.cache_data(ttl=600, show_spinner=False)
def odds_feed(api_key):
    r = _session().get("https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds",
                       params=dict(apiKey=api_key, regions="us", markets="h2h,spreads,totals",
                                   oddsFormat="american"), timeout=25)
    r.raise_for_status()
    return r.json(), str(r.headers.get("x-requests-remaining", "?"))


def _norm(s):
    s = unicodedata.normalize("NFKD", str(s or "")).encode("ascii", "ignore").decode().lower()
    return re.sub(r"[^a-z0-9 ]", "", s.replace("-", " ")).strip()


def same_team(feed_name, abbrev):
    full, fn = _norm(NAMES.get(abbrev, abbrev)), _norm(feed_name)
    if not fn or not full:
        return False
    return fn == full or fn.split()[-1] == full.split()[-1] or fn == abbrev.lower()


def odds_api_key():
    if st.session_state.get("odds_key"):
        return st.session_state["odds_key"]
    try:
        return st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        return ""


def best_lines(evt, res):
    """Best moneyline prices and the main total (with sim chances) for one odds-feed game."""
    ml = {"home": None, "away": None}
    totals = {}
    for bk in evt.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for oc in mk.get("outcomes", []):
                price = oc.get("price")
                if price is None:
                    continue
                if mk["key"] == "h2h":
                    side = "home" if same_team(oc["name"], res["home"]) else "away"
                    if ml[side] is None or price > ml[side]:
                        ml[side] = price
                elif mk["key"] == "totals" and oc.get("point") is not None:
                    t = totals.setdefault(float(oc["point"]), {"n": 0, "Over": None, "Under": None})
                    t["n"] += 1
                    if t[oc["name"]] is None or price > t[oc["name"]]:
                        t[oc["name"]] = price
    out = dict(ml_home=ml["home"], ml_away=ml["away"], total_line=None, over_price=None, under_price=None,
               p_over_line=None, p_under_line=None)
    if totals:
        line = max(totals, key=lambda k: totals[k]["n"])
        tot = res["total"]
        out.update(total_line=line, over_price=totals[line]["Over"], under_price=totals[line]["Under"],
                   p_over_line=round(float((tot > line).mean()), 4), p_under_line=round(float((tot < line).mean()), 4))
    return out


# --------------------------------------------------------------------------
# Confirmed starting goalies (DailyFaceoff's public starting-goalies page)
# --------------------------------------------------------------------------
BROWSER_UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/130.0 Safari/537.36")
GOALIE_ICON = {"Confirmed": "✅ Confirmed", "Likely": "🟡 Likely",
               "Projected": "⚪ Projected (DailyFaceoff)", "Recent": "🔁 Guess from recent starts"}


def _gstatus(v):
    v = _norm(v)
    if "unconfirm" in v:
        return "Projected"
    if "confirm" in v:
        return "Confirmed"
    if "likely" in v or "expected" in v:
        return "Likely"
    return "Projected"


@st.cache_data(ttl=900, show_spinner=False)
def dfo_goalies(date_str):
    """Every game's listed goalies for one date (empty list if the page can't be read)."""
    try:
        r = _session().get(f"https://www.dailyfaceoff.com/starting-goalies/{date_str}",
                           headers={"User-Agent": BROWSER_UA, "Accept": "text/html"}, timeout=20)
        r.raise_for_status()
        m = re.search(r'<script[^>]*id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
        data = json.loads(m.group(1)) if m else {}
    except Exception:
        return []

    def key(d, side, *words):
        for k, v in d.items():
            kl = k.lower()
            if kl.startswith(side) and all(w in kl for w in words) and isinstance(v, str):
                return v
        return None

    found = []

    def walk(o):
        if isinstance(o, dict):
            hg, ag = key(o, "home", "goalie", "name"), key(o, "away", "goalie", "name")
            if hg or ag:
                found.append(dict(
                    home_team=key(o, "home", "team", "name") or key(o, "home", "team") or "",
                    away_team=key(o, "away", "team", "name") or key(o, "away", "team") or "",
                    home_goalie=hg, away_goalie=ag,
                    home_status=_gstatus(key(o, "home", "strength") or key(o, "home", "status") or ""),
                    away_status=_gstatus(key(o, "away", "strength") or key(o, "away", "status") or "")))
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(data)
    return found


def match_goalie(name, goalies):
    n = _norm(name)
    if not n:
        return None
    for g in goalies:
        if _norm(g["name"]) == n:
            return g["id"]
    last = n.split()[-1]
    c = [g for g in goalies if _norm(g["name"]).split()[-1:] == [last]]
    return c[0]["id"] if len(c) == 1 else None


def next_game(home, away):
    """The next scheduled game with this home and away team, if there is one."""
    try:
        games = fetch(f"club-schedule-season/{home}/now").get("games", [])
    except Exception:
        return None
    up = [g for g in games if g.get("gameState") in ("FUT", "PRE", "LIVE", "CRIT")
          and g.get("homeTeam", {}).get("abbrev") == home and g.get("awayTeam", {}).get("abbrev") == away]
    up.sort(key=lambda g: (g.get("gameDate", ""), g.get("startTimeUTC", "")))
    if not up:
        return None
    g = up[0]
    return dict(id=g["id"], date=g.get("gameDate", ""), start=g.get("startTimeUTC", ""), state=g.get("gameState"),
                playoff=g.get("gameType") == 3)


def starting_goalies(home, away, date_str, hp, ap):
    """{'home'|'away': (goalie id, status, note)} from DailyFaceoff, else the recent-starts guess."""
    entry = None
    if date_str:
        entry = next((e for e in dfo_goalies(date_str)
                      if same_team(e["home_team"], home) and same_team(e["away_team"], away)), None)
    out = {}
    for side, prof in (("home", hp), ("away", ap)):
        gid, status, note = None, None, ""
        listed = entry.get(f"{side}_goalie") if entry else None
        if listed:
            gid = match_goalie(listed, prof["goalies"])
            status = entry[f"{side}_status"] if gid else None
            if gid is None:
                note = f"DailyFaceoff lists {listed}, who has no stats with this team yet."
        if gid is None:
            gid, status = default_lineup(prof)[0], "Recent"
        out[side] = (gid, status, note)
    return out


def goalie_name(prof, gid):
    return next((g["name"] for g in prof["goalies"] if g["id"] == gid), "League-average goalie")


# --------------------------------------------------------------------------
# Prediction tracker (GitHub when set up in Secrets, otherwise a local CSV)
# --------------------------------------------------------------------------
MODEL_VERSION = "v4-2026-11-11"
TRACK_COLS = ["game_id", "date", "start_utc", "away", "home", "away_goalie", "home_goalie", "goalie_status",
              "p_home", "proj_away", "proj_home", "p_ot", "likely", "sims", "n_recent", "model", "source",
              "saved_at", "ml_away", "ml_home", "total_line", "over_price", "under_price", "p_over_line",
              "p_under_line", "final_away", "final_home", "end"]
LOCAL_TRACK = Path(__file__).resolve().parent / "nhl_predictions.csv"


def _gh_conf():
    try:
        token, repo = st.secrets.get("GITHUB_TOKEN"), st.secrets.get("GITHUB_REPO")
        if token and repo:
            return dict(token=token, repo=repo, branch=st.secrets.get("TRACKER_BRANCH", "tracker-data"),
                        path=st.secrets.get("TRACKER_PATH", "nhl_predictions.csv"))
    except Exception:
        pass
    return None


def _gh(method, url, conf, **kw):
    headers = {"Authorization": f"Bearer {conf['token']}", "Accept": "application/vnd.github+json",
               "X-GitHub-Api-Version": "2022-11-28"}
    return requests.request(method, f"https://api.github.com{url}", headers=headers, timeout=20, **kw)


def _gh_ensure_branch(conf):
    """Predictions live on their own branch so saving them doesn't restart the Streamlit app."""
    repo, br = conf["repo"], conf["branch"]
    if _gh("GET", f"/repos/{repo}/git/ref/heads/{br}", conf).status_code == 200:
        return
    r = _gh("GET", f"/repos/{repo}", conf)
    r.raise_for_status()
    base = r.json()["default_branch"]
    r = _gh("GET", f"/repos/{repo}/git/ref/heads/{base}", conf)
    r.raise_for_status()
    _gh("POST", f"/repos/{repo}/git/refs", conf,
        json=dict(ref=f"refs/heads/{br}", sha=r.json()["object"]["sha"])).raise_for_status()


def _read_csv_text(text):
    df = pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame(columns=TRACK_COLS)
    for c in TRACK_COLS:
        if c not in df.columns:
            df[c] = None
    return df[TRACK_COLS]


@st.cache_data(ttl=120, show_spinner=False)
def load_tracker():
    conf = _gh_conf()
    if conf:
        r = _gh("GET", f"/repos/{conf['repo']}/contents/{conf['path']}", conf, params={"ref": conf["branch"]})
        if r.status_code == 404:
            return _read_csv_text("")
        r.raise_for_status()
        return _read_csv_text(base64.b64decode(r.json()["content"]).decode())
    if LOCAL_TRACK.exists():
        return _read_csv_text(LOCAL_TRACK.read_text())
    return _read_csv_text("")


def save_tracker(df, message):
    text = df[TRACK_COLS].to_csv(index=False)
    conf = _gh_conf()
    if conf:
        _gh_ensure_branch(conf)
        url = f"/repos/{conf['repo']}/contents/{conf['path']}"
        cur = _gh("GET", url, conf, params={"ref": conf["branch"]})
        body = dict(message=message, branch=conf["branch"], content=base64.b64encode(text.encode()).decode())
        if cur.status_code == 200:
            body["sha"] = cur.json()["sha"]
        _gh("PUT", url, conf, json=body).raise_for_status()
    else:
        LOCAL_TRACK.write_text(text)
    load_tracker.clear()


def make_record(res, game, goalies, n_recent, source, evt=None):
    s = res["top_scores"][0][0]
    rec = dict(
        game_id=int(game["id"]), date=game.get("date", ""), start_utc=game.get("start", ""),
        away=res["away"], home=res["home"], away_goalie=goalies["away"][0], home_goalie=goalies["home"][0],
        goalie_status=f"{goalies['away'][1]}/{goalies['home'][1]}",
        p_home=round(float(res["home_win"]), 4), proj_away=round(float(res["goals_mean"][1]), 2),
        proj_home=round(float(res["goals_mean"][0]), 2), p_ot=round(float(res["ot"] + res["so"]), 4),
        likely=f"{s[1]}-{s[0]}", sims=int(res["n"]), n_recent=int(n_recent), model=MODEL_VERSION, source=source,
        saved_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        final_away=None, final_home=None, end=None)
    rec.update(best_lines(evt, res) if evt else dict(ml_away=None, ml_home=None, total_line=None, over_price=None,
                                                    under_price=None, p_over_line=None, p_under_line=None))
    return rec


def _started(start_utc):
    try:
        return pd.Timestamp(start_utc) <= pd.Timestamp.now(tz="UTC")
    except Exception:
        return False


def upsert_predictions(records, message):
    """Add or refresh predictions. Once a game has started its prediction is locked."""
    df = load_tracker().copy()
    changed = 0
    for rec in records:
        if _started(rec["start_utc"]):
            continue
        hit = df.index[pd.to_numeric(df["game_id"], errors="coerce") == rec["game_id"]]
        if len(hit):
            row = df.loc[hit[0]]
            if pd.notna(row["final_home"]) or _started(row["start_utc"]):
                continue
            df = df.drop(hit)
        df = pd.concat([df, pd.DataFrame([rec])[TRACK_COLS]], ignore_index=True)
        changed += 1
    if changed:
        save_tracker(df.sort_values(["date", "start_utc", "game_id"]), message)
    return changed


@st.cache_data(ttl=600, show_spinner=False)
def final_scores(date_str):
    """{game id: (away goals, home goals, 'REG'|'OT'|'SO')} for finished games on a date."""
    out = {}
    try:
        games = fetch(f"score/{date_str}").get("games", [])
    except Exception:
        games = []
    for g in games:
        if g.get("gameState") in ("OFF", "FINAL"):
            out[g["id"]] = (g["awayTeam"].get("score"), g["homeTeam"].get("score"),
                            (g.get("gameOutcome") or {}).get("lastPeriodType", "REG"))
    return out


def grade_tracker():
    """Fill in final scores for finished games; saves only if something changed."""
    df = load_tracker().copy()
    todo = df[df["final_home"].isna() & df["start_utc"].apply(_started)]
    changed = 0
    for date_str, grp in todo.groupby("date"):
        scores = final_scores(str(date_str))
        for i, row in grp.iterrows():
            gid = int(row["game_id"])
            sc = scores.get(gid)
            if sc is None:
                try:
                    lj = fetch(f"gamecenter/{gid}/landing")
                    if lj.get("gameState") in ("OFF", "FINAL"):
                        sc = (lj["awayTeam"].get("score"), lj["homeTeam"].get("score"),
                              (lj.get("gameOutcome") or {}).get("lastPeriodType", "REG"))
                except Exception:
                    sc = None
            if sc and sc[0] is not None:
                df.loc[i, ["final_away", "final_home", "end"]] = [sc[0], sc[1], sc[2]]
                changed += 1
    if changed:
        save_tracker(df, f"Grade {changed} NHL predictions")
    return df, changed


def bet_results(df, min_ev):
    """Flat $100 on every bet whose expected profit per $100 is at least min_ev."""
    bets = []
    for _, r in df.iterrows():
        if pd.isna(r["final_home"]):
            continue
        fh, fa = float(r["final_home"]), float(r["final_away"])
        p = float(r["p_home"])
        for side, prob, price, won in (("home", p, r["ml_home"], fh > fa), ("away", 1 - p, r["ml_away"], fa > fh)):
            if pd.isna(price):
                continue
            ev = prob * payout(price) - (1 - prob) * 100
            if ev >= min_ev:
                team = r["home"] if side == "home" else r["away"]
                bets.append(dict(date=r["date"], game=f"{r['away']} @ {r['home']}", bet=f"{team} ML {int(price):+d}",
                                 ev=ev, profit=payout(price) if won else -100.0))
        if pd.notna(r["total_line"]) and pd.notna(r["p_over_line"]):
            tot, line = fh + fa, float(r["total_line"])
            p_o, p_u = float(r["p_over_line"]), float(r["p_under_line"])
            for name, prob, lose, price, won, push in (
                    ("Over", p_o, p_u, r["over_price"], tot > line, tot == line),
                    ("Under", p_u, p_o, r["under_price"], tot < line, tot == line)):
                if pd.isna(price):
                    continue
                ev = prob * payout(price) - lose * 100     # a push (whole-number line) neither wins nor loses
                if ev >= min_ev:
                    bets.append(dict(date=r["date"], game=f"{r['away']} @ {r['home']}",
                                     bet=f"{name} {line:g} {int(price):+d}", ev=ev,
                                     profit=0.0 if push else (payout(price) if won else -100.0)))
    return pd.DataFrame(bets, columns=["date", "game", "bet", "ev", "profit"])


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
st.title("🏒 NHL Game Simulator")

try:
    standings = load_standings()
    TEAMS = standings["team"].tolist()
except Exception as e:
    st.error(f"Couldn't reach the NHL data feed ({e}). Check your internet connection and refresh.")
    st.stop()
if not TEAMS:
    st.error("The NHL standings feed came back empty. Try again in a minute.")
    st.stop()
NAMES = dict(zip(standings["team"], standings["name"]))


def _label(t):
    return f"{t} – {NAMES.get(t, t)}"


for key, val in (("home", "SEA" if "SEA" in TEAMS else TEAMS[0]),
                 ("away", "VAN" if "VAN" in TEAMS else TEAMS[1])):
    if st.session_state.get(key) not in TEAMS:
        st.session_state[key] = val

with st.sidebar:
    st.header("Matchup")
    away = st.selectbox("Away team", TEAMS, key="away", format_func=_label)
    home = st.selectbox("Home team", TEAMS, key="home", format_func=_label)
    st.header("Settings")
    n_sims = st.select_slider("Simulations", [1000, 2000, 5000, 10000, 20000], value=5000)
    n_recent = st.slider("Recent games used for team play style", 5, 30, 15,
                         help="How many of each team's most recent games feed the model.")
    playoff = st.toggle("Playoff rules (sudden-death 5-on-5 OT, no shootout)", value=False)
    st.caption("Data: NHL public API (refreshes every 30 min). Model: real play-by-play shot rates "
               "by score state & strength, shot locations, penalties, finishing and goaltending.")

if home == away:
    st.warning("Pick two different teams.")
    st.stop()

with st.spinner(f"Loading real play-by-play for {away} and {home}…"):
    try:
        hp, ap = build_profile(home, n_recent), build_profile(away, n_recent)
    except Exception as e:
        st.error(f"Couldn't load team data ({e}).")
        st.stop()

tab_match, tab_anim, tab_bet, tab_slate, tab_track, tab_season, tab_playoffs, tab_style = st.tabs(
    ["🎯 Matchup", "🎬 Game animation", "💰 Betting", "📅 Every game on a date", "📈 Tracker", "🏆 Season",
     "🥅 Playoffs", "📊 Team style (real data)"])

# ---------------- Lineups ----------------
def lineup_controls(prof, side, col, gs):
    _, d_out = default_lineup(prof)
    g_def, g_status, g_note = gs
    with col:
        st.markdown(f"**{_label(prof['team'])}**")
        gk = prof["goalies"]
        if gk:
            ids = [g["id"] for g in gk]
            idx = ids.index(g_def) if g_def in ids else 0
            goalie = st.selectbox(
                "Starting goalie", ids, index=idx, key=f"gk_{side}_{prof['team']}_{g_def}",
                format_func=lambda i: next(f"{g['name']}  ({g['svp']:.3f} SV%, {g['gp']} GP)" for g in gk if g["id"] == i))
            st.caption(("✏️ Your pick" if goalie != g_def else GOALIE_ICON.get(g_status, g_status))
                       + (f" · {g_note}" if g_note else ""))
        else:
            goalie = None
            st.caption("No goalie stats yet – using league average.")
        names = {s["id"]: f"{s['name']} ({s['pos']}, {s['g']}G {s['a']}A)" for s in prof["skaters"]}
        out = st.multiselect(
            "Out tonight (injured / scratched)", list(names), default=[i for i in d_out if i in names],
            key=f"out_{side}_{prof['team']}", format_func=lambda i: names[i],
            help="Pre-filled with players who weren't in the last two lineups. "
                 "Remove a name to bring a player back from injury; add a name to sit him.")
    return goalie, tuple(sorted(out))


SHORT_ICON = {"Confirmed": "✅", "Likely": "🟡", "Projected": "⚪", "Recent": "🔁"}


def simulate_slate(day_str, games, n, n_recent_, save, progress=None):
    """Simulate every game on a date with confirmed goalies; optionally save to the tracker."""
    rows, records, evts = [], [], []
    key = odds_api_key() if save else ""
    if key:
        try:
            evts = odds_feed(key)[0]
        except Exception:
            evts = []
    for k, g in enumerate(games):
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        try:
            hpp, app = build_profile(h, n_recent_), build_profile(a, n_recent_)
            gs = starting_goalies(h, a, day_str, hpp, app)
            hg, ag = gs["home"][0], gs["away"][0]
            ho, ao = default_lineup(hpp)[1], default_lineup(app)[1]
            r = run_matchup(h, a, hg, ag, ho, ao, n, g.get("gameType") == 3, n_recent_)
            try:
                start = pd.Timestamp(g["startTimeUTC"]).tz_convert("America/Los_Angeles").strftime("%I:%M %p PT")
            except Exception:
                start = ""
            ts = r["top_scores"][0][0]
            rows.append({"Start": start, "Matchup": f"{a} @ {h}",
                         "Away win %": round(100 * r["away_win"], 1),
                         "Home win %": round(100 * r["home_win"], 1),
                         "Projected": f"{a} {r['goals_mean'][1]:.1f} – {r['goals_mean'][0]:.1f} {h}",
                         "Most likely": f"{a} {ts[1]}–{ts[0]} {h}",
                         "OT/SO %": round(100 * (r["ot"] + r["so"]), 1),
                         "Goalies": f"{goalie_name(app, ag)} {SHORT_ICON.get(gs['away'][1], '')}  vs  "
                                    f"{goalie_name(hpp, hg)} {SHORT_ICON.get(gs['home'][1], '')}",
                         "Pick": h if r["home_win"] >= 0.5 else a, "_h": h, "_a": a})
            if save:
                evt = next((e for e in evts if same_team(e.get("home_team"), h) and same_team(e.get("away_team"), a)), None)
                gnames = {sd: (goalie_name(hpp if sd == "home" else app, gs[sd][0]), gs[sd][1]) for sd in ("home", "away")}
                records.append(make_record(r, dict(id=g["id"], date=day_str, start=g.get("startTimeUTC", "")),
                                           gnames, n_recent_, "slate", evt))
        except Exception as e:
            rows.append({"Matchup": f"{a} @ {h}", "Pick": f"error: {e}", "_h": h, "_a": a})
        if progress is not None:
            progress.progress((k + 1) / len(games))
    saved = upsert_predictions(records, f"Save {len(records)} NHL predictions for {day_str}") if records else 0
    return rows, saved


def day_games(day_str):
    sched = fetch(f"schedule/{day_str}")
    wk = next((w for w in sched.get("gameWeek", []) if w.get("date") == day_str), None)
    return [g for g in (wk or {}).get("games", []) if g.get("gameType") in (2, 3)]


with tab_match:
    ng = next_game(home, away)
    gstart = starting_goalies(home, away, ng["date"] if ng else None, hp, ap)
    with st.expander("Lineups: starting goalies and injuries", expanded=False):
        c1, c2 = st.columns(2)
        a_goalie, a_out = lineup_controls(ap, "away", c1, gstart["away"])
        h_goalie, h_out = lineup_controls(hp, "home", c2, gstart["home"])

    with st.spinner(f"Simulating {n_sims:,} games…"):
        res = run_matchup(home, away, h_goalie, a_goalie, h_out, a_out, n_sims, playoff, n_recent)
    ch, ca = team_colors(home, away)

    st.subheader(f"{_label(away)}  @  {_label(home)}")

    def _gtag(prof, gid, side):
        stt = "Manual" if gid != gstart[side][0] else gstart[side][1]
        return f"{goalie_name(prof, gid)} {SHORT_ICON.get(stt, '✏️')}", stt

    a_tag, a_stt = _gtag(ap, a_goalie, "away")
    h_tag, h_stt = _gtag(hp, h_goalie, "home")
    when = f"Next meeting: **{ng['date']}**" if ng else "No upcoming game between these teams on the schedule"
    st.markdown(f"{when}  ·  Goalies: {away} **{a_tag}**  vs  {home} **{h_tag}**  "
                "<span style='color:#888;font-size:0.85em'>(✅ confirmed · 🟡 likely · ⚪ projected · "
                "🔁 guess from recent starts · ✏️ your pick)</span>", unsafe_allow_html=True)
    m = st.columns(5)
    m[0].metric(f"{away} win", f"{100 * res['away_win']:.1f}%")
    m[1].metric(f"{home} win", f"{100 * res['home_win']:.1f}%")
    m[2].metric("Projected score", f"{away} {res['goals_mean'][1]:.2f} – {res['goals_mean'][0]:.2f} {home}")
    m[3].metric("Goes to OT / SO", f"{100 * (res['ot'] + res['so']):.1f}%")
    s0 = res["top_scores"][0]
    m[4].metric("Most common final", f"{away} {s0[0][1]} – {s0[0][0]} {home}", f"{100 * s0[1]:.1f}% of sims",
                delta_color="off")

    if ng:
        if st.button("📌 Save this prediction to the tracker", key="save_match",
                     help="Records this prediction so the 📈 Tracker can grade it after the game. "
                          "Includes sportsbook odds if you've entered an Odds API key (3 credits)."):
            try:
                evt = None
                if odds_api_key():
                    evt = next((e for e in odds_feed(odds_api_key())[0] if same_team(e.get("home_team"), home)
                                and same_team(e.get("away_team"), away)), None)
                rec = make_record(res, ng, {"home": (goalie_name(hp, h_goalie), h_stt),
                                            "away": (goalie_name(ap, a_goalie), a_stt)}, n_recent, "matchup", evt)
                if upsert_predictions([rec], f"Save prediction {away} @ {home} {ng['date']}"):
                    st.success(f"Saved: {away} @ {home} on {ng['date']}. The 📈 Tracker will grade it after the game.")
                else:
                    st.info("That game has already started, so its saved prediction is locked.")
            except Exception as e:
                st.error(f"Couldn't save the prediction ({e}).")

    left, right = st.columns([1.1, 1])
    with left:
        st.markdown("**Most common final scores**")
        labels = [f"{away} {s[1]}–{s[0]} {home}" for s, _ in res["top_scores"][:10]]
        colors = [ch if s[0] > s[1] else ca for s, _ in res["top_scores"][:10]]
        fig = go.Figure(go.Bar(x=[100 * p for _, p in res["top_scores"][:10]], y=labels, orientation="h",
                               marker_color=colors, text=[f"{100 * p:.1f}%" for _, p in res["top_scores"][:10]],
                               textposition="outside"))
        fig.update_layout(height=380, yaxis=dict(autorange="reversed"), xaxis_title="% of simulations",
                          margin=dict(t=10, b=40, l=10, r=40))
        st.plotly_chart(fig, width="stretch")
    with right:
        st.markdown("**Goals per team**")
        fig = go.Figure()
        for i, (ab, col) in enumerate(((home, ch), (away, ca))):
            fig.add_trace(go.Bar(x=[str(k) if k < 9 else "9+" for k in range(10)], y=100 * res["goal_dist"][i],
                                 name=ab, marker_color=col))
        fig.update_layout(barmode="group", height=380, yaxis_title="% of simulations", xaxis_title="Goals",
                          margin=dict(t=10, b=40), legend=dict(orientation="h", y=1.1))
        st.plotly_chart(fig, width="stretch")

    st.markdown("**Team stats: simulated game vs. real recent play**")
    tot = res["total"]
    rows = [
        ("Win %", f"{100 * res['away_win']:.1f}%", f"{100 * res['home_win']:.1f}%"),
        ("Regulation win %", f"{100 * res['away_reg_win']:.1f}%", f"{100 * res['home_reg_win']:.1f}%"),
        ("Wins by 2+ (puck line)", f"{100 * res['away_cover']:.1f}%", f"{100 * res['home_cover']:.1f}%"),
        ("Sim goals", f"{res['goals_mean'][1]:.2f}", f"{res['goals_mean'][0]:.2f}"),
        ("Sim shots on goal", f"{res['sog_mean'][1]:.1f}", f"{res['sog_mean'][0]:.1f}"),
        ("Sim power-play goals / chances", f"{res['ppg_mean'][1]:.2f} / {res['ppo_mean'][1]:.2f}",
         f"{res['ppg_mean'][0]:.2f} / {res['ppo_mean'][0]:.2f}"),
        ("Sim empty-net goals", f"{res['eng_mean'][1]:.2f}", f"{res['eng_mean'][0]:.2f}"),
        ("Goalie (sim SV%)", f"{res['goalies'][1]['Goalie']} ({res['goalies'][1]['SV%']:.3f})",
         f"{res['goalies'][0]['Goalie']} ({res['goalies'][0]['SV%']:.3f})"),
    ]
    r_a = standings.set_index("team").loc[away]
    r_h = standings.set_index("team").loc[home]
    rows.append(("Season record (W-L-OTL)", f"{r_a.w}-{r_a.l}-{r_a.otl}", f"{r_h.w}-{r_h.l}-{r_h.otl}"))
    rows.append(("Season GF / GA per game", f"{r_a.gf / max(r_a.gp, 1):.2f} / {r_a.ga / max(r_a.gp, 1):.2f}",
                 f"{r_h.gf / max(r_h.gp, 1):.2f} / {r_h.ga / max(r_h.gp, 1):.2f}"))
    rows.append(("Last 10 / streak", f"{r_a.l10} / {r_a.streak}", f"{r_h.l10} / {r_h.streak}"))
    rows += [
        ("Recent shots on goal for / against", f"{ap['sog_pg']:.1f} / {ap['soga_pg']:.1f}",
         f"{hp['sog_pg']:.1f} / {hp['soga_pg']:.1f}"),
        ("Recent PP% / PK%", f"{pct(ap['pp_pct'])} / {pct(ap['pk_pct'])}", f"{pct(hp['pp_pct'])} / {pct(hp['pk_pct'])}"),
        ("Recent faceoff % / hits per game", f"{pct(ap['fo_pct'])} / {ap['hits_pg']:.1f}",
         f"{pct(hp['fo_pct'])} / {hp['hits_pg']:.1f}"),
        ("Lineup strength vs usual", f"{100 * (res['sides']['away']['off'] - 1):+.0f}%",
         f"{100 * (res['sides']['home']['off'] - 1):+.0f}%"),
    ]
    st.dataframe(pd.DataFrame(rows, columns=["", away, home]), hide_index=True, width="stretch")
    st.caption(f"Total goals: average {tot.mean():.2f}  ·  over 5.5 in {100 * (tot > 5.5).mean():.1f}%  ·  "
               f"over 6.5 in {100 * (tot > 6.5).mean():.1f}% of simulations.")

    st.markdown("**Projected player stats (average per simulated game)**")
    fmt = {"G": "{:.2f}", "A": "{:.2f}", "PTS": "{:.2f}", "SOG": "{:.1f}", "Goal %": "{:.0f}%", "Point %": "{:.0f}%"}
    c1, c2 = st.columns(2)
    for col, i, ab in ((c1, 1, away), (c2, 0, home)):
        with col:
            st.markdown(f"*{_label(ab)}*")
            st.dataframe(res["players"][i].style.format(fmt), hide_index=True, width="stretch", height=420)
            gk = res["goalies"][i]
            st.caption(f"🥅 {gk['Goalie']}: {gk['Saves']:.1f} saves, {gk['GA']:.2f} GA, "
                       f"{gk['SV%']:.3f} SV%, shutout {gk['Shutout %']:.1f}%, wins {gk['Win %']:.1f}%")

# ---------------- Animation ----------------
with tab_anim:
    f = res["featured"]
    fs = f["final"]
    suffix = " (OT)" if f["ot"] else (" (SO)" if f["so"] else "")
    st.subheader(f"How the game most often plays out: {away} {fs[1]} – {fs[0]} {home}{suffix}")
    st.caption("A full simulated game that finished with the most common final score. Press ▶ Play, pick a speed, "
               "or jump to the next goal. The ring around the puck shows which team has it; circles are saved "
               "shots and stars are goals.")
    if hasattr(st, "iframe"):            # newer Streamlit
        st.iframe(animation_html(res), height="content")
    else:                                 # older Streamlit on your PC
        components.html(animation_html(res), height=820, scrolling=False)

    st.markdown("**Win probability** (re-simulated from every moment of the game)")
    st.plotly_chart(winprob_figure(res), width="stretch", key="wp_chart")

    st.markdown("**Game flow**")
    st.plotly_chart(flow_figure(res), width="stretch")

    st.markdown("**Scoring summary & penalties**")
    log_rows = [dict(Time=clock(e[1]) if e[3] != "so" else "Shootout",
                     Play=event_text(e, f["names"], home, away),
                     Score=f"{away} {e[10]} – {e[9]} {home}")
                for e in f["events"] if e[3] in ("goal", "pen", "so")]
    st.dataframe(pd.DataFrame(log_rows), hide_index=True, width="stretch")

    st.markdown("**When goals happen: simulation vs. each team's real recent games**")
    per_lbl = ["1st", "2nd", "3rd", "OT"]
    fig = go.Figure()
    for i, (ab, prof, col) in enumerate(((home, hp, ch), (away, ap, ca))):
        fig.add_trace(go.Bar(x=per_lbl, y=res["goals_by_per"][i], name=f"{ab} simulated", marker_color=col))
        fig.add_trace(go.Bar(x=per_lbl, y=prof["gf_per"], name=f"{ab} real (last {prof['n_games']})",
                             marker=dict(color=col, pattern_shape="/", opacity=0.55)))
    fig.update_layout(barmode="group", height=330, yaxis_title="Goals per game", margin=dict(t=10),
                      legend=dict(orientation="h", y=1.15))
    st.plotly_chart(fig, width="stretch")

# ---------------- Every game on a date ----------------
with tab_slate:
    day = st.date_input("Date", dt.date.today())
    try:
        games = day_games(day.isoformat())
    except Exception as e:
        games = []
        st.error(f"Couldn't load the schedule ({e}).")
    if not games:
        st.info("No NHL games on this date.")
    else:
        st.write(f"**{len(games)} games** on {day:%A, %B %d, %Y}")
        c1, c2 = st.columns([2, 1])
        slate_n = c1.select_slider("Simulations per game", [1000, 2000, 5000], value=2000, key="slate_n")
        save_slate = c2.checkbox("Save to the 📈 Tracker", value=True, key="slate_save",
                                 help="Games that have already started aren't saved or changed.")
        if st.button("Simulate every game", type="primary"):
            try:
                rows, saved = simulate_slate(day.isoformat(), games, slate_n, n_recent, save_slate, st.progress(0.0))
                if save_slate and saved:
                    st.success(f"Saved {saved} prediction(s) to the tracker.")
                elif save_slate:
                    st.info("Nothing new to save (those games have already started).")
            except Exception as e:
                rows = []
                st.error(f"Couldn't finish the slate ({e}).")
            st.session_state["slate"] = rows
        rows = st.session_state.get("slate")
        if rows:
            df = pd.DataFrame(rows)
            st.dataframe(df.drop(columns=["_h", "_a"]), hide_index=True, width="stretch")

            def _open(h, a):
                st.session_state["home"], st.session_state["away"] = h, a

            st.markdown("Open a game in the full simulator (player stats + animation):")
            cols = st.columns(4)
            for k, r in enumerate(rows):
                cols[k % 4].button(r["Matchup"], key=f"open_{k}", on_click=_open, args=(r["_h"], r["_a"]))
            st.caption("After clicking, switch to the 🎯 Matchup or 🎬 Game animation tab.")
        else:
            for g in games:
                st.write(f"• {g['awayTeam']['abbrev']} @ {g['homeTeam']['abbrev']}")

# ---------------- Team style ----------------
with tab_style:
    st.caption("Everything below is measured from each team's real play-by-play in its last "
               f"{n_recent} games. This is what drives the simulation.")
    c1, c2 = st.columns(2)
    for col, prof, color in ((c1, ap, ca), (c2, hp, ch)):
        with col:
            st.markdown(f"### {_label(prof['team'])}")
            st.markdown("Where they shoot from (dark = more shots, ★ = goals)")
            st.plotly_chart(shot_map(prof, "for", color), width="stretch", key=f"map_for_{prof['team']}")
            st.markdown("Where opponents shoot from against them")
            st.plotly_chart(shot_map(prof, "against", "#555555"), width="stretch", key=f"map_ag_{prof['team']}")
            lab = ["Leading", "Tied", "Trailing", "Power play", "Shorthanded"]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=lab, y=[prof["r_for"][s] for s in STATES], name="Shots for /60", marker_color=color))
            fig.add_trace(go.Bar(x=lab, y=[prof["r_ag"][s] for s in STATES], name="Shots against /60", marker_color="#999"))
            fig.add_trace(go.Scatter(x=lab, y=[LG_RATE[s] for s in STATES], mode="markers", name="League avg (for)",
                                     marker=dict(symbol="line-ew-open", size=30, color="black", line=dict(width=3))))
            fig.update_layout(barmode="group", height=320, margin=dict(t=30), title="Shot rates by game state",
                              legend=dict(orientation="h", y=-0.2))
            st.plotly_chart(fig, width="stretch", key=f"rates_{prof['team']}")
            st.write(f"Penalties taken per game: **{prof['pens_pg']:.2f}**  ·  PP: **{pct(prof['pp_pct'])}**  ·  "
                     f"PK: **{pct(prof['pk_pct'])}**  ·  Faceoffs: **{pct(prof['fo_pct'])}**")
            if prof["recent_games"]:
                st.dataframe(pd.DataFrame(prof["recent_games"][::-1]).rename(
                    columns={"date": "Date", "opp": "Opponent", "result": "Result"}),
                    hide_index=True, width="stretch", height=240)

# ---------------- League model gate (season + playoffs) ----------------
def league_ready(key):
    if st.session_state.get("league_ready") == n_recent:
        return True
    st.info("The season and playoff simulators use all 32 teams' real play-by-play "
            f"(about {16 * n_recent} games). The first build takes 1–2 minutes, then it's cached for hours.")
    if st.button("Build the league model", type="primary", key=key):
        with st.spinner("Downloading play-by-play for all 32 teams and simulating every matchup…"):
            league_model(n_recent)
        st.session_state["league_ready"] = n_recent
        st.rerun()
    return False


PCT_COLS = ["Playoffs %", "Division %", "Presidents' %", "Conf final %", "Final %", "Cup %"]


def pct_config(cols):
    return {c: st.column_config.ProgressColumn(c, min_value=0, max_value=100, format="%.1f%%") for c in cols}


# ---------------- Season ----------------
with tab_season:
    st.subheader("Rest-of-season simulator")
    st.caption("Plays every remaining regular-season game thousands of times, adds the results to the "
               "real standings, then plays out the NHL playoff bracket each time.")
    if league_ready("league_btn_season"):
        n_seasons = st.select_slider("Seasons to simulate", [1000, 2000, 5000, 10000], value=2000)
        with st.spinner(f"Simulating {n_seasons:,} seasons…"):
            ss = season_sim(n_recent, n_seasons)
        tbl = ss["table"]
        st.caption(f"{ss['n_games']} games left on the schedule · {ss['n_seasons']:,} simulated seasons")
        fmt_cols = {"Proj PTS": st.column_config.NumberColumn(format="%.1f"),
                    "Low (10%)": st.column_config.NumberColumn(format="%.0f"),
                    "High (90%)": st.column_config.NumberColumn(format="%.0f")}
        view = st.radio("View", ["By division", "Whole league"], horizontal=True)
        if view == "Whole league":
            st.dataframe(tbl, hide_index=True, width="stretch", height=1150,
                         column_config={**pct_config(PCT_COLS), **fmt_cols})
        else:
            for conf, ds in CONF_DIVS.items():
                st.markdown(f"#### {conf} Conference")
                cols = st.columns(2)
                for col, d in zip(cols, ds):
                    with col:
                        st.markdown(f"**{d}**")
                        sub = tbl[tbl["Div"] == d][["Team", "PTS now", "Proj PTS", "Playoffs %",
                                                    "Division %", "Cup %"]]
                        st.dataframe(sub, hide_index=True, width="stretch", column_config={
                            **{c: st.column_config.NumberColumn(format="%.1f%%") for c in PCT_COLS}, **fmt_cols})
        top = tbl.sort_values("Cup %", ascending=False).head(16)
        fig = go.Figure(go.Bar(x=top["Team"], y=top["Cup %"],
                               marker_color=[TEAM_COLORS.get(t, "#888") for t in top["Team"]],
                               text=[f"{v:.1f}%" for v in top["Cup %"]], textposition="outside"))
        fig.update_layout(title="Stanley Cup odds", height=360, yaxis_title="%", margin=dict(t=40))
        st.plotly_chart(fig, width="stretch", key="cup_chart")

# ---------------- Playoffs ----------------
with tab_playoffs:
    st.subheader("Playoff series simulator")
    st.caption("Best-of-7, 2-2-1-1-1 format, playoff overtime (sudden-death 5-on-5).")
    c1, c2, c3 = st.columns(3)
    sa_ = c1.selectbox("Team A", TEAMS, index=TEAMS.index(home), format_func=_label, key="ser_a")
    sb_ = c2.selectbox("Team B", TEAMS, index=TEAMS.index(away), format_func=_label, key="ser_b")
    hi_choice = c3.radio("Home ice", [sa_, sb_], horizontal=True, key="ser_hi")
    if sa_ == sb_:
        st.warning("Pick two different teams.")
    else:
        with st.spinner("Simulating the series…"):
            a_home, a_road = pair_series(sa_, sb_, n_recent)
        if hi_choice == sa_:
            hi_t, lo_t, ph, pa = sa_, sb_, a_home, a_road
        else:
            hi_t, lo_t, ph, pa = sb_, sa_, 1 - a_road, 1 - a_home
        o = series_outcomes(ph, pa)
        p_hi = sum(v for (w, _), v in o.items() if w == "hi")
        m = st.columns(4)
        m[0].metric(f"{hi_t} wins series", f"{100 * p_hi:.1f}%")
        m[1].metric(f"{lo_t} wins series", f"{100 * (1 - p_hi):.1f}%")
        m[2].metric(f"{hi_t} per game at home", f"{100 * ph:.1f}%")
        m[3].metric(f"{hi_t} per game on road", f"{100 * pa:.1f}%")
        ch2, ca2 = team_colors(hi_t, lo_t)
        labels = [f"{hi_t} in {g}" for g in (4, 5, 6, 7)] + [f"{lo_t} in {g}" for g in (4, 5, 6, 7)]
        vals = [o[("hi", g)] * 100 for g in (4, 5, 6, 7)] + [o[("lo", g)] * 100 for g in (4, 5, 6, 7)]
        fig = go.Figure(go.Bar(x=labels, y=vals, marker_color=[ch2] * 4 + [ca2] * 4,
                               text=[f"{v:.1f}%" for v in vals], textposition="outside"))
        fig.update_layout(height=330, yaxis_title="% of series", margin=dict(t=20))
        st.plotly_chart(fig, width="stretch", key="series_chart")

    st.divider()
    st.subheader("If the playoffs started today")
    if league_ready("league_btn_playoffs"):
        bt = today_bracket(n_recent)
        teams_l, SP, key_now, br, counts, n_br = (bt["teams"], bt["SP"], bt["key"], bt["br"],
                                                  bt["counts"], bt["n"])
        N = len(teams_l)
        divs = standings.set_index("team").loc[teams_l, "div"].tolist()
        if not br:
            st.info("Not enough standings data to build a bracket yet.")
        else:
            seed_lbl = {}
            for conf, sides in br.items():
                for k_side, side in enumerate(sides):
                    (dw, wc), (s2, s3) = side
                    d = divs[dw]
                    seed_lbl[dw], seed_lbl[s2], seed_lbl[s3] = f"{d[0]}1", f"{d[0]}2", f"{d[0]}3"
                    seed_lbl[wc] = "WC2" if k_side == 0 else "WC1"
            cols = st.columns(2)
            for col, (conf, sides) in zip(cols, br.items()):
                with col:
                    st.markdown(f"#### {conf} Conference — first round")
                    for side in sides:
                        for a_, b_ in side:
                            hi, lo = (a_, b_) if key_now[a_] >= key_now[b_] else (b_, a_)
                            p = SP[hi, lo]
                            st.write(f"({seed_lbl[hi]}) **{teams_l[hi]}** {100 * p:.0f}%  vs  "
                                     f"({seed_lbl[lo]}) **{teams_l[lo]}** {100 * (1 - p):.0f}%")
            playoff_teams = [i for i in range(N) if counts[i, 0] > 0]
            adv = pd.DataFrame({
                "Seed": [seed_lbl.get(i, "") for i in playoff_teams],
                "Team": [teams_l[i] for i in playoff_teams],
                "Conf": [CONF_OF.get(divs[i], "") for i in playoff_teams],
                "Win round 1 %": counts[playoff_teams, 1] / n_br * 100,
                "Conf final %": counts[playoff_teams, 2] / n_br * 100,
                "Final %": counts[playoff_teams, 3] / n_br * 100,
                "Cup %": counts[playoff_teams, 4] / n_br * 100,
            }).sort_values("Cup %", ascending=False)
            st.markdown("**Odds to reach each round (this bracket, 20,000 simulations)**")
            st.dataframe(adv, hide_index=True, width="stretch", height=600,
                         column_config=pct_config(["Win round 1 %", "Conf final %", "Final %", "Cup %"]))

# ---------------- Betting ----------------
def sim_prob(res, market, is_home, outcome, point):
    """(win, push) probability for one betting outcome, from the simulated games."""
    m, tot = res["margin"], res["total"]
    if market == "h2h":
        win = (m > 0) if is_home else (m < 0)
        return win.mean(), 0.0
    if market == "spreads":
        tm = m if is_home else -m
        return (tm + point > 0).mean(), (tm + point == 0).mean()
    if outcome == "Over":
        return (tot > point).mean(), (tot == point).mean()
    return (tot < point).mean(), (tot == point).mean()


with tab_bet:
    st.subheader(f"Betting view: {away} @ {home}")
    st.caption("Fair odds are what the simulation says each bet is worth with no sportsbook margin. "
               "This is a model, not a guarantee. Bet only what you can afford to lose.")
    rows = []
    for lbl, p in ((f"{away} moneyline", res["away_win"]), (f"{home} moneyline", res["home_win"])):
        rows.append((lbl, p))
    m_, t_ = res["margin"], res["total"]
    rows += [(f"{away} +1.5", (m_ <= 1).mean()), (f"{home} -1.5", (m_ >= 2).mean()),
             (f"{away} -1.5", (m_ <= -2).mean()), (f"{home} +1.5", (m_ >= -1).mean())]
    for line in (5.5, 6.5):
        rows += [(f"Over {line}", (t_ > line).mean()), (f"Under {line}", (t_ < line).mean())]
    fair = pd.DataFrame([dict(Bet=b, **{"Sim chance": p * 100, "Fair odds": american(p)}) for b, p in rows])
    st.markdown("**Fair odds from the simulation**")
    st.dataframe(fair, hide_index=True, width="stretch",
                 column_config={"Sim chance": st.column_config.NumberColumn(format="%.1f%%")})

    st.markdown("**Compare with real sportsbook lines**")
    try:
        default_key = st.secrets.get("ODDS_API_KEY", "")
    except Exception:
        default_key = ""
    api_key = st.text_input("The Odds API key (free at the-odds-api.com)", value=default_key, type="password")
    if api_key and st.button("Get sportsbook lines (uses 3 API credits)"):
        st.session_state["odds_key"] = api_key
    if st.session_state.get("odds_key"):
        try:
            feed, left = odds_feed(st.session_state["odds_key"])
        except Exception as e:
            feed, left = [], "?"
            st.error(f"Couldn't get odds ({e}). Check the API key.")
        evt = next((e for e in feed if same_team(e.get("home_team"), home) and same_team(e.get("away_team"), away)), None)
        if feed and evt is None:
            st.info(f"No sportsbook lines found for {away} @ {home}. Books usually post lines a day or two ahead.")
        if evt:
            out = []
            for bk in evt.get("bookmakers", []):
                for mk in bk.get("markets", []):
                    outs = mk.get("outcomes", [])
                    for oc in outs:
                        price, pt = oc.get("price"), oc.get("point")
                        if price is None:
                            continue
                        if mk["key"] == "totals":
                            is_home, label = None, f"{oc['name']} {pt}"
                            pair = [x for x in outs if x.get("point") == pt]
                        else:
                            is_home = same_team(oc["name"], home)
                            ab = home if is_home else away
                            label = f"{ab} moneyline" if mk["key"] == "h2h" else f"{ab} {pt:+g}"
                            pair = [x for x in outs if mk["key"] == "h2h" or abs(x.get("point") or 0) == abs(pt or 0)]
                        pw, pp_ = sim_prob(res, mk["key"], is_home, oc["name"], pt)
                        novig = implied(price) / sum(implied(x["price"]) for x in pair)
                        ev = pw * payout(price) - (1 - pw - pp_) * 100
                        out.append({"Bet": label, "Book": bk.get("title", ""), "Odds": f"{price:+d}",
                                    "Book chance (no vig)": novig * 100, "Sim chance": pw * 100,
                                    "Edge": (pw - novig) * 100, "EV per $100": ev})
            if out:
                bets = pd.DataFrame(out)
                best = bets.sort_values("EV per $100", ascending=False).drop_duplicates("Bet")
                cfg = {"Book chance (no vig)": st.column_config.NumberColumn(format="%.1f%%"),
                       "Sim chance": st.column_config.NumberColumn(format="%.1f%%"),
                       "Edge": st.column_config.NumberColumn(format="%+.1f%%"),
                       "EV per $100": st.column_config.NumberColumn(format="$%+.2f")}
                st.markdown("**Best price for each bet** (positive EV = the sim thinks the price is too generous)")
                st.dataframe(best, hide_index=True, width="stretch", column_config=cfg)
                with st.expander("Every book"):
                    st.dataframe(bets.sort_values(["Bet", "EV per $100"], ascending=[True, False]),
                                 hide_index=True, width="stretch", column_config=cfg)
            st.caption(f"API credits left this month: {left}")

# ---------------- Tracker ----------------
with tab_track:
    st.subheader("Prediction tracker")
    st.caption("Every saved prediction is graded automatically once the game is final, so you can see whether the "
               "model (and its betting edges) actually hold up. A prediction locks when the game starts.")
    conf = _gh_conf()
    if conf:
        st.caption(f"💾 Saving to GitHub: **{conf['repo']}** · branch `{conf['branch']}` · file `{conf['path']}`")
    else:
        st.caption(f"💾 Saving to a file on this computer: `{LOCAL_TRACK}`")

    tc1, tc2 = st.columns([1.2, 2])
    today = dt.date.today().isoformat()
    if tc1.button("Simulate & save today's games", type="primary", key="track_today",
                  help="Runs every game on today's schedule (2,000 sims each, confirmed goalies) and saves the predictions."):
        try:
            g_today = day_games(today)
            if g_today:
                _, saved = simulate_slate(today, g_today, 2000, n_recent, True, st.progress(0.0))
                if saved:
                    st.success(f"Saved {saved} prediction(s) for today.")
                else:
                    st.info("Nothing new to save (today's games have already started).")
            else:
                st.info("No NHL games today.")
        except Exception as e:
            st.error(f"Couldn't run today's games ({e}).")
    tc2.caption("Tip: do this a little before the first puck drop, after goalies are confirmed. "
                "You can re-run it; predictions keep updating until each game starts.")

    try:
        tdf, newly = grade_tracker()
        if newly:
            st.toast(f"Graded {newly} finished game(s).")
    except Exception as e:
        tdf = None
        st.error(f"Couldn't read or update the tracker ({e}). If you set up GitHub, check the token and repo name.")

    if tdf is not None and len(tdf):
        tdf = tdf.copy()
        for c in ("p_home", "proj_home", "proj_away", "final_home", "final_away", "p_over_line", "p_under_line"):
            tdf[c] = pd.to_numeric(tdf[c], errors="coerce")
        done = tdf[tdf["final_home"].notna()].copy()
        if len(done):
            home_won = (done["final_home"] > done["final_away"]).astype(float)
            pick_home = done["p_home"] >= 0.5
            correct = (home_won == 1) == pick_home
            p_pick = np.where(pick_home, done["p_home"], 1 - done["p_home"])
            brier = float(((done["p_home"] - home_won) ** 2).mean())
            tot_err = float((done["proj_home"] + done["proj_away"] - done["final_home"] - done["final_away"]).abs().mean())
            m = st.columns(4)
            m[0].metric("Games graded", f"{len(done)}")
            m[1].metric("Picks correct", f"{100 * correct.mean():.1f}%", f"{int(correct.sum())}-{int((~correct).sum())}",
                        delta_color="off")
            m[2].metric("Brier score", f"{brier:.3f}", "lower is better · coin flip = 0.250", delta_color="off")
            m[3].metric("Total goals miss", f"{tot_err:.2f}", "average goals off per game", delta_color="off")

            st.markdown("**Are the percentages honest?** When the model says a team wins X% of the time, "
                        "does it actually win about X% of the time?")
            bins = [0.5, 0.55, 0.6, 0.65, 0.7, 1.01]
            labels = ["50–55%", "55–60%", "60–65%", "65–70%", "70%+"]
            cut = pd.cut(p_pick, bins, labels=labels, right=False)
            cal = pd.DataFrame({"bin": cut, "pred": p_pick, "hit": correct.to_numpy()}).groupby("bin", observed=False)
            cal = cal.agg(games=("hit", "size"), predicted=("pred", "mean"), actual=("hit", "mean")).reset_index()
            cal = cal[cal["games"] > 0]
            fig = go.Figure()
            fig.add_trace(go.Bar(x=cal["bin"].astype(str), y=cal["actual"] * 100, name="Actually won",
                                 marker_color="#2e86de", text=[f"{v:.0f}% ({n})" for v, n in zip(cal["actual"] * 100, cal["games"])],
                                 textposition="outside"))
            fig.add_trace(go.Scatter(x=cal["bin"].astype(str), y=cal["predicted"] * 100, name="Model said",
                                     mode="markers+lines", marker=dict(size=12, color="black"), line=dict(dash="dot")))
            fig.update_layout(height=330, yaxis=dict(range=[0, 105], title="% of picks that won"),
                              xaxis_title="Model's confidence in its pick (games in brackets)",
                              margin=dict(t=20), legend=dict(orientation="h", y=1.12))
            st.plotly_chart(fig, width="stretch", key="calib")
            if len(done) < 50:
                st.caption(f"Only {len(done)} games so far. These numbers bounce around a lot until you have "
                           "a few hundred, so don't read too much into them yet.")

            st.markdown("**Betting results** (flat $100 on every bet the model liked when the prediction was saved)")
            has_odds = done["ml_home"].notna().any() or done["over_price"].notna().any()
            if not has_odds:
                st.info("No sportsbook odds were saved with these predictions yet. Enter your Odds API key on the "
                        "💰 Betting tab, and odds are saved automatically with each prediction.")
            else:
                min_ev = st.select_slider("Only count bets with expected profit of at least (per $100)",
                                          [0, 2, 5, 10, 15], value=0, format_func=lambda v: f"${v}")
                bets = bet_results(done, min_ev)
                if bets.empty:
                    st.info("No bets met that bar.")
                else:
                    wins, losses = int((bets["profit"] > 0).sum()), int((bets["profit"] < 0).sum())
                    pushes = len(bets) - wins - losses
                    profit = float(bets["profit"].sum())
                    b = st.columns(4)
                    b[0].metric("Bets", f"{len(bets)}")
                    b[1].metric("Record", f"{wins}-{losses}" + (f"-{pushes}" if pushes else ""))
                    b[2].metric("Profit", f"${profit:+,.0f}")
                    b[3].metric("Return per bet", f"{profit / len(bets):+.1f}%")
                    curve = bets.sort_values("date").reset_index(drop=True)
                    fig = go.Figure(go.Scatter(x=list(range(1, len(curve) + 1)), y=curve["profit"].cumsum(),
                                               mode="lines", line=dict(width=3, color="#27ae60" if profit >= 0 else "#c0392b"),
                                               hovertext=curve["game"] + " · " + curve["bet"], hoverinfo="text+y"))
                    fig.add_hline(y=0, line=dict(color="#999", dash="dot"))
                    fig.update_layout(height=300, xaxis_title="Bet number", yaxis_title="Running profit ($)",
                                      margin=dict(t=20))
                    st.plotly_chart(fig, width="stretch", key="profit_curve")
                    with st.expander("Every bet"):
                        st.dataframe(curve.rename(columns={"date": "Date", "game": "Game", "bet": "Bet",
                                                           "ev": "Expected $/100", "profit": "Result $"}),
                                     hide_index=True, width="stretch",
                                     column_config={"Expected $/100": st.column_config.NumberColumn(format="$%+.2f"),
                                                    "Result $": st.column_config.NumberColumn(format="$%+.0f")})
        else:
            st.info("No finished games yet. Predictions are graded automatically the next time you open the app "
                    "after the game ends.")

        def _row(r):
            pick = r["home"] if r["p_home"] >= 0.5 else r["away"]
            pp = max(r["p_home"], 1 - r["p_home"]) * 100
            if pd.notna(r["final_home"]):
                won = (r["final_home"] > r["final_away"]) == (r["p_home"] >= 0.5)
                final = f"{r['away']} {int(r['final_away'])} – {int(r['final_home'])} {r['home']}" + \
                    (f" ({r['end']})" if r["end"] in ("OT", "SO") else "")
                result = "✅" if won else "❌"
            else:
                final, result = "", "⏳"
            return {"Date": r["date"], "Game": f"{r['away']} @ {r['home']}", "Pick": f"{pick} {pp:.0f}%",
                    "Projected": f"{r['proj_away']:.1f} – {r['proj_home']:.1f}", "Final": final, "": result,
                    "Goalies": f"{r['away_goalie']} vs {r['home_goalie']}", "Odds saved": "✓" if pd.notna(r["ml_home"]) else ""}

        st.markdown("**All saved predictions**")
        table = pd.DataFrame([_row(r) for _, r in tdf.sort_values(["date", "start_utc"], ascending=False).iterrows()])
        st.dataframe(table, hide_index=True, width="stretch", height=420)
        st.download_button("⬇️ Download all predictions (CSV)", tdf.to_csv(index=False), "nhl_predictions.csv", "text/csv")
    elif tdf is not None:
        st.info("No predictions saved yet. Use **Simulate & save today's games** above, the 📅 tab, "
                "or **📌 Save this prediction** on the 🎯 Matchup tab.")

    with st.expander("☁️ Keep your tracker on Streamlit Cloud (one-time GitHub setup)"):
        st.markdown('''
Streamlit Cloud wipes saved files whenever the app restarts, so on the cloud app the tracker
saves to your GitHub repo instead. On your PC you don't need this; it saves to `nhl_predictions.csv`
next to `nhl_app.py`.

1. On github.com: your profile picture → **Settings** → **Developer settings** → **Personal access tokens**
   → **Fine-grained tokens** → **Generate new token**.
2. Name it `nhl tracker`, set **Repository access** to *Only select repositories* → your `nhl-sim` repo.
3. Under **Permissions → Repository permissions**, set **Contents** to **Read and write**. Click **Generate token** and copy it.
4. In Streamlit Cloud: your app → **⋮** → **Settings** → **Secrets**, add these lines (use your GitHub username), then **Save**:
```
GITHUB_TOKEN = "github_pat_paste_your_token_here"
GITHUB_REPO = "your-github-username/nhl-sim"
```
Predictions are saved on a separate branch called `tracker-data`, so saving them never restarts your app.
You can also add `ODDS_API_KEY = "..."` there so odds are saved automatically.
''')