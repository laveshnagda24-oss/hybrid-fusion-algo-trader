"""
main.py
================================================================================
CLOUD PRODUCTION LIVE & PAPER TRADING SERVER (RENDER.COM + TURSO PERSISTENCE)
STRATEGY: F12_CAUSAL_5xVol (2 TRADES / DAY: SLOT A W12 + SLOT B FINAL_C1)
Fixed ₹60,000 Base Capital (2 x ₹30,000) | 5x MIS Leverage (₹1,50,000 / 1.0x Slot)
Strict Causal Waterfall: 09:45 OPEN (Tiers 1-3) -> 10:15 OPEN (Tiers 4-5) | 15:20 EOD
Liquidity Guard: Entry 1m Bar Volume >= 5.0 * Order Qty | 0.05% Slippage + 2026 MIS Costs
================================================================================
"""

import os
import sys
import time
import json
import logging
import datetime
import threading
import urllib.request
import urllib.error
import requests
import pandas as pd
import numpy as np
import pytz
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from concurrent.futures import ThreadPoolExecutor, as_completed
import uvicorn

LOG_FORMAT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
logger = logging.getLogger("RenderF12Causal")

IST = pytz.timezone("Asia/Kolkata")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "https://market-data-lavesh.aws-ap-south-1.turso.io")
TURSO_TOKEN = os.environ.get(
    "TURSO_AUTH_TOKEN",
    "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3OTMyODUxNjIsImlkIjoiMDFhMTg4MWYtY2EwMS03MzI5LTljNzUtOTU5NGYyZTdmYTVkIiwia2lkIjoiYUVPVmtxaENSbEVZb21JMTFSOHlZZUliWE5UTXJ1OXhpNEx2ZkJBOEI5NCIsInJpZCI6ImZkZWZhNzEzLWQxZmUtNDJjNy04NTVlLTBhOGUwY2I0NDY0NCJ9.KMq5bJSwJfGh8X-cTTeL1hWaVsTb1Sz9hAlBTorfIKS0rJBkCGA9gnN47A-yuxtsr_k84oSXOj45j6d9Ht8DDA",
)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept": "application/json",
}

SLIPPAGE = 0.0005  # 0.05% adverse slippage on entry and exit
MIN_VOL_MULT = 5.0 # Strict liquidity guard: entry 1m bar volume >= 5.0 * qty

BRACKETS = {
    "S_SL1.80_TP3.0_4.8_R0.2": {"direction": "SHORT", "split": True,  "sl": 1.80, "tp1": 3.00, "tp2": 4.80, "ratchet": 0.20},
    "S_SL1.40_TP2.4_4.0_R0.3": {"direction": "SHORT", "split": True,  "sl": 1.40, "tp1": 2.40, "tp2": 4.00, "ratchet": 0.30},
    "L_SL1.80_TP3.0_4.8_R0.2": {"direction": "LONG",  "split": True,  "sl": 1.80, "tp1": 3.00, "tp2": 4.80, "ratchet": 0.20},
    "L_SL2.00_1LEG_TP3.6":     {"direction": "LONG",  "split": False, "sl": 2.00, "tp1": 3.60, "tp2": 3.60, "ratchet": 0.00},
    "L_SL1.40_1LEG_TP2.8":     {"direction": "LONG",  "split": False, "sl": 1.40, "tp1": 2.80, "tp2": 2.80, "ratchet": 0.00},
}

log_records = []


class MemoryLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        log_records.append(msg)
        if len(log_records) > 250:
            log_records.pop(0)


mem_handler = MemoryLogHandler()
mem_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(mem_handler)


# ------------------------------------------------------------------------------
# 1. 2026 STATUTORY COST & FRICTION CALCULATOR (NSE CASH INTRADAY)
# ------------------------------------------------------------------------------
def calc_statutory_costs(entry_px, q1, q2, exit1_px, exit2_px, direction="SHORT"):
    v_ent = entry_px * (q1 + q2)
    v_ex1 = exit1_px * q1
    v_ex2 = exit2_px * q2

    if direction == "SHORT":
        sell_v = v_ent
        buy_v = v_ex1 + v_ex2
    else:
        buy_v = v_ent
        sell_v = v_ex1 + v_ex2

    brk = min(20.0, 0.0003 * v_ent) + min(20.0, 0.0003 * v_ex1)
    if q2 > 0:
        brk += min(20.0, 0.0003 * v_ex2)
    stt = 0.00025 * sell_v
    txn = 0.0000307 * (sell_v + buy_v)
    sebi = 0.000001 * (sell_v + buy_v)
    stamp = 0.00003 * buy_v
    gst = 0.18 * (brk + txn + sebi)
    return brk + stt + txn + sebi + stamp + gst


# ------------------------------------------------------------------------------
# 2. TURSO CLOUD DATABASE HELPERS (PERMANENT MULTI-DAY PERSISTENCE)
# ------------------------------------------------------------------------------
def query_turso_raw(sql, params=None):
    headers = {
        "Authorization": f"Bearer {TURSO_TOKEN}",
        "Content-Type": "application/json",
    }
    stmt = {"sql": sql}
    if params:
        args = []
        for p in params:
            if p is None:
                args.append({"type": "null"})
            elif isinstance(p, int):
                args.append({"type": "integer", "value": str(p)})
            elif isinstance(p, float):
                args.append({"type": "float", "value": p})
            else:
                args.append({"type": "text", "value": str(p)})
        stmt["args"] = args

    payload = {"requests": [{"type": "execute", "stmt": stmt}]}
    try:
        resp = requests.post(f"{TURSO_URL}/v2/pipeline", headers=headers, json=payload, timeout=8)
        if resp.status_code == 200:
            res = resp.json().get("results", [{}])[0].get("response", {}).get("result", {})
            cols = [c["name"] for c in res.get("cols", [])]
            rows = [[c.get("value") for c in r] for r in res.get("rows", [])]
            return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        logger.error(f"Turso query error: {e}")
    return pd.DataFrame()


def save_trade_to_turso(pos, exit1_px, exit1_reason, exit1_time, exit2_px, exit2_reason, exit2_time, net_inr, gross_inr, costs, ending_capital):
    today_str = datetime.datetime.now(IST).strftime("%Y-%m-%d")
    check_sql = "SELECT id FROM champion_cloud_paper_trades WHERE trade_date = ? AND symbol = ? AND entry_time = ? AND strategy = ? LIMIT 1"
    df_exists = query_turso_raw(check_sql, [today_str, pos["symbol"], pos["entry_time"], pos["strategy"]])
    if not df_exists.empty:
        logger.info(f"Trade [{pos['slot']}] {pos['symbol']} already recorded in Turso for {today_str}. Skipping duplicate.")
        return

    sql = """
    INSERT INTO champion_cloud_paper_trades
    (trade_date, symbol, direction, sector, entry_time, exit_time, entry_price, exit_price, shares, exposure_inr, pnl_net_inr, pnl_net_pct, exit_reason, strategy, ending_capital)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    if pos["q2"] > 0:
        full_reason = f"L1: {exit1_reason} @ {exit1_time} | L2: {exit2_reason} @ {exit2_time}"
        avg_exit = (exit1_px * pos["q1"] + exit2_px * pos["q2"]) / pos["qty"]
    else:
        full_reason = f"1LEG: {exit1_reason} @ {exit1_time}"
        avg_exit = exit1_px

    net_pct = (net_inr / pos["slot_exposure"]) * 100.0 if pos["slot_exposure"] > 0 else 0.0

    params = [
        today_str,
        pos["symbol"],
        pos["direction"],
        pos.get("sector", "N/A"),
        pos["entry_time"],
        exit2_time or exit1_time,
        pos["entry_px"],
        round(avg_exit, 2),
        pos["qty"],
        round(pos["slot_exposure"], 2),
        round(net_inr, 2),
        round(net_pct, 2),
        full_reason,
        pos["strategy"],
        round(ending_capital, 2),
    ]
    query_turso_raw(sql, params)
    logger.info(f"Saved [{pos['slot']}] {pos['symbol']} ({pos['direction']}) to Turso DB (Net: Rs {net_inr:+.2f}).")


def save_eod_balance_to_turso(today_str, starting_cap, day_pnl, ending_cap, trades_count, wins, losses):
    sql = """
    INSERT OR REPLACE INTO champion_daily_balance
    (trade_date, starting_capital, day_pnl, ending_capital, trades_count, win_count, loss_count, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    params = [today_str, round(starting_cap, 2), round(day_pnl, 2), round(ending_cap, 2), trades_count, wins, losses]
    query_turso_raw(sql, params)
    logger.info(f"Saved EOD Balance {today_str} to Turso: Ending Equity = Rs {ending_cap:,.2f}")


# ------------------------------------------------------------------------------
# 3. CORE F12_CAUSAL_5xVol ENGINE (2 TRADES / DAY: SLOT A W12 + SLOT B FINAL_C1)
# ------------------------------------------------------------------------------
class F12CausalEngine:
    def __init__(self, capital_base=60000.0, dry_run=True, enable_1015_catch_all=True):
        self.capital_base = capital_base          # Fixed ₹60,000/day total budget
        self.slot_capital = capital_base / 2.0    # ₹30,000 per slot
        self.leverage = 5.0                       # 5x MIS Intraday Leverage
        self.slot_exposure = self.slot_capital * self.leverage  # ₹1,50,000 per 1.0x slot
        self.dry_run = dry_run
        self.enable_1015_catch_all = enable_1015_catch_all

        self.max_trades_today = 2
        self.active_positions = {"A": None, "B": None}
        self.slot_filled_today = {"A": False, "B": False}
        self.closed_trades = []
        self.candidates = []
        self.today_pnl = 0.0
        self.cumulative_capital = self.capital_base

        self.status = "INITIALIZING"
        self.last_update_ist = ""
        self.scan_0945_done = False
        self.scan_1015_done = False
        self.eod_done = False

        self._5d_vol_cache = {}  # (isin, n_bars) -> mean 5d window volume

        self.init_capital_from_db()
        self.load_watchlist()

    def init_capital_from_db(self):
        df_bal = query_turso_raw("SELECT ending_capital FROM champion_daily_balance ORDER BY trade_date DESC LIMIT 1")
        if not df_bal.empty and df_bal.iloc[0]["ending_capital"] is not None:
            self.cumulative_capital = float(df_bal.iloc[0]["ending_capital"])
            logger.info(f"Loaded equity curve from Turso DB: Rs {self.cumulative_capital:,.2f}")
        else:
            self.cumulative_capital = self.capital_base
            logger.info(f"Initialized fresh tracking equity: Rs {self.cumulative_capital:,.2f}")

    def load_watchlist(self):
        n500_path = os.path.join(DATA_DIR, "ind_nifty500list.csv")
        if not os.path.exists(n500_path):
            n500_path = os.path.join(os.path.dirname(ROOT_DIR), "ind_nifty500list.csv")
        if os.path.exists(n500_path):
            try:
                df = pd.read_csv(n500_path)
                valid = df.dropna(subset=["Symbol", "ISIN Code"]).to_dict("records")
                self.candidates = valid
                logger.info(f"Loaded {len(valid)} stocks from Nifty 500 universe for F12_CAUSAL_5xVol.")
            except Exception as e:
                logger.error(f"Error loading universe: {e}")

    def fetch_live_1min_bars(self, isin):
        url = f"https://api.upstox.com/v2/historical-candle/intraday/NSE_EQ%7C{isin}/1minute"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candles = data.get("data", {}).get("candles", [])
                if candles:
                    return sorted(candles, key=lambda x: x[0])
        except Exception:
            pass
        return []

    def fetch_daily_bars(self, isin, today_str):
        start_dt = (datetime.datetime.strptime(today_str, "%Y-%m-%d") - datetime.timedelta(days=45)).strftime("%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/NSE_EQ%7C{isin}/day/{today_str}/{start_dt}"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candles = sorted(data.get("data", {}).get("candles", []), key=lambda x: x[0])
                return candles
        except Exception:
            pass
        return []

    def fetch_5d_avg_window_vol(self, isin, today_str, n_bars):
        key = (isin, n_bars)
        if key in self._5d_vol_cache:
            return self._5d_vol_cache[key]
        end_dt = (datetime.datetime.strptime(today_str, "%Y-%m-%d") - datetime.timedelta(days=1)).strftime("%Y-%m-%d")
        start_dt = (datetime.datetime.strptime(today_str, "%Y-%m-%d") - datetime.timedelta(days=12)).strftime("%Y-%m-%d")
        url = f"https://api.upstox.com/v2/historical-candle/NSE_EQ%7C{isin}/1minute/{end_dt}/{start_dt}"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                candles = sorted(data.get("data", {}).get("candles", []), key=lambda x: x[0])
                if candles:
                    by_day = {}
                    for c in candles:
                        d_str = str(c[0])[:10]
                        by_day.setdefault(d_str, []).append(float(c[5]))
                    last_5 = sorted(by_day.keys())[-5:]
                    vols = [sum(by_day[d][:n_bars]) for d in last_5 if len(by_day[d]) >= n_bars]
                    if vols:
                        avg_v = float(np.mean(vols))
                        self._5d_vol_cache[key] = avg_v
                        return avg_v
        except Exception:
            pass
        return None

    def compute_window_features(self, bars_1m, daily_bars, n_bars, avg_5d_vol=None):
        """
        Zero look-ahead window feature calculator.
        n_bars = 30 -> 09:15..09:44, entry at bars_1m[30] (09:45 OPEN)
        n_bars = 60 -> 09:15..10:14, entry at bars_1m[60] (10:15 OPEN)
        """
        if len(bars_1m) < n_bars + 1 or len(daily_bars) < 11:
            return None

        prev_close = float(daily_bars[-2][4])
        if prev_close < 50.0:
            return None

        c2c_moves = [
            abs(float(daily_bars[i][4]) / float(daily_bars[i - 1][4]) - 1.0) * 100.0
            for i in range(len(daily_bars) - 10, len(daily_bars))
            if float(daily_bars[i - 1][4]) > 0
        ]
        if any(m > 25.0 for m in c2c_moves):
            return None

        trs = []
        for i in range(len(daily_bars) - 11, len(daily_bars) - 1):
            h_d = float(daily_bars[i][2])
            l_d = float(daily_bars[i][3])
            pc_d = float(daily_bars[i - 1][4])
            trs.append(max(h_d - l_d, abs(h_d - pc_d), abs(l_d - pc_d)))
        atr10 = (float(np.mean(trs)) / prev_close) * 100.0

        prev1d_ret = (float(daily_bars[-2][4]) / float(daily_bars[-3][4]) - 1.0) * 100.0
        prev5d_ret = float(
            sum((float(daily_bars[i][4]) / float(daily_bars[i - 1][4]) - 1.0) * 100.0 for i in range(len(daily_bars) - 6, len(daily_bars) - 1))
        )

        win_bars = bars_1m[:n_bars]
        open_0915 = float(win_bars[0][1])
        if open_0915 <= 0:
            return None
        open_gap = ((open_0915 - prev_close) / prev_close) * 100.0
        if abs(open_gap) > 12.0:
            return None

        closes = [float(b[4]) for b in win_bars]
        highs = [float(b[2]) for b in win_bars]
        lows = [float(b[3]) for b in win_bars]
        vols = [float(b[5]) for b in win_bars]

        cum_vol = sum(vols)
        if cum_vol <= 0:
            return None

        cum_val = sum(((highs[t] + lows[t] + closes[t]) / 3.0) * vols[t] for t in range(n_bars))
        vwap = cum_val / cum_vol
        dist_vwap = ((closes[-1] - vwap) / vwap) * 100.0
        intra_ret = ((closes[-1] - open_0915) / open_0915) * 100.0

        hi_w = max(highs)
        lo_w = min(lows)
        rng_w = max(hi_w - lo_w, 1e-9)
        clv_intra = ((closes[-1] - lo_w) - (hi_w - closes[-1])) / rng_w

        diffs = [closes[t] - closes[t - 1] for t in range(n_bars - 14, n_bars)]
        up = float(np.mean([max(d, 0.0) for d in diffs]))
        dn = max(float(np.mean([max(-d, 0.0) for d in diffs])), 1e-9)
        rsi14 = 100.0 - (100.0 / (1.0 + up / dn))

        total_turnover = sum(closes[t] * vols[t] for t in range(n_bars))
        turnover_log = float(np.log1p(max(0.0, total_turnover)))

        rel_vol_5d = (cum_vol / avg_5d_vol) if (avg_5d_vol and avg_5d_vol > 0) else 1.0

        entry_bar = bars_1m[n_bars]
        return {
            "prev_close": prev_close,
            "open_gap": open_gap,
            "atr10": atr10,
            "rsi14": rsi14,
            "dist_vwap": dist_vwap,
            "intra_ret": intra_ret,
            "clv_intra": clv_intra,
            "rel_vol_5d": rel_vol_5d,
            "turnover_log": turnover_log,
            "total_turnover": total_turnover,
            "prev1d_ret": prev1d_ret,
            "prev5d_ret": prev5d_ret,
            "entry_open": float(entry_bar[1]),
            "entry_vol": float(entry_bar[5]),
        }

    def build_universe_snapshot(self, n_bars, today_str, fetch_rel_vol=False):
        raw_records = []

        def evaluate_cand(cand):
            sym = cand.get("Symbol")
            isin = cand.get("ISIN Code")
            sec = cand.get("Industry", "UNKNOWN")
            if not isin:
                return None
            bars_1m = self.fetch_live_1min_bars(isin)
            if len(bars_1m) < n_bars + 1:
                return None
            daily_bars = self.fetch_daily_bars(isin, today_str)
            if not daily_bars or len(daily_bars) < 11:
                return None
            avg_5d_vol = self.fetch_5d_avg_window_vol(isin, today_str, n_bars) if fetch_rel_vol else None
            sig = self.compute_window_features(bars_1m, daily_bars, n_bars, avg_5d_vol=avg_5d_vol)
            if not sig:
                return None
            sig["symbol"] = sym
            sig["isin"] = isin
            sig["sector"] = sec
            return sig

        with ThreadPoolExecutor(max_workers=12) as executor:
            futs = [executor.submit(evaluate_cand, c) for c in self.candidates]
            for f in as_completed(futs):
                res = f.result()
                if res:
                    raw_records.append(res)

        if not raw_records:
            return [], {}, {}

        mkt_ret = float(np.mean([r["intra_ret"] for r in raw_records]))
        pct_gap_up = float(np.mean([1.0 if r["open_gap"] >= 0.50 else 0.0 for r in raw_records]))
        mkt_prev1d = float(np.mean([r["prev1d_ret"] for r in raw_records]))
        mkt_prev5d = float(np.mean([r["prev5d_ret"] for r in raw_records]))

        by_sec = {}
        for r in raw_records:
            by_sec.setdefault(r["sector"], []).append(r)

        sec_ret_map = {sec: float(np.mean([x["intra_ret"] for x in grp])) for sec, grp in by_sec.items()}
        sec_prev5d_map = {sec: float(np.mean([x["prev5d_ret"] for x in grp])) for sec, grp in by_sec.items()}

        for r in raw_records:
            r["mkt_ret"] = mkt_ret
            r["sec_ret"] = sec_ret_map.get(r["sector"], 0.0)
            r["sec_prev5d"] = sec_prev5d_map.get(r["sector"], 0.0)
            r["sec_rel_str"] = r["intra_ret"] - r["sec_ret"]

        raw_records.sort(key=lambda x: (-x["turnover_log"], x["symbol"]))
        mkt_stats = {
            "mkt_ret": mkt_ret,
            "pct_gap_up": pct_gap_up,
            "mkt_prev1d": mkt_prev1d,
            "mkt_prev5d": mkt_prev5d,
        }
        return raw_records, mkt_stats, sec_prev5d_map

    def passes_liquidity_guard(self, rec, notional, direction):
        ep = rec["entry_open"] * (1.0 - SLIPPAGE if direction == "SHORT" else 1.0 + SLIPPAGE)
        if ep < 50.0:
            return False
        qty = int(np.floor(notional / ep))
        return qty >= 2 and rec["entry_vol"] >= (MIN_VOL_MULT * qty)

    def pick_first_valid(self, records, predicate, notional, direction):
        for r in records:
            if predicate(r) and self.passes_liquidity_guard(r, notional, direction):
                return r
        return None

    def scan_market_0945(self):
        if self.scan_0945_done:
            return
        logger.info("=" * 80)
        logger.info("09:45 AM IST: EXECUTING F12_CAUSAL_5xVol SCAN (TIERS 1-3)...")
        logger.info("=" * 80)

        today_str = datetime.datetime.now(IST).strftime("%Y-%m-%d")
        records, mkt_stats, _ = self.build_universe_snapshot(30, today_str, fetch_rel_vol=True)
        self.scan_0945_done = True

        if not records:
            logger.info("09:45 Scan: No valid universe records fetched. Pending 10:15 scan.")
            return

        # Tier 1: S3_Exact Regime Switcher LONG (shared by Slot A & Slot B)
        top_a = self.pick_first_valid(
            records,
            lambda x: x["open_gap"] >= 0.60 and x["rsi14"] <= 48.0 and x["atr10"] >= 3.30,
            self.slot_exposure,
            "SHORT",
        )
        top_b = self.pick_first_valid(
            records,
            lambda x: x["open_gap"] >= 0.80 and x["dist_vwap"] <= -0.25 and x["atr10"] >= 3.30,
            self.slot_exposure,
            "SHORT",
        )
        ref_vals = [x["sec_prev5d"] for x in (top_a, top_b) if x is not None]
        picked_sec_5d = float(np.mean(ref_vals)) if ref_vals else -999.0

        s3_r1 = (mkt_stats["pct_gap_up"] >= 0.50) and (picked_sec_5d >= 2.0)
        s3_r2 = (not s3_r1) and (mkt_stats["mkt_prev1d"] >= 0.60) and (mkt_stats["mkt_prev5d"] <= -2.50)

        t1_pick = None
        t1_bracket = None
        t1_tier = None
        if s3_r1:
            t1_pick = self.pick_first_valid(
                records,
                lambda x: x["sec_ret"] >= 0.30 and x["sec_rel_str"] >= 0.30 and x["dist_vwap"] >= 0.20,
                self.slot_exposure,
                "LONG",
            )
            t1_bracket = "L_SL1.80_TP3.0_4.8_R0.2"
            t1_tier = "T1_S3_LONG_R1"
        elif s3_r2:
            t1_pick = self.pick_first_valid(
                records,
                lambda x: x["intra_ret"] >= 0.80 and x["dist_vwap"] >= 0.20 and x["rel_vol_5d"] >= 1.10,
                self.slot_exposure,
                "LONG",
            )
            t1_bracket = "L_SL2.00_1LEG_TP3.6"
            t1_tier = "T1_S3_LONG_R2"

        # Slot A (W12)
        if not self.slot_filled_today["A"]:
            if t1_pick:
                self.enter_position("A", t1_pick, t1_bracket, t1_tier, "09:45", 1.0)
            else:
                pick_a = self.pick_first_valid(
                    records,
                    lambda x: x["open_gap"] >= 0.80 and x["rsi14"] <= 48.0 and x["atr10"] >= 3.30,
                    self.slot_exposure,
                    "SHORT",
                )
                tier_a = "T2_CHAMP2_0945"
                if not pick_a:
                    pick_a = self.pick_first_valid(
                        records,
                        lambda x: x["open_gap"] >= 0.60 and x["rsi14"] <= 48.0 and x["atr10"] >= 3.30,
                        self.slot_exposure,
                        "SHORT",
                    )
                    tier_a = "T3_POOL_A_0945"
                if pick_a:
                    self.enter_position("A", pick_a, "S_SL1.80_TP3.0_4.8_R0.2", tier_a, "09:45", 1.0)
                else:
                    logger.info("[Slot A] No 09:45 setup qualified. Holding cash for 10:15 causal scan.")

        # Slot B (FINAL_C1)
        if not self.slot_filled_today["B"]:
            if t1_pick:
                self.enter_position("B", t1_pick, t1_bracket, t1_tier, "09:45", 1.0)
            else:
                pick_b = self.pick_first_valid(
                    records,
                    lambda x: (
                        ((x["open_gap"] >= 0.60 and x["rsi14"] <= 48.0) or (x["open_gap"] >= 0.80 and x["dist_vwap"] <= -0.25))
                        and x["atr10"] >= 3.30
                        and x["sec_ret"] <= 0.20
                    ),
                    self.slot_exposure,
                    "SHORT",
                )
                if pick_b:
                    self.enter_position("B", pick_b, "S_SL1.80_TP3.0_4.8_R0.2", "T2_UNION_SEC_0945", "09:45", 1.0)
                else:
                    logger.info("[Slot B] No 09:45 setup qualified. Holding cash for 10:15 causal scan.")

    def scan_market_1015(self):
        if self.scan_1015_done:
            return
        self.scan_1015_done = True

        need_a = not self.slot_filled_today["A"]
        need_b = not self.slot_filled_today["B"]
        if not need_a and not need_b:
            logger.info("10:15 Scan: Both Slot A & Slot B already filled at 09:45. Skipping.")
            return

        logger.info("=" * 80)
        logger.info("10:15 AM IST: EXECUTING F12_CAUSAL_5xVol 10:15 FALLBACK SCAN (TIERS 4-5)...")
        logger.info("=" * 80)

        today_str = datetime.datetime.now(IST).strftime("%Y-%m-%d")
        records, mkt_stats, _ = self.build_universe_snapshot(60, today_str, fetch_rel_vol=True)
        if not records:
            logger.info("10:15 Scan: No valid universe records fetched.")
            return

        # Tier 4: 10:15 Bullish Reversal LONG (ranked by sec_rel_str DESC)
        by_sr = sorted(records, key=lambda x: (-x["sec_rel_str"], x["symbol"]))
        rev_pick = self.pick_first_valid(
            by_sr,
            lambda x: (
                x["open_gap"] <= -0.40
                and x["dist_vwap"] >= 0.15
                and x["clv_intra"] >= 0.20
                and x["rel_vol_5d"] >= 1.15
                and x["mkt_ret"] >= 0.20
                and x["sec_ret"] >= 0.0
            ),
            self.slot_exposure,
            "LONG",
        )

        def pick_catch_all(scale):
            if not self.enable_1015_catch_all:
                return None, None, None
            notional = self.slot_exposure * scale
            if mkt_stats["mkt_ret"] >= 0.0:
                rc = self.pick_first_valid(records, lambda x: x["dist_vwap"] >= 0.05, notional, "LONG")
                if rc:
                    return rc, "L_SL1.40_1LEG_TP2.8", "T5_CATCH_15L"
            rc = self.pick_first_valid(records, lambda x: True, notional, "SHORT")
            if rc:
                return rc, "S_SL1.40_TP2.4_4.0_R0.3", "T5_CATCH_15S"
            return None, None, None

        if need_a:
            if rev_pick:
                self.enter_position("A", rev_pick, "L_SL1.80_TP3.0_4.8_R0.2", "T4_1015_REV", "10:15", 1.0)
            else:
                rc, br, tr = pick_catch_all(1.0)
                if rc:
                    self.enter_position("A", rc, br, tr, "10:15", 1.0)

        if need_b:
            if rev_pick:
                self.enter_position("B", rev_pick, "L_SL1.80_TP3.0_4.8_R0.2", "T4_1015_REV", "10:15", 1.0)
            else:
                rc, br, tr = pick_catch_all(0.5)
                if rc:
                    self.enter_position("B", rc, br, tr, "10:15", 0.5)

    def enter_position(self, slot_id, cand, bracket_name, tier_name, entry_time_str, scale=1.0):
        b = BRACKETS[bracket_name]
        direction = b["direction"]
        sym = cand["symbol"]
        raw_ep = cand["entry_open"]
        entry_vol = cand["entry_vol"]

        entry_px = round(raw_ep * (1.0 - SLIPPAGE if direction == "SHORT" else 1.0 + SLIPPAGE), 2)
        notional = self.slot_exposure * scale
        qty = int(np.floor(notional / entry_px))

        if qty < 2:
            logger.warning(f"[Slot {slot_id}] {sym} price Rs {entry_px:.2f} too high for Rs {notional:,.2f}.")
            return
        if entry_vol < MIN_VOL_MULT * qty:
            logger.warning(f"[Slot {slot_id}] {sym} vol {entry_vol:.0f} < 5x qty ({MIN_VOL_MULT*qty:.0f}). Skipping.")
            return

        if b["split"]:
            q1 = qty // 2
            q2 = qty - q1
        else:
            q1 = qty
            q2 = 0

        if direction == "SHORT":
            initial_sl = round(entry_px * (1.0 + b["sl"] / 100.0), 2)
            tp1 = round(entry_px * (1.0 - b["tp1"] / 100.0), 2)
            tp2 = round(entry_px * (1.0 - b["tp2"] / 100.0), 2)
        else:
            initial_sl = round(entry_px * (1.0 - b["sl"] / 100.0), 2)
            tp1 = round(entry_px * (1.0 + b["tp1"] / 100.0), 2)
            tp2 = round(entry_px * (1.0 + b["tp2"] / 100.0), 2)

        self.active_positions[slot_id] = {
            "slot": slot_id,
            "symbol": sym,
            "isin": cand["isin"],
            "sector": cand.get("sector", "N/A"),
            "direction": direction,
            "bracket": bracket_name,
            "split": b["split"],
            "ratchet_pct": b["ratchet"],
            "entry_time": entry_time_str,
            "raw_ep": raw_ep,
            "entry_px": entry_px,
            "qty": qty,
            "q1": q1,
            "q2": q2,
            "slot_exposure": qty * entry_px,
            "initial_sl": initial_sl,
            "current_sl": initial_sl,
            "tp1": tp1,
            "tp2": tp2,
            "pending_ratchet": False,
            "t1_hit": False,
            "l1_closed": False,
            "exit1_px": None,
            "exit1_time": None,
            "exit1_reason": None,
            "l2_closed": (q2 == 0),
            "exit2_px": None,
            "exit2_time": None,
            "exit2_reason": None,
            "current_px": entry_px,
            "pool_tag": tier_name,
            "strategy": f"F12_CAUSAL_5xVol [Slot {slot_id}: {tier_name}]",
            "mfe_pct": 0.0,
            "pnl_net_inr": 0.0,
        }
        self.slot_filled_today[slot_id] = True

        logger.info("=" * 80)
        logger.info(f"[SLOT {slot_id} EXECUTED] {direction} {sym} @ Rs {entry_px:.2f} ({tier_name} | {bracket_name})")
        logger.info(f"   Scale: {scale:.1f}x | Notional: Rs {qty*entry_px:,.2f} | Qty: {qty} (L1:{q1}, L2:{q2})")
        logger.info(f"   SL: Rs {initial_sl:.2f} | TP1: Rs {tp1:.2f} | TP2: Rs {tp2:.2f}")
        logger.info("=" * 80)

    def manage_slot_position(self, slot_id, bars_1m, current_time_str):
        pos = self.active_positions.get(slot_id)
        if not pos:
            return

        direction = pos["direction"]

        # Apply pending ratchet from previous closed bar
        if pos["pending_ratchet"]:
            if direction == "SHORT":
                ratchet_sl = round(pos["entry_px"] * (1.0 - pos["ratchet_pct"] / 100.0), 2)
                if ratchet_sl < pos["current_sl"]:
                    pos["current_sl"] = ratchet_sl
                    logger.info(f"[SLOT {slot_id} RATCHET] {pos['symbol']} SL -> Rs {ratchet_sl:.2f}")
            else:
                ratchet_sl = round(pos["entry_px"] * (1.0 + pos["ratchet_pct"] / 100.0), 2)
                if ratchet_sl > pos["current_sl"]:
                    pos["current_sl"] = ratchet_sl
                    logger.info(f"[SLOT {slot_id} RATCHET] {pos['symbol']} SL -> Rs {ratchet_sl:.2f}")
            pos["pending_ratchet"] = False

        if not bars_1m:
            return

        latest_bar = bars_1m[-1]
        o_i = float(latest_bar[1])
        h_i = float(latest_bar[2])
        l_i = float(latest_bar[3])
        c_i = float(latest_bar[4])

        pos["current_px"] = c_i
        if direction == "SHORT":
            unrealized_pct = ((pos["entry_px"] - c_i) / pos["entry_px"]) * 100.0
        else:
            unrealized_pct = ((c_i - pos["entry_px"]) / pos["entry_px"]) * 100.0
        pos["mfe_pct"] = max(pos["mfe_pct"], unrealized_pct)

        exit_sl_mult = (1.0 + SLIPPAGE) if direction == "SHORT" else (1.0 - SLIPPAGE)
        exit_tp_mult = (1.0 - SLIPPAGE) if direction == "SHORT" else (1.0 + SLIPPAGE)

        # 1. Gap-Through SL
        gap_sl = (o_i >= pos["current_sl"]) if direction == "SHORT" else (o_i <= pos["current_sl"])
        if gap_sl:
            fill = round(o_i * exit_sl_mult, 2)
            if not pos["l1_closed"]:
                pos["exit1_px"], pos["exit1_time"], pos["exit1_reason"], pos["l1_closed"] = fill, current_time_str, "GAP_THROUGH_SL", True
            if not pos["l2_closed"]:
                pos["exit2_px"], pos["exit2_time"], pos["exit2_reason"], pos["l2_closed"] = fill, current_time_str, "GAP_THROUGH_SL", True
            self.finalize_trade(slot_id)
            return

        # 2. Intrabar Hard SL
        hard_sl = (h_i >= pos["current_sl"]) if direction == "SHORT" else (l_i <= pos["current_sl"])
        if hard_sl:
            fill = round(pos["current_sl"] * exit_sl_mult, 2)
            if not pos["l1_closed"]:
                pos["exit1_px"], pos["exit1_time"], pos["exit1_reason"], pos["l1_closed"] = fill, current_time_str, "HARD_SL", True
            if not pos["l2_closed"]:
                pos["exit2_px"], pos["exit2_time"], pos["exit2_reason"], pos["l2_closed"] = fill, current_time_str, "HARD_SL", True
            self.finalize_trade(slot_id)
            return

        # 3. Target 1 (or full 1-leg TP)
        if not pos["t1_hit"]:
            gap_tp1 = (o_i <= pos["tp1"]) if direction == "SHORT" else (o_i >= pos["tp1"])
            hit_tp1 = (l_i <= pos["tp1"]) if direction == "SHORT" else (h_i >= pos["tp1"])
            if gap_tp1 or hit_tp1:
                px_base = o_i if gap_tp1 else pos["tp1"]
                pos["exit1_px"] = round(px_base * exit_tp_mult, 2)
                pos["exit1_time"] = current_time_str
                pos["exit1_reason"] = "GAP_THROUGH_TP1" if gap_tp1 else "TP1_HIT"
                pos["t1_hit"] = True
                pos["l1_closed"] = True
                if not pos["split"]:
                    self.finalize_trade(slot_id)
                    return
                if pos["ratchet_pct"] > 0:
                    pos["pending_ratchet"] = True
                logger.info(f"[SLOT {slot_id} TP1] {pos['symbol']} L1 closed @ Rs {pos['exit1_px']:.2f}")

        # 4. Target 2 (Leg 2 Runner)
        if pos["split"] and pos["t1_hit"] and not pos["l2_closed"]:
            gap_tp2 = (o_i <= pos["tp2"]) if direction == "SHORT" else (o_i >= pos["tp2"])
            hit_tp2 = (l_i <= pos["tp2"]) if direction == "SHORT" else (h_i >= pos["tp2"])
            if gap_tp2 or hit_tp2:
                px_base = o_i if gap_tp2 else pos["tp2"]
                pos["exit2_px"] = round(px_base * exit_tp_mult, 2)
                pos["exit2_time"] = current_time_str
                pos["exit2_reason"] = "GAP_THROUGH_TP2" if gap_tp2 else "TP2_HIT"
                pos["l2_closed"] = True
                self.finalize_trade(slot_id)
                return

    def eod_square_off(self, slot_id, bars_1m, current_time_str):
        pos = self.active_positions.get(slot_id)
        if not pos:
            return
        mult = (1.0 + SLIPPAGE) if pos["direction"] == "SHORT" else (1.0 - SLIPPAGE)
        raw_exit = float(bars_1m[-1][1]) if bars_1m else pos["current_px"]
        exit_px = round(raw_exit * mult, 2)

        if not pos["l1_closed"]:
            pos["exit1_px"], pos["exit1_time"], pos["exit1_reason"], pos["l1_closed"] = exit_px, current_time_str, "EOD_1520_OPEN", True
        if not pos["l2_closed"]:
            pos["exit2_px"], pos["exit2_time"], pos["exit2_reason"], pos["l2_closed"] = exit_px, current_time_str, "EOD_1520_OPEN", True

        logger.info(f"[SLOT {slot_id} 15:20 EOD] Closing {pos['symbol']} @ Rs {exit_px:.2f}")
        self.finalize_trade(slot_id)

    def finalize_trade(self, slot_id):
        pos = self.active_positions.get(slot_id)
        if not pos:
            return

        ep = pos["entry_px"]
        q1, q2 = pos["q1"], pos["q2"]
        ex1 = pos["exit1_px"]
        ex2 = pos["exit2_px"] if q2 > 0 else ex1

        if pos["direction"] == "SHORT":
            gross_inr = (ep - ex1) * q1 + (ep - ex2) * q2
        else:
            gross_inr = (ex1 - ep) * q1 + (ex2 - ep) * q2

        costs = calc_statutory_costs(ep, q1, q2, ex1, ex2, direction=pos["direction"])
        net_inr = gross_inr - costs

        self.today_pnl += net_inr
        self.cumulative_capital += net_inr

        record = {
            "slot": slot_id,
            "symbol": pos["symbol"],
            "direction": pos["direction"],
            "entry_time": pos["entry_time"],
            "entry_px": ep,
            "qty": pos["qty"],
            "exit1_time": pos["exit1_time"],
            "exit1_px": ex1,
            "exit1_reason": pos["exit1_reason"],
            "exit2_time": pos["exit2_time"],
            "exit2_px": ex2 if q2 > 0 else None,
            "exit2_reason": pos["exit2_reason"],
            "gross_inr": round(gross_inr, 2),
            "costs": round(costs, 2),
            "net_inr": round(net_inr, 2),
            "is_win": net_inr > 0,
            "pool_tag": pos["pool_tag"],
            "strategy": pos["strategy"],
        }
        self.closed_trades.append(record)

        save_trade_to_turso(
            pos, ex1, pos["exit1_reason"], pos["exit1_time"],
            ex2, pos["exit2_reason"], pos["exit2_time"],
            net_inr, gross_inr, costs, self.cumulative_capital,
        )

        logger.info(
            f"[SLOT {slot_id} COMPLETE] {pos['direction']} {pos['symbol']} | Net: Rs {net_inr:+,.2f} | Costs: Rs {costs:,.2f}"
        )
        self.active_positions[slot_id] = None

    def run_cycle(self):
        now = datetime.datetime.now(IST)
        time_str = now.strftime("%H:%M")
        today_str = now.strftime("%Y-%m-%d")
        self.last_update_ist = now.strftime("%Y-%m-%d %H:%M:%S IST")

        if time_str == "09:00":
            self.scan_0945_done = False
            self.scan_1015_done = False
            self.eod_done = False
            self.active_positions = {"A": None, "B": None}
            self.slot_filled_today = {"A": False, "B": False}
            self.closed_trades.clear()
            self.today_pnl = 0.0
            self._5d_vol_cache.clear()
            self.status = "PRE-MARKET WARMUP (Standing by for 09:45 AM Entry)"

        if time_str < "09:15" or time_str > "15:30":
            self.status = "STANDBY (Market Closed) - Next Scan: 09:45 AM IST"
            return

        open_cnt = sum(1 for v in self.active_positions.values() if v is not None)
        self.status = f"RUNNING LIVE ({time_str} IST) | Mode: F12_CAUSAL_5xVol (2 Slots) | Open: {open_cnt}/2"

        if "09:45" <= time_str < "09:50" and not self.scan_0945_done:
            self.scan_market_0945()
        elif time_str >= "09:50" and not self.scan_0945_done:
            self.scan_0945_done = True

        if "10:15" <= time_str < "10:20" and not self.scan_1015_done:
            self.scan_market_1015()
        elif time_str >= "10:20" and not self.scan_1015_done:
            self.scan_1015_done = True

        for slot_id in ("A", "B"):
            pos = self.active_positions.get(slot_id)
            if pos:
                bars_1m = self.fetch_live_1min_bars(pos["isin"])
                if time_str >= "15:20" and not self.eod_done:
                    self.eod_square_off(slot_id, bars_1m, time_str)
                else:
                    self.manage_slot_position(slot_id, bars_1m, time_str)

        if time_str >= "15:20" and not self.eod_done:
            self.eod_done = True
            wins = sum(1 for t in self.closed_trades if t["net_inr"] > 0)
            losses = sum(1 for t in self.closed_trades if t["net_inr"] <= 0)
            save_eod_balance_to_turso(today_str, self.capital_base, self.today_pnl, self.cumulative_capital, len(self.closed_trades), wins, losses)


Champion1UnionEngine = F12CausalEngine
engine = F12CausalEngine(capital_base=60000.0, dry_run=True, enable_1015_catch_all=True)


def background_trader_loop():
    logger.info("F12_CAUSAL_5xVol 2-Slot Background Live Trader Daemon started.")
    while True:
        try:
            engine.run_cycle()
        except Exception as e:
            logger.error(f"Trader loop error: {e}")
        time.sleep(30)


if os.environ.get("DISABLE_TRADER_THREAD") != "1":
    trader_thread = threading.Thread(target=background_trader_loop, daemon=True)
    trader_thread.start()

# ------------------------------------------------------------------------------
# 4. FASTAPI DASHBOARD (REAL-TIME CLOUD MONITORING)
# ------------------------------------------------------------------------------
app = FastAPI(title="F12_CAUSAL_5xVol 2-Slot Live Trader")


@app.get("/ping")
def ping():
    now_ist = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    return {
        "status": "online",
        "strategy": "F12_CAUSAL_5xVol (2 Trades/Day: W12 + FINAL_C1 @ 09:45 & 10:15 OPEN)",
        "time_ist": now_ist,
        "engine_status": engine.status,
        "daily_budget": f"Rs {engine.capital_base:,.2f}",
        "slot_notional_1x": f"Rs {engine.slot_exposure:,.2f}",
        "active_slots": {k: (v is not None) for k, v in engine.active_positions.items()},
        "today_pnl_inr": round(engine.today_pnl, 2),
        "closed_trades_today": len(engine.closed_trades),
    }


@app.get("/api/state")
def get_state():
    return {
        "strategy": "F12_CAUSAL_5xVol (2 Trades/Day: W12 + FINAL_C1)",
        "status": engine.status,
        "last_update": engine.last_update_ist,
        "capital_base": engine.capital_base,
        "slot_exposure": engine.slot_exposure,
        "enable_1015_catch_all": engine.enable_1015_catch_all,
        "today_pnl": engine.today_pnl,
        "cumulative_capital": engine.cumulative_capital,
        "active_positions": engine.active_positions,
        "closed_trades": engine.closed_trades,
        "scan_0945_done": engine.scan_0945_done,
        "scan_1015_done": engine.scan_1015_done,
        "logs": log_records[-50:],
    }


@app.get("/", response_class=HTMLResponse)
def dashboard():
    now_ist = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    pnl_color = "#10b981" if engine.today_pnl >= 0 else "#ef4444"

    active_html = ""
    for slot_id in ("A", "B"):
        p = engine.active_positions.get(slot_id)
        if p:
            mult = 1.0 if p["direction"] == "LONG" else -1.0
            curr_pnl = round(mult * (p["current_px"] - p["entry_px"]) * p["qty"], 2)
            c_pnl = "#10b981" if curr_pnl >= 0 else "#ef4444"
            dir_bg = "#059669" if p["direction"] == "LONG" else "#dc2626"
            active_html += f"""
            <tr style="border-bottom: 1px solid #334155;">
                <td style="padding: 10px; font-weight: bold;">Slot {slot_id}</td>
                <td style="padding: 10px; font-weight: bold; color: #38bdf8;">{p['symbol']}</td>
                <td style="padding: 10px;"><span style="background: {dir_bg}; padding: 3px 8px; border-radius: 4px; font-size: 11px;">{p['direction']}</span></td>
                <td style="padding: 10px;">Rs {p['entry_px']:.2f}</td>
                <td style="padding: 10px;">Rs {p['current_px']:.2f}</td>
                <td style="padding: 10px;">{p['qty']} (L1:{p['q1']}, L2:{p['q2']})</td>
                <td style="padding: 10px; color: #f87171;">Rs {p['current_sl']:.2f}</td>
                <td style="padding: 10px; color: #4ade80;">Rs {p['tp1']:.2f} / Rs {p['tp2']:.2f}</td>
                <td style="padding: 10px; color: {c_pnl}; font-weight: bold;">Rs {curr_pnl:+,.2f}</td>
                <td style="padding: 10px; font-size: 12px; color: #94a3b8;">{p['pool_tag']}</td>
            </tr>
            """
    if not active_html:
        active_html = "<tr><td colspan='10' style='text-align: center; padding: 25px; color: #64748b;'>No Active Trades. (Scans at 09:45 AM & 10:15 AM IST)</td></tr>"

    closed_html = ""
    for c in reversed(engine.closed_trades):
        c_pnl = "#10b981" if c["net_inr"] >= 0 else "#ef4444"
        ex_str = f"L1: Rs {c['exit1_px']:.2f}" + (f" | L2: Rs {c['exit2_px']:.2f}" if c["exit2_px"] else "")
        rs_str = f"L1: {c['exit1_reason']}" + (f" | L2: {c['exit2_reason']}" if c["exit2_reason"] else "")
        closed_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px;">Slot {c['slot']}</td>
            <td style="padding: 10px; font-weight: bold;">{c['symbol']} ({c['direction']})</td>
            <td style="padding: 10px;">{c['entry_time']}</td>
            <td style="padding: 10px;">Rs {c['entry_px']:.2f}</td>
            <td style="padding: 10px;">{ex_str}</td>
            <td style="padding: 10px; color: {c_pnl}; font-weight: bold;">Rs {c['net_inr']:+,.2f}</td>
            <td style="padding: 10px; font-size: 12px; color: #cbd5e1;">{rs_str}</td>
        </tr>
        """
    if not closed_html:
        closed_html = "<tr><td colspan='7' style='text-align: center; padding: 25px; color: #64748b;'>No trades closed today yet.</td></tr>"

    turso_history_html = ""
    try:
        df_turso_trades = query_turso_raw("SELECT trade_date, symbol, direction, entry_price, exit_price, pnl_net_inr, exit_reason, ending_capital FROM champion_cloud_paper_trades ORDER BY id DESC LIMIT 15")
        if not df_turso_trades.empty:
            for _, r in df_turso_trades.iterrows():
                col = "#10b981" if float(r.get("pnl_net_inr", 0)) >= 0 else "#ef4444"
                turso_history_html += f"""
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 8px;">{r.get('trade_date')}</td>
                    <td style="padding: 8px; font-weight: bold;">{r.get('symbol')} ({r.get('direction')})</td>
                    <td style="padding: 8px;">Rs {float(r.get('entry_price', 0)):.2f}</td>
                    <td style="padding: 8px;">Rs {float(r.get('exit_price', 0)):.2f}</td>
                    <td style="padding: 8px; color: {col}; font-weight: bold;">Rs {float(r.get('pnl_net_inr', 0)):+.2f}</td>
                    <td style="padding: 8px; font-size: 11px; color: #cbd5e1;">{r.get('exit_reason')}</td>
                    <td style="padding: 8px; font-weight: bold; color: #38bdf8;">Rs {float(r.get('ending_capital', 0)):,.2f}</td>
                </tr>
                """
    except Exception:
        pass
    if not turso_history_html:
        turso_history_html = "<tr><td colspan='7' style='text-align: center; padding: 20px; color: #64748b;'>Turso Cloud Database connected.</td></tr>"

    logs_str = "<br>".join(log_records[-35:])
    slots_filled = sum(1 for v in engine.slot_filled_today.values() if v)

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>F12_CAUSAL_5xVol 2-Slot Cloud Trader</title>
        <meta http-equiv="refresh" content="10">
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #0f172a; color: #f8fafc; margin: 0; padding: 20px; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .card {{ background: #1e293b; border-radius: 8px; padding: 20px; margin-bottom: 20px; border: 1px solid #334155; }}
            .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 15px; margin-bottom: 20px; }}
            .stat-box {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; text-align: center; }}
            .stat-val {{ font-size: 24px; font-weight: bold; margin-top: 5px; }}
            table {{ width: 100%; border-collapse: collapse; text-align: left; }}
            th {{ background: #0f172a; padding: 10px; color: #94a3b8; font-size: 12px; text-transform: uppercase; }}
            .console {{ background: #020617; border-radius: 6px; padding: 15px; font-family: monospace; font-size: 12px; line-height: 1.6; color: #38bdf8; max-height: 250px; overflow-y: auto; }}
            .btn {{ display: inline-block; padding: 8px 14px; background: #2563eb; color: #fff; text-decoration: none; border-radius: 6px; font-size: 12px; margin-right: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
                <div>
                    <h2 style="margin: 0;">F12_CAUSAL_5xVol 2-Slot Engine</h2>
                    <span style="color: #94a3b8; font-size: 13px;">2 Trades/Day (W12 + FINAL_C1) @ 09:45 & 10:15 OPEN | Fixed Rs 60,000/Day | Vol >= 5x Qty</span>
                </div>
                <div style="text-align: right;">
                    <span style="background: #10b981; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">LIVE PRODUCTION</span>
                    <div style="font-size: 11px; color: #64748b; margin-top: 5px;">Updated: {now_ist}</div>
                </div>
            </div>

            <div class="grid">
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">DAILY FIXED BUDGET</div>
                    <div class="stat-val" style="color: #38bdf8;">Rs {engine.capital_base:,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">2 Slots x Rs {engine.slot_exposure:,.0f} (5x MIS)</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">TODAY'S NET P&L</div>
                    <div class="stat-val" style="color: {pnl_color};">Rs {engine.today_pnl:+,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Full Statutory Drag Deducted</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">DAILY SLOT ALLOCATION</div>
                    <div class="stat-val">{slots_filled} / 2 SLOTS</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Slot A (W12) + Slot B (FINAL_C1)</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">EQUITY CURVE</div>
                    <div class="stat-val" style="color: #a78bfa;">Rs {engine.cumulative_capital:,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Permanent Turso Cloud Ledger</div>
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <a href="/ping" target="_blank" class="btn" style="background: #059669;">Engine Ping</a>
                <a href="/api/state" target="_blank" class="btn" style="background: #475569;">View State JSON</a>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #38bdf8;">Active Positions (Slots A & B)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Slot</th><th>Symbol</th><th>Dir</th><th>Entry</th><th>Current</th><th>Qty (L1/L2)</th><th>Stop Loss</th><th>Targets</th><th>Net P&L</th><th>Tier</th>
                        </tr>
                    </thead>
                    <tbody>{active_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #a78bfa;">Today's Completed Trades</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Slot</th><th>Symbol</th><th>Entry</th><th>Entry Px</th><th>Exit Prices</th><th>Net P&L</th><th>Exit Mechanics</th>
                        </tr>
                    </thead>
                    <tbody>{closed_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #10b981;">Multi-Day Trade History (Turso DB)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>Net P&L</th><th>Exit Reason</th><th>Ending Capital</th>
                        </tr>
                    </thead>
                    <tbody>{turso_history_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #94a3b8;">Engine Live Console Logs</h3>
                <div class="console">{logs_str}</div>
            </div>
        </div>
    </body>
    </html>
    """
    return html


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
