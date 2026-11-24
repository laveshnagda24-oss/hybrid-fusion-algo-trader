"""
main.py
================================================================================
CLOUD PRODUCTION LIVE & PAPER TRADING SERVER (RENDER.COM + TURSO PERSISTENCE)
STRATEGY: CHAMPION #1 SPLIT-SLOT UNION (1 TRADE / DAY @ 09:45 OPEN)
Fixed ₹30,000 Base Capital | 5x MIS Leverage (₹1,50,000 Notional) | Zero Look-Ahead Bias
Leg 1: 50% Qty @ -3.20% TP1 / +2.00% SL | Leg 2: 50% Qty @ -5.20% TP2 / Next-Bar Ratchet +0.20% / 15:20 EOD
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
logger = logging.getLogger("RenderChampion1")

IST = pytz.timezone("Asia/Kolkata")
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

TURSO_URL = os.environ.get("TURSO_DATABASE_URL", "https://market-data-lavesh.aws-ap-south-1.turso.io")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN", "eyJhbGciOiJFZERTQSIsInR5cCI6IkpXVCJ9.eyJhIjoicnciLCJpYXQiOjE3OTMyODUxNjIsImlkIjoiMDFhMTg4MWYtY2EwMS03MzI5LTljNzUtOTU5NGYyZTdmYTVkIiwia2lkIjoiYUVPVmtxaENSbEVZb21JMTFSOHlZZUliWE5UTXJ1OXhpNEx2ZkJBOEI5NCIsInJpZCI6ImZkZWZhNzEzLWQxZmUtNDJjNy04NTVlLTBhOGUwY2I0NDY0NCJ9.KMq5bJSwJfGh8X-cTTeL1hWaVsTb1Sz9hAlBTorfIKS0rJBkCGA9gnN47A-yuxtsr_k84oSXOj45j6d9Ht8DDA")

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Accept': 'application/json'
}

log_records = []
class MemoryLogHandler(logging.Handler):
    def emit(self, record):
        msg = self.format(record)
        log_records.append(msg)
        if len(log_records) > 200:
            log_records.pop(0)

mem_handler = MemoryLogHandler()
mem_handler.setFormatter(logging.Formatter(LOG_FORMAT))
logger.addHandler(mem_handler)

# ------------------------------------------------------------------------------
# 1. 2026 STATUTORY COST & FRICTION CALCULATOR (NSE CASH INTRADAY)
# ------------------------------------------------------------------------------
def calc_statutory_costs(entry_px, q1, q2, exit1_px, exit2_px):
    v_ent = entry_px * (q1 + q2)
    v_ex1 = exit1_px * q1
    v_ex2 = exit2_px * q2
    sell_v = v_ent
    buy_v = v_ex1 + v_ex2
    
    brk = min(20.0, 0.0003 * v_ent) + min(20.0, 0.0003 * v_ex1) + min(20.0, 0.0003 * v_ex2)
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
        "Content-Type": "application/json"
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
            res = resp.json().get('results', [{}])[0].get('response', {}).get('result', {})
            cols = [c['name'] for c in res.get('cols', [])]
            rows = []
            for r in res.get('rows', []):
                rows.append([c.get('value') for c in r])
            return pd.DataFrame(rows, columns=cols)
    except Exception as e:
        logger.error(f"Turso query error: {e}")
    return pd.DataFrame()

def save_trade_to_turso(pos, exit1_px, exit1_reason, exit1_time, exit2_px, exit2_reason, exit2_time, net_inr, gross_inr, costs, ending_capital):
    today_str = datetime.datetime.now(IST).strftime("%Y-%m-%d")
    check_sql = "SELECT id FROM champion_cloud_paper_trades WHERE trade_date = ? AND symbol = ? AND entry_time = ? LIMIT 1"
    df_exists = query_turso_raw(check_sql, [today_str, pos['symbol'], pos['entry_time']])
    if not df_exists.empty:
        logger.info(f"Trade {pos['symbol']} already recorded in Turso for {today_str}. Skipping duplicate.")
        return

    sql = """
    INSERT INTO champion_cloud_paper_trades 
    (trade_date, symbol, direction, sector, entry_time, exit_time, entry_price, exit_price, shares, exposure_inr, pnl_net_inr, pnl_net_pct, exit_reason, strategy, ending_capital)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    full_reason = f"L1: {exit1_reason} @ {exit1_time} | L2: {exit2_reason} @ {exit2_time}"
    avg_exit = (exit1_px * pos['q1'] + exit2_px * pos['q2']) / pos['qty']
    net_pct = (net_inr / pos['slot_exposure']) * 100.0 if pos['slot_exposure'] > 0 else 0.0

    params = [
        today_str, pos['symbol'], pos['direction'], pos.get('sector', 'N/A'),
        pos['entry_time'], exit2_time, pos['entry_px'], round(avg_exit, 2),
        pos['qty'], pos['slot_exposure'], round(net_inr, 2), round(net_pct, 2),
        full_reason, pos['strategy'], round(ending_capital, 2)
    ]
    query_turso_raw(sql, params)
    logger.info(f"Permanently saved Champion #1 trade {pos['symbol']} to Turso Cloud DB (Net PnL: Rs {net_inr:+.2f}).")

def save_eod_balance_to_turso(today_str, starting_cap, day_pnl, ending_cap, trades_count, wins, losses):
    sql = """
    INSERT OR REPLACE INTO champion_daily_balance
    (trade_date, starting_capital, day_pnl, ending_capital, trades_count, win_count, loss_count, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    params = [today_str, round(starting_cap, 2), round(day_pnl, 2), round(ending_cap, 2), trades_count, wins, losses]
    query_turso_raw(sql, params)
    logger.info(f"Permanently saved EOD Balance {today_str} to Turso: Ending Capital = Rs {ending_cap:,.2f}")

# ------------------------------------------------------------------------------
# 3. CORE CHAMPION #1 ENGINE (1 TRADE / DAY @ 09:45 OPEN)
# ------------------------------------------------------------------------------
class Champion1UnionEngine:
    def __init__(self, capital_base=30000.0, dry_run=True):
        self.capital_base = capital_base  # Strictly ₹30,000 base capital per day
        self.leverage = 5.0               # 5x MIS Intraday Leverage
        self.slot_exposure = self.capital_base * self.leverage  # ₹150,000 Notional
        self.dry_run = dry_run
        
        self.max_trades_today = 1
        self.active_position = None
        self.closed_trades = []
        self.candidates = []
        self.today_pnl = 0.0
        self.today_start_capital = self.capital_base
        self.cumulative_capital = self.capital_base
        
        self.status = "INITIALIZING"
        self.last_update_ist = ""
        self.scan_0945_done = False
        self.eod_done = False
        self.screener_done = False
        
        self.init_capital_from_db()
        self.load_watchlist()

    def init_capital_from_db(self):
        df_bal = query_turso_raw("SELECT ending_capital FROM champion_daily_balance ORDER BY trade_date DESC LIMIT 1")
        if not df_bal.empty and df_bal.iloc[0]['ending_capital'] is not None:
            self.cumulative_capital = float(df_bal.iloc[0]['ending_capital'])
            logger.info(f"Loaded compounded equity curve from Turso DB: Rs {self.cumulative_capital:,.2f}")
        else:
            self.cumulative_capital = self.capital_base
            logger.info(f"Initialized fresh starting equity: Rs {self.cumulative_capital:,.2f}")

    def load_watchlist(self):
        n500_path = os.path.join(DATA_DIR, 'ind_nifty500list.csv')
        if os.path.exists(n500_path):
            try:
                df = pd.read_csv(n500_path)
                valid = df.dropna(subset=['Symbol', 'ISIN Code']).to_dict('records')
                self.candidates = valid
                logger.info(f"Loaded {len(valid)} stocks from Nifty 500 universe for 09:45 monitoring.")
            except Exception as e:
                logger.error(f"Error loading universe: {e}")

    def fetch_live_1min_bars(self, isin):
        url = f"https://api.upstox.com/v2/historical-candle/intraday/NSE_EQ%7C{isin}/1minute"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                candles = data.get('data', {}).get('candles', [])
                if candles:
                    return sorted(candles, key=lambda x: x[0])
        except Exception:
            pass
        return []

    def fetch_daily_bars(self, isin, today_str):
        url = f"https://api.upstox.com/v2/historical-candle/NSE_EQ%7C{isin}/day/{today_str}/2026-09-01"
        req = urllib.request.Request(url, headers=HEADERS)
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                candles = sorted(data.get('data', {}).get('candles', []), key=lambda x: x[0])
                return candles
        except Exception:
            pass
        return []

    def compute_zero_lookahead_signals_0945(self, bars_1m, daily_bars):
        """
        Computes exact indicators strictly from closed bars 09:15 to 09:44 (first 30 1m bars):
          - open_gap: (Open_0915 - PrevClose) / PrevClose * 100
          - atr10: mean(TR over prior 10 completed days) / PrevClose * 100
          - rsi14: 14-bar RSI over 09:30-09:44 closes
          - dist_vwap: (C_29 - VWAP_0915_0944) / VWAP * 100
          - turnover_log: ln(1 + sum(C_t * V_t))
        """
        if len(bars_1m) < 30 or len(daily_bars) < 11:
            return None

        # Daily historical metrics
        prev_close = float(daily_bars[-2][4])  # Yesterday's close
        if prev_close < 50.0:
            return None

        # Corporate action & extreme gap guard
        c2c_moves = [abs(float(daily_bars[i][4]) / float(daily_bars[i-1][4]) - 1.0) * 100.0 for i in range(len(daily_bars)-10, len(daily_bars))]
        if any(m > 25.0 for m in c2c_moves):
            return None

        trs = []
        for i in range(len(daily_bars)-11, len(daily_bars)-1):
            h_d = float(daily_bars[i][2])
            l_d = float(daily_bars[i][3])
            pc_d = float(daily_bars[i-1][4])
            trs.append(max(h_d - l_d, abs(h_d - pc_d), abs(l_d - pc_d)))
        atr10 = (np.mean(trs) / prev_close) * 100.0

        # Intraday 30-minute window (09:15 to 09:44)
        bars_30 = bars_1m[:30]
        open_0915 = float(bars_30[0][1])
        open_gap = ((open_0915 - prev_close) / prev_close) * 100.0

        if abs(open_gap) > 12.0:
            return None

        closes = [float(b[4]) for b in bars_30]
        highs = [float(b[2]) for b in bars_30]
        lows = [float(b[3]) for b in bars_30]
        vols = [float(b[5]) for b in bars_30]

        # VWAP
        cum_vol = sum(vols)
        if cum_vol <= 0:
            return None
        cum_val = sum(((highs[t] + lows[t] + closes[t]) / 3.0) * vols[t] for t in range(30))
        vwap = cum_val / cum_vol
        dist_vwap = ((closes[-1] - vwap) / vwap) * 100.0

        # RSI 14
        diffs = [closes[t] - closes[t-1] for t in range(15, 30)]  # last 14 differences
        up = np.mean([max(d, 0.0) for d in diffs])
        dn = max(np.mean([max(-d, 0.0) for d in diffs]), 1e-9)
        rsi14 = 100.0 - (100.0 / (1.0 + up / dn))

        # Turnover Log
        total_turnover = sum(closes[t] * vols[t] for t in range(30))
        turnover_log = np.log1p(max(0.0, total_turnover))

        return {
            "prev_close": prev_close,
            "open_gap": open_gap,
            "atr10": atr10,
            "rsi14": rsi14,
            "dist_vwap": dist_vwap,
            "turnover_log": turnover_log,
            "total_turnover": total_turnover,
            "close_0944": closes[-1],
            "vwap": vwap
        }

    def scan_market_0945(self):
        """
        Executes strict 09:45 AM scan across universe.
        Qualifies Pool A and Pool B, ranks by turnover_log, selects Rank #1 ONLY.
        """
        if self.scan_0945_done or self.active_position is not None or len(self.closed_trades) >= self.max_trades_today:
            return

        logger.info("=" * 80)
        logger.info("09:45 AM IST: EXECUTING CHAMPION #1 SPLIT-SLOT UNION SCREENER...")
        logger.info("=" * 80)

        now = datetime.datetime.now(IST)
        today_str = now.strftime("%Y-%m-%d")

        qualifying = []
        
        # Parallel fetch & evaluate
        def evaluate_candidate(cand):
            sym = cand.get('Symbol')
            isin = cand.get('ISIN Code')
            if not isin: return None
            bars_1m = self.fetch_live_1min_bars(isin)
            if len(bars_1m) < 31: return None  # Must have 30 closed bars + 09:45 open candle
            daily_bars = self.fetch_daily_bars(isin, today_str)
            if not daily_bars or len(daily_bars) < 11: return None

            sig = self.compute_zero_lookahead_signals_0945(bars_1m, daily_bars)
            if not sig: return None

            # ATR filter
            if sig['atr10'] < 3.30: return None

            # Pool A: open_gap >= 0.60 & rsi14 <= 48.0
            in_pool_a = (sig['open_gap'] >= 0.60) and (sig['rsi14'] <= 48.0)
            # Pool B: open_gap >= 0.80 & dist_vwap <= -0.25
            in_pool_b = (sig['open_gap'] >= 0.80) and (sig['dist_vwap'] <= -0.25)

            if in_pool_a or in_pool_b:
                bar_0945 = bars_1m[30]
                entry_open = float(bar_0945[1])
                entry_vol = float(bar_0945[5])
                pool_tag = "Pool A (Gap+RSI)" if (in_pool_a and not in_pool_b) else ("Pool B (Gap+VWAP)" if (in_pool_b and not in_pool_a) else "Both Pools (A+B Conviction)")
                return {
                    "symbol": sym,
                    "isin": isin,
                    "entry_open": entry_open,
                    "entry_vol": entry_vol,
                    "signals": sig,
                    "pool_tag": pool_tag,
                    "turnover_log": sig['turnover_log'],
                    "turnover_cr": sig['total_turnover'] / 1e7
                }
            return None

        with ThreadPoolExecutor(max_workers=12) as executor:
            futs = [executor.submit(evaluate_candidate, c) for c in self.candidates]
            for f in as_completed(futs):
                res = f.result()
                if res:
                    qualifying.append(res)

        self.scan_0945_done = True

        if not qualifying:
            logger.info("09:45 Scan Complete: 0 stocks met strict Champion #1 criteria. Capital stays 100% in cash today.")
            return

        # Sort descending by turnover_log (Rank #1 Selection)
        qualifying.sort(key=lambda x: x['turnover_log'], reverse=True)
        logger.info(f"Found {len(qualifying)} qualifying candidates:")
        for idx, q in enumerate(qualifying[:5], 1):
            logger.info(f"  Rank #{idx}: {q['symbol']:<10} | Turnover: ₹{q['turnover_cr']:.2f} Cr | Gap: {q['signals']['open_gap']:+.2f}% | ATR10: {q['signals']['atr10']:.2f}% | {q['pool_tag']}")

        # Pick Rank #1 ONLY
        top1 = qualifying[0]
        self.enter_position(top1)

    def enter_position(self, cand):
        sym = cand['symbol']
        raw_ep = cand['entry_open']
        entry_vol = cand['entry_vol']

        slip = 0.0005  # Conservative 0.05% adverse slippage
        entry_px = round(raw_ep * (1.0 - slip), 2)
        qty = int(np.floor(self.slot_exposure / entry_px))

        if qty < 2:
            logger.warning(f"Stock {sym} price ₹{entry_px:.2f} too high for ₹{self.slot_exposure:,.2f} allocation. Cannot take trade.")
            return
        if entry_vol < qty:
            logger.warning(f"Stock {sym} 09:45 candle volume ({entry_vol}) < required qty ({qty}). Skipping to avoid liquidity trap.")
            return

        q1 = qty // 2
        q2 = qty - q1

        sl_pct = 2.00
        tp1_pct = 3.20
        tp2_pct = 5.20

        initial_sl = round(entry_px * (1.0 + sl_pct / 100.0), 2)
        tp1 = round(entry_px * (1.0 - tp1_pct / 100.0), 2)
        tp2 = round(entry_px * (1.0 - tp2_pct / 100.0), 2)

        self.active_position = {
            "symbol": sym,
            "isin": cand['isin'],
            "direction": "SHORT",
            "entry_time": "09:45",
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
            "l2_closed": False,
            "exit2_px": None,
            "exit2_time": None,
            "exit2_reason": None,
            "current_px": entry_px,
            "pool_tag": cand['pool_tag'],
            "strategy": f"Champion #1 Split-Slot Union [{cand['pool_tag']}]",
            "mfe_pct": 0.0,
            "pnl_net_inr": 0.0
        }

        logger.info("=" * 80)
        logger.info(f"🎯 [ORDER EXECUTED] SHORT {sym} @ ₹{entry_px:.2f} (Raw: ₹{raw_ep:.2f})")
        logger.info(f"   Allocation: ₹{qty*entry_px:,.2f} | Total Qty: {qty} (Leg 1: {q1}, Leg 2: {q2})")
        logger.info(f"   Initial SL (+2.0%): ₹{initial_sl:.2f} | TP1 (-3.2%): ₹{tp1:.2f} | TP2 (-5.2%): ₹{tp2:.2f}")
        logger.info("=" * 80)

    def manage_active_position(self, bars_1m, current_time_str):
        if not self.active_position:
            return

        pos = self.active_position
        slip = 0.0005

        # Apply pending ratchet on the next minute candle
        if pos['pending_ratchet']:
            ratchet_sl = round(pos['entry_px'] * (1.0 - 0.0020), 2)
            if ratchet_sl < pos['current_sl']:
                pos['current_sl'] = ratchet_sl
                logger.info(f"🔒 [RATCHET TRIGGERED] Leg 2 Stop-Loss tightened to +0.20% locked profit @ ₹{ratchet_sl:.2f}")
            pos['pending_ratchet'] = False

        if not bars_1m or len(bars_1m) == 0:
            return

        latest_bar = bars_1m[-1]
        o_i = float(latest_bar[1])
        h_i = float(latest_bar[2])
        l_i = float(latest_bar[3])
        c_i = float(latest_bar[4])

        pos['current_px'] = c_i
        unrealized_pct = ((pos['entry_px'] - c_i) / pos['entry_px']) * 100.0
        if unrealized_pct > pos['mfe_pct']:
            pos['mfe_pct'] = unrealized_pct

        # -------------------------------------------------------------
        # Intrabar Order of Checks (Strict Human / Exchange Order):
        # 1. Gap-Through SL (open >= SL)
        # 2. Hard SL (high >= SL)
        # 3. Target 1 (low <= TP1)
        # 4. Target 2 (low <= TP2)
        # -------------------------------------------------------------

        # 1. Gap-Through SL
        if o_i >= pos['current_sl']:
            fill = round(o_i * (1.0 + slip), 2)
            if not pos['l1_closed']:
                pos['exit1_px'] = fill
                pos['exit1_time'] = current_time_str
                pos['exit1_reason'] = "GAP_THROUGH_SL"
                pos['l1_closed'] = True
            pos['exit2_px'] = fill
            pos['exit2_time'] = current_time_str
            pos['exit2_reason'] = "GAP_THROUGH_SL"
            pos['l2_closed'] = True
            self.finalize_trade()
            return

        # 2. Intrabar Hard SL
        if h_i >= pos['current_sl']:
            fill = round(pos['current_sl'] * (1.0 + slip), 2)
            if not pos['l1_closed']:
                pos['exit1_px'] = fill
                pos['exit1_time'] = current_time_str
                pos['exit1_reason'] = "HARD_SL"
                pos['l1_closed'] = True
            pos['exit2_px'] = fill
            pos['exit2_time'] = current_time_str
            pos['exit2_reason'] = "HARD_SL"
            pos['l2_closed'] = True
            self.finalize_trade()
            return

        # 3. Target 1 (50% Qty @ -3.20%)
        if not pos['t1_hit']:
            if o_i <= pos['tp1']:
                pos['exit1_px'] = round(o_i * (1.0 - slip), 2)
                pos['exit1_time'] = current_time_str
                pos['exit1_reason'] = "GAP_THROUGH_TP1"
                pos['t1_hit'] = True
                pos['l1_closed'] = True
                pos['pending_ratchet'] = True
                logger.info(f"🎉 [TARGET 1 HIT] 50% Qty closed @ ₹{pos['exit1_px']:.2f} (Gap TP1)!")
            elif l_i <= pos['tp1']:
                pos['exit1_px'] = round(pos['tp1'] * (1.0 - slip), 2)
                pos['exit1_time'] = current_time_str
                pos['exit1_reason'] = "TP1_HIT (-3.2%)"
                pos['t1_hit'] = True
                pos['l1_closed'] = True
                pos['pending_ratchet'] = True
                logger.info(f"🎉 [TARGET 1 HIT] 50% Qty closed @ ₹{pos['exit1_px']:.2f} (-3.20%)! Leg 2 ratchet armed.")

        # 4. Target 2 (50% Qty @ -5.20%)
        if pos['t1_hit'] and not pos['l2_closed']:
            if o_i <= pos['tp2']:
                pos['exit2_px'] = round(o_i * (1.0 - slip), 2)
                pos['exit2_time'] = current_time_str
                pos['exit2_reason'] = "GAP_THROUGH_TP2"
                pos['l2_closed'] = True
                logger.info(f"🏆 [TARGET 2 HIT - FULL RUNNER] 50% Qty closed @ ₹{pos['exit2_px']:.2f} (Gap TP2)!")
                self.finalize_trade()
                return
            elif l_i <= pos['tp2']:
                pos['exit2_px'] = round(pos['tp2'] * (1.0 - slip), 2)
                pos['exit2_time'] = current_time_str
                pos['exit2_reason'] = "TP2_HIT (-5.2%)"
                pos['l2_closed'] = True
                logger.info(f"🏆 [TARGET 2 HIT - FULL RUNNER] 50% Qty closed @ ₹{pos['exit2_px']:.2f} (-5.20%)!")
                self.finalize_trade()
                return

    def eod_square_off(self, bars_1m, current_time_str):
        if not self.active_position:
            return
        pos = self.active_position
        slip = 0.0005
        exit_px = pos['current_px'] * (1.0 + slip)
        if bars_1m and len(bars_1m) > 0:
            exit_px = float(bars_1m[-1][1]) * (1.0 + slip)  # 15:20 Open

        exit_px = round(exit_px, 2)

        if not pos['l1_closed']:
            pos['exit1_px'] = exit_px
            pos['exit1_time'] = current_time_str
            pos['exit1_reason'] = "EOD_1520_OPEN"
            pos['l1_closed'] = True
        if not pos['l2_closed']:
            pos['exit2_px'] = exit_px
            pos['exit2_time'] = current_time_str
            pos['exit2_reason'] = "EOD_1520_OPEN"
            pos['l2_closed'] = True

        logger.info(f"⏰ [15:20 EOD MANDATORY EXIT] Closing position {pos['symbol']} @ ₹{exit_px:.2f}")
        self.finalize_trade()

    def finalize_trade(self):
        pos = self.active_position
        if not pos: return

        ex1 = pos['exit1_px']
        ex2 = pos['exit2_px']
        q1 = pos['q1']
        q2 = pos['q2']
        ep = pos['entry_px']

        gross_inr = (ep - ex1) * q1 + (ep - ex2) * q2
        costs = calc_statutory_costs(ep, q1, q2, ex1, ex2)
        net_inr = gross_inr - costs

        self.today_pnl += net_inr
        self.cumulative_capital += net_inr

        record = {
            "symbol": pos['symbol'],
            "entry_time": pos['entry_time'],
            "entry_px": ep,
            "qty": pos['qty'],
            "exit1_time": pos['exit1_time'],
            "exit1_px": ex1,
            "exit1_reason": pos['exit1_reason'],
            "exit2_time": pos['exit2_time'],
            "exit2_px": ex2,
            "exit2_reason": pos['exit2_reason'],
            "gross_inr": round(gross_inr, 2),
            "costs": round(costs, 2),
            "net_inr": round(net_inr, 2),
            "is_win": net_inr > 0,
            "pool_tag": pos['pool_tag'],
            "strategy": pos['strategy']
        }
        self.closed_trades.append(record)

        save_trade_to_turso(pos, ex1, pos['exit1_reason'], pos['exit1_time'],
                            ex2, pos['exit2_reason'], pos['exit2_time'],
                            net_inr, gross_inr, costs, self.cumulative_capital)

        logger.info("=" * 80)
        logger.info(f"🏁 [TRADE COMPLETE] {pos['symbol']} | Net PnL: Rs {net_inr:+,.2f} | Costs: Rs {costs:,.2f} | Status: {'WIN' if net_inr > 0 else 'LOSS'}")
        logger.info(f"   L1: {pos['exit1_time']} @ Rs {ex1:.2f} ({pos['exit1_reason']}) | L2: {pos['exit2_time']} @ Rs {ex2:.2f} ({pos['exit2_reason']})")
        logger.info("=" * 80)

        self.active_position = None

    def run_cycle(self):
        now = datetime.datetime.now(IST)
        time_str = now.strftime("%H:%M")
        today_str = now.strftime("%Y-%m-%d")
        self.last_update_ist = now.strftime("%Y-%m-%d %H:%M:%S IST")

        # Morning Reset at 09:00 IST
        if time_str == "09:00":
            self.scan_0945_done = False
            self.eod_done = False
            self.active_position = None
            self.closed_trades.clear()
            self.today_pnl = 0.0
            self.status = "PRE-MARKET WARMUP (Standing by for 09:45 AM Entry)"

        # Outside Market Hours
        if time_str < "09:15" or time_str > "15:30":
            self.status = f"STANDBY (Market Closed) - Next Scan: 09:45 AM IST"
            return

        self.status = f"RUNNING LIVE ({time_str} IST) | Mode: 1 Trade/Day | Slot: {'1/1 OPEN' if self.active_position else 'WAITING/CASH'}"

        # 09:45 AM Screener Execution
        if "09:45" <= time_str < "09:50" and not self.scan_0945_done:
            self.scan_market_0945()
        elif time_str >= "09:50" and not self.scan_0945_done:
            self.scan_0945_done = True

        # Position Management
        if self.active_position:
            isin = self.active_position['isin']
            bars_1m = self.fetch_live_1min_bars(isin)
            if time_str >= "15:20" and not self.eod_done:
                self.eod_square_off(bars_1m, time_str)
                wins = sum(1 for t in self.closed_trades if t['net_inr'] > 0)
                losses = sum(1 for t in self.closed_trades if t['net_inr'] <= 0)
                save_eod_balance_to_turso(today_str, self.capital_base, self.today_pnl, self.cumulative_capital, len(self.closed_trades), wins, losses)
                self.eod_done = True
            else:
                self.manage_active_position(bars_1m, time_str)

engine = Champion1UnionEngine(capital_base=30000.0, dry_run=True)

def background_trader_loop():
    logger.info("Champion #1 Background Live Trader Daemon started on Render Cloud.")
    while True:
        try:
            engine.run_cycle()
        except Exception as e:
            logger.error(f"Trader loop error: {e}")
        time.sleep(30)

trader_thread = threading.Thread(target=background_trader_loop, daemon=True)
trader_thread.start()

# ------------------------------------------------------------------------------
# 4. FASTAPI DASHBOARD (REAL-TIME CLOUD MONITORING)
# ------------------------------------------------------------------------------
app = FastAPI(title="Champion #1 Split-Slot Union Live Trader")

@app.get("/ping")
def ping():
    now_ist = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    return {
        "status": "online",
        "strategy": "Champion #1 Split-Slot Union (1 Trade/Day @ 09:45)",
        "time_ist": now_ist,
        "engine_status": engine.status,
        "daily_budget": f"Rs {engine.capital_base:,.2f}",
        "notional_power": f"Rs {engine.slot_exposure:,.2f}",
        "has_active_position": engine.active_position is not None,
        "today_pnl_inr": round(engine.today_pnl, 2),
        "closed_trades_today": len(engine.closed_trades)
    }

@app.get("/api/state")
def get_state():
    return {
        "strategy": "Champion #1 Split-Slot Union (1 Trade/Day @ 09:45)",
        "status": engine.status,
        "last_update": engine.last_update_ist,
        "capital_base": engine.capital_base,
        "notional_buying_power": engine.slot_exposure,
        "today_pnl": engine.today_pnl,
        "cumulative_capital": engine.cumulative_capital,
        "active_position": engine.active_position,
        "closed_trades": engine.closed_trades,
        "scan_0945_done": engine.scan_0945_done,
        "logs": log_records[-50:]
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    now_ist = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    pnl_color = "#10b981" if engine.today_pnl >= 0 else "#ef4444"

    active_html = ""
    if engine.active_position:
        p = engine.active_position
        curr_pnl = round(((p['entry_px'] - p['current_px']) * p['qty']), 2)
        c_pnl = "#10b981" if curr_pnl >= 0 else "#ef4444"
        active_html = f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: bold; color: #38bdf8;">{p['symbol']}</td>
            <td style="padding: 10px;"><span style="background: #dc2626; padding: 3px 8px; border-radius: 4px; font-size: 11px;">SHORT</span></td>
            <td style="padding: 10px;">Rs {p['entry_px']:.2f}</td>
            <td style="padding: 10px;">Rs {p['current_px']:.2f}</td>
            <td style="padding: 10px;">{p['qty']} (L1:{p['q1']}, L2:{p['q2']})</td>
            <td style="padding: 10px; color: #f87171;">Rs {p['current_sl']:.2f}</td>
            <td style="padding: 10px; color: #4ade80;">Rs {p['tp1']:.2f} / Rs {p['tp2']:.2f}</td>
            <td style="padding: 10px; color: {c_pnl}; font-weight: bold;">Rs {curr_pnl:+,.2f}</td>
            <td style="padding: 10px; font-size: 12px; color: #94a3b8;">{p['pool_tag']}</td>
        </tr>
        """
    else:
        active_html = "<tr><td colspan='9' style='text-align: center; padding: 25px; color: #64748b;'>No Active Trade. (1 Trade/Day Limit strictly enforced; enters at 09:45 AM IST if criteria met)</td></tr>"

    closed_html = ""
    for c in reversed(engine.closed_trades):
        c_pnl = "#10b981" if c['net_inr'] >= 0 else "#ef4444"
        closed_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: bold;">{c['symbol']}</td>
            <td style="padding: 10px;">09:45</td>
            <td style="padding: 10px;">Rs {c['entry_px']:.2f}</td>
            <td style="padding: 10px;">L1: Rs {c['exit1_px']:.2f} | L2: Rs {c['exit2_px']:.2f}</td>
            <td style="padding: 10px; color: {c_pnl}; font-weight: bold;">Rs {c['net_inr']:+,.2f}</td>
            <td style="padding: 10px; font-size: 12px; color: #cbd5e1;">L1: {c['exit1_reason']} | L2: {c['exit2_reason']}</td>
        </tr>
        """
    if not closed_html:
        closed_html = "<tr><td colspan='6' style='text-align: center; padding: 25px; color: #64748b;'>No trades closed today yet.</td></tr>"

    turso_history_html = ""
    try:
        df_turso_trades = query_turso_raw("SELECT trade_date, symbol, direction, entry_price, exit_price, pnl_net_inr, pnl_net_pct, exit_reason, ending_capital FROM champion_cloud_paper_trades ORDER BY id DESC LIMIT 15")
        if not df_turso_trades.empty:
            for _, r in df_turso_trades.iterrows():
                col = "#10b981" if float(r.get('pnl_net_inr', 0)) >= 0 else "#ef4444"
                turso_history_html += f"""
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 8px;">{r.get('trade_date')}</td>
                    <td style="padding: 8px; font-weight: bold;">{r.get('symbol')}</td>
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
        turso_history_html = "<tr><td colspan='7' style='text-align: center; padding: 20px; color: #64748b;'>Turso Cloud Database connected. Trades permanently accumulate here.</td></tr>"

    logs_str = "<br>".join(log_records[-35:])

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Champion #1 Split-Slot Union Cloud Trader</title>
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
                    <h2 style="margin: 0;">👑 Champion #1 Split-Slot Union Engine</h2>
                    <span style="color: #94a3b8; font-size: 13px;">1 Trade / Day @ 09:45 OPEN | Rank #1 Turnover Selection | Fixed ₹30,000 Capital (Render + Turso)</span>
                </div>
                <div style="text-align: right;">
                    <span style="background: #10b981; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">LIVE PRODUCTION</span>
                    <div style="font-size: 11px; color: #64748b; margin-top: 5px;">Updated: {now_ist} (Auto-refresh 10s)</div>
                </div>
            </div>

            <div class="grid">
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">DAILY BASE BUDGET</div>
                    <div class="stat-val" style="color: #38bdf8;">Rs {engine.capital_base:,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">5x Leverage = Rs {engine.slot_exposure:,.2f} Power</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">TODAY'S NET P&L</div>
                    <div class="stat-val" style="color: {pnl_color};">Rs {engine.today_pnl:+,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Full Statutory Drag Deducted</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">DAILY TRADE ALLOCATION</div>
                    <div class="stat-val">{'1 / 1 EXECUTED' if (engine.active_position or len(engine.closed_trades)>0) else '0 / 1 ACTIVE'}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Strict 1 Trade/Day Limit</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">COMPOUNDED EQUITY CURVE</div>
                    <div class="stat-val" style="color: #a78bfa;">Rs {engine.cumulative_capital:,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Permanent Turso Cloud Ledger</div>
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <a href="/ping" target="_blank" class="btn" style="background: #059669;">📡 Engine Ping</a>
                <a href="/api/state" target="_blank" class="btn" style="background: #475569;">📄 View State JSON</a>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #38bdf8;">📊 Active Today's Position (Rank #1 Stock)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th><th>Direction</th><th>Entry</th><th>Current</th><th>Qty (L1/L2)</th><th>Stop Loss</th><th>Targets</th><th>Net P&L</th><th>Candidate Pool</th>
                        </tr>
                    </thead>
                    <tbody>{active_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #a78bfa;">📋 Today's Completed Trade</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th><th>Entry Time</th><th>Entry Px</th><th>Exit Prices</th><th>Net P&L</th><th>Exit Mechanics</th>
                        </tr>
                    </thead>
                    <tbody>{closed_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #10b981;">💾 Multi-Day Trade History (Permanent Turso DB)</h3>
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
                <h3 style="margin-top: 0; color: #94a3b8;">📜 Engine Live Console Logs</h3>
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
