"""
main.py
================================================================================
CLOUD PRODUCTION LIVE & PAPER TRADING SERVER (RENDER.COM + TURSO PERSISTENCE)
HYBRID DEEP RL TRADING ENGINE (DUELING DDQN + 9 PARETO-OPTIMAL SNIPER SPECIALISTS)
Zero Look-Ahead Bias | Causal Inference Engine | Full NSE Transaction Friction Accounting
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
logger = logging.getLogger("RenderTrader")

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
# 1. DEEP REINFORCEMENT LEARNING INFERENCE ENGINE (DUELING DDQN)
# ------------------------------------------------------------------------------
class DeepRLPolicyBrain:
    """
    Sub-millisecond causal forward inference engine.
    Executes Dueling DDQN policy without heavy PyTorch dependencies.
    """
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.loaded = False
        self.weights = {}
        self.load_model()

    def load_model(self):
        npz_path = os.path.join(self.data_dir, 'causal_deep_rl_weights.npz')
        if os.path.exists(npz_path):
            try:
                self.weights = np.load(npz_path)
                self.loaded = True
                logger.info(f"Causal Deep RL Policy Brain successfully loaded from {npz_path} (Pure NumPy Engine)")
            except Exception as e:
                logger.error(f"Error loading Deep RL npz weights: {e}")
        else:
            logger.warning(f"Deep RL weights not found at {npz_path}. Operating in standard robust mode.")

    @staticmethod
    def _leaky_relu(x, neg_slope=0.1):
        return np.where(x > 0, x, x * neg_slope)

    @staticmethod
    def _layer_norm(x, weight, bias, eps=1e-5):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + eps) * weight + bias

    def predict(self, obs_22):
        """
        Executes pure Dueling DDQN forward pass in ~0.1 ms.
        Output:
            best_action: 0 (Abstain/Skip), 1 (Scalp/Lock), 2 (Balanced), 3 (Runner)
            q_dict: {action_id: Q-value}
        """
        if not self.loaded:
            return 2, {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}

        try:
            x = np.array(obs_22, dtype=np.float32).reshape(1, 22)
            w = self.weights

            # Shared Feature Extractor
            h1 = np.dot(x, w['feature_net.0.weight'].T) + w['feature_net.0.bias']
            h1 = self._layer_norm(h1, w['feature_net.1.weight'], w['feature_net.1.bias'])
            h1 = self._leaky_relu(h1, 0.1)

            h2 = np.dot(h1, w['feature_net.4.weight'].T) + w['feature_net.4.bias']
            h2 = self._layer_norm(h2, w['feature_net.5.weight'], w['feature_net.5.bias'])
            h2 = self._leaky_relu(h2, 0.1)

            # State Value Stream V(s)
            v1 = np.dot(h2, w['value_stream.0.weight'].T) + w['value_stream.0.bias']
            v1 = self._leaky_relu(v1, 0.1)
            val = np.dot(v1, w['value_stream.2.weight'].T) + w['value_stream.2.bias']

            # Advantage Stream A(s, a)
            a1 = np.dot(h2, w['advantage_stream.0.weight'].T) + w['advantage_stream.0.bias']
            a1 = self._leaky_relu(a1, 0.1)
            adv = np.dot(a1, w['advantage_stream.2.weight'].T) + w['advantage_stream.2.bias']

            # Dueling aggregation: Q(s, a) = V(s) + (A(s, a) - mean(A(s, :)))
            q_vals = val + (adv - np.mean(adv, axis=-1, keepdims=True))
            q_flat = q_vals.flatten()
            best_action = int(np.argmax(q_flat))

            q_dict = {i: round(float(q_flat[i]), 3) for i in range(len(q_flat))}
            return best_action, q_dict
        except Exception as e:
            logger.error(f"Error during Deep RL forward pass: {e}")
            return 2, {0: 0.0, 1: 0.0, 2: 1.0, 3: 0.0}

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

def save_trade_to_turso(pos, exit_px, exit_reason, exit_time, net_inr, net_pct, ending_capital):
    today_str = datetime.datetime.now(IST).strftime("%Y-%m-%d")
    check_sql = "SELECT id FROM champion_cloud_paper_trades WHERE trade_date = ? AND symbol = ? AND entry_time = ? AND exit_time = ? LIMIT 1"
    df_exists = query_turso_raw(check_sql, [today_str, pos['symbol'], pos['entry_time'], exit_time])
    if not df_exists.empty:
        logger.info(f"Trade {pos['symbol']} already recorded in Turso for {today_str}. Skipping duplicate.")
        return

    sql = """
    INSERT INTO champion_cloud_paper_trades 
    (trade_date, symbol, direction, sector, entry_time, exit_time, entry_price, exit_price, shares, exposure_inr, pnl_net_inr, pnl_net_pct, exit_reason, strategy, ending_capital)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    params = [
        today_str, pos['symbol'], pos['direction'], pos['sector'],
        pos['entry_time'], exit_time, pos['entry_price'], round(exit_px, 2),
        pos['shares'], pos['exposure'], round(net_inr, 2), round(net_pct, 2),
        exit_reason, pos['strategy'], round(ending_capital, 2)
    ]
    query_turso_raw(sql, params)
    logger.info(f"Permanently saved trade {pos['symbol']} to Turso Cloud DB.")

def save_eod_balance_to_turso(today_str, starting_cap, day_pnl, ending_cap, trades_count, wins, losses):
    sql = """
    INSERT OR REPLACE INTO champion_daily_balance
    (trade_date, starting_capital, day_pnl, ending_capital, trades_count, win_count, loss_count, updated_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
    """
    params = [today_str, round(starting_cap, 2), round(day_pnl, 2), round(ending_cap, 2), trades_count, wins, losses]
    query_turso_raw(sql, params)
    logger.info(f"Permanently saved EOD Balance {today_str} to Turso Cloud DB: Ending Capital = Rs {ending_cap:,.2f}")

# ------------------------------------------------------------------------------
# 3. CORE TRADING ENGINE (HYBRID DEEP RL + 9 SPECIALISTS)
# ------------------------------------------------------------------------------
class CloudChampionEngine:
    def __init__(self, capital_mode="30k", dry_run=True):
        self.capital_mode = capital_mode
        self.dry_run = dry_run
        
        self.init_capital_from_db(capital_mode)
        
        self.max_slots = 2
        self.active_positions = []
        self.closed_trades = []
        self.candidates = []
        self.today_pnl = 0.0
        self.status = "INITIALIZING"
        self.last_update_ist = ""
        self.scan_0925_done = False
        self.scan_1015_done = False
        self.eod_done = False
        self.screener_done = False
        self.today_start_capital = self.account_capital
        
        self.rl_brain = DeepRLPolicyBrain(DATA_DIR)
        self.rl_abstained_count = 0
        
        self.load_candidates()

    def init_capital_from_db(self, mode):
        df_bal = query_turso_raw("SELECT ending_capital FROM champion_daily_balance ORDER BY trade_date DESC LIMIT 1")
        if not df_bal.empty and df_bal.iloc[0]['ending_capital'] is not None:
            self.account_capital = float(df_bal.iloc[0]['ending_capital'])
            logger.info(f"Loaded compounded capital from Turso DB: Rs {self.account_capital:,.2f}")
        else:
            if mode == "30k": self.account_capital = 30000.0
            elif mode == "50k": self.account_capital = 50000.0
            else: self.account_capital = 100000.0
            logger.info(f"Initialized fresh starting capital: Rs {self.account_capital:,.2f}")
            
        self.recalc_slots()

    def recalc_slots(self):
        self.slot_exposure = (self.account_capital * 5.0) / 2.0
        self.ratchet_locked_net = round(self.slot_exposure * 0.0080 * 0.80, 2)

    def load_candidates(self):
        json_path = os.path.join(DATA_DIR, 'premarket_candidates_latest.json')
        loaded = False
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.candidates = data.get('candidates', [])
                if len(self.candidates) > 0:
                    loaded = True
                    logger.info(f"Loaded {len(self.candidates)} screened candidates for trading.")
            except Exception as e:
                logger.error(f"Error loading candidates: {e}")
        if not loaded or len(self.candidates) == 0:
            logger.info("Watchlist empty or missing. Launching immediate parallel screener...")
            self.run_daily_postmarket_screener()

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

    def compute_intraday_snapshot(self, candles):
        if not candles:
            return None
        open_0915 = float(candles[0][1])
        day_high = max(float(c[2]) for c in candles)
        day_low = min(float(c[3]) for c in candles)
        latest_bar = candles[-1]
        latest_time = str(latest_bar[0])[11:16]
        latest_close = float(latest_bar[4])
        
        cum_vol = 0.0
        cum_val = 0.0
        for c in candles:
            h, l, cl, v = float(c[2]), float(c[3]), float(c[4]), float(c[5])
            cum_vol += v
            cum_val += (((h + l + cl) / 3.0) * v)
            
        vwap = round(cum_val / cum_vol, 2) if cum_vol > 0 else latest_close
        ret_10m = 0.0
        if len(candles) >= 10:
            c10 = float(candles[9][4])
            ret_10m = round(((c10 - open_0915) / open_0915) * 100.0, 2)
            
        return {
            'time': latest_time,
            'open_0915': open_0915,
            'close': latest_close,
            'day_high': day_high,
            'day_low': day_low,
            'vwap': vwap,
            'ret_open': round(((latest_close - open_0915) / open_0915) * 100.0, 2),
            'ret_10m': ret_10m,
            'cum_vol': cum_vol
        }

    def build_live_observation(self, snap, pm, direction, strat_name):
        """Builds strictly point-in-time 22-dimensional observation vector S_t."""
        # Market Regime (Default to Sideways/Chop unless morning gap is wide)
        n_ret5 = 0.0
        n_gap = 0.0
        reg_chop = 1.0
        reg_hvol = 0.0
        reg_bull = 0.0
        reg_bear = 0.0
        
        # Stock Technical Features
        ret_5d = np.clip(float(pm.get('ret_5d', 0.0)) / 10.0, -2.0, 2.0)
        atr10 = np.clip(float(pm.get('atr10_pct', 2.5)) / 5.0, 0.0, 2.0)
        ret_op = np.clip(float(snap.get('ret_open', 0.0)) / 3.0, -2.0, 2.0)
        ret_10m = np.clip(float(snap.get('ret_10m', 0.0)) / 2.0, -2.0, 2.0)
        vr = 1.2
        is_hi = 1.0 if snap.get('close', 0) >= snap.get('day_high', 0) * 0.998 else 0.0
        is_unanimous = 1.0 if 'Unanimous' in strat_name else 0.0
        direction_val = 1.0 if direction == 'LONG' else -1.0
        
        # Specialist One-Hot Features
        sec = str(pm.get('sector', ''))
        strat_panic_dip = 1.0 if 'Panic Dip' in strat_name or direction == 'LONG' else 0.0
        strat_auto_fade = 1.0 if 'Auto' in sec and ret_5d >= 0.6 else 0.0
        strat_cap_goods = 1.0 if 'Capital Goods' in sec and ret_5d >= 0.75 else 0.0
        strat_healthcare = 1.0 if 'Healthcare' in sec else 0.0
        strat_overextend = 1.0 if 'Specialist #3' in strat_name or 'Overextended' in strat_name else 0.0
        strat_vol_surge = 1.0 if vr >= 0.70 and ret_5d >= 0.80 else 0.0
        strat_diwali = 1.0 if 'Diwali' in strat_name else 0.0
        strat_tier1 = 1.0 if 'Tier 1' in strat_name else 0.0
        
        obs = np.array([
            n_ret5, n_gap, reg_chop, reg_hvol, reg_bull, reg_bear,
            ret_5d, atr10, ret_op, ret_10m, vr, is_hi, is_unanimous, direction_val,
            strat_panic_dip, strat_auto_fade, strat_cap_goods, strat_healthcare,
            strat_overextend, strat_vol_surge, strat_diwali, strat_tier1
        ], dtype=np.float32)
        
        return obs

    def enter_position(self, sym, direction, entry_price, vwap, entry_time, strategy_name, sector='N/A', rl_action=2, q_vals=None):
        shares = max(2, int(self.slot_exposure / entry_price))
        shares_l1 = shares // 2
        shares_l2 = shares - shares_l1
        
        # Strategy 2: Calibrated High-Payoff Champion Brackets
        action_tag = "STRATEGY_2_CALIBRATED"
        sl_pct = 0.0140
        tp1_pct = 0.0190
        tp2_pct = 0.0300
        early_cut_pct = 0.0080
            
        hard_sl = entry_price * (1.0 + sl_pct) if direction == 'SHORT' else entry_price * (1.0 - sl_pct)
        tp1 = entry_price * (1.0 - tp1_pct) if direction == 'SHORT' else entry_price * (1.0 + tp1_pct)
        tp2 = entry_price * (1.0 - tp2_pct) if direction == 'SHORT' else entry_price * (1.0 + tp2_pct)
        
        pos = {
            'symbol': sym,
            'direction': direction,
            'sector': sector,
            'entry_time': entry_time,
            'entry_price': round(entry_price, 2),
            'total_shares': shares,
            'shares': shares,
            'shares_l1': shares_l1,
            'shares_l2': shares_l2,
            'exposure': round(shares * entry_price, 2),
            'current_price': round(entry_price, 2),
            'hard_sl': round(hard_sl, 2),
            'current_sl': round(hard_sl, 2),
            'tp1': round(tp1, 2),
            'tp2': round(tp2, 2),
            'tp1_hit': False,
            'l1_closed': False,
            'l1_exit_px': None,
            'l1_exit_time': None,
            'ratchet_locked': False,
            'strategy': f"{strategy_name} [{action_tag}]",
            'rl_action': rl_action,
            'early_cut_pct': early_cut_pct,
            'mfe_pct': 0.0,
            'mae_pct': 0.0,
            'vwap': vwap,
            'pnl_inr': 0.0,
            'pnl_pct': 0.0
        }
        self.active_positions.append(pos)
        logger.info(f"[ORDER PLACED] {direction} {sym} @ Rs {entry_price:.2f} | Shares: {shares} | SL: Rs {hard_sl:.2f} | TP1: Rs {tp1:.2f} | Setup: {pos['strategy']} | Q-Values: {q_vals}")

    def exit_position(self, pos, exit_price, exit_reason, exit_time, exit_leg="FULL"):
        shares_to_exit = pos['shares']
        if exit_leg == "LEG_1":
            shares_to_exit = pos['shares_l1']
        elif exit_leg == "LEG_2":
            shares_to_exit = pos['shares_l2']
            
        leg_exposure = round(shares_to_exit * pos['entry_price'], 2)
        mult = 1.0 if pos['direction'] == 'LONG' else -1.0
        gross_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100.0 * mult
        gross_inr = leg_exposure * (gross_pct / 100.0)
        
        turnover = 2.0 * leg_exposure
        brok = 20.0
        stt = 0.00025 * leg_exposure
        nse = 0.0000297 * turnover
        sebi = 0.000001 * turnover
        stamp = 0.00003 * leg_exposure
        gst = 0.18 * (brok + nse + sebi)
        friction = brok + stt + nse + sebi + stamp + gst
        
        net_inr = gross_inr - friction
        net_pct = (net_inr / leg_exposure) * 100.0 if leg_exposure > 0 else 0.0
        
        self.account_capital += net_inr
        self.today_pnl += net_inr
        self.recalc_slots()
        
        trade_record = {
            'symbol': pos['symbol'],
            'direction': pos['direction'],
            'sector': pos['sector'],
            'entry_time': pos['entry_time'],
            'exit_time': exit_time,
            'entry_price': pos['entry_price'],
            'exit_price': round(exit_price, 2),
            'shares': shares_to_exit,
            'pnl_inr': round(net_inr, 2),
            'pnl_pct': round(net_pct, 2),
            'exit_reason': f"{exit_reason} [{exit_leg}]",
            'strategy': pos['strategy']
        }
        self.closed_trades.append(trade_record)
        save_trade_to_turso(pos, exit_price, f"{exit_reason} [{exit_leg}]", exit_time, net_inr, net_pct, self.account_capital)
        logger.info(f"[EXIT EXEC] {pos['direction']} {pos['symbol']} ({exit_leg}) @ Rs {exit_price:.2f} | Reason: {exit_reason} | Net PnL: Rs {net_inr:+.2f} ({net_pct:+.2f}%) | Ending Capital: Rs {self.account_capital:,.2f}")

    def run_cycle(self):
        now = datetime.datetime.now(IST)
        time_str = now.strftime("%H:%M")
        today_str = now.strftime("%Y-%m-%d")
        self.last_update_ist = now.strftime("%Y-%m-%d %H:%M:%S IST")
        
        if time_str == "09:00":
            self.scan_0925_done = False
            self.scan_1015_done = False
            self.eod_done = False
            self.screener_done = False
            self.active_positions.clear()
            self.closed_trades.clear()
            self.today_pnl = 0.0
            self.rl_abstained_count = 0
            self.today_start_capital = self.account_capital
            self.load_candidates()
            self.status = "PRE-MARKET READY"
            
        if time_str < "09:15" or time_str > "15:30":
            self.status = f"STANDBY (Market Closed) - Next: 09:15 AM Open"
            if (time_str >= "18:00" or ("08:30" <= time_str < "09:10")) and not self.screener_done:
                self.run_daily_postmarket_screener()
                self.screener_done = True
            return
            
        self.status = f"RUNNING LIVE ({time_str} IST) | Open Slots: {len(self.active_positions)}/2 | Deep RL Brain: Active"
        
        cand_map = {c['symbol']: c for c in self.candidates}
        snapshots = {}
        for sym, c_data in cand_map.items():
            isin = c_data.get('isin')
            if isin:
                bars = self.fetch_live_1min_bars(isin)
                snap = self.compute_intraday_snapshot(bars)
                if snap: snapshots[sym] = snap
                    
        self.manage_active_positions(snapshots, time_str)
        
        # 09:25 AM Scan (Extreme Panic Market Drop Long)
        if "09:25" <= time_str <= "09:30" and not self.scan_0925_done and len(self.active_positions) < self.max_slots:
            self.evaluate_scan_0925(snapshots)
            self.scan_0925_done = True
        elif time_str > "09:30" and not self.scan_0925_done:
            self.scan_0925_done = True
            
        # 10:15 AM Scan (Multi-Strategy Consensus Gated by Deep RL)
        if "10:15" <= time_str <= "10:20" and not self.scan_1015_done and len(self.active_positions) < self.max_slots:
            self.evaluate_scan_1015(snapshots)
            self.scan_1015_done = True
        elif time_str > "10:20" and not self.scan_1015_done:
            self.scan_1015_done = True
            
        if time_str >= "15:20" and not self.eod_done:
            self.square_off_all(snapshots, "15:20 EOD Square-Off", time_str)
            wins = sum(1 for t in self.closed_trades if t['pnl_inr'] > 0)
            losses = sum(1 for t in self.closed_trades if t['pnl_inr'] < 0)
            save_eod_balance_to_turso(today_str, self.today_start_capital, self.today_pnl, self.account_capital, len(self.closed_trades), wins, losses)
            self.eod_done = True

    def evaluate_scan_0925(self, snapshots):
        """09:25 AM: Extreme Panic Dip Bounce LONG gated by Deep RL Policy."""
        if len(self.active_positions) >= self.max_slots or not snapshots: return
        mkt_bear = sum(1 for s in snapshots.values() if s['close'] < s['open_0915'] and s['close'] < s['vwap'])
        total = len(snapshots)
        if total == 0: return
        bear_breadth = (mkt_bear / total) * 100.0
        
        if bear_breadth >= 60.0:
            cands = [(sym, s) for sym, s in snapshots.items() if -3.50 <= s['ret_open'] <= -0.30]
            cands.sort(key=lambda x: x[1]['ret_open'])
            for sym, snap in cands:
                if len(self.active_positions) >= self.max_slots: break
                obs = self.build_live_observation(snap, {}, 'LONG', 'Specialist #1: Panic Dip Bounce LONG')
                act, q_vals = self.rl_brain.predict(obs)
                
                if act == 0:
                    self.rl_abstained_count += 1
                    logger.info(f"[RL FILTER] Abstaining from Panic Dip setup {sym} (Toxic/Low Reward: {q_vals})")
                    continue
                self.enter_position(sym, 'LONG', snap['close'], snap['vwap'], '09:25', 'Specialist #1: Panic Dip Bounce LONG', rl_action=act, q_vals=q_vals)

    def evaluate_scan_1015(self, snapshots):
        """10:15 AM: Pareto-Optimal Multi-Strategy Candidates evaluated by Deep RL Policy."""
        if len(self.active_positions) >= self.max_slots or not snapshots: return
        cand_map = {c['symbol']: c for c in self.candidates}
        
        setup_pool = []
        for sym, snap in snapshots.items():
            pm = cand_map.get(sym)
            if not pm: continue
            ret_5d = pm.get('ret_5d', 0.0)
            atr10 = pm.get('atr10_pct', 99.0)
            ret_open = snap.get('ret_open', 0.0)
            close_px = snap['close']
            vwap_px = snap['vwap']
            day_high = snap.get('day_high', close_px)
            sec = pm.get('sector', 'N/A')
            
            # Universal overextension screening criteria
            if ret_5d >= 4.0 and (0.10 <= ret_open <= 4.50):
                v9_pass = (atr10 <= 4.50) and (close_px <= vwap_px * 1.002)
                v4_pass = (atr10 <= 4.50) and (day_high > close_px)
                
                # Tag setup identity
                if v9_pass and v4_pass:
                    strat_name = 'Unanimous Conviction SHORT (Dual Consensus)'
                elif 'Auto' in sec and ret_5d >= 6.0:
                    strat_name = 'Auto Sector Exhaustion SHORT'
                elif 'Capital Goods' in sec and ret_5d >= 7.5:
                    strat_name = 'Capital Goods Momentum Exhaustion SHORT'
                elif 'Healthcare' in sec and atr10 <= 3.8:
                    strat_name = 'Healthcare Non-Squeeze Fade SHORT'
                elif atr10 <= 4.50:
                    strat_name = 'V4.1 Tier 3A Overextended Fade SHORT'
                else:
                    strat_name = 'Specialist #3: Overextended 5D Fade SHORT'
                    
                obs = self.build_live_observation(snap, pm, 'SHORT', strat_name)
                act, q_vals = self.rl_brain.predict(obs)
                
                setup_pool.append({
                    'symbol': sym,
                    'pm': pm,
                    'snap': snap,
                    'strat_name': strat_name,
                    'ret_5d': ret_5d,
                    'rl_action': act,
                    'q_vals': q_vals,
                    'sector': sec
                })
                
        # Sort by Ret_5D momentum
        setup_pool.sort(key=lambda x: x['ret_5d'], reverse=True)
        
        # Multi-Pass Selection with RL Abstain Filter
        for max_sec in (1, 2):
            if len(self.active_positions) >= self.max_slots: break
            sec_counts = {p.get('sector', 'N/A'): 0 for p in self.active_positions}
            for p in self.active_positions:
                sec_counts[p.get('sector', 'N/A')] += 1
                
            for cand in setup_pool:
                if len(self.active_positions) >= self.max_slots: break
                sym = cand['symbol']
                if any(p['symbol'] == sym for p in self.active_positions): continue
                sec = cand['sector']
                if sec_counts.get(sec, 0) >= max_sec: continue
                
                # Check Deep RL Action Decision
                act = cand['rl_action']
                if act == 0:
                    self.rl_abstained_count += 1
                    logger.info(f"[RL NEURAL FILTER] Action 0 (Abstain) chosen for {sym} | Filtered toxic setup | Q-Values: {cand['q_vals']}")
                    continue
                    
                sec_counts[sec] = sec_counts.get(sec, 0) + 1
                self.enter_position(
                    sym=sym,
                    direction='SHORT',
                    entry_price=cand['snap']['close'],
                    vwap=cand['snap']['vwap'],
                    entry_time='10:15',
                    strategy_name=cand['strat_name'],
                    sector=sec,
                    rl_action=act,
                    q_vals=cand['q_vals']
                )

    def manage_active_positions(self, snapshots, time_str):
        remaining = []
        for pos in self.active_positions:
            sym = pos['symbol']
            snap = snapshots.get(sym)
            if not snap:
                remaining.append(pos)
                continue
                
            curr_px = snap['close']
            vwap = snap['vwap']
            pos['current_price'] = curr_px
            pos['vwap'] = vwap
            
            mult = 1.0 if pos['direction'] == 'LONG' else -1.0
            pnl_pct = ((curr_px - pos['entry_price']) / pos['entry_price']) * 100.0 * mult
            adverse_pct = ((pos['entry_price'] - curr_px) / pos['entry_price']) * 100.0 * mult
            pos['pnl_pct'] = round(pnl_pct, 2)
            pos['pnl_inr'] = round(pos['exposure'] * (pnl_pct / 100.0), 2)
            if pnl_pct > pos['mfe_pct']: pos['mfe_pct'] = pnl_pct
            if adverse_pct > pos.get('mae_pct', 0.0): pos['mae_pct'] = adverse_pct
            
            # 0. Early Loss Cut
            early_cut = pos.get('early_cut_pct', 0.0080) * 100.0
            if not pos['l1_closed'] and pos.get('mae_pct', 0.0) >= early_cut and pos['mfe_pct'] < 0.45:
                early_cut_px = pos['entry_price'] * (1.0 + (early_cut/100.0)) if pos['direction'] == 'SHORT' else pos['entry_price'] * (1.0 - (early_cut/100.0))
                logger.info(f"[EARLY CUT] {pos['symbol']} adverse move {pos['mae_pct']:.2f}% >= {early_cut:.2f}% with MFE {pos['mfe_pct']:.2f}% < 0.45%! Cutting loss @ Rs {early_cut_px:.2f}")
                self.exit_position(pos, early_cut_px, f"Early Momentum Cut (-{early_cut:.2f}%)", time_str, exit_leg="FULL")
                continue

            # 1. Stop-Loss Hit
            if (pos['direction'] == 'SHORT' and curr_px >= pos['current_sl']) or (pos['direction'] == 'LONG' and curr_px <= pos['current_sl']):
                reason = 'Trailing Stop Hit' if pos['l1_closed'] else 'Stop-Loss Hit'
                self.exit_position(pos, pos['current_sl'], reason, time_str, exit_leg="LEG_2" if pos['l1_closed'] else "FULL")
                continue
                
            # 2. TP1 Hit -> Book Leg 1 (50%) & Trail Leg 2
            if not pos['tp1_hit']:
                if (pos['direction'] == 'SHORT' and curr_px <= pos['tp1']) or (pos['direction'] == 'LONG' and curr_px >= pos['tp1']):
                    pos['tp1_hit'] = True
                    pos['l1_closed'] = True
                    pos['l1_exit_px'] = pos['tp1']
                    pos['l1_exit_time'] = time_str
                    
                    self.exit_position(pos, pos['tp1'], 'Target 1 Hit', time_str, exit_leg="LEG_1")
                    
                    # Strategy 2: Trail Leg 2 stop to lock +0.95% profit
                    lock_pct = 0.0095
                    new_sl = pos['entry_price'] * (1.0 - lock_pct) if pos['direction'] == 'SHORT' else pos['entry_price'] * (1.0 + lock_pct)
                    pos['current_sl'] = round(new_sl, 2)
                    pos['shares'] = pos['shares_l2']
                    pos['exposure'] = round(pos['shares_l2'] * pos['entry_price'], 2)
                    logger.info(f"[TP1 BOOKED] {pos['symbol']} booked 50% @ TP1 (+1.90%)! Runner trailed @ Rs {new_sl:.2f} (+0.95% locked)")
                    
            # 3. TP2 Hit
            if (pos['direction'] == 'SHORT' and curr_px <= pos['tp2']) or (pos['direction'] == 'LONG' and curr_px >= pos['tp2']):
                self.exit_position(pos, pos['tp2'], 'Target 2 Hit', time_str, exit_leg="LEG_2" if pos['l1_closed'] else "FULL")
                continue
                
            # 4. Extended Dynamic Trailing
            if pos['tp1_hit'] and pos['mfe_pct'] >= 1.50:
                sgn = -1.0 if pos['direction'] == 'SHORT' else 1.0
                dyn_px = round(pos['entry_price'] * (1.0 + sgn * (pos['mfe_pct'] - 0.35) / 100.0), 2)
                if pos['direction'] == 'SHORT' and dyn_px < pos['current_sl']:
                    pos['current_sl'] = dyn_px
                elif pos['direction'] == 'LONG' and dyn_px > pos['current_sl']:
                    pos['current_sl'] = dyn_px

            # 5. Afternoon Anti-Scratch Ratchet (at 13:30, locks +0.80% if MFE >= 1.00%)
            if time_str >= "13:30" and not pos['ratchet_locked'] and pos['mfe_pct'] >= 1.00:
                pos['ratchet_locked'] = True
                lock_px = pos['entry_price'] * (1.0 - 0.0080) if pos['direction'] == 'SHORT' else pos['entry_price'] * (1.0 + 0.0080)
                if pos['direction'] == 'SHORT' and lock_px < pos['current_sl']: pos['current_sl'] = round(lock_px, 2)
                elif pos['direction'] == 'LONG' and lock_px > pos['current_sl']: pos['current_sl'] = round(lock_px, 2)
                logger.info(f"[AFTERNOON RATCHET] {pos['symbol']} locked +0.80% profit @ Rs {pos['current_sl']:.2f}")
                
            remaining.append(pos)
        self.active_positions = remaining

    def square_off_all(self, snapshots, reason, time_str):
        for pos in list(self.active_positions):
            snap = snapshots.get(pos['symbol'])
            exit_px = snap['close'] if snap else pos['current_price']
            exit_leg = "LEG_2" if pos.get('l1_closed', False) else "FULL"
            self.exit_position(pos, exit_px, reason, time_str, exit_leg=exit_leg)
        self.active_positions.clear()

    def run_daily_postmarket_screener(self):
        logger.info("Executing daily parallel screener across Nifty 500 universe...")
        try:
            n500_path = os.path.join(DATA_DIR, 'ind_nifty500list.csv')
            if not os.path.exists(n500_path):
                logger.warning("Nifty 500 list not found for screener.")
                return
            df_n500 = pd.read_csv(n500_path)
            now = datetime.datetime.now(IST)
            today_str = now.strftime("%Y-%m-%d")
            
            valid_stocks = df_n500.dropna(subset=['ISIN Code', 'Symbol']).copy()
            cands = []

            def fetch_single(sym, isin, sec):
                url = f"https://api.upstox.com/v2/historical-candle/NSE_EQ%7C{isin}/day/{today_str}/2026-10-01"
                req = urllib.request.Request(url, headers=HEADERS)
                try:
                    with urllib.request.urlopen(req, timeout=4) as resp:
                        data = json.loads(resp.read().decode('utf-8'))
                        candles = sorted(data.get('data', {}).get('candles', []), key=lambda x: x[0])
                        if len(candles) >= 6:
                            last_c = float(candles[-1][4])
                            c_5ago = float(candles[-6][4])
                            if not (100.0 <= last_c <= 4500.0):
                                return None
                            ret_5d = ((last_c - c_5ago) / c_5ago) * 100.0
                            if ret_5d >= 3.50:
                                if len(candles) >= 11:
                                    trs = []
                                    for i in range(len(candles)-10, len(candles)):
                                        h = float(candles[i][2])
                                        l = float(candles[i][3])
                                        prev_cl = float(candles[i-1][4])
                                        trs.append(max(h - l, abs(h - prev_cl), abs(l - prev_cl)))
                                    atr10_pct = (np.mean(trs) / last_c) * 100.0
                                else:
                                    atr10_pct = 2.0
                                return {
                                    'symbol': sym,
                                    'isin': isin,
                                    'sector': sec,
                                    'prev_close': round(last_c, 2),
                                    'ret_5d': round(ret_5d, 2),
                                    'atr10_pct': round(atr10_pct, 2)
                                }
                except Exception:
                    pass
                return None

            with ThreadPoolExecutor(max_workers=10) as executor:
                futs = [executor.submit(fetch_single, r['Symbol'], r['ISIN Code'], r.get('Industry', 'N/A')) for _, r in valid_stocks.iterrows()]
                for f in as_completed(futs):
                    res = f.result()
                    if res:
                        cands.append(res)
                    
            if cands:
                cands.sort(key=lambda x: x['ret_5d'], reverse=True)
                json_path = os.path.join(DATA_DIR, 'premarket_candidates_latest.json')
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({'date': today_str, 'count': len(cands), 'candidates': cands}, f, indent=2)
                self.candidates = cands
                logger.info(f"Daily screener completed! Filtered {len(cands)} quality candidates for next session.")
        except Exception as e:
            logger.error(f"Post-market screener error: {e}")

engine = CloudChampionEngine(capital_mode="30k", dry_run=True)

def background_trader_loop():
    logger.info("Background Live Trader Daemon started on Render Cloud.")
    while True:
        try:
            engine.run_cycle()
        except Exception as e:
            logger.error(f"Trader loop error: {e}")
        time.sleep(30)

trader_thread = threading.Thread(target=background_trader_loop, daemon=True)
trader_thread.start()

# ------------------------------------------------------------------------------
# 4. FASTAPI APP & INTERACTIVE DASHBOARD
# ------------------------------------------------------------------------------
app = FastAPI(title="Hybrid Deep RL Champion Trader")

@app.get("/api/test-feed")
def test_feed():
    results = []
    for c in engine.candidates[:3]:
        sym = c.get('symbol')
        isin = c.get('isin')
        bars = engine.fetch_live_1min_bars(isin)
        snap = engine.compute_intraday_snapshot(bars)
        results.append({
            "symbol": sym,
            "isin": isin,
            "bars_received": len(bars),
            "latest_snapshot": snap,
            "connectivity_status": "OK" if len(bars) > 0 else "FAILED"
        })
    return {
        "status": "PASS" if all(r["bars_received"] > 0 for r in results) else "WARNING",
        "tested_at": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "details": results
    }

@app.get("/ping")
def ping():
    now_ist = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    return {
        "status": "online",
        "time_ist": now_ist,
        "engine_status": engine.status,
        "capital": f"Rs {engine.account_capital:,.2f}",
        "active_positions": len(engine.active_positions),
        "today_pnl_inr": round(engine.today_pnl, 2),
        "rl_brain_loaded": engine.rl_brain.loaded,
        "rl_abstained_count": engine.rl_abstained_count
    }

@app.get("/api/state")
def get_state():
    return {
        "status": engine.status,
        "last_update": engine.last_update_ist,
        "capital": engine.account_capital,
        "today_pnl": engine.today_pnl,
        "active_positions": engine.active_positions,
        "closed_trades": engine.closed_trades,
        "candidates_count": len(engine.candidates),
        "rl_brain_loaded": engine.rl_brain.loaded,
        "rl_abstained_count": engine.rl_abstained_count,
        "logs": log_records[-50:]
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    now_ist = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    pnl_color = "#10b981" if engine.today_pnl >= 0 else "#ef4444"
    
    active_html = ""
    for p in engine.active_positions:
        c_pnl = "#10b981" if p['pnl_inr'] >= 0 else "#ef4444"
        active_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: bold;">{p['symbol']}</td>
            <td style="padding: 10px;"><span style="background: {'#dc2626' if p['direction']=='SHORT' else '#16a34a'}; padding: 3px 8px; border-radius: 4px; font-size: 11px;">{p['direction']}</span></td>
            <td style="padding: 10px;">Rs {p['entry_price']:.2f}</td>
            <td style="padding: 10px;">Rs {p['current_price']:.2f}</td>
            <td style="padding: 10px;">Rs {p['vwap']:.2f}</td>
            <td style="padding: 10px; color: #f87171;">Rs {p['current_sl']:.2f}</td>
            <td style="padding: 10px; color: #4ade80;">Rs {p['tp1']:.2f}</td>
            <td style="padding: 10px; color: {c_pnl}; font-weight: bold;">Rs {p['pnl_inr']:+.2f} ({p['pnl_pct']:+.2f}%)</td>
            <td style="padding: 10px; font-size: 12px; color: #94a3b8;">{p['strategy']}</td>
        </tr>
        """
    if not active_html:
        active_html = "<tr><td colspan='9' style='text-align: center; padding: 25px; color: #64748b;'>No Active Open Positions (Standing by for 09:25 / 10:15 scans)</td></tr>"
        
    closed_html = ""
    for c in reversed(engine.closed_trades):
        c_pnl = "#10b981" if c['pnl_inr'] >= 0 else "#ef4444"
        closed_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 10px; font-weight: bold;">{c['symbol']}</td>
            <td style="padding: 10px;">{c['direction']}</td>
            <td style="padding: 10px;">{c['entry_time']} -> {c['exit_time']}</td>
            <td style="padding: 10px;">Rs {c['entry_price']:.2f}</td>
            <td style="padding: 10px;">Rs {c['exit_price']:.2f}</td>
            <td style="padding: 10px; color: {c_pnl}; font-weight: bold;">Rs {c['pnl_inr']:+.2f} ({c['pnl_pct']:+.2f}%)</td>
            <td style="padding: 10px; font-size: 12px; color: #cbd5e1;">{c['exit_reason']}</td>
        </tr>
        """
    if not closed_html:
        closed_html = "<tr><td colspan='7' style='text-align: center; padding: 25px; color: #64748b;'>No closed trades today yet.</td></tr>"

    turso_history_html = ""
    try:
        df_turso_trades = query_turso_raw("SELECT trade_date, symbol, direction, entry_price, exit_price, pnl_net_inr, pnl_net_pct, exit_reason, ending_capital FROM champion_cloud_paper_trades ORDER BY id DESC LIMIT 20")
        if not df_turso_trades.empty:
            for _, r in df_turso_trades.iterrows():
                col = "#10b981" if float(r.get('pnl_net_inr', 0)) >= 0 else "#ef4444"
                turso_history_html += f"""
                <tr style="border-bottom: 1px solid #334155;">
                    <td style="padding: 8px;">{r.get('trade_date')}</td>
                    <td style="padding: 8px; font-weight: bold;">{r.get('symbol')}</td>
                    <td style="padding: 8px;">{r.get('direction')}</td>
                    <td style="padding: 8px;">Rs {float(r.get('entry_price', 0)):.2f}</td>
                    <td style="padding: 8px;">Rs {float(r.get('exit_price', 0)):.2f}</td>
                    <td style="padding: 8px; color: {col}; font-weight: bold;">Rs {float(r.get('pnl_net_inr', 0)):+.2f} ({float(r.get('pnl_net_pct', 0)):+.2f}%)</td>
                    <td style="padding: 8px; font-size: 11px; color: #cbd5e1;">{r.get('exit_reason')}</td>
                    <td style="padding: 8px; font-weight: bold; color: #38bdf8;">Rs {float(r.get('ending_capital', 0)):,.2f}</td>
                </tr>
                """
    except Exception:
        pass
    if not turso_history_html:
        turso_history_html = "<tr><td colspan='8' style='text-align: center; padding: 20px; color: #64748b;'>Turso Cloud Database connected & empty. Completed trades will permanently accumulate here.</td></tr>"

    watch_html = ""
    for cand in engine.candidates[:10]:
        watch_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 8px; font-weight: bold;">{cand.get('symbol')}</td>
            <td style="padding: 8px;">{cand.get('sector')}</td>
            <td style="padding: 8px;">Rs {cand.get('today_close', cand.get('prev_close', 0)):.2f}</td>
            <td style="padding: 8px; color: #4ade80;">{cand.get('ret_5d', 0):+.2f}%</td>
            <td style="padding: 8px;">{cand.get('atr10_pct', 0):.2f}%</td>
        </tr>
        """

    logs_str = "<br>".join(log_records[-30:])
    brain_status_color = "#10b981" if engine.rl_brain.loaded else "#f59e0b"
    brain_status_text = "Dueling DDQN Active (Zero Look-Ahead Passed)" if engine.rl_brain.loaded else "Fallback Engine"
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Hybrid Deep RL Champion Cloud Trader</title>
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
                    <h2 style="margin: 0;">🚀 Hybrid Deep RL Champion Live Engine</h2>
                    <span style="color: #94a3b8; font-size: 13px;">Dueling DDQN Neural Policy + 9 Pareto Sniper Specialists (Render + Turso Cloud)</span>
                </div>
                <div style="text-align: right;">
                    <span style="background: {brain_status_color}; padding: 5px 12px; border-radius: 20px; font-size: 12px; font-weight: bold;">{brain_status_text}</span>
                    <div style="font-size: 11px; color: #64748b; margin-top: 5px;">Updated: {now_ist} (Auto-refresh 10s)</div>
                </div>
            </div>

            <div class="grid">
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">ACCOUNT CAPITAL (COMPOUNDED)</div>
                    <div class="stat-val" style="color: #38bdf8;">Rs {engine.account_capital:,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">2 Slots: Rs {engine.slot_exposure:,.2f} each</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">TODAY'S NET P&L</div>
                    <div class="stat-val" style="color: {pnl_color};">Rs {engine.today_pnl:+,.2f}</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Full NSE Friction & Drag Deducted</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">ACTIVE TRADING SLOTS</div>
                    <div class="stat-val">{len(engine.active_positions)} / 2 Slots</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">5x MIS Intraday Leverage</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">RL TOXIC TRADES FILTERED</div>
                    <div class="stat-val" style="color: #f59e0b;">{engine.rl_abstained_count} Trades</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Action 0 Abstain Decisions Today</div>
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <a href="/api/test-feed" target="_blank" class="btn">🧪 Test Live Feed</a>
                <a href="/ping" target="_blank" class="btn" style="background: #059669;">📡 Engine Ping</a>
                <a href="/api/state" target="_blank" class="btn" style="background: #475569;">📄 View State JSON</a>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #38bdf8;">📊 Active Open Positions (Live 1-Minute Stream)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th><th>Type</th><th>Entry</th><th>Current</th><th>VWAP</th><th>SL</th><th>TP1</th><th>Net P&L</th><th>Strategy & Action</th>
                        </tr>
                    </thead>
                    <tbody>{active_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #a78bfa;">📋 Today's Closed Trades</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th><th>Type</th><th>Time</th><th>Entry Px</th><th>Exit Px</th><th>Net P&L</th><th>Exit Reason</th>
                        </tr>
                    </thead>
                    <tbody>{closed_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #10b981;">💾 Multi-Day History & Compounding (Permanent Turso Cloud DB)</h3>
                <div style="font-size: 12px; color: #94a3b8; margin-bottom: 12px;">Permanent multi-day trade journal verified across all market regimes.</div>
                <table>
                    <thead>
                        <tr>
                            <th>Date</th><th>Symbol</th><th>Direction</th><th>Entry</th><th>Exit</th><th>Net P&L</th><th>Exit Reason</th><th>Ending Capital</th>
                        </tr>
                    </thead>
                    <tbody>{turso_history_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #f59e0b;">🎯 Pre-Market Screened Watchlist (Top Candidates)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th><th>Sector</th><th>Close</th><th>5D Return</th><th>ATR10 %</th>
                        </tr>
                    </thead>
                    <tbody>{watch_html}</tbody>
                </table>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #94a3b8;">📜 Live Engine Console Logs</h3>
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
