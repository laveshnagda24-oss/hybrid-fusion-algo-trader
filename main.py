"""
main.py
================================================================================
CLOUD PRODUCTION LIVE & PAPER TRADING SERVER (RENDER.COM + TURSO PERSISTENCE)
HYBRID FUSION CHAMPION 1.0 (WITH MULTI-DAY COMPOUNDING & LIVE MOBILE DASHBOARD)
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
# TURSO CLOUD DATABASE HELPERS (PERMANENT MULTI-DAY PERSISTENCE)
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
# CORE TRADING ENGINE
# ------------------------------------------------------------------------------
class CloudChampionEngine:
    def __init__(self, capital_mode="30k", dry_run=True):
        self.capital_mode = capital_mode
        self.dry_run = dry_run
        
        # Load latest compounded balance from Turso Cloud DB if exists!
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
        self.ratchet_locked_net = round(self.slot_exposure * 0.0060 * 0.80, 2)

    def load_candidates(self):
        json_path = os.path.join(DATA_DIR, 'premarket_candidates_latest.json')
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.candidates = data.get('candidates', [])
                logger.info(f"Loaded {len(self.candidates)} screened candidates for trading.")
            except Exception as e:
                logger.error(f"Error loading candidates: {e}")

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
        return {
            'time': latest_time,
            'open_0915': open_0915,
            'close': latest_close,
            'day_high': day_high,
            'day_low': day_low,
            'vwap': vwap,
            'ret_open': round(((latest_close - open_0915) / open_0915) * 100.0, 2)
        }

    def enter_position(self, sym, direction, entry_price, vwap, entry_time, strategy_name, sector='N/A'):
        shares = max(1, int(self.slot_exposure / entry_price))
        sl_pct = 0.0120 if entry_time == '09:25' else 0.0140
        hard_sl = entry_price * (1.0 + sl_pct) if direction == 'SHORT' else entry_price * (1.0 - sl_pct)
        tp1 = entry_price * (1.0 - 0.0195) if direction == 'SHORT' else entry_price * (1.0 + 0.0195)
        tp2 = entry_price * (1.0 - 0.0340) if direction == 'SHORT' else entry_price * (1.0 + 0.0340)
        
        pos = {
            'symbol': sym,
            'direction': direction,
            'sector': sector,
            'entry_time': entry_time,
            'entry_price': round(entry_price, 2),
            'shares': shares,
            'exposure': round(shares * entry_price, 2),
            'current_price': round(entry_price, 2),
            'hard_sl': round(hard_sl, 2),
            'current_sl': round(hard_sl, 2),
            'tp1': round(tp1, 2),
            'tp2': round(tp2, 2),
            'tp1_hit': False,
            'ratchet_locked': False,
            'strategy': strategy_name,
            'mfe_pct': 0.0,
            'vwap': vwap,
            'pnl_inr': 0.0,
            'pnl_pct': 0.0
        }
        self.active_positions.append(pos)
        logger.info(f"[PAPER ORDER] ENTER {direction} {sym} @ Rs {entry_price:.2f} | Shares: {shares} | SL: Rs {hard_sl:.2f} | TP1: Rs {tp1:.2f} | Setup: {strategy_name}")

    def exit_position(self, pos, exit_price, exit_reason, exit_time):
        mult = 1.0 if pos['direction'] == 'LONG' else -1.0
        gross_pct = ((exit_price - pos['entry_price']) / pos['entry_price']) * 100.0 * mult
        gross_inr = pos['exposure'] * (gross_pct / 100.0)
        
        turnover = 2.0 * pos['exposure']
        brok = 40.0 if not pos['tp1_hit'] else 60.0
        stt = 0.00025 * pos['exposure']
        nse = 0.0000297 * turnover
        sebi = 0.000001 * turnover
        stamp = 0.00003 * pos['exposure']
        gst = 0.18 * (brok + nse + sebi)
        friction = brok + stt + nse + sebi + stamp + gst
        
        net_inr = gross_inr - friction
        net_pct = (net_inr / pos['exposure']) * 100.0
        
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
            'shares': pos['shares'],
            'pnl_inr': round(net_inr, 2),
            'pnl_pct': round(net_pct, 2),
            'exit_reason': exit_reason,
            'strategy': pos['strategy']
        }
        self.closed_trades.append(trade_record)
        save_trade_to_turso(pos, exit_price, exit_reason, exit_time, net_inr, net_pct, self.account_capital)
        logger.info(f"[PAPER EXIT] {pos['direction']} {pos['symbol']} @ Rs {exit_price:.2f} | Reason: {exit_reason} | Net PnL: Rs {net_inr:+.2f} ({net_pct:+.2f}%) | New Capital: Rs {self.account_capital:,.2f}")

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
            self.today_start_capital = self.account_capital
            self.load_candidates()
            self.status = "PRE-MARKET READY"
            
        if time_str < "09:15" or time_str > "15:30":
            self.status = f"STANDBY (Market Closed) - Next: 09:15 AM Open"
            if time_str >= "15:35" and not self.screener_done:
                self.run_daily_postmarket_screener()
                self.screener_done = True
            return
            
        self.status = f"RUNNING LIVE ({time_str} IST) | Open Slots: {len(self.active_positions)}/2"
        
        cand_map = {c['symbol']: c for c in self.candidates}
        snapshots = {}
        for sym, c_data in cand_map.items():
            isin = c_data.get('isin')
            if isin:
                bars = self.fetch_live_1min_bars(isin)
                snap = self.compute_intraday_snapshot(bars)
                if snap: snapshots[sym] = snap
                    
        self.manage_active_positions(snapshots, time_str)
        
        # Strict 5-minute execution windows (avoids executing late if server boots outside window)
        if "09:25" <= time_str <= "09:30" and not self.scan_0925_done and len(self.active_positions) < self.max_slots:
            self.evaluate_scan_0925(snapshots)
            self.scan_0925_done = True
        elif time_str > "09:30" and not self.scan_0925_done:
            self.scan_0925_done = True
            
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
        if len(self.active_positions) >= self.max_slots or not snapshots: return
        mkt_bear = sum(1 for s in snapshots.values() if s['close'] < s['open_0915'] and s['close'] < s['vwap'])
        mkt_bull = sum(1 for s in snapshots.values() if s['close'] > s['open_0915'] and s['close'] > s['vwap'])
        total = len(snapshots)
        if total == 0: return
        bear_breadth = (mkt_bear / total) * 100.0
        bull_breadth = (mkt_bull / total) * 100.0
        
        if bear_breadth >= 60.0:
            cands = [(sym, s) for sym, s in snapshots.items() if -3.50 <= s['ret_open'] <= -0.30]
            cands.sort(key=lambda x: x[1]['ret_open'])
            for sym, snap in cands:
                if len(self.active_positions) >= self.max_slots: break
                self.enter_position(sym, 'LONG', snap['close'], snap['vwap'], '09:25', 'Tier #1 Panic Bounce LONG')
        elif bull_breadth >= 60.0:
            cands = [(sym, s) for sym, s in snapshots.items() if 0.40 <= s['ret_open'] <= 3.50]
            cands.sort(key=lambda x: x[1]['ret_open'], reverse=True)
            for sym, snap in cands:
                if len(self.active_positions) >= self.max_slots: break
                self.enter_position(sym, 'SHORT', snap['close'], snap['vwap'], '09:25', 'Tier #2 Euphoria Fade SHORT')

    def evaluate_scan_1015(self, snapshots):
        if len(self.active_positions) >= self.max_slots or not snapshots: return
        cand_map = {c['symbol']: c for c in self.candidates}
        tier_3a = []
        tier_3b = []
        for sym, snap in snapshots.items():
            pm = cand_map.get(sym)
            if pm and pm.get('ret_5d', 0) >= 5.0 and (0.20 <= snap['ret_open'] <= 4.00):
                if pm.get('atr10_pct', 99) <= 4.50: tier_3a.append((sym, pm, snap))
                else: tier_3b.append((sym, pm, snap))
                    
        tier_3a.sort(key=lambda x: x[1]['ret_5d'], reverse=True)
        tier_3b.sort(key=lambda x: x[1]['ret_5d'], reverse=True)
        
        used_secs = {p['sector'] for p in self.active_positions}
        for sym, pm, snap in tier_3a:
            if len(self.active_positions) >= self.max_slots: break
            if pm.get('sector') in used_secs: continue
            used_secs.add(pm.get('sector'))
            self.enter_position(sym, 'SHORT', snap['close'], snap['vwap'], '10:15', 'Tier #3A 5D Fade SHORT (Prime)', sector=pm.get('sector', 'N/A'))
            
        if len(self.active_positions) < self.max_slots:
            for sym, pm, snap in tier_3b:
                if len(self.active_positions) >= self.max_slots: break
                if pm.get('sector') in used_secs: continue
                used_secs.add(pm.get('sector'))
                self.enter_position(sym, 'SHORT', snap['close'], snap['vwap'], '10:15', 'Tier #3B High-ATR Fade SHORT', sector=pm.get('sector', 'N/A'))

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
            pos['pnl_pct'] = round(pnl_pct, 2)
            pos['pnl_inr'] = round(pos['exposure'] * (pnl_pct / 100.0), 2)
            if pnl_pct > pos['mfe_pct']: pos['mfe_pct'] = pnl_pct
            
            # 1. Check Hard SL / Trailed Stop
            if pos['direction'] == 'SHORT' and curr_px >= pos['current_sl']:
                reason = 'Hard Stop-Loss Hit' if not pos['tp1_hit'] else 'Trailing Stop Hit (+1.40% locked)'
                self.exit_position(pos, pos['current_sl'], reason, time_str)
                continue
            elif pos['direction'] == 'LONG' and curr_px <= pos['current_sl']:
                reason = 'Hard Stop-Loss Hit' if not pos['tp1_hit'] else 'Trailing Stop Hit (+1.40% locked)'
                self.exit_position(pos, pos['current_sl'], reason, time_str)
                continue
                
            # 2. Check TP1 Hit (+1.95%) -> Trail stop to lock +1.40% profit
            if not pos['tp1_hit']:
                if (pos['direction'] == 'SHORT' and curr_px <= pos['tp1']) or (pos['direction'] == 'LONG' and curr_px >= pos['tp1']):
                    pos['tp1_hit'] = True
                    new_sl = pos['entry_price'] * (1.0 - 0.0140) if pos['direction'] == 'SHORT' else pos['entry_price'] * (1.0 + 0.0140)
                    pos['current_sl'] = round(new_sl, 2)
                    logger.info(f"[TARGET 1 HIT] {pos['symbol']} reached +1.95%! SL trailed to lock +1.40% @ Rs {new_sl:.2f}")
                    
            # 3. Check TP2 Hit (+3.40%)
            if (pos['direction'] == 'SHORT' and curr_px <= pos['tp2']) or (pos['direction'] == 'LONG' and curr_px >= pos['tp2']):
                self.exit_position(pos, pos['tp2'], 'Target 2 Profit Reached (+3.40%)', time_str)
                continue
                
            # 4. Extended Dynamic Trailing after +1.50% MFE (pure 5-year model)
            if pos['tp1_hit'] and pos['mfe_pct'] >= 1.50:
                sgn = -1.0 if pos['direction'] == 'SHORT' else 1.0
                dyn_px = round(pos['entry_price'] * (1.0 + sgn * (pos['mfe_pct'] - 0.35) / 100.0), 2)
                if pos['direction'] == 'SHORT' and dyn_px < pos['current_sl']:
                    pos['current_sl'] = dyn_px
                elif pos['direction'] == 'LONG' and dyn_px > pos['current_sl']:
                    pos['current_sl'] = dyn_px

            # 5. Afternoon Anti-Scratch Ratchet at 13:30 (lock +0.60% if MFE >= +0.70%)
            if time_str >= "13:30" and not pos['ratchet_locked'] and pos['mfe_pct'] >= 0.70:
                pos['ratchet_locked'] = True
                lock_px = pos['entry_price'] * (1.0 - 0.0060) if pos['direction'] == 'SHORT' else pos['entry_price'] * (1.0 + 0.0060)
                if pos['direction'] == 'SHORT' and lock_px < pos['current_sl']: pos['current_sl'] = round(lock_px, 2)
                elif pos['direction'] == 'LONG' and lock_px > pos['current_sl']: pos['current_sl'] = round(lock_px, 2)
                logger.info(f"[AFTERNOON RATCHET] {pos['symbol']} locked +0.60% profit @ Rs {pos['current_sl']:.2f}")
                
            remaining.append(pos)
        self.active_positions = remaining

    def square_off_all(self, snapshots, reason, time_str):
        for pos in list(self.active_positions):
            snap = snapshots.get(pos['symbol'])
            exit_px = snap['close'] if snap else pos['current_price']
            self.exit_position(pos, exit_px, reason, time_str)
        self.active_positions.clear()

    def run_daily_postmarket_screener(self):
        logger.info("Executing daily post-market screener to prepare candidates for next trading day...")
        self.load_candidates()

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
# FASTAPI APP & DASHBOARD
# ------------------------------------------------------------------------------
app = FastAPI(title="Hybrid Fusion Champion Trader")

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

@app.get("/api/test-scan")
def test_scan():
    cand_map = {c['symbol']: c for c in engine.candidates}
    snapshots = {}
    for sym, c_data in list(cand_map.items())[:12]:
        isin = c_data.get('isin')
        if isin:
            bars = engine.fetch_live_1min_bars(isin)
            snap = engine.compute_intraday_snapshot(bars)
            if snap: snapshots[sym] = snap
                
    simulated_orders = []
    used_secs = set()
    for sym, snap in snapshots.items():
        pm = cand_map.get(sym)
        if pm and pm.get('ret_5d', 0) >= 5.0 and pm.get('atr10_pct', 99) <= 4.50:
            sec = pm.get('sector', 'N/A')
            if sec not in used_secs and len(simulated_orders) < 2:
                used_secs.add(sec)
                shares = max(1, int(engine.slot_exposure / snap['close']))
                sl = round(snap['close'] * (1.0 + 0.0140), 2)
                tp1 = round(snap['close'] * (1.0 - 0.0195), 2)
                tp2 = round(snap['close'] * (1.0 - 0.0340), 2)
                simulated_orders.append({
                    "symbol": sym,
                    "direction": "SHORT",
                    "strategy": "Tier #3A 5D Fade SHORT (Prime)",
                    "sector": sec,
                    "5d_ret": pm.get('ret_5d'),
                    "atr10": pm.get('atr10_pct'),
                    "test_price": snap['close'],
                    "vwap": snap['vwap'],
                    "shares": shares,
                    "exposure_inr": round(shares * snap['close'], 2),
                    "hard_sl_1_40pct": sl,
                    "tp1_1_95pct": tp1,
                    "tp2_3_40pct": tp2
                })
                    
    return {
        "status": "PASS",
        "evaluated_at": datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST"),
        "snapshots_processed": len(snapshots),
        "simulated_trades_count": len(simulated_orders),
        "simulated_orders": simulated_orders
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
        "today_pnl_inr": round(engine.today_pnl, 2)
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
        "logs": log_records[-50:]
    }

@app.get("/", response_class=HTMLResponse)
def dashboard():
    now_ist = datetime.datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S IST")
    pnl_color = "#10b981" if engine.today_pnl >= 0 else "#ef4444"
    
    # Active table rows
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
        active_html = "<tr><td colspan='9' style='text-align: center; padding: 25px; color: #64748b;'>No Active Open Positions (Waiting for market scans)</td></tr>"
        
    # Closed table rows
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

    # Multi-Day Turso Cloud History
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

    # Watchlist preview
    watch_html = ""
    for cand in engine.candidates[:10]:
        tier = "Tier 3A (Prime)" if cand.get('is_tier3a') else "Tier 3B"
        watch_html += f"""
        <tr style="border-bottom: 1px solid #334155;">
            <td style="padding: 8px; font-weight: bold;">{cand.get('symbol')}</td>
            <td style="padding: 8px;">{cand.get('sector')}</td>
            <td style="padding: 8px;">Rs {cand.get('today_close', cand.get('prev_close', 0)):.2f}</td>
            <td style="padding: 8px; color: #4ade80;">{cand.get('ret_5d', 0):+.2f}%</td>
            <td style="padding: 8px;">{cand.get('atr10_pct', 0):.2f}%</td>
            <td style="padding: 8px;"><span style="background: #1e293b; padding: 2px 6px; border-radius: 4px; font-size: 11px;">{tier}</span></td>
        </tr>
        """

    logs_str = "<br>".join(log_records[-30:])
    
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Hybrid Fusion Champion Live Cloud</title>
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
                    <h2 style="margin: 0;">🚀 Hybrid Fusion Champion Live Engine</h2>
                    <span style="color: #94a3b8; font-size: 13px;">Cloud Production Server (Render + Turso Cloud Multi-Day DB)</span>
                </div>
                <div style="text-align: right;">
                    <span style="background: #0284c7; padding: 5px 12px; border-radius: 20px; font-size: 12px;">{engine.status}</span>
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
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Kite Brokerage & Taxes Deducted</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">ACTIVE TRADING SLOTS</div>
                    <div class="stat-val">{len(engine.active_positions)} / 2 Slots</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">5x MIS Intraday Leverage</div>
                </div>
                <div class="stat-box">
                    <div style="color: #94a3b8; font-size: 12px;">SCREENED TARGETS</div>
                    <div class="stat-val">{len(engine.candidates)} Stocks</div>
                    <div style="color: #64748b; font-size: 11px; margin-top: 4px;">Ret_5D >= +3.5% (Nifty 500)</div>
                </div>
            </div>

            <div style="margin-bottom: 20px;">
                <a href="/api/test-feed" target="_blank" class="btn">🧪 Test Live Data Feed</a>
                <a href="/api/test-scan" target="_blank" class="btn" style="background: #059669;">🧪 Simulate 10:15 Scan</a>
                <a href="/api/state" target="_blank" class="btn" style="background: #475569;">📄 View Raw JSON State</a>
            </div>

            <div class="card">
                <h3 style="margin-top: 0; color: #38bdf8;">📊 Active Open Positions (Live 1-Minute Stream)</h3>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th><th>Type</th><th>Entry</th><th>Current</th><th>VWAP</th><th>SL</th><th>TP1</th><th>Net P&L</th><th>Strategy</th>
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
                <div style="font-size: 12px; color: #94a3b8; margin-bottom: 12px;">Every trade from every day is permanently recorded here. You can open this weeks or months later to view full performance.</div>
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
                            <th>Symbol</th><th>Sector</th><th>Close</th><th>5D Return</th><th>ATR10 %</th><th>Tier</th>
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
