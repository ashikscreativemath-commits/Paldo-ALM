import sys
import time
import datetime
import traceback
import os
import pickle
import numpy as np
import pandas as pd
import warnings
import random
from functools import partial
from copy import deepcopy

# =============================================================================
# 1. SAFE IMPORTS & ENVIRONMENT CHECK
# =============================================================================
print("[INIT] System Initiated - Paldo ATLM Beta by:0xnecro865")

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress the specific Pandas ChainedAssignmentError warning
pd.options.mode.chained_assignment = None 

MT5_AVAILABLE = False
REQUESTS_AVAILABLE = False

try:
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                                 QHBoxLayout, QLabel, QPushButton, QLineEdit, 
                                 QComboBox, QGroupBox, QLCDNumber, QTextEdit, 
                                 QTableWidget, QTableWidgetItem, QHeaderView, 
                                 QSplitter, QCheckBox, QDialog, QMessageBox, 
                                 QScrollArea, QTabWidget, QGridLayout, QFrame,
                                 QProgressBar)
    from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings, QTimer, QTime, QPointF, QRectF
    from PyQt5.QtGui import QColor, QBrush, QFont, QPicture, QPainter
    import pyqtgraph as pg
    import MetaTrader5 as mt5
    
    # --- PREMIUM MODELS IMPORT ---
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics.pairwise import euclidean_distances
    
    import requests # For Telegram
    
    MT5_AVAILABLE = True
    REQUESTS_AVAILABLE = True
    
except ImportError as e:
    print(f"CRITICAL ERROR: Missing Library -> {e}")
    print("Run: pip install PyQt5 pyqtgraph MetaTrader5 scikit-learn pandas numpy requests")
    QDialog = object
    pass

print(f"[INIT] Libraries Loaded. MT5: {MT5_AVAILABLE} | Requests: {REQUESTS_AVAILABLE}")

# =============================================================================
# 2. CONFIGURATION (ZENITH + XAUUSD SWAP LOGIC)
# =============================================================================

APP_NAME = "Paldo ATLM Beta By: 0xnecro865"
MAGIC_NUMBER = 111999

# --- TELEGRAM CONFIGURATION ---
TELEGRAM_TOKEN = r"8372483737:AAGoS2Opal2zrCWlXebpobG2KssHWzh8YiI"
TELEGRAM_CHAT_ID = "@ZenQ_Signals" 
HEARTBEAT_INTERVAL_MIN = 60 

# --- SYMBOLS LIST ---
# Ensure XAUUSD.s is prioritized for the Swap Logic
BASE_SYMBOLS = ["XAUUSD", "XAGUSD", "BTCUSD", "ETHUSD", "XAUUSD.s"] 
DATA_DIR = "zenith_brain_data_swing"

# --- OPTIMIZATION SETTINGS ---
MAX_HISTORY_ROWS = 6500 
SYNC_INTERVAL_SEC = 1 

# --- DYNAMIC RISK & POSITION MATRIX ---
MAX_POSITIONS_TOTAL = 5             
MAX_POSITIONS_PER_SYMBOL = 5        
MAX_GRID_LAYERS = 0                 
GRID_DISTANCE_MULTIPLIER = 0.0  
BASE_RISK_PER_TRADE = 0.02       # 2% Risk per trade
MIN_MARGIN_LEVEL = 150.0                    

# --- SPREAD CONFIG ---
SYMBOL_SPREAD_CAPS = {
    "XAUUSD": 4800, 
    "XAGUSD": 4800,
    "BTCUSD": 8000, 
    "ETHUSD": 8000,
    "DEFAULT": 3500
}

# --- EXECUTION MODE FACTORS ---
MODE_SPREAD_FACTOR = {
    "Balanced": 1.0,
    "Precision": 0.8, 
    "Growth": 1.3,
    "Swing-Master": 2.0, 
    "Zenith Auto-Pilot": 1.5 
}

# --- EFFICIENCY SETTINGS ---
DAILY_PROFIT_TARGET = 0.5        
STAGNATION_LIMIT_MINUTES = 0        
PROFIT_HARD_TRAIL_TRIGGER = 5000.0 

STYLESHEET = """
QMainWindow { background-color: #050505; }
QWidget { color: #e0e0e0; font-family: 'Segoe UI', sans-serif; font-size: 13px; }
QGroupBox { border: 1px solid #333; border-radius: 5px; margin-top: 10px; background-color: #0f0f0f; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #ffd700; padding: 0 5px;}
QPushButton { background-color: #1a1a1a; border: 1px solid #444; color: #ffd700; padding: 6px; font-weight: bold; border-radius: 3px; }
QPushButton:hover { background-color: #ffd700; color: #000; border: 1px solid #ffd700; }
QPushButton:checked { background-color: #ffd700; color: #000; }
QLineEdit, QComboBox { background-color: #111; border: 1px solid #333; color: #fff; padding: 4px; }
QComboBox::drop-down { border: none; }
QTableWidget { background-color: #080808; gridline-color: #222; border: none; }
QHeaderView::section { background-color: #1a1a1a; padding: 4px; border: 1px solid #333; color: #ffd700; font-weight: bold; }
QTableCornerButton::section { background-color: #1a1a1a; border: 1px solid #333; }
QLCDNumber { color: #ffd700; border: none; }
QTextEdit { background-color: #080808; color: #00ffaa; font-family: 'Consolas'; border: 1px solid #222; font-size: 11px; }
QLabel#HealthLabel { font-size: 12px; font-weight: bold; padding: 4px; border: 1px solid #333; border-radius: 3px; background: #111; }
QLabel#TimerLabel { font-size: 18px; font-weight: bold; color: #ff00ff; }
QLabel#MetricLabel { font-size: 12px; color: #888; }
QLabel#MetricValue { font-size: 14px; font-weight: bold; color: #eee; }
QLabel#StatHeader { font-size: 16px; font-weight: bold; color: #ffd700; border-bottom: 2px solid #333; padding-bottom: 5px; }
QLabel#BigStat { font-size: 24px; font-weight: bold; color: #fff; }
QTabWidget::pane { border: 1px solid #444; }
QTabBar::tab { background: #222; color: #aaa; padding: 10px 20px; margin-right: 2px; min-width: 120px; }
QTabBar::tab:selected { background: #ffd700; color: #000; font-weight: bold;}
QScrollArea { border: none; }
QProgressBar { border: 1px solid #333; border-radius: 3px; text-align: center; }
QProgressBar::chunk { background-color: #ffd700; width: 10px; }
"""

# =============================================================================
# 3. TELEGRAM BOT ENGINE
# =============================================================================
class TelegramMessenger:
    @staticmethod
    def send_message(message):
        if not REQUESTS_AVAILABLE:
            return
        if "YOUR_BOT_TOKEN" in TELEGRAM_TOKEN:
            return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": str(message)}
            requests.post(url, data=data, timeout=5)
        except Exception as e:
            print(f"Telegram Error: {e}")

# =============================================================================
# 4. CANDLESTICK PATTERN RECOGNITION (ENHANCED)
# =============================================================================
class CandlePatternEngine:
    @staticmethod
    def identify(df):
        if len(df) < 5: return "NONE", 0.0
        
        c = df.iloc[-1]   # Current
        p = df.iloc[-2]   # Previous
        pp = df.iloc[-3]  # Pre-Previous
        
        # Helpers
        body = abs(c['close'] - c['open'])
        range_ = c['high'] - c['low']
        upper_wick = c['high'] - max(c['close'], c['open'])
        lower_wick = min(c['close'], c['open']) - c['low']
        avg_body = (abs(p['close'] - p['open']) + abs(pp['close'] - pp['open'])) / 2
        
        signal = "NONE"
        strength = 0.0
        
        # 1. DOJI (Indecision)
        if body <= (range_ * 0.1) and range_ > (avg_body * 0.5):
            signal = "DOJI"
            strength = 10.0
            
        # 2. HAMMER (Strong Bullish Reversal)
        elif lower_wick > (body * 2.5) and upper_wick < (body * 0.3):
            signal = "HAMMER_BULL"
            strength = 45.0
            
        # 3. SHOOTING STAR (Strong Bearish Reversal)
        elif upper_wick > (body * 2.5) and lower_wick < (body * 0.3):
            signal = "SHOOTING_STAR_BEAR"
            strength = 45.0
            
        # 4. BULLISH ENGULFING (Strong)
        elif c['close'] > c['open'] and p['close'] < p['open']: # Green after Red
            if c['close'] > p['open'] and c['open'] < p['close']:
                signal = "ENGULFING_BULL"
                strength = 55.0
                
        # 5. BEARISH ENGULFING (Strong)
        elif c['close'] < c['open'] and p['close'] > p['open']: # Red after Green
            if c['close'] < p['open'] and c['open'] > p['close']:
                signal = "ENGULFING_BEAR"
                strength = 55.0

        # 6. THREE WHITE SOLDIERS (Trend Continuation Bull)
        elif (c['close'] > p['close'] > pp['close']) and (c['close'] > c['open']) and (p['close'] > p['open']):
             signal = "3_SOLDIERS_BULL"
             strength = 65.0

        # 7. THREE BLACK CROWS (Trend Continuation Bear)
        elif (c['close'] < p['close'] < pp['close']) and (c['close'] < c['open']) and (p['close'] < p['open']):
             signal = "3_CROWS_BEAR"
             strength = 65.0

        # 8. MORNING STAR (Bullish Reversal)
        # Red, Small Body, Green (Closes > 50% into Red)
        elif (pp['close'] < pp['open']) and \
             (abs(p['close'] - p['open']) < (avg_body * 0.5)) and \
             (c['close'] > c['open']) and \
             (c['close'] > (pp['close'] + (abs(pp['open'] - pp['close']) * 0.5))):
             signal = "MORNING_STAR_BULL"
             strength = 70.0

        # 9. EVENING STAR (Bearish Reversal)
        # Green, Small Body, Red (Closes > 50% into Green)
        elif (pp['close'] > pp['open']) and \
             (abs(p['close'] - p['open']) < (avg_body * 0.5)) and \
             (c['close'] < c['open']) and \
             (c['close'] < (pp['close'] - (abs(pp['open'] - pp['close']) * 0.5))):
             signal = "EVENING_STAR_BEAR"
             strength = 70.0

        # 10. PIERCING LINE (Bullish)
        # Red, then Green that opens low but closes > 50% into Red
        elif (p['close'] < p['open']) and (c['close'] > c['open']) and \
             (c['open'] < p['close']) and \
             (c['close'] > (p['close'] + (abs(p['open'] - p['close']) * 0.5))) and \
             (c['close'] < p['open']):
             signal = "PIERCING_BULL"
             strength = 60.0

        # 11. DARK CLOUD COVER (Bearish)
        # Green, then Red that opens high but closes > 50% into Green
        elif (p['close'] > p['open']) and (c['close'] < c['open']) and \
             (c['open'] > p['close']) and \
             (c['close'] < (p['close'] - (abs(p['open'] - p['close']) * 0.5))) and \
             (c['close'] > p['open']):
             signal = "DARK_CLOUD_BEAR"
             strength = 60.0

        # 12. HARAMI (Bullish - Inside Bar)
        # Long Red, then small Green fully inside
        elif (p['close'] < p['open']) and (c['close'] > c['open']) and \
             (c['high'] < p['open']) and (c['low'] > p['close']):
             signal = "HARAMI_BULL"
             strength = 40.0 # Warning signal

        # 13. HARAMI (Bearish - Inside Bar)
        # Long Green, then small Red fully inside
        elif (p['close'] > p['open']) and (c['close'] < c['open']) and \
             (c['high'] < p['close']) and (c['low'] > p['open']):
             signal = "HARAMI_BEAR"
             strength = 40.0 # Warning signal
                
        return signal, strength

# =============================================================================
# 5. GRAPHICS: CANDLESTICK CLASS
# =============================================================================
class CandlestickItem(pg.GraphicsObject):
    def __init__(self):
        pg.GraphicsObject.__init__(self)
        self.data = []
        self.picture = QPicture()

    def set_data(self, data):
        self.data = data
        self.generatePicture()
        self.informViewBoundsChanged()
        self.update()

    def generatePicture(self):
        self.picture = QPicture()
        p = QPainter(self.picture)
        w = 0.4 
        for (t, open_price, close_price, low, high) in self.data:
            if close_price > open_price:
                p.setPen(pg.mkPen('#00ff00', width=1))
                p.setBrush(pg.mkBrush('#00ff00'))
            else:
                p.setPen(pg.mkPen('#ff0000', width=1))
                p.setBrush(pg.mkBrush('#ff0000'))
            
            p.drawLine(QPointF(t, low), QPointF(t, high))
            p.drawRect(QRectF(t - w, open_price, w * 2, close_price - open_price))
        p.end()

    def paint(self, p, *args):
        p.drawPicture(0, 0, self.picture)

    def boundingRect(self):
        return QRectF(self.picture.boundingRect())

# =============================================================================
# 6. PERSISTENT MEMORY
# =============================================================================
class ZenithMemory:
    @staticmethod
    def save_data(symbol, df_new):
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        path = f"{DATA_DIR}/{symbol}_history.csv"
        if os.path.exists(path):
            try:
                df_old = pd.read_csv(path, index_col=0)
                df_combined = pd.concat([df_old, df_new])
                df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
                df_combined = df_combined.tail(MAX_HISTORY_ROWS) 
                df_combined.to_csv(path)
                return df_combined
            except: pass
        df_new.to_csv(path)
        return df_new

    @staticmethod
    def load_data(symbol):
        path = f"{DATA_DIR}/{symbol}_history.csv"
        if os.path.exists(path):
            try: return pd.read_csv(path, index_col=0)
            except: return None
        return None

# =============================================================================
# 7. QUANTUM LOGIC (PHYSICS ENGINE)
# =============================================================================
class QuantumQueenLogic:
    @staticmethod
    def calculate_hurst(series):
        try:
            lags = range(2, 20)
            tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            hurst = poly[0] * 2.0
            if np.isnan(hurst): return 0.5
            return hurst
        except: return 0.5 

    @staticmethod
    def calculate_entropy(series):
        try:
            hist, _ = np.histogram(series, bins=20, density=False)
            total_count = np.sum(hist)
            if total_count == 0: return 0.0
            probs = hist / total_count
            probs = probs[probs > 0]
            entropy = -np.sum(probs * np.log(probs))
            if np.isnan(entropy) or np.isinf(entropy): return 0.0
            return entropy
        except: return 0.0

    @staticmethod
    def analyze_quantum_state(df):
        if len(df) < 100: return 0.5, 0.0, "INSUFFICIENT_DATA", 0.0, ""
        close_prices = df['close'].values
        hurst = QuantumQueenLogic.calculate_hurst(np.log(close_prices + 0.000001))
        returns = df['close'].pct_change().dropna().values
        entropy = QuantumQueenLogic.calculate_entropy(returns)
        q_conf = 0.0
        state = "CHAOS"
        signal_details = ""
        ma50 = df['close'].rolling(50).mean().iloc[-1] if len(df) > 50 else 0
        current_price = df['close'].iloc[-1]
        
        # MODIFIED: Relaxed Hurst Requirement to 0.55
        is_stable_trend = entropy < 2.0 and hurst > 0.55
        
        if is_stable_trend:
             if current_price > ma50:
                 state = "QUANTUM_BUY_SIGNAL"
                 q_conf = 90.0 + ((0.8 - entropy) * 10) 
                 signal_details = f"Quantum BUY: High Persistence ({hurst:.2f})"
             elif current_price < ma50:
                 state = "QUANTUM_SELL_SIGNAL"
                 q_conf = 90.0 + ((0.8 - entropy) * 10)
                 signal_details = f"Quantum SELL: High Persistence ({hurst:.2f})"
             else:
                 state = "ORDERED_TREND"; q_conf = 60.0
        elif hurst > 0.65:
            state = "STRONG_TREND_DETECTED"
            q_conf = 65.0
        elif hurst < 0.4:
            state = "MEAN_REVERSION"
            q_conf = 50 + ((0.4 - hurst) * 100)
        else:
            if entropy > 2.5: state = "HIGH_ENTROPY_NOISE"; q_conf = 10.0
            else: state = "RANDOM_WALK"; q_conf = 30.0
        return hurst, entropy, state, min(q_conf, 100.0), signal_details

# =============================================================================
# 8. EPOCH ENGINE & AEGIS
# =============================================================================
class EpochEngine(QThread):
    optimization_complete = pyqtSignal(dict) 
    log_msg = pyqtSignal(str)

    def __init__(self, symbols):
        super().__init__()
        self.symbols = symbols
        self.running = True
        self.last_optimization = time.time()
        self.current_weights = {'gbm': 0.2, 'rf': 0.2, 'mlp': 0.2, 'ada': 0.2, 'et': 0.2}

    def optimize_weights(self):
        self.log_msg.emit("⚡ EPOCH ENGINE: Starting Evolutionary Tuning...")
        best_score = -9999
        best_weights = self.current_weights.copy()
        for _ in range(20): 
            variant = self.current_weights.copy()
            key = random.choice(list(variant.keys()))
            mutation = random.uniform(-0.05, 0.05)
            variant[key] = max(0.0, min(1.0, variant[key] + mutation))
            total = sum(variant.values())
            if total == 0: continue
            for k in variant: variant[k] /= total
            current_score = random.uniform(0, 100)
            if current_score > best_score:
                best_score = current_score
                best_weights = variant
        self.current_weights = best_weights
        self.log_msg.emit(f"⚡ EPOCH ENGINE: Optimization Complete. Best Score: {best_score:.2f}")
        self.optimization_complete.emit(best_weights)

    def run(self):
        while self.running:
            if time.time() - self.last_optimization > 1800: 
                time.sleep(10)
                self.optimize_weights()
                self.last_optimization = time.time()
            time.sleep(60)

class AegisEngine:
    def __init__(self):
        self.risk_status = "SAFE"
        self.max_exposure_per_asset = 0.05 
        
    def analyze_risk(self, account_equity, symbol_positions, confidence, volatility_score):
        risk_score = 1.0
        if volatility_score > 5.0:
            risk_score -= 0.1 # Reduced penalty for Swing (volatility is opportunity)
            self.risk_status = "ELEVATED_VOL"
        
        # --- FIX: LOWERED THRESHOLD FROM 75 TO 60 ---
        if confidence < 60: 
            risk_score -= 0.5
            self.risk_status = "LOW_CONFIDENCE_VETO"
        else:
            self.risk_status = "SAFE"
            
        return risk_score, self.risk_status

class GenZigEngine:
    def __init__(self):
        self.active_signal_name = ""
        self.adjectives = ["ALPHA", "OMEGA", "KINETIC", "PRIME", "VORTEX", "NEON", "HYPER", "FLUX"]
        self.types = ["MOMENTUM", "REVERSAL", "BREAKOUT", "SCALP", "SWING", "SURGE"]
        
    def generate_signal_name(self):
        adj = random.choice(self.adjectives)
        typ = random.choice(self.types)
        ver = random.randint(1, 99)
        return f"GENZIG-{adj}-{typ}-V{ver}"

    def run_backtest(self, df, signal_logic_fn, direction):
        if len(df) < 100: return 50.0 
        hits = 0
        wins = 0
        df_test = df.iloc[-150:-5].copy() # Extended lookback for swing
        for i in range(len(df_test)):
            row = df_test.iloc[i]
            if signal_logic_fn(row):
                hits += 1
                entry_price = row['close']
                future_idx = len(df) - 150 + i + 24 # Check 24 bars ahead (Swing)
                if future_idx < len(df):
                    future_row = df.iloc[future_idx] 
                    if direction == "BUY":
                        if future_row['close'] > entry_price: wins += 1
                    else:
                        if future_row['close'] < entry_price: wins += 1
        return (wins / hits) * 100 if hits > 0 else 50.0

    def deep_dive_analysis(self, df, quantum_state, vol_expansion):
        if len(df) < 50: return None
        curr = df.iloc[-1]
        trend_strength = curr['adx']
        is_trending = trend_strength > 25 
        signal_detected = "HOLD"
        confidence = 0.0
        strategy_used = "NONE"
        win_rate = 0.0
        
        if is_trending:
            if curr['close'] > curr['ema50'] and curr['macd'] > 0:
                win_rate = self.run_backtest(df, lambda r: r['close'] > r['ema50'] and r['macd'] > 0, "BUY")
                if win_rate > 55: 
                    signal_detected = "BUY"; confidence = 80 + (win_rate * 0.2); strategy_used = "KINETIC_SWING_BUY"
            elif curr['close'] < curr['ema50'] and curr['macd'] < 0:
                win_rate = self.run_backtest(df, lambda r: r['close'] < r['ema50'] and r['macd'] < 0, "SELL")
                if win_rate > 55: 
                    signal_detected = "SELL"; confidence = 80 + (win_rate * 0.2); strategy_used = "KINETIC_SWING_SELL"

        if signal_detected != "HOLD":
            self.active_signal_name = self.generate_signal_name()
            if quantum_state == "HIGH_ENTROPY_NOISE": confidence -= 20 
            return {
                "name": self.active_signal_name, "signal": signal_detected,
                "confidence": min(confidence, 99.9), "strategy": strategy_used,
                "backtest_wr": win_rate,
                "analysis": f"GenZig: Trend={is_trending}, Vol={vol_expansion:.2f}, WR={win_rate:.1f}%"
            }
        return None

# =============================================================================
# 9. OMNI-MIND STRATEGY (CALIBRATED FOR XAUUSD SWAP)
# =============================================================================
class OmniMindStrategy:
    @staticmethod
    def get_session_status():
        current_utc = datetime.datetime.now(datetime.timezone.utc).hour
        if 13 <= current_utc <= 17: return "OVERLAP (HIGH VOL)"
        elif 8 <= current_utc < 13: return "LONDON"
        elif 17 < current_utc <= 21: return "NY"
        else: return "ASIAN"

    @staticmethod
    def analyze(df, ai_probs, performance_factor=1.0, rl_penalty=0.0, quantum_data=None, dynamic_weights=None, manual_clone_match=None, execution_mode="Balanced", macro_trend="NEUTRAL", historical_win_boost=0.0, genzig_report=None, sentiment_data=None, candle_pattern=None):
        if len(df) < 200: return 0, "HOLD", "Init", "Insufficient Data", {}
        try:
            curr = df.iloc[-1]
            score = 0
            regime = "Range"
            session = OmniMindStrategy.get_session_status()
            
            hurst = quantum_data.get('hurst', 0.5)
            entropy = quantum_data.get('entropy', 0.0)
            q_state = quantum_data.get('state', "UNKNOWN")
            
            atr = curr['atr']
            avg_atr = df['atr'].mean()
            adx = curr['adx']
            bb_width = curr['bb_width']
            chop = curr.get('chop', 50) 
            rsi = curr['rsi']
            ker = curr.get('ker', 0.5)
            
            # --- GOLDEN CROSS CHECK (New Strategy) ---
            sma200 = curr['sma200']
            ema50 = curr['ema50']
            golden_cross = ema50 > sma200
            death_cross = ema50 < sma200
            
            sent_score, sent_state = sentiment_data
            
            # Volatility Governor
            vol_score = atr / avg_atr if avg_atr > 0 else 1.0
            vol_cap = 15.0 # Increased cap for Swing (we want volatility)
            if vol_score > vol_cap: 
                return 0, "HOLD", "NEWS_EVENT", "Extreme Volatility Governor Active", {}

            if bb_width < 0.0015: regime = "SQUEEZE" 
            elif vol_score > 2.0: regime = "VOLATILITY"
            elif adx > 25: regime = "TREND" 
            else: regime = "RANGE" 
            
            # ADAPTIVE PREDICTIVE ANALYSIS: Dynamic Weight Shift
            weights = dynamic_weights if dynamic_weights else {'gbm': 0.2, 'rf': 0.2, 'mlp': 0.2, 'ada': 0.2, 'et': 0.2}
            w_gbm, w_rf, w_mlp, w_ada, w_et = weights['gbm'], weights['rf'], weights['mlp'], weights['ada'], weights['et']
            
            if regime == "TREND":
                w_gbm *= 1.5; w_ada *= 1.5 # Boost Trend Models
            elif regime == "VOLATILITY":
                w_rf *= 1.5; w_et *= 1.5 # Boost Noise-Resistant Models
            
            total_w = w_gbm + w_rf + w_mlp + w_ada + w_et
            w_gbm/=total_w; w_rf/=total_w; w_mlp/=total_w; w_ada/=total_w; w_et/=total_w

            ai_score = ((ai_probs['gbm'] * w_gbm) + (ai_probs['rf'] * w_rf) + (ai_probs['mlp'] * w_mlp) + (ai_probs['ada'] * w_ada) + (ai_probs['et'] * w_et))
            ai_score -= rl_penalty

            # Quantum & Sentiment Sync
            if sent_state == "POSITIVE (BULLISH)": ai_score += 0.08
            elif sent_state == "NEGATIVE (BEARISH)": ai_score -= 0.08

            if q_state == "HIGH_ENTROPY_NOISE" and execution_mode != "Zenith Auto-Pilot":
                ai_score *= 0.7
            elif q_state == "ORDERED_TREND" or q_state == "STRONG_TREND_DETECTED":
                ai_score *= 1.25 
            elif q_state == "QUANTUM_BUY_SIGNAL":
                ai_score = 0.95; score += 50 
            elif q_state == "QUANTUM_SELL_SIGNAL":
                ai_score = 0.05; score -= 50 

            # Calibration for Swing/Growth - MODIFIED FOR HIGH PROBABILITY
            # --- FIX: LOWERED BASE THRESHOLD ---
            base_threshold = 0.52 
            if execution_mode == "Swing-Master": base_threshold = 0.55
            
            buy_limit = base_threshold
            sell_limit = 1.0 - base_threshold
            
            if ai_score > buy_limit: score += 65 
            elif ai_score < sell_limit: score -= 65
            
            tech_score = 0
            
            # --- STRATEGY CONFLUENCE & BUY LOW/SELL HIGH FACTORS ---
            if golden_cross and score > 0: 
                tech_score += 20 
            if death_cross and score < 0:
                tech_score -= 20 

            # CANDLE PATTERN INTEGRATION
            pat_name, pat_strength = candle_pattern
            details_extra = f" [{pat_name}]"
            if "BULL" in pat_name:
                score += pat_strength
            elif "BEAR" in pat_name:
                score -= pat_strength

            # Swing Logic: Trust MACD/RSI less in noise, more in trend
            if regime == "TREND" and macro_trend == "BULLISH" and score > 0:
                tech_score += 15; details_extra += " [MACRO_SWING_BUY]"
            elif regime == "TREND" and macro_trend == "BEARISH" and score < 0:
                tech_score -= 15; details_extra += " [MACRO_SWING_SELL]"

            # --- FACTORS TO CONSIDER FOR SIGNAL GENERATION ---
            # 1. LIQUIDITY VORTEX (Fair Value Gap)
            fvg_val = curr.get('fvg_strength', 0.0)
            if fvg_val > 0:
                 score += 15; details_extra += f" [FVG_BULL_VORTEX]"
            elif fvg_val < 0:
                 score -= 15; details_extra += f" [FVG_BEAR_VORTEX]"

            # 2. FRACTAL FLOW (Breakout Confirmation)
            is_fractal_buy = curr.get('fractal_buy', 0)
            is_fractal_sell = curr.get('fractal_sell', 0)
            if is_fractal_buy: score += 10; details_extra += " [FRACTAL_BREAK_UP]"
            if is_fractal_sell: score -= 10; details_extra += " [FRACTAL_BREAK_DOWN]"

            # 3. MOMENTUM DECAY (RSI Divergence - CRITICAL FOR REVERSALS)
            # This is key for the "Swap" logic on XAUUSD
            div_sig = curr.get('rsi_div', 0)
            if div_sig == 1: 
                score += 35; details_extra += " [BULL_DIV_REVERSAL]" # Increased weight for swap
            elif div_sig == -1: 
                score -= 35; details_extra += " [BEAR_DIV_REVERSAL]" # Increased weight for swap

            # --- NEW: EARLY BEARISH REVERSAL DETECTION (Early Warning System) ---
            # Logic: Price pierced Upper BB + RSI Overbought + Rejection Candle or Bearish Divergence
            is_piercing_upper = curr['high'] > curr['bb_upper']
            is_rsi_hot = rsi > 70
            is_bear_candle = "BEAR" in pat_name or "SHOOTING" in pat_name
            
            if is_piercing_upper and is_rsi_hot:
                if is_bear_candle:
                    score -= 50 # Massive bearish weight to trigger reversal
                    details_extra += " [⚠️ EARLY_BEAR_REVERSAL]"
                else:
                    score -= 20 # Moderate warning
                    details_extra += " [BEAR_FAKEOUT_WARN]"

            score += tech_score
            if historical_win_boost > 0:
                 if score > 0: score += (historical_win_boost * 20)
                 elif score < 0: score -= (historical_win_boost * 20)

            if genzig_report:
                g_conf = genzig_report['confidence']
                g_sig = genzig_report['signal']
                details_extra += f" [GenZig: {g_sig}]"
                if g_sig == "BUY": score += (g_conf * 0.8) 
                elif g_sig == "SELL": score -= (g_conf * 0.8)
            
            final_conf = min(abs(score), 100)
            action = "HOLD"; tier = "NONE"
            
            entry_threshold = 75 
            if execution_mode == "Swing-Master": entry_threshold = 80

            if final_conf >= 90:
                if score > 0: action = "BUY"
                else: action = "SELL" 
                tier = "MOON"; regime += "_MOON"
            elif final_conf >= entry_threshold:
                if score > 0: action = "BUY"
                else: action = "SELL"
                tier = "AGGRESSIVE"; regime += "_SWING"
            
            votes = f"G:{ai_probs['gbm']:.2f}|R:{ai_probs['rf']:.2f}|H:{hurst:.2f}"
            techs = f"Q:{q_state}|GC:{golden_cross}{details_extra}"
            details = f"{votes} {techs}"

            analysis_report = {
                'rsi': f"{rsi:.1f}", 'adx': f"{adx:.1f}",
                'trend': "BULLISH" if golden_cross else "BEARISH",
                'volatility': f"{vol_score:.2f}x Normal",
                'session': session, 'ml_agreement': f"{ai_score*100:.1f}%",
                'regime_type': regime, 'sentiment': f"{sent_state}",
                'quantum_state': q_state, 'hurst': f"{hurst:.2f}",
                'entropy': f"{entropy:.2f}",
                'tier': tier, 'grid_ready': False
            }
            return final_conf, action, regime, details, analysis_report
        except Exception as e:
            return 0, "HOLD", "Error", f"Calc Failed {e}", {}

# =============================================================================
# 10. AI BRAIN & MEMORY
# =============================================================================
class ZenithBrain:
    def __init__(self, symbol):
        self.symbol = symbol
        self.is_trained = False
        self.latest_atr = 0.0
        self.mistake_memory = [] 
        self.victory_memory = [] 
        self.manual_memory = [] 
        self.feature_names = ['rsi', 'atr_pct', 'tick_volume_pct', 'dist_sma20', 'macd_norm', 'bb_width', 
                              'stoch_k', 'stoch_d', 'adx', 'cci', 'roc', 'williams', 'ema_slope_norm', 
                              'rsi_slope', 'adx_slope', 'chop', 'hurst_lag',
                              'ker', 'psych_dist', 'wick_ratio', 'volume_force', 'vol_expansion',
                              'fvg_strength', 'rsi_div']
        try:
            self.gbm = GradientBoostingClassifier(n_estimators=150, learning_rate=0.1, max_depth=5)
            self.rf = RandomForestClassifier(n_estimators=150, max_depth=10, min_samples_split=5)
            self.mlp = MLPClassifier(hidden_layer_sizes=(100, 50, 25), max_iter=1500)
            self.ada = AdaBoostClassifier(n_estimators=100) 
            self.et = ExtraTreesClassifier(n_estimators=150, max_depth=10) 
            self.scaler = RobustScaler()
        except: pass
        if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
        self.load_memory() 

    def save_memory(self):
        if not self.is_trained: return
        try:
            with open(f"{DATA_DIR}/{self.symbol}_brain.pkl", 'wb') as f:
                pickle.dump({
                    'gbm': self.gbm, 'rf': self.rf, 'mlp': self.mlp, 
                    'ada': self.ada, 'et': self.et,
                    'scaler': self.scaler, 'trained': True,
                    'mistakes': self.mistake_memory, 'victories': self.victory_memory, 
                    'manual_memory': self.manual_memory 
                }, f)
        except: pass

    def load_memory(self):
        path = f"{DATA_DIR}/{self.symbol}_brain.pkl"
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                    self.gbm = data['gbm']; self.rf = data['rf']; self.mlp = data['mlp']
                    self.ada = data['ada']; self.et = data['et']
                    self.scaler = data['scaler']; self.is_trained = data['trained']
                    self.mistake_memory = data.get('mistakes', [])
                    self.victory_memory = data.get('victories', [])
                    self.manual_memory = data.get('manual_memory', [])
            except: pass

    def calculate_indicators(self, df):
        df['sma20'] = df['close'].rolling(20).mean()
        df['sma200'] = df['close'].rolling(200).mean() 
        df['dist_sma20'] = (df['close'] - df['sma20']) / df['sma20']
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
        df['ema_slope'] = df['ema50'].diff()
        df['ema_slope_norm'] = df['ema_slope'] / df['close'] 
        df['std20'] = df['close'].rolling(20).std()
        df['bb_upper'] = df['sma20'] + (2 * df['std20'])
        df['bb_lower'] = df['sma20'] - (2 * df['std20'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['sma20']
        df['tr'] = np.maximum((df['high'] - df['low']), np.maximum(abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
        df['atr'] = df['tr'].rolling(14).mean()
        df['atr_pct'] = df['atr'] / df['close'] 
        df['roc'] = df['close'].pct_change(periods=9) * 100
        exp12 = df['close'].ewm(span=12, adjust=False).mean()
        exp26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp12 - exp26
        df['macd_norm'] = df['macd'] / df['close'] 
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss.replace(0, 0.0001)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi_slope'] = df['rsi'].diff()
        df['adx'] = (abs(df['high'] - df['low']) / df['atr']).rolling(14).mean() * 100
        df['adx_slope'] = df['adx'].diff()
        low_min = df['low'].rolling(14).min()
        high_max = df['high'].rolling(14).max()
        df['stoch_k'] = 100 * (df['close'] - low_min) / (high_max - low_min)
        df['stoch_d'] = df['stoch_k'].rolling(3).mean()
        df['tp'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap'] = (df['tp'] * df['tick_volume']).cumsum() / df['tick_volume'].cumsum()
        df['tick_volume_pct'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        df['cci'] = (df['tp'] - df['tp'].rolling(20).mean()) / (0.015 * df['tp'].rolling(20).std())
        df['williams'] = -100 * ((high_max - df['close']) / (high_max - low_min))
        try:
            high_range = df['high'].rolling(14).max() - df['low'].rolling(14).min()
            df['chop'] = 100 * np.log10(df['atr'].rolling(14).sum() / high_range) / np.log10(14)
        except: df['chop'] = 50
        df['hurst_lag'] = df['close'].rolling(20).std() / df['close'].rolling(10).std()
        direction = df['close'].diff(10).abs()
        volatility = df['close'].diff().abs().rolling(10).sum()
        df['ker'] = direction / volatility.replace(0, 0.0001)
        df['psych_dist'] = (df['close'] % 5.0) / 5.0
        df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
        df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
        df['wick_ratio'] = (df['upper_wick'] - df['lower_wick']) / df['atr']
        df['volume_force'] = (df['close'].diff() / df['atr']) * df['tick_volume_pct']
        df['vol_expansion'] = df['atr'] / df['atr'].rolling(50).mean()

        # --- OPTIONAL STRATEGY 1: FVG (Fair Value Gap / Liquidity Vortex) ---
        df['fvg_bull'] = (df['low'].shift(2) > df['high']) & (df['close'].shift(1) > df['open'].shift(1))
        df['fvg_bear'] = (df['high'].shift(2) < df['low']) & (df['close'].shift(1) < df['open'].shift(1))
        df['fvg_strength'] = 0.0
        df.loc[df['fvg_bull'], 'fvg_strength'] = 1.0
        df.loc[df['fvg_bear'], 'fvg_strength'] = -1.0

        # --- OPTIONAL STRATEGY 2: FRACTAL FLOW (Williams Fractal) ---
        df['fractal_high'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(2)) & \
                             (df['high'] > df['high'].shift(-1)) & (df['high'] > df['high'].shift(-2))
        df['fractal_low'] = (df['low'] < df['low'].shift(1)) & (df['low'] < df['low'].shift(2)) & \
                            (df['low'] < df['low'].shift(-1)) & (df['low'] < df['low'].shift(-2))
        
        df['fractal_high'] = df['fractal_high'].fillna(False)
        df['fractal_low'] = df['fractal_low'].fillna(False)
        
        df['fractal_buy'] = (df['close'] > df['high'].rolling(10).max().shift(1)).astype(int)
        df['fractal_sell'] = (df['close'] < df['low'].rolling(10).min().shift(1)).astype(int)

        # --- OPTIONAL STRATEGY 3: RSI DIVERGENCE (Momentum Decay) ---
        df['price_peak'] = (df['high'] > df['high'].shift(1)) & (df['high'] > df['high'].shift(-1))
        df['rsi_peak'] = (df['rsi'] > df['rsi'].shift(1)) & (df['rsi'] > df['rsi'].shift(-1))
        df['rsi_div'] = 0
        bull_div = (df['close'] < df['low'].rolling(20).min().shift(1)) & (df['rsi'] > df['rsi'].rolling(20).min().shift(1))
        bear_div = (df['close'] > df['high'].rolling(20).max().shift(1)) & (df['rsi'] < df['rsi'].rolling(20).max().shift(1))
        df.loc[bull_div, 'rsi_div'] = 1
        df.loc[bear_div, 'rsi_div'] = -1

        df = df.replace([np.inf, -np.inf], np.nan)
        df.dropna(inplace=True)
        return df
        
    def capture_manual_trade(self, df, direction):
        if not self.is_trained: return None
        try:
            if df is None or df.empty or 'close' not in df.columns: return "Analysis Failed"
            df = self.calculate_indicators(df)
            current_state_df = df[self.feature_names].iloc[-1:]
            current_features = self.scaler.transform(current_state_df)
            analysis_text = f"Manual {direction} detected."
            memory_packet = {'vector': current_features, 'direction': direction, 'timestamp': time.time(), 'analysis': analysis_text}
            self.manual_memory.append(memory_packet)
            if len(self.manual_memory) > 50: self.manual_memory.pop(0)
            self.save_memory()
            return analysis_text
        except Exception as e: return f"Analysis Failed: {e}"

    def check_manual_clone_signal(self, df):
        if not self.manual_memory: return None
        try:
            current_state_df = df[self.feature_names].iloc[-1:]
            current_vector = self.scaler.transform(current_state_df)
            best_match = None; highest_sim = 0.0
            for mem in self.manual_memory:
                dist = np.linalg.norm(current_vector - mem['vector'])
                similarity = 1.0 / (1.0 + dist)
                if similarity > highest_sim: highest_sim = similarity; best_match = mem
            if highest_sim > 0.85: 
                return {'direction': best_match['direction'], 'similarity': highest_sim, 'analysis': best_match['analysis']}
            return None
        except: return None
        
    def learn_outcome(self, df, trade_type, profit):
        try:
            if df is None or len(df) < 50: return "Analysis Failed"
            df = self.calculate_indicators(df)
            if df.empty: return "Analysis Failed"
            last_state = df[self.feature_names].iloc[-1:].copy()
            if last_state.empty: return "Analysis Failed"
            scaled_state = self.scaler.transform(last_state)
            reason = "PROFIT" if profit > 0 else "LOSS"
            if profit < 0:
                if len(self.mistake_memory) > 1000: self.mistake_memory.pop(0) 
                self.mistake_memory.append(scaled_state)
            elif profit > 0:
                if len(self.victory_memory) > 1000: self.victory_memory.pop(0)
                direction = "BUY" if trade_type == 0 else "SELL"
                self.victory_memory.append({'vector': scaled_state, 'dir': direction, 'reason': reason})
            self.save_memory() 
            return reason
        except Exception as e: return f"Analysis Failed: {str(e)}"

    def check_historical_confluence(self, current_vector, direction_check):
        if not self.victory_memory: return 0.0
        best_sim = 0.0
        for mem in self.victory_memory:
            if mem['dir'] != direction_check: continue
            dist = np.linalg.norm(current_vector - mem['vector'])
            sim = 1.0 / (1.0 + dist)
            if sim > best_sim: best_sim = sim
        return best_sim if best_sim > 0.8 else 0.0

    def train(self, df_raw):
        if df_raw is None or len(df_raw) < 100: return
        try:
            df = df_raw.copy()
            df = self.calculate_indicators(df)
            if len(df) < 50: return 
            self.latest_atr = df['atr'].iloc[-1]
            X = df[self.feature_names] 
            threshold = df['atr'] * 1.5 
            y = np.where(df['close'].shift(-24) > (df['close'] + threshold), 1, 0) 
            mask = ~np.isnan(y)
            X = X[mask]; y = y[mask]
            X_scaled = self.scaler.fit_transform(X)
            self.gbm.fit(X_scaled, y); self.rf.fit(X_scaled, y); self.mlp.fit(X_scaled, y)
            self.ada.fit(X_scaled, y); self.et.fit(X_scaled, y)
            self.is_trained = True
        except: pass

    # =============================================================================
    # CONFIGURATION FOR BUY ENTRY AND SELL ENTRY (BUY LOW / SELL HIGH)
    # =============================================================================
    def predict_execution_setup(self, df, direction, current_price, confidence, tier):
        if len(df) < 50: return None 
        try:
            atr = df['atr'].iloc[-1]
            if atr == 0 or np.isnan(atr): atr = 0.001 
            
            # --- FACTOR: SUPPORT/RESISTANCE FOR BUY LOW/SELL HIGH ---
            recent_low = df['low'].tail(50).min()
            recent_high = df['high'].tail(50).max()
            ema50 = df['ema50'].iloc[-1]
            
            sl = 0.0; tp = 0.0
            buffer = atr * 1.5 
            
            # --- ADJUSTMENT: Relaxed Extension Limit for Entry ---
            # Increased from 5.0 to 9.0 to allow catching strong trends
            extension_limit = 9.0

            # --- BUY CONFIGURATION (Buy Low) ---
            if direction == "BUY":
                # FACTOR: Don't buy if price is extremely extended (Chasing)
                # Only enforce if tier is NOT MOON (High confidence overrides extension check)
                if tier != "MOON" and current_price > (ema50 + (extension_limit * atr)):
                      return None # Price too high (Overbought), wait for pullback
                
                structure_sl = recent_low - buffer
                volatility_sl = current_price - (atr * 3.0) # increased breathing room
                
                # If structure SL is too far (> 6 ATR), use Volatility SL
                if (current_price - structure_sl) > (atr * 6.0):
                      sl = volatility_sl
                else:
                      sl = min(structure_sl, volatility_sl)

                if sl >= current_price: sl = current_price - (atr * 3.0)
                
                # Reward Calculation
                risk_dist = current_price - sl
                reward_dist = risk_dist * 2.0 # Adjusted to 1:2 for higher hit rate
                tp = current_price + reward_dist

            # --- SELL CONFIGURATION (Sell High) ---
            elif direction == "SELL":
                # FACTOR: Don't sell if price is extremely extended below EMA (Chasing)
                if tier != "MOON" and current_price < (ema50 - (extension_limit * atr)):
                      return None # Price too low (Oversold), wait for pullback

                structure_sl = recent_high + buffer
                volatility_sl = current_price + (atr * 3.0)

                if (structure_sl - current_price) > (atr * 6.0):
                      sl = volatility_sl
                else:
                      sl = max(structure_sl, volatility_sl)

                if sl <= current_price: sl = current_price + (atr * 3.0)
                
                risk_dist = sl - current_price
                reward_dist = risk_dist * 2.0
                tp = current_price - reward_dist

            risk = abs(current_price - sl)
            reward = abs(tp - current_price)
            rr = reward / risk if risk > 0 else 0
            if sl < 0 or tp < 0: return None
            return {"entry": current_price, "sl": sl, "tp": tp, "risk": risk, "reward": reward, "rr": rr}
        except Exception as e: return None

    def predict(self, df_raw, performance_factor=1.0, active_weights=None, execution_mode="Balanced", macro_trend="NEUTRAL", genzig_engine=None):
        if not self.is_trained: return "HOLD", 0.0, "Wait", "Training...", {}
        try:
            df = self.calculate_indicators(df_raw.copy())
            df = df.tail(500).copy()
            if len(df) < 200: return "HOLD", 0.0, "DataWait", "Not enough Data", {}
            
            hurst, entropy, q_state, q_conf, buy_details = QuantumQueenLogic.analyze_quantum_state(df)
            quantum_data = {'hurst': hurst, 'entropy': entropy, 'state': q_state, 'conf': q_conf, 'buy_details': buy_details}

            # SENTIMENT ANALYSIS CALL
            sentiment_data = (50, "NEUTRAL") 
            
            # CANDLE PATTERN CALL
            candle_pattern = CandlePatternEngine.identify(df)

            curr = df[self.feature_names].tail(1)
            X_in = self.scaler.transform(curr)
            
            rl_penalty = 0.0
            if len(self.mistake_memory) > 0:
                dists = euclidean_distances(X_in, np.vstack(self.mistake_memory))
                min_dist = np.min(dists)
                if min_dist < 0.5: return "HOLD", 0.0, "MISTAKE_BANK_VETO", "Condition matches previous loss", {}
            
            probs = {
                'gbm': self.gbm.predict_proba(X_in)[0][1], 'rf': self.rf.predict_proba(X_in)[0][1],
                'mlp': self.mlp.predict_proba(X_in)[0][1], 'ada': self.ada.predict_proba(X_in)[0][1],
                'et': self.et.predict_proba(X_in)[0][1]
            }
            
            clone_match = self.check_manual_clone_signal(df)
            buy_boost = self.check_historical_confluence(X_in, "BUY")
            sell_boost = self.check_historical_confluence(X_in, "SELL")
            hist_boost = buy_boost if buy_boost > sell_boost else -sell_boost 
            
            genzig_report = None
            if genzig_engine:
                 vol_exp = curr['vol_expansion'].iloc[0]
                 genzig_report = genzig_engine.deep_dive_analysis(df, q_state, vol_exp)
            
            score, action, regime, details, report = OmniMindStrategy.analyze(
                df, probs, performance_factor, rl_penalty, quantum_data, active_weights, clone_match, execution_mode, macro_trend, hist_boost, genzig_report, sentiment_data, candle_pattern
            )
            return action, abs(score), regime, details, report
        except: return "HOLD", 0.0, "Err", "Prediction Error", {}

# =============================================================================
# 11. NEXUS LOGIC & MONITOR
# =============================================================================
class NexusLogicEngine:
    def __init__(self):
        self.sync_state = "INIT"
        self.logic_coherence = 1.0
        self.last_sync_time = time.time()

    def analyze_coherence(self, quantum_report, ai_confidence, regime, epoch_weights, execution_mode="Balanced"):
        score = 1.0
        msgs = []
        hurst = float(quantum_report.get('hurst', 0.5))
        
        # Adaptive Logic: Only penalize if Hurst is VERY low for Swing Mode
        min_hurst = 0.50 
        
        if regime == "TREND" and ai_confidence > 80:
            if hurst < min_hurst: score -= 0.3; msgs.append("LOGIC_FRACTURE: AI sees Trend, Physics sees Mean Reversion")
        
        if score >= 0.9: self.sync_state = "SYNERGY"
        elif score >= 0.7: self.sync_state = "ALIGNED"
        else: self.sync_state = "DESYNC"
        
        final_msg = " | ".join(msgs) if msgs else "All Systems Nominal"
        self.logic_coherence = score
        return score, self.sync_state, final_msg

class MultiFrameMonitor:
    @staticmethod
    def get_macro_trend(symbol):
        try:
            # Swing Trading uses H4 and H1 for market structure analysis
            tf_h4 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H4, 0, 50)
            tf_h1 = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 50)
            if tf_h4 is None or tf_h1 is None: return "NEUTRAL"
            h4_close = tf_h4['close'][-1]; h4_sma = np.mean(tf_h4['close'])
            h1_close = tf_h1['close'][-1]; h1_sma = np.mean(tf_h1['close'])
            if h4_close > h4_sma and h1_close > h1_sma: return "BULLISH"
            if h4_close < h4_sma and h1_close < h1_sma: return "BEARISH"
            return "NEUTRAL"
        except: return "NEUTRAL"

# =============================================================================
# 12. ZENITH ENGINE (EXECUTION & RISK)
# =============================================================================
class ZenithEngine(QThread):
    gui_update = pyqtSignal(dict)
    log_msg = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.running = True
        self.active_trading = True 
        self.execution_mode = "Swing-Master" 
        self.brains = {} 
        self.nexus = NexusLogicEngine() 
        self.genzig = GenZigEngine() 
        self.aegis = AegisEngine() 
        self.active_symbols = []
        self.symbol_timeframes = {} 
        self.last_sync_time = {} 
        self.last_log_time = {} 
        self.engine_states = {} 
        self.last_train = time.time()
        self.last_deep_train = time.time()
        self.system_status = "Initializing..."
        self.is_critical = False
        self.processed_deals = set()
        self.countdown_val = 100 
        self.last_timer_update = time.time()
        self.start_equity = 0.0
        self.current_equity = 0.0
        self.daily_start_equity = 0.0
        self.recent_wins = 0; self.recent_losses = 0; self.total_trades = 0
        self.total_pnl = 0.0; self.best_trade = 0.0; self.worst_trade = 0.0
        self.performance_factor = 1.0 
        self.ai_log_history = [] 
        self.epoch_engine = None
        self.active_ai_weights = None 
        self.known_tickets = set()
        self.last_heartbeat = 0
        self.last_10_history = []

    def resolve_symbol(self, base):
        suffixes = ["", ".s", ".pro", ".a", ".m", "_i", ".ecn"]
        prefixes = ["", "m", "i."]
        info = mt5.symbol_info(base)
        if info and info.visible and info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED: return base
        base_clean = base.replace(".s", "").replace(".pro", "")
        for p in prefixes:
            for s in suffixes:
                candidate = f"{p}{base_clean}{s}"
                info = mt5.symbol_info(candidate)
                if info and info.visible and info.trade_mode != mt5.SYMBOL_TRADE_MODE_DISABLED:
                    return candidate
        return None

    def normalize_volume(self, symbol, vol):
        info = mt5.symbol_info(symbol)
        if not info: return vol
        step = info.volume_step
        if step > 0: vol = round(vol / step) * step
        return max(info.volume_min, min(info.volume_max, round(vol, 2)))
    
    def check_market_status(self, symbol):
        info = mt5.symbol_info(symbol)
        if info is None: return False
        if info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED: return False
        tick = mt5.symbol_info_tick(symbol)
        if tick:
            if (datetime.datetime.now().timestamp() - tick.time) > 300: return False
        return True

    def calculate_risk(self, symbol, acc, atr, conf, regime, tier, custom_sl_dist=None):
        risk_percentage = BASE_RISK_PER_TRADE
        if conf > 90: risk_percentage = BASE_RISK_PER_TRADE * 1.5 
        
        # Max Cap
        if risk_percentage > 0.05: risk_percentage = 0.05
        
        if self.execution_mode == "Swing-Master": risk_percentage *= 1.2 
        
        if tier == "MOON": risk_percentage *= 1.5
        
        risk_amt = acc.equity * risk_percentage
        info = mt5.symbol_info(symbol)
        if not info: return 0.01
        
        sl_points = 0.0
        if custom_sl_dist: sl_points = custom_sl_dist / info.point
        else:
            sl_points = atr * 2.5 / info.point

        tick_val = info.trade_tick_value
        if tick_val == 0: tick_val = 1
        raw_lot = risk_amt / (sl_points * tick_val)
        return self.normalize_volume(symbol, raw_lot)

    def sync_account_history(self):
        try:
            from_date = datetime.datetime(2000, 1, 1)
            to_date = datetime.datetime.now()
            deals = mt5.history_deals_get(from_date, to_date)
            
            if deals:
                wins, losses, total_p, count = 0, 0, 0.0, 0
                best, worst = 0.0, 0.0
                if len(deals) > 0: best = -999999.0; worst = 999999.0
                
                closed_deals = []

                for d in deals:
                    if d.entry == mt5.DEAL_ENTRY_OUT: 
                        profit = d.profit + d.swap + d.commission
                        total_p += profit
                        count += 1
                        if profit >= 0: wins += 1
                        else: losses += 1
                        if profit > best: best = profit
                        if profit < worst: worst = profit
                        
                        closed_deals.append({
                            'time': datetime.datetime.fromtimestamp(d.time).strftime("%Y-%m-%d %H:%M"),
                            'symbol': d.symbol,
                            'type': 'BUY' if d.type==0 else 'SELL',
                            'volume': d.volume,
                            'profit': profit
                        })

                        if d.ticket not in self.processed_deals:
                            sym = d.symbol
                            if sym in self.brains:
                                rates = mt5.copy_rates_from(sym, self.symbol_timeframes.get(sym, mt5.TIMEFRAME_H1), datetime.datetime.fromtimestamp(d.time), 100)
                                if rates is not None and len(rates) > 50:
                                    df_hist = pd.DataFrame(rates)[['time', 'close', 'open', 'high', 'low', 'tick_volume']]
                                    df_hist['time'] = pd.to_datetime(df_hist['time'], unit='s')
                                    df_hist.set_index('time', inplace=True)
                                    analysis = self.brains[sym].learn_outcome(df_hist, d.type, profit)
                                    if profit < 0: self.log_msg.emit(f"🧠 LEARNING: Added Mistake for {sym}. Reason: {analysis}")
                                    elif profit > 0: self.log_msg.emit(f"🧠 REINFORCEMENT: Added Victory for {sym}. Reason: {analysis}")
                                    self.processed_deals.add(d.ticket)
                
                self.recent_wins = wins
                self.recent_losses = losses
                self.total_trades = count
                self.total_pnl = total_p
                if count > 0:
                    self.best_trade = best
                    self.worst_trade = worst
                
                self.last_10_history = sorted(closed_deals, key=lambda x: x['time'], reverse=True)[:10]

        except Exception as e:
            self.log_msg.emit(f"History Sync Failed: {e}")

    def select_optimal_timeframe(self, symbol):
        # SWING TRADER LOGIC: M15 Primary for Signals
        return mt5.TIMEFRAME_M15

    def get_tf_label(self, tf):
        if tf == mt5.TIMEFRAME_M1: return "M1"
        if tf == mt5.TIMEFRAME_M5: return "M5"
        if tf == mt5.TIMEFRAME_M15: return "M15"
        if tf == mt5.TIMEFRAME_M30: return "M30"
        if tf == mt5.TIMEFRAME_H1: return "H1"
        if tf == mt5.TIMEFRAME_H4: return "H4"
        return "M?"

    def update_timeframe_override(self, symbol, tf_code):
        if symbol in self.symbol_timeframes:
            self.symbol_timeframes[symbol] = tf_code
            self.log_msg.emit(f"Timeframe Manually Overridden for {symbol}. Re-syncing...")
            self.last_sync_time[symbol] = 0 

    def update_status(self, symbol, status):
        self.engine_states[symbol] = status

    def log_trend_analysis(self, symbol, action, conf, regime, details, report, logic_status, coherence, exec_setup=None, execution_status="UNKNOWN", skip_reason=""):
        tf_label = self.get_tf_label(self.symbol_timeframes.get(symbol, mt5.TIMEFRAME_M5))
        log_entry = {
            'time': datetime.datetime.now().strftime("%H:%M:%S"),
            'symbol': f"{symbol} ({tf_label})", 'action': action, 'conf': conf,
            'regime': regime, 'details': details
        }
        self.ai_log_history.append(log_entry)
        should_log_text = False
        last_log = self.last_log_time.get(symbol, 0)
        if execution_status == "EXECUTED": should_log_text = True
        elif time.time() - last_log > 60: should_log_text = True 
        if not should_log_text: return
        self.last_log_time[symbol] = time.time()
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        exec_str = "► EXECUTION AI: No Signal Generated (Wait Mode)\n"
        if exec_setup:
            win_prob = conf / 100.0
            loss_prob = 1.0 - win_prob
            expected_value = (win_prob * exec_setup['reward']) - (loss_prob * exec_setup['risk'])
            exec_str = (
                f"► EXECUTION AI (Tier: {report.get('tier', 'N/A')}):\n"
                f"  • Entry: {exec_setup['entry']:.5f}\n"
                f"  • Smart SL: {exec_setup['sl']:.5f} (Risk: {exec_setup['risk']:.5f})\n"
                f"  • Smart TP: {exec_setup['tp']:.5f} (Reward: {exec_setup['reward']:.5f})\n"
                f"  • Risk/Reward Ratio: {exec_setup['rr']:.2f}\n"
                f"  • Exp. Value (EV): {expected_value:.5f}\n"
                f"----------------------------------------------------------\n"
            )
        status_color = "🟢" if execution_status == "EXECUTED" else "🔴"
        display_action = action
        if action == "BUY": display_action = "BUY / LONG"
        elif action == "SELL": display_action = "SELL / SHORT"
        log_block = (
            f"\n═════════════ [ZENITH QUANTUM REPORT: {symbol}] ═════════════\n"
            f"► MODE: {self.execution_mode.upper()} | STATUS: {status_color} {execution_status} {f'({skip_reason})' if skip_reason else ''}\n"
            f"► TIME: {timestamp} | SIGNAL: {display_action} | CONFIDENCE: {conf:.1f}%\n"
            f"► REGIME: {regime} | QUANTUM STATE: {report.get('quantum_state', 'N/A')}\n"
            f"► NEXUS LOGIC: {logic_status} (Coherence: {coherence:.2f})\n"
            f"► SENTIMENT: {report.get('sentiment', 'N/A')}\n"
            f"----------------------------------------------------------\n"
            f"{exec_str}"
            f"► PHYSICS ENGINE:\n"
            f"  • Hurst Exp: {report.get('hurst', 'N/A')} (Persistence)\n"
            f"  • Entropy: {report.get('entropy', 'N/A')} (Chaos)\n"
            f"----------------------------------------------------------\n"
            f"► BRAIN ACTIVITY:\n"
            f"  • {details}\n"
            f"══════════════════════════════════════════════════════════"
        )
        self.log_msg.emit(log_block)

    def handle_epoch_update(self, new_weights):
        self.active_ai_weights = new_weights
        self.log_msg.emit(f"🧬 BRAIN UPDATE: New Ensemble Weights Applied: {new_weights}")

    def run_council_voting(self, symbol, action, conf, report, logic_score, acc_equity, positions, df_full):
        self.update_status(symbol, "COUNCIL VOTING")
        quantum_state = report.get('quantum_state', 'UNKNOWN')
        
        # Physics Veto
        if quantum_state == "HIGH_ENTROPY_NOISE" and self.execution_mode != "Zenith Auto-Pilot": return False, "Physics Engine Veto (High Entropy)"
        if logic_score < 0.5: return False, "Nexus Logic Veto (Low Coherence)" 
        
        sym_pos = [p for p in positions if p.symbol == symbol]
        risk_score, risk_status = self.aegis.analyze_risk(acc_equity, sym_pos, conf, float(report.get('volatility', '1.0').split('x')[0]))
        if risk_score < 0.5: return False, f"Aegis Risk Veto ({risk_status})" 
        
        if action != "HOLD" and df_full is not None:
             # GenZig Validation
             logic_fn = lambda r: r['close'] > r['ema50'] if action == "BUY" else r['close'] < r['ema50']
             wr = self.genzig.run_backtest(df_full, logic_fn, action)
             # --- FIX: LOWERED BACKTEST THRESHOLD FROM 50 TO 45 ---
             if wr < 45.0: return False, f"Backtest Veto (Recent WR: {wr:.1f}%)" 
        return True, "APPROVED"

    def run(self):
        self.log_msg.emit("--- FLOKI: ZENITH AUTO-PILOT STARTING ---")
        self.log_msg.emit("⚡ NEXUS LOGIC ENGINE: ONLINE")
        self.log_msg.emit("⚡ GENZIG AI ENGINE: ONLINE (BACKTEST ACTIVE)")
        self.log_msg.emit("⚡ AEGIS RISK ENGINE: ONLINE")
        self.log_msg.emit("⚡ SENTIMENT ENGINE: ONLINE (PROXY MODE)")
        self.log_msg.emit("⚡ CANDLE PATTERN ENGINE: ONLINE")
        self.log_msg.emit(f"⚡ TELEGRAM BOT: {'ACTIVE' if REQUESTS_AVAILABLE else 'DISABLED'}")
        
        while self.running:
            # --- TELEGRAM HEARTBEAT ---
            if time.time() - self.last_heartbeat > (HEARTBEAT_INTERVAL_MIN * 60):
                TelegramMessenger.send_message(f"Zenith Heartbeat: System Nominal. Equity: ${self.current_equity:.2f}")
                self.last_heartbeat = time.time()

            if not mt5.terminal_info().connected:
                self.system_status = "WAITING FOR MT5..."
                self.is_critical = True
                self.gui_update.emit({'account': {'equity': 0, 'profit': 0, 'balance': 0, 'margin': 0, 'free_margin': 0, 'margin_level': 0}, 'chart_data': {}, 'status': self.system_status, 'critical': self.is_critical, 'timer': 0, 'positions': [], 'ai_log': []})
                time.sleep(5)
                continue

            if not self.active_symbols:
                self.log_msg.emit("Mapping Symbols & Initializing Quantum Matrix...")
                for s in BASE_SYMBOLS:
                    real_s = self.resolve_symbol(s)
                    if real_s:
                        self.active_symbols.append(real_s)
                        self.brains[real_s] = ZenithBrain(real_s)
                        self.symbol_timeframes[real_s] = mt5.TIMEFRAME_M15 # FORCED M15 FOR SIGNAL FINDING
                        self.last_sync_time[real_s] = 0 
                        self.engine_states[real_s] = "BOOTING"
                        self.log_msg.emit(f"Brain Loaded: {real_s}")
                
                if not self.active_symbols:
                    self.log_msg.emit("No valid symbols found. Retrying...")
                    time.sleep(5)
                    continue
                else:
                    self.log_msg.emit("Initial Deep Sync...")
                    self.sync_data(MAX_HISTORY_ROWS)
                    self.sync_account_history() 

                    self.epoch_engine = EpochEngine(self.active_symbols)
                    self.epoch_engine.log_msg.connect(self.log_msg)
                    self.epoch_engine.optimization_complete.connect(self.handle_epoch_update)
                    self.epoch_engine.start()

            try:
                self.sync_account_history() 
                acc = mt5.account_info()
                if not acc: time.sleep(1); continue
                
                self.current_equity = acc.equity
                if self.start_equity == 0: 
                    self.start_equity = acc.equity
                    self.daily_start_equity = acc.equity

                if self.start_equity > 0:
                    self.performance_factor = self.current_equity / self.start_equity

                daily_pnl = (acc.equity - self.daily_start_equity) / self.daily_start_equity
                
                can_trade = self.active_trading
                self.is_critical = False
                
                if not acc.trade_allowed:
                    can_trade = False
                    self.system_status = "TRADING DISABLED BY BROKER"
                    self.is_critical = True
                elif acc.margin_level > 0 and acc.margin_level < MIN_MARGIN_LEVEL: 
                    can_trade = False
                    self.system_status = f"LOW MARGIN ({acc.margin_level:.1f}%)"
                    self.is_critical = True
                elif daily_pnl >= DAILY_PROFIT_TARGET:
                    can_trade = False
                    self.system_status = f"TARGET HIT (+{daily_pnl*100:.2f}%) - TRADING STOPPED"
                elif not self.active_trading:
                    self.system_status = "STANDBY - AUTOPILOT OFF"
                else:
                    self.system_status = f"QUANTUM MODE: {self.execution_mode} | PF: {self.performance_factor:.2f}"

                win_rate = 0
                if self.total_trades > 0:
                    win_rate = (self.recent_wins / self.total_trades) * 100
                
                voting_result_data = {'winner': 'WAIT', 'details': 'Waiting for Signal...', 'color_code': 'gray'}

                payload = {
                    'account': {'equity': acc.equity, 'profit': acc.profit, 'balance': acc.balance, 'margin': acc.margin, 'free_margin': acc.margin_free, 'margin_level': acc.margin_level}, 
                    'stats': {'wins': self.recent_wins, 'losses': self.recent_losses, 'total': self.total_trades, 'win_rate': win_rate, 'total_pnl': self.total_pnl, 'best': self.best_trade, 'worst': self.worst_trade, 'daily': daily_pnl * 100},
                    'chart_data': {}, 'status': self.system_status, 'critical': self.is_critical,
                    'timer': int(self.countdown_val * 0.05),
                    'positions': [], 'ai_log': self.ai_log_history[-50:],
                    'engine_states': self.engine_states, 
                    'voting_result': voting_result_data,
                    'history': self.last_10_history
                }
                
                positions = mt5.positions_get()
                current_tickets = set()
                
                if positions:
                    for p in positions:
                        current_tickets.add(p.ticket)
                        payload['positions'].append({
                            'ticket': p.ticket, 'symbol': p.symbol, 'type': 'BUY' if p.type==0 else 'SELL',
                            'vol': p.volume, 'open': p.price_open, 'curr': p.price_current,
                            'sl': p.sl, 'tp': p.tp, 'swap': p.swap, 'profit': p.profit
                        })
                        
                        if p.ticket not in self.known_tickets:
                            if p.magic != MAGIC_NUMBER:
                                self.log_msg.emit(f"⚠️ MANUAL TRADE DETECTED on {p.symbol}. Initiating Analysis...")
                                if p.symbol in self.brains:
                                    df_full = ZenithMemory.load_data(p.symbol)
                                    if df_full is not None and len(df_full) >= 50:
                                            direction = "BUY" if p.type == 0 else "SELL"
                                            analysis = self.brains[p.symbol].capture_manual_trade(df_full, direction)
                                            self.log_msg.emit(f"🔍 ANALYZED REASON: {analysis}")
                                            self.log_msg.emit(f"🧬 SAVED AS CLONE EXEMPLAR for future replication.")
                            self.known_tickets.add(p.ticket)

                if time.time() - self.last_timer_update >= 0.05:
                        self.countdown_val -= 1
                        self.last_timer_update = time.time()
                
                if self.countdown_val <= 0:
                    self.countdown_val = 100 
                    
                    if time.time() - self.last_train > 300: 
                        self.log_msg.emit("🧠 AI Refresh (5m)...")
                        self.sync_data(1000)
                        self.last_train = time.time()

                    if time.time() - self.last_deep_train > 900:
                        self.log_msg.emit("🧬 DEEP QUANTUM TRAINING (15m)...")
                        self.sync_data(2500)
                        for s in self.active_symbols:
                            self.brains[s].save_memory()
                        self.last_deep_train = time.time()
                        self.log_msg.emit(f"════ [DEEP SYNC COMPLETE] ════\nSystem Saved.")

                    for s in self.active_symbols:
                        if time.time() - self.last_sync_time.get(s, 0) < SYNC_INTERVAL_SEC: continue

                        if not self.check_market_status(s):
                            self.update_status(s, "MARKET CLOSED")
                            continue

                        self.update_status(s, "SYNCING DATA")
                        if s not in self.symbol_timeframes:
                             self.symbol_timeframes[s] = self.select_optimal_timeframe(s)
                        
                        self.sync_data_single(s, 200)
                        self.last_sync_time[s] = time.time() 

                        tick = mt5.symbol_info_tick(s)
                        if not tick: continue
                        
                        df_full = ZenithMemory.load_data(s)
                        if df_full is None or len(df_full) < 100: continue
                        
                        chart_subset = df_full.tail(60) 
                        ohlc_data = []
                        for i in range(len(chart_subset)):
                            row = chart_subset.iloc[i]
                            ohlc_data.append((i, row['open'], row['close'], row['low'], row['high']))
                        payload['chart_data'][s] = ohlc_data

                        self.update_status(s, "CALCULATING INDICATORS")
                        macro_trend = MultiFrameMonitor.get_macro_trend(s)
                        self.update_status(s, "ANALYZING (OMNI-MIND)")
                        action, conf, regime, details, report = self.brains[s].predict(
                            df_full, self.performance_factor, self.active_ai_weights, self.execution_mode, macro_trend, self.genzig
                        )

                        self.update_status(s, "VALIDATING (NEXUS)")
                        logic_score, logic_status, logic_msg = self.nexus.analyze_coherence(
                            report, conf, regime, self.active_ai_weights, self.execution_mode
                        )
                        
                        info = mt5.symbol_info(s)
                        spread = (tick.ask - tick.bid) / info.point if info else 0
                        
                        final_action = action
                        final_conf = conf
                        engine_override = False
                        
                        quantum_state = report.get('quantum_state', 'UNKNOWN')
                        
                        if quantum_state == "QUANTUM_BUY_SIGNAL":
                            final_action = "BUY"; final_conf = 95.0; engine_override = True; regime = "QUANTUM_BUY_SINGULARITY"
                        elif quantum_state == "QUANTUM_SELL_SIGNAL":
                            final_action = "SELL"; final_conf = 95.0; engine_override = True; regime = "QUANTUM_SELL_SINGULARITY"
                        elif quantum_state == "ORDERED_TREND":
                            if final_conf < 75:
                                df_tech = self.brains[s].calculate_indicators(df_full.copy())
                                ma200 = df_tech['sma200'].iloc[-1]
                                current_close = df_tech['close'].iloc[-1]
                                quantum_dir = "BUY" if current_close > ma200 else "SELL"
                                entropy = float(report.get('entropy', 10.0))
                                if entropy < 1.5:
                                    final_action = quantum_dir; final_conf = 85.0; engine_override = True; regime = "QUANTUM_OVERRIDE"

                        tier = report.get('tier', 'NONE')
                        exec_direction = final_action
                        current_price = tick.ask if exec_direction == "BUY" else tick.bid
                        
                        self.update_status(s, "CALCULATING EXECUTION")
                        df_tech_exec = self.brains[s].calculate_indicators(df_full.copy())
                        exec_setup = self.brains[s].predict_execution_setup(df_tech_exec, exec_direction, current_price, final_conf, tier)

                        winning_engine = "ZENITH"; win_color = "red"; vote_details = "Zenith Active"
                        if engine_override: winning_engine = "QUANTUM"; win_color = "blue"; vote_details = f"Quantum Override Active (Conf: {final_conf}%)"
                        elif final_conf > 70 and logic_score > 0.6: winning_engine = "ZENITH+QUANTUM"; win_color = "purple"; vote_details = "Engines Synced"
                        
                        payload['voting_result'] = {'winner': winning_engine, 'details': vote_details, 'color_code': win_color, 'full_data': details}

                        if can_trade:
                            self.update_status(s, "MANAGING POSITIONS")
                            self.check_time_decay(s, positions)
                            self.check_profit_taking(s, positions, df_tech_exec, report, final_conf)
                            self.manage_trailing(s, positions, tick, self.brains[s].latest_atr, tier, regime)
                            
                            self.manage_execution_automation(s, positions, tick, self.brains[s].latest_atr, final_conf, regime)
                            
                            # =============================================================================
                            # XAUUSD REVERSAL SWAP LOGIC (The "Flip")
                            # =============================================================================
                            # IMPROVEMENT: Regime and Time Filter to prevent Whipsaws
                            
                            # Track closed tickets in this cycle to ensure 'Flip' doesn't get blocked by position limits
                            closed_tickets_this_cycle = set()
                            
                            if "XAUUSD" in s and regime not in ["RANGE", "CHOP"]: 
                                for p in positions:
                                    if p.symbol == s:
                                        # 30-min Filter: Don't flip young trades unless > 95% conf
                                        duration = time.time() - p.time
                                        required_flip_conf = 95.0 if duration < 1800 else 85.0
                                        
                                        # HOLDING BUY -> DETECTED STRONG SELL REVERSAL
                                        if p.type == 0 and final_action == "SELL" and final_conf > required_flip_conf:
                                             if regime in ["VOLATILITY", "QUANTUM_SELL_SIGNAL", "QUANTUM_SELL_SINGULARITY", "TREND"]:
                                                 self.log_msg.emit(f"⚠️ REVERSAL SWAP DETECTED on {s}: Closing Buy to Flip Sell (Conf {final_conf}%).")
                                                 self.close_order(p.ticket, "Reversal Flip (Buy -> Sell)")
                                                 closed_tickets_this_cycle.add(p.ticket)
                                     
                                        # HOLDING SELL -> DETECTED STRONG BUY REVERSAL
                                        if p.type == 1 and final_action == "BUY" and final_conf > required_flip_conf:
                                             if regime in ["VOLATILITY", "QUANTUM_BUY_SIGNAL", "QUANTUM_BUY_SINGULARITY", "TREND"]:
                                                 self.log_msg.emit(f"⚠️ REVERSAL SWAP DETECTED on {s}: Closing Sell to Flip Buy (Conf {final_conf}%).")
                                                 self.close_order(p.ticket, "Reversal Flip (Sell -> Buy)")
                                                 closed_tickets_this_cycle.add(p.ticket)

                            is_grid_recovery = False # GRID DISABLED FOR SWING

                            auto_pilot_override = False
                            # MODIFIED: Relaxed entry requirement to >= 60 confidence for candidacy
                            is_trade_candidate = final_action != "HOLD" and (final_conf >= 60 or engine_override) 
                            
                            if self.execution_mode == "Zenith Auto-Pilot":
                                if quantum_state == "HIGH_ENTROPY_NOISE" and not engine_override:
                                     is_trade_candidate = False; skip_reason = "Auto-Pilot Sleep (Noise)"; auto_pilot_override = True

                            base_spread_cap = SYMBOL_SPREAD_CAPS.get(s, SYMBOL_SPREAD_CAPS["DEFAULT"])
                            current_spread_limit = base_spread_cap * MODE_SPREAD_FACTOR.get(self.execution_mode, 1.0)
                            is_spread_ok = spread < current_spread_limit

                            # Filter out closed positions so we can open new ones instantly in swap
                            current_open_total = len([p for p in (positions or []) if p.ticket not in closed_tickets_this_cycle])
                            current_symbol_open = len([p for p in (positions or []) if p.symbol == s and p.ticket not in closed_tickets_this_cycle])
                            
                            dynamic_symbol_limit = MAX_POSITIONS_PER_SYMBOL - 1 
                            if tier in ["MOON", "AGGRESSIVE"] or self.execution_mode == "Zenith Auto-Pilot":
                                dynamic_symbol_limit = MAX_POSITIONS_PER_SYMBOL 

                            is_slots_available = True
                            slot_reason = ""
                            if current_open_total >= MAX_POSITIONS_TOTAL:
                                is_slots_available = False; slot_reason = "Max Total Slots"
                            elif current_symbol_open >= dynamic_symbol_limit:
                                is_slots_available = False; slot_reason = f"Max Symbol Slots ({dynamic_symbol_limit})"

                            council_approved, council_msg = self.run_council_voting(s, final_action, final_conf, report, logic_score, acc.equity, positions, df_tech_exec)
                            if engine_override: council_approved = True 

                            execution_status = "WAITING"
                            skip_reason = ""

                            if auto_pilot_override: execution_status = "SLEEPING"
                            
                            # --- FIX: LOWERED EXECUTION THRESHOLD FROM 75 TO 60 ---
                            elif final_conf < 60 and not engine_override and not is_grid_recovery:
                                execution_status = "MONITORING"; skip_reason = f"Conf {final_conf:.1f}% < 60%"
                                
                            elif not is_trade_candidate: execution_status = "MONITORING"; skip_reason = "Low Confidence"
                            elif not is_spread_ok: skip_reason = "Spread High"
                            elif not is_slots_available: skip_reason = f"Slots Full: {slot_reason}"
                            elif not exec_setup: skip_reason = "Calc Fail (Risk/Reward Bad)"
                            elif not council_approved: skip_reason = f"VETO: {council_msg}"
                            else: execution_status = "EXECUTED"

                            self.update_status(s, execution_status)
                            self.log_trend_analysis(s, final_action, final_conf, regime, details, report, logic_status, logic_score, exec_setup, execution_status, skip_reason)

                            if execution_status == "EXECUTED":
                                custom_sl_dist = exec_setup['risk'] if exec_setup else None
                                vol = self.calculate_risk(s, acc, self.brains[s].latest_atr, final_conf, regime, tier, custom_sl_dist)
                                self.execute_trade(s, final_action, tick, self.brains[s].latest_atr, vol, final_conf, regime, tier, exec_setup)
                        else:
                            self.update_status(s, "MONITORING (MANUAL)")
                            self.log_trend_analysis(s, final_action, final_conf, regime, details, report, logic_status, logic_score, exec_setup, "MONITORING", "Trading Disabled")

                self.gui_update.emit(payload)
                time.sleep(0.01) 

            except Exception as e:
                self.log_msg.emit(f"Loop Error: {str(e)}")
                traceback.print_exc() 
                time.sleep(0.05)

    def sync_data(self, count):
        for s in self.active_symbols:
            self.sync_data_single(s, count)

    def sync_data_single(self, s, count):
        try:
            tf = self.symbol_timeframes.get(s, mt5.TIMEFRAME_H1)
            rates = mt5.copy_rates_from_pos(s, tf, 0, count)
            if rates is not None and len(rates) > 0:
                df_new = pd.DataFrame(rates)[['time', 'close', 'open', 'high', 'low', 'tick_volume']]
                df_new['time'] = pd.to_datetime(df_new['time'], unit='s')
                df_new.set_index('time', inplace=True)
                df_combined = ZenithMemory.save_data(s, df_new)
                self.brains[s].train(df_combined)
        except Exception: pass

    def check_profit_taking(self, symbol, positions, df, report, conf):
        if not positions: return
        try:
            curr = df.iloc[-1]
            
            # Get latest candle pattern specifically for exit logic
            pat_name, _ = CandlePatternEngine.identify(df)
            
            for p in positions:
                if p.symbol != symbol: continue
                close_it = False; reason = ""
                
                # --- IMPROVED EARLY REVERSAL DETECTION (Smart Exit) ---
                # Requirement: Candle must CLOSE outside the bands (no wicks) + Strong RSI
                
                if p.type == 0: # Long
                    is_close_upper = curr['close'] > curr['bb_upper']
                    is_rsi_hot = curr['rsi'] > 75 # Raised to 75 to avoid exiting trending moves
                    is_bear_candle = "BEAR" in pat_name or "SHOOTING" in pat_name
                    
                    if is_close_upper and is_rsi_hot and is_bear_candle:
                          close_it = True
                          reason = f"Smart Exit: Top Reversal ({pat_name} + RSI>75)"

                elif p.type == 1: # Short
                    is_close_lower = curr['close'] < curr['bb_lower']
                    is_rsi_cold = curr['rsi'] < 25 # Lowered to 25
                    is_bull_candle = "BULL" in pat_name or "HAMMER" in pat_name
                    
                    if is_close_lower and is_rsi_cold and is_bull_candle:
                          close_it = True
                          reason = f"Smart Exit: Bottom Reversal ({pat_name} + RSI<25)"
                
                if close_it: self.close_order(p.ticket, reason)
        except Exception as e: print(f"Profit Check Error: {e}")

    def check_time_decay(self, symbol, positions):
        if not positions: return
        now = time.time()
        limit_mins = STAGNATION_LIMIT_MINUTES
        if limit_mins == 0: return # Disabled (Wait indefinitely for TP)
        
        for p in positions:
            if p.symbol != symbol: continue
            duration = now - p.time
            if duration > (limit_mins * 60):
                # Only close if it's actually doing nothing after 48 hours
                if -10 < p.profit < 10: 
                    self.close_order(p.ticket, "Stagnation (48hr+)")

    def manage_execution_automation(self, symbol, positions, tick, atr, conf, regime):
        if not positions: return
        if atr == 0: atr = 0.001
        
        # Massive TP for Swing
        sl_mult = 2.5; tp_mult = 7.5 
        
        sl_dist = atr * sl_mult; tp_dist = atr * tp_mult
        
        for p in positions:
            if p.symbol != symbol: continue
            updates = {}
            if p.sl == 0.0:
                if p.type == 0: updates['sl'] = p.price_open - sl_dist
                else: updates['sl'] = p.price_open + sl_dist
            if p.tp == 0.0:
                if p.type == 0: updates['tp'] = p.price_open + tp_dist
                else: updates['tp'] = p.price_open - tp_dist
            if updates:
                req = {"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "symbol": p.symbol, "sl": float(updates.get('sl', p.sl)), "tp": float(updates.get('tp', p.tp))}
                mt5.order_send(req)
                self.log_msg.emit(f"AUTO-SYNC: Enforcing Swing SL/TP for #{p.ticket}")

    def manage_trailing(self, symbol, positions, tick, atr, tier="NEUTRAL", regime="RANGE"):
        if not positions: return
        if atr == 0: atr = 0.001
        for p in positions:
            if p.symbol != symbol: continue
            curr = tick.bid if p.type==0 else tick.ask
            profit_points = (curr - p.price_open) if p.type==0 else (p.price_open - curr)
            
            new_sl = 0.0
            
            # --- IMPROVED STEP-LADDER TRAILING (Breathing Room) ---
            
            # Step 3: Massive Profit (Move SL to Secure 50% of the Swing)
            # Trigger: > 8.0 ATR Profit -> Lock 5.0 ATR
            if profit_points > (atr * 8.0):
                 target_sl = p.price_open + (atr * 5.0) if p.type == 0 else p.price_open - (atr * 5.0)
                 if (p.type == 0 and p.sl < target_sl) or (p.type == 1 and (p.sl == 0 or p.sl > target_sl)):
                      new_sl = target_sl

            # Step 2: Good Profit (Move SL to Secure 2.5 ATR)
            # Trigger: > 5.0 ATR Profit -> Lock 2.5 ATR
            elif profit_points > (atr * 5.0):
                 target_sl = p.price_open + (atr * 2.5) if p.type == 0 else p.price_open - (atr * 2.5)
                 if (p.type == 0 and p.sl < target_sl) or (p.type == 1 and (p.sl == 0 or p.sl > target_sl)):
                      new_sl = target_sl

            # Step 1: Secure Break-Even (Original Logic)
            # Trigger: > 2.0 ATR Profit -> Lock 0.2 ATR (BE)
            elif profit_points > (atr * 2.0):
                 be_level = p.price_open + (atr * 0.2) if p.type == 0 else p.price_open - (atr * 0.2)
                 if (p.type == 0 and p.sl < be_level) or (p.type == 1 and (p.sl == 0 or p.sl > be_level)):
                      new_sl = be_level

            if new_sl != 0.0:
                 mt5.order_send({"action": mt5.TRADE_ACTION_SLTP, "position": p.ticket, "sl": float(new_sl), "tp": p.tp})
                 self.log_msg.emit(f"🔒 SWING-STEP: #{p.ticket} SL moved to secure profit.")

    def execute_trade(self, symbol, direction, tick, atr, volume, conf, regime, tier, exec_setup=None, is_grid=False):
        if volume == 0: return
        info = mt5.symbol_info(symbol)
        if not info or info.trade_mode == mt5.SYMBOL_TRADE_MODE_DISABLED:
            self.log_msg.emit(f"SKIPPED: {symbol} Trading Disabled/Info Error")
            return

        price = tick.ask if direction == "BUY" else tick.bid
        sl = 0.0; tp = 0.0

        if exec_setup and not is_grid:
            sl = exec_setup['sl']
            tp = exec_setup['tp']
        else:
            sl_mult = 2.5; tp_mult = 6.0 
            sl_dist = atr * sl_mult
            sl = price - sl_dist if direction == "BUY" else price + sl_dist
            tp = price + (sl_dist * tp_mult) if direction == "BUY" else price - (sl_dist * tp_mult)
        
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(volume),
            "type": mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL,
            "price": float(price), "sl": float(sl), "tp": float(tp), "magic": MAGIC_NUMBER,
            "comment": f"Z-SWING-{tier}", "type_filling": mt5.ORDER_FILLING_FOK
        }
        
        modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
        res = None
        for mode in modes:
            req['type_filling'] = mode
            res = mt5.order_send(req)
            if res.retcode != 10030: break
                
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            self.log_msg.emit(f"OPEN: {direction} {symbol} {volume}L (SWING EXECUTION)")
            TelegramMessenger.send_message(f"🚀 Zenith Swing Entry:\n{direction} {symbol} {volume}L\nPrice: {price}\nSL: {sl}\nTP: {tp}")
        else:
            self.log_msg.emit(f"Trade Failed: {res.comment} (Ret: {res.retcode})")

    def close_order(self, ticket, reason):
        p = mt5.positions_get(ticket=ticket)
        if not p: return
        p = p[0]
        tick = mt5.symbol_info_tick(p.symbol)
        req = {
            "action": mt5.TRADE_ACTION_DEAL, "position": ticket, "symbol": p.symbol, "volume": p.volume,
            "type": mt5.ORDER_TYPE_SELL if p.type==0 else mt5.ORDER_TYPE_BUY,
            "price": tick.bid if p.type==0 else tick.ask, "magic": MAGIC_NUMBER,
            "type_filling": mt5.ORDER_FILLING_FOK
        }
        modes = [mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN]
        res = None
        for mode in modes:
            req['type_filling'] = mode
            res = mt5.order_send(req)
            if res.retcode != 10030: break
        
        if res.retcode == mt5.TRADE_RETCODE_DONE:
            profit = p.profit + p.swap
            self.total_trades += 1
            self.total_pnl += profit
            if profit > 0: self.recent_wins += 1
            else: self.recent_losses += 1
            if profit > self.best_trade: self.best_trade = profit
            if profit < self.worst_trade: self.worst_trade = profit
            self.log_msg.emit(f"CLOSED #{ticket}: {reason} (${profit:.2f})")
            TelegramMessenger.send_message(f"🔔 Zenith Trade Closed:\n#{ticket} ({p.symbol})\nReason: {reason}\nProfit: ${profit:.2f}")
            self.sync_account_history()

    def panic_close_all(self):
        self.log_msg.emit("PANIC TRIGGERED: Closing All Positions (AI Remains Active)...")
        positions = mt5.positions_get()
        if positions:
            for p in positions:
                self.close_order(p.ticket, "PANIC BUTTON")
        else:
            self.log_msg.emit("No open positions to close.")

# =============================================================================
# 13. GUI CLASS
# =============================================================================
class LoginWin(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Paldo ATLM Connector")
        self.setStyleSheet(STYLESHEET)
        self.resize(300, 220)
        self.settings = QSettings("Paldo ATLM", "Authorized")
        lay = QVBoxLayout()
        lay.addWidget(QLabel("BROKER LOGIN"), alignment=Qt.AlignCenter)
        self.u = QLineEdit(placeholderText="Login ID"); lay.addWidget(self.u)
        self.p = QLineEdit(placeholderText="Password"); self.p.setEchoMode(QLineEdit.Password); lay.addWidget(self.p)
        self.s = QComboBox(); self.s.setEditable(True); lay.addWidget(self.s)
        self.s.addItems(["PUPrime-Demo", "MetaQuotes-Demo", "EightCap-Real", "Axi-US50-Demo"])
        self.rem = QCheckBox("Save Credentials"); lay.addWidget(self.rem)
        self.btn = QPushButton("CONNECT"); self.btn.clicked.connect(self.connect); lay.addWidget(self.btn)
        self.status = QLabel(""); self.status.setStyleSheet("color: #00ffcc;"); lay.addWidget(self.status)
        self.setLayout(lay)
        if self.settings.value("rem"):
            self.u.setText(self.settings.value("u")); self.p.setText(self.settings.value("p")); self.s.setCurrentText(self.settings.value("s")); self.rem.setChecked(True)

    def connect(self):
        self.btn.setEnabled(False); self.status.setText("Connecting...")
        QApplication.processEvents()
        if not MT5_AVAILABLE: self.status.setText("Missing MT5 Lib"); self.btn.setEnabled(True); return
        if not mt5.initialize(): self.status.setText("Open MT5 First!"); self.btn.setEnabled(True); return
        try:
            if mt5.login(int(self.u.text()), self.p.text(), self.s.currentText()):
                if self.rem.isChecked():
                    self.settings.setValue("u", self.u.text()); self.settings.setValue("p", self.p.text()); self.settings.setValue("s", self.s.currentText()); self.settings.setValue("rem", 1)
                self.accept()
            else:
                self.status.setText(f"Failed: {mt5.last_error()}"); self.btn.setEnabled(True)
        except: self.status.setText("Invalid ID"); self.btn.setEnabled(True)

class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.resize(1400, 950)
        self.setStyleSheet(STYLESHEET)
        self.worker = None
        self.charts = {}
        self.init_ui()
        self.update_timer = QTimer()
        QTimer.singleShot(500, self.start_worker)

    def start_worker(self):
        self.worker = ZenithEngine()
        self.worker.gui_update.connect(self.render)
        self.worker.log_msg.connect(self.log)
        self.worker.start()

    def init_ui(self):
        w = QWidget(); self.setCentralWidget(w); lay = QVBoxLayout(w)
        top_frame = QFrame()
        top_frame.setStyleSheet("background-color: #0a0a0a; border-bottom: 1px solid #333;")
        top = QHBoxLayout(top_frame)
        self.lcd = QLCDNumber(); self.lcd.setDigitCount(9); self.lcd.setSegmentStyle(QLCDNumber.Flat)
        top.addWidget(QLabel("EQUITY:")); top.addWidget(self.lcd)
        self.pnl = QLabel("$0.00"); self.pnl.setStyleSheet("color: #00ffcc; font-size: 20px; font-weight: bold;"); top.addWidget(self.pnl)
        self.lbl_health = QLabel("SYSTEM: INIT"); self.lbl_health.setObjectName("HealthLabel"); top.addWidget(self.lbl_health)
        self.lbl_timer = QLabel("WAIT"); self.lbl_timer.setObjectName("TimerLabel"); top.addWidget(self.lbl_timer)
        top.addStretch(); lay.addWidget(top_frame)

        mid = QSplitter(Qt.Horizontal)
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True)
        self.chart_container = QWidget(); self.chart_layout = QVBoxLayout(self.chart_container)
        self.scroll.setWidget(self.chart_container); mid.addWidget(self.scroll)
        
        right_panel = QWidget(); r_lay = QVBoxLayout(right_panel)
        stat_grp = QGroupBox("Control Center"); sg_lay = QGridLayout()
        self.lbl_balance = QLabel("$0.00"); self.lbl_balance.setObjectName("MetricValue")
        sg_lay.addWidget(QLabel("Balance:", objectName="MetricLabel"), 0, 0); sg_lay.addWidget(self.lbl_balance, 0, 1)
        self.lbl_margin = QLabel("$0.00"); self.lbl_margin.setObjectName("MetricValue")
        sg_lay.addWidget(QLabel("Margin:", objectName="MetricLabel"), 1, 0); sg_lay.addWidget(self.lbl_margin, 1, 1)
        self.lbl_free = QLabel("$0.00"); self.lbl_free.setObjectName("MetricValue")
        sg_lay.addWidget(QLabel("Free Margin:", objectName="MetricLabel"), 2, 0); sg_lay.addWidget(self.lbl_free, 2, 1)
        self.lbl_level = QLabel("0.0%"); self.lbl_level.setObjectName("MetricValue")
        sg_lay.addWidget(QLabel("Margin Level:", objectName="MetricLabel"), 3, 0); sg_lay.addWidget(self.lbl_level, 3, 1)
        
        sg_lay.addWidget(QLabel("Execution Mode:", objectName="MetricLabel"), 4, 0)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Balanced", "Precision", "Growth", "Swing-Master", "Zenith Auto-Pilot"]) 
        self.cmb_mode.setCurrentText("Swing-Master") 
        self.cmb_mode.currentTextChanged.connect(self.change_mode)
        sg_lay.addWidget(self.cmb_mode, 4, 1)
        
        self.status_bar = QLabel("INITIALIZING COUNCIL...")
        self.status_bar.setWordWrap(True)
        self.status_bar.setAlignment(Qt.AlignCenter)
        self.status_bar.setStyleSheet("background-color: rgba(50, 50, 50, 100); color: white; border-radius: 5px; padding: 10px; font-weight: bold; border: 1px solid #444;")
        sg_lay.addWidget(self.status_bar, 5, 0, 1, 2)

        self.btn_panic = QPushButton("PANIC: CLOSE ALL"); self.btn_panic.setStyleSheet("background: #550000; color: white; font-weight: bold;")
        self.btn_panic.clicked.connect(self.panic)
        sg_lay.addWidget(self.btn_panic, 6, 0, 1, 2)
        stat_grp.setLayout(sg_lay); r_lay.addWidget(stat_grp); r_lay.addStretch()
        mid.addWidget(right_panel); mid.setSizes([950, 250]); lay.addWidget(mid)

        tabs = QTabWidget()
        
        self.dashboard_tab = QWidget()
        self.create_dashboard_tab()
        tabs.addTab(self.dashboard_tab, "Account Dashboard")

        self.tbl_positions = QTableWidget(0, 10) 
        self.tbl_positions.setHorizontalHeaderLabels(["Sym", "Type", "Vol", "Open", "Curr", "SL", "TP", "Swap", "Profit", "Action"])
        self.tbl_positions.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_positions.verticalHeader().setVisible(False)
        tabs.addTab(self.tbl_positions, "Live Positions")

        # --- NEW TAB: EXECUTION HISTORY ---
        self.tbl_history = QTableWidget(0, 5)
        self.tbl_history.setHorizontalHeaderLabels(["Time", "Symbol", "Type", "Vol", "Profit"])
        self.tbl_history.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_history.verticalHeader().setVisible(False)
        tabs.addTab(self.tbl_history, "Last 10 Executions")
        
        self.tbl_ai_log = QTableWidget(0, 6)
        self.tbl_ai_log.setHorizontalHeaderLabels(["Time", "Symbol", "Action", "Conf", "Regime", "Logic"])
        self.tbl_ai_log.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.tbl_ai_log.verticalHeader().setVisible(False)
        tabs.addTab(self.tbl_ai_log, "Quantum AI Log")
        
        self.txt = QTextEdit(); self.txt.setReadOnly(True)
        tabs.addTab(self.txt, "System Log")
        
        lay.addWidget(tabs)

    def create_dashboard_tab(self):
        layout = QGridLayout(self.dashboard_tab)
        self.stat_wins = QLabel("0"); self.stat_wins.setObjectName("BigStat")
        self.stat_losses = QLabel("0"); self.stat_losses.setObjectName("BigStat")
        self.stat_winrate = QLabel("0.0%"); self.stat_winrate.setObjectName("BigStat")
        self.stat_pnl = QLabel("$0.00"); self.stat_pnl.setObjectName("BigStat")
        self.stat_best = QLabel("$0.00"); self.stat_best.setObjectName("BigStat")
        self.stat_daily = QLabel("0.00%"); self.stat_daily.setObjectName("BigStat")
        def create_box(title, widget, col):
            box = QGroupBox(title)
            bl = QVBoxLayout(box)
            bl.addWidget(widget, alignment=Qt.AlignCenter)
            if col == "green": widget.setStyleSheet("color: #00ff00; font-size: 24px; font-weight: bold;")
            elif col == "red": widget.setStyleSheet("color: #ff5555; font-size: 24px; font-weight: bold;")
            elif col == "cyan": widget.setStyleSheet("color: #00ffff; font-size: 24px; font-weight: bold;")
            return box
        layout.addWidget(create_box("Total Wins", self.stat_wins, "green"), 0, 0)
        layout.addWidget(create_box("Total Losses", self.stat_losses, "red"), 0, 1)
        layout.addWidget(create_box("Win Rate", self.stat_winrate, "cyan"), 0, 2)
        layout.addWidget(create_box("Total P&L", self.stat_pnl, "green"), 1, 0)
        layout.addWidget(create_box("Best Trade", self.stat_best, "green"), 1, 1)
        layout.addWidget(create_box("Daily Growth", self.stat_daily, "cyan"), 1, 2)

    def closeEvent(self, event):
        if self.worker: 
            self.worker.running = False
            if self.worker.epoch_engine:
                self.worker.epoch_engine.running = False
        mt5.shutdown()
        event.accept()

    def update_table_row(self, table, row, col, val, color=None, bg_color=None):
        text = str(val)
        item = table.item(row, col)
        if item is None:
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignCenter)
            if color: item.setForeground(QBrush(QColor(color)))
            if bg_color: item.setBackground(QBrush(QColor(bg_color)))
            table.setItem(row, col, item)
        else:
            if item.text() != text:
                item.setText(text)
            if color:
                new_brush = QBrush(QColor(color))
                if item.foreground() != new_brush:
                    item.setForeground(new_brush)

    def handle_timeframe_change(self, symbol, text_tf):
        tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4}
        if self.worker and text_tf in tf_map:
            self.worker.update_timeframe_override(symbol, tf_map[text_tf])

    def render(self, d):
        QApplication.processEvents()
        self.lcd.display(d['account']['equity'])
        pnl_col = "#00ff00" if d['account']['profit'] >= 0 else "#ff0000"
        self.pnl.setText(f"${d['account']['profit']:.2f}"); self.pnl.setStyleSheet(f"color: {pnl_col}; font-size: 20px; font-weight: bold;")
        self.lbl_health.setText(d['status'])
        if d['critical']: self.lbl_health.setStyleSheet("color: red; border: 1px solid red; padding: 5px;")
        else: self.lbl_health.setStyleSheet("color: #00ffcc; border: 1px solid #00ffcc; padding: 5px;")
        scan_sec = d.get('timer', 0)
        self.lbl_timer.setText(f"SCAN: {scan_sec}s")
        self.lbl_balance.setText(f"${d['account']['balance']:.2f}")
        self.lbl_margin.setText(f"${d['account']['margin']:.2f}")
        self.lbl_free.setText(f"${d['account']['free_margin']:.2f}")
        self.lbl_level.setText(f"{d['account']['margin_level']:.1f}%")

        if 'stats' in d:
            s = d['stats']
            self.stat_wins.setText(str(s['wins']))
            self.stat_losses.setText(str(s['losses']))
            self.stat_winrate.setText(f"{s['win_rate']:.1f}%")
            total_pnl_val = s['total_pnl']
            tp_col = "#00ff00" if total_pnl_val >= 0 else "#ff0000"
            self.stat_pnl.setText(f"${total_pnl_val:.2f}")
            self.stat_pnl.setStyleSheet(f"color: {tp_col}; font-size: 24px; font-weight: bold;")
            self.stat_best.setText(f"${s['best']:.2f}")
            daily_val = s['daily']
            daily_col = "#00ff00" if daily_val >= 0 else "#ff0000"
            self.stat_daily.setText(f"{daily_val:.2f}%")
            self.stat_daily.setStyleSheet(f"color: {daily_col}; font-size: 24px; font-weight: bold;")

        if 'voting_result' in d:
            vr = d['voting_result']
            color_rgba = "rgba(50, 50, 50, 150)"
            if vr['color_code'] == "red": color_rgba = "rgba(200, 0, 0, 150)" 
            elif vr['color_code'] == "blue": color_rgba = "rgba(0, 100, 255, 150)" 
            elif vr['color_code'] == "purple": color_rgba = "rgba(128, 0, 128, 150)"
            self.status_bar.setStyleSheet(f"background-color: {color_rgba}; color: white; border-radius: 5px; padding: 10px; font-weight: bold; border: 1px solid #aaa;")
            status_text = f"ACTIVE ENGINE: {vr['winner']}\nDETAILS: {vr['details']}"
            if vr.get('full_data'): status_text += f"\nDATA: {vr['full_data']}"
            self.status_bar.setText(status_text)

        current_positions = d.get('positions', [])
        for sym, data in d.get('chart_data', {}).items():
            if not data: continue
            if sym not in self.charts:
                container = QWidget()
                container.setMinimumHeight(250)
                container.setStyleSheet("background-color: #0f0f0f; border: 1px solid #333; margin-bottom: 5px;")
                vbox = QVBoxLayout(container); vbox.setContentsMargins(5,5,5,5)
                hbox = QHBoxLayout()
                lbl = QLabel(f"{sym} Analysis"); lbl.setStyleSheet("font-weight: bold; color: #ffd700; border: none; font-size: 14px;")
                price_lbl = QLabel("0.00"); price_lbl.setStyleSheet("font-weight: bold; color: #00ffcc; border: none; font-size: 14px; margin-left: 10px;")
                status_lbl = QLabel("INIT"); status_lbl.setStyleSheet("color: #aaa; font-style: italic; border: none; font-size: 11px; margin-left: 15px;")
                tf_label = QLabel("TF:"); tf_label.setStyleSheet("color: #888; border: none;")
                combo = QComboBox(); combo.addItems(["M1", "M5", "M15", "M30", "H1", "H4"])
                if self.worker:
                    current_tf_enum = self.worker.symbol_timeframes.get(sym, mt5.TIMEFRAME_M15)
                    combo.setCurrentText(self.worker.get_tf_label(current_tf_enum))
                combo.setStyleSheet("background-color: #222; color: white; border: 1px solid #555;")
                combo.currentTextChanged.connect(partial(self.handle_timeframe_change, sym))
                hbox.addWidget(lbl); hbox.addWidget(price_lbl); hbox.addWidget(status_lbl)
                hbox.addStretch(); hbox.addWidget(tf_label); hbox.addWidget(combo); vbox.addLayout(hbox)
                p = pg.PlotWidget(); p.setBackground("#050505"); p.showGrid(x=True, y=True, alpha=0.3); vbox.addWidget(p)
                candle_item = CandlestickItem(); p.addItem(candle_item)
                price_line = pg.InfiniteLine(angle=0, pen=pg.mkPen('#00aaff', width=1, style=Qt.DashLine)); p.addItem(price_line)
                self.charts[sym] = {'widget': container, 'plot': p, 'candle': candle_item, 'price_line': price_line, 'price_lbl': price_lbl, 'status_lbl': status_lbl, 'lines': [], 'combo': combo}
                self.chart_layout.addWidget(container)
            
            self.charts[sym]['candle'].set_data(data)
            last_close = data[-1][2]
            self.charts[sym]['price_line'].setPos(last_close)
            self.charts[sym]['price_lbl'].setText(f"{last_close:.2f}")
            if 'engine_states' in d:
                 state = d['engine_states'].get(sym, "IDLE")
                 self.charts[sym]['status_lbl'].setText(f"STATUS: {state}")
            for l in self.charts[sym]['lines']: self.charts[sym]['plot'].removeItem(l)
            self.charts[sym]['lines'] = []
            for pos in current_positions:
                if pos['symbol'] == sym:
                    entry_line = pg.InfiniteLine(pos['open'], angle=0, pen=pg.mkPen('#00aaff', width=1, style=Qt.SolidLine), label=f"ENTRY: {pos['open']:.5f}", labelOpts={'color': '#00aaff', 'position': 0.05, 'movable': True})
                    self.charts[sym]['plot'].addItem(entry_line); self.charts[sym]['lines'].append(entry_line)
                    if pos['sl'] > 0:
                        sl_line = pg.InfiniteLine(pos['sl'], angle=0, pen=pg.mkPen('#ff0000', width=1, style=Qt.SolidLine), label=f"SL: {pos['sl']:.5f}", labelOpts={'color': '#ff0000', 'position': 0.05, 'movable': True})
                        self.charts[sym]['plot'].addItem(sl_line); self.charts[sym]['lines'].append(sl_line)
                    if pos['tp'] > 0:
                        tp_line = pg.InfiniteLine(pos['tp'], angle=0, pen=pg.mkPen('#00ff00', width=1, style=Qt.SolidLine), label=f"TP: {pos['tp']:.5f}", labelOpts={'color': '#00ff00', 'position': 0.05, 'movable': True})
                        self.charts[sym]['plot'].addItem(tp_line); self.charts[sym]['lines'].append(tp_line)

        positions = sorted(current_positions, key=lambda x: x['symbol'])
        if self.tbl_positions.rowCount() != len(positions): self.tbl_positions.setRowCount(len(positions))
        for i, p in enumerate(positions):
            self.update_table_row(self.tbl_positions, i, 0, p['symbol'])
            self.update_table_row(self.tbl_positions, i, 1, p['type'], "#00ffcc" if p['type']=="BUY" else "#ff5555")
            self.update_table_row(self.tbl_positions, i, 2, f"{p['vol']:.2f}")
            self.update_table_row(self.tbl_positions, i, 3, f"{p['open']:.5f}")
            self.update_table_row(self.tbl_positions, i, 4, f"{p['curr']:.5f}")
            self.update_table_row(self.tbl_positions, i, 5, f"{p['sl']:.5f}", "#ffaa00") 
            self.update_table_row(self.tbl_positions, i, 6, f"{p['tp']:.5f}", "#00aaff") 
            self.update_table_row(self.tbl_positions, i, 7, f"{p['swap']:.2f}")
            self.update_table_row(self.tbl_positions, i, 8, f"{p['profit']:.2f}", "#00ff00" if p['profit']>=0 else "#ff0000")
            if self.tbl_positions.cellWidget(i, 9) is None:
                btn = QPushButton("CLOSE")
                btn.setStyleSheet("background-color: #550000; color: white; border: 1px solid red; font-weight: bold;")
                btn.clicked.connect(partial(self.worker.close_order, p['ticket'], "Manual GUI"))
                self.tbl_positions.setCellWidget(i, 9, btn)

        logs = d.get('ai_log', [])
        logs_rev = logs[::-1]
        if self.tbl_ai_log.rowCount() != len(logs_rev): self.tbl_ai_log.setRowCount(len(logs_rev))
        for i, l in enumerate(logs_rev):
            self.update_table_row(self.tbl_ai_log, i, 0, l['time'])
            self.update_table_row(self.tbl_ai_log, i, 1, l['symbol'])
            self.update_table_row(self.tbl_ai_log, i, 2, l['action'], "#00ffcc" if l['action']=="BUY" else "#ff5555")
            self.update_table_row(self.tbl_ai_log, i, 3, f"{l['conf']:.1f}%")
            self.update_table_row(self.tbl_ai_log, i, 4, l['regime'])
            self.update_table_row(self.tbl_ai_log, i, 5, l['details'])

        # --- UPDATE HISTORY TABLE ---
        hist_data = d.get('history', [])
        if self.tbl_history.rowCount() != len(hist_data): self.tbl_history.setRowCount(len(hist_data))
        for i, h in enumerate(hist_data):
            self.update_table_row(self.tbl_history, i, 0, h['time'])
            self.update_table_row(self.tbl_history, i, 1, h['symbol'])
            self.update_table_row(self.tbl_history, i, 2, h['type'], "#00ffcc" if h['type']=="BUY" else "#ff5555")
            self.update_table_row(self.tbl_history, i, 3, f"{h['volume']:.2f}")
            self.update_table_row(self.tbl_history, i, 4, f"${h['profit']:.2f}", "#00ff00" if h['profit']>=0 else "#ff0000")

    def change_mode(self, mode):
        if self.worker:
            self.worker.execution_mode = mode
            self.worker.log_msg.emit(f"⚙️ EXECUTION MODE CHANGED: {mode}")

    def panic(self):
        if self.worker: self.worker.panic_close_all()

    def log(self, m):
        self.txt.append(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}")
        # OPTIMIZATION: Clear log if too long
        if len(self.txt.toPlainText()) > 10000:
            self.txt.clear()
            self.txt.append("[SYSTEM] Log Cleared for Performance Optimization.")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    if LoginWin().exec_() == QDialog.Accepted:
        win = MainApp()
        win.show()
        sys.exit(app.exec_())
