import os
import time
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
import requests
import pandas as pd
import numpy as np
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# ================= 1. 環境變數讀取 =================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "NFLX",
    "AMD", "QCOM", "CSCO", "IBM", "TXN", "AMAT", "INTU", "NOW", "CRM", "UBER",
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "WMT", "COST", "HD",
    "PG", "KO", "PEP", "MCD", "PM", "DIS", "UNH", "JNJ", "ABBV", "MRK",
    "TMO", "ABT", "DHR", "XOM", "CVX", "LIN", "ACN", "GE", "CAT", "VZ"
]

last_alert_time = {}
ALERT_COOLDOWN_SEC = 1800

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ================= 2. 滿足 Render 與 UptimeRobot 的 Web 服務 =================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"Bot is running 24/7! Technical Analysis Rules v2 Active.")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def log_message(self, format, *args):
        return

def start_web_server():
    port = int(os.environ.get("PORT", 10000))
    server = HTTPServer(('0.0.0.0', port), HealthCheckHandler)
    server.serve_forever()

# ================= 3. 策略與推播邏輯 =================
def send_discord_alert(ticker, l1, l2, current_price, vwap, vol_ratio=1.0):
    stop_loss = round(l2 * 0.998, 2)
    payload = {
        "content": (
            f"⚡ **【美股短線】W底右側啟動：{ticker}** ⚡\n"
            f"• 當前價格: `${current_price:.2f}`\n"
            f"• 左底 (L1): `${l1:.2f}` | 右底 (L2): `${l2:.2f}`\n"
            f"• 當日日內 VWAP: `${vwap:.2f}` (放量站上)\n"
            f"• 量能動態: 放量 `{vol_ratio}x` (突破量 > 前5棒均量)\n"
            f"• 參考防守停損: `${stop_loss}` (跌破右底停損)\n"
            f"• 觸發時間: {datetime.now().strftime('%H:%M:%S')}"
        )
    }
    try:
        if DISCORD_WEBHOOK_URL.startswith("http"):
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 觸發通知: {ticker}", flush=True)
    except Exception as err:
        print(f"推播失敗: {err}", flush=True)

def calculate_intraday_vwap(df):
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date
    
    data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3
    data["tp_vol"] = data["typical_price"] * data["volume"]
    
    data["cum_vol"] = data.groupby("date")["volume"].cumsum()
    data["cum_tp_vol"] = data.groupby("date")["tp_vol"].cumsum()
    
    data["vwap"] = np.where(data["cum_vol"] > 0, data["cum_tp_vol"] / data["cum_vol"], data["close"])
    return data

def evaluate_w_bottom(bars_df, tolerance=0.015):
    if len(bars_df) < 25:
        return False, None
    
    df = calculate_intraday_vwap(bars_df)
    recent = df.tail(25).reset_index(drop=True)
    
    lows = recent["low"].values
    highs = recent["high"].values
    closes = recent["close"].values
    vwaps = recent["vwap"].values
    volumes = recent["volume"].values
    
    l1_idx = int(np.argmin(lows[3:15])) + 3
    l1 = lows[l1_idx]
    
    peak_idx = int(np.argmax(highs[l1_idx:len(highs)-3])) + l1_idx
    peak = highs[peak_idx]
    if (peak - l1) / l1 < 0.012:
        return False, None
    
    l2_idx = int(np.argmin(lows[peak_idx:])) + peak_idx
    l2 = lows[l2_idx]
    if abs(l2 - l1) / l1 > tolerance:
        return False, None
        
    curr_close = closes[-1]
    curr_vwap = vwaps[-1]
    prev_close = closes[-2]
    curr_vol = volumes[-1]
    
    prev_volumes = volumes[-6:-1]
    avg_vol = np.mean(prev_volumes) if len(prev_volumes) > 0 else 0
    
    is_recent = (len(lows) - 1 - l2_idx) <= 3
    cross_vwap = (curr_close >= curr_vwap) and (prev_close <= curr_vwap * 1.002)
    volume_confirmed = (curr_vol >= avg_vol * 1.15) if avg_vol > 0 else True
    not_overextended = curr_close < peak
    
    if is_recent and cross_vwap and volume_confirmed and not_overextended:
        vol_ratio = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        return True, {
            "l1": l1,
            "l2": l2,
            "price": curr_close,
            "vwap": curr_vwap,
            "volume_ratio": vol_ratio
        }
        
    return False, None

def run_scan_cycle():
    now = time.time()
    chunk_size = 10
    for i in range(0, len(WATCHLIST), chunk_size):
        chunk = WATCHLIST[i:i + chunk_size]
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            limit=500
        )
        try:
            bars_data = client.get_stock_bars(req)
        except Exception as e:
            print(f"批次查詢失敗 ({chunk[0]}...): {e}", flush=True)
            continue
            
        bars_dict = bars_data.data if hasattr(bars_data, 'data') else bars_data
        
        for ticker in chunk:
            bars = bars_dict.get(ticker, [])
            if not bars:
                continue
                
            df = pd.DataFrame([{
                "timestamp": b.timestamp,
                "open": b.open,
                "high": b.high,
                "low": b.low,
                "close": b.close,
                "volume": b.volume
            } for b in bars])
            
            matched, info = evaluate_w_bottom(df)
            if matched and (now - last_alert_time.get(ticker, 0) > ALERT_COOLDOWN_SEC):
                send_discord_alert(ticker, info["l1"], info["l2"], info["price"], info["vwap"], info["volume_ratio"])
                last_alert_time[ticker] = now

if __name__ == "__main__":
    threading.Thread(target=start_web_server, daemon=True).start()
    print("美股短線 W 底雲端監控啟動！技術分析 v2 (當日VWAP + 放量確認) 運行中...", flush=True)
    
    while True:
        run_scan_cycle()
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成前 50 大股票 5分K 掃描，等待下一根收線...", flush=True)
        time.sleep(300)
