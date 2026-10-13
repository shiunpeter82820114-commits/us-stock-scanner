from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import threading
import time
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
import numpy as np
import pandas as pd
import requests

# ================= 1. 從環境變數讀取金鑰 =================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

WATCHLIST = ["TSLA", "NVDA", "AMD", "AAPL", "MSFT", "AMZN", "QQQ", "COIN"]
last_alert_time = {}
ALERT_COOLDOWN_SEC = 1800

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


# ================= 2. 滿足 Render 的輕量 Web Server =================
class HealthCheckHandler(BaseHTTPRequestHandler):

  def do_GET(self):
    self.send_response(200)
    self.end_headers()
    self.wfile.write(b"Bot is running 24/7!")

  def do_HEAD(self):
    # 專門回應 UptimeRobot 的 HEAD 請求，避免被判定 501 錯誤
    self.send_response(200)
    self.end_headers()

  def log_message(self, format, *args):
    return


def start_web_server():
  port = int(os.environ.get("PORT", 10000))
  server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
  server.serve_forever()


# ================= 3. 策略與推播運算 =================
def send_discord_alert(ticker, l1, l2, current_price, vwap):
  stop_loss = round(l2 * 0.998, 2)
  payload = {
      "content": (
          f"⚡ **【美股短線】W底右側啟動：{ticker}** ⚡\n"
          f"• 當前價格: `${current_price:.2f}`\n"
          f"• 左底 (L1): `${l1:.2f}` | 右底 (L2): `${l2:.2f}`\n"
          f"• 即時 VWAP: `${vwap:.2f}` (已向上站上)\n"
          f"• 參考防守停損: `${stop_loss}` (跌破右底)\n"
          f"• 觸發時間: {datetime.now().strftime('%H:%M:%S')}"
      )
  }
  try:
    if DISCORD_WEBHOOK_URL.startswith("http"):
      requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 發送通知: {ticker}")
  except Exception as err:
    print(f"推播失敗: {err}")


def calculate_vwap(df):
  typical = (df["high"] + df["low"] + df["close"]) / 3
  return (typical * df["volume"]).cumsum() / df["volume"].cumsum()


def evaluate_w_bottom(bars_df, tolerance=0.015):
  if len(bars_df) < 25:
    return False, None
  df = bars_df.copy()
  df["vwap"] = calculate_vwap(df)
  recent = df.tail(25).reset_index(drop=True)
  lows = recent["low"].values
  highs = recent["high"].values
  closes = recent["close"].values
  vwap = recent["vwap"].values

  l1_idx = int(np.argmin(lows[3:15])) + 3
  l1 = lows[l1_idx]
  peak_idx = int(np.argmax(highs[l1_idx : len(highs) - 3])) + l1_idx
  peak = highs[peak_idx]
  if (peak - l1) / l1 < 0.012:
    return False, None

  l2_idx = int(np.argmin(lows[peak_idx:])) + peak_idx
  l2 = lows[l2_idx]
  if abs(l2 - l1) / l1 > tolerance:
    return False, None

  curr_close = closes[-1]
  curr_vwap = vwap[-1]
  prev_close = closes[-2]
  is_recent = (len(lows) - 1 - l2_idx) <= 3
  cross_vwap = curr_close >= curr_vwap and prev_close <= curr_vwap * 1.002

  if is_recent and cross_vwap and curr_close < peak:
    return True, {
        "l1": l1,
        "l2": l2,
        "price": curr_close,
        "vwap": curr_vwap,
    }
  return False, None


def run_scan_cycle():
  # 修正：最新版 alpaca-py 的 5 分 K 正確寫法
  req = StockBarsRequest(
      symbol_or_symbols=WATCHLIST,
      timeframe=TimeFrame(5, TimeFrameUnit.Minute),
      limit=50,
  )
  try:
    bars_data = client.get_stock_bars(req)
  except Exception as e:
    print(f"API 請求失敗: {e}")
    return

  # 兼容量化物件與 dict 結構
  bars_dict = bars_data.data if hasattr(bars_data, "data") else bars_data
  now = time.time()

  for ticker in WATCHLIST:
    bars = bars_dict.get(ticker, [])
    if not bars:
      continue
    df = pd.DataFrame([
        {
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ])
    matched, info = evaluate_w_bottom(df)
    if matched and now - last_alert_time.get(ticker, 0) > ALERT_COOLDOWN_SEC:
      send_discord_alert(
          ticker, info["l1"], info["l2"], info["price"], info["vwap"]
      )
      last_alert_time[ticker] = now


if __name__ == "__main__":
  threading.Thread(target=start_web_server, daemon=True).start()
  print("美股短線 W 底雲端監控中...", flush=True)

  while True:
    run_scan_cycle()
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 完成一輪 5分K"
        " 掃描，等待下一根收線...",
        flush=True,
    )
    time.sleep(300)
