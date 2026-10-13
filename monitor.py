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

# ================= 1. 環境變數讀取 =================
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
DISCORD_WEBHOOK_URL = os.environ.get("DISCORD_WEBHOOK_URL", "")

# 美股市值與流動性前 50 大標的清單
WATCHLIST = [
    # 科技與半導體巨頭
    "AAPL",
    "MSFT",
    "NVDA",
    "AMZN",
    "GOOGL",
    "META",
    "TSLA",
    "AVGO",
    "ORCL",
    "NFLX",
    "AMD",
    "QCOM",
    "CSCO",
    "IBM",
    "TXN",
    "AMAT",
    "INTU",
    "NOW",
    "CRM",
    "UBER",
    # 金融巨頭
    "JPM",
    "V",
    "MA",
    "BAC",
    "WFC",
    "MS",
    "GS",
    # 零售與民生消費
    "WMT",
    "COST",
    "HD",
    "PG",
    "KO",
    "PEP",
    "MCD",
    "PM",
    "DIS",
    # 醫藥健康
    "UNH",
    "JNJ",
    "ABBV",
    "MRK",
    "TMO",
    "ABT",
    "DHR",
    # 能源、工業與電信
    "XOM",
    "CVX",
    "LIN",
    "ACN",
    "GE",
    "CAT",
    "VZ",
]

last_alert_time = {}
ALERT_COOLDOWN_SEC = 1800  # 同一標的 30 分鐘內不重複發送

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)


# ================= 2. 滿足 Render 與 UptimeRobot 的 Web 服務 =================
class HealthCheckHandler(BaseHTTPRequestHandler):

  def do_GET(self):
    self.send_response(200)
    self.end_headers()
    self.wfile.write(b"Bot is running 24/7! Monitoring Top 50 Stocks.")

  def do_HEAD(self):
    self.send_response(200)
    self.end_headers()

  def log_message(self, format, *args):
    return


def start_web_server():
  port = int(os.environ.get("PORT", 10000))
  server = HTTPServer(("0.0.0.0", port), HealthCheckHandler)
  server.serve_forever()


# ================= 3. 策略與推播邏輯 =================
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
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 觸發通知: {ticker}",
        flush=True,
    )
  except Exception as err:
    print(f"推播失敗: {err}", flush=True)


def calculate_vwap(df):
  typical = (df["high"] + df["low"] + df["close"]) / 3
  return (typical * df["volume"]).cumsum() / df["volume"].cumsum()


def evaluate_w_bottom(bars_df, tolerance=0.015):
  """檢測 5分K 雙底且最新價格突破/踩穩 VWAP"""
  if len(bars_df) < 25:
    return False, None

  df = bars_df.copy()
  df["vwap"] = calculate_vwap(df)

  recent = df.tail(25).reset_index(drop=True)
  lows = recent["low"].values
  highs = recent["high"].values
  closes = recent["close"].values
  vwap = recent["vwap"].values

  # 1. 抓取左底 L1
  l1_idx = int(np.argmin(lows[3:15])) + 3
  l1 = lows[l1_idx]

  # 2. 抓取反彈頸線高點 (至少反彈 1.2%)
  peak_idx = int(np.argmax(highs[l1_idx : len(highs) - 3])) + l1_idx
  peak = highs[peak_idx]
  if (peak - l1) / l1 < 0.012:
    return False, None

  # 3. 抓取右底 L2 (誤差幅度在 1.5% 內)
  l2_idx = int(np.argmin(lows[peak_idx:])) + peak_idx
  l2 = lows[l2_idx]
  if abs(l2 - l1) / l1 > tolerance:
    return False, None

  curr_close = closes[-1]
  curr_vwap = vwap[-1]
  prev_close = closes[-2]

  # 4. 判斷右側啟動條件
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
  now = time.time()
  # 將 50 檔股票分成每組 10 檔批次查詢，避免單次 API 棒線總數被上限截斷
  chunk_size = 10
  for i in range(0, len(WATCHLIST), chunk_size):
    chunk = WATCHLIST[i : i + chunk_size]
    req = StockBarsRequest(
        symbol_or_symbols=chunk,
        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
        limit=500,  # 10 檔 x 50 根 = 500 根充足額度
    )
    try:
      bars_data = client.get_stock_bars(req)
    except Exception as e:
      print(f"批次查詢失敗 ({chunk[0]}...): {e}", flush=True)
      continue

    bars_dict = bars_data.data if hasattr(bars_data, "data") else bars_data

    for ticker in chunk:
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
  print("美股短線 W 底雲端監控啟動！監控標的：前 50 大股票", flush=True)

  while True:
    run_scan_cycle()
    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] 完成前 50 大股票 5分K"
        " 掃描，等待下一根收線...",
        flush=True,
    )
    time.sleep(300)
