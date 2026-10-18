import os
import time
import threading
from datetime import datetime
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
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

# 美股市值與流動性前 50 大標的
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "ORCL", "NFLX",
    "AMD", "QCOM", "CSCO", "IBM", "TXN", "AMAT", "INTU", "NOW", "CRM", "UBER",
    "JPM", "V", "MA", "BAC", "WFC", "MS", "GS", "WMT", "COST", "HD",
    "PG", "KO", "PEP", "MCD", "PM", "DIS", "UNH", "JNJ", "ABBV", "MRK",
    "TMO", "ABT", "DHR", "XOM", "CVX", "LIN", "ACN", "GE", "CAT", "VZ"
]

last_alert_time = {}
ALERT_COOLDOWN_SEC = 1800  # 同一標的 30 分鐘冷卻

client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)

# ================= 2. 多執行緒 Web 服務 (防止 502 堵塞) =================
class HealthCheckHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(b"Bot is running 24/7! Pro Technical Analysis v3 Active.")

    def do_HEAD(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Connection", "close")
        self.end_headers()

    def log_message(self, format, *args):
        return

def start_web_server():
    try:
        port = int(os.environ.get("PORT", 10000))
        server = ThreadingHTTPServer(('0.0.0.0', port), HealthCheckHandler)
        server.serve_forever()
    except Exception as e:
        print(f"Web 伺服器異常: {e}", flush=True)

# ================= 3. 技術指標與量價運算核心 =================
def enrich_technical_indicators(df):
    """計算當日日內 VWAP、EMA20、RSI(14) 與 ATR(14)"""
    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data["date"] = data["timestamp"].dt.date

    # 1. 當日日內 VWAP (每日開盤歸零重新累加)
    data["typical_price"] = (data["high"] + data["low"] + data["close"]) / 3
    data["tp_vol"] = data["typical_price"] * data["volume"]
    data["cum_vol"] = data.groupby("date")["volume"].cumsum()
    data["cum_tp_vol"] = data.groupby("date")["tp_vol"].cumsum()
    data["vwap"] = np.where(data["cum_vol"] > 0, data["cum_tp_vol"] / data["cum_vol"], data["close"])

    # 2. 指數移動平均線 EMA20 (短線趨勢生命線)
    data["ema20"] = data["close"].ewm(span=20, adjust=False).mean()

    # 3. 相對強弱指標 RSI(14)
    delta = data["close"].diff()
    gain = delta.clip(lower=0).ewm(alpha=1/14, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1/14, adjust=False).mean()
    rs = gain / np.where(loss == 0, 1e-9, loss)
    data["rsi"] = np.where(loss == 0, 100.0, 100.0 - (100.0 / (1.0 + rs)))

    # 4. 平均真實波幅 ATR(14) (用於動態防洗盤停損)
    high_low = data["high"] - data["low"]
    high_close = (data["high"] - data["close"].shift(1)).abs()
    low_close = (data["low"] - data["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    data["atr"] = tr.rolling(14, min_periods=1).mean()

    return data

def get_market_context():
    """獲取 QQQ 大盤狀態，作為相對強弱度 (RS) 與市場環境基準"""
    context = {"qqq_bullish": True, "qqq_1h_ret": 0.0, "status_text": "⚖️ 中性震盪"}
    try:
        req = StockBarsRequest(
            symbol_or_symbols=["QQQ"],
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            limit=50
        )
        bars_data = client.get_stock_bars(req)
        bars_dict = bars_data.data if hasattr(bars_data, 'data') else bars_data
        bars = bars_dict.get("QQQ", [])
        if len(bars) >= 15:
            df = pd.DataFrame([{
                "timestamp": b.timestamp, "open": b.open, "high": b.high,
                "low": b.low, "close": b.close, "volume": b.volume
            } for b in bars])
            df = enrich_technical_indicators(df)
            curr_close = df["close"].iloc[-1]
            curr_vwap = df["vwap"].iloc[-1]
            curr_ema20 = df["ema20"].iloc[-1]
            ret_1h = (curr_close - df["close"].iloc[-12]) / df["close"].iloc[-12]
            context["qqq_1h_ret"] = ret_1h

            if curr_close >= curr_vwap and curr_close >= curr_ema20:
                context["qqq_bullish"] = True
                context["status_text"] = f"🟢 多頭順風 (QQQ 站上 VWAP, 近1h `{ret_1h*100:+.2f}%`)"
            elif curr_close < curr_vwap and ret_1h < -0.005:
                context["qqq_bullish"] = False
                context["status_text"] = f"🔴 大盤偏弱 (QQQ 跌破 VWAP, 近1h `{ret_1h*100:+.2f}%` ⚠️嚴控倉位)"
            else:
                context["qqq_bullish"] = True
                context["status_text"] = f"🟡 盤整觀望 (QQQ 近1h `{ret_1h*100:+.2f}%`)"
    except Exception:
        pass
    return context

def evaluate_w_bottom(bars_df, market_ctx, tolerance=0.015):
    try:
        if len(bars_df) < 25:
            return False, None

        df = enrich_technical_indicators(bars_df)
        recent = df.tail(25).reset_index(drop=True)

        opens = recent["open"].values
        lows = recent["low"].values
        highs = recent["high"].values
        closes = recent["close"].values
        vwaps = recent["vwap"].values
        volumes = recent["volume"].values
        ema20s = recent["ema20"].values
        rsis = recent["rsi"].values
        atrs = recent["atr"].values

        if len(lows) < 20:
            return False, None

        # 1. 抓取左底 L1
        l1_idx = int(np.argmin(lows[3:15])) + 3
        l1 = lows[l1_idx]
        l1_vol = volumes[l1_idx]

        # 2. 抓取反彈頸線高點 Peak (至少反彈 1.2%)
        peak_slice = highs[l1_idx:len(highs)-3]
        if len(peak_slice) == 0:
            return False, None

        peak_idx = int(np.argmax(peak_slice)) + l1_idx
        peak = highs[peak_idx]
        if (peak - l1) / l1 < 0.012:
            return False, None

        # 3. 抓取右底 L2 (誤差在 1.5% 內)
        l2_slice = lows[peak_idx:]
        if len(l2_slice) == 0:
            return False, None

        l2_idx = int(np.argmin(l2_slice)) + peak_idx
        l2 = lows[l2_idx]
        l2_vol = volumes[l2_idx]
        if abs(l2 - l1) / l1 > tolerance:
            return False, None

        curr_open = opens[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]
        curr_close = closes[-1]
        curr_vwap = vwaps[-1]
        prev_close = closes[-2]
        curr_vol = volumes[-1]
        curr_ema20 = ema20s[-1]
        curr_rsi = rsis[-1]
        curr_atr = atrs[-1]

        prev_volumes = volumes[-6:-1]
        avg_vol = np.mean(prev_volumes) if len(prev_volumes) > 0 else 0

        # ================= 硬性生存濾網 (淘汰陷阱與假突破) =================
        # 條件 A: 右底剛在最近 15 分鐘內成型，且突破站上當日 VWAP
        is_recent = (len(lows) - 1 - l2_idx) <= 3
        cross_vwap = (curr_close >= curr_vwap) and (prev_close <= curr_vwap * 1.003)
        volume_confirmed = (curr_vol >= avg_vol * 1.15) if avg_vol > 0 else True
        not_overextended = curr_close < peak

        if not (is_recent and cross_vwap and volume_confirmed and not_overextended):
            return False, None

        # 條件 B: K棒實體檢查 (必須收紅K，且拒絕長上影線避雷針)
        candle_range = curr_high - curr_low
        is_bullish_candle = (curr_close > curr_open) and (
            (curr_close - curr_low) / candle_range >= 0.5 if candle_range > 0 else True
        )
        if not is_bullish_candle:
            return False, None

        # 條件 C: RSI 不得極端超買 (>75 不追高)
        if curr_rsi > 75:
            return False, None

        # 條件 D: ATR 動態防洗盤停損與最低風報比檢查 (R:R >= 1.5 才出手)
        atr_stop = l2 - (0.5 * curr_atr)
        pct_stop = l2 * 0.998
        stop_loss = round(min(atr_stop, pct_stop), 2)
        risk = curr_close - stop_loss
        if risk <= 0:
            return False, None

        pattern_height = peak - l2
        tp1 = round(peak, 2)
        tp2 = round(peak + pattern_height, 2)
        reward = tp2 - curr_close
        rr_ratio = round(reward / risk, 1)

        if rr_ratio < 1.5:
            return False, None

        # ================= 實戰技術評分與細節診斷 =================
        score = 60  # 基礎合格分
        checklist = []

        # 1. 底部型態判定 (底底高 vs 破底翻)
        if l2 >= l1:
            pattern_type = "🔥 強勢底底高 (Higher Low)"
            score += 10
            checklist.append("✅ 右腳高於左腳 (買盤積極墊高)")
        else:
            pattern_type = "⚡ 破底翻洗盤 (Wyckoff Spring)"
            score += 8
            checklist.append("✅ 假跌破左底後迅速收復 (洗出浮額)")

        # 2. 威科夫右底量縮檢查 (賣壓枯竭)
        vol_dry_up = l2_vol < l1_vol
        if vol_dry_up:
            score += 10
            checklist.append("✅ 右底測試量縮 (賣壓枯竭確認)")
        else:
            checklist.append("⚪ 右底未明顯量縮 (留意防守位)")

        # 3. 均線共振 (站上 20 EMA 生命線)
        above_ema20 = curr_close >= curr_ema20
        if above_ema20:
            score += 10
            checklist.append("✅ 股價同步站上 20 EMA 均線")
        else:
            checklist.append("⚠️ 尚未站上 20 EMA (上方仍有短壓)")

        # 4. 相對強弱度 (RS vs QQQ 大盤)
        stock_1h_ret = (curr_close - closes[-12]) / closes[-12] if len(closes) >= 12 else 0.0
        rs_diff = stock_1h_ret - market_ctx.get("qqq_1h_ret", 0.0)
        if rs_diff > 0:
            score += 10
            rs_text = f"💪 強於大盤 (RS `{rs_diff*100:+.2f}%`)"
            checklist.append("✅ 相對強弱度跑贏大盤 QQQ")
        else:
            rs_text = f"📉 弱於大盤 (RS `{rs_diff*100:+.2f}%`)"
            checklist.append("⚠️ 個股近1小時表現落後大盤")

        # 綜合評級判定
        if score >= 90:
            grade = "🏆 S 級 (極高勝率・多維度完美共振)"
        elif score >= 80:
            grade = "⭐ A 級 (標準右側好球帶)"
        else:
            grade = "🔹 B 級 (基本達標・建議輕倉試單)"

        vol_ratio = round(curr_vol / avg_vol, 2) if avg_vol > 0 else 1.0
        # 倉位控管試算：假設單筆交易最高只願意賠 $100 USD
        shares_per_100_risk = int(100 / risk) if risk > 0 else 0

        return True, {
            "l1": l1,
            "l2": l2,
            "peak": peak,
            "price": curr_close,
            "vwap": curr_vwap,
            "ema20": curr_ema20,
            "rsi": round(curr_rsi, 1),
            "atr": round(curr_atr, 2),
            "volume_ratio": vol_ratio,
            "stop_loss": stop_loss,
            "tp1": tp1,
            "tp2": tp2,
            "rr_ratio": rr_ratio,
            "pattern_type": pattern_type,
            "rs_text": rs_text,
            "grade": grade,
            "score": score,
            "checklist": "\n".join(checklist),
            "shares_100": shares_per_100_risk,
            "market_status": market_ctx.get("status_text", "")
        }
    except Exception:
        return False, None

def send_discord_alert(ticker, info):
    payload = {
        "content": (
            f"⚡ **【美股短線】W底右側啟動：{ticker}** ⚡\n"
            f"🏅 **訊號評級**: **{info['grade']}** (`{info['score']}分`)\n"
            f"📊 **大盤環境**: {info['market_status']}\n"
            f"-----------------------------------------\n"
            f"🔍 **【技術結構與動能診斷】**\n"
            f"• **型態分類**: {info['pattern_type']}\n"
            f"• **當前價格**: `${info['price']:.2f}` (站上 VWAP `${info['vwap']:.2f}` | 20EMA `${info['ema20']:.2f}`)\n"
            f"• **雙底支撐**: 左底 (L1) `${info['l1']:.2f}` | 右底 (L2) `${info['l2']:.2f}`\n"
            f"• **頸線反壓**: `${info['peak']:.2f}`\n"
            f"• **量價動能**: 攻擊放量 `{info['volume_ratio']}x` | RSI(14): `{info['rsi']}` | {info['rs_text']}\n"
            f"{info['checklist']}\n"
            f"-----------------------------------------\n"
            f"🎯 **【出場規劃與生存風控】**\n"
            f"• 🛑 **ATR 防洗盤停損 (SL)**: `${info['stop_loss']:.2f}` (結合右底與 `0.5x ATR` 緩衝)\n"
            f"• 🚩 **第一止盈 (TP1)**: `${info['tp1']:.2f}` (頸線測試，建議減倉 1/2 並將停損移至成本價)\n"
            f"• 🚀 **測幅目標 (TP2)**: `${info['tp2']:.2f}` (突破頸線等距滿足點)\n"
            f"• ⚖️ **預估風報比**: `1 : {info['rr_ratio']}`\n"
            f"• 🛡️ **生存倉位上限**: 每承受 `$100 USD` 風險，建議最多買入 **`{info['shares_100']} 股`**\n"
            f"• 🕒 **觸發時間**: {datetime.now().strftime('%H:%M:%S')} (UTC)"
        )
    }
    try:
        if DISCORD_WEBHOOK_URL.startswith("http"):
            requests.post(DISCORD_WEBHOOK_URL, json=payload, timeout=5)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] 🚨 觸發高精準通知: {ticker} ({info['grade']})", flush=True)
    except Exception as err:
        print(f"推播失敗: {err}", flush=True)

def run_scan_cycle():
    now = time.time()
    # 先取得大盤 QQQ 的即時多空與強弱基準
    market_ctx = get_market_context()

    chunk_size = 10
    for i in range(0, len(WATCHLIST), chunk_size):
        chunk = WATCHLIST[i:i + chunk_size]
        try:
            req = StockBarsRequest(
                symbol_or_symbols=chunk,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                limit=500
            )
            bars_data = client.get_stock_bars(req)
        except Exception as e:
            print(f"API 請求逾時 ({chunk[0]}...): {e}", flush=True)
            continue

        bars_dict = bars_data.data if hasattr(bars_data, 'data') else bars_data

        for ticker in chunk:
            try:
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

                matched, info = evaluate_w_bottom(df, market_ctx)
                if matched and (now - last_alert_time.get(ticker, 0) > ALERT_COOLDOWN_SEC):
                    send_discord_alert(ticker, info)
                    last_alert_time[ticker] = now
            except Exception:
                continue

if __name__ == "__main__":
    web_thread = threading.Thread(target=start_web_server, daemon=True)
    web_thread.start()
    print("美股短線 W 底雲端監控啟動！Pro v3 (RS相對強弱 + 威科夫量價 + ATR動態風控) 運行中...", flush=True)

    while True:
        try:
            run_scan_cycle()
            print(f"[{datetime.now().strftime('%H:%M:%S')}] 完成前 50 大股票 Pro v3 掃描，伺服器健康運行中...", flush=True)
        except Exception as cycle_err:
            print(f"掃描週期異常（已自動恢復）: {cycle_err}", flush=True)
        time.sleep(300)
