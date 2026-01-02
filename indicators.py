import numpy as np

# ==================== TREND INDICATORS ====================

def calculate_sma(prices, period):
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period

def calculate_ema(prices, period):
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema

def calculate_wma(prices, period):
    if len(prices) < period:
        return None
    weights = list(range(1, period + 1))
    weighted_sum = sum(p * w for p, w in zip(prices[-period:], weights))
    return weighted_sum / sum(weights)

def calculate_adx(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None, None, None
    
    plus_dm, minus_dm, tr_list = [], [], []
    for i in range(1, len(closes)):
        high_diff = highs[i] - highs[i-1]
        low_diff = lows[i-1] - lows[i]
        plus_dm.append(high_diff if high_diff > low_diff and high_diff > 0 else 0)
        minus_dm.append(low_diff if low_diff > high_diff and low_diff > 0 else 0)
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
    
    atr = sum(tr_list[-period:]) / period
    plus_di = (sum(plus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
    minus_di = (sum(minus_dm[-period:]) / period) / atr * 100 if atr > 0 else 0
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
    return dx, plus_di, minus_di


# ==================== MOMENTUM OSCILLATORS ====================

def calculate_rsi(prices, period=14):
    if len(prices) < period + 1:
        return None
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(prices, fast=12, slow=26, signal=9):
    if len(prices) < slow:
        return None, None, None
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(prices[-signal:], signal) if len(prices) >= signal else macd_line
    histogram = macd_line - signal_line if signal_line else 0
    return macd_line, signal_line, histogram

def calculate_stochastic(highs, lows, closes, period=14):
    if len(closes) < period:
        return None, None
    lowest_low = min(lows[-period:])
    highest_high = max(highs[-period:])
    if highest_high == lowest_low:
        return 50, 50
    k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
    d = k
    return k, d

def calculate_williams_r(highs, lows, closes, period=14):
    if len(closes) < period:
        return None
    highest_high = max(highs[-period:])
    lowest_low = min(lows[-period:])
    if highest_high == lowest_low:
        return -50
    return ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100

def calculate_cci(highs, lows, closes, period=20):
    if len(closes) < period:
        return None
    tp = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    sma_tp = sum(tp[-period:]) / period
    mean_dev = sum(abs(t - sma_tp) for t in tp[-period:]) / period
    if mean_dev == 0:
        return 0
    return (tp[-1] - sma_tp) / (0.015 * mean_dev)

def calculate_momentum(prices, period=10):
    if len(prices) < period + 1:
        return None
    return prices[-1] - prices[-period-1]

def calculate_roc(prices, period=10):
    if len(prices) < period + 1:
        return None
    return ((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100


# ==================== VOLATILITY INDICATORS ====================

def calculate_bollinger(prices, period=20, std_dev=2):
    if len(prices) < period:
        return None, None, None
    sma = calculate_sma(prices, period)
    std = np.std(prices[-period:])
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, sma, lower

def calculate_atr(highs, lows, closes, period=14):
    if len(closes) < period + 1:
        return None
    trs = []
    for i in range(1, len(closes)):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        trs.append(tr)
    return sum(trs[-period:]) / period

def calculate_keltner(highs, lows, closes, period=20, atr_mult=2):
    if len(closes) < period:
        return None, None, None
    ema = calculate_ema(closes, period)
    atr = calculate_atr(highs, lows, closes, period)
    if not ema or not atr:
        return None, None, None
    upper = ema + (atr * atr_mult)
    lower = ema - (atr * atr_mult)
    return upper, ema, lower

def calculate_donchian(highs, lows, period=20):
    if len(highs) < period:
        return None, None, None
    upper = max(highs[-period:])
    lower = min(lows[-period:])
    middle = (upper + lower) / 2
    return upper, middle, lower

# ==================== ICHIMOKU CLOUD ====================

def calculate_ichimoku(highs, lows, closes):
    if len(closes) < 52:
        return None
    
    tenkan = (max(highs[-9:]) + min(lows[-9:])) / 2
    kijun = (max(highs[-26:]) + min(lows[-26:])) / 2
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (max(highs[-52:]) + min(lows[-52:])) / 2
    chikou = closes[-1]
    
    return {
        "tenkan": tenkan,
        "kijun": kijun,
        "senkou_a": senkou_a,
        "senkou_b": senkou_b,
        "chikou": chikou,
        "cloud": "bullish" if senkou_a > senkou_b else "bearish",
        "price_vs_cloud": "above" if closes[-1] > max(senkou_a, senkou_b) else "below" if closes[-1] < min(senkou_a, senkou_b) else "inside"
    }


# ==================== PIVOT POINTS ====================

def calculate_pivot_points(high, low, close):
    """Calculate Standard Pivot Points"""
    pivot = (high + low + close) / 3
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)
    
    return {
        "pivot": pivot,
        "r1": r1, "r2": r2, "r3": r3,
        "s1": s1, "s2": s2, "s3": s3
    }

def calculate_fibonacci_pivots(high, low, close):
    """Calculate Fibonacci Pivot Points"""
    pivot = (high + low + close) / 3
    diff = high - low
    
    return {
        "pivot": pivot,
        "r1": pivot + (0.382 * diff),
        "r2": pivot + (0.618 * diff),
        "r3": pivot + (1.0 * diff),
        "s1": pivot - (0.382 * diff),
        "s2": pivot - (0.618 * diff),
        "s3": pivot - (1.0 * diff)
    }

def calculate_camarilla_pivots(high, low, close):
    """Calculate Camarilla Pivot Points"""
    diff = high - low
    
    return {
        "r4": close + (diff * 1.1/2),
        "r3": close + (diff * 1.1/4),
        "r2": close + (diff * 1.1/6),
        "r1": close + (diff * 1.1/12),
        "s1": close - (diff * 1.1/12),
        "s2": close - (diff * 1.1/6),
        "s3": close - (diff * 1.1/4),
        "s4": close - (diff * 1.1/2)
    }

# ==================== FIBONACCI RETRACEMENT ====================

def calculate_fibonacci_retracement(high, low, trend="bullish"):
    """Calculate Fibonacci Retracement Levels"""
    diff = high - low
    
    if trend == "bullish":
        return {
            "0.0": low,
            "0.236": low + (diff * 0.236),
            "0.382": low + (diff * 0.382),
            "0.5": low + (diff * 0.5),
            "0.618": low + (diff * 0.618),
            "0.786": low + (diff * 0.786),
            "1.0": high
        }
    else:
        return {
            "0.0": high,
            "0.236": high - (diff * 0.236),
            "0.382": high - (diff * 0.382),
            "0.5": high - (diff * 0.5),
            "0.618": high - (diff * 0.618),
            "0.786": high - (diff * 0.786),
            "1.0": low
        }

def calculate_fibonacci_extension(high, low, trend="bullish"):
    """Calculate Fibonacci Extension Levels"""
    diff = high - low
    
    if trend == "bullish":
        return {
            "1.0": high,
            "1.272": high + (diff * 0.272),
            "1.414": high + (diff * 0.414),
            "1.618": high + (diff * 0.618),
            "2.0": high + diff,
            "2.618": high + (diff * 1.618)
        }
    else:
        return {
            "1.0": low,
            "1.272": low - (diff * 0.272),
            "1.414": low - (diff * 0.414),
            "1.618": low - (diff * 0.618),
            "2.0": low - diff,
            "2.618": low - (diff * 1.618)
        }


# ==================== DIVERGENCE DETECTION ====================

def detect_rsi_divergence(prices, rsi_values, lookback=14):
    """Detect RSI Divergence"""
    if len(prices) < lookback or len(rsi_values) < lookback:
        return None
    
    prices = prices[-lookback:]
    rsi_vals = rsi_values[-lookback:]
    
    # Find price highs/lows
    price_high_idx = prices.index(max(prices))
    price_low_idx = prices.index(min(prices))
    
    # Bullish Divergence: Price makes lower low, RSI makes higher low
    if price_low_idx > len(prices) // 2:
        first_half_low = min(prices[:len(prices)//2])
        second_half_low = min(prices[len(prices)//2:])
        first_rsi_low = min(rsi_vals[:len(rsi_vals)//2])
        second_rsi_low = min(rsi_vals[len(rsi_vals)//2:])
        
        if second_half_low < first_half_low and second_rsi_low > first_rsi_low:
            return {"type": "bullish", "signal": "BUY", "confidence": 75}
    
    # Bearish Divergence: Price makes higher high, RSI makes lower high
    if price_high_idx > len(prices) // 2:
        first_half_high = max(prices[:len(prices)//2])
        second_half_high = max(prices[len(prices)//2:])
        first_rsi_high = max(rsi_vals[:len(rsi_vals)//2])
        second_rsi_high = max(rsi_vals[len(rsi_vals)//2:])
        
        if second_half_high > first_half_high and second_rsi_high < first_rsi_high:
            return {"type": "bearish", "signal": "SELL", "confidence": 75}
    
    return None

def detect_macd_divergence(prices, macd_values, lookback=14):
    """Detect MACD Divergence"""
    if len(prices) < lookback or len(macd_values) < lookback:
        return None
    
    prices = prices[-lookback:]
    macd_vals = macd_values[-lookback:]
    
    # Bullish Divergence
    first_half_low = min(prices[:len(prices)//2])
    second_half_low = min(prices[len(prices)//2:])
    first_macd_low = min(macd_vals[:len(macd_vals)//2])
    second_macd_low = min(macd_vals[len(macd_vals)//2:])
    
    if second_half_low < first_half_low and second_macd_low > first_macd_low:
        return {"type": "bullish", "signal": "BUY", "confidence": 70}
    
    # Bearish Divergence
    first_half_high = max(prices[:len(prices)//2])
    second_half_high = max(prices[len(prices)//2:])
    first_macd_high = max(macd_vals[:len(macd_vals)//2])
    second_macd_high = max(macd_vals[len(macd_vals)//2:])
    
    if second_half_high > first_half_high and second_macd_high < first_macd_high:
        return {"type": "bearish", "signal": "SELL", "confidence": 70}
    
    return None

# ==================== PARABOLIC SAR ====================

def calculate_parabolic_sar(highs, lows, af_start=0.02, af_max=0.2):
    """Calculate Parabolic SAR"""
    if len(highs) < 5:
        return None, None
    
    # Simplified calculation for current SAR
    trend = "bullish" if highs[-1] > highs[-5] else "bearish"
    
    if trend == "bullish":
        sar = min(lows[-5:])
        signal = "BUY" if lows[-1] > sar else "SELL"
    else:
        sar = max(highs[-5:])
        signal = "SELL" if highs[-1] < sar else "BUY"
    
    return sar, signal

# ==================== OBV (On Balance Volume) ====================

def calculate_obv(closes, volumes):
    """Calculate On Balance Volume"""
    if len(closes) < 2 or len(volumes) < 2:
        return None
    
    obv = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
    
    return obv

# ==================== VWAP ====================

def calculate_vwap(highs, lows, closes, volumes):
    """Calculate Volume Weighted Average Price"""
    if len(closes) < 1 or len(volumes) < 1:
        return None
    
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    cumulative_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
    cumulative_vol = sum(volumes)
    
    if cumulative_vol == 0:
        return None
    
    return cumulative_tp_vol / cumulative_vol
