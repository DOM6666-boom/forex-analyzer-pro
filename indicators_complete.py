"""
COMPLETE TECHNICAL INDICATORS - 103 Indicators
Trend, Momentum, Volume, Volatility, Support/Resistance, Cycle, Custom
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ==================== TREND INDICATORS (19) ====================

def calculate_sma(prices: List[float], period: int) -> Optional[float]:
    """Simple Moving Average"""
    if len(prices) < period:
        return None
    return sum(prices[-period:]) / period


def calculate_ema(prices: List[float], period: int) -> Optional[float]:
    """Exponential Moving Average"""
    if len(prices) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(prices[:period]) / period
    for price in prices[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_wma(prices: List[float], period: int) -> Optional[float]:
    """Weighted Moving Average"""
    if len(prices) < period:
        return None
    weights = list(range(1, period + 1))
    weighted_sum = sum(p * w for p, w in zip(prices[-period:], weights))
    return weighted_sum / sum(weights)


def calculate_hma(prices: List[float], period: int) -> Optional[float]:
    """Hull Moving Average"""
    if len(prices) < period:
        return None
    half_period = period // 2
    sqrt_period = int(np.sqrt(period))
    
    wma_half = calculate_wma(prices, half_period)
    wma_full = calculate_wma(prices, period)
    
    if wma_half is None or wma_full is None:
        return None
    
    raw_hma = 2 * wma_half - wma_full
    return raw_hma


def calculate_kama(prices: List[float], period: int = 10, fast: int = 2, slow: int = 30) -> Optional[float]:
    """Kaufman Adaptive Moving Average"""
    if len(prices) < period + 1:
        return None
    
    # Efficiency Ratio
    change = abs(prices[-1] - prices[-period-1])
    volatility = sum(abs(prices[i] - prices[i-1]) for i in range(-period, 0))
    
    if volatility == 0:
        return prices[-1]
    
    er = change / volatility
    
    # Smoothing Constant
    fast_sc = 2 / (fast + 1)
    slow_sc = 2 / (slow + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    # KAMA
    kama = prices[-period-1]
    for i in range(-period, 0):
        kama = kama + sc * (prices[i] - kama)
    
    return kama


def calculate_t3(prices: List[float], period: int = 5, v_factor: float = 0.7) -> Optional[float]:
    """T3 Moving Average"""
    if len(prices) < period * 6:
        return None
    
    ema1 = calculate_ema(prices, period)
    if ema1 is None:
        return None
    
    # Simplified T3
    c1 = -v_factor ** 3
    c2 = 3 * v_factor ** 2 + 3 * v_factor ** 3
    c3 = -6 * v_factor ** 2 - 3 * v_factor - 3 * v_factor ** 3
    c4 = 1 + 3 * v_factor + v_factor ** 3 + 3 * v_factor ** 2
    
    return ema1  # Simplified


def calculate_dema(prices: List[float], period: int) -> Optional[float]:
    """Double Exponential Moving Average"""
    ema1 = calculate_ema(prices, period)
    if ema1 is None:
        return None
    
    # Calculate EMA of EMA
    ema_prices = []
    for i in range(period, len(prices) + 1):
        e = calculate_ema(prices[:i], period)
        if e:
            ema_prices.append(e)
    
    if len(ema_prices) < period:
        return ema1
    
    ema2 = calculate_ema(ema_prices, period)
    if ema2 is None:
        return ema1
    
    return 2 * ema1 - ema2


def calculate_tema(prices: List[float], period: int) -> Optional[float]:
    """Triple Exponential Moving Average"""
    ema1 = calculate_ema(prices, period)
    if ema1 is None:
        return None
    
    dema = calculate_dema(prices, period)
    if dema is None:
        return ema1
    
    return 3 * ema1 - 3 * dema + ema1


def calculate_zlema(prices: List[float], period: int) -> Optional[float]:
    """Zero Lag Exponential Moving Average"""
    if len(prices) < period:
        return None
    
    lag = (period - 1) // 2
    zlema_prices = [2 * prices[i] - prices[i - lag] if i >= lag else prices[i] for i in range(len(prices))]
    
    return calculate_ema(zlema_prices, period)


def calculate_vidya(prices: List[float], period: int = 14, alpha: float = 0.2) -> Optional[float]:
    """Variable Index Dynamic Average"""
    if len(prices) < period + 1:
        return None
    
    # Calculate CMO for volatility
    gains = []
    losses = []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    sum_gains = sum(gains[-period:])
    sum_losses = sum(losses[-period:])
    
    if sum_gains + sum_losses == 0:
        return prices[-1]
    
    cmo = abs(sum_gains - sum_losses) / (sum_gains + sum_losses)
    
    # VIDYA
    vidya = prices[-period-1]
    for i in range(-period, 0):
        vidya = alpha * cmo * prices[i] + (1 - alpha * cmo) * vidya
    
    return vidya


def calculate_frama(prices: List[float], period: int = 16) -> Optional[float]:
    """Fractal Adaptive Moving Average"""
    if len(prices) < period * 2:
        return None
    
    half = period // 2
    
    # Calculate fractal dimension
    n1 = (max(prices[-half:]) - min(prices[-half:])) / half
    n2 = (max(prices[-period:-half]) - min(prices[-period:-half])) / half
    n3 = (max(prices[-period:]) - min(prices[-period:])) / period
    
    if n1 + n2 == 0 or n3 == 0:
        return calculate_ema(prices, period)
    
    d = (np.log(n1 + n2) - np.log(n3)) / np.log(2)
    alpha = np.exp(-4.6 * (d - 1))
    alpha = max(0.01, min(1, alpha))
    
    frama = prices[-period]
    for i in range(-period + 1, 0):
        frama = alpha * prices[i] + (1 - alpha) * frama
    
    return frama


def calculate_adx(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Average Directional Index with +DI and -DI"""
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
    
    if atr == 0:
        return 0, 0, 0
    
    plus_di = (sum(plus_dm[-period:]) / period) / atr * 100
    minus_di = (sum(minus_dm[-period:]) / period) / atr * 100
    
    if plus_di + minus_di == 0:
        return 0, plus_di, minus_di
    
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100
    
    return dx, plus_di, minus_di


def calculate_parabolic_sar(highs: List[float], lows: List[float], af_start: float = 0.02, af_max: float = 0.2) -> Tuple[Optional[float], str]:
    """Parabolic SAR"""
    if len(highs) < 5:
        return None, "neutral"
    
    # Simplified calculation
    trend = "bullish" if highs[-1] > highs[-5] else "bearish"
    
    if trend == "bullish":
        sar = min(lows[-5:])
        signal = "BUY" if lows[-1] > sar else "SELL"
    else:
        sar = max(highs[-5:])
        signal = "SELL" if highs[-1] < sar else "BUY"
    
    return sar, signal


def calculate_aroon(highs: List[float], lows: List[float], period: int = 25) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Aroon Indicator"""
    if len(highs) < period:
        return None, None, None
    
    high_idx = highs[-period:].index(max(highs[-period:]))
    low_idx = lows[-period:].index(min(lows[-period:]))
    
    aroon_up = ((period - (period - 1 - high_idx)) / period) * 100
    aroon_down = ((period - (period - 1 - low_idx)) / period) * 100
    aroon_osc = aroon_up - aroon_down
    
    return aroon_up, aroon_down, aroon_osc


def calculate_vortex(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """Vortex Indicator"""
    if len(closes) < period + 1:
        return None, None
    
    vm_plus, vm_minus, tr_list = [], [], []
    
    for i in range(1, len(closes)):
        vm_plus.append(abs(highs[i] - lows[i-1]))
        vm_minus.append(abs(lows[i] - highs[i-1]))
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
    
    sum_tr = sum(tr_list[-period:])
    
    if sum_tr == 0:
        return 1, 1
    
    vi_plus = sum(vm_plus[-period:]) / sum_tr
    vi_minus = sum(vm_minus[-period:]) / sum_tr
    
    return vi_plus, vi_minus


def calculate_mass_index(highs: List[float], lows: List[float], period: int = 25) -> Optional[float]:
    """Mass Index"""
    if len(highs) < period + 18:
        return None
    
    ema_range = []
    for i in range(len(highs)):
        ema_range.append(highs[i] - lows[i])
    
    ema9 = calculate_ema(ema_range, 9)
    if ema9 is None or ema9 == 0:
        return None
    
    # Simplified
    return sum(ema_range[-period:]) / period / ema9 * 9


def calculate_trix(prices: List[float], period: int = 15) -> Optional[float]:
    """TRIX Indicator"""
    ema1 = calculate_ema(prices, period)
    if ema1 is None:
        return None
    
    # Simplified TRIX
    if len(prices) < 2:
        return 0
    
    prev_ema = calculate_ema(prices[:-1], period)
    if prev_ema is None or prev_ema == 0:
        return 0
    
    return ((ema1 - prev_ema) / prev_ema) * 100


# ==================== MOMENTUM OSCILLATORS (13) ====================

def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Relative Strength Index"""
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


def calculate_stochastic(highs: List[float], lows: List[float], closes: List[float], k_period: int = 14, d_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """Stochastic Oscillator"""
    if len(closes) < k_period:
        return None, None
    
    lowest_low = min(lows[-k_period:])
    highest_high = max(highs[-k_period:])
    
    if highest_high == lowest_low:
        return 50, 50
    
    k = ((closes[-1] - lowest_low) / (highest_high - lowest_low)) * 100
    
    # Calculate %D (SMA of %K)
    k_values = []
    for i in range(d_period):
        if len(closes) >= k_period + i:
            ll = min(lows[-(k_period+i):len(lows)-i] if i > 0 else lows[-k_period:])
            hh = max(highs[-(k_period+i):len(highs)-i] if i > 0 else highs[-k_period:])
            if hh != ll:
                k_values.append(((closes[-(i+1)] - ll) / (hh - ll)) * 100)
    
    d = sum(k_values) / len(k_values) if k_values else k
    
    return k, d


def calculate_macd(prices: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """MACD"""
    if len(prices) < slow:
        return None, None, None
    
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    if ema_fast is None or ema_slow is None:
        return None, None, None
    
    macd_line = ema_fast - ema_slow
    
    # Calculate signal line
    macd_values = []
    for i in range(slow, len(prices) + 1):
        ef = calculate_ema(prices[:i], fast)
        es = calculate_ema(prices[:i], slow)
        if ef and es:
            macd_values.append(ef - es)
    
    signal_line = calculate_ema(macd_values, signal) if len(macd_values) >= signal else macd_line
    histogram = macd_line - signal_line if signal_line else 0
    
    return macd_line, signal_line, histogram


def calculate_williams_r(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Williams %R"""
    if len(closes) < period:
        return None
    
    highest_high = max(highs[-period:])
    lowest_low = min(lows[-period:])
    
    if highest_high == lowest_low:
        return -50
    
    return ((highest_high - closes[-1]) / (highest_high - lowest_low)) * -100


def calculate_cci(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Optional[float]:
    """Commodity Channel Index"""
    if len(closes) < period:
        return None
    
    tp = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    sma_tp = sum(tp[-period:]) / period
    mean_dev = sum(abs(t - sma_tp) for t in tp[-period:]) / period
    
    if mean_dev == 0:
        return 0
    
    return (tp[-1] - sma_tp) / (0.015 * mean_dev)


def calculate_momentum(prices: List[float], period: int = 10) -> Optional[float]:
    """Momentum Indicator"""
    if len(prices) < period + 1:
        return None
    return prices[-1] - prices[-period-1]


def calculate_roc(prices: List[float], period: int = 10) -> Optional[float]:
    """Rate of Change"""
    if len(prices) < period + 1 or prices[-period-1] == 0:
        return None
    return ((prices[-1] - prices[-period-1]) / prices[-period-1]) * 100


def calculate_ultimate_oscillator(highs: List[float], lows: List[float], closes: List[float], p1: int = 7, p2: int = 14, p3: int = 28) -> Optional[float]:
    """Ultimate Oscillator"""
    if len(closes) < p3 + 1:
        return None
    
    bp, tr = [], []
    for i in range(1, len(closes)):
        bp.append(closes[i] - min(lows[i], closes[i-1]))
        tr.append(max(highs[i], closes[i-1]) - min(lows[i], closes[i-1]))
    
    if sum(tr[-p1:]) == 0 or sum(tr[-p2:]) == 0 or sum(tr[-p3:]) == 0:
        return 50
    
    avg1 = sum(bp[-p1:]) / sum(tr[-p1:])
    avg2 = sum(bp[-p2:]) / sum(tr[-p2:])
    avg3 = sum(bp[-p3:]) / sum(tr[-p3:])
    
    return 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7


def calculate_awesome_oscillator(highs: List[float], lows: List[float]) -> Optional[float]:
    """Awesome Oscillator"""
    if len(highs) < 34:
        return None
    
    midpoints = [(h + l) / 2 for h, l in zip(highs, lows)]
    sma5 = sum(midpoints[-5:]) / 5
    sma34 = sum(midpoints[-34:]) / 34
    
    return sma5 - sma34


def calculate_accelerator_oscillator(highs: List[float], lows: List[float]) -> Optional[float]:
    """Accelerator Oscillator"""
    ao = calculate_awesome_oscillator(highs, lows)
    if ao is None:
        return None
    
    # Simplified - would need AO history for proper calculation
    return ao * 0.9


def calculate_dpo(prices: List[float], period: int = 20) -> Optional[float]:
    """Detrended Price Oscillator"""
    if len(prices) < period + period // 2:
        return None
    
    shift = period // 2 + 1
    sma = sum(prices[-(period + shift):-shift]) / period
    
    return prices[-1] - sma


def calculate_ppo(prices: List[float], fast: int = 12, slow: int = 26) -> Optional[float]:
    """Percentage Price Oscillator"""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)
    
    if ema_fast is None or ema_slow is None or ema_slow == 0:
        return None
    
    return ((ema_fast - ema_slow) / ema_slow) * 100


def calculate_cmo(prices: List[float], period: int = 14) -> Optional[float]:
    """Chande Momentum Oscillator"""
    if len(prices) < period + 1:
        return None
    
    gains, losses = [], []
    for i in range(1, len(prices)):
        change = prices[i] - prices[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    sum_gains = sum(gains[-period:])
    sum_losses = sum(losses[-period:])
    
    if sum_gains + sum_losses == 0:
        return 0
    
    return ((sum_gains - sum_losses) / (sum_gains + sum_losses)) * 100


# ==================== VOLUME INDICATORS (12) ====================

def calculate_obv(closes: List[float], volumes: List[float]) -> Optional[float]:
    """On Balance Volume"""
    if len(closes) < 2 or len(volumes) < 2:
        return None
    
    obv = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            obv += volumes[i]
        elif closes[i] < closes[i-1]:
            obv -= volumes[i]
    
    return obv


def calculate_vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Optional[float]:
    """Volume Weighted Average Price"""
    if len(closes) < 1 or len(volumes) < 1:
        return None
    
    typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    cumulative_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
    cumulative_vol = sum(volumes)
    
    if cumulative_vol == 0:
        return None
    
    return cumulative_tp_vol / cumulative_vol


def calculate_ad_line(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Optional[float]:
    """Accumulation/Distribution Line"""
    if len(closes) < 1:
        return None
    
    ad = 0
    for i in range(len(closes)):
        if highs[i] != lows[i]:
            mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
            ad += mfm * volumes[i]
    
    return ad


def calculate_cmf(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 20) -> Optional[float]:
    """Chaikin Money Flow"""
    if len(closes) < period:
        return None
    
    mfv_sum = 0
    vol_sum = 0
    
    for i in range(-period, 0):
        if highs[i] != lows[i]:
            mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i])
            mfv_sum += mfm * volumes[i]
        vol_sum += volumes[i]
    
    if vol_sum == 0:
        return 0
    
    return mfv_sum / vol_sum


def calculate_mfi(highs: List[float], lows: List[float], closes: List[float], volumes: List[float], period: int = 14) -> Optional[float]:
    """Money Flow Index"""
    if len(closes) < period + 1:
        return None
    
    tp = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]
    
    pos_mf, neg_mf = 0, 0
    for i in range(-period, 0):
        mf = tp[i] * volumes[i]
        if tp[i] > tp[i-1]:
            pos_mf += mf
        else:
            neg_mf += mf
    
    if neg_mf == 0:
        return 100
    
    mfr = pos_mf / neg_mf
    return 100 - (100 / (1 + mfr))


def calculate_force_index(closes: List[float], volumes: List[float], period: int = 13) -> Optional[float]:
    """Force Index"""
    if len(closes) < period + 1:
        return None
    
    fi = []
    for i in range(1, len(closes)):
        fi.append((closes[i] - closes[i-1]) * volumes[i])
    
    return calculate_ema(fi, period)


def calculate_eom(highs: List[float], lows: List[float], volumes: List[float], period: int = 14) -> Optional[float]:
    """Ease of Movement"""
    if len(highs) < period + 1:
        return None
    
    eom = []
    for i in range(1, len(highs)):
        dm = ((highs[i] + lows[i]) / 2) - ((highs[i-1] + lows[i-1]) / 2)
        br = (volumes[i] / 100000000) / (highs[i] - lows[i]) if highs[i] != lows[i] else 0
        eom.append(dm / br if br != 0 else 0)
    
    return sum(eom[-period:]) / period


def calculate_nvi(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Negative Volume Index"""
    if len(closes) < 2:
        return None
    
    nvi = 1000
    for i in range(1, len(closes)):
        if volumes[i] < volumes[i-1]:
            nvi = nvi + (nvi * (closes[i] - closes[i-1]) / closes[i-1])
    
    return nvi


def calculate_pvi(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Positive Volume Index"""
    if len(closes) < 2:
        return None
    
    pvi = 1000
    for i in range(1, len(closes)):
        if volumes[i] > volumes[i-1]:
            pvi = pvi + (pvi * (closes[i] - closes[i-1]) / closes[i-1])
    
    return pvi


def calculate_klinger(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Optional[float]:
    """Klinger Oscillator"""
    if len(closes) < 55:
        return None
    
    # Simplified Klinger
    vf = []
    cm = 0
    prev_trend = 1
    for i in range(1, len(closes)):
        hlc = highs[i] + lows[i] + closes[i]
        hlc_prev = highs[i-1] + lows[i-1] + closes[i-1]
        trend = 1 if hlc > hlc_prev else -1
        dm = highs[i] - lows[i]
        if i == 1:
            cm = dm
        elif trend == prev_trend:
            cm = cm + dm
        else:
            cm = dm
        prev_trend = trend
        vf.append(volumes[i] * abs(2 * dm / cm - 1) * trend * 100 if cm != 0 else 0)
    
    ema34 = calculate_ema(vf, 34)
    ema55 = calculate_ema(vf, 55)
    
    if ema34 is None or ema55 is None:
        return None
    
    return ema34 - ema55


def calculate_vpt(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Volume Price Trend"""
    if len(closes) < 2:
        return None
    
    vpt = 0
    for i in range(1, len(closes)):
        if closes[i-1] != 0:
            vpt += volumes[i] * ((closes[i] - closes[i-1]) / closes[i-1])
    
    return vpt


def calculate_vroc(volumes: List[float], period: int = 14) -> Optional[float]:
    """Volume Rate of Change"""
    if len(volumes) < period + 1 or volumes[-period-1] == 0:
        return None
    return ((volumes[-1] - volumes[-period-1]) / volumes[-period-1]) * 100


# ==================== VOLATILITY INDICATORS (15) ====================

def calculate_atr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Average True Range"""
    if len(closes) < period + 1:
        return None
    
    tr_list = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        tr_list.append(tr)
    
    return sum(tr_list[-period:]) / period


def calculate_bollinger_bands(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Bollinger Bands"""
    if len(prices) < period:
        return None, None, None
    
    sma = sum(prices[-period:]) / period
    variance = sum((p - sma) ** 2 for p in prices[-period:]) / period
    std = np.sqrt(variance)
    
    upper = sma + (std_dev * std)
    lower = sma - (std_dev * std)
    
    return upper, sma, lower


def calculate_bollinger_bandwidth(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[float]:
    """Bollinger Bandwidth"""
    upper, middle, lower = calculate_bollinger_bands(prices, period, std_dev)
    if middle is None or middle == 0:
        return None
    return ((upper - lower) / middle) * 100


def calculate_bollinger_percent_b(prices: List[float], period: int = 20, std_dev: float = 2.0) -> Optional[float]:
    """Bollinger %B"""
    upper, middle, lower = calculate_bollinger_bands(prices, period, std_dev)
    if upper is None or upper == lower:
        return None
    return (prices[-1] - lower) / (upper - lower)


def calculate_keltner_channels(highs: List[float], lows: List[float], closes: List[float], period: int = 20, multiplier: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Keltner Channels"""
    if len(closes) < period:
        return None, None, None
    
    ema = calculate_ema(closes, period)
    atr = calculate_atr(highs, lows, closes, period)
    
    if ema is None or atr is None:
        return None, None, None
    
    upper = ema + (multiplier * atr)
    lower = ema - (multiplier * atr)
    
    return upper, ema, lower


def calculate_donchian_channels(highs: List[float], lows: List[float], period: int = 20) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Donchian Channels"""
    if len(highs) < period:
        return None, None, None
    
    upper = max(highs[-period:])
    lower = min(lows[-period:])
    middle = (upper + lower) / 2
    
    return upper, middle, lower


def calculate_chandelier_exit(highs: List[float], lows: List[float], closes: List[float], period: int = 22, multiplier: float = 3.0) -> Tuple[Optional[float], Optional[float]]:
    """Chandelier Exit"""
    if len(closes) < period:
        return None, None
    
    atr = calculate_atr(highs, lows, closes, period)
    if atr is None:
        return None, None
    
    highest_high = max(highs[-period:])
    lowest_low = min(lows[-period:])
    
    long_exit = highest_high - (multiplier * atr)
    short_exit = lowest_low + (multiplier * atr)
    
    return long_exit, short_exit


def calculate_standard_deviation(prices: List[float], period: int = 20) -> Optional[float]:
    """Standard Deviation"""
    if len(prices) < period:
        return None
    
    mean = sum(prices[-period:]) / period
    variance = sum((p - mean) ** 2 for p in prices[-period:]) / period
    return np.sqrt(variance)


def calculate_historical_volatility(prices: List[float], period: int = 20) -> Optional[float]:
    """Historical Volatility (Annualized)"""
    if len(prices) < period + 1:
        return None
    
    returns = []
    for i in range(1, len(prices)):
        if prices[i-1] != 0:
            returns.append(np.log(prices[i] / prices[i-1]))
    
    if len(returns) < period:
        return None
    
    std = np.std(returns[-period:])
    return std * np.sqrt(252) * 100  # Annualized


def calculate_chaikin_volatility(highs: List[float], lows: List[float], period: int = 10, roc_period: int = 10) -> Optional[float]:
    """Chaikin Volatility"""
    if len(highs) < period + roc_period:
        return None
    
    hl_range = [h - l for h, l in zip(highs, lows)]
    ema_range = calculate_ema(hl_range, period)
    ema_range_prev = calculate_ema(hl_range[:-roc_period], period)
    
    if ema_range is None or ema_range_prev is None or ema_range_prev == 0:
        return None
    
    return ((ema_range - ema_range_prev) / ema_range_prev) * 100


def calculate_ulcer_index(prices: List[float], period: int = 14) -> Optional[float]:
    """Ulcer Index"""
    if len(prices) < period:
        return None
    
    max_price = max(prices[-period:])
    squared_drawdowns = []
    
    for p in prices[-period:]:
        pct_drawdown = ((p - max_price) / max_price) * 100
        squared_drawdowns.append(pct_drawdown ** 2)
    
    return np.sqrt(sum(squared_drawdowns) / period)


def calculate_natr(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Normalized Average True Range"""
    atr = calculate_atr(highs, lows, closes, period)
    if atr is None or closes[-1] == 0:
        return None
    return (atr / closes[-1]) * 100


def calculate_rvi(highs: List[float], lows: List[float], closes: List[float], period: int = 10) -> Optional[float]:
    """Relative Volatility Index"""
    if len(closes) < period + 1:
        return None
    
    std_up, std_down = [], []
    
    for i in range(period, len(closes)):
        std = calculate_standard_deviation(closes[i-period:i+1], period)
        if std is None:
            continue
        if closes[i] > closes[i-1]:
            std_up.append(std)
            std_down.append(0)
        else:
            std_up.append(0)
            std_down.append(std)
    
    if not std_up or not std_down:
        return 50
    
    avg_up = sum(std_up[-period:]) / period
    avg_down = sum(std_down[-period:]) / period
    
    if avg_up + avg_down == 0:
        return 50
    
    return (avg_up / (avg_up + avg_down)) * 100


def calculate_true_range(high: float, low: float, prev_close: float) -> float:
    """True Range (single bar)"""
    return max(
        high - low,
        abs(high - prev_close),
        abs(low - prev_close)
    )


def calculate_atrp(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """ATR Percentage"""
    atr = calculate_atr(highs, lows, closes, period)
    if atr is None or closes[-1] == 0:
        return None
    return (atr / closes[-1]) * 100


# ==================== SUPPORT/RESISTANCE INDICATORS (10) ====================

def calculate_pivot_points_standard(high: float, low: float, close: float) -> Dict[str, float]:
    """Standard Pivot Points"""
    pivot = (high + low + close) / 3
    
    return {
        'R3': high + 2 * (pivot - low),
        'R2': pivot + (high - low),
        'R1': 2 * pivot - low,
        'P': pivot,
        'S1': 2 * pivot - high,
        'S2': pivot - (high - low),
        'S3': low - 2 * (high - pivot)
    }


def calculate_pivot_points_fibonacci(high: float, low: float, close: float) -> Dict[str, float]:
    """Fibonacci Pivot Points"""
    pivot = (high + low + close) / 3
    range_hl = high - low
    
    return {
        'R3': pivot + range_hl * 1.000,
        'R2': pivot + range_hl * 0.618,
        'R1': pivot + range_hl * 0.382,
        'P': pivot,
        'S1': pivot - range_hl * 0.382,
        'S2': pivot - range_hl * 0.618,
        'S3': pivot - range_hl * 1.000
    }


def calculate_pivot_points_woodie(high: float, low: float, close: float) -> Dict[str, float]:
    """Woodie Pivot Points"""
    pivot = (high + low + 2 * close) / 4
    
    return {
        'R2': pivot + (high - low),
        'R1': 2 * pivot - low,
        'P': pivot,
        'S1': 2 * pivot - high,
        'S2': pivot - (high - low)
    }


def calculate_pivot_points_camarilla(high: float, low: float, close: float) -> Dict[str, float]:
    """Camarilla Pivot Points"""
    range_hl = high - low
    
    return {
        'R4': close + range_hl * 1.1 / 2,
        'R3': close + range_hl * 1.1 / 4,
        'R2': close + range_hl * 1.1 / 6,
        'R1': close + range_hl * 1.1 / 12,
        'P': (high + low + close) / 3,
        'S1': close - range_hl * 1.1 / 12,
        'S2': close - range_hl * 1.1 / 6,
        'S3': close - range_hl * 1.1 / 4,
        'S4': close - range_hl * 1.1 / 2
    }


def calculate_pivot_points_demark(high: float, low: float, close: float, open_price: float) -> Dict[str, float]:
    """DeMark Pivot Points"""
    if close < open_price:
        x = high + 2 * low + close
    elif close > open_price:
        x = 2 * high + low + close
    else:
        x = high + low + 2 * close
    
    pivot = x / 4
    
    return {
        'R1': x / 2 - low,
        'P': pivot,
        'S1': x / 2 - high
    }


def calculate_fibonacci_retracements(high: float, low: float, trend: str = "up") -> Dict[str, float]:
    """Fibonacci Retracement Levels"""
    diff = high - low
    
    if trend == "up":
        return {
            '0.0%': high,
            '23.6%': high - diff * 0.236,
            '38.2%': high - diff * 0.382,
            '50.0%': high - diff * 0.500,
            '61.8%': high - diff * 0.618,
            '78.6%': high - diff * 0.786,
            '100.0%': low
        }
    else:
        return {
            '0.0%': low,
            '23.6%': low + diff * 0.236,
            '38.2%': low + diff * 0.382,
            '50.0%': low + diff * 0.500,
            '61.8%': low + diff * 0.618,
            '78.6%': low + diff * 0.786,
            '100.0%': high
        }


def calculate_fibonacci_extensions(high: float, low: float, retracement: float, trend: str = "up") -> Dict[str, float]:
    """Fibonacci Extension Levels"""
    diff = high - low
    
    if trend == "up":
        return {
            '61.8%': retracement + diff * 0.618,
            '100.0%': retracement + diff * 1.000,
            '127.2%': retracement + diff * 1.272,
            '161.8%': retracement + diff * 1.618,
            '200.0%': retracement + diff * 2.000,
            '261.8%': retracement + diff * 2.618
        }
    else:
        return {
            '61.8%': retracement - diff * 0.618,
            '100.0%': retracement - diff * 1.000,
            '127.2%': retracement - diff * 1.272,
            '161.8%': retracement - diff * 1.618,
            '200.0%': retracement - diff * 2.000,
            '261.8%': retracement - diff * 2.618
        }


def calculate_support_resistance_levels(highs: List[float], lows: List[float], closes: List[float], lookback: int = 20) -> Dict[str, List[float]]:
    """Dynamic Support/Resistance Levels"""
    if len(closes) < lookback:
        return {'support': [], 'resistance': []}
    
    # Find swing highs and lows
    resistance_levels = []
    support_levels = []
    
    for i in range(2, len(highs) - 2):
        # Swing high
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            resistance_levels.append(highs[i])
        # Swing low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            support_levels.append(lows[i])
    
    return {
        'support': sorted(support_levels)[-3:] if support_levels else [],
        'resistance': sorted(resistance_levels)[:3] if resistance_levels else []
    }


def calculate_price_channels(highs: List[float], lows: List[float], period: int = 20) -> Tuple[Optional[float], Optional[float]]:
    """Price Channels"""
    if len(highs) < period:
        return None, None
    
    upper = max(highs[-period:])
    lower = min(lows[-period:])
    
    return upper, lower


def calculate_envelope(prices: List[float], period: int = 20, percent: float = 2.5) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Moving Average Envelope"""
    sma = calculate_sma(prices, period)
    if sma is None:
        return None, None, None
    
    upper = sma * (1 + percent / 100)
    lower = sma * (1 - percent / 100)
    
    return upper, sma, lower


# ==================== CYCLE INDICATORS (8) ====================

def calculate_schaff_trend_cycle(prices: List[float], period: int = 10, fast: int = 23, slow: int = 50) -> Optional[float]:
    """Schaff Trend Cycle"""
    if len(prices) < slow:
        return None
    
    # Calculate MACD
    macd, _, _ = calculate_macd(prices, fast, slow, period)
    if macd is None:
        return None
    
    # Simplified STC
    return 50 + (macd * 10)  # Normalized


def calculate_detrended_price_oscillator(prices: List[float], period: int = 20) -> Optional[float]:
    """Detrended Price Oscillator"""
    return calculate_dpo(prices, period)


def calculate_sine_wave(prices: List[float], period: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """Sine Wave Indicator (Ehlers)"""
    if len(prices) < period * 2:
        return None, None
    
    # Simplified sine wave calculation
    cycle_period = period
    phase = 0
    
    for i in range(-period, 0):
        phase += 2 * np.pi / cycle_period
    
    sine = np.sin(phase)
    lead_sine = np.sin(phase + np.pi / 4)
    
    return sine, lead_sine


def calculate_even_better_sine_wave(prices: List[float], period: int = 40) -> Optional[float]:
    """Even Better Sine Wave (Ehlers)"""
    if len(prices) < period:
        return None
    
    # Simplified EBSW
    hp = []
    for i in range(1, len(prices)):
        hp.append(prices[i] - prices[i-1])
    
    if len(hp) < period:
        return None
    
    # Bandpass filter approximation
    bp = sum(hp[-period:]) / period
    return np.sin(bp * np.pi)


def calculate_cyber_cycle(prices: List[float], alpha: float = 0.07) -> Optional[float]:
    """Cyber Cycle (Ehlers)"""
    if len(prices) < 4:
        return None
    
    # Simplified cyber cycle
    smooth = (prices[-1] + 2 * prices[-2] + 2 * prices[-3] + prices[-4]) / 6
    return smooth


def calculate_cg_oscillator(prices: List[float], period: int = 10) -> Optional[float]:
    """Center of Gravity Oscillator"""
    if len(prices) < period:
        return None
    
    num = sum((i + 1) * prices[-(period-i)] for i in range(period))
    den = sum(prices[-period:])
    
    if den == 0:
        return 0
    
    return -num / den


def calculate_rvi_cycle(highs: List[float], lows: List[float], closes: List[float], opens: List[float], period: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Relative Vigor Index"""
    if len(closes) < period + 3:
        return None, None
    
    num, den = [], []
    
    for i in range(3, len(closes)):
        close_open = closes[i] - opens[i]
        high_low = highs[i] - lows[i]
        
        num.append((close_open + 2 * (closes[i-1] - opens[i-1]) + 2 * (closes[i-2] - opens[i-2]) + (closes[i-3] - opens[i-3])) / 6)
        den.append((high_low + 2 * (highs[i-1] - lows[i-1]) + 2 * (highs[i-2] - lows[i-2]) + (highs[i-3] - lows[i-3])) / 6)
    
    sum_num = sum(num[-period:])
    sum_den = sum(den[-period:])
    
    if sum_den == 0:
        return 0, 0
    
    rvi = sum_num / sum_den
    signal = (rvi + 2 * rvi + 2 * rvi + rvi) / 6  # Simplified
    
    return rvi, signal


def calculate_fisher_transform(highs: List[float], lows: List[float], period: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Fisher Transform"""
    if len(highs) < period:
        return None, None
    
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    
    if highest == lowest:
        return 0, 0
    
    mid = (highs[-1] + lows[-1]) / 2
    value = 0.66 * ((mid - lowest) / (highest - lowest) - 0.5) + 0.67 * 0  # Simplified
    
    value = max(-0.999, min(0.999, value))
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    
    return fisher, fisher * 0.9  # Signal approximation


# ==================== CUSTOM/ADVANCED INDICATORS (16) ====================

def calculate_ichimoku(highs: List[float], lows: List[float], closes: List[float], 
                       tenkan: int = 9, kijun: int = 26, senkou_b: int = 52) -> Dict[str, Optional[float]]:
    """Ichimoku Cloud"""
    result = {
        'tenkan_sen': None,
        'kijun_sen': None,
        'senkou_span_a': None,
        'senkou_span_b': None,
        'chikou_span': None
    }
    
    if len(highs) < senkou_b:
        return result
    
    # Tenkan-sen (Conversion Line)
    result['tenkan_sen'] = (max(highs[-tenkan:]) + min(lows[-tenkan:])) / 2
    
    # Kijun-sen (Base Line)
    result['kijun_sen'] = (max(highs[-kijun:]) + min(lows[-kijun:])) / 2
    
    # Senkou Span A (Leading Span A)
    if result['tenkan_sen'] and result['kijun_sen']:
        result['senkou_span_a'] = (result['tenkan_sen'] + result['kijun_sen']) / 2
    
    # Senkou Span B (Leading Span B)
    result['senkou_span_b'] = (max(highs[-senkou_b:]) + min(lows[-senkou_b:])) / 2
    
    # Chikou Span (Lagging Span)
    result['chikou_span'] = closes[-1]
    
    return result


def calculate_supertrend(highs: List[float], lows: List[float], closes: List[float], period: int = 10, multiplier: float = 3.0) -> Tuple[Optional[float], str]:
    """SuperTrend Indicator"""
    if len(closes) < period + 1:
        return None, "neutral"
    
    atr = calculate_atr(highs, lows, closes, period)
    if atr is None:
        return None, "neutral"
    
    hl2 = (highs[-1] + lows[-1]) / 2
    
    upper_band = hl2 + (multiplier * atr)
    lower_band = hl2 - (multiplier * atr)
    
    # Determine trend
    if closes[-1] > upper_band:
        return lower_band, "bullish"
    elif closes[-1] < lower_band:
        return upper_band, "bearish"
    else:
        return hl2, "neutral"


def calculate_zigzag(highs: List[float], lows: List[float], deviation: float = 5.0) -> List[Dict]:
    """ZigZag Indicator"""
    if len(highs) < 3:
        return []
    
    pivots = []
    last_pivot_type = None
    last_pivot_price = highs[0]
    last_pivot_idx = 0
    
    for i in range(1, len(highs)):
        high_change = ((highs[i] - last_pivot_price) / last_pivot_price) * 100
        low_change = ((lows[i] - last_pivot_price) / last_pivot_price) * 100
        
        if last_pivot_type != "high" and high_change >= deviation:
            pivots.append({'index': i, 'type': 'high', 'price': highs[i]})
            last_pivot_type = "high"
            last_pivot_price = highs[i]
            last_pivot_idx = i
        elif last_pivot_type != "low" and low_change <= -deviation:
            pivots.append({'index': i, 'type': 'low', 'price': lows[i]})
            last_pivot_type = "low"
            last_pivot_price = lows[i]
            last_pivot_idx = i
    
    return pivots


def calculate_heikin_ashi(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict[str, float]:
    """Heikin Ashi Candles"""
    if len(closes) < 2:
        return {}
    
    ha_close = (opens[-1] + highs[-1] + lows[-1] + closes[-1]) / 4
    ha_open = (opens[-2] + closes[-2]) / 2
    ha_high = max(highs[-1], ha_open, ha_close)
    ha_low = min(lows[-1], ha_open, ha_close)
    
    return {
        'open': ha_open,
        'high': ha_high,
        'low': ha_low,
        'close': ha_close
    }


def calculate_elder_ray(highs: List[float], lows: List[float], closes: List[float], period: int = 13) -> Tuple[Optional[float], Optional[float]]:
    """Elder Ray Index (Bull/Bear Power)"""
    ema = calculate_ema(closes, period)
    if ema is None:
        return None, None
    
    bull_power = highs[-1] - ema
    bear_power = lows[-1] - ema
    
    return bull_power, bear_power


def calculate_elder_force_index(closes: List[float], volumes: List[float], period: int = 13) -> Optional[float]:
    """Elder Force Index"""
    return calculate_force_index(closes, volumes, period)


def calculate_tsi(prices: List[float], long_period: int = 25, short_period: int = 13) -> Optional[float]:
    """True Strength Index"""
    if len(prices) < long_period + short_period:
        return None
    
    # Price changes
    pc = [prices[i] - prices[i-1] for i in range(1, len(prices))]
    abs_pc = [abs(p) for p in pc]
    
    # Double smoothed
    ema_pc = calculate_ema(pc, long_period)
    ema_abs_pc = calculate_ema(abs_pc, long_period)
    
    if ema_pc is None or ema_abs_pc is None or ema_abs_pc == 0:
        return None
    
    return (ema_pc / ema_abs_pc) * 100


def calculate_connors_rsi(prices: List[float], rsi_period: int = 3, streak_period: int = 2, rank_period: int = 100) -> Optional[float]:
    """Connors RSI"""
    if len(prices) < max(rsi_period, streak_period, rank_period) + 1:
        return None
    
    # Standard RSI
    rsi = calculate_rsi(prices, rsi_period)
    if rsi is None:
        return None
    
    # Streak RSI (simplified)
    streak = 0
    for i in range(-1, -10, -1):
        if prices[i] > prices[i-1]:
            streak += 1
        elif prices[i] < prices[i-1]:
            streak -= 1
        else:
            break
    
    streak_rsi = 50 + streak * 5  # Simplified
    
    # Percent Rank
    roc = calculate_roc(prices, 1)
    if roc is None:
        roc = 0
    
    percent_rank = 50  # Simplified
    
    return (rsi + streak_rsi + percent_rank) / 3


def calculate_know_sure_thing(prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Know Sure Thing (KST)"""
    if len(prices) < 30:
        return None, None
    
    roc1 = calculate_roc(prices, 10) or 0
    roc2 = calculate_roc(prices, 15) or 0
    roc3 = calculate_roc(prices, 20) or 0
    roc4 = calculate_roc(prices, 30) or 0
    
    kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
    signal = kst * 0.9  # Simplified signal
    
    return kst, signal


def calculate_coppock_curve(prices: List[float], wma_period: int = 10, roc1: int = 14, roc2: int = 11) -> Optional[float]:
    """Coppock Curve"""
    if len(prices) < max(roc1, roc2) + wma_period:
        return None
    
    roc_sum = []
    for i in range(wma_period):
        idx = -(wma_period - i)
        r1 = calculate_roc(prices[:len(prices)+idx+1], roc1) or 0
        r2 = calculate_roc(prices[:len(prices)+idx+1], roc2) or 0
        roc_sum.append(r1 + r2)
    
    return calculate_wma(roc_sum, wma_period)


def calculate_chande_forecast_oscillator(prices: List[float], period: int = 14) -> Optional[float]:
    """Chande Forecast Oscillator"""
    if len(prices) < period:
        return None
    
    # Linear regression forecast
    x = list(range(period))
    y = prices[-period:]
    
    x_mean = sum(x) / period
    y_mean = sum(y) / period
    
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(period))
    den = sum((x[i] - x_mean) ** 2 for i in range(period))
    
    if den == 0:
        return 0
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    forecast = slope * period + intercept
    
    return ((prices[-1] - forecast) / prices[-1]) * 100


def calculate_qstick(opens: List[float], closes: List[float], period: int = 8) -> Optional[float]:
    """QStick Indicator"""
    if len(closes) < period:
        return None
    
    diff = [c - o for c, o in zip(closes[-period:], opens[-period:])]
    return sum(diff) / period


def calculate_intraday_momentum_index(opens: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Intraday Momentum Index"""
    if len(closes) < period:
        return None
    
    gains, losses = 0, 0
    
    for i in range(-period, 0):
        if closes[i] > opens[i]:
            gains += closes[i] - opens[i]
        else:
            losses += opens[i] - closes[i]
    
    if gains + losses == 0:
        return 50
    
    return (gains / (gains + losses)) * 100


def calculate_pretty_good_oscillator(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Pretty Good Oscillator"""
    if len(closes) < period:
        return None
    
    sma = calculate_sma(closes, period)
    atr = calculate_atr(highs, lows, closes, period)
    
    if sma is None or atr is None or atr == 0:
        return None
    
    return (closes[-1] - sma) / atr


def calculate_psychological_line(closes: List[float], period: int = 12) -> Optional[float]:
    """Psychological Line"""
    if len(closes) < period + 1:
        return None
    
    up_days = sum(1 for i in range(-period, 0) if closes[i] > closes[i-1])
    return (up_days / period) * 100


# ==================== STATISTICAL INDICATORS (10) ====================

def calculate_linear_regression(prices: List[float], period: int = 14) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Linear Regression (Value, Slope, Intercept)"""
    if len(prices) < period:
        return None, None, None
    
    x = list(range(period))
    y = prices[-period:]
    
    x_mean = sum(x) / period
    y_mean = sum(y) / period
    
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(period))
    den = sum((x[i] - x_mean) ** 2 for i in range(period))
    
    if den == 0:
        return y_mean, 0, y_mean
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    value = slope * (period - 1) + intercept
    
    return value, slope, intercept


def calculate_linear_regression_channel(prices: List[float], period: int = 14, std_dev: float = 2.0) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Linear Regression Channel"""
    value, slope, intercept = calculate_linear_regression(prices, period)
    if value is None:
        return None, None, None
    
    # Calculate standard error
    y = prices[-period:]
    x = list(range(period))
    
    predicted = [slope * xi + intercept for xi in x]
    errors = [(y[i] - predicted[i]) ** 2 for i in range(period)]
    std_error = np.sqrt(sum(errors) / period)
    
    upper = value + std_dev * std_error
    lower = value - std_dev * std_error
    
    return upper, value, lower


def calculate_r_squared(prices: List[float], period: int = 14) -> Optional[float]:
    """R-Squared (Coefficient of Determination)"""
    if len(prices) < period:
        return None
    
    y = prices[-period:]
    y_mean = sum(y) / period
    
    # Linear regression
    x = list(range(period))
    x_mean = sum(x) / period
    
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(period))
    den = sum((x[i] - x_mean) ** 2 for i in range(period))
    
    if den == 0:
        return 0
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    
    # Calculate R²
    ss_res = sum((y[i] - (slope * x[i] + intercept)) ** 2 for i in range(period))
    ss_tot = sum((y[i] - y_mean) ** 2 for i in range(period))
    
    if ss_tot == 0:
        return 1
    
    return 1 - (ss_res / ss_tot)


def calculate_standard_error(prices: List[float], period: int = 14) -> Optional[float]:
    """Standard Error"""
    if len(prices) < period:
        return None
    
    value, slope, intercept = calculate_linear_regression(prices, period)
    if slope is None:
        return None
    
    y = prices[-period:]
    x = list(range(period))
    
    predicted = [slope * xi + intercept for xi in x]
    errors = [(y[i] - predicted[i]) ** 2 for i in range(period)]
    
    return np.sqrt(sum(errors) / (period - 2)) if period > 2 else 0


def calculate_correlation_coefficient(prices1: List[float], prices2: List[float], period: int = 14) -> Optional[float]:
    """Correlation Coefficient"""
    if len(prices1) < period or len(prices2) < period:
        return None
    
    x = prices1[-period:]
    y = prices2[-period:]
    
    x_mean = sum(x) / period
    y_mean = sum(y) / period
    
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(period))
    den_x = sum((x[i] - x_mean) ** 2 for i in range(period))
    den_y = sum((y[i] - y_mean) ** 2 for i in range(period))
    
    if den_x == 0 or den_y == 0:
        return 0
    
    return num / np.sqrt(den_x * den_y)


def calculate_beta(asset_prices: List[float], market_prices: List[float], period: int = 20) -> Optional[float]:
    """Beta Coefficient"""
    if len(asset_prices) < period + 1 or len(market_prices) < period + 1:
        return None
    
    # Calculate returns
    asset_returns = [(asset_prices[i] - asset_prices[i-1]) / asset_prices[i-1] 
                     for i in range(-period, 0) if asset_prices[i-1] != 0]
    market_returns = [(market_prices[i] - market_prices[i-1]) / market_prices[i-1] 
                      for i in range(-period, 0) if market_prices[i-1] != 0]
    
    if len(asset_returns) < period or len(market_returns) < period:
        return None
    
    # Covariance / Variance
    asset_mean = sum(asset_returns) / len(asset_returns)
    market_mean = sum(market_returns) / len(market_returns)
    
    covariance = sum((asset_returns[i] - asset_mean) * (market_returns[i] - market_mean) 
                     for i in range(len(asset_returns))) / len(asset_returns)
    variance = sum((market_returns[i] - market_mean) ** 2 
                   for i in range(len(market_returns))) / len(market_returns)
    
    if variance == 0:
        return 0
    
    return covariance / variance


def calculate_sharpe_ratio(prices: List[float], risk_free_rate: float = 0.02, period: int = 252) -> Optional[float]:
    """Sharpe Ratio (Annualized)"""
    if len(prices) < period + 1:
        return None
    
    # Calculate returns
    returns = [(prices[i] - prices[i-1]) / prices[i-1] 
               for i in range(-period, 0) if prices[i-1] != 0]
    
    if not returns:
        return None
    
    avg_return = sum(returns) / len(returns) * 252  # Annualized
    std_return = np.std(returns) * np.sqrt(252)  # Annualized
    
    if std_return == 0:
        return 0
    
    return (avg_return - risk_free_rate) / std_return


def calculate_sortino_ratio(prices: List[float], risk_free_rate: float = 0.02, period: int = 252) -> Optional[float]:
    """Sortino Ratio"""
    if len(prices) < period + 1:
        return None
    
    # Calculate returns
    returns = [(prices[i] - prices[i-1]) / prices[i-1] 
               for i in range(-period, 0) if prices[i-1] != 0]
    
    if not returns:
        return None
    
    avg_return = sum(returns) / len(returns) * 252  # Annualized
    
    # Downside deviation
    negative_returns = [r for r in returns if r < 0]
    if not negative_returns:
        return float('inf')
    
    downside_std = np.sqrt(sum(r ** 2 for r in negative_returns) / len(negative_returns)) * np.sqrt(252)
    
    if downside_std == 0:
        return 0
    
    return (avg_return - risk_free_rate) / downside_std


def calculate_max_drawdown(prices: List[float]) -> Tuple[Optional[float], Optional[int], Optional[int]]:
    """Maximum Drawdown"""
    if len(prices) < 2:
        return None, None, None
    
    max_dd = 0
    peak = prices[0]
    peak_idx = 0
    trough_idx = 0
    
    for i, price in enumerate(prices):
        if price > peak:
            peak = price
            peak_idx = i
        
        dd = (peak - price) / peak
        if dd > max_dd:
            max_dd = dd
            trough_idx = i
    
    return max_dd * 100, peak_idx, trough_idx


def calculate_calmar_ratio(prices: List[float], period: int = 252) -> Optional[float]:
    """Calmar Ratio"""
    if len(prices) < period + 1:
        return None
    
    # Annual return
    annual_return = (prices[-1] / prices[-period] - 1) * 100
    
    # Max drawdown
    max_dd, _, _ = calculate_max_drawdown(prices[-period:])
    
    if max_dd is None or max_dd == 0:
        return 0
    
    return annual_return / max_dd


# ==================== MASTER ANALYSIS FUNCTION ====================

def analyze_all_indicators(opens: List[float], highs: List[float], lows: List[float], 
                           closes: List[float], volumes: List[float]) -> Dict:
    """
    Comprehensive analysis using all 103 indicators
    Returns categorized signals and values
    """
    results = {
        'trend': {},
        'momentum': {},
        'volume': {},
        'volatility': {},
        'support_resistance': {},
        'cycle': {},
        'custom': {},
        'statistical': {},
        'signals': {
            'bullish': 0,
            'bearish': 0,
            'neutral': 0
        },
        'overall_bias': 'neutral',
        'strength': 0
    }
    
    # ===== TREND INDICATORS =====
    results['trend']['sma_20'] = calculate_sma(closes, 20)
    results['trend']['sma_50'] = calculate_sma(closes, 50)
    results['trend']['sma_200'] = calculate_sma(closes, 200)
    results['trend']['ema_12'] = calculate_ema(closes, 12)
    results['trend']['ema_26'] = calculate_ema(closes, 26)
    results['trend']['wma_20'] = calculate_wma(closes, 20)
    results['trend']['hma_20'] = calculate_hma(closes, 20)
    results['trend']['kama'] = calculate_kama(closes)
    results['trend']['dema'] = calculate_dema(closes, 20)
    results['trend']['tema'] = calculate_tema(closes, 20)
    results['trend']['zlema'] = calculate_zlema(closes, 20)
    results['trend']['vidya'] = calculate_vidya(closes)
    results['trend']['frama'] = calculate_frama(closes)
    
    adx, plus_di, minus_di = calculate_adx(highs, lows, closes)
    results['trend']['adx'] = adx
    results['trend']['plus_di'] = plus_di
    results['trend']['minus_di'] = minus_di
    
    sar, sar_signal = calculate_parabolic_sar(highs, lows)
    results['trend']['parabolic_sar'] = sar
    results['trend']['sar_signal'] = sar_signal
    
    aroon_up, aroon_down, aroon_osc = calculate_aroon(highs, lows)
    results['trend']['aroon_up'] = aroon_up
    results['trend']['aroon_down'] = aroon_down
    results['trend']['aroon_osc'] = aroon_osc
    
    vi_plus, vi_minus = calculate_vortex(highs, lows, closes)
    results['trend']['vortex_plus'] = vi_plus
    results['trend']['vortex_minus'] = vi_minus
    
    results['trend']['mass_index'] = calculate_mass_index(highs, lows)
    results['trend']['trix'] = calculate_trix(closes)
    
    # ===== MOMENTUM INDICATORS =====
    results['momentum']['rsi'] = calculate_rsi(closes)
    
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
    results['momentum']['stoch_k'] = stoch_k
    results['momentum']['stoch_d'] = stoch_d
    
    macd, macd_signal, macd_hist = calculate_macd(closes)
    results['momentum']['macd'] = macd
    results['momentum']['macd_signal'] = macd_signal
    results['momentum']['macd_histogram'] = macd_hist
    
    results['momentum']['williams_r'] = calculate_williams_r(highs, lows, closes)
    results['momentum']['cci'] = calculate_cci(highs, lows, closes)
    results['momentum']['momentum'] = calculate_momentum(closes)
    results['momentum']['roc'] = calculate_roc(closes)
    results['momentum']['ultimate_osc'] = calculate_ultimate_oscillator(highs, lows, closes)
    results['momentum']['awesome_osc'] = calculate_awesome_oscillator(highs, lows)
    results['momentum']['accelerator_osc'] = calculate_accelerator_oscillator(highs, lows)
    results['momentum']['dpo'] = calculate_dpo(closes)
    results['momentum']['ppo'] = calculate_ppo(closes)
    results['momentum']['cmo'] = calculate_cmo(closes)
    
    # ===== VOLUME INDICATORS =====
    results['volume']['obv'] = calculate_obv(closes, volumes)
    results['volume']['vwap'] = calculate_vwap(highs, lows, closes, volumes)
    results['volume']['ad_line'] = calculate_ad_line(highs, lows, closes, volumes)
    results['volume']['cmf'] = calculate_cmf(highs, lows, closes, volumes)
    results['volume']['mfi'] = calculate_mfi(highs, lows, closes, volumes)
    results['volume']['force_index'] = calculate_force_index(closes, volumes)
    results['volume']['eom'] = calculate_eom(highs, lows, volumes)
    results['volume']['nvi'] = calculate_nvi(closes, volumes)
    results['volume']['pvi'] = calculate_pvi(closes, volumes)
    results['volume']['klinger'] = calculate_klinger(highs, lows, closes, volumes)
    results['volume']['vpt'] = calculate_vpt(closes, volumes)
    results['volume']['vroc'] = calculate_vroc(volumes)
    
    # ===== VOLATILITY INDICATORS =====
    results['volatility']['atr'] = calculate_atr(highs, lows, closes)
    
    bb_upper, bb_middle, bb_lower = calculate_bollinger_bands(closes)
    results['volatility']['bb_upper'] = bb_upper
    results['volatility']['bb_middle'] = bb_middle
    results['volatility']['bb_lower'] = bb_lower
    results['volatility']['bb_bandwidth'] = calculate_bollinger_bandwidth(closes)
    results['volatility']['bb_percent_b'] = calculate_bollinger_percent_b(closes)
    
    kc_upper, kc_middle, kc_lower = calculate_keltner_channels(highs, lows, closes)
    results['volatility']['kc_upper'] = kc_upper
    results['volatility']['kc_middle'] = kc_middle
    results['volatility']['kc_lower'] = kc_lower
    
    dc_upper, dc_middle, dc_lower = calculate_donchian_channels(highs, lows)
    results['volatility']['dc_upper'] = dc_upper
    results['volatility']['dc_middle'] = dc_middle
    results['volatility']['dc_lower'] = dc_lower
    
    chand_long, chand_short = calculate_chandelier_exit(highs, lows, closes)
    results['volatility']['chandelier_long'] = chand_long
    results['volatility']['chandelier_short'] = chand_short
    
    results['volatility']['std_dev'] = calculate_standard_deviation(closes)
    results['volatility']['hist_volatility'] = calculate_historical_volatility(closes)
    results['volatility']['chaikin_volatility'] = calculate_chaikin_volatility(highs, lows)
    results['volatility']['ulcer_index'] = calculate_ulcer_index(closes)
    results['volatility']['natr'] = calculate_natr(highs, lows, closes)
    results['volatility']['rvi'] = calculate_rvi(highs, lows, closes)
    
    # ===== SUPPORT/RESISTANCE =====
    if len(highs) > 0 and len(lows) > 0 and len(closes) > 0:
        results['support_resistance']['pivot_standard'] = calculate_pivot_points_standard(highs[-1], lows[-1], closes[-1])
        results['support_resistance']['pivot_fibonacci'] = calculate_pivot_points_fibonacci(highs[-1], lows[-1], closes[-1])
        results['support_resistance']['pivot_woodie'] = calculate_pivot_points_woodie(highs[-1], lows[-1], closes[-1])
        results['support_resistance']['pivot_camarilla'] = calculate_pivot_points_camarilla(highs[-1], lows[-1], closes[-1])
        
        if len(opens) > 0:
            results['support_resistance']['pivot_demark'] = calculate_pivot_points_demark(highs[-1], lows[-1], closes[-1], opens[-1])
        
        swing_high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
        swing_low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
        results['support_resistance']['fib_retracements'] = calculate_fibonacci_retracements(swing_high, swing_low)
        
        results['support_resistance']['sr_levels'] = calculate_support_resistance_levels(highs, lows, closes)
        
        price_upper, price_lower = calculate_price_channels(highs, lows)
        results['support_resistance']['price_channel_upper'] = price_upper
        results['support_resistance']['price_channel_lower'] = price_lower
        
        env_upper, env_middle, env_lower = calculate_envelope(closes)
        results['support_resistance']['envelope_upper'] = env_upper
        results['support_resistance']['envelope_middle'] = env_middle
        results['support_resistance']['envelope_lower'] = env_lower
    
    # ===== CYCLE INDICATORS =====
    results['cycle']['stc'] = calculate_schaff_trend_cycle(closes)
    
    sine, lead_sine = calculate_sine_wave(closes)
    results['cycle']['sine_wave'] = sine
    results['cycle']['lead_sine'] = lead_sine
    
    results['cycle']['ebsw'] = calculate_even_better_sine_wave(closes)
    results['cycle']['cyber_cycle'] = calculate_cyber_cycle(closes)
    results['cycle']['cg_oscillator'] = calculate_cg_oscillator(closes)
    
    if len(opens) > 0:
        rvi_val, rvi_sig = calculate_rvi_cycle(highs, lows, closes, opens)
        results['cycle']['rvi'] = rvi_val
        results['cycle']['rvi_signal'] = rvi_sig
    
    fisher, fisher_sig = calculate_fisher_transform(highs, lows)
    results['cycle']['fisher'] = fisher
    results['cycle']['fisher_signal'] = fisher_sig
    
    # ===== CUSTOM INDICATORS =====
    results['custom']['ichimoku'] = calculate_ichimoku(highs, lows, closes)
    
    supertrend, st_signal = calculate_supertrend(highs, lows, closes)
    results['custom']['supertrend'] = supertrend
    results['custom']['supertrend_signal'] = st_signal
    
    results['custom']['zigzag'] = calculate_zigzag(highs, lows)
    
    if len(opens) > 0:
        results['custom']['heikin_ashi'] = calculate_heikin_ashi(opens, highs, lows, closes)
    
    bull_power, bear_power = calculate_elder_ray(highs, lows, closes)
    results['custom']['bull_power'] = bull_power
    results['custom']['bear_power'] = bear_power
    
    results['custom']['tsi'] = calculate_tsi(closes)
    results['custom']['connors_rsi'] = calculate_connors_rsi(closes)
    
    kst, kst_sig = calculate_know_sure_thing(closes)
    results['custom']['kst'] = kst
    results['custom']['kst_signal'] = kst_sig
    
    results['custom']['coppock'] = calculate_coppock_curve(closes)
    results['custom']['cfo'] = calculate_chande_forecast_oscillator(closes)
    
    if len(opens) > 0:
        results['custom']['qstick'] = calculate_qstick(opens, closes)
        results['custom']['imi'] = calculate_intraday_momentum_index(opens, closes)
    
    results['custom']['pgo'] = calculate_pretty_good_oscillator(highs, lows, closes)
    results['custom']['psy_line'] = calculate_psychological_line(closes)
    
    # ===== STATISTICAL INDICATORS =====
    lr_value, lr_slope, lr_intercept = calculate_linear_regression(closes)
    results['statistical']['lr_value'] = lr_value
    results['statistical']['lr_slope'] = lr_slope
    
    lrc_upper, lrc_middle, lrc_lower = calculate_linear_regression_channel(closes)
    results['statistical']['lrc_upper'] = lrc_upper
    results['statistical']['lrc_middle'] = lrc_middle
    results['statistical']['lrc_lower'] = lrc_lower
    
    results['statistical']['r_squared'] = calculate_r_squared(closes)
    results['statistical']['std_error'] = calculate_standard_error(closes)
    results['statistical']['sharpe_ratio'] = calculate_sharpe_ratio(closes)
    results['statistical']['sortino_ratio'] = calculate_sortino_ratio(closes)
    
    max_dd, _, _ = calculate_max_drawdown(closes)
    results['statistical']['max_drawdown'] = max_dd
    results['statistical']['calmar_ratio'] = calculate_calmar_ratio(closes)
    
    # ===== CALCULATE OVERALL SIGNALS =====
    current_price = closes[-1] if closes else 0
    
    # Trend signals
    if results['trend']['sma_20'] and current_price > results['trend']['sma_20']:
        results['signals']['bullish'] += 1
    elif results['trend']['sma_20'] and current_price < results['trend']['sma_20']:
        results['signals']['bearish'] += 1
    
    if results['trend']['sma_50'] and current_price > results['trend']['sma_50']:
        results['signals']['bullish'] += 1
    elif results['trend']['sma_50'] and current_price < results['trend']['sma_50']:
        results['signals']['bearish'] += 1
    
    if results['trend']['adx'] and results['trend']['adx'] > 25:
        if results['trend']['plus_di'] and results['trend']['minus_di']:
            if results['trend']['plus_di'] > results['trend']['minus_di']:
                results['signals']['bullish'] += 2
            else:
                results['signals']['bearish'] += 2
    
    # Momentum signals
    if results['momentum']['rsi']:
        if results['momentum']['rsi'] < 30:
            results['signals']['bullish'] += 1  # Oversold
        elif results['momentum']['rsi'] > 70:
            results['signals']['bearish'] += 1  # Overbought
    
    if results['momentum']['stoch_k']:
        if results['momentum']['stoch_k'] < 20:
            results['signals']['bullish'] += 1
        elif results['momentum']['stoch_k'] > 80:
            results['signals']['bearish'] += 1
    
    if results['momentum']['macd'] and results['momentum']['macd_signal']:
        if results['momentum']['macd'] > results['momentum']['macd_signal']:
            results['signals']['bullish'] += 1
        else:
            results['signals']['bearish'] += 1
    
    # Volume signals
    if results['volume']['cmf']:
        if results['volume']['cmf'] > 0:
            results['signals']['bullish'] += 1
        else:
            results['signals']['bearish'] += 1
    
    if results['volume']['mfi']:
        if results['volume']['mfi'] < 20:
            results['signals']['bullish'] += 1
        elif results['volume']['mfi'] > 80:
            results['signals']['bearish'] += 1
    
    # Custom signals
    if results['custom']['supertrend_signal'] == 'bullish':
        results['signals']['bullish'] += 2
    elif results['custom']['supertrend_signal'] == 'bearish':
        results['signals']['bearish'] += 2
    
    # Calculate overall bias
    total_signals = results['signals']['bullish'] + results['signals']['bearish']
    if total_signals > 0:
        bull_pct = results['signals']['bullish'] / total_signals
        if bull_pct > 0.6:
            results['overall_bias'] = 'BULLISH'
            results['strength'] = int(bull_pct * 100)
        elif bull_pct < 0.4:
            results['overall_bias'] = 'BEARISH'
            results['strength'] = int((1 - bull_pct) * 100)
        else:
            results['overall_bias'] = 'NEUTRAL'
            results['strength'] = 50
    
    return results


# ==================== INDICATOR COUNT SUMMARY ====================
"""
TOTAL INDICATORS: 103

TREND INDICATORS (19):
1. SMA - Simple Moving Average
2. EMA - Exponential Moving Average
3. WMA - Weighted Moving Average
4. HMA - Hull Moving Average
5. KAMA - Kaufman Adaptive Moving Average
6. T3 - T3 Moving Average
7. DEMA - Double Exponential Moving Average
8. TEMA - Triple Exponential Moving Average
9. ZLEMA - Zero Lag EMA
10. VIDYA - Variable Index Dynamic Average
11. FRAMA - Fractal Adaptive Moving Average
12. ADX - Average Directional Index
13. Parabolic SAR
14. Aroon Indicator
15. Vortex Indicator
16. Mass Index
17. TRIX
18. +DI / -DI
19. Aroon Oscillator

MOMENTUM OSCILLATORS (13):
20. RSI - Relative Strength Index
21. Stochastic Oscillator
22. MACD
23. Williams %R
24. CCI - Commodity Channel Index
25. Momentum
26. ROC - Rate of Change
27. Ultimate Oscillator
28. Awesome Oscillator
29. Accelerator Oscillator
30. DPO - Detrended Price Oscillator
31. PPO - Percentage Price Oscillator
32. CMO - Chande Momentum Oscillator

VOLUME INDICATORS (12):
33. OBV - On Balance Volume
34. VWAP - Volume Weighted Average Price
35. A/D Line - Accumulation/Distribution
36. CMF - Chaikin Money Flow
37. MFI - Money Flow Index
38. Force Index
39. EOM - Ease of Movement
40. NVI - Negative Volume Index
41. PVI - Positive Volume Index
42. Klinger Oscillator
43. VPT - Volume Price Trend
44. VROC - Volume Rate of Change

VOLATILITY INDICATORS (15):
45. ATR - Average True Range
46. Bollinger Bands (Upper, Middle, Lower)
47. Bollinger Bandwidth
48. Bollinger %B
49. Keltner Channels
50. Donchian Channels
51. Chandelier Exit
52. Standard Deviation
53. Historical Volatility
54. Chaikin Volatility
55. Ulcer Index
56. NATR - Normalized ATR
57. RVI - Relative Volatility Index
58. True Range
59. ATRP - ATR Percentage

SUPPORT/RESISTANCE (10):
60. Standard Pivot Points
61. Fibonacci Pivot Points
62. Woodie Pivot Points
63. Camarilla Pivot Points
64. DeMark Pivot Points
65. Fibonacci Retracements
66. Fibonacci Extensions
67. Support/Resistance Levels
68. Price Channels
69. Moving Average Envelope

CYCLE INDICATORS (8):
70. Schaff Trend Cycle
71. Detrended Price Oscillator
72. Sine Wave Indicator
73. Even Better Sine Wave
74. Cyber Cycle
75. CG Oscillator
76. RVI (Relative Vigor Index)
77. Fisher Transform

CUSTOM/ADVANCED (16):
78. Ichimoku Cloud
79. SuperTrend
80. ZigZag
81. Heikin Ashi
82. Elder Ray (Bull/Bear Power)
83. Elder Force Index
84. TSI - True Strength Index
85. Connors RSI
86. KST - Know Sure Thing
87. Coppock Curve
88. Chande Forecast Oscillator
89. QStick
90. IMI - Intraday Momentum Index
91. PGO - Pretty Good Oscillator
92. Psychological Line
93. Linear Regression Value

STATISTICAL (10):
94. Linear Regression
95. Linear Regression Channel
96. R-Squared
97. Standard Error
98. Correlation Coefficient
99. Beta
100. Sharpe Ratio
101. Sortino Ratio
102. Maximum Drawdown
103. Calmar Ratio
"""


# ==================== ADDITIONAL INDICATORS (47) ====================

def calculate_williams_ad(highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
    """Williams Accumulation/Distribution"""
    if len(closes) < 2:
        return None
    
    wad = 0
    for i in range(1, len(closes)):
        if closes[i] > closes[i-1]:
            wad += closes[i] - min(lows[i], closes[i-1])
        elif closes[i] < closes[i-1]:
            wad += closes[i] - max(highs[i], closes[i-1])
    
    return wad


def calculate_elder_impulse(closes: List[float], period: int = 13) -> str:
    """Elder Impulse System"""
    if len(closes) < period + 1:
        return "neutral"
    
    # EMA direction
    ema = sum(closes[-period:]) / period
    ema_prev = sum(closes[-period-1:-1]) / period
    ema_rising = ema > ema_prev
    
    # MACD Histogram direction
    macd_hist = closes[-1] - closes[-2]
    macd_rising = macd_hist > 0
    
    if ema_rising and macd_rising:
        return "green"  # Bullish
    elif not ema_rising and not macd_rising:
        return "red"  # Bearish
    else:
        return "blue"  # Neutral


def calculate_squeeze_momentum(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Dict:
    """Squeeze Momentum Indicator"""
    if len(closes) < period:
        return {"squeeze": False, "momentum": 0}
    
    # Bollinger Bands
    sma = sum(closes[-period:]) / period
    std = np.sqrt(sum((c - sma) ** 2 for c in closes[-period:]) / period)
    bb_upper = sma + 2 * std
    bb_lower = sma - 2 * std
    
    # Keltner Channels
    tr_list = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) 
               for i in range(-period, 0)]
    atr = sum(tr_list) / period
    kc_upper = sma + 1.5 * atr
    kc_lower = sma - 1.5 * atr
    
    # Squeeze: BB inside KC
    squeeze = bb_lower > kc_lower and bb_upper < kc_upper
    
    # Momentum
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    midline = (highest + lowest) / 2
    momentum = closes[-1] - midline
    
    return {
        "squeeze": squeeze,
        "momentum": momentum,
        "signal": "bullish" if momentum > 0 else "bearish"
    }


def calculate_vwma(closes: List[float], volumes: List[float], period: int = 20) -> Optional[float]:
    """Volume Weighted Moving Average"""
    if len(closes) < period:
        return None
    
    vwma = sum(closes[i] * volumes[i] for i in range(-period, 0)) / sum(volumes[-period:])
    return vwma


def calculate_alma(prices: List[float], period: int = 9, offset: float = 0.85, sigma: float = 6) -> Optional[float]:
    """Arnaud Legoux Moving Average"""
    if len(prices) < period:
        return None
    
    m = offset * (period - 1)
    s = period / sigma
    
    weights = [np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(period)]
    weight_sum = sum(weights)
    
    alma = sum(prices[-(period-i)] * weights[i] for i in range(period)) / weight_sum
    return alma


def calculate_smma(prices: List[float], period: int = 14) -> Optional[float]:
    """Smoothed Moving Average"""
    if len(prices) < period:
        return None
    
    smma = sum(prices[:period]) / period
    for price in prices[period:]:
        smma = (smma * (period - 1) + price) / period
    
    return smma


def calculate_lsma(prices: List[float], period: int = 25) -> Optional[float]:
    """Least Squares Moving Average"""
    if len(prices) < period:
        return None
    
    x = list(range(period))
    y = prices[-period:]
    
    x_mean = sum(x) / period
    y_mean = sum(y) / period
    
    num = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(period))
    den = sum((x[i] - x_mean) ** 2 for i in range(period))
    
    if den == 0:
        return y_mean
    
    slope = num / den
    intercept = y_mean - slope * x_mean
    
    return slope * (period - 1) + intercept


def calculate_mcginley_dynamic(prices: List[float], period: int = 14) -> Optional[float]:
    """McGinley Dynamic"""
    if len(prices) < period:
        return None
    
    md = prices[-period]
    k = 0.6
    
    for price in prices[-period:]:
        md = md + (price - md) / (k * period * (price / md) ** 4)
    
    return md


def calculate_rma(prices: List[float], period: int = 14) -> Optional[float]:
    """Running Moving Average (Wilder's)"""
    if len(prices) < period:
        return None
    
    rma = sum(prices[:period]) / period
    alpha = 1 / period
    
    for price in prices[period:]:
        rma = alpha * price + (1 - alpha) * rma
    
    return rma


def calculate_stoch_rsi(prices: List[float], period: int = 14, k_period: int = 3, d_period: int = 3) -> Tuple[Optional[float], Optional[float]]:
    """Stochastic RSI"""
    if len(prices) < period + k_period:
        return None, None
    
    # Calculate RSI values
    rsi_values = []
    for i in range(period, len(prices) + 1):
        rsi = calculate_rsi(prices[:i], period)
        if rsi is not None:
            rsi_values.append(rsi)
    
    if len(rsi_values) < k_period:
        return None, None
    
    # Stochastic of RSI
    lowest_rsi = min(rsi_values[-k_period:])
    highest_rsi = max(rsi_values[-k_period:])
    
    if highest_rsi == lowest_rsi:
        stoch_rsi = 50
    else:
        stoch_rsi = ((rsi_values[-1] - lowest_rsi) / (highest_rsi - lowest_rsi)) * 100
    
    # %D (SMA of %K)
    d = stoch_rsi  # Simplified
    
    return stoch_rsi, d


def calculate_rvi_indicator(opens: List[float], highs: List[float], lows: List[float], closes: List[float], period: int = 10) -> Optional[float]:
    """Relative Vigor Index"""
    if len(closes) < period + 3:
        return None
    
    num = []
    den = []
    
    for i in range(3, len(closes)):
        close_open = closes[i] - opens[i]
        high_low = highs[i] - lows[i]
        
        num.append((close_open + 2 * (closes[i-1] - opens[i-1]) + 2 * (closes[i-2] - opens[i-2]) + (closes[i-3] - opens[i-3])) / 6)
        den.append((high_low + 2 * (highs[i-1] - lows[i-1]) + 2 * (highs[i-2] - lows[i-2]) + (highs[i-3] - lows[i-3])) / 6)
    
    if len(num) < period:
        return None
    
    sum_num = sum(num[-period:])
    sum_den = sum(den[-period:])
    
    if sum_den == 0:
        return 0
    
    return sum_num / sum_den


def calculate_dmi(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict:
    """Directional Movement Index"""
    adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period)
    
    return {
        "adx": adx,
        "plus_di": plus_di,
        "minus_di": minus_di,
        "trend_strength": "strong" if adx and adx > 25 else "weak",
        "direction": "bullish" if plus_di and minus_di and plus_di > minus_di else "bearish"
    }


def calculate_choppiness_index(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Choppiness Index"""
    if len(closes) < period + 1:
        return None
    
    # Sum of ATR
    atr_sum = 0
    for i in range(-period, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr_sum += tr
    
    # Highest high - Lowest low
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    hl_range = highest - lowest
    
    if hl_range == 0:
        return 50
    
    ci = 100 * np.log10(atr_sum / hl_range) / np.log10(period)
    return ci


def calculate_adl(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Optional[float]:
    """Advance Decline Line"""
    return calculate_ad_line(highs, lows, closes, volumes)


def calculate_trin(advances: int, declines: int, adv_volume: float, dec_volume: float) -> Optional[float]:
    """Arms Index (TRIN)"""
    if declines == 0 or dec_volume == 0:
        return None
    
    ad_ratio = advances / declines
    vol_ratio = adv_volume / dec_volume
    
    return ad_ratio / vol_ratio


def calculate_mcclellan_oscillator(advances: List[int], declines: List[int]) -> Optional[float]:
    """McClellan Oscillator"""
    if len(advances) < 39:
        return None
    
    net_advances = [a - d for a, d in zip(advances, declines)]
    
    # 19-day EMA
    ema19 = sum(net_advances[-19:]) / 19
    
    # 39-day EMA
    ema39 = sum(net_advances[-39:]) / 39
    
    return ema19 - ema39


def calculate_put_call_ratio(put_volume: float, call_volume: float) -> Optional[float]:
    """Put/Call Ratio"""
    if call_volume == 0:
        return None
    return put_volume / call_volume


def calculate_vix_fix(closes: List[float], period: int = 22) -> Optional[float]:
    """Williams VIX Fix"""
    if len(closes) < period:
        return None
    
    highest_close = max(closes[-period:])
    wvf = ((highest_close - closes[-1]) / highest_close) * 100
    
    return wvf


def calculate_hurst_exponent(prices: List[float], max_lag: int = 20) -> Optional[float]:
    """Hurst Exponent (simplified)"""
    if len(prices) < max_lag * 2:
        return None
    
    # Simplified R/S analysis
    returns = [prices[i] / prices[i-1] - 1 for i in range(1, len(prices))]
    
    mean_return = sum(returns) / len(returns)
    std_return = np.std(returns)
    
    if std_return == 0:
        return 0.5
    
    # Simplified Hurst calculation
    cumulative = [sum(returns[:i+1]) - (i+1) * mean_return for i in range(len(returns))]
    r = max(cumulative) - min(cumulative)
    s = std_return
    
    if s == 0:
        return 0.5
    
    rs = r / s
    hurst = np.log(rs) / np.log(len(returns))
    
    return max(0, min(1, hurst))


def calculate_fractal_dimension(prices: List[float], period: int = 30) -> Optional[float]:
    """Fractal Dimension Index"""
    if len(prices) < period:
        return None
    
    # Box counting method (simplified)
    n = period
    
    # Calculate price range
    price_range = max(prices[-period:]) - min(prices[-period:])
    
    if price_range == 0:
        return 1.5
    
    # Count boxes
    box_size = price_range / 10
    boxes = set()
    
    for i, price in enumerate(prices[-period:]):
        box_x = i
        box_y = int((price - min(prices[-period:])) / box_size)
        boxes.add((box_x, box_y))
    
    # Fractal dimension approximation
    fd = np.log(len(boxes)) / np.log(period)
    
    return fd


def calculate_efficiency_ratio(prices: List[float], period: int = 10) -> Optional[float]:
    """Kaufman Efficiency Ratio"""
    if len(prices) < period + 1:
        return None
    
    change = abs(prices[-1] - prices[-period-1])
    volatility = sum(abs(prices[i] - prices[i-1]) for i in range(-period, 0))
    
    if volatility == 0:
        return 0
    
    return change / volatility


def calculate_price_oscillator(prices: List[float], fast: int = 12, slow: int = 26) -> Optional[float]:
    """Price Oscillator"""
    fast_ma = calculate_sma(prices, fast)
    slow_ma = calculate_sma(prices, slow)
    
    if fast_ma is None or slow_ma is None:
        return None
    
    return fast_ma - slow_ma


def calculate_volume_oscillator(volumes: List[float], fast: int = 5, slow: int = 10) -> Optional[float]:
    """Volume Oscillator"""
    if len(volumes) < slow:
        return None
    
    fast_ma = sum(volumes[-fast:]) / fast
    slow_ma = sum(volumes[-slow:]) / slow
    
    if slow_ma == 0:
        return 0
    
    return ((fast_ma - slow_ma) / slow_ma) * 100


def calculate_accumulation_swing_index(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
    """Accumulation Swing Index"""
    if len(closes) < 2:
        return None
    
    asi = 0
    for i in range(1, len(closes)):
        c = closes[i]
        c1 = closes[i-1]
        o = opens[i]
        o1 = opens[i-1]
        h = highs[i]
        l = lows[i]
        
        k = max(h - c1, l - c1)
        tr = max(h - l, abs(h - c1), abs(l - c1))
        
        if tr == 0:
            continue
        
        er = 0
        if c1 >= l and c1 <= h:
            er = 0
        elif c1 > h:
            er = abs(h - c1)
        else:
            er = abs(l - c1)
        
        sh = c1 - o1
        r = tr - 0.5 * er + 0.25 * sh
        
        if r != 0:
            si = 50 * ((c - c1) + 0.5 * (c - o) + 0.25 * (c1 - o1)) / r * k / tr
            asi += si
    
    return asi


def calculate_swing_index(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
    """Swing Index"""
    if len(closes) < 2:
        return None
    
    c = closes[-1]
    c1 = closes[-2]
    o = opens[-1]
    o1 = opens[-2]
    h = highs[-1]
    l = lows[-1]
    
    k = max(h - c1, l - c1)
    tr = max(h - l, abs(h - c1), abs(l - c1))
    
    if tr == 0:
        return 0
    
    r = tr
    
    if r != 0:
        si = 50 * ((c - c1) + 0.5 * (c - o) + 0.25 * (c1 - o1)) / r * k / tr
        return si
    
    return 0


def calculate_rainbow_oscillator(prices: List[float]) -> Dict:
    """Rainbow Oscillator"""
    if len(prices) < 50:
        return {"value": 0, "signal": "neutral"}
    
    # Multiple SMAs
    sma2 = calculate_sma(prices, 2)
    sma5 = calculate_sma(prices, 5)
    sma10 = calculate_sma(prices, 10)
    sma20 = calculate_sma(prices, 20)
    sma50 = calculate_sma(prices, 50)
    
    if None in [sma2, sma5, sma10, sma20, sma50]:
        return {"value": 0, "signal": "neutral"}
    
    # Rainbow value
    avg_sma = (sma2 + sma5 + sma10 + sma20 + sma50) / 5
    rainbow = ((prices[-1] - avg_sma) / avg_sma) * 100
    
    signal = "bullish" if rainbow > 0 else "bearish"
    
    return {"value": rainbow, "signal": signal}


def calculate_gator_oscillator(highs: List[float], lows: List[float]) -> Dict:
    """Gator Oscillator (Alligator derivative)"""
    if len(highs) < 21:
        return {"upper": 0, "lower": 0}
    
    # Median prices
    median = [(h + l) / 2 for h, l in zip(highs, lows)]
    
    # Alligator lines (simplified)
    jaw = sum(median[-13:]) / 13
    teeth = sum(median[-8:]) / 8
    lips = sum(median[-5:]) / 5
    
    upper = abs(jaw - teeth)
    lower = -abs(teeth - lips)
    
    return {"upper": upper, "lower": lower}


def calculate_awesome_oscillator_histogram(highs: List[float], lows: List[float]) -> List[float]:
    """Awesome Oscillator Histogram"""
    if len(highs) < 35:
        return []
    
    ao_values = []
    for i in range(34, len(highs)):
        midpoints = [(highs[j] + lows[j]) / 2 for j in range(i-33, i+1)]
        sma5 = sum(midpoints[-5:]) / 5
        sma34 = sum(midpoints) / 34
        ao_values.append(sma5 - sma34)
    
    return ao_values


def calculate_market_facilitation_index(highs: List[float], lows: List[float], volumes: List[float]) -> Optional[float]:
    """Market Facilitation Index"""
    if len(highs) < 1 or volumes[-1] == 0:
        return None
    
    return (highs[-1] - lows[-1]) / volumes[-1]


def calculate_balance_of_power(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Optional[float]:
    """Balance of Power"""
    if len(closes) < 1:
        return None
    
    hl_range = highs[-1] - lows[-1]
    if hl_range == 0:
        return 0
    
    return (closes[-1] - opens[-1]) / hl_range


def calculate_price_zone_oscillator(closes: List[float], period: int = 14) -> Optional[float]:
    """Price Zone Oscillator"""
    if len(closes) < period + 1:
        return None
    
    up_sum = 0
    down_sum = 0
    
    for i in range(-period, 0):
        change = closes[i] - closes[i-1]
        if change > 0:
            up_sum += change
        else:
            down_sum += abs(change)
    
    total = up_sum + down_sum
    if total == 0:
        return 50
    
    return (up_sum / total) * 100


def calculate_trend_intensity_index(closes: List[float], period: int = 30) -> Optional[float]:
    """Trend Intensity Index"""
    if len(closes) < period:
        return None
    
    sma = sum(closes[-period:]) / period
    
    above = sum(1 for c in closes[-period:] if c > sma)
    
    return (above / period) * 100


def calculate_vertical_horizontal_filter(closes: List[float], period: int = 28) -> Optional[float]:
    """Vertical Horizontal Filter"""
    if len(closes) < period:
        return None
    
    highest = max(closes[-period:])
    lowest = min(closes[-period:])
    
    numerator = abs(highest - lowest)
    denominator = sum(abs(closes[i] - closes[i-1]) for i in range(-period+1, 0))
    
    if denominator == 0:
        return 0
    
    return numerator / denominator
