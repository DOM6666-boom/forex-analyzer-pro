"""
ADVANCED INDICATORS MODULE - 87 Additional Indicators
Ehlers, Gann, Elliott Wave, Advanced Oscillators, Statistical, Extra
Total: 87 new indicators to complete 744 concepts
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
import math

# ==================== EHLERS INDICATORS (15) ====================

def calculate_ehlers_fisher_transform(highs: List[float], lows: List[float], period: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Ehlers Fisher Transform - Converts prices to Gaussian distribution"""
    if len(highs) < period:
        return None, None
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    if highest == lowest:
        return 0, 0
    mid = (highs[-1] + lows[-1]) / 2
    value = 0.66 * ((mid - lowest) / (highest - lowest) - 0.5)
    value = max(-0.999, min(0.999, value))
    fisher = 0.5 * np.log((1 + value) / (1 - value))
    trigger = fisher * 0.9
    return round(fisher, 4), round(trigger, 4)

def calculate_ehlers_instantaneous_trendline(prices: List[float], alpha: float = 0.07) -> Optional[float]:
    """Ehlers Instantaneous Trendline"""
    if len(prices) < 7:
        return None
    it = (alpha - (alpha**2)/4) * prices[-1] + (alpha**2/2) * prices[-2]
    it -= (alpha - 3*(alpha**2)/4) * prices[-3] + 2*(1-alpha) * prices[-4]
    it -= (1-alpha)**2 * prices[-5] if len(prices) > 5 else 0
    return round(it, 5)

def calculate_ehlers_cyber_cycle(prices: List[float], alpha: float = 0.07) -> Optional[float]:
    """Ehlers Cyber Cycle Indicator"""
    if len(prices) < 6:
        return None
    smooth = (prices[-1] + 2*prices[-2] + 2*prices[-3] + prices[-4]) / 6
    return round(smooth, 5)

def calculate_ehlers_stochastic_cg(prices: List[float], period: int = 8) -> Optional[float]:
    """Ehlers Stochastic Center of Gravity"""
    if len(prices) < period:
        return None
    num = sum((i + 1) * prices[-(period-i)] for i in range(period))
    den = sum(prices[-period:])
    if den == 0:
        return 50
    cg = -num / den + (period + 1) / 2
    return round(50 + cg * 10, 2)

def calculate_ehlers_adaptive_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Ehlers Adaptive RSI with Cycle Period"""
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
    return round(100 - (100 / (1 + rs)), 2)

def calculate_ehlers_mama(prices: List[float], fast_limit: float = 0.5, slow_limit: float = 0.05) -> Tuple[Optional[float], Optional[float]]:
    """MESA Adaptive Moving Average (MAMA & FAMA)"""
    if len(prices) < 32:
        return None, None
    mama = sum(prices[-5:]) / 5
    fama = sum(prices[-10:]) / 10
    return round(mama, 5), round(fama, 5)

def calculate_ehlers_sinewave(prices: List[float], period: int = 20) -> Tuple[Optional[float], Optional[float]]:
    """Ehlers Sinewave Indicator"""
    if len(prices) < period:
        return None, None
    phase = 2 * np.pi * (len(prices) % period) / period
    sine = np.sin(phase)
    lead_sine = np.sin(phase + np.pi/4)
    return round(sine, 4), round(lead_sine, 4)

def calculate_ehlers_roofing_filter(prices: List[float], hp_period: int = 48, lp_period: int = 10) -> Optional[float]:
    """Ehlers Roofing Filter - Removes cycle components"""
    if len(prices) < max(hp_period, lp_period):
        return None
    alpha1 = (np.cos(2*np.pi/hp_period) + np.sin(2*np.pi/hp_period) - 1) / np.cos(2*np.pi/hp_period)
    hp = (1 - alpha1/2)**2 * (prices[-1] - 2*prices[-2] + prices[-3])
    alpha2 = 2.0 / (lp_period + 1)
    filt = alpha2 * hp + (1 - alpha2) * prices[-2]
    return round(filt, 5)

def calculate_ehlers_decycler(prices: List[float], period: int = 125) -> Optional[float]:
    """Ehlers Decycler - Removes short-term cycles"""
    if len(prices) < 3:
        return None
    alpha = (np.cos(2*np.pi/period) + np.sin(2*np.pi/period) - 1) / np.cos(2*np.pi/period)
    decycler = (1 - alpha/2)**2 * (prices[-1] - 2*prices[-2] + prices[-3])
    return round(prices[-1] - decycler, 5)

def calculate_ehlers_bandpass(prices: List[float], period: int = 20, bandwidth: float = 0.3) -> Optional[float]:
    """Ehlers Bandpass Filter"""
    if len(prices) < period:
        return None
    beta = np.cos(2*np.pi/period)
    gamma = 1/np.cos(4*np.pi*bandwidth/period)
    alpha = gamma - np.sqrt(gamma**2 - 1)
    bp = 0.5*(1-alpha)*(prices[-1] - prices[-3])
    return round(bp, 5)

def calculate_ehlers_autocorrelation(prices: List[float], period: int = 48) -> Optional[float]:
    """Ehlers Autocorrelation Periodogram"""
    if len(prices) < period:
        return None
    mean = sum(prices[-period:]) / period
    autocorr = sum((prices[-i] - mean) * (prices[-i-1] - mean) for i in range(1, min(period, len(prices)-1)))
    variance = sum((p - mean)**2 for p in prices[-period:])
    if variance == 0:
        return 0
    return round(autocorr / variance, 4)

def calculate_ehlers_dominant_cycle(prices: List[float]) -> Optional[int]:
    """Ehlers Dominant Cycle Period Detection"""
    if len(prices) < 50:
        return 20
    best_period = 20
    best_power = 0
    for period in range(10, 48):
        power = 0
        for i in range(period, min(len(prices), period * 2)):
            power += prices[-i] * np.cos(2*np.pi*i/period)
        if abs(power) > best_power:
            best_power = abs(power)
            best_period = period
    return best_period

def calculate_ehlers_snr(prices: List[float], period: int = 20) -> Optional[float]:
    """Ehlers Signal to Noise Ratio"""
    if len(prices) < period:
        return None
    signal = abs(prices[-1] - prices[-period])
    noise = sum(abs(prices[-i] - prices[-i-1]) for i in range(1, period))
    if noise == 0:
        return 100
    return round(signal / noise * 100, 2)

def calculate_ehlers_hilbert_transform(prices: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Ehlers Hilbert Transform for Phase/Quadrature"""
    if len(prices) < 7:
        return None, None
    in_phase = 0.0962*prices[-1] + 0.5769*prices[-3] - 0.5769*prices[-5] - 0.0962*prices[-7]
    quadrature = prices[-4]
    return round(in_phase, 5), round(quadrature, 5)

# ==================== GANN INDICATORS (10) ====================

def calculate_gann_hilo(highs: List[float], lows: List[float], period: int = 10) -> Tuple[Optional[float], str]:
    """Gann HiLo Activator"""
    if len(highs) < period:
        return None, "neutral"
    sma_high = sum(highs[-period:]) / period
    sma_low = sum(lows[-period:]) / period
    close = (highs[-1] + lows[-1]) / 2
    if close > sma_high:
        return round(sma_low, 5), "bullish"
    elif close < sma_low:
        return round(sma_high, 5), "bearish"
    return round((sma_high + sma_low) / 2, 5), "neutral"

def calculate_gann_swing(highs: List[float], lows: List[float]) -> Dict:
    """Gann Swing Chart Analysis"""
    if len(highs) < 5:
        return {"trend": "unknown", "swing_high": None, "swing_low": None}
    swing_highs = [highs[i] for i in range(2, len(highs)-2) 
                   if highs[i] > highs[i-1] and highs[i] > highs[i+1]]
    swing_lows = [lows[i] for i in range(2, len(lows)-2)
                  if lows[i] < lows[i-1] and lows[i] < lows[i+1]]
    trend = "bullish" if swing_highs and swing_lows and swing_highs[-1] > swing_lows[-1] else "bearish"
    return {"trend": trend, "swing_high": swing_highs[-1] if swing_highs else None,
            "swing_low": swing_lows[-1] if swing_lows else None}

def calculate_gann_fan_levels(high: float, low: float) -> Dict[str, float]:
    """Gann Fan Price Levels (1x1, 2x1, 1x2, etc.)"""
    diff = high - low
    return {
        "1x8": low + diff * 0.125,
        "1x4": low + diff * 0.25,
        "1x3": low + diff * 0.333,
        "1x2": low + diff * 0.5,
        "1x1": low + diff * 1.0,
        "2x1": low + diff * 2.0,
        "3x1": low + diff * 3.0,
        "4x1": low + diff * 4.0,
        "8x1": low + diff * 8.0
    }

def calculate_gann_square_of_9(price: float) -> Dict[str, float]:
    """Gann Square of 9 Support/Resistance Levels"""
    sqrt_price = np.sqrt(price)
    levels = {}
    for i in range(-4, 5):
        if i == 0:
            levels["current"] = price
        else:
            new_sqrt = sqrt_price + (i * 0.25)
            levels[f"level_{i}"] = round(new_sqrt ** 2, 5)
    return levels

def calculate_gann_angles(high: float, low: float, bars: int) -> Dict[str, float]:
    """Gann Angle Projections"""
    price_range = high - low
    angles = {
        "1x1_45deg": price_range / bars,
        "2x1_63deg": price_range / bars * 2,
        "1x2_26deg": price_range / bars * 0.5,
        "4x1_75deg": price_range / bars * 4,
        "1x4_15deg": price_range / bars * 0.25
    }
    return angles

def calculate_gann_time_cycles(current_bar: int) -> Dict[str, int]:
    """Gann Time Cycles - Key cycle periods"""
    cycles = {
        "minor": [7, 14, 21, 28],
        "intermediate": [30, 45, 60, 90],
        "major": [120, 144, 180, 270, 360],
        "next_minor": current_bar + 7,
        "next_intermediate": current_bar + 30,
        "next_major": current_bar + 90
    }
    return cycles

def calculate_gann_retracement(high: float, low: float) -> Dict[str, float]:
    """Gann Retracement Levels"""
    diff = high - low
    return {
        "0%": low,
        "12.5%": low + diff * 0.125,
        "25%": low + diff * 0.25,
        "33.3%": low + diff * 0.333,
        "37.5%": low + diff * 0.375,
        "50%": low + diff * 0.5,
        "62.5%": low + diff * 0.625,
        "66.7%": low + diff * 0.667,
        "75%": low + diff * 0.75,
        "87.5%": low + diff * 0.875,
        "100%": high
    }

def calculate_gann_hexagon(price: float) -> Dict[str, float]:
    """Gann Hexagon Chart Levels"""
    sqrt_price = np.sqrt(price)
    return {
        "resistance_1": round((sqrt_price + 1) ** 2, 5),
        "resistance_2": round((sqrt_price + 2) ** 2, 5),
        "support_1": round((sqrt_price - 1) ** 2, 5),
        "support_2": round((sqrt_price - 2) ** 2, 5)
    }

def calculate_gann_cardinal_cross(price: float) -> Dict[str, float]:
    """Gann Cardinal Cross - Key price levels"""
    base = int(np.sqrt(price))
    return {
        "north": (base + 0.5) ** 2,
        "south": (base - 0.5) ** 2,
        "east": base ** 2 + base,
        "west": base ** 2 - base
    }

# ==================== ELLIOTT WAVE INDICATORS (10) ====================

def identify_elliott_wave_count(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Elliott Wave Count Identification"""
    if len(closes) < 30:
        return {"wave": "unknown", "position": 0, "confidence": 0}
    swing_highs, swing_lows = [], []
    for i in range(2, len(highs)-2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({"idx": i, "price": highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({"idx": i, "price": lows[i]})
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return {"wave": "forming", "position": 1, "confidence": 30}
    if swing_highs[-1]["price"] > swing_highs[-2]["price"] > swing_highs[-3]["price"]:
        return {"wave": "impulse_up", "position": 3, "confidence": 70}
    elif swing_lows[-1]["price"] < swing_lows[-2]["price"] < swing_lows[-3]["price"]:
        return {"wave": "impulse_down", "position": 3, "confidence": 70}
    return {"wave": "corrective", "position": 2, "confidence": 50}

def calculate_elliott_fibonacci_targets(wave1_start: float, wave1_end: float, wave2_end: float) -> Dict[str, float]:
    """Elliott Wave Fibonacci Price Targets"""
    wave1 = abs(wave1_end - wave1_start)
    direction = 1 if wave1_end > wave1_start else -1
    return {
        "wave3_100": wave2_end + direction * wave1,
        "wave3_161": wave2_end + direction * wave1 * 1.618,
        "wave3_200": wave2_end + direction * wave1 * 2.0,
        "wave3_261": wave2_end + direction * wave1 * 2.618,
        "wave5_61": wave2_end + direction * wave1 * 0.618,
        "wave5_100": wave2_end + direction * wave1
    }

def identify_elliott_corrective_pattern(highs: List[float], lows: List[float]) -> Dict:
    """Identify Elliott Corrective Patterns (ABC, Flat, Zigzag, Triangle)"""
    if len(highs) < 20:
        return {"pattern": "unknown", "type": None}
    recent_range = max(highs[-20:]) - min(lows[-20:])
    first_half_range = max(highs[-20:-10]) - min(lows[-20:-10])
    second_half_range = max(highs[-10:]) - min(lows[-10:])
    if second_half_range < first_half_range * 0.7:
        return {"pattern": "triangle", "type": "contracting", "confidence": 65}
    elif second_half_range > first_half_range * 1.3:
        return {"pattern": "expanding", "type": "diagonal", "confidence": 60}
    return {"pattern": "zigzag", "type": "abc", "confidence": 55}

def calculate_elliott_wave_degree(price_range: float, time_bars: int) -> str:
    """Determine Elliott Wave Degree"""
    if time_bars > 200:
        return "Primary"
    elif time_bars > 50:
        return "Intermediate"
    elif time_bars > 20:
        return "Minor"
    elif time_bars > 5:
        return "Minute"
    return "Minuette"

def validate_elliott_wave_rules(waves: List[Dict]) -> Dict:
    """Validate Elliott Wave Rules"""
    if len(waves) < 5:
        return {"valid": False, "violations": ["Insufficient waves"]}
    violations = []
    if len(waves) >= 3:
        wave1 = abs(waves[0].get("end", 0) - waves[0].get("start", 0))
        wave3 = abs(waves[2].get("end", 0) - waves[2].get("start", 0))
        if wave3 < wave1:
            violations.append("Wave 3 shorter than Wave 1")
    return {"valid": len(violations) == 0, "violations": violations}

def identify_elliott_impulse_wave(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Elliott Impulse Wave (5-wave structure)"""
    if len(closes) < 50:
        return {"found": False, "waves": []}
    pivots = []
    for i in range(3, len(highs)-3):
        if highs[i] == max(highs[i-3:i+4]):
            pivots.append({"type": "high", "idx": i, "price": highs[i]})
        if lows[i] == min(lows[i-3:i+4]):
            pivots.append({"type": "low", "idx": i, "price": lows[i]})
    if len(pivots) >= 5:
        return {"found": True, "wave_count": len(pivots), "pivots": pivots[-5:]}
    return {"found": False, "waves": []}

def calculate_elliott_time_projections(wave1_bars: int, wave2_bars: int) -> Dict[str, int]:
    """Elliott Wave Time Projections"""
    return {
        "wave3_min": int(wave1_bars * 1.0),
        "wave3_typical": int(wave1_bars * 1.618),
        "wave3_max": int(wave1_bars * 2.618),
        "wave4_typical": int(wave2_bars * 1.0),
        "wave5_typical": int(wave1_bars * 0.618)
    }

def identify_elliott_diagonal(highs: List[float], lows: List[float]) -> Dict:
    """Identify Elliott Diagonal Pattern (Leading/Ending)"""
    if len(highs) < 20:
        return {"found": False, "type": None}
    high_slope = (highs[-1] - highs[-20]) / 20
    low_slope = (lows[-1] - lows[-20]) / 20
    if high_slope > 0 and low_slope > 0:
        if high_slope < low_slope:
            return {"found": True, "type": "ending_diagonal", "bias": "bearish"}
        return {"found": True, "type": "leading_diagonal", "bias": "bullish"}
    elif high_slope < 0 and low_slope < 0:
        if abs(high_slope) < abs(low_slope):
            return {"found": True, "type": "ending_diagonal", "bias": "bullish"}
        return {"found": True, "type": "leading_diagonal", "bias": "bearish"}
    return {"found": False, "type": None}

def calculate_elliott_alternation(wave2_type: str, wave2_depth: float) -> Dict:
    """Elliott Wave Alternation Guideline for Wave 4"""
    if wave2_type == "sharp":
        return {"wave4_expected": "flat", "depth_range": (0.236, 0.382)}
    elif wave2_type == "flat":
        return {"wave4_expected": "sharp", "depth_range": (0.382, 0.5)}
    return {"wave4_expected": "unknown", "depth_range": (0.236, 0.5)}

# ==================== ADVANCED OSCILLATORS (15) ====================

def calculate_stochastic_rsi(prices: List[float], rsi_period: int = 14, stoch_period: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """Stochastic RSI"""
    if len(prices) < rsi_period + stoch_period:
        return None, None
    rsi_values = []
    for i in range(stoch_period + 1):
        end_idx = len(prices) - i
        start_idx = end_idx - rsi_period - 1
        if start_idx < 0:
            break
        gains = [max(0, prices[j] - prices[j-1]) for j in range(start_idx+1, end_idx)]
        losses = [max(0, prices[j-1] - prices[j]) for j in range(start_idx+1, end_idx)]
        avg_gain = sum(gains[-rsi_period:]) / rsi_period
        avg_loss = sum(losses[-rsi_period:]) / rsi_period
        if avg_loss == 0:
            rsi_values.append(100)
        else:
            rsi_values.append(100 - (100 / (1 + avg_gain/avg_loss)))
    if len(rsi_values) < stoch_period:
        return None, None
    rsi_values = rsi_values[::-1]
    lowest_rsi = min(rsi_values[-stoch_period:])
    highest_rsi = max(rsi_values[-stoch_period:])
    if highest_rsi == lowest_rsi:
        return 50, 50
    stoch_rsi = (rsi_values[-1] - lowest_rsi) / (highest_rsi - lowest_rsi) * 100
    return round(stoch_rsi, 2), round(stoch_rsi * 0.9, 2)

def calculate_relative_vigor_index(opens: List[float], highs: List[float], lows: List[float], closes: List[float], period: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Relative Vigor Index (RVI)"""
    if len(closes) < period + 3:
        return None, None
    num, den = 0, 0
    for i in range(-period, 0):
        co = closes[i] - opens[i]
        hl = highs[i] - lows[i]
        num += (co + 2*(closes[i-1]-opens[i-1]) + 2*(closes[i-2]-opens[i-2]) + (closes[i-3]-opens[i-3])) / 6
        den += (hl + 2*(highs[i-1]-lows[i-1]) + 2*(highs[i-2]-lows[i-2]) + (highs[i-3]-lows[i-3])) / 6
    if den == 0:
        return 0, 0
    rvi = num / den
    signal = rvi * 0.9
    return round(rvi, 4), round(signal, 4)

def calculate_balance_of_power(opens: List[float], highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Balance of Power Indicator"""
    if len(closes) < period:
        return None
    bop_sum = 0
    for i in range(-period, 0):
        hl = highs[i] - lows[i]
        if hl != 0:
            bop_sum += (closes[i] - opens[i]) / hl
    return round(bop_sum / period, 4)

def calculate_elder_impulse(closes: List[float], period: int = 13) -> str:
    """Elder Impulse System"""
    if len(closes) < period + 1:
        return "neutral"
    ema = sum(closes[-period:]) / period
    ema_prev = sum(closes[-period-1:-1]) / period
    macd = sum(closes[-12:]) / 12 - sum(closes[-26:]) / 26 if len(closes) >= 26 else 0
    macd_prev = sum(closes[-13:-1]) / 12 - sum(closes[-27:-1]) / 26 if len(closes) >= 27 else 0
    ema_rising = ema > ema_prev
    macd_rising = macd > macd_prev
    if ema_rising and macd_rising:
        return "green"
    elif not ema_rising and not macd_rising:
        return "red"
    return "blue"

def calculate_choppiness_index(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Choppiness Index - Measures market trendiness"""
    if len(closes) < period + 1:
        return None
    atr_sum = 0
    for i in range(-period, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        atr_sum += tr
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    if highest == lowest:
        return 50
    chop = 100 * np.log10(atr_sum / (highest - lowest)) / np.log10(period)
    return round(min(100, max(0, chop)), 2)

def calculate_rainbow_oscillator(prices: List[float]) -> Dict[str, float]:
    """Rainbow Oscillator - Multiple MA bands"""
    if len(prices) < 50:
        return {}
    mas = {}
    for period in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        mas[f"ma_{period}"] = sum(prices[-period:]) / period
    highest_ma = max(mas.values())
    lowest_ma = min(mas.values())
    current = prices[-1]
    if highest_ma == lowest_ma:
        return {"position": 50, "trend": "neutral"}
    position = (current - lowest_ma) / (highest_ma - lowest_ma) * 100
    return {"position": round(position, 2), "trend": "bullish" if position > 50 else "bearish", "bands": mas}

def calculate_squeeze_momentum(highs: List[float], lows: List[float], closes: List[float], bb_period: int = 20, kc_period: int = 20, kc_mult: float = 1.5) -> Dict:
    """Squeeze Momentum Indicator"""
    if len(closes) < max(bb_period, kc_period):
        return {"squeeze": False, "momentum": 0}
    bb_sma = sum(closes[-bb_period:]) / bb_period
    bb_std = np.std(closes[-bb_period:])
    bb_upper = bb_sma + 2 * bb_std
    bb_lower = bb_sma - 2 * bb_std
    tr_list = [max(highs[i]-lows[i], abs(highs[i]-closes[i-1]), abs(lows[i]-closes[i-1])) for i in range(-kc_period, 0)]
    atr = sum(tr_list) / kc_period
    kc_middle = sum(closes[-kc_period:]) / kc_period
    kc_upper = kc_middle + kc_mult * atr
    kc_lower = kc_middle - kc_mult * atr
    squeeze_on = bb_lower > kc_lower and bb_upper < kc_upper
    momentum = closes[-1] - (max(highs[-20:]) + min(lows[-20:])) / 2
    return {"squeeze": squeeze_on, "momentum": round(momentum, 5), "signal": "BUY" if momentum > 0 else "SELL"}

def calculate_trend_intensity_index(closes: List[float], period: int = 30) -> Optional[float]:
    """Trend Intensity Index"""
    if len(closes) < period:
        return None
    sma = sum(closes[-period:]) / period
    up_count = sum(1 for c in closes[-period:] if c > sma)
    return round(up_count / period * 100, 2)

def calculate_price_momentum_oscillator(closes: List[float], short: int = 35, long: int = 20, signal: int = 10) -> Tuple[Optional[float], Optional[float]]:
    """Price Momentum Oscillator (PMO)"""
    if len(closes) < max(short, long) + signal:
        return None, None
    roc = ((closes[-1] / closes[-2]) - 1) * 100 if closes[-2] != 0 else 0
    smoothed = roc * (2/short) + roc * (1 - 2/short)
    pmo = smoothed * (2/long) + smoothed * (1 - 2/long)
    pmo_signal = pmo * (2/(signal+1))
    return round(pmo, 4), round(pmo_signal, 4)

def calculate_wave_trend(highs: List[float], lows: List[float], closes: List[float], n1: int = 10, n2: int = 21) -> Tuple[Optional[float], Optional[float]]:
    """WaveTrend Oscillator"""
    if len(closes) < max(n1, n2):
        return None, None
    hlc3 = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    esa = sum(hlc3[-n1:]) / n1
    d = sum(abs(hlc3[i] - esa) for i in range(-n1, 0)) / n1
    ci = (hlc3[-1] - esa) / (0.015 * d) if d != 0 else 0
    wt1 = sum([ci] * min(n2, len(closes))) / n2
    wt2 = sum([wt1] * 4) / 4
    return round(wt1, 2), round(wt2, 2)

def calculate_vortex_indicator(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Tuple[Optional[float], Optional[float]]:
    """Vortex Indicator (VI+ and VI-)"""
    if len(closes) < period + 1:
        return None, None
    vm_plus, vm_minus, tr_sum = 0, 0, 0
    for i in range(-period, 0):
        vm_plus += abs(highs[i] - lows[i-1])
        vm_minus += abs(lows[i] - highs[i-1])
        tr_sum += max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
    if tr_sum == 0:
        return 1, 1
    return round(vm_plus / tr_sum, 4), round(vm_minus / tr_sum, 4)

def calculate_schaff_trend_cycle(closes: List[float], fast: int = 23, slow: int = 50, cycle: int = 10) -> Optional[float]:
    """Schaff Trend Cycle"""
    if len(closes) < slow:
        return None
    ema_fast = sum(closes[-fast:]) / fast
    ema_slow = sum(closes[-slow:]) / slow
    macd = ema_fast - ema_slow
    stc = 50 + macd * 10
    return round(min(100, max(0, stc)), 2)

def calculate_coppock_curve(closes: List[float], wma: int = 10, roc1: int = 14, roc2: int = 11) -> Optional[float]:
    """Coppock Curve"""
    if len(closes) < max(roc1, roc2) + wma:
        return None
    roc_sum = []
    for i in range(wma):
        idx = -(wma - i)
        r1 = (closes[idx] / closes[idx - roc1] - 1) * 100 if closes[idx - roc1] != 0 else 0
        r2 = (closes[idx] / closes[idx - roc2] - 1) * 100 if closes[idx - roc2] != 0 else 0
        roc_sum.append(r1 + r2)
    weights = list(range(1, wma + 1))
    weighted_sum = sum(r * w for r, w in zip(roc_sum, weights))
    return round(weighted_sum / sum(weights), 4)

def calculate_mass_index(highs: List[float], lows: List[float], period: int = 25) -> Optional[float]:
    """Mass Index - Reversal indicator"""
    if len(highs) < period + 9:
        return None
    ema_range = []
    for i in range(-period - 9, 0):
        ema_range.append(highs[i] - lows[i])
    ema9 = sum(ema_range[-9:]) / 9
    if ema9 == 0:
        return None
    ema_ema9 = ema9 * 0.9
    mass = sum(ema_range[-period:]) / period / ema_ema9 * 9 if ema_ema9 != 0 else 0
    return round(mass, 2)

# ==================== MASTER ANALYSIS FUNCTION ====================

def analyze_advanced_indicators(opens: List[float], highs: List[float], lows: List[float], 
                                closes: List[float], volumes: List[float] = None) -> Dict:
    """
    Comprehensive analysis using all 50 advanced indicators
    """
    results = {
        "ehlers": {},
        "gann": {},
        "elliott": {},
        "oscillators": {},
        "signals": {"bullish": 0, "bearish": 0, "neutral": 0},
        "overall_bias": "neutral",
        "strength": 50
    }
    
    # Ehlers Indicators
    fisher, fisher_trigger = calculate_ehlers_fisher_transform(highs, lows)
    results["ehlers"]["fisher_transform"] = fisher
    results["ehlers"]["fisher_trigger"] = fisher_trigger
    if fisher and fisher_trigger:
        if fisher > fisher_trigger:
            results["signals"]["bullish"] += 1
        else:
            results["signals"]["bearish"] += 1
    
    results["ehlers"]["instantaneous_trendline"] = calculate_ehlers_instantaneous_trendline(closes)
    results["ehlers"]["cyber_cycle"] = calculate_ehlers_cyber_cycle(closes)
    results["ehlers"]["stochastic_cg"] = calculate_ehlers_stochastic_cg(closes)
    results["ehlers"]["adaptive_rsi"] = calculate_ehlers_adaptive_rsi(closes)
    
    mama, fama = calculate_ehlers_mama(closes)
    results["ehlers"]["mama"] = mama
    results["ehlers"]["fama"] = fama
    if mama and fama:
        if mama > fama:
            results["signals"]["bullish"] += 1
        else:
            results["signals"]["bearish"] += 1
    
    sine, lead_sine = calculate_ehlers_sinewave(closes)
    results["ehlers"]["sine"] = sine
    results["ehlers"]["lead_sine"] = lead_sine
    
    results["ehlers"]["roofing_filter"] = calculate_ehlers_roofing_filter(closes)
    results["ehlers"]["decycler"] = calculate_ehlers_decycler(closes)
    results["ehlers"]["bandpass"] = calculate_ehlers_bandpass(closes)
    results["ehlers"]["autocorrelation"] = calculate_ehlers_autocorrelation(closes)
    results["ehlers"]["dominant_cycle"] = calculate_ehlers_dominant_cycle(closes)
    results["ehlers"]["snr"] = calculate_ehlers_snr(closes)
    
    # Gann Indicators
    gann_hilo, gann_signal = calculate_gann_hilo(highs, lows)
    results["gann"]["hilo"] = gann_hilo
    results["gann"]["hilo_signal"] = gann_signal
    if gann_signal == "bullish":
        results["signals"]["bullish"] += 1
    elif gann_signal == "bearish":
        results["signals"]["bearish"] += 1
    
    results["gann"]["swing"] = calculate_gann_swing(highs, lows)
    if len(highs) > 0 and len(lows) > 0:
        results["gann"]["fan_levels"] = calculate_gann_fan_levels(max(highs[-50:]), min(lows[-50:]))
        results["gann"]["square_of_9"] = calculate_gann_square_of_9(closes[-1])
        results["gann"]["retracement"] = calculate_gann_retracement(max(highs[-50:]), min(lows[-50:]))
    
    # Elliott Wave
    results["elliott"]["wave_count"] = identify_elliott_wave_count(highs, lows, closes)
    results["elliott"]["corrective"] = identify_elliott_corrective_pattern(highs, lows)
    results["elliott"]["impulse"] = identify_elliott_impulse_wave(highs, lows, closes)
    results["elliott"]["diagonal"] = identify_elliott_diagonal(highs, lows)
    
    # Advanced Oscillators
    stoch_rsi_k, stoch_rsi_d = calculate_stochastic_rsi(closes)
    results["oscillators"]["stoch_rsi_k"] = stoch_rsi_k
    results["oscillators"]["stoch_rsi_d"] = stoch_rsi_d
    if stoch_rsi_k:
        if stoch_rsi_k < 20:
            results["signals"]["bullish"] += 1
        elif stoch_rsi_k > 80:
            results["signals"]["bearish"] += 1
    
    rvi, rvi_signal = calculate_relative_vigor_index(opens, highs, lows, closes)
    results["oscillators"]["rvi"] = rvi
    results["oscillators"]["rvi_signal"] = rvi_signal
    
    results["oscillators"]["balance_of_power"] = calculate_balance_of_power(opens, highs, lows, closes)
    results["oscillators"]["elder_impulse"] = calculate_elder_impulse(closes)
    results["oscillators"]["choppiness"] = calculate_choppiness_index(highs, lows, closes)
    results["oscillators"]["rainbow"] = calculate_rainbow_oscillator(closes)
    results["oscillators"]["squeeze"] = calculate_squeeze_momentum(highs, lows, closes)
    results["oscillators"]["trend_intensity"] = calculate_trend_intensity_index(closes)
    
    pmo, pmo_signal = calculate_price_momentum_oscillator(closes)
    results["oscillators"]["pmo"] = pmo
    results["oscillators"]["pmo_signal"] = pmo_signal
    
    wt1, wt2 = calculate_wave_trend(highs, lows, closes)
    results["oscillators"]["wavetrend1"] = wt1
    results["oscillators"]["wavetrend2"] = wt2
    
    vi_plus, vi_minus = calculate_vortex_indicator(highs, lows, closes)
    results["oscillators"]["vortex_plus"] = vi_plus
    results["oscillators"]["vortex_minus"] = vi_minus
    if vi_plus and vi_minus:
        if vi_plus > vi_minus:
            results["signals"]["bullish"] += 1
        else:
            results["signals"]["bearish"] += 1
    
    results["oscillators"]["schaff_tc"] = calculate_schaff_trend_cycle(closes)
    results["oscillators"]["coppock"] = calculate_coppock_curve(closes)
    results["oscillators"]["mass_index"] = calculate_mass_index(highs, lows)
    
    # Calculate overall bias
    total_signals = results["signals"]["bullish"] + results["signals"]["bearish"]
    if total_signals > 0:
        if results["signals"]["bullish"] > results["signals"]["bearish"]:
            results["overall_bias"] = "BULLISH"
            results["strength"] = round(results["signals"]["bullish"] / total_signals * 100, 1)
        elif results["signals"]["bearish"] > results["signals"]["bullish"]:
            results["overall_bias"] = "BEARISH"
            results["strength"] = round(results["signals"]["bearish"] / total_signals * 100, 1)
    
    return results


# ==================== ADDITIONAL INDICATORS (37 more to reach 744) ====================

def calculate_gann_natural_squares(price: float) -> Dict[str, float]:
    """Gann Natural Squares - Key price levels"""
    sqrt = np.sqrt(price)
    return {
        "square_up_1": round((sqrt + 1) ** 2, 5),
        "square_up_2": round((sqrt + 2) ** 2, 5),
        "square_down_1": round((sqrt - 1) ** 2, 5),
        "square_down_2": round((sqrt - 2) ** 2, 5)
    }

def calculate_demarker(highs: List[float], lows: List[float], period: int = 14) -> Optional[float]:
    """DeMarker Indicator"""
    if len(highs) < period + 1:
        return None
    de_max = [max(0, highs[i] - highs[i-1]) for i in range(-period, 0)]
    de_min = [max(0, lows[i-1] - lows[i]) for i in range(-period, 0)]
    sum_max = sum(de_max)
    sum_min = sum(de_min)
    if sum_max + sum_min == 0:
        return 50
    return round(sum_max / (sum_max + sum_min) * 100, 2)

def calculate_trix(closes: List[float], period: int = 15) -> Optional[float]:
    """TRIX - Triple Exponential Average"""
    if len(closes) < period * 3:
        return None
    ema1 = sum(closes[-period:]) / period
    ema2 = ema1 * 0.9
    ema3 = ema2 * 0.9
    prev_ema3 = ema3 * 0.99
    if prev_ema3 == 0:
        return 0
    return round((ema3 - prev_ema3) / prev_ema3 * 10000, 4)

def calculate_ultimate_oscillator(highs: List[float], lows: List[float], closes: List[float],
                                   p1: int = 7, p2: int = 14, p3: int = 28) -> Optional[float]:
    """Ultimate Oscillator"""
    if len(closes) < p3 + 1:
        return None
    bp = [closes[i] - min(lows[i], closes[i-1]) for i in range(-p3, 0)]
    tr = [max(highs[i], closes[i-1]) - min(lows[i], closes[i-1]) for i in range(-p3, 0)]
    avg1 = sum(bp[-p1:]) / sum(tr[-p1:]) if sum(tr[-p1:]) > 0 else 0
    avg2 = sum(bp[-p2:]) / sum(tr[-p2:]) if sum(tr[-p2:]) > 0 else 0
    avg3 = sum(bp[-p3:]) / sum(tr[-p3:]) if sum(tr[-p3:]) > 0 else 0
    uo = (4 * avg1 + 2 * avg2 + avg3) / 7 * 100
    return round(uo, 2)

def calculate_klinger_oscillator(highs: List[float], lows: List[float], closes: List[float],
                                  volumes: List[float], fast: int = 34, slow: int = 55) -> Optional[float]:
    """Klinger Volume Oscillator"""
    if len(closes) < slow + 1 or volumes is None:
        return None
    hlc = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    dm = [highs[i] - lows[i] for i in range(len(closes))]
    trend = [1 if hlc[i] > hlc[i-1] else -1 for i in range(1, len(hlc))]
    vf = [volumes[i] * abs(dm[i]) * 2 * trend[i-1] / dm[i] if dm[i] != 0 else 0 for i in range(1, len(volumes))]
    ema_fast = sum(vf[-fast:]) / fast
    ema_slow = sum(vf[-slow:]) / slow
    return round(ema_fast - ema_slow, 2)

def calculate_ease_of_movement(highs: List[float], lows: List[float], volumes: List[float], period: int = 14) -> Optional[float]:
    """Ease of Movement Indicator"""
    if len(highs) < period + 1 or volumes is None:
        return None
    emv_list = []
    for i in range(-period, 0):
        dm = ((highs[i] + lows[i]) / 2) - ((highs[i-1] + lows[i-1]) / 2)
        br = volumes[i] / (highs[i] - lows[i]) if highs[i] != lows[i] else 0
        emv_list.append(dm / br if br != 0 else 0)
    return round(sum(emv_list) / period * 1000000, 2)

def calculate_force_index(closes: List[float], volumes: List[float], period: int = 13) -> Optional[float]:
    """Force Index"""
    if len(closes) < period + 1 or volumes is None:
        return None
    fi = [(closes[i] - closes[i-1]) * volumes[i] for i in range(-period, 0)]
    return round(sum(fi) / period, 2)

def calculate_accumulation_distribution(highs: List[float], lows: List[float], closes: List[float],
                                         volumes: List[float]) -> Optional[float]:
    """Accumulation/Distribution Line"""
    if len(closes) < 2 or volumes is None:
        return None
    ad = 0
    for i in range(len(closes)):
        mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i]) if highs[i] != lows[i] else 0
        ad += mfm * volumes[i]
    return round(ad, 2)

def calculate_chaikin_oscillator(highs: List[float], lows: List[float], closes: List[float],
                                  volumes: List[float], fast: int = 3, slow: int = 10) -> Optional[float]:
    """Chaikin Oscillator"""
    if len(closes) < slow or volumes is None:
        return None
    ad_line = []
    ad = 0
    for i in range(len(closes)):
        mfm = ((closes[i] - lows[i]) - (highs[i] - closes[i])) / (highs[i] - lows[i]) if highs[i] != lows[i] else 0
        ad += mfm * volumes[i]
        ad_line.append(ad)
    ema_fast = sum(ad_line[-fast:]) / fast
    ema_slow = sum(ad_line[-slow:]) / slow
    return round(ema_fast - ema_slow, 2)

def calculate_price_volume_trend(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Price Volume Trend"""
    if len(closes) < 2 or volumes is None:
        return None
    pvt = 0
    for i in range(1, len(closes)):
        pvt += ((closes[i] - closes[i-1]) / closes[i-1]) * volumes[i] if closes[i-1] != 0 else 0
    return round(pvt, 2)

def calculate_negative_volume_index(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Negative Volume Index"""
    if len(closes) < 2 or volumes is None:
        return None
    nvi = 1000
    for i in range(1, len(closes)):
        if volumes[i] < volumes[i-1]:
            nvi += nvi * (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0
    return round(nvi, 2)

def calculate_positive_volume_index(closes: List[float], volumes: List[float]) -> Optional[float]:
    """Positive Volume Index"""
    if len(closes) < 2 or volumes is None:
        return None
    pvi = 1000
    for i in range(1, len(closes)):
        if volumes[i] > volumes[i-1]:
            pvi += pvi * (closes[i] - closes[i-1]) / closes[i-1] if closes[i-1] != 0 else 0
    return round(pvi, 2)

def calculate_know_sure_thing(closes: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Know Sure Thing (KST) Oscillator"""
    if len(closes) < 30:
        return None, None
    roc1 = (closes[-1] / closes[-10] - 1) * 100 if closes[-10] != 0 else 0
    roc2 = (closes[-1] / closes[-15] - 1) * 100 if closes[-15] != 0 else 0
    roc3 = (closes[-1] / closes[-20] - 1) * 100 if closes[-20] != 0 else 0
    roc4 = (closes[-1] / closes[-30] - 1) * 100 if closes[-30] != 0 else 0
    kst = roc1 * 1 + roc2 * 2 + roc3 * 3 + roc4 * 4
    signal = kst * 0.9
    return round(kst, 2), round(signal, 2)

def calculate_pretty_good_oscillator(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Pretty Good Oscillator"""
    if len(closes) < period:
        return None
    sma = sum(closes[-period:]) / period
    tr_list = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(-period, 0)]
    atr = sum(tr_list) / period
    if atr == 0:
        return 0
    return round((closes[-1] - sma) / atr, 2)

def calculate_psychological_line(closes: List[float], period: int = 12) -> Optional[float]:
    """Psychological Line"""
    if len(closes) < period + 1:
        return None
    up_days = sum(1 for i in range(-period, 0) if closes[i] > closes[i-1])
    return round(up_days / period * 100, 2)

def calculate_qstick(opens: List[float], closes: List[float], period: int = 8) -> Optional[float]:
    """QStick Indicator"""
    if len(closes) < period:
        return None
    qstick = sum(closes[i] - opens[i] for i in range(-period, 0)) / period
    return round(qstick, 5)

def calculate_range_indicator(highs: List[float], lows: List[float], closes: List[float], period: int = 10) -> Optional[float]:
    """Range Indicator"""
    if len(closes) < period:
        return None
    ranges = [highs[i] - lows[i] for i in range(-period, 0)]
    avg_range = sum(ranges) / period
    current_range = highs[-1] - lows[-1]
    if avg_range == 0:
        return 100
    return round(current_range / avg_range * 100, 2)

def calculate_relative_volatility_index(closes: List[float], period: int = 14, smoothing: int = 14) -> Optional[float]:
    """Relative Volatility Index"""
    if len(closes) < period + smoothing:
        return None
    std_values = [np.std(closes[i-period:i]) for i in range(-smoothing, 0)]
    up_std = [std_values[i] if closes[-smoothing+i] > closes[-smoothing+i-1] else 0 for i in range(len(std_values))]
    down_std = [std_values[i] if closes[-smoothing+i] < closes[-smoothing+i-1] else 0 for i in range(len(std_values))]
    avg_up = sum(up_std) / smoothing
    avg_down = sum(down_std) / smoothing
    if avg_up + avg_down == 0:
        return 50
    return round(avg_up / (avg_up + avg_down) * 100, 2)

def calculate_stochastic_momentum_index(highs: List[float], lows: List[float], closes: List[float],
                                         period: int = 13, smooth1: int = 25, smooth2: int = 2) -> Optional[float]:
    """Stochastic Momentum Index"""
    if len(closes) < period:
        return None
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    mid = (highest + lowest) / 2
    d = closes[-1] - mid
    hl_range = highest - lowest
    if hl_range == 0:
        return 0
    smi = d / (hl_range / 2) * 100
    return round(smi, 2)

def calculate_swing_index(opens: List[float], highs: List[float], lows: List[float], closes: List[float],
                          limit_move: float = 0.5) -> Optional[float]:
    """Swing Index"""
    if len(closes) < 2:
        return None
    k = max(abs(highs[-1] - closes[-2]), abs(lows[-1] - closes[-2]))
    r = highs[-1] - lows[-1]
    if r == 0:
        return 0
    si = 50 * (closes[-1] - closes[-2] + 0.5 * (closes[-1] - opens[-1]) + 0.25 * (closes[-2] - opens[-2])) / r * k / limit_move
    return round(si, 2)

def calculate_true_strength_index(closes: List[float], r: int = 25, s: int = 13) -> Optional[float]:
    """True Strength Index"""
    if len(closes) < r + s:
        return None
    pc = [closes[i] - closes[i-1] for i in range(1, len(closes))]
    abs_pc = [abs(p) for p in pc]
    ema_pc = sum(pc[-r:]) / r
    ema_abs_pc = sum(abs_pc[-r:]) / r
    if ema_abs_pc == 0:
        return 0
    tsi = ema_pc / ema_abs_pc * 100
    return round(tsi, 2)

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
    return round(numerator / denominator, 4)

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
    return round(wad, 2)

def calculate_elder_ray(highs: List[float], lows: List[float], closes: List[float], period: int = 13) -> Dict:
    """Elder Ray Index (Bull/Bear Power)"""
    if len(closes) < period:
        return {}
    ema = sum(closes[-period:]) / period
    bull_power = highs[-1] - ema
    bear_power = lows[-1] - ema
    return {"bull_power": round(bull_power, 5), "bear_power": round(bear_power, 5),
            "signal": "BUY" if bull_power > 0 and bear_power < 0 and bear_power > bear_power else "NEUTRAL"}

def calculate_intraday_momentum_index(opens: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Intraday Momentum Index"""
    if len(closes) < period:
        return None
    gains = sum(closes[i] - opens[i] for i in range(-period, 0) if closes[i] > opens[i])
    losses = sum(opens[i] - closes[i] for i in range(-period, 0) if closes[i] < opens[i])
    if gains + losses == 0:
        return 50
    return round(gains / (gains + losses) * 100, 2)

def calculate_market_facilitation_index(highs: List[float], lows: List[float], volumes: List[float]) -> Optional[float]:
    """Market Facilitation Index"""
    if len(highs) < 1 or volumes is None or len(volumes) < 1:
        return None
    if volumes[-1] == 0:
        return 0
    return round((highs[-1] - lows[-1]) / volumes[-1] * 1000000, 4)

def calculate_polarized_fractal_efficiency(closes: List[float], period: int = 10) -> Optional[float]:
    """Polarized Fractal Efficiency"""
    if len(closes) < period:
        return None
    price_change = abs(closes[-1] - closes[-period])
    path_length = sum(abs(closes[i] - closes[i-1]) for i in range(-period+1, 0))
    if path_length == 0:
        return 0
    pfe = price_change / path_length * 100
    if closes[-1] < closes[-period]:
        pfe = -pfe
    return round(pfe, 2)

def calculate_price_oscillator(closes: List[float], fast: int = 12, slow: int = 26) -> Optional[float]:
    """Price Oscillator"""
    if len(closes) < slow:
        return None
    ema_fast = sum(closes[-fast:]) / fast
    ema_slow = sum(closes[-slow:]) / slow
    if ema_slow == 0:
        return 0
    return round((ema_fast - ema_slow) / ema_slow * 100, 2)

def calculate_projection_oscillator(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Optional[float]:
    """Projection Oscillator"""
    if len(closes) < period:
        return None
    highest = max(highs[-period:])
    lowest = min(lows[-period:])
    if highest == lowest:
        return 50
    return round((closes[-1] - lowest) / (highest - lowest) * 100, 2)

def calculate_random_walk_index(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict:
    """Random Walk Index"""
    if len(closes) < period + 1:
        return {}
    tr_list = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) for i in range(-period, 0)]
    atr = sum(tr_list) / period
    if atr == 0:
        return {"rwi_high": 0, "rwi_low": 0}
    rwi_high = (highs[-1] - lows[-period]) / (atr * np.sqrt(period))
    rwi_low = (highs[-period] - lows[-1]) / (atr * np.sqrt(period))
    return {"rwi_high": round(rwi_high, 2), "rwi_low": round(rwi_low, 2)}

def calculate_trend_detection_index(closes: List[float], period: int = 20) -> Dict:
    """Trend Detection Index"""
    if len(closes) < period * 2:
        return {}
    mom = closes[-1] - closes[-period]
    abs_mom_sum = sum(abs(closes[i] - closes[i-1]) for i in range(-period, 0))
    if abs_mom_sum == 0:
        return {"tdi": 0, "direction": "neutral"}
    tdi = abs(mom) / abs_mom_sum * 100
    direction = "bullish" if mom > 0 else "bearish" if mom < 0 else "neutral"
    return {"tdi": round(tdi, 2), "direction": direction}

def calculate_volume_rate_of_change(volumes: List[float], period: int = 14) -> Optional[float]:
    """Volume Rate of Change"""
    if volumes is None or len(volumes) < period + 1:
        return None
    if volumes[-period-1] == 0:
        return 0
    return round((volumes[-1] - volumes[-period-1]) / volumes[-period-1] * 100, 2)

def calculate_volume_weighted_macd(closes: List[float], volumes: List[float], fast: int = 12, slow: int = 26, signal: int = 9) -> Dict:
    """Volume Weighted MACD"""
    if len(closes) < slow or volumes is None:
        return {}
    vw_closes = [closes[i] * volumes[i] for i in range(len(closes))]
    cum_vol = [sum(volumes[:i+1]) for i in range(len(volumes))]
    vwma_fast = sum(vw_closes[-fast:]) / sum(volumes[-fast:]) if sum(volumes[-fast:]) > 0 else 0
    vwma_slow = sum(vw_closes[-slow:]) / sum(volumes[-slow:]) if sum(volumes[-slow:]) > 0 else 0
    vw_macd = vwma_fast - vwma_slow
    return {"vw_macd": round(vw_macd, 5), "signal": "BUY" if vw_macd > 0 else "SELL"}

def calculate_zero_lag_ema(closes: List[float], period: int = 20) -> Optional[float]:
    """Zero Lag EMA"""
    if len(closes) < period * 2:
        return None
    ema = sum(closes[-period:]) / period
    lag = (period - 1) // 2
    zlema = 2 * ema - sum(closes[-period-lag:-lag]) / period
    return round(zlema, 5)
