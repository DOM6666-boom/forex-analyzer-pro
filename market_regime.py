"""
MARKET REGIME ANALYSIS MODULE - 20 Concepts
Trend Detection, Volatility Regime, Mean Reversion, Momentum Regime
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# ==================== TREND REGIME (5) ====================

def detect_trend_regime(closes: List[float], period: int = 20) -> Dict:
    """Detect Current Trend Regime"""
    if len(closes) < period * 2:
        return {"regime": "unknown", "strength": 0}
    
    sma_short = sum(closes[-period:]) / period
    sma_long = sum(closes[-period*2:]) / (period * 2)
    
    slope = (closes[-1] - closes[-period]) / period
    
    if closes[-1] > sma_short > sma_long and slope > 0:
        regime = "strong_uptrend"
        strength = min(100, abs(slope) * 10000)
    elif closes[-1] > sma_short and slope > 0:
        regime = "uptrend"
        strength = min(80, abs(slope) * 8000)
    elif closes[-1] < sma_short < sma_long and slope < 0:
        regime = "strong_downtrend"
        strength = min(100, abs(slope) * 10000)
    elif closes[-1] < sma_short and slope < 0:
        regime = "downtrend"
        strength = min(80, abs(slope) * 8000)
    else:
        regime = "ranging"
        strength = 50
    
    return {"regime": regime, "strength": round(strength, 1), "slope": round(slope, 6),
            "price_vs_sma": "above" if closes[-1] > sma_short else "below"}

def calculate_adx_regime(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict:
    """ADX-based Trend Regime Detection"""
    if len(closes) < period + 1:
        return {"regime": "unknown", "adx": None}
    
    tr_list, plus_dm, minus_dm = [], [], []
    for i in range(-period, 0):
        tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
        tr_list.append(tr)
        plus_dm.append(max(0, highs[i] - highs[i-1]) if highs[i] - highs[i-1] > lows[i-1] - lows[i] else 0)
        minus_dm.append(max(0, lows[i-1] - lows[i]) if lows[i-1] - lows[i] > highs[i] - highs[i-1] else 0)
    
    atr = sum(tr_list) / period
    if atr == 0:
        return {"regime": "unknown", "adx": 0}
    
    plus_di = sum(plus_dm) / atr * 100 / period
    minus_di = sum(minus_dm) / atr * 100 / period
    dx = abs(plus_di - minus_di) / (plus_di + minus_di) * 100 if (plus_di + minus_di) > 0 else 0
    adx = dx
    
    if adx > 40:
        regime = "strong_trend"
    elif adx > 25:
        regime = "trending"
    elif adx > 15:
        regime = "weak_trend"
    else:
        regime = "ranging"
    
    direction = "bullish" if plus_di > minus_di else "bearish"
    
    return {"regime": regime, "adx": round(adx, 2), "plus_di": round(plus_di, 2),
            "minus_di": round(minus_di, 2), "direction": direction}

def detect_trend_exhaustion(closes: List[float], period: int = 14) -> Dict:
    """Detect Trend Exhaustion"""
    if len(closes) < period * 2:
        return {"exhaustion": False}
    
    recent_move = abs(closes[-1] - closes[-period])
    prior_move = abs(closes[-period] - closes[-period*2])
    
    if prior_move == 0:
        return {"exhaustion": False}
    
    momentum_ratio = recent_move / prior_move
    
    if momentum_ratio < 0.5:
        return {"exhaustion": True, "type": "momentum_declining", "ratio": round(momentum_ratio, 2)}
    elif momentum_ratio > 2:
        return {"exhaustion": True, "type": "climax_move", "ratio": round(momentum_ratio, 2)}
    
    return {"exhaustion": False, "ratio": round(momentum_ratio, 2)}

def calculate_trend_consistency(closes: List[float], period: int = 20) -> Optional[float]:
    """Calculate Trend Consistency Score"""
    if len(closes) < period:
        return None
    
    up_days = sum(1 for i in range(-period+1, 0) if closes[i] > closes[i-1])
    consistency = up_days / (period - 1) * 100
    
    if consistency > 50:
        return round(consistency, 1)
    else:
        return round(100 - consistency, 1)

def identify_trend_change(closes: List[float], period: int = 20) -> Dict:
    """Identify Potential Trend Change"""
    if len(closes) < period * 2:
        return {"change_detected": False}
    
    sma = sum(closes[-period:]) / period
    prev_sma = sum(closes[-period-5:-5]) / period
    
    price_cross = (closes[-2] < prev_sma and closes[-1] > sma) or (closes[-2] > prev_sma and closes[-1] < sma)
    
    recent_trend = "up" if closes[-1] > closes[-period] else "down"
    prior_trend = "up" if closes[-period] > closes[-period*2] else "down"
    
    if recent_trend != prior_trend or price_cross:
        return {"change_detected": True, "from": prior_trend, "to": recent_trend,
                "confirmation": "sma_cross" if price_cross else "direction_change"}
    
    return {"change_detected": False}

# ==================== VOLATILITY REGIME (5) ====================

def detect_volatility_regime(highs: List[float], lows: List[float], closes: List[float], period: int = 14) -> Dict:
    """Detect Current Volatility Regime"""
    if len(closes) < period * 2:
        return {"regime": "unknown"}
    
    tr_recent = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) 
                 for i in range(-period, 0)]
    tr_prior = [max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) 
                for i in range(-period*2, -period)]
    
    atr_recent = sum(tr_recent) / period
    atr_prior = sum(tr_prior) / period
    
    if atr_prior == 0:
        return {"regime": "unknown"}
    
    vol_ratio = atr_recent / atr_prior
    
    if vol_ratio > 1.5:
        regime = "high_volatility"
    elif vol_ratio > 1.2:
        regime = "expanding"
    elif vol_ratio < 0.7:
        regime = "low_volatility"
    elif vol_ratio < 0.85:
        regime = "contracting"
    else:
        regime = "normal"
    
    return {"regime": regime, "atr_current": round(atr_recent, 5), "atr_prior": round(atr_prior, 5),
            "vol_ratio": round(vol_ratio, 2)}

def calculate_volatility_percentile(highs: List[float], lows: List[float], closes: List[float], 
                                    lookback: int = 100, period: int = 14) -> Optional[float]:
    """Calculate Volatility Percentile"""
    if len(closes) < lookback:
        return None
    
    atr_values = []
    for i in range(lookback - period, 0):
        if i - period >= -len(closes):
            tr_sum = sum(max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])) 
                         for j in range(i-period, i) if j >= -len(closes) and j-1 >= -len(closes))
            if tr_sum > 0:
                atr_values.append(tr_sum / period)
    
    if not atr_values:
        return 50
    
    current_atr = atr_values[-1] if atr_values else 0
    percentile = sum(1 for a in atr_values if a < current_atr) / len(atr_values) * 100 if len(atr_values) > 0 else 50
    
    return round(percentile, 1)

def detect_volatility_breakout(highs: List[float], lows: List[float], closes: List[float], 
                               period: int = 20, threshold: float = 1.5) -> Dict:
    """Detect Volatility Breakout"""
    if len(closes) < period + 5:
        return {"breakout": False}
    
    bb_std = np.std(closes[-period:])
    bb_mean = sum(closes[-period:]) / period
    bb_width = (bb_std * 2) / bb_mean * 100
    
    prior_widths = []
    for i in range(-5, 0):
        std = np.std(closes[i-period:i])
        mean = sum(closes[i-period:i]) / period
        prior_widths.append((std * 2) / mean * 100 if mean != 0 else 0)
    
    avg_prior_width = sum(prior_widths) / len(prior_widths) if prior_widths else bb_width
    
    if bb_width > avg_prior_width * threshold:
        return {"breakout": True, "type": "expansion", "current_width": round(bb_width, 2),
                "avg_width": round(avg_prior_width, 2)}
    elif bb_width < avg_prior_width / threshold:
        return {"breakout": True, "type": "squeeze", "current_width": round(bb_width, 2),
                "avg_width": round(avg_prior_width, 2)}
    
    return {"breakout": False, "current_width": round(bb_width, 2)}

def calculate_historical_volatility(closes: List[float], period: int = 20) -> Optional[float]:
    """Calculate Historical Volatility (Annualized)"""
    if len(closes) < period + 1:
        return None
    
    returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(-period, 0)]
    std_return = np.std(returns)
    hv = std_return * np.sqrt(252) * 100
    
    return round(hv, 2)

def detect_volatility_clustering(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Dict:
    """Detect Volatility Clustering"""
    if len(closes) < period * 2:
        return {"clustering": False}
    
    ranges = [highs[i] - lows[i] for i in range(-period*2, 0)]
    
    high_vol_count = 0
    low_vol_count = 0
    avg_range = sum(ranges) / len(ranges)
    
    for i in range(-5, 0):
        if ranges[i] > avg_range * 1.2:
            high_vol_count += 1
        elif ranges[i] < avg_range * 0.8:
            low_vol_count += 1
    
    if high_vol_count >= 3:
        return {"clustering": True, "type": "high_volatility_cluster", "count": high_vol_count}
    elif low_vol_count >= 3:
        return {"clustering": True, "type": "low_volatility_cluster", "count": low_vol_count}
    
    return {"clustering": False}

# ==================== MEAN REVERSION REGIME (5) ====================

def detect_mean_reversion_setup(closes: List[float], period: int = 20) -> Dict:
    """Detect Mean Reversion Setup"""
    if len(closes) < period:
        return {"setup": False}
    
    mean = sum(closes[-period:]) / period
    std = np.std(closes[-period:])
    
    if std == 0:
        return {"setup": False}
    
    z_score = (closes[-1] - mean) / std
    
    if z_score > 2:
        return {"setup": True, "type": "overbought", "z_score": round(z_score, 2),
                "target": round(mean, 5), "signal": "SELL"}
    elif z_score < -2:
        return {"setup": True, "type": "oversold", "z_score": round(z_score, 2),
                "target": round(mean, 5), "signal": "BUY"}
    
    return {"setup": False, "z_score": round(z_score, 2)}

def calculate_mean_reversion_bands(closes: List[float], period: int = 20, num_std: float = 2) -> Dict:
    """Calculate Mean Reversion Bands"""
    if len(closes) < period:
        return {}
    
    mean = sum(closes[-period:]) / period
    std = np.std(closes[-period:])
    
    return {
        "mean": round(mean, 5),
        "upper_1std": round(mean + std, 5),
        "upper_2std": round(mean + 2*std, 5),
        "lower_1std": round(mean - std, 5),
        "lower_2std": round(mean - 2*std, 5),
        "current_position": "above_2std" if closes[-1] > mean + 2*std else
                          "above_1std" if closes[-1] > mean + std else
                          "below_2std" if closes[-1] < mean - 2*std else
                          "below_1std" if closes[-1] < mean - std else "within_1std"
    }

def calculate_rsi_mean_reversion(closes: List[float], period: int = 14) -> Dict:
    """RSI-based Mean Reversion Signal"""
    if len(closes) < period + 1:
        return {}
    
    gains, losses = [], []
    for i in range(-period, 0):
        change = closes[i] - closes[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        rsi = 100
    else:
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
    
    signal = None
    if rsi > 70:
        signal = "SELL"
    elif rsi < 30:
        signal = "BUY"
    
    return {"rsi": round(rsi, 2), "signal": signal, "zone": "overbought" if rsi > 70 else "oversold" if rsi < 30 else "neutral"}

def detect_price_extreme(closes: List[float], period: int = 50) -> Dict:
    """Detect Price Extreme for Mean Reversion"""
    if len(closes) < period:
        return {"extreme": False}
    
    highest = max(closes[-period:])
    lowest = min(closes[-period:])
    current = closes[-1]
    
    range_pct = (current - lowest) / (highest - lowest) * 100 if highest != lowest else 50
    
    if range_pct > 95:
        return {"extreme": True, "type": "near_high", "percentile": round(range_pct, 1), "signal": "SELL"}
    elif range_pct < 5:
        return {"extreme": True, "type": "near_low", "percentile": round(range_pct, 1), "signal": "BUY"}
    
    return {"extreme": False, "percentile": round(range_pct, 1)}

def calculate_mean_reversion_probability(closes: List[float], period: int = 20) -> Dict:
    """Calculate Mean Reversion Probability"""
    if len(closes) < period:
        return {}
    
    mean = sum(closes[-period:]) / period
    std = np.std(closes[-period:])
    
    if std == 0:
        return {"probability": 50}
    
    z = abs((closes[-1] - mean) / std)
    
    if z > 3:
        prob = 99
    elif z > 2:
        prob = 95
    elif z > 1.5:
        prob = 85
    elif z > 1:
        prob = 70
    else:
        prob = 50
    
    return {"probability": prob, "z_score": round(z, 2), "direction": "down" if closes[-1] > mean else "up"}

# ==================== MOMENTUM REGIME (5) ====================

def detect_momentum_regime(closes: List[float], period: int = 14) -> Dict:
    """Detect Momentum Regime"""
    if len(closes) < period * 2:
        return {"regime": "unknown"}
    
    roc = (closes[-1] - closes[-period]) / closes[-period] * 100
    prev_roc = (closes[-period] - closes[-period*2]) / closes[-period*2] * 100
    
    if roc > 5 and roc > prev_roc:
        regime = "strong_bullish_momentum"
    elif roc > 2:
        regime = "bullish_momentum"
    elif roc < -5 and roc < prev_roc:
        regime = "strong_bearish_momentum"
    elif roc < -2:
        regime = "bearish_momentum"
    else:
        regime = "no_momentum"
    
    return {"regime": regime, "roc": round(roc, 2), "prev_roc": round(prev_roc, 2),
            "accelerating": abs(roc) > abs(prev_roc)}

def calculate_momentum_score(closes: List[float], periods: List[int] = [5, 10, 20, 50]) -> Dict:
    """Calculate Multi-Period Momentum Score"""
    if len(closes) < max(periods):
        return {}
    
    scores = {}
    total_score = 0
    
    for p in periods:
        roc = (closes[-1] - closes[-p]) / closes[-p] * 100
        scores[f"roc_{p}"] = round(roc, 2)
        if roc > 0:
            total_score += 1
        else:
            total_score -= 1
    
    return {"scores": scores, "total_score": total_score, "max_score": len(periods),
            "bias": "bullish" if total_score > 0 else "bearish" if total_score < 0 else "neutral"}

def detect_momentum_divergence(closes: List[float], period: int = 14) -> Dict:
    """Detect Momentum Divergence"""
    if len(closes) < period * 3:
        return {"divergence": False}
    
    price_higher = closes[-1] > closes[-period]
    
    gains, losses = [], []
    for i in range(-period, 0):
        change = closes[i] - closes[i-1]
        gains.append(max(0, change))
        losses.append(max(0, -change))
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    rsi_current = 100 - (100 / (1 + avg_gain/avg_loss)) if avg_loss > 0 else 100
    
    gains_prev, losses_prev = [], []
    for i in range(-period*2, -period):
        change = closes[i] - closes[i-1]
        gains_prev.append(max(0, change))
        losses_prev.append(max(0, -change))
    
    avg_gain_prev = sum(gains_prev) / period
    avg_loss_prev = sum(losses_prev) / period
    rsi_prev = 100 - (100 / (1 + avg_gain_prev/avg_loss_prev)) if avg_loss_prev > 0 else 100
    
    rsi_higher = rsi_current > rsi_prev
    
    if price_higher and not rsi_higher:
        return {"divergence": True, "type": "bearish", "signal": "SELL"}
    elif not price_higher and rsi_higher:
        return {"divergence": True, "type": "bullish", "signal": "BUY"}
    
    return {"divergence": False}

def calculate_momentum_acceleration(closes: List[float], period: int = 10) -> Dict:
    """Calculate Momentum Acceleration"""
    if len(closes) < period * 3:
        return {}
    
    mom1 = closes[-1] - closes[-period]
    mom2 = closes[-period] - closes[-period*2]
    mom3 = closes[-period*2] - closes[-period*3]
    
    accel1 = mom1 - mom2
    accel2 = mom2 - mom3
    
    if accel1 > 0 and accel1 > accel2:
        status = "accelerating_up"
    elif accel1 < 0 and accel1 < accel2:
        status = "accelerating_down"
    elif accel1 > 0:
        status = "decelerating_up"
    elif accel1 < 0:
        status = "decelerating_down"
    else:
        status = "neutral"
    
    return {"acceleration": round(accel1, 5), "prev_acceleration": round(accel2, 5), "status": status}

def identify_momentum_shift(closes: List[float], period: int = 14) -> Dict:
    """Identify Momentum Shift"""
    if len(closes) < period * 2:
        return {"shift": False}
    
    recent_mom = sum(closes[i] - closes[i-1] for i in range(-period, 0))
    prior_mom = sum(closes[i] - closes[i-1] for i in range(-period*2, -period))
    
    if recent_mom > 0 and prior_mom < 0:
        return {"shift": True, "from": "bearish", "to": "bullish", "signal": "BUY"}
    elif recent_mom < 0 and prior_mom > 0:
        return {"shift": True, "from": "bullish", "to": "bearish", "signal": "SELL"}
    
    return {"shift": False}

# ==================== MASTER ANALYSIS FUNCTION ====================

def analyze_all_regimes(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Comprehensive Market Regime Analysis - 20 concepts"""
    results = {
        "trend": {},
        "volatility": {},
        "mean_reversion": {},
        "momentum": {},
        "signals": {"bullish": 0, "bearish": 0},
        "overall_regime": "unknown",
        "trading_approach": "neutral"
    }
    
    # Trend Regime
    results["trend"]["regime"] = detect_trend_regime(closes)
    results["trend"]["adx"] = calculate_adx_regime(highs, lows, closes)
    results["trend"]["exhaustion"] = detect_trend_exhaustion(closes)
    results["trend"]["consistency"] = calculate_trend_consistency(closes)
    results["trend"]["change"] = identify_trend_change(closes)
    
    # Volatility Regime
    results["volatility"]["regime"] = detect_volatility_regime(highs, lows, closes)
    results["volatility"]["percentile"] = calculate_volatility_percentile(highs, lows, closes)
    results["volatility"]["breakout"] = detect_volatility_breakout(highs, lows, closes)
    results["volatility"]["historical"] = calculate_historical_volatility(closes)
    results["volatility"]["clustering"] = detect_volatility_clustering(highs, lows, closes)
    
    # Mean Reversion
    results["mean_reversion"]["setup"] = detect_mean_reversion_setup(closes)
    results["mean_reversion"]["bands"] = calculate_mean_reversion_bands(closes)
    results["mean_reversion"]["rsi"] = calculate_rsi_mean_reversion(closes)
    results["mean_reversion"]["extreme"] = detect_price_extreme(closes)
    results["mean_reversion"]["probability"] = calculate_mean_reversion_probability(closes)
    
    # Momentum
    results["momentum"]["regime"] = detect_momentum_regime(closes)
    results["momentum"]["score"] = calculate_momentum_score(closes)
    results["momentum"]["divergence"] = detect_momentum_divergence(closes)
    results["momentum"]["acceleration"] = calculate_momentum_acceleration(closes)
    results["momentum"]["shift"] = identify_momentum_shift(closes)
    
    # Calculate signals
    trend_regime = results["trend"]["regime"].get("regime", "")
    if "uptrend" in trend_regime:
        results["signals"]["bullish"] += 2
    elif "downtrend" in trend_regime:
        results["signals"]["bearish"] += 2
    
    adx_dir = results["trend"]["adx"].get("direction", "")
    if adx_dir == "bullish":
        results["signals"]["bullish"] += 1
    elif adx_dir == "bearish":
        results["signals"]["bearish"] += 1
    
    mr_signal = results["mean_reversion"]["setup"].get("signal", "")
    if mr_signal == "BUY":
        results["signals"]["bullish"] += 1
    elif mr_signal == "SELL":
        results["signals"]["bearish"] += 1
    
    mom_regime = results["momentum"]["regime"].get("regime", "")
    if "bullish" in mom_regime:
        results["signals"]["bullish"] += 1
    elif "bearish" in mom_regime:
        results["signals"]["bearish"] += 1
    
    # Determine overall regime and trading approach
    trend_r = results["trend"]["regime"].get("regime", "ranging")
    vol_r = results["volatility"]["regime"].get("regime", "normal")
    
    if "trend" in trend_r or results["trend"]["adx"].get("adx", 0) > 25:
        results["overall_regime"] = "trending"
        results["trading_approach"] = "trend_following"
    elif vol_r in ["low_volatility", "contracting"]:
        results["overall_regime"] = "ranging"
        results["trading_approach"] = "mean_reversion"
    elif vol_r in ["high_volatility", "expanding"]:
        results["overall_regime"] = "volatile"
        results["trading_approach"] = "breakout"
    else:
        results["overall_regime"] = "mixed"
        results["trading_approach"] = "selective"
    
    # Overall bias
    if results["signals"]["bullish"] > results["signals"]["bearish"]:
        results["overall_bias"] = "BULLISH"
    elif results["signals"]["bearish"] > results["signals"]["bullish"]:
        results["overall_bias"] = "BEARISH"
    else:
        results["overall_bias"] = "NEUTRAL"
    
    return results
