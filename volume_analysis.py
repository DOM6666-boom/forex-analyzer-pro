"""
ADVANCED VOLUME ANALYSIS MODULE - 35 Concepts
Volume Profile, VWAP Bands, Volume Clusters, Footprint Analysis, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ==================== VOLUME PROFILE ADVANCED (10) ====================

def calculate_volume_profile_advanced(highs: List[float], lows: List[float], closes: List[float], 
                                      volumes: List[float] = None, num_levels: int = 30) -> Dict:
    """Advanced Volume Profile with POC, VAH, VAL, HVN, LVN"""
    if len(closes) < 10:
        return {}
    if volumes is None:
        volumes = [abs(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    price_min, price_max = min(lows), max(highs)
    level_size = (price_max - price_min) / num_levels
    profile = defaultdict(lambda: {"volume": 0, "buy_vol": 0, "sell_vol": 0})
    for i in range(len(closes)):
        level = int((closes[i] - price_min) / level_size) if level_size > 0 else 0
        level = min(level, num_levels - 1)
        level_price = price_min + (level + 0.5) * level_size
        is_bullish = closes[i] > (highs[i] + lows[i]) / 2
        profile[round(level_price, 5)]["volume"] += volumes[i]
        if is_bullish:
            profile[round(level_price, 5)]["buy_vol"] += volumes[i]
        else:
            profile[round(level_price, 5)]["sell_vol"] += volumes[i]
    poc = max(profile.keys(), key=lambda x: profile[x]["volume"])
    total_vol = sum(p["volume"] for p in profile.values())
    sorted_levels = sorted(profile.items(), key=lambda x: x[1]["volume"], reverse=True)
    va_vol, va_levels = 0, []
    for price, data in sorted_levels:
        va_levels.append(price)
        va_vol += data["volume"]
        if va_vol >= total_vol * 0.7:
            break
    vah, val = max(va_levels), min(va_levels)
    avg_vol = total_vol / len(profile)
    hvn = sorted([p for p, v in profile.items() if v["volume"] > avg_vol * 1.5], reverse=True)[:5]
    lvn = sorted([p for p, v in profile.items() if v["volume"] < avg_vol * 0.5])[:5]
    return {"poc": poc, "vah": vah, "val": val, "hvn": hvn, "lvn": lvn, "profile": dict(profile), "total_volume": total_vol}

def calculate_developing_poc(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> List[float]:
    """Calculate Developing POC over time"""
    if len(closes) < 5:
        return []
    developing_pocs = []
    for i in range(5, len(closes) + 1):
        vp = calculate_volume_profile_advanced(highs[:i], lows[:i], closes[:i], volumes[:i] if volumes else None, 20)
        developing_pocs.append(vp.get("poc", closes[i-1]))
    return developing_pocs

def calculate_volume_weighted_price_levels(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> Dict:
    """Calculate Volume-Weighted Support/Resistance Levels"""
    vp = calculate_volume_profile_advanced(highs, lows, closes, volumes)
    current = closes[-1]
    resistance = sorted([p for p in vp.get("hvn", []) if p > current])[:3]
    support = sorted([p for p in vp.get("hvn", []) if p < current], reverse=True)[:3]
    return {"vw_resistance": resistance, "vw_support": support, "poc": vp.get("poc")}

def calculate_volume_delta_profile(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> Dict:
    """Volume Delta Profile - Buy vs Sell volume at each level"""
    if len(closes) < 10:
        return {}
    if volumes is None:
        volumes = [abs(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    delta_profile = defaultdict(lambda: {"delta": 0, "buy": 0, "sell": 0})
    price_min, price_max = min(lows), max(highs)
    level_size = (price_max - price_min) / 20
    for i in range(len(closes)):
        level = int((closes[i] - price_min) / level_size) if level_size > 0 else 0
        level_price = round(price_min + (level + 0.5) * level_size, 5)
        if closes[i] > opens[i]:
            delta_profile[level_price]["buy"] += volumes[i]
            delta_profile[level_price]["delta"] += volumes[i]
        else:
            delta_profile[level_price]["sell"] += volumes[i]
            delta_profile[level_price]["delta"] -= volumes[i]
    max_buy_level = max(delta_profile.keys(), key=lambda x: delta_profile[x]["buy"])
    max_sell_level = max(delta_profile.keys(), key=lambda x: delta_profile[x]["sell"])
    return {"delta_profile": dict(delta_profile), "max_buy_level": max_buy_level, "max_sell_level": max_sell_level}

def calculate_session_volume_profile(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None, session_bars: int = 24) -> Dict:
    """Calculate Volume Profile for Current Session"""
    if len(closes) < session_bars:
        session_bars = len(closes)
    return calculate_volume_profile_advanced(highs[-session_bars:], lows[-session_bars:], closes[-session_bars:], volumes[-session_bars:] if volumes else None)

def identify_volume_clusters(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None, threshold: float = 1.5) -> List[Dict]:
    """Identify High Volume Clusters (Institutional Activity)"""
    vp = calculate_volume_profile_advanced(highs, lows, closes, volumes)
    profile = vp.get("profile", {})
    avg_vol = vp.get("total_volume", 0) / len(profile) if profile else 0
    clusters = []
    for price, data in profile.items():
        if data["volume"] > avg_vol * threshold:
            clusters.append({"price": price, "volume": data["volume"], "buy_vol": data["buy_vol"], "sell_vol": data["sell_vol"],
                           "bias": "bullish" if data["buy_vol"] > data["sell_vol"] else "bearish"})
    return sorted(clusters, key=lambda x: x["volume"], reverse=True)[:5]

# ==================== VWAP ANALYSIS (8) ====================

def calculate_vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> Optional[float]:
    """Calculate VWAP"""
    if len(closes) < 1:
        return None
    if volumes is None:
        volumes = [1] * len(closes)
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    cum_tp_vol = sum(tp[i] * volumes[i] for i in range(len(closes)))
    cum_vol = sum(volumes)
    return round(cum_tp_vol / cum_vol, 5) if cum_vol > 0 else None

def calculate_vwap_bands(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None, std_mult: List[float] = [1, 2, 3]) -> Dict:
    """Calculate VWAP with Standard Deviation Bands"""
    vwap = calculate_vwap(highs, lows, closes, volumes)
    if vwap is None:
        return {}
    if volumes is None:
        volumes = [1] * len(closes)
    tp = [(highs[i] + lows[i] + closes[i]) / 3 for i in range(len(closes))]
    cum_vol = sum(volumes)
    variance = sum(volumes[i] * (tp[i] - vwap) ** 2 for i in range(len(closes))) / cum_vol if cum_vol > 0 else 0
    std = np.sqrt(variance)
    bands = {"vwap": vwap}
    for mult in std_mult:
        bands[f"upper_{mult}"] = round(vwap + mult * std, 5)
        bands[f"lower_{mult}"] = round(vwap - mult * std, 5)
    return bands

def calculate_anchored_vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None, anchor_idx: int = 0) -> Optional[float]:
    """Calculate Anchored VWAP from specific bar"""
    if anchor_idx >= len(closes):
        return None
    return calculate_vwap(highs[anchor_idx:], lows[anchor_idx:], closes[anchor_idx:], volumes[anchor_idx:] if volumes else None)

def calculate_rolling_vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None, period: int = 20) -> List[float]:
    """Calculate Rolling VWAP"""
    if len(closes) < period:
        return []
    rolling_vwaps = []
    for i in range(period, len(closes) + 1):
        vwap = calculate_vwap(highs[i-period:i], lows[i-period:i], closes[i-period:i], volumes[i-period:i] if volumes else None)
        rolling_vwaps.append(vwap)
    return rolling_vwaps

def calculate_vwap_deviation(closes: List[float], vwap: float) -> Dict:
    """Calculate Price Deviation from VWAP"""
    if vwap is None or vwap == 0:
        return {}
    current = closes[-1]
    deviation = (current - vwap) / vwap * 100
    return {"deviation_pct": round(deviation, 2), "above_vwap": current > vwap,
            "signal": "overbought" if deviation > 2 else "oversold" if deviation < -2 else "neutral"}

def calculate_vwap_cross_signals(closes: List[float], vwaps: List[float]) -> List[Dict]:
    """Detect VWAP Cross Signals"""
    if len(closes) < 2 or len(vwaps) < 2:
        return []
    signals = []
    for i in range(1, min(len(closes), len(vwaps))):
        if closes[i-1] < vwaps[i-1] and closes[i] > vwaps[i]:
            signals.append({"index": i, "type": "bullish_cross", "price": closes[i]})
        elif closes[i-1] > vwaps[i-1] and closes[i] < vwaps[i]:
            signals.append({"index": i, "type": "bearish_cross", "price": closes[i]})
    return signals

def calculate_multi_session_vwap(highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None, sessions: List[int] = [24, 48, 120]) -> Dict:
    """Calculate VWAP for Multiple Sessions"""
    result = {}
    for session in sessions:
        if len(closes) >= session:
            result[f"vwap_{session}"] = calculate_vwap(highs[-session:], lows[-session:], closes[-session:], volumes[-session:] if volumes else None)
    return result

# ==================== ORDER FLOW ANALYSIS (10) ====================

def calculate_cumulative_delta(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> Dict:
    """Calculate Cumulative Volume Delta"""
    if len(closes) < 2:
        return {}
    if volumes is None:
        volumes = [abs(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    deltas, cum_delta = [], 0
    for i in range(len(closes)):
        body = abs(closes[i] - opens[i])
        total_range = highs[i] - lows[i] if highs[i] != lows[i] else 0.0001
        if closes[i] > opens[i]:
            buy_ratio = 0.5 + (body / total_range) * 0.3
        elif closes[i] < opens[i]:
            buy_ratio = 0.5 - (body / total_range) * 0.3
        else:
            buy_ratio = 0.5
        buy_vol = volumes[i] * buy_ratio
        sell_vol = volumes[i] * (1 - buy_ratio)
        delta = buy_vol - sell_vol
        cum_delta += delta
        deltas.append({"delta": delta, "cum_delta": cum_delta})
    return {"current_delta": deltas[-1]["delta"], "cumulative_delta": cum_delta, "delta_history": deltas[-20:],
            "trend": "bullish" if cum_delta > 0 else "bearish"}

def detect_delta_divergence(closes: List[float], deltas: List[Dict]) -> Dict:
    """Detect Delta Divergence"""
    if len(closes) < 10 or len(deltas) < 10:
        return {"divergence": False}
    price_trend = "up" if closes[-1] > closes[-10] else "down"
    delta_trend = "up" if deltas[-1]["cum_delta"] > deltas[-10]["cum_delta"] else "down"
    if price_trend == "up" and delta_trend == "down":
        return {"divergence": True, "type": "bearish", "signal": "SELL", "confidence": 75}
    elif price_trend == "down" and delta_trend == "up":
        return {"divergence": True, "type": "bullish", "signal": "BUY", "confidence": 75}
    return {"divergence": False}

def calculate_delta_momentum(deltas: List[Dict], period: int = 14) -> Optional[float]:
    """Calculate Delta Momentum"""
    if len(deltas) < period:
        return None
    recent_deltas = [d["delta"] for d in deltas[-period:]]
    return sum(recent_deltas) / period

def identify_absorption_zones(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> List[Dict]:
    """Identify Volume Absorption Zones"""
    if len(closes) < 10:
        return []
    if volumes is None:
        volumes = [abs(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
    avg_range = sum(highs[i] - lows[i] for i in range(-20, 0)) / 20 if len(closes) >= 20 else sum(highs[i] - lows[i] for i in range(len(closes))) / len(closes)
    zones = []
    for i in range(-10, 0):
        vol = volumes[i]
        rng = highs[i] - lows[i]
        if vol > avg_vol * 1.5 and rng < avg_range * 0.7:
            zones.append({"index": len(closes) + i, "price": closes[i], "type": "absorption",
                         "bias": "bullish" if closes[i] > opens[i] else "bearish"})
    return zones

def calculate_buying_selling_pressure(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Calculate Buying and Selling Pressure"""
    if len(closes) < 1:
        return {}
    buying_pressure = sum(closes[i] - lows[i] for i in range(-min(20, len(closes)), 0))
    selling_pressure = sum(highs[i] - closes[i] for i in range(-min(20, len(closes)), 0))
    total = buying_pressure + selling_pressure
    if total == 0:
        return {"buying_pct": 50, "selling_pct": 50, "bias": "neutral"}
    return {"buying_pct": round(buying_pressure / total * 100, 1), "selling_pct": round(selling_pressure / total * 100, 1),
            "bias": "bullish" if buying_pressure > selling_pressure else "bearish"}

def identify_imbalance_zones(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None, threshold: float = 3.0) -> List[Dict]:
    """Identify Volume Imbalance Zones"""
    if len(closes) < 5:
        return []
    if volumes is None:
        volumes = [abs(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    avg_vol = sum(volumes) / len(volumes)
    imbalances = []
    for i in range(len(closes)):
        if volumes[i] > avg_vol * threshold:
            imbalances.append({"index": i, "price": closes[i], "volume": volumes[i], "volume_ratio": volumes[i] / avg_vol,
                              "type": "bullish_imbalance" if closes[i] > opens[i] else "bearish_imbalance"})
    return imbalances[-5:]

def calculate_volume_momentum(volumes: List[float], period: int = 14) -> Optional[float]:
    """Calculate Volume Momentum"""
    if len(volumes) < period + 1:
        return None
    return volumes[-1] / volumes[-period-1] * 100 - 100 if volumes[-period-1] != 0 else 0

def calculate_relative_volume(volumes: List[float], period: int = 20) -> Optional[float]:
    """Calculate Relative Volume (RVOL)"""
    if len(volumes) < period:
        return None
    avg_vol = sum(volumes[-period:-1]) / (period - 1)
    return round(volumes[-1] / avg_vol, 2) if avg_vol > 0 else 1

def identify_volume_climax(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> Dict:
    """Identify Volume Climax (Exhaustion)"""
    if len(closes) < 20:
        return {"climax": False}
    if volumes is None:
        volumes = [abs(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    max_vol = max(volumes[-20:])
    current_vol = volumes[-1]
    if current_vol >= max_vol * 0.9:
        return {"climax": True, "type": "buying_climax" if closes[-1] > opens[-1] else "selling_climax",
                "signal": "potential_reversal", "confidence": 70}
    return {"climax": False}

def calculate_volume_spread_analysis(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> Dict:
    """Volume Spread Analysis (VSA)"""
    if len(closes) < 5:
        return {}
    if volumes is None:
        volumes = [abs(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    avg_vol = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else sum(volumes) / len(volumes)
    avg_spread = sum(highs[i] - lows[i] for i in range(-20, 0)) / 20 if len(closes) >= 20 else sum(highs[i] - lows[i] for i in range(len(closes))) / len(closes)
    current_vol = volumes[-1]
    current_spread = highs[-1] - lows[-1]
    vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
    spread_ratio = current_spread / avg_spread if avg_spread > 0 else 1
    if vol_ratio > 1.5 and spread_ratio < 0.7:
        signal = "no_demand" if closes[-1] < opens[-1] else "no_supply"
    elif vol_ratio < 0.7 and spread_ratio > 1.3:
        signal = "test" if closes[-1] > opens[-1] else "upthrust"
    elif vol_ratio > 1.5 and spread_ratio > 1.3:
        signal = "strength" if closes[-1] > opens[-1] else "weakness"
    else:
        signal = "neutral"
    return {"signal": signal, "vol_ratio": round(vol_ratio, 2), "spread_ratio": round(spread_ratio, 2)}

# ==================== MASTER ANALYSIS FUNCTION ====================

def analyze_all_volume(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float] = None) -> Dict:
    """Comprehensive Volume Analysis - 35 concepts"""
    results = {
        "volume_profile": {},
        "vwap": {},
        "order_flow": {},
        "signals": {"bullish": 0, "bearish": 0},
        "overall_bias": "neutral",
        "strength": 50
    }
    # Volume Profile
    results["volume_profile"] = calculate_volume_profile_advanced(highs, lows, closes, volumes)
    results["volume_profile"]["clusters"] = identify_volume_clusters(highs, lows, closes, volumes)
    results["volume_profile"]["vw_levels"] = calculate_volume_weighted_price_levels(highs, lows, closes, volumes)
    # VWAP Analysis
    results["vwap"]["current"] = calculate_vwap(highs, lows, closes, volumes)
    results["vwap"]["bands"] = calculate_vwap_bands(highs, lows, closes, volumes)
    results["vwap"]["multi_session"] = calculate_multi_session_vwap(highs, lows, closes, volumes)
    if results["vwap"]["current"]:
        results["vwap"]["deviation"] = calculate_vwap_deviation(closes, results["vwap"]["current"])
    # Order Flow
    delta_data = calculate_cumulative_delta(opens, highs, lows, closes, volumes)
    results["order_flow"]["delta"] = delta_data
    if delta_data.get("delta_history"):
        results["order_flow"]["divergence"] = detect_delta_divergence(closes, delta_data["delta_history"])
    results["order_flow"]["pressure"] = calculate_buying_selling_pressure(opens, highs, lows, closes)
    results["order_flow"]["absorption"] = identify_absorption_zones(opens, highs, lows, closes, volumes)
    results["order_flow"]["imbalances"] = identify_imbalance_zones(opens, highs, lows, closes, volumes)
    results["order_flow"]["climax"] = identify_volume_climax(opens, highs, lows, closes, volumes)
    results["order_flow"]["vsa"] = calculate_volume_spread_analysis(opens, highs, lows, closes, volumes)
    results["order_flow"]["rvol"] = calculate_relative_volume(volumes) if volumes else None
    # Calculate signals
    if delta_data.get("trend") == "bullish":
        results["signals"]["bullish"] += 1
    else:
        results["signals"]["bearish"] += 1
    pressure = results["order_flow"].get("pressure", {})
    if pressure.get("bias") == "bullish":
        results["signals"]["bullish"] += 1
    elif pressure.get("bias") == "bearish":
        results["signals"]["bearish"] += 1
    vsa = results["order_flow"].get("vsa", {})
    if vsa.get("signal") in ["strength", "no_supply"]:
        results["signals"]["bullish"] += 1
    elif vsa.get("signal") in ["weakness", "no_demand"]:
        results["signals"]["bearish"] += 1
    # Overall bias
    if results["signals"]["bullish"] > results["signals"]["bearish"]:
        results["overall_bias"] = "BULLISH"
        results["strength"] = round(results["signals"]["bullish"] / (results["signals"]["bullish"] + results["signals"]["bearish"]) * 100, 1)
    elif results["signals"]["bearish"] > results["signals"]["bullish"]:
        results["overall_bias"] = "BEARISH"
        results["strength"] = round(results["signals"]["bearish"] / (results["signals"]["bullish"] + results["signals"]["bearish"]) * 100, 1)
    return results
