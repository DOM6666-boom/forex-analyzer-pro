"""
MULTI-TIMEFRAME ANALYSIS MODULE - 15 Concepts
MTF Confluence, HTF/LTF Alignment, Timeframe Correlation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# ==================== MTF TREND ANALYSIS (5) ====================

def analyze_mtf_trend(closes_htf: List[float], closes_ltf: List[float], period: int = 20) -> Dict:
    """Analyze Multi-Timeframe Trend Alignment"""
    if len(closes_htf) < period or len(closes_ltf) < period:
        return {"aligned": False}
    
    htf_sma = sum(closes_htf[-period:]) / period
    ltf_sma = sum(closes_ltf[-period:]) / period
    
    htf_trend = "bullish" if closes_htf[-1] > htf_sma else "bearish"
    ltf_trend = "bullish" if closes_ltf[-1] > ltf_sma else "bearish"
    
    aligned = htf_trend == ltf_trend
    
    return {
        "htf_trend": htf_trend,
        "ltf_trend": ltf_trend,
        "aligned": aligned,
        "confluence_strength": 100 if aligned else 50,
        "signal": htf_trend.upper() if aligned else "WAIT"
    }

def calculate_mtf_momentum(closes_htf: List[float], closes_ltf: List[float], period: int = 14) -> Dict:
    """Calculate Multi-Timeframe Momentum"""
    if len(closes_htf) < period or len(closes_ltf) < period:
        return {}
    
    htf_roc = (closes_htf[-1] - closes_htf[-period]) / closes_htf[-period] * 100
    ltf_roc = (closes_ltf[-1] - closes_ltf[-period]) / closes_ltf[-period] * 100
    
    htf_mom = "bullish" if htf_roc > 0 else "bearish"
    ltf_mom = "bullish" if ltf_roc > 0 else "bearish"
    
    return {
        "htf_momentum": htf_mom,
        "htf_roc": round(htf_roc, 2),
        "ltf_momentum": ltf_mom,
        "ltf_roc": round(ltf_roc, 2),
        "aligned": htf_mom == ltf_mom,
        "combined_strength": round((abs(htf_roc) + abs(ltf_roc)) / 2, 2)
    }

def identify_mtf_divergence(closes_htf: List[float], closes_ltf: List[float], period: int = 20) -> Dict:
    """Identify Multi-Timeframe Divergence"""
    if len(closes_htf) < period or len(closes_ltf) < period:
        return {"divergence": False}
    
    htf_direction = "up" if closes_htf[-1] > closes_htf[-period] else "down"
    ltf_direction = "up" if closes_ltf[-1] > closes_ltf[-period] else "down"
    
    if htf_direction != ltf_direction:
        return {
            "divergence": True,
            "htf_direction": htf_direction,
            "ltf_direction": ltf_direction,
            "signal": "CAUTION - Timeframe conflict",
            "recommendation": f"Follow HTF ({htf_direction})"
        }
    
    return {"divergence": False, "direction": htf_direction}

def calculate_mtf_strength(closes_list: List[List[float]], period: int = 20) -> Dict:
    """Calculate Strength Across Multiple Timeframes"""
    if not closes_list or any(len(c) < period for c in closes_list):
        return {}
    
    bullish_count = 0
    bearish_count = 0
    
    for closes in closes_list:
        sma = sum(closes[-period:]) / period
        if closes[-1] > sma:
            bullish_count += 1
        else:
            bearish_count += 1
    
    total = len(closes_list)
    
    return {
        "bullish_tfs": bullish_count,
        "bearish_tfs": bearish_count,
        "total_tfs": total,
        "alignment_pct": round(max(bullish_count, bearish_count) / total * 100, 1),
        "bias": "BULLISH" if bullish_count > bearish_count else "BEARISH" if bearish_count > bullish_count else "NEUTRAL"
    }

def detect_mtf_trend_change(closes_htf: List[float], closes_ltf: List[float], period: int = 20) -> Dict:
    """Detect Trend Change Across Timeframes"""
    if len(closes_htf) < period * 2 or len(closes_ltf) < period * 2:
        return {"change": False}
    
    htf_prev = "up" if closes_htf[-period] > closes_htf[-period*2] else "down"
    htf_curr = "up" if closes_htf[-1] > closes_htf[-period] else "down"
    
    ltf_prev = "up" if closes_ltf[-period] > closes_ltf[-period*2] else "down"
    ltf_curr = "up" if closes_ltf[-1] > closes_ltf[-period] else "down"
    
    htf_changed = htf_prev != htf_curr
    ltf_changed = ltf_prev != ltf_curr
    
    if htf_changed and ltf_changed:
        return {"change": True, "type": "confirmed", "new_direction": htf_curr, "signal": "STRONG"}
    elif ltf_changed and not htf_changed:
        return {"change": True, "type": "early_warning", "ltf_direction": ltf_curr, "signal": "WATCH"}
    elif htf_changed:
        return {"change": True, "type": "htf_shift", "htf_direction": htf_curr, "signal": "IMPORTANT"}
    
    return {"change": False}

# ==================== MTF STRUCTURE ANALYSIS (5) ====================

def analyze_mtf_structure(highs_htf: List[float], lows_htf: List[float], 
                          highs_ltf: List[float], lows_ltf: List[float]) -> Dict:
    """Analyze Market Structure Across Timeframes"""
    if len(highs_htf) < 10 or len(highs_ltf) < 10:
        return {}
    
    def get_structure(highs, lows):
        if len(highs) < 5:
            return "unknown"
        recent_high = max(highs[-5:])
        prev_high = max(highs[-10:-5])
        recent_low = min(lows[-5:])
        prev_low = min(lows[-10:-5])
        
        if recent_high > prev_high and recent_low > prev_low:
            return "bullish"
        elif recent_high < prev_high and recent_low < prev_low:
            return "bearish"
        return "ranging"
    
    htf_structure = get_structure(highs_htf, lows_htf)
    ltf_structure = get_structure(highs_ltf, lows_ltf)
    
    return {
        "htf_structure": htf_structure,
        "ltf_structure": ltf_structure,
        "aligned": htf_structure == ltf_structure,
        "recommendation": f"Trade {htf_structure}" if htf_structure == ltf_structure else "Wait for alignment"
    }

def find_mtf_key_levels(highs_htf: List[float], lows_htf: List[float],
                        highs_ltf: List[float], lows_ltf: List[float]) -> Dict:
    """Find Key Levels Across Timeframes"""
    if len(highs_htf) < 20 or len(highs_ltf) < 20:
        return {}
    
    htf_resistance = max(highs_htf[-20:])
    htf_support = min(lows_htf[-20:])
    ltf_resistance = max(highs_ltf[-20:])
    ltf_support = min(lows_ltf[-20:])
    
    confluence_resistance = []
    confluence_support = []
    
    tolerance = (htf_resistance - htf_support) * 0.02
    
    if abs(htf_resistance - ltf_resistance) < tolerance:
        confluence_resistance.append(round((htf_resistance + ltf_resistance) / 2, 5))
    
    if abs(htf_support - ltf_support) < tolerance:
        confluence_support.append(round((htf_support + ltf_support) / 2, 5))
    
    return {
        "htf_resistance": round(htf_resistance, 5),
        "htf_support": round(htf_support, 5),
        "ltf_resistance": round(ltf_resistance, 5),
        "ltf_support": round(ltf_support, 5),
        "confluence_resistance": confluence_resistance,
        "confluence_support": confluence_support,
        "has_confluence": len(confluence_resistance) > 0 or len(confluence_support) > 0
    }

def calculate_mtf_volatility(highs_htf: List[float], lows_htf: List[float], closes_htf: List[float],
                             highs_ltf: List[float], lows_ltf: List[float], closes_ltf: List[float],
                             period: int = 14) -> Dict:
    """Calculate Volatility Across Timeframes"""
    if len(closes_htf) < period or len(closes_ltf) < period:
        return {}
    
    def calc_atr(highs, lows, closes, p):
        tr_sum = sum(max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1])) 
                     for i in range(-p, 0))
        return tr_sum / p
    
    htf_atr = calc_atr(highs_htf, lows_htf, closes_htf, period)
    ltf_atr = calc_atr(highs_ltf, lows_ltf, closes_ltf, period)
    
    htf_atr_pct = htf_atr / closes_htf[-1] * 100
    ltf_atr_pct = ltf_atr / closes_ltf[-1] * 100
    
    return {
        "htf_atr": round(htf_atr, 5),
        "htf_atr_pct": round(htf_atr_pct, 2),
        "ltf_atr": round(ltf_atr, 5),
        "ltf_atr_pct": round(ltf_atr_pct, 2),
        "volatility_ratio": round(ltf_atr_pct / htf_atr_pct, 2) if htf_atr_pct > 0 else 1,
        "environment": "high_vol" if htf_atr_pct > 2 else "low_vol" if htf_atr_pct < 0.5 else "normal"
    }

def identify_mtf_order_blocks(opens_htf: List[float], highs_htf: List[float], lows_htf: List[float], closes_htf: List[float],
                               opens_ltf: List[float], highs_ltf: List[float], lows_ltf: List[float], closes_ltf: List[float]) -> Dict:
    """Identify Order Blocks Across Timeframes"""
    if len(closes_htf) < 10 or len(closes_ltf) < 10:
        return {}
    
    def find_ob(opens, highs, lows, closes):
        obs = []
        for i in range(-10, -2):
            if closes[i] < opens[i] and closes[i+1] > opens[i+1] and closes[i+2] > highs[i]:
                obs.append({"type": "bullish", "top": highs[i], "bottom": lows[i]})
            elif closes[i] > opens[i] and closes[i+1] < opens[i+1] and closes[i+2] < lows[i]:
                obs.append({"type": "bearish", "top": highs[i], "bottom": lows[i]})
        return obs[-3:] if obs else []
    
    htf_obs = find_ob(opens_htf, highs_htf, lows_htf, closes_htf)
    ltf_obs = find_ob(opens_ltf, highs_ltf, lows_ltf, closes_ltf)
    
    return {
        "htf_order_blocks": htf_obs,
        "ltf_order_blocks": ltf_obs,
        "htf_count": len(htf_obs),
        "ltf_count": len(ltf_obs)
    }

def analyze_mtf_fvg(opens_htf: List[float], highs_htf: List[float], lows_htf: List[float], closes_htf: List[float],
                    opens_ltf: List[float], highs_ltf: List[float], lows_ltf: List[float], closes_ltf: List[float]) -> Dict:
    """Analyze Fair Value Gaps Across Timeframes"""
    if len(closes_htf) < 5 or len(closes_ltf) < 5:
        return {}
    
    def find_fvg(highs, lows):
        fvgs = []
        for i in range(-5, -1):
            if lows[i+1] > highs[i-1]:
                fvgs.append({"type": "bullish", "top": lows[i+1], "bottom": highs[i-1]})
            elif highs[i+1] < lows[i-1]:
                fvgs.append({"type": "bearish", "top": lows[i-1], "bottom": highs[i+1]})
        return fvgs
    
    htf_fvgs = find_fvg(highs_htf, lows_htf)
    ltf_fvgs = find_fvg(highs_ltf, lows_ltf)
    
    return {
        "htf_fvgs": htf_fvgs,
        "ltf_fvgs": ltf_fvgs,
        "htf_count": len(htf_fvgs),
        "ltf_count": len(ltf_fvgs)
    }

# ==================== MTF ENTRY/EXIT (5) ====================

def find_mtf_entry_zone(closes_htf: List[float], highs_ltf: List[float], lows_ltf: List[float], 
                        closes_ltf: List[float], period: int = 20) -> Dict:
    """Find Optimal Entry Zone Using MTF"""
    if len(closes_htf) < period or len(closes_ltf) < period:
        return {}
    
    htf_sma = sum(closes_htf[-period:]) / period
    htf_trend = "bullish" if closes_htf[-1] > htf_sma else "bearish"
    
    ltf_high = max(highs_ltf[-10:])
    ltf_low = min(lows_ltf[-10:])
    ltf_range = ltf_high - ltf_low
    
    if htf_trend == "bullish":
        entry_zone_top = ltf_low + ltf_range * 0.382
        entry_zone_bottom = ltf_low
        stop_loss = ltf_low - ltf_range * 0.1
        take_profit = ltf_high + ltf_range * 0.5
    else:
        entry_zone_top = ltf_high
        entry_zone_bottom = ltf_high - ltf_range * 0.382
        stop_loss = ltf_high + ltf_range * 0.1
        take_profit = ltf_low - ltf_range * 0.5
    
    return {
        "htf_trend": htf_trend,
        "entry_zone_top": round(entry_zone_top, 5),
        "entry_zone_bottom": round(entry_zone_bottom, 5),
        "stop_loss": round(stop_loss, 5),
        "take_profit": round(take_profit, 5),
        "risk_reward": round(abs(take_profit - closes_ltf[-1]) / abs(closes_ltf[-1] - stop_loss), 2) if abs(closes_ltf[-1] - stop_loss) > 0 else 0
    }

def calculate_mtf_confluence_score(closes_htf: List[float], closes_mtf: List[float], closes_ltf: List[float],
                                   period: int = 20) -> Dict:
    """Calculate MTF Confluence Score"""
    if len(closes_htf) < period or len(closes_mtf) < period or len(closes_ltf) < period:
        return {}
    
    score = 0
    details = []
    
    htf_sma = sum(closes_htf[-period:]) / period
    mtf_sma = sum(closes_mtf[-period:]) / period
    ltf_sma = sum(closes_ltf[-period:]) / period
    
    htf_above = closes_htf[-1] > htf_sma
    mtf_above = closes_mtf[-1] > mtf_sma
    ltf_above = closes_ltf[-1] > ltf_sma
    
    if htf_above:
        score += 3
        details.append("HTF bullish (+3)")
    else:
        score -= 3
        details.append("HTF bearish (-3)")
    
    if mtf_above:
        score += 2
        details.append("MTF bullish (+2)")
    else:
        score -= 2
        details.append("MTF bearish (-2)")
    
    if ltf_above:
        score += 1
        details.append("LTF bullish (+1)")
    else:
        score -= 1
        details.append("LTF bearish (-1)")
    
    if htf_above == mtf_above == ltf_above:
        score += 2
        details.append("Full alignment (+2)")
    
    return {
        "score": score,
        "max_score": 8,
        "min_score": -8,
        "details": details,
        "signal": "STRONG BUY" if score >= 6 else "BUY" if score >= 3 else "STRONG SELL" if score <= -6 else "SELL" if score <= -3 else "NEUTRAL"
    }

def identify_mtf_pullback_entry(closes_htf: List[float], highs_ltf: List[float], lows_ltf: List[float],
                                 closes_ltf: List[float], period: int = 20) -> Dict:
    """Identify Pullback Entry Using MTF"""
    if len(closes_htf) < period or len(closes_ltf) < period:
        return {}
    
    htf_sma = sum(closes_htf[-period:]) / period
    htf_trend = "bullish" if closes_htf[-1] > htf_sma else "bearish"
    
    ltf_sma = sum(closes_ltf[-period:]) / period
    
    if htf_trend == "bullish":
        if closes_ltf[-1] < ltf_sma and closes_ltf[-2] >= sum(closes_ltf[-period-1:-1]) / period:
            return {
                "pullback": True,
                "type": "bullish_pullback",
                "entry": round(ltf_sma, 5),
                "signal": "BUY on pullback to LTF SMA"
            }
    else:
        if closes_ltf[-1] > ltf_sma and closes_ltf[-2] <= sum(closes_ltf[-period-1:-1]) / period:
            return {
                "pullback": True,
                "type": "bearish_pullback",
                "entry": round(ltf_sma, 5),
                "signal": "SELL on pullback to LTF SMA"
            }
    
    return {"pullback": False}

def calculate_mtf_exit_levels(closes_htf: List[float], highs_htf: List[float], lows_htf: List[float],
                              closes_ltf: List[float], position_type: str = "long") -> Dict:
    """Calculate Exit Levels Using MTF"""
    if len(closes_htf) < 20 or len(closes_ltf) < 20:
        return {}
    
    htf_high = max(highs_htf[-20:])
    htf_low = min(lows_htf[-20:])
    htf_range = htf_high - htf_low
    
    current = closes_ltf[-1]
    
    if position_type == "long":
        tp1 = current + htf_range * 0.382
        tp2 = current + htf_range * 0.618
        tp3 = htf_high
        sl = current - htf_range * 0.236
    else:
        tp1 = current - htf_range * 0.382
        tp2 = current - htf_range * 0.618
        tp3 = htf_low
        sl = current + htf_range * 0.236
    
    return {
        "take_profit_1": round(tp1, 5),
        "take_profit_2": round(tp2, 5),
        "take_profit_3": round(tp3, 5),
        "stop_loss": round(sl, 5),
        "rr_tp1": round(abs(tp1 - current) / abs(current - sl), 2) if abs(current - sl) > 0 else 0,
        "rr_tp2": round(abs(tp2 - current) / abs(current - sl), 2) if abs(current - sl) > 0 else 0,
        "rr_tp3": round(abs(tp3 - current) / abs(current - sl), 2) if abs(current - sl) > 0 else 0
    }

def analyze_mtf_timing(closes_htf: List[float], closes_mtf: List[float], closes_ltf: List[float]) -> Dict:
    """Analyze Optimal Entry Timing Using MTF"""
    if len(closes_htf) < 20 or len(closes_mtf) < 20 or len(closes_ltf) < 20:
        return {}
    
    htf_mom = closes_htf[-1] - closes_htf[-5]
    mtf_mom = closes_mtf[-1] - closes_mtf[-5]
    ltf_mom = closes_ltf[-1] - closes_ltf[-5]
    
    htf_dir = "up" if htf_mom > 0 else "down"
    mtf_dir = "up" if mtf_mom > 0 else "down"
    ltf_dir = "up" if ltf_mom > 0 else "down"
    
    if htf_dir == mtf_dir == ltf_dir:
        timing = "optimal"
        confidence = 90
    elif htf_dir == mtf_dir:
        timing = "good"
        confidence = 70
    elif htf_dir == ltf_dir:
        timing = "moderate"
        confidence = 50
    else:
        timing = "poor"
        confidence = 30
    
    return {
        "timing": timing,
        "confidence": confidence,
        "htf_direction": htf_dir,
        "mtf_direction": mtf_dir,
        "ltf_direction": ltf_dir,
        "recommendation": f"{'Enter' if timing in ['optimal', 'good'] else 'Wait'} - {htf_dir.upper()} bias"
    }

# ==================== MASTER ANALYSIS FUNCTION ====================

def analyze_all_mtf(opens_htf: List[float], highs_htf: List[float], lows_htf: List[float], closes_htf: List[float],
                    opens_ltf: List[float], highs_ltf: List[float], lows_ltf: List[float], closes_ltf: List[float]) -> Dict:
    """Comprehensive Multi-Timeframe Analysis - 15 concepts"""
    results = {
        "trend": {},
        "structure": {},
        "entry_exit": {},
        "signals": {"bullish": 0, "bearish": 0},
        "overall_bias": "neutral",
        "confluence_score": 0
    }
    
    # Trend Analysis
    results["trend"]["alignment"] = analyze_mtf_trend(closes_htf, closes_ltf)
    results["trend"]["momentum"] = calculate_mtf_momentum(closes_htf, closes_ltf)
    results["trend"]["divergence"] = identify_mtf_divergence(closes_htf, closes_ltf)
    results["trend"]["change"] = detect_mtf_trend_change(closes_htf, closes_ltf)
    
    # Structure Analysis
    results["structure"]["analysis"] = analyze_mtf_structure(highs_htf, lows_htf, highs_ltf, lows_ltf)
    results["structure"]["key_levels"] = find_mtf_key_levels(highs_htf, lows_htf, highs_ltf, lows_ltf)
    results["structure"]["volatility"] = calculate_mtf_volatility(highs_htf, lows_htf, closes_htf, highs_ltf, lows_ltf, closes_ltf)
    results["structure"]["order_blocks"] = identify_mtf_order_blocks(opens_htf, highs_htf, lows_htf, closes_htf, opens_ltf, highs_ltf, lows_ltf, closes_ltf)
    results["structure"]["fvg"] = analyze_mtf_fvg(opens_htf, highs_htf, lows_htf, closes_htf, opens_ltf, highs_ltf, lows_ltf, closes_ltf)
    
    # Entry/Exit Analysis
    results["entry_exit"]["entry_zone"] = find_mtf_entry_zone(closes_htf, highs_ltf, lows_ltf, closes_ltf)
    results["entry_exit"]["pullback"] = identify_mtf_pullback_entry(closes_htf, highs_ltf, lows_ltf, closes_ltf)
    results["entry_exit"]["timing"] = analyze_mtf_timing(closes_htf, closes_htf, closes_ltf)
    
    # Calculate signals
    trend_align = results["trend"]["alignment"]
    if trend_align.get("aligned"):
        if trend_align.get("htf_trend") == "bullish":
            results["signals"]["bullish"] += 2
        else:
            results["signals"]["bearish"] += 2
    
    mom = results["trend"]["momentum"]
    if mom.get("aligned"):
        if mom.get("htf_momentum") == "bullish":
            results["signals"]["bullish"] += 1
        else:
            results["signals"]["bearish"] += 1
    
    structure = results["structure"]["analysis"]
    if structure.get("aligned"):
        if structure.get("htf_structure") == "bullish":
            results["signals"]["bullish"] += 1
        else:
            results["signals"]["bearish"] += 1
    
    # Overall bias
    total = results["signals"]["bullish"] + results["signals"]["bearish"]
    if total > 0:
        if results["signals"]["bullish"] > results["signals"]["bearish"]:
            results["overall_bias"] = "BULLISH"
            results["confluence_score"] = round(results["signals"]["bullish"] / total * 100, 1)
        elif results["signals"]["bearish"] > results["signals"]["bullish"]:
            results["overall_bias"] = "BEARISH"
            results["confluence_score"] = round(results["signals"]["bearish"] / total * 100, 1)
    
    return results
