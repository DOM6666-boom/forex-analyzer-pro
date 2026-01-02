"""
SENTIMENT ANALYSIS MODULE - 15 Concepts
Fear/Greed, COT Analysis, Retail Positioning, Market Sentiment
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# ==================== FEAR & GREED ANALYSIS (5) ====================

def calculate_fear_greed_index(rsi: float, volatility_percentile: float, 
                                momentum: float, safe_haven_demand: float = 50) -> Dict:
    """Calculate Custom Fear & Greed Index"""
    
    rsi_score = (rsi - 50) * 2 if rsi else 0
    vol_score = (50 - volatility_percentile) if volatility_percentile else 0
    mom_score = min(50, max(-50, momentum * 10)) if momentum else 0
    safe_score = (50 - safe_haven_demand)
    
    index = 50 + (rsi_score * 0.3 + vol_score * 0.25 + mom_score * 0.25 + safe_score * 0.2)
    index = max(0, min(100, index))
    
    if index >= 80:
        classification = "Extreme Greed"
        signal = "SELL"
    elif index >= 60:
        classification = "Greed"
        signal = "CAUTION_LONG"
    elif index <= 20:
        classification = "Extreme Fear"
        signal = "BUY"
    elif index <= 40:
        classification = "Fear"
        signal = "CAUTION_SHORT"
    else:
        classification = "Neutral"
        signal = "NEUTRAL"
    
    return {
        "index": round(index, 1),
        "classification": classification,
        "signal": signal,
        "components": {
            "rsi_contribution": round(rsi_score * 0.3, 2),
            "volatility_contribution": round(vol_score * 0.25, 2),
            "momentum_contribution": round(mom_score * 0.25, 2),
            "safe_haven_contribution": round(safe_score * 0.2, 2)
        }
    }

def analyze_market_sentiment_score(bullish_signals: int, bearish_signals: int, 
                                   neutral_signals: int = 0) -> Dict:
    """Analyze Overall Market Sentiment Score"""
    total = bullish_signals + bearish_signals + neutral_signals
    if total == 0:
        return {"score": 50, "sentiment": "neutral"}
    
    score = (bullish_signals - bearish_signals) / total * 50 + 50
    
    if score >= 70:
        sentiment = "very_bullish"
    elif score >= 55:
        sentiment = "bullish"
    elif score <= 30:
        sentiment = "very_bearish"
    elif score <= 45:
        sentiment = "bearish"
    else:
        sentiment = "neutral"
    
    return {
        "score": round(score, 1),
        "sentiment": sentiment,
        "bullish_pct": round(bullish_signals / total * 100, 1),
        "bearish_pct": round(bearish_signals / total * 100, 1),
        "signal": "BUY" if score >= 60 else "SELL" if score <= 40 else "NEUTRAL"
    }

def calculate_sentiment_extremes(sentiment_history: List[float], period: int = 20) -> Dict:
    """Identify Sentiment Extremes for Contrarian Signals"""
    if len(sentiment_history) < period:
        return {}
    
    current = sentiment_history[-1]
    avg = sum(sentiment_history[-period:]) / period
    std = np.std(sentiment_history[-period:])
    
    if std == 0:
        z_score = 0
    else:
        z_score = (current - avg) / std
    
    if z_score > 2:
        extreme = "extreme_bullish"
        contrarian_signal = "SELL"
    elif z_score < -2:
        extreme = "extreme_bearish"
        contrarian_signal = "BUY"
    elif z_score > 1:
        extreme = "bullish"
        contrarian_signal = "CAUTION_LONG"
    elif z_score < -1:
        extreme = "bearish"
        contrarian_signal = "CAUTION_SHORT"
    else:
        extreme = "neutral"
        contrarian_signal = "NEUTRAL"
    
    return {
        "current": round(current, 1),
        "average": round(avg, 1),
        "z_score": round(z_score, 2),
        "extreme": extreme,
        "contrarian_signal": contrarian_signal
    }

def analyze_sentiment_divergence(price_direction: str, sentiment_direction: str) -> Dict:
    """Analyze Price vs Sentiment Divergence"""
    if price_direction == sentiment_direction:
        return {
            "divergence": False,
            "alignment": "confirmed",
            "signal": price_direction.upper()
        }
    
    if price_direction == "up" and sentiment_direction == "down":
        return {
            "divergence": True,
            "type": "bearish_divergence",
            "signal": "SELL",
            "note": "Price rising but sentiment falling - potential reversal"
        }
    elif price_direction == "down" and sentiment_direction == "up":
        return {
            "divergence": True,
            "type": "bullish_divergence",
            "signal": "BUY",
            "note": "Price falling but sentiment rising - potential reversal"
        }
    
    return {"divergence": False}

def calculate_sentiment_momentum(sentiment_history: List[float], period: int = 5) -> Dict:
    """Calculate Sentiment Momentum"""
    if len(sentiment_history) < period + 1:
        return {}
    
    current = sentiment_history[-1]
    previous = sentiment_history[-period-1]
    
    momentum = current - previous
    
    if momentum > 10:
        trend = "rapidly_improving"
    elif momentum > 5:
        trend = "improving"
    elif momentum < -10:
        trend = "rapidly_deteriorating"
    elif momentum < -5:
        trend = "deteriorating"
    else:
        trend = "stable"
    
    return {
        "momentum": round(momentum, 2),
        "trend": trend,
        "current": round(current, 1),
        "previous": round(previous, 1)
    }

# ==================== COT ANALYSIS (5) ====================

def analyze_cot_positioning(commercial_net: int, speculator_net: int, 
                            retail_net: int = 0) -> Dict:
    """Analyze COT Positioning"""
    
    if commercial_net > 0 and speculator_net < 0:
        smart_money_bias = "bullish"
        signal = "BUY"
        note = "Commercials accumulating, speculators selling"
    elif commercial_net < 0 and speculator_net > 0:
        smart_money_bias = "bearish"
        signal = "SELL"
        note = "Commercials distributing, speculators buying"
    elif commercial_net > 0:
        smart_money_bias = "slightly_bullish"
        signal = "LEAN_LONG"
        note = "Commercials net long"
    elif commercial_net < 0:
        smart_money_bias = "slightly_bearish"
        signal = "LEAN_SHORT"
        note = "Commercials net short"
    else:
        smart_money_bias = "neutral"
        signal = "NEUTRAL"
        note = "No clear positioning"
    
    return {
        "commercial_net": commercial_net,
        "speculator_net": speculator_net,
        "retail_net": retail_net,
        "smart_money_bias": smart_money_bias,
        "signal": signal,
        "note": note
    }

def calculate_cot_extremes(commercial_history: List[int], speculator_history: List[int],
                           lookback: int = 52) -> Dict:
    """Calculate COT Extreme Readings"""
    if len(commercial_history) < lookback or len(speculator_history) < lookback:
        return {}
    
    comm_current = commercial_history[-1]
    comm_max = max(commercial_history[-lookback:])
    comm_min = min(commercial_history[-lookback:])
    
    spec_current = speculator_history[-1]
    spec_max = max(speculator_history[-lookback:])
    spec_min = min(speculator_history[-lookback:])
    
    comm_percentile = (comm_current - comm_min) / (comm_max - comm_min) * 100 if comm_max != comm_min else 50
    spec_percentile = (spec_current - spec_min) / (spec_max - spec_min) * 100 if spec_max != spec_min else 50
    
    extreme_signal = None
    if comm_percentile > 90:
        extreme_signal = "Commercial extreme long - BULLISH"
    elif comm_percentile < 10:
        extreme_signal = "Commercial extreme short - BEARISH"
    elif spec_percentile > 90:
        extreme_signal = "Speculator extreme long - Contrarian BEARISH"
    elif spec_percentile < 10:
        extreme_signal = "Speculator extreme short - Contrarian BULLISH"
    
    return {
        "commercial_percentile": round(comm_percentile, 1),
        "speculator_percentile": round(spec_percentile, 1),
        "extreme_signal": extreme_signal,
        "is_extreme": extreme_signal is not None
    }

def analyze_cot_change(commercial_current: int, commercial_previous: int,
                       speculator_current: int, speculator_previous: int) -> Dict:
    """Analyze Week-over-Week COT Changes"""
    comm_change = commercial_current - commercial_previous
    spec_change = speculator_current - speculator_previous
    
    if comm_change > 0 and spec_change < 0:
        signal = "BULLISH"
        note = "Smart money buying, weak hands selling"
    elif comm_change < 0 and spec_change > 0:
        signal = "BEARISH"
        note = "Smart money selling, weak hands buying"
    elif comm_change > 0:
        signal = "LEAN_BULLISH"
        note = "Commercials adding longs"
    elif comm_change < 0:
        signal = "LEAN_BEARISH"
        note = "Commercials adding shorts"
    else:
        signal = "NEUTRAL"
        note = "No significant change"
    
    return {
        "commercial_change": comm_change,
        "speculator_change": spec_change,
        "signal": signal,
        "note": note
    }

def calculate_cot_index(net_position: int, history: List[int], lookback: int = 52) -> Optional[float]:
    """Calculate COT Index (0-100)"""
    if len(history) < lookback:
        return None
    
    max_pos = max(history[-lookback:])
    min_pos = min(history[-lookback:])
    
    if max_pos == min_pos:
        return 50
    
    index = (net_position - min_pos) / (max_pos - min_pos) * 100
    return round(index, 1)

def identify_cot_divergence(price_trend: str, commercial_trend: str) -> Dict:
    """Identify COT Divergence with Price"""
    if price_trend == "up" and commercial_trend == "down":
        return {
            "divergence": True,
            "type": "bearish",
            "signal": "SELL",
            "note": "Price rising but commercials reducing longs"
        }
    elif price_trend == "down" and commercial_trend == "up":
        return {
            "divergence": True,
            "type": "bullish",
            "signal": "BUY",
            "note": "Price falling but commercials adding longs"
        }
    
    return {"divergence": False, "alignment": "confirmed"}

# ==================== RETAIL POSITIONING (5) ====================

def analyze_retail_positioning(long_pct: float, short_pct: float) -> Dict:
    """Analyze Retail Positioning for Contrarian Signals"""
    
    if long_pct > 70:
        signal = "SELL"
        note = "Extreme retail long - contrarian bearish"
        strength = "strong"
    elif long_pct > 60:
        signal = "LEAN_SHORT"
        note = "Retail majority long - contrarian bearish"
        strength = "moderate"
    elif short_pct > 70:
        signal = "BUY"
        note = "Extreme retail short - contrarian bullish"
        strength = "strong"
    elif short_pct > 60:
        signal = "LEAN_LONG"
        note = "Retail majority short - contrarian bullish"
        strength = "moderate"
    else:
        signal = "NEUTRAL"
        note = "Balanced retail positioning"
        strength = "weak"
    
    return {
        "long_pct": round(long_pct, 1),
        "short_pct": round(short_pct, 1),
        "ratio": round(long_pct / short_pct, 2) if short_pct > 0 else 0,
        "contrarian_signal": signal,
        "strength": strength,
        "note": note
    }

def calculate_retail_sentiment_index(long_pct: float) -> Dict:
    """Calculate Retail Sentiment Index"""
    
    index = long_pct
    
    if index >= 75:
        zone = "extreme_greed"
        contrarian = "STRONG_SELL"
    elif index >= 60:
        zone = "greed"
        contrarian = "SELL"
    elif index <= 25:
        zone = "extreme_fear"
        contrarian = "STRONG_BUY"
    elif index <= 40:
        zone = "fear"
        contrarian = "BUY"
    else:
        zone = "neutral"
        contrarian = "NEUTRAL"
    
    return {
        "index": round(index, 1),
        "zone": zone,
        "contrarian_signal": contrarian
    }

def analyze_retail_flow(current_long_pct: float, previous_long_pct: float) -> Dict:
    """Analyze Retail Flow Changes"""
    change = current_long_pct - previous_long_pct
    
    if change > 10:
        flow = "strong_buying"
        contrarian = "SELL"
    elif change > 5:
        flow = "buying"
        contrarian = "LEAN_SHORT"
    elif change < -10:
        flow = "strong_selling"
        contrarian = "BUY"
    elif change < -5:
        flow = "selling"
        contrarian = "LEAN_LONG"
    else:
        flow = "stable"
        contrarian = "NEUTRAL"
    
    return {
        "change": round(change, 1),
        "flow": flow,
        "contrarian_signal": contrarian
    }

def calculate_retail_extreme_percentile(long_pct_history: List[float], current: float,
                                        lookback: int = 100) -> Dict:
    """Calculate Retail Positioning Percentile"""
    if len(long_pct_history) < lookback:
        return {}
    
    sorted_history = sorted(long_pct_history[-lookback:])
    percentile = sum(1 for x in sorted_history if x < current) / len(sorted_history) * 100
    
    if percentile > 90:
        extreme = "extreme_long"
        signal = "STRONG_SELL"
    elif percentile > 75:
        extreme = "high_long"
        signal = "SELL"
    elif percentile < 10:
        extreme = "extreme_short"
        signal = "STRONG_BUY"
    elif percentile < 25:
        extreme = "high_short"
        signal = "BUY"
    else:
        extreme = "normal"
        signal = "NEUTRAL"
    
    return {
        "percentile": round(percentile, 1),
        "extreme": extreme,
        "contrarian_signal": signal
    }

def analyze_retail_vs_price(price_direction: str, retail_direction: str) -> Dict:
    """Analyze Retail Positioning vs Price Movement"""
    
    if price_direction == "up" and retail_direction == "long":
        return {
            "alignment": "retail_chasing",
            "signal": "CAUTION_LONG",
            "note": "Retail chasing price higher - late to the move"
        }
    elif price_direction == "down" and retail_direction == "short":
        return {
            "alignment": "retail_chasing",
            "signal": "CAUTION_SHORT",
            "note": "Retail chasing price lower - late to the move"
        }
    elif price_direction == "up" and retail_direction == "short":
        return {
            "alignment": "retail_fighting",
            "signal": "BUY",
            "note": "Retail fighting the trend - bullish continuation"
        }
    elif price_direction == "down" and retail_direction == "long":
        return {
            "alignment": "retail_fighting",
            "signal": "SELL",
            "note": "Retail fighting the trend - bearish continuation"
        }
    
    return {"alignment": "neutral", "signal": "NEUTRAL"}

# ==================== MASTER ANALYSIS FUNCTION ====================

def analyze_all_sentiment(rsi: float = 50, volatility_percentile: float = 50,
                          momentum: float = 0, bullish_signals: int = 0,
                          bearish_signals: int = 0, commercial_net: int = 0,
                          speculator_net: int = 0, retail_long_pct: float = 50) -> Dict:
    """Comprehensive Sentiment Analysis - 15 concepts"""
    results = {
        "fear_greed": {},
        "market_sentiment": {},
        "cot": {},
        "retail": {},
        "signals": {"bullish": 0, "bearish": 0},
        "overall_sentiment": "neutral",
        "contrarian_signal": "NEUTRAL"
    }
    
    # Fear & Greed
    results["fear_greed"] = calculate_fear_greed_index(rsi, volatility_percentile, momentum)
    results["market_sentiment"]["score"] = analyze_market_sentiment_score(bullish_signals, bearish_signals)
    
    # COT Analysis
    results["cot"]["positioning"] = analyze_cot_positioning(commercial_net, speculator_net)
    
    # Retail Analysis
    results["retail"]["positioning"] = analyze_retail_positioning(retail_long_pct, 100 - retail_long_pct)
    results["retail"]["sentiment_index"] = calculate_retail_sentiment_index(retail_long_pct)
    
    # Calculate signals
    fg = results["fear_greed"]
    if fg.get("signal") == "BUY":
        results["signals"]["bullish"] += 2
    elif fg.get("signal") == "SELL":
        results["signals"]["bearish"] += 2
    
    cot = results["cot"]["positioning"]
    if cot.get("signal") in ["BUY", "LEAN_LONG"]:
        results["signals"]["bullish"] += 1
    elif cot.get("signal") in ["SELL", "LEAN_SHORT"]:
        results["signals"]["bearish"] += 1
    
    retail = results["retail"]["positioning"]
    if retail.get("contrarian_signal") in ["BUY", "LEAN_LONG"]:
        results["signals"]["bullish"] += 1
    elif retail.get("contrarian_signal") in ["SELL", "LEAN_SHORT"]:
        results["signals"]["bearish"] += 1
    
    # Overall sentiment
    if results["signals"]["bullish"] > results["signals"]["bearish"]:
        results["overall_sentiment"] = "BULLISH"
        results["contrarian_signal"] = "BUY"
    elif results["signals"]["bearish"] > results["signals"]["bullish"]:
        results["overall_sentiment"] = "BEARISH"
        results["contrarian_signal"] = "SELL"
    
    return results
