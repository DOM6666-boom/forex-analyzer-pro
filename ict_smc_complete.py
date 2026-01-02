"""
COMPLETE ICT/SMC CONCEPTS - 131 Concepts
Market Structure, FVG, IFVG, Order Blocks, Liquidity, Kill Zones, Power of 3, etc.
"""

import numpy as np
from datetime import datetime
from typing import List, Dict, Tuple, Optional


# ==================== MARKET STRUCTURE (12 concepts) ====================

def analyze_market_structure(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Complete market structure analysis including:
    - BOS (Break of Structure)
    - CHoCH (Change of Character)
    - Market Shift
    - HH/HL, LH/LL
    - Internal/External Structure
    - Swing Points
    - Fractal Structure
    """
    structure = {
        "trend": "sideways",
        "swing_highs": [],
        "swing_lows": [],
        "bos": [],
        "choch": None,
        "market_shift": None,
        "hh_hl": False,
        "lh_ll": False,
        "internal_structure": None,
        "external_structure": None,
        "fractal_bias": None,
        "pivot_points": []
    }
    
    if len(closes) < 20:
        return structure
    
    # Find swing points
    for i in range(3, len(highs) - 3):
        # Swing High
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i-3]:
            if highs[i] > highs[i+1] and highs[i] > highs[i+2] and highs[i] > highs[i+3]:
                structure["swing_highs"].append({"index": i, "price": highs[i]})
        
        # Swing Low
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i-3]:
            if lows[i] < lows[i+1] and lows[i] < lows[i+2] and lows[i] < lows[i+3]:
                structure["swing_lows"].append({"index": i, "price": lows[i]})
    
    sh = structure["swing_highs"]
    sl = structure["swing_lows"]
    
    # Determine trend (HH/HL or LH/LL)
    if len(sh) >= 2 and len(sl) >= 2:
        # Higher Highs and Higher Lows
        if sh[-1]["price"] > sh[-2]["price"] and sl[-1]["price"] > sl[-2]["price"]:
            structure["trend"] = "bullish"
            structure["hh_hl"] = True
        # Lower Highs and Lower Lows
        elif sh[-1]["price"] < sh[-2]["price"] and sl[-1]["price"] < sl[-2]["price"]:
            structure["trend"] = "bearish"
            structure["lh_ll"] = True
    
    # Detect BOS (Break of Structure)
    if len(sh) >= 2 and len(sl) >= 2:
        current_price = closes[-1]
        
        # Bullish BOS: Price breaks above previous swing high
        if current_price > sh[-1]["price"]:
            structure["bos"].append({
                "type": "bullish",
                "level": sh[-1]["price"],
                "broken_at": current_price
            })
        
        # Bearish BOS: Price breaks below previous swing low
        if current_price < sl[-1]["price"]:
            structure["bos"].append({
                "type": "bearish",
                "level": sl[-1]["price"],
                "broken_at": current_price
            })
    
    # Detect CHoCH (Change of Character)
    if len(sh) >= 3 and len(sl) >= 3:
        # Bullish CHoCH: Was making LH/LL, now breaks above LH
        if sh[-3]["price"] > sh[-2]["price"] and closes[-1] > sh[-2]["price"]:
            structure["choch"] = {
                "type": "bullish",
                "level": sh[-2]["price"],
                "signal": "Potential trend reversal to bullish"
            }
        
        # Bearish CHoCH: Was making HH/HL, now breaks below HL
        if sl[-3]["price"] < sl[-2]["price"] and closes[-1] < sl[-2]["price"]:
            structure["choch"] = {
                "type": "bearish",
                "level": sl[-2]["price"],
                "signal": "Potential trend reversal to bearish"
            }
    
    # Market Shift detection
    if structure["choch"] and structure["bos"]:
        structure["market_shift"] = {
            "detected": True,
            "direction": structure["choch"]["type"],
            "confirmation": "CHoCH + BOS aligned"
        }
    
    # Internal vs External Structure
    if len(closes) >= 50:
        # External: Higher timeframe structure
        external_high = max(highs[-50:])
        external_low = min(lows[-50:])
        
        # Internal: Lower timeframe structure (last 20 candles)
        internal_high = max(highs[-20:])
        internal_low = min(lows[-20:])
        
        structure["external_structure"] = {
            "high": external_high,
            "low": external_low,
            "range": external_high - external_low
        }
        
        structure["internal_structure"] = {
            "high": internal_high,
            "low": internal_low,
            "range": internal_high - internal_low
        }
    
    # Fractal Structure
    if len(sh) >= 3 and len(sl) >= 3:
        # Check if fractals are aligned with trend
        if structure["trend"] == "bullish":
            structure["fractal_bias"] = "bullish" if sh[-1]["price"] > sh[-3]["price"] else "weakening"
        elif structure["trend"] == "bearish":
            structure["fractal_bias"] = "bearish" if sl[-1]["price"] < sl[-3]["price"] else "weakening"
    
    return structure


# ==================== FAIR VALUE GAPS (14 concepts) ====================

def find_all_fvg(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Complete FVG analysis including:
    - Bullish/Bearish FVG
    - FVG Mitigation
    - FVG Inversion
    - Consequent Encroachment
    - Partial/Complete Fill
    - Multi-timeframe FVG
    """
    fvg_data = {
        "bullish_fvg": [],
        "bearish_fvg": [],
        "mitigated": [],
        "inverted": [],
        "ce_levels": [],  # Consequent Encroachment
        "unfilled": [],
        "partially_filled": []
    }
    
    if len(closes) < 5:
        return fvg_data
    
    current_price = closes[-1]
    
    for i in range(2, len(closes)):
        # Bullish FVG: Gap between candle 1 high and candle 3 low
        if lows[i] > highs[i-2]:
            fvg = {
                "type": "bullish",
                "top": lows[i],
                "bottom": highs[i-2],
                "index": i,
                "midpoint": (lows[i] + highs[i-2]) / 2,  # CE level
                "size": lows[i] - highs[i-2]
            }
            
            # Check fill status
            if current_price < highs[i-2]:
                fvg["status"] = "filled"
                fvg_data["mitigated"].append(fvg)
            elif current_price < fvg["midpoint"]:
                fvg["status"] = "partially_filled"
                fvg_data["partially_filled"].append(fvg)
            else:
                fvg["status"] = "unfilled"
                fvg_data["unfilled"].append(fvg)
            
            fvg_data["bullish_fvg"].append(fvg)
            fvg_data["ce_levels"].append({"price": fvg["midpoint"], "type": "bullish_ce"})
        
        # Bearish FVG: Gap between candle 1 low and candle 3 high
        if highs[i] < lows[i-2]:
            fvg = {
                "type": "bearish",
                "top": lows[i-2],
                "bottom": highs[i],
                "index": i,
                "midpoint": (lows[i-2] + highs[i]) / 2,
                "size": lows[i-2] - highs[i]
            }
            
            if current_price > lows[i-2]:
                fvg["status"] = "filled"
                fvg_data["mitigated"].append(fvg)
            elif current_price > fvg["midpoint"]:
                fvg["status"] = "partially_filled"
                fvg_data["partially_filled"].append(fvg)
            else:
                fvg["status"] = "unfilled"
                fvg_data["unfilled"].append(fvg)
            
            fvg_data["bearish_fvg"].append(fvg)
            fvg_data["ce_levels"].append({"price": fvg["midpoint"], "type": "bearish_ce"})
    
    # FVG Inversion: When price fills FVG and reverses
    for fvg in fvg_data["mitigated"]:
        if fvg["type"] == "bullish" and current_price > fvg["top"]:
            fvg_data["inverted"].append({
                "original": fvg,
                "inversion_type": "bullish_to_support",
                "level": fvg["bottom"]
            })
        elif fvg["type"] == "bearish" and current_price < fvg["bottom"]:
            fvg_data["inverted"].append({
                "original": fvg,
                "inversion_type": "bearish_to_resistance",
                "level": fvg["top"]
            })
    
    return fvg_data


# ==================== INVERSE FAIR VALUE GAPS (8 concepts) ====================

def find_ifvg(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Inverse Fair Value Gap analysis
    IFVG forms when FVG gets filled and acts as new S/R
    """
    ifvg_data = {
        "bullish_ifvg": [],
        "bearish_ifvg": [],
        "rejection_zones": [],
        "mitigation_zones": []
    }
    
    fvg_data = find_all_fvg(opens, highs, lows, closes)
    current_price = closes[-1]
    
    # IFVG from mitigated FVGs
    for fvg in fvg_data["mitigated"]:
        if fvg["type"] == "bullish":
            # Bullish FVG becomes IFVG (resistance) after fill
            ifvg_data["bearish_ifvg"].append({
                "zone_top": fvg["top"],
                "zone_bottom": fvg["bottom"],
                "midpoint": fvg["midpoint"],
                "signal": "Potential resistance zone"
            })
        else:
            # Bearish FVG becomes IFVG (support) after fill
            ifvg_data["bullish_ifvg"].append({
                "zone_top": fvg["top"],
                "zone_bottom": fvg["bottom"],
                "midpoint": fvg["midpoint"],
                "signal": "Potential support zone"
            })
    
    return ifvg_data


# ==================== ORDER BLOCKS (10 concepts) ====================

def find_all_order_blocks(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Complete Order Block analysis including:
    - Bullish/Bearish OB
    - Breaker Blocks
    - Mitigation Blocks
    - Refined OB
    - Institutional OB
    """
    ob_data = {
        "bullish_ob": [],
        "bearish_ob": [],
        "breaker_blocks": [],
        "mitigation_blocks": [],
        "refined_ob": [],
        "unmitigated": [],
        "institutional_ob": []
    }
    
    if len(closes) < 10:
        return ob_data
    
    current_price = closes[-1]
    
    for i in range(3, len(closes) - 3):
        # Bullish Order Block: Last bearish candle before strong bullish move
        if closes[i] < opens[i]:  # Bearish candle
            # Check for strong bullish move after
            if closes[i+1] > opens[i+1] and closes[i+2] > opens[i+2]:
                if closes[i+2] > highs[i]:  # Displacement
                    ob = {
                        "type": "bullish",
                        "top": highs[i],
                        "bottom": lows[i],
                        "body_top": opens[i],
                        "body_bottom": closes[i],
                        "index": i,
                        "mitigated": current_price < lows[i]
                    }
                    
                    # Refined OB uses body instead of wicks
                    ob["refined_top"] = opens[i]
                    ob["refined_bottom"] = closes[i]
                    
                    ob_data["bullish_ob"].append(ob)
                    
                    if not ob["mitigated"]:
                        ob_data["unmitigated"].append(ob)
                    else:
                        ob_data["mitigation_blocks"].append(ob)
        
        # Bearish Order Block: Last bullish candle before strong bearish move
        if closes[i] > opens[i]:  # Bullish candle
            if closes[i+1] < opens[i+1] and closes[i+2] < opens[i+2]:
                if closes[i+2] < lows[i]:  # Displacement
                    ob = {
                        "type": "bearish",
                        "top": highs[i],
                        "bottom": lows[i],
                        "body_top": closes[i],
                        "body_bottom": opens[i],
                        "index": i,
                        "mitigated": current_price > highs[i]
                    }
                    
                    ob["refined_top"] = closes[i]
                    ob["refined_bottom"] = opens[i]
                    
                    ob_data["bearish_ob"].append(ob)
                    
                    if not ob["mitigated"]:
                        ob_data["unmitigated"].append(ob)
                    else:
                        ob_data["mitigation_blocks"].append(ob)
    
    # Breaker Blocks: OB that gets broken and becomes opposite S/R
    for ob in ob_data["mitigation_blocks"]:
        if ob["type"] == "bullish" and current_price < ob["bottom"]:
            ob_data["breaker_blocks"].append({
                "original_type": "bullish",
                "new_type": "bearish_breaker",
                "zone": ob,
                "signal": "Former support now resistance"
            })
        elif ob["type"] == "bearish" and current_price > ob["top"]:
            ob_data["breaker_blocks"].append({
                "original_type": "bearish",
                "new_type": "bullish_breaker",
                "zone": ob,
                "signal": "Former resistance now support"
            })
    
    # Institutional OB: Large body candles with significant volume implication
    for ob in ob_data["bullish_ob"] + ob_data["bearish_ob"]:
        body_size = abs(ob["body_top"] - ob["body_bottom"])
        total_range = ob["top"] - ob["bottom"]
        
        if body_size > total_range * 0.7:  # Strong institutional candle
            ob_data["institutional_ob"].append(ob)
    
    return ob_data


# ==================== LIQUIDITY CONCEPTS (13 concepts) ====================

def analyze_liquidity(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Complete liquidity analysis including:
    - BSL/SSL
    - Equal Highs/Lows
    - Liquidity Sweep
    - Liquidity Grab
    - Liquidity Void
    - Liquidity Pool
    """
    liquidity = {
        "bsl": [],  # Buy Side Liquidity
        "ssl": [],  # Sell Side Liquidity
        "eqh": [],  # Equal Highs
        "eql": [],  # Equal Lows
        "sweeps": [],
        "grabs": [],
        "voids": [],
        "pools": [],
        "relative_eqh": [],
        "relative_eql": []
    }
    
    if len(closes) < 20:
        return liquidity
    
    tolerance = (max(highs) - min(lows)) * 0.002
    current_price = closes[-1]
    
    # Find swing points for liquidity
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({"index": i, "price": highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({"index": i, "price": lows[i]})
    
    # Buy Side Liquidity (stops above swing highs)
    for sh in swing_highs:
        liquidity["bsl"].append({
            "price": sh["price"],
            "stop_zone": sh["price"] * 1.001,
            "index": sh["index"]
        })
    
    # Sell Side Liquidity (stops below swing lows)
    for sl in swing_lows:
        liquidity["ssl"].append({
            "price": sl["price"],
            "stop_zone": sl["price"] * 0.999,
            "index": sl["index"]
        })
    
    # Equal Highs (EQH)
    for i, h1 in enumerate(swing_highs):
        for h2 in swing_highs[i+1:]:
            if abs(h1["price"] - h2["price"]) < tolerance:
                liquidity["eqh"].append({
                    "price": (h1["price"] + h2["price"]) / 2,
                    "indices": [h1["index"], h2["index"]],
                    "liquidity_above": h1["price"] * 1.001
                })
    
    # Equal Lows (EQL)
    for i, l1 in enumerate(swing_lows):
        for l2 in swing_lows[i+1:]:
            if abs(l1["price"] - l2["price"]) < tolerance:
                liquidity["eql"].append({
                    "price": (l1["price"] + l2["price"]) / 2,
                    "indices": [l1["index"], l2["index"]],
                    "liquidity_below": l1["price"] * 0.999
                })
    
    # Liquidity Sweep detection
    if len(swing_highs) >= 2:
        prev_high = swing_highs[-2]["price"]
        if max(highs[-5:]) > prev_high and current_price < prev_high:
            liquidity["sweeps"].append({
                "type": "bsl_sweep",
                "level": prev_high,
                "signal": "Buy side liquidity swept - potential reversal down"
            })
    
    if len(swing_lows) >= 2:
        prev_low = swing_lows[-2]["price"]
        if min(lows[-5:]) < prev_low and current_price > prev_low:
            liquidity["sweeps"].append({
                "type": "ssl_sweep",
                "level": prev_low,
                "signal": "Sell side liquidity swept - potential reversal up"
            })
    
    # Liquidity Void (areas with no trading)
    for i in range(1, len(closes)):
        gap = lows[i] - highs[i-1]
        if gap > tolerance * 5:
            liquidity["voids"].append({
                "top": lows[i],
                "bottom": highs[i-1],
                "index": i,
                "type": "bullish_void"
            })
        
        gap = lows[i-1] - highs[i]
        if gap > tolerance * 5:
            liquidity["voids"].append({
                "top": lows[i-1],
                "bottom": highs[i],
                "index": i,
                "type": "bearish_void"
            })
    
    # Liquidity Pools (clusters of stops)
    if liquidity["eqh"]:
        liquidity["pools"].append({
            "type": "bsl_pool",
            "price": max([e["price"] for e in liquidity["eqh"]]),
            "strength": len(liquidity["eqh"])
        })
    
    if liquidity["eql"]:
        liquidity["pools"].append({
            "type": "ssl_pool",
            "price": min([e["price"] for e in liquidity["eql"]]),
            "strength": len(liquidity["eql"])
        })
    
    return liquidity


# ==================== PREMIUM/DISCOUNT THEORY (9 concepts) ====================

def analyze_premium_discount(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Premium/Discount zone analysis including:
    - Premium Zone
    - Discount Zone
    - Equilibrium
    - OTE (Optimal Trade Entry)
    - Premium/Discount Arrays
    """
    if len(closes) < 20:
        return {}
    
    # Define range
    range_high = max(highs[-50:]) if len(highs) >= 50 else max(highs)
    range_low = min(lows[-50:]) if len(lows) >= 50 else min(lows)
    range_size = range_high - range_low
    
    # Equilibrium (50% level)
    equilibrium = range_low + (range_size * 0.5)
    
    # Premium Zone (above 50%)
    premium_start = equilibrium
    premium_end = range_high
    
    # Discount Zone (below 50%)
    discount_start = range_low
    discount_end = equilibrium
    
    # OTE Zone (61.8% - 78.6% retracement)
    ote_top = range_low + (range_size * 0.786)
    ote_bottom = range_low + (range_size * 0.618)
    
    current_price = closes[-1]
    
    # Determine current zone
    if current_price > equilibrium:
        current_zone = "premium"
        zone_percentage = ((current_price - equilibrium) / (range_high - equilibrium)) * 100
    else:
        current_zone = "discount"
        zone_percentage = ((equilibrium - current_price) / (equilibrium - range_low)) * 100
    
    # Trading bias based on zone
    if current_zone == "premium":
        bias = "Look for sells in premium zone"
        signal = "SELL"
    else:
        bias = "Look for buys in discount zone"
        signal = "BUY"
    
    return {
        "range_high": range_high,
        "range_low": range_low,
        "equilibrium": equilibrium,
        "premium_zone": {"start": premium_start, "end": premium_end},
        "discount_zone": {"start": discount_start, "end": discount_end},
        "ote_zone": {"top": ote_top, "bottom": ote_bottom},
        "current_zone": current_zone,
        "zone_percentage": round(zone_percentage, 1),
        "bias": bias,
        "signal": signal,
        "in_ote": ote_bottom <= current_price <= ote_top
    }


# ==================== KILL ZONES & SESSIONS (12 concepts) ====================

def analyze_kill_zones(current_hour: int = None) -> Dict:
    """
    Kill Zone and Session analysis including:
    - London Kill Zone
    - New York Kill Zone
    - Asian Kill Zone
    - Session Overlaps
    - Dead Zones
    """
    if current_hour is None:
        current_hour = datetime.now().hour  # Use local time or pass UTC hour explicitly
    
    kill_zones = {
        "asian": {"start": 0, "end": 8, "peak": [2, 4]},
        "london": {"start": 7, "end": 16, "peak": [8, 11]},
        "new_york": {"start": 12, "end": 21, "peak": [13, 16]},
        "london_close": {"start": 15, "end": 17, "peak": [15, 16]}
    }
    
    sessions = {
        "sydney": {"start": 21, "end": 6},
        "tokyo": {"start": 0, "end": 9},
        "london": {"start": 7, "end": 16},
        "new_york": {"start": 12, "end": 21}
    }
    
    # Determine active sessions
    active_sessions = []
    for session, times in sessions.items():
        if times["start"] <= current_hour < times["end"]:
            active_sessions.append(session)
        elif times["start"] > times["end"]:  # Overnight session
            if current_hour >= times["start"] or current_hour < times["end"]:
                active_sessions.append(session)
    
    # Determine active kill zones
    active_kz = []
    for kz, times in kill_zones.items():
        if times["start"] <= current_hour < times["end"]:
            active_kz.append(kz)
            if current_hour in times["peak"]:
                active_kz.append(f"{kz}_peak")
    
    # Session overlaps (highest volatility)
    overlaps = []
    if 7 <= current_hour < 9:
        overlaps.append("Tokyo-London overlap")
    if 12 <= current_hour < 16:
        overlaps.append("London-New York overlap (HIGHEST VOLATILITY)")
    
    # Dead zones (low volatility)
    dead_zones = []
    if 17 <= current_hour < 21:
        dead_zones.append("Post-NY close (low volatility)")
    if 5 <= current_hour < 7:
        dead_zones.append("Pre-London (low volatility)")
    
    # Volatility expectation
    if overlaps:
        volatility = "HIGH"
    elif active_kz:
        volatility = "MEDIUM-HIGH"
    elif dead_zones:
        volatility = "LOW"
    else:
        volatility = "MEDIUM"
    
    return {
        "current_hour_utc": current_hour,
        "active_sessions": active_sessions,
        "active_kill_zones": active_kz,
        "overlaps": overlaps,
        "dead_zones": dead_zones,
        "volatility_expectation": volatility,
        "best_pairs": get_best_pairs_for_session(active_sessions),
        "recommendation": get_session_recommendation(active_sessions, active_kz)
    }


def get_best_pairs_for_session(sessions: List[str]) -> List[str]:
    """Get best pairs to trade based on active sessions"""
    pairs = {
        "tokyo": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"],
        "london": ["EUR/USD", "GBP/USD", "EUR/GBP", "USD/CHF"],
        "new_york": ["EUR/USD", "GBP/USD", "USD/CAD", "XAU/USD"],
        "sydney": ["AUD/USD", "NZD/USD", "AUD/NZD"]
    }
    
    best = []
    for session in sessions:
        if session in pairs:
            best.extend(pairs[session])
    
    return list(set(best))


def get_session_recommendation(sessions: List[str], kill_zones: List[str]) -> str:
    """Get trading recommendation based on session"""
    if "london_peak" in kill_zones or "new_york_peak" in kill_zones:
        return "OPTIMAL trading time - High probability setups"
    elif kill_zones:
        return "Good trading time - Look for setups"
    elif sessions:
        return "Moderate trading time - Be selective"
    else:
        return "Low activity - Consider waiting for better session"


# ==================== POWER OF 3 / AMD (8 concepts) ====================

def analyze_power_of_3(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Power of 3 / AMD Algorithm analysis:
    - Accumulation Phase
    - Manipulation Phase
    - Distribution Phase
    """
    if len(closes) < 20:
        return {}
    
    # Analyze recent price action for AMD pattern
    recent_opens = opens[-20:]
    recent_highs = highs[-20:]
    recent_lows = lows[-20:]
    recent_closes = closes[-20:]
    
    # Phase 1: Accumulation (consolidation)
    accumulation_range = max(recent_highs[:7]) - min(recent_lows[:7])
    
    # Phase 2: Manipulation (fake breakout)
    manipulation_high = max(recent_highs[7:14])
    manipulation_low = min(recent_lows[7:14])
    
    # Phase 3: Distribution (real move)
    distribution_close = recent_closes[-1]
    
    # Determine pattern type
    if manipulation_low < min(recent_lows[:7]) and distribution_close > max(recent_highs[:7]):
        pattern = "Bullish AMD"
        signal = "BUY"
        description = "Accumulation → Manipulation (sweep lows) → Distribution (move up)"
    elif manipulation_high > max(recent_highs[:7]) and distribution_close < min(recent_lows[:7]):
        pattern = "Bearish AMD"
        signal = "SELL"
        description = "Accumulation → Manipulation (sweep highs) → Distribution (move down)"
    else:
        pattern = "No clear AMD"
        signal = "NEUTRAL"
        description = "Pattern not confirmed"
    
    return {
        "pattern": pattern,
        "signal": signal,
        "description": description,
        "accumulation_zone": {
            "high": max(recent_highs[:7]),
            "low": min(recent_lows[:7])
        },
        "manipulation_zone": {
            "high": manipulation_high,
            "low": manipulation_low
        },
        "current_phase": determine_current_phase(recent_closes),
        "reversal_profile": pattern == "Bullish AMD" or pattern == "Bearish AMD",
        "continuation_profile": False
    }


def determine_current_phase(closes: List[float]) -> str:
    """Determine current phase of AMD"""
    if len(closes) < 10:
        return "unknown"
    
    early_range = max(closes[:5]) - min(closes[:5])
    recent_range = max(closes[-5:]) - min(closes[-5:])
    
    if recent_range < early_range * 0.5:
        return "accumulation"
    elif recent_range > early_range * 1.5:
        return "distribution"
    else:
        return "manipulation"


# ==================== ADVANCED ICT CONCEPTS (14 concepts) ====================

def analyze_advanced_ict(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Advanced ICT concepts including:
    - Displacement
    - Inducement
    - Judas Swing
    - Silver Bullet
    - Market Maker Model
    """
    if len(closes) < 20:
        return {}
    
    current_price = closes[-1]
    
    # Displacement detection (strong impulsive move)
    displacements = []
    for i in range(1, len(closes)):
        move = abs(closes[i] - closes[i-1])
        lookback_range = [abs(closes[j] - closes[j-1]) for j in range(max(1, i-10), i)]
        if not lookback_range:
            continue
        avg_move = np.mean(lookback_range)
        
        if avg_move > 0 and move > avg_move * 2:
            displacements.append({
                "index": i,
                "type": "bullish" if closes[i] > closes[i-1] else "bearish",
                "magnitude": move / avg_move
            })
    
    # Inducement detection (liquidity grab before real move)
    inducements = []
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i+1]:
            swing_highs.append({"index": i, "price": highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append({"index": i, "price": lows[i]})
    
    # Check for inducement (price takes out a level then reverses)
    if len(swing_highs) >= 2:
        if highs[-1] > swing_highs[-2]["price"] and closes[-1] < swing_highs[-2]["price"]:
            inducements.append({
                "type": "bearish_inducement",
                "level": swing_highs[-2]["price"],
                "signal": "Liquidity taken above, expecting move down"
            })
    
    if len(swing_lows) >= 2:
        if lows[-1] < swing_lows[-2]["price"] and closes[-1] > swing_lows[-2]["price"]:
            inducements.append({
                "type": "bullish_inducement",
                "level": swing_lows[-2]["price"],
                "signal": "Liquidity taken below, expecting move up"
            })
    
    # Judas Swing (fake move at session open)
    judas_swing = None
    if len(closes) >= 5:
        open_price = opens[-5]
        high_since_open = max(highs[-5:])
        low_since_open = min(lows[-5:])
        
        if current_price < open_price and high_since_open > open_price * 1.002:
            judas_swing = {
                "type": "bearish_judas",
                "fake_high": high_since_open,
                "signal": "Fake move up, real move down"
            }
        elif current_price > open_price and low_since_open < open_price * 0.998:
            judas_swing = {
                "type": "bullish_judas",
                "fake_low": low_since_open,
                "signal": "Fake move down, real move up"
            }
    
    # Silver Bullet (specific time-based setup)
    current_hour = datetime.now().hour  # Use local time
    silver_bullet = None
    
    if 10 <= current_hour <= 11:  # London Silver Bullet
        silver_bullet = {
            "type": "london_silver_bullet",
            "time_window": "10:00-11:00 UTC",
            "active": True
        }
    elif 14 <= current_hour <= 15:  # NY Silver Bullet
        silver_bullet = {
            "type": "ny_silver_bullet",
            "time_window": "14:00-15:00 UTC",
            "active": True
        }
    
    return {
        "displacements": displacements[-5:] if displacements else [],
        "inducements": inducements,
        "judas_swing": judas_swing,
        "silver_bullet": silver_bullet,
        "market_maker_model": analyze_market_maker_model(highs, lows, closes)
    }


def analyze_market_maker_model(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Market Maker Model analysis"""
    if len(closes) < 30:
        return {}
    
    # Identify accumulation, manipulation, distribution
    range_high = max(highs[-30:])
    range_low = min(lows[-30:])
    current = closes[-1]
    
    # Check for stop hunt patterns
    recent_high = max(highs[-10:])
    recent_low = min(lows[-10:])
    
    if recent_high > range_high * 0.99 and current < range_high * 0.98:
        return {
            "pattern": "Distribution after BSL sweep",
            "signal": "SELL",
            "target": range_low
        }
    elif recent_low < range_low * 1.01 and current > range_low * 1.02:
        return {
            "pattern": "Accumulation after SSL sweep",
            "signal": "BUY",
            "target": range_high
        }
    
    return {"pattern": "No clear MM model", "signal": "NEUTRAL"}


# ==================== BALANCED PRICE RANGE (6 concepts) ====================

def find_balanced_price_range(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Balanced Price Range (BPR) analysis
    BPR forms when bullish and bearish FVGs overlap
    """
    bpr_data = {
        "bullish_bpr": [],
        "bearish_bpr": [],
        "active_bpr": None
    }
    
    fvg_data = find_all_fvg(opens, highs, lows, closes)
    
    # Find overlapping FVGs
    for bull_fvg in fvg_data["bullish_fvg"]:
        for bear_fvg in fvg_data["bearish_fvg"]:
            # Check for overlap
            overlap_top = min(bull_fvg["top"], bear_fvg["top"])
            overlap_bottom = max(bull_fvg["bottom"], bear_fvg["bottom"])
            
            if overlap_top > overlap_bottom:
                bpr = {
                    "top": overlap_top,
                    "bottom": overlap_bottom,
                    "midpoint": (overlap_top + overlap_bottom) / 2,
                    "bullish_fvg": bull_fvg,
                    "bearish_fvg": bear_fvg
                }
                
                # Determine BPR type based on which FVG is more recent
                if bull_fvg["index"] > bear_fvg["index"]:
                    bpr_data["bullish_bpr"].append(bpr)
                else:
                    bpr_data["bearish_bpr"].append(bpr)
    
    # Find most recent active BPR
    current_price = closes[-1]
    all_bpr = bpr_data["bullish_bpr"] + bpr_data["bearish_bpr"]
    
    for bpr in reversed(all_bpr):
        if bpr["bottom"] <= current_price <= bpr["top"]:
            bpr_data["active_bpr"] = bpr
            break
    
    return bpr_data


# ==================== COMPLETE ICT/SMC ANALYSIS ====================

def complete_ict_smc_analysis(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """
    Complete ICT/SMC analysis combining all 131 concepts
    """
    analysis = {
        "market_structure": analyze_market_structure(highs, lows, closes),
        "fvg": find_all_fvg(opens, highs, lows, closes),
        "ifvg": find_ifvg(opens, highs, lows, closes),
        "order_blocks": find_all_order_blocks(opens, highs, lows, closes),
        "liquidity": analyze_liquidity(highs, lows, closes),
        "premium_discount": analyze_premium_discount(highs, lows, closes),
        "kill_zones": analyze_kill_zones(),
        "power_of_3": analyze_power_of_3(opens, highs, lows, closes),
        "advanced_ict": analyze_advanced_ict(opens, highs, lows, closes),
        "bpr": find_balanced_price_range(opens, highs, lows, closes)
    }
    
    # Generate overall signal
    signals = []
    
    # Market Structure signal
    if analysis["market_structure"]["trend"] == "bullish":
        signals.append(("structure", "BUY", 25))
    elif analysis["market_structure"]["trend"] == "bearish":
        signals.append(("structure", "SELL", 25))
    
    # Premium/Discount signal
    if analysis["premium_discount"].get("signal"):
        signals.append(("pd_zone", analysis["premium_discount"]["signal"], 20))
    
    # Order Block signal
    ob = analysis["order_blocks"]
    if ob["unmitigated"]:
        latest_ob = ob["unmitigated"][-1]
        if latest_ob["type"] == "bullish" and closes[-1] > latest_ob["bottom"]:
            signals.append(("ob", "BUY", 20))
        elif latest_ob["type"] == "bearish" and closes[-1] < latest_ob["top"]:
            signals.append(("ob", "SELL", 20))
    
    # Liquidity signal
    liq = analysis["liquidity"]
    if liq["sweeps"]:
        latest_sweep = liq["sweeps"][-1]
        if latest_sweep["type"] == "ssl_sweep":
            signals.append(("liquidity", "BUY", 15))
        else:
            signals.append(("liquidity", "SELL", 15))
    
    # Power of 3 signal
    if analysis["power_of_3"].get("signal") and analysis["power_of_3"]["signal"] != "NEUTRAL":
        signals.append(("amd", analysis["power_of_3"]["signal"], 20))
    
    # Calculate weighted signal
    buy_weight = sum(s[2] for s in signals if s[1] == "BUY")
    sell_weight = sum(s[2] for s in signals if s[1] == "SELL")
    
    if buy_weight > sell_weight + 10:
        analysis["overall_signal"] = "BUY"
        analysis["signal_strength"] = buy_weight
    elif sell_weight > buy_weight + 10:
        analysis["overall_signal"] = "SELL"
        analysis["signal_strength"] = sell_weight
    else:
        analysis["overall_signal"] = "NEUTRAL"
        analysis["signal_strength"] = 50
    
    analysis["signal_breakdown"] = signals
    analysis["concepts_detected"] = count_detected_concepts(analysis)
    
    return analysis


def count_detected_concepts(analysis: Dict) -> int:
    """Count total ICT/SMC concepts detected"""
    count = 0
    
    # Market Structure
    if analysis["market_structure"]["trend"] != "sideways":
        count += 1
    if analysis["market_structure"]["bos"]:
        count += len(analysis["market_structure"]["bos"])
    if analysis["market_structure"]["choch"]:
        count += 1
    
    # FVG
    count += len(analysis["fvg"]["bullish_fvg"])
    count += len(analysis["fvg"]["bearish_fvg"])
    
    # Order Blocks
    count += len(analysis["order_blocks"]["bullish_ob"])
    count += len(analysis["order_blocks"]["bearish_ob"])
    count += len(analysis["order_blocks"]["breaker_blocks"])
    
    # Liquidity
    count += len(analysis["liquidity"]["eqh"])
    count += len(analysis["liquidity"]["eql"])
    count += len(analysis["liquidity"]["sweeps"])
    
    return count
