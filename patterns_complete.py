"""
COMPLETE CANDLESTICK PATTERNS - 64 Patterns
All single, double, triple, and advanced candlestick patterns
"""

import numpy as np

def identify_all_candlestick_patterns(opens, highs, lows, closes):
    """
    Identify ALL 64 candlestick patterns
    Returns list of detected patterns with type and confidence
    """
    patterns = []
    if len(closes) < 5:
        return patterns
    
    # Current candle metrics
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body = abs(c - o)
    total_range = h - l if h != l else 0.0001
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_bullish = c > o
    is_bearish = c < o
    body_pct = body / total_range if total_range > 0 else 0
    
    # Previous candles
    o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2]
    body1 = abs(c1 - o1)
    range1 = h1 - l1 if h1 != l1 else 0.0001
    
    o2, h2, l2, c2 = opens[-3], highs[-3], lows[-3], closes[-3] if len(closes) >= 3 else (o, h, l, c)
    body2 = abs(c2 - o2)
    
    o3, h3, l3, c3 = opens[-4], highs[-4], lows[-4], closes[-4] if len(closes) >= 4 else (o, h, l, c)
    o4, h4, l4, c4 = opens[-5], highs[-5], lows[-5], closes[-5] if len(closes) >= 5 else (o, h, l, c)
    
    # ==================== SINGLE CANDLE PATTERNS (14) ====================
    
    # 1. Doji
    if body < total_range * 0.1:
        patterns.append({"name": "Doji", "type": "neutral", "confidence": 75, "category": "single"})
    
    # 2. Long Legged Doji
    if body < total_range * 0.1 and upper_wick > total_range * 0.3 and lower_wick > total_range * 0.3:
        patterns.append({"name": "Long Legged Doji", "type": "neutral", "confidence": 80, "category": "single"})
    
    # 3. Dragonfly Doji
    if body < total_range * 0.1 and upper_wick < total_range * 0.1 and lower_wick > total_range * 0.6:
        patterns.append({"name": "Dragonfly Doji", "type": "bullish", "confidence": 85, "category": "single"})
    
    # 4. Gravestone Doji
    if body < total_range * 0.1 and lower_wick < total_range * 0.1 and upper_wick > total_range * 0.6:
        patterns.append({"name": "Gravestone Doji", "type": "bearish", "confidence": 85, "category": "single"})
    
    # 5. Four Price Doji
    if body < total_range * 0.02 and upper_wick < total_range * 0.02 and lower_wick < total_range * 0.02:
        patterns.append({"name": "Four Price Doji", "type": "neutral", "confidence": 90, "category": "single"})
    
    # 6. Hammer
    if lower_wick > body * 2 and upper_wick < body * 0.3 and body > total_range * 0.1:
        patterns.append({"name": "Hammer", "type": "bullish", "confidence": 82, "category": "single"})
    
    # 7. Inverted Hammer
    if upper_wick > body * 2 and lower_wick < body * 0.3 and is_bullish:
        patterns.append({"name": "Inverted Hammer", "type": "bullish", "confidence": 78, "category": "single"})
    
    # 8. Shooting Star
    if upper_wick > body * 2 and lower_wick < body * 0.3 and is_bearish:
        patterns.append({"name": "Shooting Star", "type": "bearish", "confidence": 82, "category": "single"})
    
    # 9. Hanging Man
    if lower_wick > body * 2 and upper_wick < body * 0.3 and is_bearish:
        patterns.append({"name": "Hanging Man", "type": "bearish", "confidence": 78, "category": "single"})
    
    # 10. Bullish Marubozu
    if upper_wick < body * 0.05 and lower_wick < body * 0.05 and body > total_range * 0.8 and is_bullish:
        patterns.append({"name": "Bullish Marubozu", "type": "bullish", "confidence": 88, "category": "single"})
    
    # 11. Bearish Marubozu
    if upper_wick < body * 0.05 and lower_wick < body * 0.05 and body > total_range * 0.8 and is_bearish:
        patterns.append({"name": "Bearish Marubozu", "type": "bearish", "confidence": 88, "category": "single"})
    
    # 12. Bullish Belt Hold
    if is_bullish and lower_wick < body * 0.1 and body > total_range * 0.6:
        patterns.append({"name": "Bullish Belt Hold", "type": "bullish", "confidence": 75, "category": "single"})
    
    # 13. Bearish Belt Hold
    if is_bearish and upper_wick < body * 0.1 and body > total_range * 0.6:
        patterns.append({"name": "Bearish Belt Hold", "type": "bearish", "confidence": 75, "category": "single"})
    
    # 14. Spinning Top
    if body < total_range * 0.3 and upper_wick > body and lower_wick > body:
        patterns.append({"name": "Spinning Top", "type": "neutral", "confidence": 70, "category": "single"})
    
    # ==================== TWO CANDLE PATTERNS (10) ====================
    
    if len(closes) >= 2:
        # 15. Bullish Engulfing
        if is_bullish and c1 < o1 and c > o1 and o < c1 and body > body1:
            patterns.append({"name": "Bullish Engulfing", "type": "bullish", "confidence": 85, "category": "double"})
        
        # 16. Bearish Engulfing
        if is_bearish and c1 > o1 and c < o1 and o > c1 and body > body1:
            patterns.append({"name": "Bearish Engulfing", "type": "bearish", "confidence": 85, "category": "double"})
        
        # 17. Piercing Line
        if c1 < o1 and is_bullish and o < l1 and c > (o1 + c1) / 2 and c < o1:
            patterns.append({"name": "Piercing Line", "type": "bullish", "confidence": 80, "category": "double"})
        
        # 18. Dark Cloud Cover
        if c1 > o1 and is_bearish and o > h1 and c < (o1 + c1) / 2 and c > o1:
            patterns.append({"name": "Dark Cloud Cover", "type": "bearish", "confidence": 80, "category": "double"})
        
        # 19. Bullish Harami
        if c1 < o1 and is_bullish and o > c1 and c < o1 and body < body1 * 0.5:
            patterns.append({"name": "Bullish Harami", "type": "bullish", "confidence": 75, "category": "double"})
        
        # 20. Bearish Harami
        if c1 > o1 and is_bearish and o < c1 and c > o1 and body < body1 * 0.5:
            patterns.append({"name": "Bearish Harami", "type": "bearish", "confidence": 75, "category": "double"})
        
        # 21. Tweezer Bottom
        if abs(l - l1) < total_range * 0.05 and c1 < o1 and is_bullish:
            patterns.append({"name": "Tweezer Bottom", "type": "bullish", "confidence": 78, "category": "double"})
        
        # 22. Tweezer Top
        if abs(h - h1) < total_range * 0.05 and c1 > o1 and is_bearish:
            patterns.append({"name": "Tweezer Top", "type": "bearish", "confidence": 78, "category": "double"})
        
        # 23. Bullish Kicker
        if c1 < o1 and is_bullish and o > o1 and c > h1:
            patterns.append({"name": "Bullish Kicker", "type": "bullish", "confidence": 90, "category": "double"})
        
        # 24. Bearish Kicker
        if c1 > o1 and is_bearish and o < o1 and c < l1:
            patterns.append({"name": "Bearish Kicker", "type": "bearish", "confidence": 90, "category": "double"})

    # ==================== THREE CANDLE PATTERNS (16) ====================
    
    if len(closes) >= 3:
        # 25. Morning Star
        if c2 < o2 and body1 < body2 * 0.3 and is_bullish and c > (o2 + c2) / 2:
            patterns.append({"name": "Morning Star", "type": "bullish", "confidence": 88, "category": "triple"})
        
        # 26. Evening Star
        if c2 > o2 and body1 < body2 * 0.3 and is_bearish and c < (o2 + c2) / 2:
            patterns.append({"name": "Evening Star", "type": "bearish", "confidence": 88, "category": "triple"})
        
        # 27. Morning Doji Star
        if c2 < o2 and body1 < range1 * 0.1 and is_bullish and c > (o2 + c2) / 2:
            patterns.append({"name": "Morning Doji Star", "type": "bullish", "confidence": 90, "category": "triple"})
        
        # 28. Evening Doji Star
        if c2 > o2 and body1 < range1 * 0.1 and is_bearish and c < (o2 + c2) / 2:
            patterns.append({"name": "Evening Doji Star", "type": "bearish", "confidence": 90, "category": "triple"})
        
        # 29. Three White Soldiers
        if c2 > o2 and c1 > o1 and is_bullish and c > c1 > c2 and o > o1 > o2:
            patterns.append({"name": "Three White Soldiers", "type": "bullish", "confidence": 90, "category": "triple"})
        
        # 30. Three Black Crows
        if c2 < o2 and c1 < o1 and is_bearish and c < c1 < c2 and o < o1 < o2:
            patterns.append({"name": "Three Black Crows", "type": "bearish", "confidence": 90, "category": "triple"})
        
        # 31. Three Inside Up
        if c2 < o2 and c1 > o1 and o1 > c2 and c1 < o2 and is_bullish and c > o2:
            patterns.append({"name": "Three Inside Up", "type": "bullish", "confidence": 82, "category": "triple"})
        
        # 32. Three Inside Down
        if c2 > o2 and c1 < o1 and o1 < c2 and c1 > o2 and is_bearish and c < o2:
            patterns.append({"name": "Three Inside Down", "type": "bearish", "confidence": 82, "category": "triple"})
        
        # 33. Three Outside Up
        if c2 < o2 and c1 > o1 and c1 > o2 and o1 < c2 and is_bullish and c > c1:
            patterns.append({"name": "Three Outside Up", "type": "bullish", "confidence": 85, "category": "triple"})
        
        # 34. Three Outside Down
        if c2 > o2 and c1 < o1 and c1 < o2 and o1 > c2 and is_bearish and c < c1:
            patterns.append({"name": "Three Outside Down", "type": "bearish", "confidence": 85, "category": "triple"})
        
        # 35. Bullish Abandoned Baby
        gap_down = h1 < l2
        gap_up = l > h1
        if c2 < o2 and body1 < range1 * 0.1 and gap_down and gap_up and is_bullish:
            patterns.append({"name": "Bullish Abandoned Baby", "type": "bullish", "confidence": 92, "category": "triple"})
        
        # 36. Bearish Abandoned Baby
        gap_up1 = l1 > h2
        gap_down1 = h < l1
        if c2 > o2 and body1 < range1 * 0.1 and gap_up1 and gap_down1 and is_bearish:
            patterns.append({"name": "Bearish Abandoned Baby", "type": "bearish", "confidence": 92, "category": "triple"})
        
        # 37. Rising Three Methods
        if c2 > o2 and body1 < body2 * 0.5 and is_bullish and c > h2:
            patterns.append({"name": "Rising Three Methods", "type": "bullish", "confidence": 80, "category": "triple"})
        
        # 38. Falling Three Methods
        if c2 < o2 and body1 < body2 * 0.5 and is_bearish and c < l2:
            patterns.append({"name": "Falling Three Methods", "type": "bearish", "confidence": 80, "category": "triple"})
        
        # 39. Tri Star Bullish
        doji2 = abs(c2 - o2) < (h2 - l2) * 0.1
        doji1 = abs(c1 - o1) < (h1 - l1) * 0.1
        doji0 = body < total_range * 0.1
        if doji2 and doji1 and doji0 and l1 < l2 and l > l1:
            patterns.append({"name": "Tri Star Bullish", "type": "bullish", "confidence": 85, "category": "triple"})
        
        # 40. Tri Star Bearish
        if doji2 and doji1 and doji0 and h1 > h2 and h < h1:
            patterns.append({"name": "Tri Star Bearish", "type": "bearish", "confidence": 85, "category": "triple"})
    
    # ==================== ADVANCED PATTERNS (24) ====================
    
    if len(closes) >= 4:
        # 41. Unique Three River Bottom
        if c3 < o3 and c2 < o2 and l2 < l3 and c2 > l2 and is_bullish and c < h2:
            patterns.append({"name": "Unique Three River Bottom", "type": "bullish", "confidence": 78, "category": "advanced"})
        
        # 42. Upside Gap Two Crows
        if c2 > o2 and l1 > h2 and c1 < o1 and is_bearish and c < c1 and c > c2:
            patterns.append({"name": "Upside Gap Two Crows", "type": "bearish", "confidence": 75, "category": "advanced"})
        
        # 43. Concealing Baby Swallow
        if c3 < o3 and c2 < o2 and c1 < o1 and is_bearish:
            if abs(c3 - o3) > (h3 - l3) * 0.8 and abs(c2 - o2) > (h2 - l2) * 0.8:
                patterns.append({"name": "Concealing Baby Swallow", "type": "bullish", "confidence": 80, "category": "advanced"})
        
        # 44. Stick Sandwich
        if c2 < o2 and c1 > o1 and is_bearish and abs(c - c2) < total_range * 0.1:
            patterns.append({"name": "Stick Sandwich", "type": "bullish", "confidence": 75, "category": "advanced"})
        
        # 45. Homing Pigeon
        if c2 < o2 and c1 < o1 and o1 < o2 and c1 > c2 and l1 > l2:
            patterns.append({"name": "Homing Pigeon", "type": "bullish", "confidence": 72, "category": "advanced"})
        
        # 46. Ladder Bottom
        if c3 < o3 and c2 < o2 and c1 < o1 and c1 > c2 > c3:
            if is_bullish and lower_wick > body:
                patterns.append({"name": "Ladder Bottom", "type": "bullish", "confidence": 78, "category": "advanced"})
        
        # 47. Ladder Top
        if c3 > o3 and c2 > o2 and c1 > o1 and c1 < c2 < c3:
            if is_bearish and upper_wick > body:
                patterns.append({"name": "Ladder Top", "type": "bearish", "confidence": 78, "category": "advanced"})
        
        # 48. Three Stars in South
        if c2 < o2 and c1 < o1 and is_bearish:
            if body2 > body1 > body and l2 < l1 < l:
                patterns.append({"name": "Three Stars in South", "type": "bullish", "confidence": 75, "category": "advanced"})
        
        # 49. Deliberation
        if c2 > o2 and c1 > o1 and is_bullish:
            if body2 > body1 and body < body1 * 0.5:
                patterns.append({"name": "Deliberation", "type": "bearish", "confidence": 70, "category": "advanced"})
        
        # 50. Advance Block
        if c2 > o2 and c1 > o1 and is_bullish:
            uw2 = h2 - c2
            uw1 = h1 - c1
            uw0 = upper_wick
            if uw0 > uw1 > uw2 and body < body1 < body2:
                patterns.append({"name": "Advance Block", "type": "bearish", "confidence": 75, "category": "advanced"})
        
        # 51. Stalled Pattern
        if c2 > o2 and c1 > o1 and is_bullish:
            if body < body1 * 0.3 and h > h1 and l > l1:
                patterns.append({"name": "Stalled Pattern", "type": "bearish", "confidence": 72, "category": "advanced"})

    # ==================== GAP PATTERNS (13) ====================
    
    if len(closes) >= 2:
        gap_up = l > h1
        gap_down = h < l1
        
        # 52. Breakaway Gap Up
        if gap_up and body > body1 * 1.5 and is_bullish:
            patterns.append({"name": "Breakaway Gap Up", "type": "bullish", "confidence": 82, "category": "gap"})
        
        # 53. Breakaway Gap Down
        if gap_down and body > body1 * 1.5 and is_bearish:
            patterns.append({"name": "Breakaway Gap Down", "type": "bearish", "confidence": 82, "category": "gap"})
        
        # 54. Runaway Gap Up
        if gap_up and c1 > o1 and is_bullish:
            patterns.append({"name": "Runaway Gap Up", "type": "bullish", "confidence": 78, "category": "gap"})
        
        # 55. Runaway Gap Down
        if gap_down and c1 < o1 and is_bearish:
            patterns.append({"name": "Runaway Gap Down", "type": "bearish", "confidence": 78, "category": "gap"})
        
        # 56. Exhaustion Gap Up
        if gap_up and upper_wick > body and is_bearish:
            patterns.append({"name": "Exhaustion Gap Up", "type": "bearish", "confidence": 80, "category": "gap"})
        
        # 57. Exhaustion Gap Down
        if gap_down and lower_wick > body and is_bullish:
            patterns.append({"name": "Exhaustion Gap Down", "type": "bullish", "confidence": 80, "category": "gap"})
        
        # 58. Island Reversal Top
        if len(closes) >= 3:
            gap_up_prev = l1 > h2
            if gap_up_prev and gap_down:
                patterns.append({"name": "Island Reversal Top", "type": "bearish", "confidence": 88, "category": "gap"})
        
        # 59. Island Reversal Bottom
        if len(closes) >= 3:
            gap_down_prev = h1 < l2
            if gap_down_prev and gap_up:
                patterns.append({"name": "Island Reversal Bottom", "type": "bullish", "confidence": 88, "category": "gap"})
    
    # ==================== SPECIAL PATTERNS (5) ====================
    
    # 60. Key Reversal Day Bullish
    if l < l1 and c > h1 and is_bullish:
        patterns.append({"name": "Key Reversal Day Bullish", "type": "bullish", "confidence": 85, "category": "special"})
    
    # 61. Key Reversal Day Bearish
    if h > h1 and c < l1 and is_bearish:
        patterns.append({"name": "Key Reversal Day Bearish", "type": "bearish", "confidence": 85, "category": "special"})
    
    # 62. Outside Day Bullish
    if h > h1 and l < l1 and is_bullish:
        patterns.append({"name": "Outside Day Bullish", "type": "bullish", "confidence": 75, "category": "special"})
    
    # 63. Outside Day Bearish
    if h > h1 and l < l1 and is_bearish:
        patterns.append({"name": "Outside Day Bearish", "type": "bearish", "confidence": 75, "category": "special"})
    
    # 64. Inside Day
    if h < h1 and l > l1:
        patterns.append({"name": "Inside Day", "type": "neutral", "confidence": 70, "category": "special"})
    
    return patterns


# ==================== PATTERN STRENGTH CALCULATOR ====================

def calculate_pattern_strength(patterns):
    """Calculate overall pattern strength and bias"""
    if not patterns:
        return {"bias": "neutral", "strength": 0, "confidence": 0}
    
    bullish_score = 0
    bearish_score = 0
    total_confidence = 0
    
    for p in patterns:
        conf = p.get("confidence", 50)
        total_confidence += conf
        
        if p["type"] == "bullish":
            bullish_score += conf
        elif p["type"] == "bearish":
            bearish_score += conf
    
    avg_confidence = total_confidence / len(patterns) if patterns else 0
    
    if bullish_score > bearish_score * 1.2:
        bias = "bullish"
        strength = (bullish_score - bearish_score) / (bullish_score + bearish_score) * 100 if (bullish_score + bearish_score) > 0 else 0
    elif bearish_score > bullish_score * 1.2:
        bias = "bearish"
        strength = (bearish_score - bullish_score) / (bullish_score + bearish_score) * 100 if (bullish_score + bearish_score) > 0 else 0
    else:
        bias = "neutral"
        strength = 0
    
    return {
        "bias": bias,
        "strength": round(abs(strength), 1),
        "confidence": round(avg_confidence, 1),
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "pattern_count": len(patterns)
    }


def analyze_pattern_confluence(patterns):
    """
    Analyze confluence of detected patterns
    Returns overall bias and strength
    """
    if not patterns:
        return {
            "bias": "neutral",
            "strength": 0,
            "confidence": 0,
            "bullish_score": 0,
            "bearish_score": 0,
            "pattern_count": 0
        }
    
    bullish_score = 0
    bearish_score = 0
    total_confidence = 0
    
    for p in patterns:
        conf = p.get("confidence", 50)
        total_confidence += conf
        
        if p.get("type") == "bullish":
            bullish_score += conf
        elif p.get("type") == "bearish":
            bearish_score += conf
    
    avg_confidence = total_confidence / len(patterns) if patterns else 0
    
    if bullish_score > bearish_score * 1.2:
        bias = "bullish"
        strength = (bullish_score - bearish_score) / (bullish_score + bearish_score) * 100 if (bullish_score + bearish_score) > 0 else 0
    elif bearish_score > bullish_score * 1.2:
        bias = "bearish"
        strength = (bearish_score - bullish_score) / (bullish_score + bearish_score) * 100 if (bullish_score + bearish_score) > 0 else 0
    else:
        bias = "neutral"
        strength = 0
    
    return {
        "bias": bias,
        "strength": round(abs(strength), 1),
        "confidence": round(avg_confidence, 1),
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "pattern_count": len(patterns)
    }
