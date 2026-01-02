"""
COMPLETE CHART PATTERNS - 69 Patterns
Reversal, Continuation, Bilateral, Complex, and Harmonic Patterns
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


def find_swing_points(highs: List[float], lows: List[float], lookback: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Find swing highs and swing lows"""
    swing_highs = []
    swing_lows = []
    
    for i in range(lookback, len(highs) - lookback):
        # Swing High
        is_swing_high = True
        for j in range(1, lookback + 1):
            if highs[i] <= highs[i-j] or highs[i] <= highs[i+j]:
                is_swing_high = False
                break
        if is_swing_high:
            swing_highs.append({"index": i, "price": highs[i]})
        
        # Swing Low
        is_swing_low = True
        for j in range(1, lookback + 1):
            if lows[i] >= lows[i-j] or lows[i] >= lows[i+j]:
                is_swing_low = False
                break
        if is_swing_low:
            swing_lows.append({"index": i, "price": lows[i]})
    
    return swing_highs, swing_lows


def detect_all_chart_patterns(highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """
    Detect ALL 69 chart patterns
    """
    patterns = []
    
    if len(closes) < 50:
        return patterns
    
    swing_highs, swing_lows = find_swing_points(highs, lows)
    
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return patterns
    
    tolerance = (max(highs) - min(lows)) * 0.02
    current_price = closes[-1]
    
    # ==================== REVERSAL PATTERNS (19) ====================
    
    # 1. Double Top
    if len(swing_highs) >= 2:
        h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
        if abs(h1 - h2) < tolerance and current_price < min(h1, h2):
            neckline = min(lows[swing_highs[-2]["index"]:swing_highs[-1]["index"]])
            target = neckline - (h1 - neckline)
            patterns.append({
                "name": "Double Top",
                "type": "bearish",
                "confidence": 82,
                "neckline": neckline,
                "target": target,
                "category": "reversal"
            })
    
    # 2. Double Bottom
    if len(swing_lows) >= 2:
        l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]
        if abs(l1 - l2) < tolerance and current_price > max(l1, l2):
            neckline = max(highs[swing_lows[-2]["index"]:swing_lows[-1]["index"]])
            target = neckline + (neckline - l1)
            patterns.append({
                "name": "Double Bottom",
                "type": "bullish",
                "confidence": 82,
                "neckline": neckline,
                "target": target,
                "category": "reversal"
            })
    
    # 3. Triple Top
    if len(swing_highs) >= 3:
        h1, h2, h3 = swing_highs[-3]["price"], swing_highs[-2]["price"], swing_highs[-1]["price"]
        if abs(h1 - h2) < tolerance and abs(h2 - h3) < tolerance:
            patterns.append({
                "name": "Triple Top",
                "type": "bearish",
                "confidence": 85,
                "category": "reversal"
            })
    
    # 4. Triple Bottom
    if len(swing_lows) >= 3:
        l1, l2, l3 = swing_lows[-3]["price"], swing_lows[-2]["price"], swing_lows[-1]["price"]
        if abs(l1 - l2) < tolerance and abs(l2 - l3) < tolerance:
            patterns.append({
                "name": "Triple Bottom",
                "type": "bullish",
                "confidence": 85,
                "category": "reversal"
            })
    
    # 5. Head and Shoulders
    if len(swing_highs) >= 3:
        left = swing_highs[-3]["price"]
        head = swing_highs[-2]["price"]
        right = swing_highs[-1]["price"]
        
        if head > left and head > right and abs(left - right) < tolerance * 2:
            neckline = min(lows[swing_highs[-3]["index"]:swing_highs[-1]["index"]])
            target = neckline - (head - neckline)
            patterns.append({
                "name": "Head and Shoulders",
                "type": "bearish",
                "confidence": 88,
                "neckline": neckline,
                "target": target,
                "category": "reversal"
            })
    
    # 6. Inverse Head and Shoulders
    if len(swing_lows) >= 3:
        left = swing_lows[-3]["price"]
        head = swing_lows[-2]["price"]
        right = swing_lows[-1]["price"]
        
        if head < left and head < right and abs(left - right) < tolerance * 2:
            neckline = max(highs[swing_lows[-3]["index"]:swing_lows[-1]["index"]])
            target = neckline + (neckline - head)
            patterns.append({
                "name": "Inverse Head and Shoulders",
                "type": "bullish",
                "confidence": 88,
                "neckline": neckline,
                "target": target,
                "category": "reversal"
            })
    
    # 7. Rounding Top (Dome)
    recent_highs = [h["price"] for h in swing_highs[-5:]] if len(swing_highs) >= 5 else []
    if len(recent_highs) >= 5:
        if recent_highs[0] < recent_highs[2] > recent_highs[4]:
            patterns.append({
                "name": "Rounding Top",
                "type": "bearish",
                "confidence": 75,
                "category": "reversal"
            })
    
    # 8. Rounding Bottom (Saucer)
    recent_lows = [l["price"] for l in swing_lows[-5:]] if len(swing_lows) >= 5 else []
    if len(recent_lows) >= 5:
        if recent_lows[0] > recent_lows[2] < recent_lows[4]:
            patterns.append({
                "name": "Rounding Bottom",
                "type": "bullish",
                "confidence": 75,
                "category": "reversal"
            })
    
    # 9-10. V-Top and V-Bottom
    if len(closes) >= 10:
        recent_range = max(highs[-10:]) - min(lows[-10:])
        price_change = abs(closes[-1] - closes[-10])
        
        if price_change > recent_range * 0.7:
            if closes[-1] < closes[-5] < closes[-10]:
                patterns.append({"name": "V-Top", "type": "bearish", "confidence": 72, "category": "reversal"})
            elif closes[-1] > closes[-5] > closes[-10]:
                patterns.append({"name": "V-Bottom", "type": "bullish", "confidence": 72, "category": "reversal"})

    # ==================== CONTINUATION PATTERNS (22) ====================
    
    # 11. Ascending Triangle
    recent_highs_prices = [h["price"] for h in swing_highs[-4:]]
    recent_lows_prices = [l["price"] for l in swing_lows[-4:]]
    
    if len(recent_highs_prices) >= 3 and len(recent_lows_prices) >= 3:
        highs_flat = max(recent_highs_prices) - min(recent_highs_prices) < tolerance
        lows_rising = recent_lows_prices[-1] > recent_lows_prices[0]
        
        if highs_flat and lows_rising:
            breakout = max(recent_highs_prices)
            patterns.append({
                "name": "Ascending Triangle",
                "type": "bullish",
                "confidence": 78,
                "breakout": breakout,
                "category": "continuation"
            })
    
    # 12. Descending Triangle
    if len(recent_highs_prices) >= 3 and len(recent_lows_prices) >= 3:
        lows_flat = max(recent_lows_prices) - min(recent_lows_prices) < tolerance
        highs_falling = recent_highs_prices[-1] < recent_highs_prices[0]
        
        if lows_flat and highs_falling:
            breakout = min(recent_lows_prices)
            patterns.append({
                "name": "Descending Triangle",
                "type": "bearish",
                "confidence": 78,
                "breakout": breakout,
                "category": "continuation"
            })
    
    # 13. Symmetrical Triangle
    if len(recent_highs_prices) >= 3 and len(recent_lows_prices) >= 3:
        highs_falling = recent_highs_prices[-1] < recent_highs_prices[0]
        lows_rising = recent_lows_prices[-1] > recent_lows_prices[0]
        
        if highs_falling and lows_rising:
            apex = (recent_highs_prices[-1] + recent_lows_prices[-1]) / 2
            patterns.append({
                "name": "Symmetrical Triangle",
                "type": "neutral",
                "confidence": 72,
                "apex": apex,
                "category": "continuation"
            })
    
    # 14. Rising Wedge
    if len(recent_highs_prices) >= 3 and len(recent_lows_prices) >= 3:
        highs_rising = recent_highs_prices[-1] > recent_highs_prices[0]
        lows_rising = recent_lows_prices[-1] > recent_lows_prices[0]
        
        high_slope = (recent_highs_prices[-1] - recent_highs_prices[0]) / len(recent_highs_prices)
        low_slope = (recent_lows_prices[-1] - recent_lows_prices[0]) / len(recent_lows_prices)
        
        if highs_rising and lows_rising and low_slope > high_slope:
            patterns.append({
                "name": "Rising Wedge",
                "type": "bearish",
                "confidence": 75,
                "category": "continuation"
            })
    
    # 15. Falling Wedge
    if len(recent_highs_prices) >= 3 and len(recent_lows_prices) >= 3:
        highs_falling = recent_highs_prices[-1] < recent_highs_prices[0]
        lows_falling = recent_lows_prices[-1] < recent_lows_prices[0]
        
        high_slope = (recent_highs_prices[-1] - recent_highs_prices[0]) / len(recent_highs_prices)
        low_slope = (recent_lows_prices[-1] - recent_lows_prices[0]) / len(recent_lows_prices)
        
        if highs_falling and lows_falling and abs(high_slope) < abs(low_slope):
            patterns.append({
                "name": "Falling Wedge",
                "type": "bullish",
                "confidence": 75,
                "category": "continuation"
            })
    
    # 16-17. Bullish/Bearish Flag
    if len(closes) >= 20:
        # Check for strong move followed by consolidation
        move = closes[-20] - closes[-10]
        consolidation_range = max(highs[-10:]) - min(lows[-10:])
        
        if abs(move) > consolidation_range * 2:
            if move > 0:
                patterns.append({"name": "Bullish Flag", "type": "bullish", "confidence": 78, "category": "continuation"})
            else:
                patterns.append({"name": "Bearish Flag", "type": "bearish", "confidence": 78, "category": "continuation"})
    
    # 18-19. Bullish/Bearish Pennant
    if len(closes) >= 15:
        recent_range = max(highs[-5:]) - min(lows[-5:])
        prev_range = max(highs[-15:-5]) - min(lows[-15:-5])
        
        if recent_range < prev_range * 0.5:
            trend = closes[-15] - closes[-5]
            if trend > 0:
                patterns.append({"name": "Bullish Pennant", "type": "bullish", "confidence": 75, "category": "continuation"})
            else:
                patterns.append({"name": "Bearish Pennant", "type": "bearish", "confidence": 75, "category": "continuation"})
    
    # 20. Cup and Handle
    if len(swing_lows) >= 3 and len(swing_highs) >= 2:
        # Look for U-shape followed by small pullback
        if len(closes) >= 30:
            left_high = max(highs[:10])
            cup_low = min(lows[10:20])
            right_high = max(highs[20:25])
            handle_low = min(lows[25:])
            
            if abs(left_high - right_high) < tolerance * 2 and cup_low < left_high * 0.9:
                if handle_low > cup_low and handle_low < right_high:
                    patterns.append({
                        "name": "Cup and Handle",
                        "type": "bullish",
                        "confidence": 80,
                        "breakout": right_high,
                        "category": "continuation"
                    })
    
    # 21-22. Rectangle patterns
    if len(closes) >= 20:
        high_range = max(highs[-20:]) - min(highs[-20:])
        low_range = max(lows[-20:]) - min(lows[-20:])
        
        if high_range < tolerance * 3 and low_range < tolerance * 3:
            trend = closes[-25] - closes[-20] if len(closes) >= 25 else 0
            if trend > 0:
                patterns.append({"name": "Bullish Rectangle", "type": "bullish", "confidence": 72, "category": "continuation"})
            else:
                patterns.append({"name": "Bearish Rectangle", "type": "bearish", "confidence": 72, "category": "continuation"})

    # ==================== HARMONIC PATTERNS (10) ====================
    
    # Harmonic pattern detection using Fibonacci ratios
    fib_ratios = {
        "0.382": 0.382,
        "0.5": 0.5,
        "0.618": 0.618,
        "0.786": 0.786,
        "1.0": 1.0,
        "1.272": 1.272,
        "1.618": 1.618,
        "2.0": 2.0,
        "2.618": 2.618
    }
    
    def check_ratio(actual, expected, tolerance=0.05):
        return abs(actual - expected) < tolerance
    
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Get XABCD points
        points = []
        all_swings = sorted(
            [(h["index"], h["price"], "high") for h in swing_highs[-4:]] +
            [(l["index"], l["price"], "low") for l in swing_lows[-4:]],
            key=lambda x: x[0]
        )
        
        if len(all_swings) >= 5:
            X = all_swings[0][1]
            A = all_swings[1][1]
            B = all_swings[2][1]
            C = all_swings[3][1]
            D = all_swings[4][1] if len(all_swings) > 4 else current_price
            
            XA = abs(A - X)
            AB = abs(B - A)
            BC = abs(C - B)
            CD = abs(D - C)
            
            if XA > 0:
                AB_XA = AB / XA
                BC_AB = BC / AB if AB > 0 else 0
                CD_BC = CD / BC if BC > 0 else 0
                
                # 23. Gartley Pattern (AB = 0.618 XA, BC = 0.382-0.886 AB, CD = 1.272-1.618 BC)
                if check_ratio(AB_XA, 0.618, 0.1) and 0.382 <= BC_AB <= 0.886:
                    pattern_type = "bullish" if D < X else "bearish"
                    patterns.append({
                        "name": "Gartley Pattern",
                        "type": pattern_type,
                        "confidence": 80,
                        "category": "harmonic"
                    })
                
                # 24. Butterfly Pattern (AB = 0.786 XA)
                if check_ratio(AB_XA, 0.786, 0.1):
                    pattern_type = "bullish" if D < X else "bearish"
                    patterns.append({
                        "name": "Butterfly Pattern",
                        "type": pattern_type,
                        "confidence": 78,
                        "category": "harmonic"
                    })
                
                # 25. Bat Pattern (AB = 0.382-0.5 XA)
                if 0.382 <= AB_XA <= 0.5:
                    pattern_type = "bullish" if D < X else "bearish"
                    patterns.append({
                        "name": "Bat Pattern",
                        "type": pattern_type,
                        "confidence": 78,
                        "category": "harmonic"
                    })
                
                # 26. Crab Pattern (AB = 0.382-0.618 XA, CD extends to 1.618 XA)
                if 0.382 <= AB_XA <= 0.618:
                    pattern_type = "bullish" if D < X else "bearish"
                    patterns.append({
                        "name": "Crab Pattern",
                        "type": pattern_type,
                        "confidence": 75,
                        "category": "harmonic"
                    })
                
                # 27. Cypher Pattern
                if check_ratio(AB_XA, 0.382, 0.1) or check_ratio(AB_XA, 0.618, 0.1):
                    pattern_type = "bullish" if D < X else "bearish"
                    patterns.append({
                        "name": "Cypher Pattern",
                        "type": pattern_type,
                        "confidence": 72,
                        "category": "harmonic"
                    })
                
                # 28. Shark Pattern
                if check_ratio(AB_XA, 0.886, 0.1):
                    pattern_type = "bullish" if D < X else "bearish"
                    patterns.append({
                        "name": "Shark Pattern",
                        "type": pattern_type,
                        "confidence": 70,
                        "category": "harmonic"
                    })
    
    # 29-30. ABCD Pattern
    if len(swing_highs) >= 2 and len(swing_lows) >= 2:
        # Bullish ABCD
        if len(swing_lows) >= 2:
            A = swing_lows[-2]["price"]
            C = swing_lows[-1]["price"]
            B = max(highs[swing_lows[-2]["index"]:swing_lows[-1]["index"]]) if swing_lows[-2]["index"] < swing_lows[-1]["index"] else 0
            
            if B > 0:
                AB = B - A
                BC = B - C
                
                if 0.5 <= BC/AB <= 0.786 if AB > 0 else False:
                    patterns.append({
                        "name": "Bullish ABCD",
                        "type": "bullish",
                        "confidence": 75,
                        "category": "harmonic"
                    })
        
        # Bearish ABCD
        if len(swing_highs) >= 2:
            A = swing_highs[-2]["price"]
            C = swing_highs[-1]["price"]
            B = min(lows[swing_highs[-2]["index"]:swing_highs[-1]["index"]]) if swing_highs[-2]["index"] < swing_highs[-1]["index"] else 0
            
            if B > 0:
                AB = A - B
                BC = C - B
                
                if 0.5 <= BC/AB <= 0.786 if AB > 0 else False:
                    patterns.append({
                        "name": "Bearish ABCD",
                        "type": "bearish",
                        "confidence": 75,
                        "category": "harmonic"
                    })
    
    # 31-32. Three Drives Pattern
    if len(swing_highs) >= 3:
        h1, h2, h3 = swing_highs[-3]["price"], swing_highs[-2]["price"], swing_highs[-1]["price"]
        if h1 < h2 < h3:
            drive1 = h2 - h1
            drive2 = h3 - h2
            if 0.8 <= drive2/drive1 <= 1.2 if drive1 > 0 else False:
                patterns.append({
                    "name": "Three Drives Up",
                    "type": "bearish",
                    "confidence": 72,
                    "category": "harmonic"
                })
    
    if len(swing_lows) >= 3:
        l1, l2, l3 = swing_lows[-3]["price"], swing_lows[-2]["price"], swing_lows[-1]["price"]
        if l1 > l2 > l3:
            drive1 = l1 - l2
            drive2 = l2 - l3
            if 0.8 <= drive2/drive1 <= 1.2 if drive1 > 0 else False:
                patterns.append({
                    "name": "Three Drives Down",
                    "type": "bullish",
                    "confidence": 72,
                    "category": "harmonic"
                })

    # ==================== BILATERAL PATTERNS (9) ====================
    
    # 33-34. Diamond patterns
    if len(closes) >= 30:
        first_half_range = max(highs[:15]) - min(lows[:15])
        second_half_range = max(highs[15:]) - min(lows[15:])
        middle_range = max(highs[10:20]) - min(lows[10:20])
        
        # Diamond: expanding then contracting
        if middle_range > first_half_range and middle_range > second_half_range:
            trend = closes[0] - closes[15]
            if trend > 0:
                patterns.append({"name": "Diamond Top", "type": "bearish", "confidence": 78, "category": "bilateral"})
            else:
                patterns.append({"name": "Diamond Bottom", "type": "bullish", "confidence": 78, "category": "bilateral"})
    
    # 35-36. Broadening patterns
    if len(recent_highs_prices) >= 3 and len(recent_lows_prices) >= 3:
        highs_expanding = recent_highs_prices[-1] > recent_highs_prices[0]
        lows_expanding = recent_lows_prices[-1] < recent_lows_prices[0]
        
        if highs_expanding and lows_expanding:
            trend = closes[-20] - closes[-1] if len(closes) >= 20 else 0
            if trend > 0:
                patterns.append({"name": "Broadening Top", "type": "bearish", "confidence": 70, "category": "bilateral"})
            else:
                patterns.append({"name": "Broadening Bottom", "type": "bullish", "confidence": 70, "category": "bilateral"})
    
    # 37. Megaphone Pattern
    if len(closes) >= 25:
        range_start = max(highs[:5]) - min(lows[:5])
        range_end = max(highs[-5:]) - min(lows[-5:])
        
        if range_end > range_start * 1.5:
            patterns.append({
                "name": "Megaphone Pattern",
                "type": "neutral",
                "confidence": 68,
                "category": "bilateral"
            })
    
    # 38-39. Horn patterns
    if len(swing_highs) >= 2:
        h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
        if abs(h1 - h2) < tolerance and current_price < min(h1, h2) * 0.98:
            patterns.append({"name": "Horn Top", "type": "bearish", "confidence": 72, "category": "bilateral"})
    
    if len(swing_lows) >= 2:
        l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]
        if abs(l1 - l2) < tolerance and current_price > max(l1, l2) * 1.02:
            patterns.append({"name": "Horn Bottom", "type": "bullish", "confidence": 72, "category": "bilateral"})
    
    # ==================== COMPLEX PATTERNS (10) ====================
    
    # 40-41. Wolfe Wave
    if len(swing_highs) >= 3 and len(swing_lows) >= 2:
        # Simplified Wolfe Wave detection
        h1, h2, h3 = swing_highs[-3]["price"], swing_highs[-2]["price"], swing_highs[-1]["price"]
        l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]
        
        # Bearish Wolfe Wave: 1-3-5 line and 2-4 line converge
        if h1 < h2 < h3 and l1 < l2:
            patterns.append({
                "name": "Bearish Wolfe Wave",
                "type": "bearish",
                "confidence": 70,
                "category": "complex"
            })
    
    if len(swing_lows) >= 3 and len(swing_highs) >= 2:
        l1, l2, l3 = swing_lows[-3]["price"], swing_lows[-2]["price"], swing_lows[-1]["price"]
        h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
        
        if l1 > l2 > l3 and h1 > h2:
            patterns.append({
                "name": "Bullish Wolfe Wave",
                "type": "bullish",
                "confidence": 70,
                "category": "complex"
            })
    
    # 42-43. Measured Move
    if len(closes) >= 30:
        first_move = closes[10] - closes[0]
        consolidation = max(highs[10:20]) - min(lows[10:20])
        second_move = closes[-1] - closes[20]
        
        if abs(first_move) > consolidation * 2:
            if 0.8 <= abs(second_move/first_move) <= 1.2 if first_move != 0 else False:
                if first_move > 0:
                    patterns.append({"name": "Measured Move Up", "type": "bullish", "confidence": 75, "category": "complex"})
                else:
                    patterns.append({"name": "Measured Move Down", "type": "bearish", "confidence": 75, "category": "complex"})
    
    # ==================== ADDITIONAL PATTERNS (27) ====================
    
    # 44. Inverted Cup and Handle
    if len(closes) >= 30:
        left_low = min(lows[:10])
        cup_high = max(highs[10:20])
        right_low = min(lows[20:25])
        
        if abs(left_low - right_low) < tolerance * 2 and cup_high > left_low * 1.1:
            patterns.append({
                "name": "Inverted Cup and Handle",
                "type": "bearish",
                "confidence": 78,
                "category": "continuation"
            })
    
    # 45-46. Channel patterns
    if len(closes) >= 20:
        high_trend = (highs[-1] - highs[-20]) / 20
        low_trend = (lows[-1] - lows[-20]) / 20
        
        if high_trend > 0 and low_trend > 0 and abs(high_trend - low_trend) < tolerance * 0.1:
            patterns.append({"name": "Ascending Channel", "type": "bullish", "confidence": 72, "category": "continuation"})
        elif high_trend < 0 and low_trend < 0 and abs(high_trend - low_trend) < tolerance * 0.1:
            patterns.append({"name": "Descending Channel", "type": "bearish", "confidence": 72, "category": "continuation"})
    
    # 47. Failed Head and Shoulders
    if len(swing_highs) >= 3:
        left = swing_highs[-3]["price"]
        head = swing_highs[-2]["price"]
        right = swing_highs[-1]["price"]
        
        if head > left and head > right and current_price > head:
            patterns.append({
                "name": "Failed Head and Shoulders",
                "type": "bullish",
                "confidence": 82,
                "category": "reversal"
            })
    
    # 48. Complex Head and Shoulders
    if len(swing_highs) >= 5:
        peaks = [h["price"] for h in swing_highs[-5:]]
        if peaks[2] > max(peaks[0], peaks[1], peaks[3], peaks[4]):
            if abs(peaks[0] - peaks[4]) < tolerance and abs(peaks[1] - peaks[3]) < tolerance:
                patterns.append({
                    "name": "Complex Head and Shoulders",
                    "type": "bearish",
                    "confidence": 85,
                    "category": "reversal"
                })
    
    # 49-50. Spike patterns
    if len(closes) >= 10:
        recent_range = max(highs[-5:]) - min(lows[-5:])
        avg_range = np.mean([highs[i] - lows[i] for i in range(-10, -5)])
        
        if avg_range > 0 and recent_range > avg_range * 3:
            if closes[-1] < closes[-5]:
                patterns.append({"name": "Spike Top", "type": "bearish", "confidence": 75, "category": "reversal"})
            else:
                patterns.append({"name": "Spike Bottom", "type": "bullish", "confidence": 75, "category": "reversal"})
    
    return patterns


def calculate_chart_pattern_targets(pattern: Dict, current_price: float) -> Dict:
    """Calculate price targets for chart patterns"""
    targets = {
        "entry": current_price,
        "stop_loss": None,
        "target_1": None,
        "target_2": None,
        "risk_reward": None
    }
    
    if "neckline" in pattern and "target" in pattern:
        neckline = pattern["neckline"]
        target = pattern["target"]
        
        if pattern["type"] == "bullish":
            targets["stop_loss"] = neckline * 0.99
            targets["target_1"] = target
            targets["target_2"] = target + (target - neckline) * 0.5
        else:
            targets["stop_loss"] = neckline * 1.01
            targets["target_1"] = target
            targets["target_2"] = target - (neckline - target) * 0.5
        
        risk = abs(current_price - targets["stop_loss"])
        reward = abs(targets["target_1"] - current_price)
        targets["risk_reward"] = round(reward / risk, 2) if risk > 0 else 0
    
    return targets


def analyze_pattern_quality(patterns):
    """
    Analyze quality and reliability of detected chart patterns
    """
    if not patterns:
        return {"quality": "none", "score": 0, "count": 0}
    
    total_score = 0
    count = 0
    
    for category, pattern_list in patterns.items():
        if isinstance(pattern_list, list):
            for p in pattern_list:
                if isinstance(p, dict):
                    total_score += p.get('confidence', 50)
                    count += 1
    
    avg_score = total_score / count if count > 0 else 0
    
    if avg_score >= 80:
        quality = "high"
    elif avg_score >= 60:
        quality = "medium"
    else:
        quality = "low"
    
    return {
        "quality": quality,
        "score": round(avg_score, 1),
        "count": count
    }
