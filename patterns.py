# ==================== CANDLESTICK PATTERNS (64 Patterns) ====================

def identify_candlestick_patterns(opens, highs, lows, closes):
    patterns = []
    if len(closes) < 3:
        return patterns
    
    # Current candle
    o, h, l, c = opens[-1], highs[-1], lows[-1], closes[-1]
    body = abs(c - o)
    total_range = h - l if h != l else 0.0001
    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    is_bullish = c > o
    is_bearish = c < o
    
    # Previous candles
    o1, h1, l1, c1 = opens[-2], highs[-2], lows[-2], closes[-2] if len(closes) >= 2 else (o, h, l, c)
    o2, h2, l2, c2 = opens[-3], highs[-3], lows[-3], closes[-3] if len(closes) >= 3 else (o, h, l, c)
    body1 = abs(c1 - o1)
    body2 = abs(c2 - o2)
    
    # ========== SINGLE CANDLE PATTERNS ==========
    
    # Doji variations
    if body < total_range * 0.1:
        if upper_wick > total_range * 0.3 and lower_wick > total_range * 0.3:
            patterns.append({"name": "Long Legged Doji", "type": "neutral", "confidence": 80})
        elif upper_wick < total_range * 0.1 and lower_wick > total_range * 0.6:
            patterns.append({"name": "Dragonfly Doji", "type": "bullish", "confidence": 85})
        elif lower_wick < total_range * 0.1 and upper_wick > total_range * 0.6:
            patterns.append({"name": "Gravestone Doji", "type": "bearish", "confidence": 85})
        else:
            patterns.append({"name": "Doji", "type": "neutral", "confidence": 75})
    
    # Hammer
    if lower_wick > body * 2 and upper_wick < body * 0.3 and body > total_range * 0.1:
        patterns.append({"name": "Hammer", "type": "bullish", "confidence": 82})
    
    # Inverted Hammer
    if upper_wick > body * 2 and lower_wick < body * 0.3 and is_bullish:
        patterns.append({"name": "Inverted Hammer", "type": "bullish", "confidence": 78})
    
    # Shooting Star
    if upper_wick > body * 2 and lower_wick < body * 0.3 and is_bearish:
        patterns.append({"name": "Shooting Star", "type": "bearish", "confidence": 82})
    
    # Hanging Man
    if lower_wick > body * 2 and upper_wick < body * 0.3 and is_bearish:
        patterns.append({"name": "Hanging Man", "type": "bearish", "confidence": 78})

    # Marubozu
    if upper_wick < body * 0.05 and lower_wick < body * 0.05 and body > total_range * 0.8:
        if is_bullish:
            patterns.append({"name": "Bullish Marubozu", "type": "bullish", "confidence": 88})
        else:
            patterns.append({"name": "Bearish Marubozu", "type": "bearish", "confidence": 88})
    
    # Belt Hold
    if is_bullish and lower_wick < body * 0.1 and body > total_range * 0.6:
        patterns.append({"name": "Bullish Belt Hold", "type": "bullish", "confidence": 75})
    if is_bearish and upper_wick < body * 0.1 and body > total_range * 0.6:
        patterns.append({"name": "Bearish Belt Hold", "type": "bearish", "confidence": 75})
    
    # Spinning Top
    if body < total_range * 0.3 and upper_wick > body and lower_wick > body:
        patterns.append({"name": "Spinning Top", "type": "neutral", "confidence": 70})
    
    # ========== TWO CANDLE PATTERNS ==========
    if len(closes) >= 2:
        # Bullish Engulfing
        if is_bullish and c1 < o1 and c > o1 and o < c1 and body > body1:
            patterns.append({"name": "Bullish Engulfing", "type": "bullish", "confidence": 85})
        
        # Bearish Engulfing
        if is_bearish and c1 > o1 and c < o1 and o > c1 and body > body1:
            patterns.append({"name": "Bearish Engulfing", "type": "bearish", "confidence": 85})
        
        # Piercing Line
        if c1 < o1 and is_bullish and o < l1 and c > (o1 + c1) / 2 and c < o1:
            patterns.append({"name": "Piercing Line", "type": "bullish", "confidence": 80})
        
        # Dark Cloud Cover
        if c1 > o1 and is_bearish and o > h1 and c < (o1 + c1) / 2 and c > o1:
            patterns.append({"name": "Dark Cloud Cover", "type": "bearish", "confidence": 80})
        
        # Bullish Harami
        if c1 < o1 and is_bullish and o > c1 and c < o1 and body < body1 * 0.5:
            patterns.append({"name": "Bullish Harami", "type": "bullish", "confidence": 75})
        
        # Bearish Harami
        if c1 > o1 and is_bearish and o < c1 and c > o1 and body < body1 * 0.5:
            patterns.append({"name": "Bearish Harami", "type": "bearish", "confidence": 75})
        
        # Tweezer Bottom
        if abs(l - l1) < total_range * 0.05 and c1 < o1 and is_bullish:
            patterns.append({"name": "Tweezer Bottom", "type": "bullish", "confidence": 78})
        
        # Tweezer Top
        if abs(h - h1) < total_range * 0.05 and c1 > o1 and is_bearish:
            patterns.append({"name": "Tweezer Top", "type": "bearish", "confidence": 78})

    # ========== THREE CANDLE PATTERNS ==========
    if len(closes) >= 3:
        # Morning Star
        if c2 < o2 and body1 < body2 * 0.3 and is_bullish and c > (o2 + c2) / 2:
            patterns.append({"name": "Morning Star", "type": "bullish", "confidence": 88})
        
        # Evening Star
        if c2 > o2 and body1 < body2 * 0.3 and is_bearish and c < (o2 + c2) / 2:
            patterns.append({"name": "Evening Star", "type": "bearish", "confidence": 88})
        
        # Three White Soldiers
        if c2 > o2 and c1 > o1 and is_bullish and c > c1 > c2 and o > o1 > o2:
            patterns.append({"name": "Three White Soldiers", "type": "bullish", "confidence": 90})
        
        # Three Black Crows
        if c2 < o2 and c1 < o1 and is_bearish and c < c1 < c2 and o < o1 < o2:
            patterns.append({"name": "Three Black Crows", "type": "bearish", "confidence": 90})
        
        # Three Inside Up
        if c2 < o2 and c1 > o1 and o1 > c2 and c1 < o2 and is_bullish and c > o2:
            patterns.append({"name": "Three Inside Up", "type": "bullish", "confidence": 82})
        
        # Three Inside Down
        if c2 > o2 and c1 < o1 and o1 < c2 and c1 > o2 and is_bearish and c < o2:
            patterns.append({"name": "Three Inside Down", "type": "bearish", "confidence": 82})
        
        # Rising Three Methods
        if c2 > o2 and body1 < body2 * 0.5 and is_bullish and c > h2:
            patterns.append({"name": "Rising Three Methods", "type": "bullish", "confidence": 80})
        
        # Falling Three Methods
        if c2 < o2 and body1 < body2 * 0.5 and is_bearish and c < l2:
            patterns.append({"name": "Falling Three Methods", "type": "bearish", "confidence": 80})
    
    return patterns


# ==================== ICT/SMC CONCEPTS ====================

def identify_market_structure(highs, lows, closes):
    structure = {
        "trend": "sideways",
        "swing_highs": [],
        "swing_lows": [],
        "bos": [],
        "choch": None,
        "hh_hl": False,
        "lh_ll": False
    }
    
    if len(closes) < 15:
        return structure
    
    # Find swing points
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            structure["swing_highs"].append({"index": i, "price": highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            structure["swing_lows"].append({"index": i, "price": lows[i]})
    
    # Determine trend
    sh = structure["swing_highs"]
    sl = structure["swing_lows"]
    
    if len(sh) >= 2 and len(sl) >= 2:
        if sh[-1]["price"] > sh[-2]["price"] and sl[-1]["price"] > sl[-2]["price"]:
            structure["trend"] = "bullish"
            structure["hh_hl"] = True
        elif sh[-1]["price"] < sh[-2]["price"] and sl[-1]["price"] < sl[-2]["price"]:
            structure["trend"] = "bearish"
            structure["lh_ll"] = True
    
    return structure


def find_fvg(opens, highs, lows, closes):
    """Find Fair Value Gaps (FVG)"""
    fvgs = []
    if len(closes) < 3:
        return fvgs
    
    for i in range(2, len(closes)):
        # Bullish FVG: gap between candle 1 high and candle 3 low
        if lows[i] > highs[i-2]:
            fvgs.append({
                "type": "bullish",
                "top": lows[i],
                "bottom": highs[i-2],
                "index": i,
                "filled": closes[-1] < lows[i]
            })
        
        # Bearish FVG: gap between candle 1 low and candle 3 high
        if highs[i] < lows[i-2]:
            fvgs.append({
                "type": "bearish",
                "top": lows[i-2],
                "bottom": highs[i],
                "index": i,
                "filled": closes[-1] > highs[i]
            })
    
    return fvgs[-5:] if len(fvgs) > 5 else fvgs


def find_order_blocks(opens, highs, lows, closes):
    """Find Order Blocks"""
    obs = []
    if len(closes) < 5:
        return obs
    
    for i in range(2, len(closes) - 2):
        # Bullish OB: last bearish candle before strong bullish move
        if closes[i] < opens[i] and closes[i+1] > opens[i+1] and closes[i+2] > opens[i+2]:
            if closes[i+2] > highs[i]:
                obs.append({
                    "type": "bullish",
                    "top": highs[i],
                    "bottom": lows[i],
                    "index": i,
                    "mitigated": closes[-1] < lows[i]
                })
        
        # Bearish OB: last bullish candle before strong bearish move
        if closes[i] > opens[i] and closes[i+1] < opens[i+1] and closes[i+2] < opens[i+2]:
            if closes[i+2] < lows[i]:
                obs.append({
                    "type": "bearish",
                    "top": highs[i],
                    "bottom": lows[i],
                    "index": i,
                    "mitigated": closes[-1] > highs[i]
                })
    
    return obs[-5:] if len(obs) > 5 else obs


def find_liquidity_zones(highs, lows, closes):
    """Find Equal Highs/Lows (Liquidity)"""
    liquidity = {"eqh": [], "eql": [], "bsl": [], "ssl": []}
    
    if len(closes) < 10:
        return liquidity
    
    tolerance = (max(highs) - min(lows)) * 0.002
    
    # Find equal highs
    for i in range(len(highs) - 1):
        for j in range(i + 1, len(highs)):
            if abs(highs[i] - highs[j]) < tolerance:
                liquidity["eqh"].append({"price": highs[i], "indices": [i, j]})
                liquidity["bsl"].append(highs[i])
    
    # Find equal lows
    for i in range(len(lows) - 1):
        for j in range(i + 1, len(lows)):
            if abs(lows[i] - lows[j]) < tolerance:
                liquidity["eql"].append({"price": lows[i], "indices": [i, j]})
                liquidity["ssl"].append(lows[i])
    
    return liquidity


def find_support_resistance(highs, lows, closes):
    """Find Support and Resistance levels"""
    levels = {"support": [], "resistance": []}
    
    if len(closes) < 20:
        return levels
    
    current_price = closes[-1]
    all_levels = []
    
    # Find pivot points
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            all_levels.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            all_levels.append(lows[i])
    
    for level in sorted(set(all_levels)):
        if level > current_price:
            levels["resistance"].append(round(level, 5))
        else:
            levels["support"].append(round(level, 5))
    
    levels["support"] = sorted(levels["support"], reverse=True)[:3]
    levels["resistance"] = sorted(levels["resistance"])[:3]
    
    return levels


# ==================== CHART PATTERNS ====================

def detect_chart_patterns(highs, lows, closes):
    """Detect major chart patterns"""
    patterns = []
    
    if len(closes) < 30:
        return patterns
    
    # Find swing points for pattern detection
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append({"index": i, "price": highs[i]})
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append({"index": i, "price": lows[i]})
    
    if len(swing_highs) < 3 or len(swing_lows) < 3:
        return patterns
    
    tolerance = (max(highs) - min(lows)) * 0.02
    
    # Double Top
    if len(swing_highs) >= 2:
        h1, h2 = swing_highs[-2]["price"], swing_highs[-1]["price"]
        if abs(h1 - h2) < tolerance and h2 < h1:
            patterns.append({
                "name": "Double Top",
                "type": "bearish",
                "confidence": 80,
                "target": min(lows[-20:])
            })
    
    # Double Bottom
    if len(swing_lows) >= 2:
        l1, l2 = swing_lows[-2]["price"], swing_lows[-1]["price"]
        if abs(l1 - l2) < tolerance and l2 > l1:
            patterns.append({
                "name": "Double Bottom",
                "type": "bullish",
                "confidence": 80,
                "target": max(highs[-20:])
            })
    
    # Head and Shoulders
    if len(swing_highs) >= 3:
        left = swing_highs[-3]["price"]
        head = swing_highs[-2]["price"]
        right = swing_highs[-1]["price"]
        
        if head > left and head > right and abs(left - right) < tolerance:
            patterns.append({
                "name": "Head and Shoulders",
                "type": "bearish",
                "confidence": 85,
                "neckline": min(lows[-15:])
            })
    
    # Inverse Head and Shoulders
    if len(swing_lows) >= 3:
        left = swing_lows[-3]["price"]
        head = swing_lows[-2]["price"]
        right = swing_lows[-1]["price"]
        
        if head < left and head < right and abs(left - right) < tolerance:
            patterns.append({
                "name": "Inverse Head and Shoulders",
                "type": "bullish",
                "confidence": 85,
                "neckline": max(highs[-15:])
            })
    
    # Ascending Triangle
    recent_highs = [h["price"] for h in swing_highs[-4:]]
    recent_lows = [l["price"] for l in swing_lows[-4:]]
    
    if len(recent_highs) >= 3 and len(recent_lows) >= 3:
        highs_flat = max(recent_highs) - min(recent_highs) < tolerance
        lows_rising = recent_lows[-1] > recent_lows[0]
        
        if highs_flat and lows_rising:
            patterns.append({
                "name": "Ascending Triangle",
                "type": "bullish",
                "confidence": 75,
                "breakout": max(recent_highs)
            })
    
    # Descending Triangle
    if len(recent_highs) >= 3 and len(recent_lows) >= 3:
        lows_flat = max(recent_lows) - min(recent_lows) < tolerance
        highs_falling = recent_highs[-1] < recent_highs[0]
        
        if lows_flat and highs_falling:
            patterns.append({
                "name": "Descending Triangle",
                "type": "bearish",
                "confidence": 75,
                "breakout": min(recent_lows)
            })
    
    # Symmetrical Triangle
    if len(recent_highs) >= 3 and len(recent_lows) >= 3:
        highs_falling = recent_highs[-1] < recent_highs[0]
        lows_rising = recent_lows[-1] > recent_lows[0]
        
        if highs_falling and lows_rising:
            patterns.append({
                "name": "Symmetrical Triangle",
                "type": "neutral",
                "confidence": 70,
                "apex": (recent_highs[-1] + recent_lows[-1]) / 2
            })
    
    return patterns
