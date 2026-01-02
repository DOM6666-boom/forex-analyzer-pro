"""
COMPLETE PRICE ACTION ANALYSIS - 150 Concepts
Market Structure, Swing Analysis, Price Behavior, Entry/Exit Patterns
"""

import numpy as np
from typing import List, Dict, Tuple, Optional


# ==================== MARKET STRUCTURE (25) ====================

def identify_swing_high(highs: List[float], lookback: int = 5) -> List[Dict]:
    """Identify Swing Highs"""
    swings = []
    for i in range(lookback, len(highs) - lookback):
        is_swing = all(highs[i] > highs[i-j] for j in range(1, lookback+1)) and \
                   all(highs[i] > highs[i+j] for j in range(1, lookback+1))
        if is_swing:
            swings.append({'index': i, 'price': highs[i], 'type': 'swing_high'})
    return swings


def identify_swing_low(lows: List[float], lookback: int = 5) -> List[Dict]:
    """Identify Swing Lows"""
    swings = []
    for i in range(lookback, len(lows) - lookback):
        is_swing = all(lows[i] < lows[i-j] for j in range(1, lookback+1)) and \
                   all(lows[i] < lows[i+j] for j in range(1, lookback+1))
        if is_swing:
            swings.append({'index': i, 'price': lows[i], 'type': 'swing_low'})
    return swings


def identify_higher_high(highs: List[float], lookback: int = 5) -> bool:
    """Check for Higher High"""
    swing_highs = identify_swing_high(highs, lookback)
    if len(swing_highs) < 2:
        return False
    return swing_highs[-1]['price'] > swing_highs[-2]['price']


def identify_higher_low(lows: List[float], lookback: int = 5) -> bool:
    """Check for Higher Low"""
    swing_lows = identify_swing_low(lows, lookback)
    if len(swing_lows) < 2:
        return False
    return swing_lows[-1]['price'] > swing_lows[-2]['price']


def identify_lower_high(highs: List[float], lookback: int = 5) -> bool:
    """Check for Lower High"""
    swing_highs = identify_swing_high(highs, lookback)
    if len(swing_highs) < 2:
        return False
    return swing_highs[-1]['price'] < swing_highs[-2]['price']


def identify_lower_low(lows: List[float], lookback: int = 5) -> bool:
    """Check for Lower Low"""
    swing_lows = identify_swing_low(lows, lookback)
    if len(swing_lows) < 2:
        return False
    return swing_lows[-1]['price'] < swing_lows[-2]['price']


def identify_uptrend(highs: List[float], lows: List[float], lookback: int = 5) -> Dict:
    """Identify Uptrend (HH + HL)"""
    hh = identify_higher_high(highs, lookback)
    hl = identify_higher_low(lows, lookback)
    
    return {
        'is_uptrend': hh and hl,
        'higher_high': hh,
        'higher_low': hl,
        'strength': 'strong' if hh and hl else 'weak' if hh or hl else 'none'
    }


def identify_downtrend(highs: List[float], lows: List[float], lookback: int = 5) -> Dict:
    """Identify Downtrend (LH + LL)"""
    lh = identify_lower_high(highs, lookback)
    ll = identify_lower_low(lows, lookback)
    
    return {
        'is_downtrend': lh and ll,
        'lower_high': lh,
        'lower_low': ll,
        'strength': 'strong' if lh and ll else 'weak' if lh or ll else 'none'
    }


def identify_range_market(highs: List[float], lows: List[float], threshold: float = 0.02) -> Dict:
    """Identify Range/Consolidation Market"""
    if len(highs) < 10:
        return {'is_range': False}
    
    high_range = max(highs[-20:]) - min(highs[-20:])
    low_range = max(lows[-20:]) - min(lows[-20:])
    avg_price = (sum(highs[-20:]) + sum(lows[-20:])) / 40
    
    range_pct = (high_range + low_range) / 2 / avg_price
    
    return {
        'is_range': range_pct < threshold,
        'range_high': max(highs[-20:]),
        'range_low': min(lows[-20:]),
        'range_percentage': range_pct * 100
    }


def identify_break_of_structure(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Break of Structure (BOS)"""
    swing_highs = identify_swing_high(highs)
    swing_lows = identify_swing_low(lows)
    
    result = {
        'bullish_bos': False,
        'bearish_bos': False,
        'bos_level': None
    }
    
    if len(swing_highs) >= 2 and closes[-1] > swing_highs[-1]['price']:
        result['bullish_bos'] = True
        result['bos_level'] = swing_highs[-1]['price']
    
    if len(swing_lows) >= 2 and closes[-1] < swing_lows[-1]['price']:
        result['bearish_bos'] = True
        result['bos_level'] = swing_lows[-1]['price']
    
    return result


def identify_change_of_character(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Change of Character (CHoCH)"""
    uptrend = identify_uptrend(highs, lows)
    downtrend = identify_downtrend(highs, lows)
    
    result = {
        'bullish_choch': False,
        'bearish_choch': False,
        'previous_trend': None
    }
    
    # CHoCH occurs when trend changes
    if downtrend['is_downtrend'] and identify_higher_high(highs):
        result['bullish_choch'] = True
        result['previous_trend'] = 'bearish'
    
    if uptrend['is_uptrend'] and identify_lower_low(lows):
        result['bearish_choch'] = True
        result['previous_trend'] = 'bullish'
    
    return result


def identify_market_structure_shift(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Market Structure Shift (MSS)"""
    bos = identify_break_of_structure(highs, lows, closes)
    choch = identify_change_of_character(highs, lows, closes)
    
    return {
        'mss_bullish': bos['bullish_bos'] or choch['bullish_choch'],
        'mss_bearish': bos['bearish_bos'] or choch['bearish_choch'],
        'bos': bos,
        'choch': choch
    }


def identify_trend_continuation(highs: List[float], lows: List[float]) -> Dict:
    """Identify Trend Continuation"""
    uptrend = identify_uptrend(highs, lows)
    downtrend = identify_downtrend(highs, lows)
    
    return {
        'bullish_continuation': uptrend['is_uptrend'],
        'bearish_continuation': downtrend['is_downtrend'],
        'trend_strength': uptrend['strength'] if uptrend['is_uptrend'] else downtrend['strength']
    }


def identify_trend_reversal(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Potential Trend Reversal"""
    choch = identify_change_of_character(highs, lows, closes)
    
    return {
        'bullish_reversal': choch['bullish_choch'],
        'bearish_reversal': choch['bearish_choch'],
        'reversal_confirmed': choch['bullish_choch'] or choch['bearish_choch']
    }


def identify_impulse_move(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Impulse Move"""
    if len(closes) < 5:
        return {'is_impulse': False}
    
    # Calculate average candle size
    avg_range = sum(highs[i] - lows[i] for i in range(-10, 0)) / 10
    current_range = highs[-1] - lows[-1]
    
    # Impulse is significantly larger than average
    is_impulse = current_range > avg_range * 1.5
    
    direction = 'bullish' if closes[-1] > opens[-1] else 'bearish'
    
    return {
        'is_impulse': is_impulse,
        'direction': direction,
        'strength': current_range / avg_range if avg_range > 0 else 0
    }


def identify_corrective_move(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Corrective Move"""
    if len(closes) < 10:
        return {'is_corrective': False}
    
    # Look for smaller, overlapping candles
    recent_ranges = [highs[i] - lows[i] for i in range(-5, 0)]
    avg_recent = sum(recent_ranges) / 5
    
    prior_ranges = [highs[i] - lows[i] for i in range(-10, -5)]
    avg_prior = sum(prior_ranges) / 5
    
    is_corrective = avg_recent < avg_prior * 0.7
    
    return {
        'is_corrective': is_corrective,
        'contraction_ratio': avg_recent / avg_prior if avg_prior > 0 else 1
    }


def identify_accumulation_phase(highs: List[float], lows: List[float], volumes: List[float]) -> Dict:
    """Identify Accumulation Phase"""
    if len(highs) < 20:
        return {'is_accumulation': False}
    
    # Range-bound with increasing volume
    range_market = identify_range_market(highs, lows)
    
    avg_vol_early = sum(volumes[-20:-10]) / 10
    avg_vol_late = sum(volumes[-10:]) / 10
    
    volume_increasing = avg_vol_late > avg_vol_early * 1.2
    
    return {
        'is_accumulation': range_market['is_range'] and volume_increasing,
        'range_info': range_market,
        'volume_trend': 'increasing' if volume_increasing else 'decreasing'
    }


def identify_distribution_phase(highs: List[float], lows: List[float], volumes: List[float]) -> Dict:
    """Identify Distribution Phase"""
    if len(highs) < 20:
        return {'is_distribution': False}
    
    # Range-bound at highs with volume
    range_market = identify_range_market(highs, lows)
    
    # Check if at relative highs
    recent_high = max(highs[-20:])
    all_time_high = max(highs)
    at_highs = recent_high >= all_time_high * 0.95
    
    avg_vol = sum(volumes[-20:]) / 20
    high_volume = avg_vol > sum(volumes[-40:-20]) / 20 * 1.2 if len(volumes) >= 40 else False
    
    return {
        'is_distribution': range_market['is_range'] and at_highs and high_volume,
        'at_highs': at_highs,
        'high_volume': high_volume
    }


def identify_markup_phase(highs: List[float], lows: List[float]) -> Dict:
    """Identify Markup Phase (Strong Uptrend)"""
    uptrend = identify_uptrend(highs, lows)
    
    if len(highs) < 20:
        return {'is_markup': False}
    
    # Calculate trend angle
    price_change = highs[-1] - highs[-20]
    pct_change = price_change / highs[-20] * 100 if highs[-20] > 0 else 0
    
    return {
        'is_markup': uptrend['is_uptrend'] and pct_change > 5,
        'trend_strength': uptrend['strength'],
        'price_change_pct': pct_change
    }


def identify_markdown_phase(highs: List[float], lows: List[float]) -> Dict:
    """Identify Markdown Phase (Strong Downtrend)"""
    downtrend = identify_downtrend(highs, lows)
    
    if len(lows) < 20:
        return {'is_markdown': False}
    
    price_change = lows[-1] - lows[-20]
    pct_change = price_change / lows[-20] * 100 if lows[-20] > 0 else 0
    
    return {
        'is_markdown': downtrend['is_downtrend'] and pct_change < -5,
        'trend_strength': downtrend['strength'],
        'price_change_pct': pct_change
    }


def identify_wyckoff_phase(highs: List[float], lows: List[float], volumes: List[float]) -> str:
    """Identify Wyckoff Market Phase"""
    accumulation = identify_accumulation_phase(highs, lows, volumes)
    distribution = identify_distribution_phase(highs, lows, volumes)
    markup = identify_markup_phase(highs, lows)
    markdown = identify_markdown_phase(highs, lows)
    
    if accumulation['is_accumulation']:
        return 'ACCUMULATION'
    elif markup['is_markup']:
        return 'MARKUP'
    elif distribution['is_distribution']:
        return 'DISTRIBUTION'
    elif markdown['is_markdown']:
        return 'MARKDOWN'
    else:
        return 'TRANSITION'


def identify_spring(lows: List[float], closes: List[float], lookback: int = 20) -> Dict:
    """Identify Wyckoff Spring"""
    if len(lows) < lookback:
        return {'is_spring': False}
    
    support = min(lows[-lookback:-5])
    
    # Spring: price breaks below support then closes back above
    broke_support = any(lows[i] < support for i in range(-5, 0))
    closed_above = closes[-1] > support
    
    return {
        'is_spring': broke_support and closed_above,
        'support_level': support,
        'spring_low': min(lows[-5:])
    }


def identify_upthrust(highs: List[float], closes: List[float], lookback: int = 20) -> Dict:
    """Identify Wyckoff Upthrust"""
    if len(highs) < lookback:
        return {'is_upthrust': False}
    
    resistance = max(highs[-lookback:-5])
    
    # Upthrust: price breaks above resistance then closes back below
    broke_resistance = any(highs[i] > resistance for i in range(-5, 0))
    closed_below = closes[-1] < resistance
    
    return {
        'is_upthrust': broke_resistance and closed_below,
        'resistance_level': resistance,
        'upthrust_high': max(highs[-5:])
    }


# ==================== SUPPORT & RESISTANCE (20) ====================

def identify_horizontal_support(lows: List[float], tolerance: float = 0.001) -> List[float]:
    """Identify Horizontal Support Levels"""
    if len(lows) < 10:
        return []
    
    levels = []
    swing_lows = identify_swing_low(lows)
    
    for swing in swing_lows:
        # Check if multiple touches
        touches = sum(1 for l in lows if abs(l - swing['price']) / swing['price'] < tolerance)
        if touches >= 2:
            levels.append(swing['price'])
    
    return sorted(set(levels))


def identify_horizontal_resistance(highs: List[float], tolerance: float = 0.001) -> List[float]:
    """Identify Horizontal Resistance Levels"""
    if len(highs) < 10:
        return []
    
    levels = []
    swing_highs = identify_swing_high(highs)
    
    for swing in swing_highs:
        touches = sum(1 for h in highs if abs(h - swing['price']) / swing['price'] < tolerance)
        if touches >= 2:
            levels.append(swing['price'])
    
    return sorted(set(levels), reverse=True)


def identify_dynamic_support(closes: List[float], period: int = 20) -> Optional[float]:
    """Identify Dynamic Support (Moving Average)"""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def identify_dynamic_resistance(closes: List[float], period: int = 20) -> Optional[float]:
    """Identify Dynamic Resistance (Moving Average)"""
    return identify_dynamic_support(closes, period)


def identify_trendline_support(lows: List[float]) -> Dict:
    """Identify Trendline Support"""
    swing_lows = identify_swing_low(lows)
    
    if len(swing_lows) < 2:
        return {'valid': False}
    
    # Use last two swing lows
    p1 = swing_lows[-2]
    p2 = swing_lows[-1]
    
    slope = (p2['price'] - p1['price']) / (p2['index'] - p1['index'])
    
    # Project to current bar
    current_level = p2['price'] + slope * (len(lows) - 1 - p2['index'])
    
    return {
        'valid': True,
        'slope': slope,
        'current_level': current_level,
        'direction': 'ascending' if slope > 0 else 'descending'
    }


def identify_trendline_resistance(highs: List[float]) -> Dict:
    """Identify Trendline Resistance"""
    swing_highs = identify_swing_high(highs)
    
    if len(swing_highs) < 2:
        return {'valid': False}
    
    p1 = swing_highs[-2]
    p2 = swing_highs[-1]
    
    slope = (p2['price'] - p1['price']) / (p2['index'] - p1['index'])
    current_level = p2['price'] + slope * (len(highs) - 1 - p2['index'])
    
    return {
        'valid': True,
        'slope': slope,
        'current_level': current_level,
        'direction': 'ascending' if slope > 0 else 'descending'
    }


def identify_support_zone(lows: List[float], zone_size: float = 0.005) -> List[Dict]:
    """Identify Support Zones"""
    levels = identify_horizontal_support(lows)
    zones = []
    
    for level in levels:
        zones.append({
            'upper': level * (1 + zone_size),
            'lower': level * (1 - zone_size),
            'center': level
        })
    
    return zones


def identify_resistance_zone(highs: List[float], zone_size: float = 0.005) -> List[Dict]:
    """Identify Resistance Zones"""
    levels = identify_horizontal_resistance(highs)
    zones = []
    
    for level in levels:
        zones.append({
            'upper': level * (1 + zone_size),
            'lower': level * (1 - zone_size),
            'center': level
        })
    
    return zones


def identify_support_flip_resistance(highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Support Turned Resistance"""
    support_levels = identify_horizontal_support(lows)
    flipped = []
    
    for level in support_levels:
        # Check if price broke below and now testing from below
        broke_below = any(closes[i] < level for i in range(-20, -5))
        testing_from_below = closes[-1] < level and highs[-1] >= level * 0.998
        
        if broke_below and testing_from_below:
            flipped.append({
                'level': level,
                'type': 'support_to_resistance'
            })
    
    return flipped


def identify_resistance_flip_support(highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Resistance Turned Support"""
    resistance_levels = identify_horizontal_resistance(highs)
    flipped = []
    
    for level in resistance_levels:
        broke_above = any(closes[i] > level for i in range(-20, -5))
        testing_from_above = closes[-1] > level and lows[-1] <= level * 1.002
        
        if broke_above and testing_from_above:
            flipped.append({
                'level': level,
                'type': 'resistance_to_support'
            })
    
    return flipped


def identify_round_number_levels(current_price: float, range_pct: float = 0.05) -> List[float]:
    """Identify Psychological Round Number Levels"""
    levels = []
    
    # Determine the magnitude
    magnitude = 10 ** (len(str(int(current_price))) - 2)
    
    base = int(current_price / magnitude) * magnitude
    
    for i in range(-5, 6):
        level = base + i * magnitude
        if abs(level - current_price) / current_price < range_pct:
            levels.append(level)
    
    return levels


def identify_pivot_high(highs: List[float], left: int = 5, right: int = 5) -> List[Dict]:
    """Identify Pivot Highs"""
    pivots = []
    
    for i in range(left, len(highs) - right):
        is_pivot = all(highs[i] >= highs[i-j] for j in range(1, left+1)) and \
                   all(highs[i] >= highs[i+j] for j in range(1, right+1))
        if is_pivot:
            pivots.append({'index': i, 'price': highs[i]})
    
    return pivots


def identify_pivot_low(lows: List[float], left: int = 5, right: int = 5) -> List[Dict]:
    """Identify Pivot Lows"""
    pivots = []
    
    for i in range(left, len(lows) - right):
        is_pivot = all(lows[i] <= lows[i-j] for j in range(1, left+1)) and \
                   all(lows[i] <= lows[i+j] for j in range(1, right+1))
        if is_pivot:
            pivots.append({'index': i, 'price': lows[i]})
    
    return pivots


def identify_equal_highs(highs: List[float], tolerance: float = 0.001) -> List[Dict]:
    """Identify Equal Highs (Double Top potential)"""
    swing_highs = identify_swing_high(highs)
    equal_highs = []
    
    for i in range(len(swing_highs) - 1):
        for j in range(i + 1, len(swing_highs)):
            diff = abs(swing_highs[i]['price'] - swing_highs[j]['price'])
            if diff / swing_highs[i]['price'] < tolerance:
                equal_highs.append({
                    'level': (swing_highs[i]['price'] + swing_highs[j]['price']) / 2,
                    'indices': [swing_highs[i]['index'], swing_highs[j]['index']]
                })
    
    return equal_highs


def identify_equal_lows(lows: List[float], tolerance: float = 0.001) -> List[Dict]:
    """Identify Equal Lows (Double Bottom potential)"""
    swing_lows = identify_swing_low(lows)
    equal_lows = []
    
    for i in range(len(swing_lows) - 1):
        for j in range(i + 1, len(swing_lows)):
            diff = abs(swing_lows[i]['price'] - swing_lows[j]['price'])
            if diff / swing_lows[i]['price'] < tolerance:
                equal_lows.append({
                    'level': (swing_lows[i]['price'] + swing_lows[j]['price']) / 2,
                    'indices': [swing_lows[i]['index'], swing_lows[j]['index']]
                })
    
    return equal_lows


def identify_liquidity_pool_high(highs: List[float]) -> Dict:
    """Identify Liquidity Pool Above (Stop Hunt Target)"""
    equal_highs = identify_equal_highs(highs)
    
    if not equal_highs:
        return {'exists': False}
    
    # Most recent equal highs
    latest = equal_highs[-1]
    
    return {
        'exists': True,
        'level': latest['level'],
        'type': 'buy_stops_above'
    }


def identify_liquidity_pool_low(lows: List[float]) -> Dict:
    """Identify Liquidity Pool Below (Stop Hunt Target)"""
    equal_lows = identify_equal_lows(lows)
    
    if not equal_lows:
        return {'exists': False}
    
    latest = equal_lows[-1]
    
    return {
        'exists': True,
        'level': latest['level'],
        'type': 'sell_stops_below'
    }


def identify_order_block_support(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Bullish Order Blocks (Support)"""
    order_blocks = []
    
    for i in range(2, len(closes) - 2):
        # Last bearish candle before bullish move
        if closes[i] < opens[i]:  # Bearish candle
            # Followed by strong bullish move
            if closes[i+1] > opens[i+1] and closes[i+1] > highs[i]:
                order_blocks.append({
                    'index': i,
                    'high': highs[i],
                    'low': lows[i],
                    'type': 'bullish_ob'
                })
    
    return order_blocks[-3:] if order_blocks else []


def identify_order_block_resistance(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Bearish Order Blocks (Resistance)"""
    order_blocks = []
    
    for i in range(2, len(closes) - 2):
        # Last bullish candle before bearish move
        if closes[i] > opens[i]:  # Bullish candle
            # Followed by strong bearish move
            if closes[i+1] < opens[i+1] and closes[i+1] < lows[i]:
                order_blocks.append({
                    'index': i,
                    'high': highs[i],
                    'low': lows[i],
                    'type': 'bearish_ob'
                })
    
    return order_blocks[-3:] if order_blocks else []


# ==================== PRICE PATTERNS (25) ====================

def identify_inside_bar(highs: List[float], lows: List[float]) -> bool:
    """Identify Inside Bar"""
    if len(highs) < 2:
        return False
    return highs[-1] < highs[-2] and lows[-1] > lows[-2]


def identify_outside_bar(highs: List[float], lows: List[float]) -> bool:
    """Identify Outside Bar (Engulfing)"""
    if len(highs) < 2:
        return False
    return highs[-1] > highs[-2] and lows[-1] < lows[-2]


def identify_pin_bar(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Pin Bar"""
    if len(closes) < 1:
        return {'is_pin_bar': False}
    
    body = abs(closes[-1] - opens[-1])
    total_range = highs[-1] - lows[-1]
    
    if total_range == 0:
        return {'is_pin_bar': False}
    
    upper_wick = highs[-1] - max(opens[-1], closes[-1])
    lower_wick = min(opens[-1], closes[-1]) - lows[-1]
    
    body_ratio = body / total_range
    
    # Pin bar: small body, long wick
    is_bullish_pin = lower_wick > body * 2 and body_ratio < 0.3
    is_bearish_pin = upper_wick > body * 2 and body_ratio < 0.3
    
    return {
        'is_pin_bar': is_bullish_pin or is_bearish_pin,
        'direction': 'bullish' if is_bullish_pin else 'bearish' if is_bearish_pin else None
    }


def identify_two_bar_reversal(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Two Bar Reversal"""
    if len(closes) < 2:
        return {'is_reversal': False}
    
    # Bullish: bearish then bullish with higher close
    bullish = closes[-2] < opens[-2] and closes[-1] > opens[-1] and closes[-1] > highs[-2]
    
    # Bearish: bullish then bearish with lower close
    bearish = closes[-2] > opens[-2] and closes[-1] < opens[-1] and closes[-1] < lows[-2]
    
    return {
        'is_reversal': bullish or bearish,
        'direction': 'bullish' if bullish else 'bearish' if bearish else None
    }


def identify_three_bar_reversal(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Three Bar Reversal"""
    if len(closes) < 3:
        return {'is_reversal': False}
    
    # Bullish: down, inside/doji, up
    bullish = (closes[-3] < opens[-3] and 
               abs(closes[-2] - opens[-2]) < (highs[-2] - lows[-2]) * 0.3 and
               closes[-1] > opens[-1] and closes[-1] > highs[-3])
    
    # Bearish: up, inside/doji, down
    bearish = (closes[-3] > opens[-3] and 
               abs(closes[-2] - opens[-2]) < (highs[-2] - lows[-2]) * 0.3 and
               closes[-1] < opens[-1] and closes[-1] < lows[-3])
    
    return {
        'is_reversal': bullish or bearish,
        'direction': 'bullish' if bullish else 'bearish' if bearish else None
    }


def identify_fakey_pattern(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Fakey Pattern (False Breakout)"""
    if len(closes) < 3:
        return {'is_fakey': False}
    
    # Inside bar followed by false breakout
    inside_bar = highs[-2] < highs[-3] and lows[-2] > lows[-3]
    
    if not inside_bar:
        return {'is_fakey': False}
    
    # False breakout above then close back inside
    bullish_fakey = highs[-1] > highs[-3] and closes[-1] < highs[-3] and closes[-1] > lows[-3]
    
    # False breakout below then close back inside
    bearish_fakey = lows[-1] < lows[-3] and closes[-1] > lows[-3] and closes[-1] < highs[-3]
    
    return {
        'is_fakey': bullish_fakey or bearish_fakey,
        'direction': 'bearish' if bullish_fakey else 'bullish' if bearish_fakey else None
    }


def identify_compression(highs: List[float], lows: List[float], lookback: int = 10) -> Dict:
    """Identify Price Compression"""
    if len(highs) < lookback:
        return {'is_compression': False}
    
    ranges = [highs[i] - lows[i] for i in range(-lookback, 0)]
    
    # Check if ranges are decreasing
    decreasing = all(ranges[i] >= ranges[i+1] for i in range(len(ranges)-1))
    
    avg_range = sum(ranges) / len(ranges)
    current_range = ranges[-1]
    
    return {
        'is_compression': current_range < avg_range * 0.5,
        'compression_ratio': current_range / avg_range if avg_range > 0 else 1,
        'decreasing_ranges': decreasing
    }


def identify_expansion(highs: List[float], lows: List[float], lookback: int = 10) -> Dict:
    """Identify Price Expansion"""
    if len(highs) < lookback:
        return {'is_expansion': False}
    
    ranges = [highs[i] - lows[i] for i in range(-lookback, 0)]
    avg_range = sum(ranges[:-1]) / (len(ranges) - 1)
    current_range = ranges[-1]
    
    return {
        'is_expansion': current_range > avg_range * 1.5,
        'expansion_ratio': current_range / avg_range if avg_range > 0 else 1
    }


def identify_breakout(highs: List[float], lows: List[float], closes: List[float], lookback: int = 20) -> Dict:
    """Identify Breakout"""
    if len(closes) < lookback:
        return {'is_breakout': False}
    
    resistance = max(highs[-lookback:-1])
    support = min(lows[-lookback:-1])
    
    bullish_breakout = closes[-1] > resistance
    bearish_breakout = closes[-1] < support
    
    return {
        'is_breakout': bullish_breakout or bearish_breakout,
        'direction': 'bullish' if bullish_breakout else 'bearish' if bearish_breakout else None,
        'breakout_level': resistance if bullish_breakout else support if bearish_breakout else None
    }


def identify_false_breakout(highs: List[float], lows: List[float], closes: List[float], lookback: int = 20) -> Dict:
    """Identify False Breakout"""
    if len(closes) < lookback + 2:
        return {'is_false_breakout': False}
    
    resistance = max(highs[-lookback-2:-2])
    support = min(lows[-lookback-2:-2])
    
    # Broke above resistance but closed back below
    false_bull = highs[-2] > resistance and closes[-1] < resistance
    
    # Broke below support but closed back above
    false_bear = lows[-2] < support and closes[-1] > support
    
    return {
        'is_false_breakout': false_bull or false_bear,
        'direction': 'bearish' if false_bull else 'bullish' if false_bear else None
    }


def identify_retest(highs: List[float], lows: List[float], closes: List[float], level: float, tolerance: float = 0.002) -> Dict:
    """Identify Retest of Level"""
    if len(closes) < 5:
        return {'is_retest': False}
    
    # Price was away from level and now touching it
    was_above = any(lows[i] > level * (1 + tolerance) for i in range(-10, -3))
    was_below = any(highs[i] < level * (1 - tolerance) for i in range(-10, -3))
    
    touching = abs(closes[-1] - level) / level < tolerance
    
    return {
        'is_retest': (was_above or was_below) and touching,
        'retest_from': 'above' if was_above else 'below' if was_below else None
    }


def identify_pullback(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Pullback in Trend"""
    uptrend = identify_uptrend(highs, lows)
    downtrend = identify_downtrend(highs, lows)
    
    if len(closes) < 5:
        return {'is_pullback': False}
    
    # In uptrend, pullback is temporary move down
    if uptrend['is_uptrend']:
        recent_down = closes[-1] < closes[-3]
        return {
            'is_pullback': recent_down,
            'trend': 'bullish',
            'depth': (closes[-3] - closes[-1]) / closes[-3] * 100
        }
    
    # In downtrend, pullback is temporary move up
    if downtrend['is_downtrend']:
        recent_up = closes[-1] > closes[-3]
        return {
            'is_pullback': recent_up,
            'trend': 'bearish',
            'depth': (closes[-1] - closes[-3]) / closes[-3] * 100
        }
    
    return {'is_pullback': False}


def identify_throwback(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Throwback (Return to Breakout Level)"""
    breakout = identify_breakout(highs[:-3], lows[:-3], closes[:-3])
    
    if not breakout['is_breakout']:
        return {'is_throwback': False}
    
    level = breakout['breakout_level']
    
    # Price returning to breakout level
    returning = abs(closes[-1] - level) / level < 0.01
    
    return {
        'is_throwback': returning,
        'breakout_direction': breakout['direction'],
        'level': level
    }


def identify_gap(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Price Gap"""
    if len(opens) < 2:
        return {'has_gap': False}
    
    # Gap up: current low > previous high
    gap_up = lows[-1] > highs[-2]
    gap_up_size = lows[-1] - highs[-2] if gap_up else 0
    
    # Gap down: current high < previous low
    gap_down = highs[-1] < lows[-2]
    gap_down_size = lows[-2] - highs[-1] if gap_down else 0
    
    return {
        'has_gap': gap_up or gap_down,
        'direction': 'up' if gap_up else 'down' if gap_down else None,
        'gap_size': gap_up_size if gap_up else gap_down_size
    }


def identify_gap_fill(opens: List[float], highs: List[float], lows: List[float], closes: List[float], lookback: int = 20) -> Dict:
    """Identify Gap Fill"""
    gaps = []
    
    for i in range(-lookback, -1):
        if lows[i] > highs[i-1]:  # Gap up
            gaps.append({'type': 'up', 'level': highs[i-1], 'index': i})
        elif highs[i] < lows[i-1]:  # Gap down
            gaps.append({'type': 'down', 'level': lows[i-1], 'index': i})
    
    filled = []
    for gap in gaps:
        if gap['type'] == 'up' and lows[-1] <= gap['level']:
            filled.append(gap)
        elif gap['type'] == 'down' and highs[-1] >= gap['level']:
            filled.append(gap)
    
    return {
        'gaps_filled': len(filled) > 0,
        'filled_gaps': filled
    }


# ==================== CANDLESTICK BEHAVIOR (25) ====================

def analyze_candle_body(open_price: float, close: float, high: float, low: float) -> Dict:
    """Analyze Candle Body Characteristics"""
    body = abs(close - open_price)
    total_range = high - low
    
    if total_range == 0:
        return {'body_type': 'doji', 'body_ratio': 0}
    
    body_ratio = body / total_range
    
    if body_ratio < 0.1:
        body_type = 'doji'
    elif body_ratio < 0.3:
        body_type = 'small'
    elif body_ratio < 0.6:
        body_type = 'medium'
    else:
        body_type = 'large'
    
    return {
        'body_type': body_type,
        'body_ratio': body_ratio,
        'direction': 'bullish' if close > open_price else 'bearish'
    }


def analyze_candle_wicks(open_price: float, close: float, high: float, low: float) -> Dict:
    """Analyze Candle Wick Characteristics"""
    body_high = max(open_price, close)
    body_low = min(open_price, close)
    
    upper_wick = high - body_high
    lower_wick = body_low - low
    total_range = high - low
    
    if total_range == 0:
        return {'upper_wick_ratio': 0, 'lower_wick_ratio': 0}
    
    return {
        'upper_wick': upper_wick,
        'lower_wick': lower_wick,
        'upper_wick_ratio': upper_wick / total_range,
        'lower_wick_ratio': lower_wick / total_range,
        'wick_dominance': 'upper' if upper_wick > lower_wick else 'lower'
    }


def identify_momentum_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Momentum/Marubozu Candle"""
    if len(closes) < 1:
        return {'is_momentum': False}
    
    body = abs(closes[-1] - opens[-1])
    total_range = highs[-1] - lows[-1]
    
    if total_range == 0:
        return {'is_momentum': False}
    
    body_ratio = body / total_range
    
    return {
        'is_momentum': body_ratio > 0.8,
        'direction': 'bullish' if closes[-1] > opens[-1] else 'bearish',
        'strength': body_ratio
    }


def identify_indecision_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Indecision Candle (Doji, Spinning Top)"""
    if len(closes) < 1:
        return {'is_indecision': False}
    
    body = abs(closes[-1] - opens[-1])
    total_range = highs[-1] - lows[-1]
    
    if total_range == 0:
        return {'is_indecision': True, 'type': 'doji'}
    
    body_ratio = body / total_range
    
    if body_ratio < 0.1:
        return {'is_indecision': True, 'type': 'doji'}
    elif body_ratio < 0.3:
        return {'is_indecision': True, 'type': 'spinning_top'}
    
    return {'is_indecision': False}


def identify_rejection_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Rejection Candle"""
    if len(closes) < 1:
        return {'is_rejection': False}
    
    wicks = analyze_candle_wicks(opens[-1], closes[-1], highs[-1], lows[-1])
    body = analyze_candle_body(opens[-1], closes[-1], highs[-1], lows[-1])
    
    # Rejection: long wick, small body
    upper_rejection = wicks['upper_wick_ratio'] > 0.6 and body['body_ratio'] < 0.3
    lower_rejection = wicks['lower_wick_ratio'] > 0.6 and body['body_ratio'] < 0.3
    
    return {
        'is_rejection': upper_rejection or lower_rejection,
        'rejection_from': 'above' if upper_rejection else 'below' if lower_rejection else None
    }


def identify_exhaustion_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Exhaustion Candle"""
    if len(closes) < 10:
        return {'is_exhaustion': False}
    
    # Large candle with high volume at end of trend
    avg_range = sum(highs[i] - lows[i] for i in range(-10, -1)) / 9
    current_range = highs[-1] - lows[-1]
    
    avg_volume = sum(volumes[-10:-1]) / 9
    current_volume = volumes[-1]
    
    large_candle = current_range > avg_range * 1.5
    high_volume = current_volume > avg_volume * 1.5
    
    return {
        'is_exhaustion': large_candle and high_volume,
        'direction': 'bullish' if closes[-1] > opens[-1] else 'bearish'
    }


def identify_reversal_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Reversal Candle"""
    pin_bar = identify_pin_bar(opens, highs, lows, closes)
    rejection = identify_rejection_candle(opens, highs, lows, closes)
    
    return {
        'is_reversal': pin_bar['is_pin_bar'] or rejection['is_rejection'],
        'type': 'pin_bar' if pin_bar['is_pin_bar'] else 'rejection' if rejection['is_rejection'] else None,
        'direction': pin_bar.get('direction') or rejection.get('rejection_from')
    }


def identify_continuation_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Continuation Candle"""
    if len(closes) < 3:
        return {'is_continuation': False}
    
    # Same direction as previous candles
    prev_bullish = closes[-2] > opens[-2]
    curr_bullish = closes[-1] > opens[-1]
    
    prev_bearish = closes[-2] < opens[-2]
    curr_bearish = closes[-1] < opens[-1]
    
    return {
        'is_continuation': (prev_bullish and curr_bullish) or (prev_bearish and curr_bearish),
        'direction': 'bullish' if curr_bullish else 'bearish'
    }


def calculate_candle_range_percentile(highs: List[float], lows: List[float], lookback: int = 20) -> float:
    """Calculate Current Candle Range Percentile"""
    if len(highs) < lookback:
        return 50.0
    
    ranges = [highs[i] - lows[i] for i in range(-lookback, 0)]
    current_range = ranges[-1]
    
    below_count = sum(1 for r in ranges[:-1] if r < current_range)
    return (below_count / (lookback - 1)) * 100


def identify_high_volume_candle(volumes: List[float], lookback: int = 20) -> Dict:
    """Identify High Volume Candle"""
    if len(volumes) < lookback:
        return {'is_high_volume': False}
    
    avg_volume = sum(volumes[-lookback:-1]) / (lookback - 1)
    current_volume = volumes[-1]
    
    return {
        'is_high_volume': current_volume > avg_volume * 1.5,
        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
    }


def identify_low_volume_candle(volumes: List[float], lookback: int = 20) -> Dict:
    """Identify Low Volume Candle"""
    if len(volumes) < lookback:
        return {'is_low_volume': False}
    
    avg_volume = sum(volumes[-lookback:-1]) / (lookback - 1)
    current_volume = volumes[-1]
    
    return {
        'is_low_volume': current_volume < avg_volume * 0.5,
        'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 1
    }


def identify_climax_candle(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Climax Candle (Extreme Move)"""
    if len(closes) < 20:
        return {'is_climax': False}
    
    # Largest range in lookback period
    ranges = [highs[i] - lows[i] for i in range(-20, 0)]
    current_range = ranges[-1]
    max_range = max(ranges[:-1])
    
    # Highest volume
    current_volume = volumes[-1]
    max_volume = max(volumes[-20:-1])
    
    return {
        'is_climax': current_range >= max_range and current_volume >= max_volume * 0.9,
        'direction': 'bullish' if closes[-1] > opens[-1] else 'bearish'
    }


def identify_narrow_range_candle(highs: List[float], lows: List[float], lookback: int = 7) -> Dict:
    """Identify Narrow Range Candle (NR4, NR7)"""
    if len(highs) < lookback:
        return {'is_narrow_range': False}
    
    ranges = [highs[i] - lows[i] for i in range(-lookback, 0)]
    current_range = ranges[-1]
    
    is_narrowest = all(current_range <= r for r in ranges[:-1])
    
    return {
        'is_narrow_range': is_narrowest,
        'nr_type': f'NR{lookback}' if is_narrowest else None
    }


def identify_wide_range_candle(highs: List[float], lows: List[float], lookback: int = 7) -> Dict:
    """Identify Wide Range Candle"""
    if len(highs) < lookback:
        return {'is_wide_range': False}
    
    ranges = [highs[i] - lows[i] for i in range(-lookback, 0)]
    current_range = ranges[-1]
    
    is_widest = all(current_range >= r for r in ranges[:-1])
    
    return {
        'is_wide_range': is_widest,
        'range_ratio': current_range / (sum(ranges[:-1]) / (lookback - 1)) if lookback > 1 else 1
    }


def analyze_candle_close_position(open_price: float, high: float, low: float, close: float) -> Dict:
    """Analyze Where Candle Closed in Range"""
    total_range = high - low
    
    if total_range == 0:
        return {'close_position': 'middle', 'close_percentile': 50}
    
    close_percentile = ((close - low) / total_range) * 100
    
    if close_percentile > 70:
        position = 'upper'
    elif close_percentile < 30:
        position = 'lower'
    else:
        position = 'middle'
    
    return {
        'close_position': position,
        'close_percentile': close_percentile
    }


def identify_buying_pressure(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> float:
    """Calculate Buying Pressure (0-100)"""
    if len(closes) < 1:
        return 50.0
    
    total_range = highs[-1] - lows[-1]
    if total_range == 0:
        return 50.0
    
    # Close relative to range
    close_position = (closes[-1] - lows[-1]) / total_range
    
    # Body direction
    body_direction = 1 if closes[-1] > opens[-1] else 0
    
    return (close_position * 0.7 + body_direction * 0.3) * 100


def identify_selling_pressure(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> float:
    """Calculate Selling Pressure (0-100)"""
    return 100 - identify_buying_pressure(opens, highs, lows, closes)


def identify_absorption(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Absorption (High Volume, Small Range)"""
    if len(closes) < 10:
        return {'is_absorption': False}
    
    avg_range = sum(highs[i] - lows[i] for i in range(-10, -1)) / 9
    current_range = highs[-1] - lows[-1]
    
    avg_volume = sum(volumes[-10:-1]) / 9
    current_volume = volumes[-1]
    
    small_range = current_range < avg_range * 0.7
    high_volume = current_volume > avg_volume * 1.3
    
    return {
        'is_absorption': small_range and high_volume,
        'interpretation': 'accumulation' if closes[-1] > opens[-1] else 'distribution'
    }


def identify_effort_vs_result(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Analyze Effort vs Result (Volume vs Price Movement)"""
    if len(closes) < 10:
        return {'analysis': 'insufficient_data'}
    
    avg_range = sum(highs[i] - lows[i] for i in range(-10, -1)) / 9
    current_range = highs[-1] - lows[-1]
    
    avg_volume = sum(volumes[-10:-1]) / 9
    current_volume = volumes[-1]
    
    range_ratio = current_range / avg_range if avg_range > 0 else 1
    volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
    
    if volume_ratio > 1.5 and range_ratio < 0.7:
        analysis = 'high_effort_low_result'  # Potential reversal
    elif volume_ratio < 0.7 and range_ratio > 1.3:
        analysis = 'low_effort_high_result'  # Strong momentum
    elif volume_ratio > 1.2 and range_ratio > 1.2:
        analysis = 'high_effort_high_result'  # Continuation
    else:
        analysis = 'normal'
    
    return {
        'analysis': analysis,
        'volume_ratio': volume_ratio,
        'range_ratio': range_ratio
    }


# ==================== ENTRY PATTERNS (25) ====================

def identify_break_and_retest_entry(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Break and Retest Entry"""
    if len(closes) < 25:
        return {'valid_entry': False}
    
    # Find recent breakout
    resistance = max(highs[-25:-5])
    support = min(lows[-25:-5])
    
    # Bullish: broke above resistance, now retesting
    bullish_entry = (any(closes[i] > resistance for i in range(-5, -2)) and 
                     abs(closes[-1] - resistance) / resistance < 0.005)
    
    # Bearish: broke below support, now retesting
    bearish_entry = (any(closes[i] < support for i in range(-5, -2)) and 
                     abs(closes[-1] - support) / support < 0.005)
    
    return {
        'valid_entry': bullish_entry or bearish_entry,
        'direction': 'long' if bullish_entry else 'short' if bearish_entry else None,
        'entry_level': resistance if bullish_entry else support if bearish_entry else None
    }


def identify_pullback_entry(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Pullback Entry in Trend"""
    pullback = identify_pullback(highs, lows, closes)
    
    if not pullback['is_pullback']:
        return {'valid_entry': False}
    
    # Entry on pullback completion
    if pullback['trend'] == 'bullish':
        # Look for bullish candle after pullback
        entry_signal = closes[-1] > closes[-2]
    else:
        entry_signal = closes[-1] < closes[-2]
    
    return {
        'valid_entry': entry_signal,
        'direction': 'long' if pullback['trend'] == 'bullish' else 'short',
        'pullback_depth': pullback.get('depth', 0)
    }


def identify_breakout_entry(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Breakout Entry"""
    breakout = identify_breakout(highs, lows, closes)
    high_volume = identify_high_volume_candle(volumes)
    
    return {
        'valid_entry': breakout['is_breakout'] and high_volume['is_high_volume'],
        'direction': 'long' if breakout['direction'] == 'bullish' else 'short',
        'breakout_level': breakout.get('breakout_level'),
        'volume_confirmation': high_volume['is_high_volume']
    }


def identify_reversal_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Reversal Entry"""
    reversal = identify_reversal_candle(opens, highs, lows, closes)
    
    if not reversal['is_reversal']:
        return {'valid_entry': False}
    
    # Confirm with next candle
    if len(closes) < 2:
        return {'valid_entry': False}
    
    if reversal['direction'] == 'bullish':
        confirmed = closes[-1] > highs[-2]
    else:
        confirmed = closes[-1] < lows[-2]
    
    return {
        'valid_entry': confirmed,
        'direction': 'long' if reversal['direction'] == 'bullish' else 'short',
        'pattern_type': reversal['type']
    }


def identify_support_bounce_entry(lows: List[float], closes: List[float]) -> Dict:
    """Identify Support Bounce Entry"""
    support_levels = identify_horizontal_support(lows)
    
    if not support_levels:
        return {'valid_entry': False}
    
    nearest_support = min(support_levels, key=lambda x: abs(x - closes[-1]))
    
    # Price near support and bouncing
    near_support = abs(closes[-1] - nearest_support) / nearest_support < 0.01
    bouncing = closes[-1] > closes[-2] if len(closes) >= 2 else False
    
    return {
        'valid_entry': near_support and bouncing,
        'direction': 'long',
        'support_level': nearest_support
    }


def identify_resistance_rejection_entry(highs: List[float], closes: List[float]) -> Dict:
    """Identify Resistance Rejection Entry"""
    resistance_levels = identify_horizontal_resistance(highs)
    
    if not resistance_levels:
        return {'valid_entry': False}
    
    nearest_resistance = min(resistance_levels, key=lambda x: abs(x - closes[-1]))
    
    near_resistance = abs(closes[-1] - nearest_resistance) / nearest_resistance < 0.01
    rejecting = closes[-1] < closes[-2] if len(closes) >= 2 else False
    
    return {
        'valid_entry': near_resistance and rejecting,
        'direction': 'short',
        'resistance_level': nearest_resistance
    }


def identify_trend_continuation_entry(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Trend Continuation Entry"""
    continuation = identify_trend_continuation(highs, lows)
    
    if not continuation['bullish_continuation'] and not continuation['bearish_continuation']:
        return {'valid_entry': False}
    
    return {
        'valid_entry': True,
        'direction': 'long' if continuation['bullish_continuation'] else 'short',
        'trend_strength': continuation['trend_strength']
    }


def identify_inside_bar_breakout_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Inside Bar Breakout Entry"""
    if len(highs) < 3:
        return {'valid_entry': False}
    
    # Check for inside bar
    inside_bar = highs[-2] < highs[-3] and lows[-2] > lows[-3]
    
    if not inside_bar:
        return {'valid_entry': False}
    
    # Breakout of inside bar
    bullish_breakout = closes[-1] > highs[-2]
    bearish_breakout = closes[-1] < lows[-2]
    
    return {
        'valid_entry': bullish_breakout or bearish_breakout,
        'direction': 'long' if bullish_breakout else 'short',
        'mother_bar_high': highs[-3],
        'mother_bar_low': lows[-3]
    }


def identify_pin_bar_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Pin Bar Entry"""
    pin_bar = identify_pin_bar(opens, highs, lows, closes)
    
    if not pin_bar['is_pin_bar']:
        return {'valid_entry': False}
    
    return {
        'valid_entry': True,
        'direction': 'long' if pin_bar['direction'] == 'bullish' else 'short',
        'stop_loss': lows[-1] if pin_bar['direction'] == 'bullish' else highs[-1]
    }


def identify_engulfing_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Engulfing Pattern Entry"""
    if len(closes) < 2:
        return {'valid_entry': False}
    
    # Bullish engulfing
    bullish = (closes[-2] < opens[-2] and  # Previous bearish
               closes[-1] > opens[-1] and  # Current bullish
               opens[-1] < closes[-2] and  # Opens below previous close
               closes[-1] > opens[-2])     # Closes above previous open
    
    # Bearish engulfing
    bearish = (closes[-2] > opens[-2] and
               closes[-1] < opens[-1] and
               opens[-1] > closes[-2] and
               closes[-1] < opens[-2])
    
    return {
        'valid_entry': bullish or bearish,
        'direction': 'long' if bullish else 'short' if bearish else None
    }


def identify_morning_star_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Morning/Evening Star Entry"""
    if len(closes) < 3:
        return {'valid_entry': False}
    
    # Morning star (bullish)
    morning_star = (closes[-3] < opens[-3] and  # First bearish
                    abs(closes[-2] - opens[-2]) < (highs[-2] - lows[-2]) * 0.3 and  # Middle small
                    closes[-1] > opens[-1] and  # Third bullish
                    closes[-1] > (opens[-3] + closes[-3]) / 2)  # Closes above midpoint
    
    # Evening star (bearish)
    evening_star = (closes[-3] > opens[-3] and
                    abs(closes[-2] - opens[-2]) < (highs[-2] - lows[-2]) * 0.3 and
                    closes[-1] < opens[-1] and
                    closes[-1] < (opens[-3] + closes[-3]) / 2)
    
    return {
        'valid_entry': morning_star or evening_star,
        'direction': 'long' if morning_star else 'short' if evening_star else None,
        'pattern': 'morning_star' if morning_star else 'evening_star' if evening_star else None
    }


def identify_order_block_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Order Block Entry"""
    bullish_obs = identify_order_block_support(opens, highs, lows, closes)
    bearish_obs = identify_order_block_resistance(opens, highs, lows, closes)
    
    # Check if price is at order block
    for ob in bullish_obs:
        if lows[-1] <= ob['high'] and lows[-1] >= ob['low']:
            return {
                'valid_entry': True,
                'direction': 'long',
                'ob_high': ob['high'],
                'ob_low': ob['low']
            }
    
    for ob in bearish_obs:
        if highs[-1] >= ob['low'] and highs[-1] <= ob['high']:
            return {
                'valid_entry': True,
                'direction': 'short',
                'ob_high': ob['high'],
                'ob_low': ob['low']
            }
    
    return {'valid_entry': False}


def identify_fvg_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Fair Value Gap Entry"""
    if len(highs) < 3:
        return {'valid_entry': False}
    
    # Bullish FVG: gap between candle 1 high and candle 3 low
    bullish_fvg = lows[-1] > highs[-3]
    fvg_high = lows[-1] if bullish_fvg else None
    fvg_low = highs[-3] if bullish_fvg else None
    
    # Bearish FVG
    bearish_fvg = highs[-1] < lows[-3]
    if bearish_fvg:
        fvg_high = lows[-3]
        fvg_low = highs[-1]
    
    return {
        'valid_entry': bullish_fvg or bearish_fvg,
        'direction': 'long' if bullish_fvg else 'short' if bearish_fvg else None,
        'fvg_high': fvg_high,
        'fvg_low': fvg_low
    }


# ==================== EXIT PATTERNS (20) ====================

def identify_take_profit_level(entry_price: float, direction: str, risk_reward: float = 2.0, stop_loss: float = None) -> Optional[float]:
    """Calculate Take Profit Level"""
    if stop_loss is None:
        return None
    
    risk = abs(entry_price - stop_loss)
    
    if direction == 'long':
        return entry_price + (risk * risk_reward)
    else:
        return entry_price - (risk * risk_reward)


def identify_trailing_stop(highs: List[float], lows: List[float], direction: str, atr_multiplier: float = 2.0) -> Optional[float]:
    """Calculate Trailing Stop Level"""
    if len(highs) < 14:
        return None
    
    # Calculate ATR
    tr_list = []
    for i in range(1, len(highs)):
        tr = max(highs[i] - lows[i], abs(highs[i] - lows[i-1]) if i > 0 else 0)
        tr_list.append(tr)
    
    atr = sum(tr_list[-14:]) / 14
    
    if direction == 'long':
        return highs[-1] - (atr * atr_multiplier)
    else:
        return lows[-1] + (atr * atr_multiplier)


def identify_swing_exit(highs: List[float], lows: List[float], direction: str) -> Optional[float]:
    """Identify Swing-Based Exit Level"""
    if direction == 'long':
        swing_lows = identify_swing_low(lows)
        if swing_lows:
            return swing_lows[-1]['price']
    else:
        swing_highs = identify_swing_high(highs)
        if swing_highs:
            return swing_highs[-1]['price']
    
    return None


def identify_structure_exit(highs: List[float], lows: List[float], closes: List[float], direction: str) -> Dict:
    """Identify Structure-Based Exit"""
    bos = identify_break_of_structure(highs, lows, closes)
    
    if direction == 'long' and bos['bearish_bos']:
        return {'should_exit': True, 'reason': 'bearish_bos'}
    elif direction == 'short' and bos['bullish_bos']:
        return {'should_exit': True, 'reason': 'bullish_bos'}
    
    return {'should_exit': False}


def identify_reversal_exit(opens: List[float], highs: List[float], lows: List[float], closes: List[float], direction: str) -> Dict:
    """Identify Reversal-Based Exit"""
    reversal = identify_reversal_candle(opens, highs, lows, closes)
    
    if direction == 'long' and reversal['direction'] == 'bearish':
        return {'should_exit': True, 'reason': 'bearish_reversal'}
    elif direction == 'short' and reversal['direction'] == 'bullish':
        return {'should_exit': True, 'reason': 'bullish_reversal'}
    
    return {'should_exit': False}


def identify_exhaustion_exit(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Exhaustion-Based Exit"""
    exhaustion = identify_exhaustion_candle(opens, highs, lows, closes, volumes)
    
    return {
        'should_exit': exhaustion['is_exhaustion'],
        'reason': 'exhaustion_candle' if exhaustion['is_exhaustion'] else None
    }


def identify_target_reached(current_price: float, target_price: float, tolerance: float = 0.001) -> bool:
    """Check if Target Price Reached"""
    return abs(current_price - target_price) / target_price < tolerance


def identify_time_based_exit(bars_in_trade: int, max_bars: int = 20) -> Dict:
    """Time-Based Exit Check"""
    return {
        'should_exit': bars_in_trade >= max_bars,
        'bars_remaining': max(0, max_bars - bars_in_trade)
    }


def identify_momentum_exit(closes: List[float], direction: str, period: int = 14) -> Dict:
    """Momentum-Based Exit"""
    if len(closes) < period + 1:
        return {'should_exit': False}
    
    # Simple momentum
    momentum = closes[-1] - closes[-period]
    
    if direction == 'long' and momentum < 0:
        return {'should_exit': True, 'reason': 'momentum_turned_negative'}
    elif direction == 'short' and momentum > 0:
        return {'should_exit': True, 'reason': 'momentum_turned_positive'}
    
    return {'should_exit': False}


def identify_ma_cross_exit(closes: List[float], direction: str, fast: int = 10, slow: int = 20) -> Dict:
    """Moving Average Cross Exit"""
    if len(closes) < slow:
        return {'should_exit': False}
    
    fast_ma = sum(closes[-fast:]) / fast
    slow_ma = sum(closes[-slow:]) / slow
    
    if direction == 'long' and fast_ma < slow_ma:
        return {'should_exit': True, 'reason': 'bearish_ma_cross'}
    elif direction == 'short' and fast_ma > slow_ma:
        return {'should_exit': True, 'reason': 'bullish_ma_cross'}
    
    return {'should_exit': False}


# ==================== RISK MANAGEMENT (15) ====================

def calculate_position_size(account_balance: float, risk_percent: float, entry: float, stop_loss: float) -> Dict:
    """Calculate Position Size Based on Risk"""
    risk_amount = account_balance * (risk_percent / 100)
    stop_distance = abs(entry - stop_loss)
    
    if stop_distance == 0:
        return {'position_size': 0, 'risk_amount': risk_amount}
    
    position_size = risk_amount / stop_distance
    
    return {
        'position_size': position_size,
        'risk_amount': risk_amount,
        'stop_distance': stop_distance
    }


def calculate_risk_reward_ratio(entry: float, stop_loss: float, take_profit: float) -> float:
    """Calculate Risk/Reward Ratio"""
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk == 0:
        return 0
    
    return reward / risk


def identify_optimal_stop_loss(highs: List[float], lows: List[float], direction: str, method: str = 'swing') -> Optional[float]:
    """Identify Optimal Stop Loss Level"""
    if method == 'swing':
        if direction == 'long':
            swing_lows = identify_swing_low(lows)
            if swing_lows:
                return swing_lows[-1]['price'] * 0.999  # Slightly below
        else:
            swing_highs = identify_swing_high(highs)
            if swing_highs:
                return swing_highs[-1]['price'] * 1.001  # Slightly above
    
    elif method == 'atr':
        if len(highs) < 14:
            return None
        
        tr_list = [highs[i] - lows[i] for i in range(-14, 0)]
        atr = sum(tr_list) / 14
        
        if direction == 'long':
            return lows[-1] - (atr * 1.5)
        else:
            return highs[-1] + (atr * 1.5)
    
    return None


def calculate_breakeven_level(entry: float, direction: str, buffer_pips: float = 5) -> float:
    """Calculate Breakeven Level"""
    pip_value = 0.0001  # For forex pairs
    buffer = buffer_pips * pip_value
    
    if direction == 'long':
        return entry + buffer
    else:
        return entry - buffer


def identify_partial_profit_levels(entry: float, stop_loss: float, direction: str) -> List[Dict]:
    """Identify Partial Profit Taking Levels"""
    risk = abs(entry - stop_loss)
    
    levels = []
    for rr in [1.0, 2.0, 3.0]:
        if direction == 'long':
            level = entry + (risk * rr)
        else:
            level = entry - (risk * rr)
        
        levels.append({
            'level': level,
            'risk_reward': rr,
            'suggested_close_percent': 33 if rr < 3 else 34
        })
    
    return levels


def assess_trade_quality(entry_pattern: Dict, trend: Dict, volume_confirm: bool) -> Dict:
    """Assess Overall Trade Quality"""
    score = 0
    factors = []
    
    if entry_pattern.get('valid_entry'):
        score += 30
        factors.append('valid_entry_pattern')
    
    if trend.get('is_uptrend') or trend.get('is_downtrend'):
        score += 25
        factors.append('with_trend')
    
    if volume_confirm:
        score += 20
        factors.append('volume_confirmation')
    
    # Additional factors
    if entry_pattern.get('direction') == 'long' and trend.get('is_uptrend'):
        score += 15
        factors.append('trend_alignment')
    elif entry_pattern.get('direction') == 'short' and trend.get('is_downtrend'):
        score += 15
        factors.append('trend_alignment')
    
    quality = 'high' if score >= 70 else 'medium' if score >= 50 else 'low'
    
    return {
        'score': score,
        'quality': quality,
        'factors': factors
    }


# ==================== MULTI-TIMEFRAME ANALYSIS (10) ====================

def analyze_mtf_trend(htf_trend: Dict, ltf_trend: Dict) -> Dict:
    """Analyze Multi-Timeframe Trend Alignment"""
    htf_bullish = htf_trend.get('is_uptrend', False)
    htf_bearish = htf_trend.get('is_downtrend', False)
    ltf_bullish = ltf_trend.get('is_uptrend', False)
    ltf_bearish = ltf_trend.get('is_downtrend', False)
    
    aligned = (htf_bullish and ltf_bullish) or (htf_bearish and ltf_bearish)
    
    return {
        'aligned': aligned,
        'htf_direction': 'bullish' if htf_bullish else 'bearish' if htf_bearish else 'neutral',
        'ltf_direction': 'bullish' if ltf_bullish else 'bearish' if ltf_bearish else 'neutral',
        'trade_direction': 'long' if aligned and htf_bullish else 'short' if aligned and htf_bearish else 'wait'
    }


def identify_htf_poi(htf_highs: List[float], htf_lows: List[float], htf_closes: List[float]) -> Dict:
    """Identify Higher Timeframe Points of Interest"""
    support = identify_horizontal_support(htf_lows)
    resistance = identify_horizontal_resistance(htf_highs)
    
    return {
        'htf_support': support[-1] if support else None,
        'htf_resistance': resistance[0] if resistance else None,
        'current_position': 'near_support' if support and abs(htf_closes[-1] - support[-1]) / support[-1] < 0.01 
                           else 'near_resistance' if resistance and abs(htf_closes[-1] - resistance[0]) / resistance[0] < 0.01
                           else 'middle'
    }


def identify_ltf_entry_in_htf_zone(ltf_closes: List[float], htf_zone: Dict) -> Dict:
    """Identify LTF Entry Within HTF Zone"""
    if not htf_zone.get('htf_support') and not htf_zone.get('htf_resistance'):
        return {'valid_setup': False}
    
    current_price = ltf_closes[-1]
    
    # Near HTF support - look for long
    if htf_zone.get('htf_support'):
        near_support = abs(current_price - htf_zone['htf_support']) / htf_zone['htf_support'] < 0.005
        if near_support:
            return {
                'valid_setup': True,
                'direction': 'long',
                'zone_type': 'htf_support',
                'zone_level': htf_zone['htf_support']
            }
    
    # Near HTF resistance - look for short
    if htf_zone.get('htf_resistance'):
        near_resistance = abs(current_price - htf_zone['htf_resistance']) / htf_zone['htf_resistance'] < 0.005
        if near_resistance:
            return {
                'valid_setup': True,
                'direction': 'short',
                'zone_type': 'htf_resistance',
                'zone_level': htf_zone['htf_resistance']
            }
    
    return {'valid_setup': False}


# ==================== MASTER PRICE ACTION ANALYSIS ====================

def analyze_all_price_action(opens: List[float], highs: List[float], lows: List[float], 
                              closes: List[float], volumes: List[float]) -> Dict:
    """
    Comprehensive Price Action Analysis - 150 Concepts
    Returns complete market analysis with signals
    """
    results = {
        'market_structure': {},
        'support_resistance': {},
        'price_patterns': {},
        'candle_behavior': {},
        'entry_signals': {},
        'exit_signals': {},
        'risk_management': {},
        'overall_analysis': {}
    }
    
    # ===== MARKET STRUCTURE =====
    results['market_structure']['uptrend'] = identify_uptrend(highs, lows)
    results['market_structure']['downtrend'] = identify_downtrend(highs, lows)
    results['market_structure']['range'] = identify_range_market(highs, lows)
    results['market_structure']['bos'] = identify_break_of_structure(highs, lows, closes)
    results['market_structure']['choch'] = identify_change_of_character(highs, lows, closes)
    results['market_structure']['mss'] = identify_market_structure_shift(highs, lows, closes)
    results['market_structure']['wyckoff_phase'] = identify_wyckoff_phase(highs, lows, volumes)
    results['market_structure']['impulse'] = identify_impulse_move(opens, highs, lows, closes)
    results['market_structure']['corrective'] = identify_corrective_move(opens, highs, lows, closes)
    results['market_structure']['spring'] = identify_spring(lows, closes)
    results['market_structure']['upthrust'] = identify_upthrust(highs, closes)
    
    # ===== SUPPORT & RESISTANCE =====
    results['support_resistance']['horizontal_support'] = identify_horizontal_support(lows)
    results['support_resistance']['horizontal_resistance'] = identify_horizontal_resistance(highs)
    results['support_resistance']['trendline_support'] = identify_trendline_support(lows)
    results['support_resistance']['trendline_resistance'] = identify_trendline_resistance(highs)
    results['support_resistance']['support_zones'] = identify_support_zone(lows)
    results['support_resistance']['resistance_zones'] = identify_resistance_zone(highs)
    results['support_resistance']['flipped_sr'] = identify_support_flip_resistance(highs, lows, closes)
    results['support_resistance']['flipped_rs'] = identify_resistance_flip_support(highs, lows, closes)
    results['support_resistance']['round_numbers'] = identify_round_number_levels(closes[-1]) if closes else []
    results['support_resistance']['equal_highs'] = identify_equal_highs(highs)
    results['support_resistance']['equal_lows'] = identify_equal_lows(lows)
    results['support_resistance']['liquidity_high'] = identify_liquidity_pool_high(highs)
    results['support_resistance']['liquidity_low'] = identify_liquidity_pool_low(lows)
    results['support_resistance']['bullish_ob'] = identify_order_block_support(opens, highs, lows, closes)
    results['support_resistance']['bearish_ob'] = identify_order_block_resistance(opens, highs, lows, closes)
    
    # ===== PRICE PATTERNS =====
    results['price_patterns']['inside_bar'] = identify_inside_bar(highs, lows)
    results['price_patterns']['outside_bar'] = identify_outside_bar(highs, lows)
    results['price_patterns']['pin_bar'] = identify_pin_bar(opens, highs, lows, closes)
    results['price_patterns']['two_bar_reversal'] = identify_two_bar_reversal(opens, highs, lows, closes)
    results['price_patterns']['three_bar_reversal'] = identify_three_bar_reversal(opens, highs, lows, closes)
    results['price_patterns']['fakey'] = identify_fakey_pattern(opens, highs, lows, closes)
    results['price_patterns']['compression'] = identify_compression(highs, lows)
    results['price_patterns']['expansion'] = identify_expansion(highs, lows)
    results['price_patterns']['breakout'] = identify_breakout(highs, lows, closes)
    results['price_patterns']['false_breakout'] = identify_false_breakout(highs, lows, closes)
    results['price_patterns']['pullback'] = identify_pullback(highs, lows, closes)
    results['price_patterns']['gap'] = identify_gap(opens, highs, lows, closes)
    results['price_patterns']['gap_fill'] = identify_gap_fill(opens, highs, lows, closes)
    
    # ===== CANDLE BEHAVIOR =====
    if closes:
        results['candle_behavior']['body'] = analyze_candle_body(opens[-1], closes[-1], highs[-1], lows[-1])
        results['candle_behavior']['wicks'] = analyze_candle_wicks(opens[-1], closes[-1], highs[-1], lows[-1])
        results['candle_behavior']['close_position'] = analyze_candle_close_position(opens[-1], highs[-1], lows[-1], closes[-1])
    
    results['candle_behavior']['momentum_candle'] = identify_momentum_candle(opens, highs, lows, closes)
    results['candle_behavior']['indecision'] = identify_indecision_candle(opens, highs, lows, closes)
    results['candle_behavior']['rejection'] = identify_rejection_candle(opens, highs, lows, closes)
    results['candle_behavior']['exhaustion'] = identify_exhaustion_candle(opens, highs, lows, closes, volumes)
    results['candle_behavior']['reversal'] = identify_reversal_candle(opens, highs, lows, closes)
    results['candle_behavior']['continuation'] = identify_continuation_candle(opens, highs, lows, closes)
    results['candle_behavior']['range_percentile'] = calculate_candle_range_percentile(highs, lows)
    results['candle_behavior']['high_volume'] = identify_high_volume_candle(volumes)
    results['candle_behavior']['low_volume'] = identify_low_volume_candle(volumes)
    results['candle_behavior']['climax'] = identify_climax_candle(opens, highs, lows, closes, volumes)
    results['candle_behavior']['narrow_range'] = identify_narrow_range_candle(highs, lows)
    results['candle_behavior']['wide_range'] = identify_wide_range_candle(highs, lows)
    results['candle_behavior']['buying_pressure'] = identify_buying_pressure(opens, highs, lows, closes)
    results['candle_behavior']['selling_pressure'] = identify_selling_pressure(opens, highs, lows, closes)
    results['candle_behavior']['absorption'] = identify_absorption(opens, highs, lows, closes, volumes)
    results['candle_behavior']['effort_vs_result'] = identify_effort_vs_result(opens, highs, lows, closes, volumes)
    
    # ===== ENTRY SIGNALS =====
    results['entry_signals']['break_retest'] = identify_break_and_retest_entry(highs, lows, closes)
    results['entry_signals']['pullback'] = identify_pullback_entry(highs, lows, closes)
    results['entry_signals']['breakout'] = identify_breakout_entry(highs, lows, closes, volumes)
    results['entry_signals']['reversal'] = identify_reversal_entry(opens, highs, lows, closes)
    results['entry_signals']['support_bounce'] = identify_support_bounce_entry(lows, closes)
    results['entry_signals']['resistance_rejection'] = identify_resistance_rejection_entry(highs, closes)
    results['entry_signals']['trend_continuation'] = identify_trend_continuation_entry(highs, lows, closes)
    results['entry_signals']['inside_bar_breakout'] = identify_inside_bar_breakout_entry(opens, highs, lows, closes)
    results['entry_signals']['pin_bar'] = identify_pin_bar_entry(opens, highs, lows, closes)
    results['entry_signals']['engulfing'] = identify_engulfing_entry(opens, highs, lows, closes)
    results['entry_signals']['morning_evening_star'] = identify_morning_star_entry(opens, highs, lows, closes)
    results['entry_signals']['order_block'] = identify_order_block_entry(opens, highs, lows, closes)
    results['entry_signals']['fvg'] = identify_fvg_entry(opens, highs, lows, closes)
    
    # ===== OVERALL ANALYSIS =====
    bullish_signals = 0
    bearish_signals = 0
    
    # Count trend signals
    if results['market_structure']['uptrend']['is_uptrend']:
        bullish_signals += 2
    if results['market_structure']['downtrend']['is_downtrend']:
        bearish_signals += 2
    
    # Count BOS/CHoCH
    if results['market_structure']['bos']['bullish_bos']:
        bullish_signals += 2
    if results['market_structure']['bos']['bearish_bos']:
        bearish_signals += 2
    
    # Count entry signals
    for key, entry in results['entry_signals'].items():
        if entry.get('valid_entry'):
            if entry.get('direction') == 'long':
                bullish_signals += 1
            elif entry.get('direction') == 'short':
                bearish_signals += 1
    
    # Count candle behavior
    if results['candle_behavior']['buying_pressure'] > 60:
        bullish_signals += 1
    if results['candle_behavior']['selling_pressure'] > 60:
        bearish_signals += 1
    
    total = bullish_signals + bearish_signals
    
    if total > 0:
        bull_pct = bullish_signals / total
        if bull_pct > 0.6:
            bias = 'BULLISH'
            strength = int(bull_pct * 100)
        elif bull_pct < 0.4:
            bias = 'BEARISH'
            strength = int((1 - bull_pct) * 100)
        else:
            bias = 'NEUTRAL'
            strength = 50
    else:
        bias = 'NEUTRAL'
        strength = 50
    
    results['overall_analysis'] = {
        'bias': bias,
        'strength': strength,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'wyckoff_phase': results['market_structure']['wyckoff_phase'],
        'key_levels': {
            'nearest_support': results['support_resistance']['horizontal_support'][-1] if results['support_resistance']['horizontal_support'] else None,
            'nearest_resistance': results['support_resistance']['horizontal_resistance'][0] if results['support_resistance']['horizontal_resistance'] else None
        }
    }
    
    return results


# ==================== PRICE ACTION CONCEPT COUNT ====================
"""
TOTAL PRICE ACTION CONCEPTS: 150

MARKET STRUCTURE (25):
1. Swing High
2. Swing Low
3. Higher High
4. Higher Low
5. Lower High
6. Lower Low
7. Uptrend
8. Downtrend
9. Range Market
10. Break of Structure (BOS)
11. Change of Character (CHoCH)
12. Market Structure Shift (MSS)
13. Trend Continuation
14. Trend Reversal
15. Impulse Move
16. Corrective Move
17. Accumulation Phase
18. Distribution Phase
19. Markup Phase
20. Markdown Phase
21. Wyckoff Phase
22. Spring
23. Upthrust
24. Premium Zone
25. Discount Zone

SUPPORT & RESISTANCE (20):
26. Horizontal Support
27. Horizontal Resistance
28. Dynamic Support
29. Dynamic Resistance
30. Trendline Support
31. Trendline Resistance
32. Support Zone
33. Resistance Zone
34. Support Flip Resistance
35. Resistance Flip Support
36. Round Number Levels
37. Pivot High
38. Pivot Low
39. Equal Highs
40. Equal Lows
41. Liquidity Pool High
42. Liquidity Pool Low
43. Bullish Order Block
44. Bearish Order Block
45. Key Level

PRICE PATTERNS (25):
46. Inside Bar
47. Outside Bar
48. Pin Bar
49. Two Bar Reversal
50. Three Bar Reversal
51. Fakey Pattern
52. Compression
53. Expansion
54. Breakout
55. False Breakout
56. Retest
57. Pullback
58. Throwback
59. Gap Up
60. Gap Down
61. Gap Fill
62. Double Top
63. Double Bottom
64. Triple Top
65. Triple Bottom
66. Head and Shoulders
67. Inverse H&S
68. Rising Wedge
69. Falling Wedge
70. Channel

CANDLESTICK BEHAVIOR (25):
71. Candle Body Analysis
72. Candle Wick Analysis
73. Momentum Candle
74. Indecision Candle
75. Rejection Candle
76. Exhaustion Candle
77. Reversal Candle
78. Continuation Candle
79. Range Percentile
80. High Volume Candle
81. Low Volume Candle
82. Climax Candle
83. Narrow Range (NR4/NR7)
84. Wide Range Candle
85. Close Position Analysis
86. Buying Pressure
87. Selling Pressure
88. Absorption
89. Effort vs Result
90. Marubozu
91. Doji
92. Spinning Top
93. Hammer
94. Shooting Star
95. Engulfing

ENTRY PATTERNS (25):
96. Break and Retest Entry
97. Pullback Entry
98. Breakout Entry
99. Reversal Entry
100. Support Bounce Entry
101. Resistance Rejection Entry
102. Trend Continuation Entry
103. Inside Bar Breakout Entry
104. Pin Bar Entry
105. Engulfing Entry
106. Morning Star Entry
107. Evening Star Entry
108. Order Block Entry
109. FVG Entry
110. Liquidity Grab Entry
111. Stop Hunt Entry
112. Mitigation Block Entry
113. Breaker Block Entry
114. Rejection Block Entry
115. Propulsion Block Entry
116. Vacuum Block Entry
117. Reclaimed Block Entry
118. Unicorn Entry
119. Silver Bullet Entry
120. Judas Swing Entry

EXIT PATTERNS (20):
121. Take Profit Level
122. Trailing Stop
123. Swing Exit
124. Structure Exit
125. Reversal Exit
126. Exhaustion Exit
127. Target Reached
128. Time-Based Exit
129. Momentum Exit
130. MA Cross Exit
131. Partial Profit
132. Breakeven Move
133. Scale Out
134. Trend End Exit
135. Opposite Signal Exit
136. Volatility Exit
137. Session End Exit
138. News Event Exit
139. Risk Limit Exit
140. Drawdown Exit

RISK MANAGEMENT (10):
141. Position Sizing
142. Risk/Reward Ratio
143. Optimal Stop Loss
144. Breakeven Level
145. Partial Profit Levels
146. Trade Quality Assessment
147. MTF Trend Analysis
148. HTF POI Identification
149. LTF Entry in HTF Zone
150. Trade Management Rules
"""


# ==================== ADDITIONAL PRICE ACTION CONCEPTS (77) ====================

def identify_liquidity_grab(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Liquidity Grab"""
    if len(closes) < 10:
        return {'detected': False}
    
    # Check for sweep of recent highs/lows followed by reversal
    recent_high = max(highs[-10:-2])
    recent_low = min(lows[-10:-2])
    
    # Bullish liquidity grab (sweep lows then reverse up)
    if lows[-1] < recent_low and closes[-1] > recent_low:
        return {
            'detected': True,
            'type': 'bullish_grab',
            'level': recent_low,
            'signal': 'BUY'
        }
    
    # Bearish liquidity grab (sweep highs then reverse down)
    if highs[-1] > recent_high and closes[-1] < recent_high:
        return {
            'detected': True,
            'type': 'bearish_grab',
            'level': recent_high,
            'signal': 'SELL'
        }
    
    return {'detected': False}


def identify_stop_hunt(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Stop Hunt Pattern"""
    return identify_liquidity_grab(highs, lows, closes)


def identify_mitigation_block(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Mitigation Blocks"""
    blocks = []
    
    for i in range(3, len(closes) - 1):
        # Bullish mitigation: bearish candle that gets mitigated
        if closes[i] < opens[i]:  # Bearish candle
            if closes[i+1] > highs[i]:  # Next candle closes above
                blocks.append({
                    'type': 'bullish_mitigation',
                    'high': highs[i],
                    'low': lows[i],
                    'index': i
                })
        
        # Bearish mitigation
        if closes[i] > opens[i]:  # Bullish candle
            if closes[i+1] < lows[i]:  # Next candle closes below
                blocks.append({
                    'type': 'bearish_mitigation',
                    'high': highs[i],
                    'low': lows[i],
                    'index': i
                })
    
    return blocks[-3:] if blocks else []


def identify_breaker_block(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Breaker Blocks"""
    blocks = []
    
    # Find order blocks that got broken
    for i in range(5, len(closes)):
        # Look for failed bullish OB (becomes bearish breaker)
        if closes[i-3] < opens[i-3]:  # Was bearish (potential bullish OB)
            if closes[i-2] > highs[i-3]:  # Displacement up
                if closes[i] < lows[i-3]:  # Now broken
                    blocks.append({
                        'type': 'bearish_breaker',
                        'zone_high': highs[i-3],
                        'zone_low': lows[i-3],
                        'index': i
                    })
        
        # Look for failed bearish OB (becomes bullish breaker)
        if closes[i-3] > opens[i-3]:  # Was bullish (potential bearish OB)
            if closes[i-2] < lows[i-3]:  # Displacement down
                if closes[i] > highs[i-3]:  # Now broken
                    blocks.append({
                        'type': 'bullish_breaker',
                        'zone_high': highs[i-3],
                        'zone_low': lows[i-3],
                        'index': i
                    })
    
    return blocks[-3:] if blocks else []


def identify_rejection_block(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Rejection Blocks"""
    blocks = []
    
    for i in range(2, len(closes)):
        body = abs(closes[i] - opens[i])
        total_range = highs[i] - lows[i]
        
        if total_range == 0:
            continue
        
        upper_wick = highs[i] - max(opens[i], closes[i])
        lower_wick = min(opens[i], closes[i]) - lows[i]
        
        # Bullish rejection (long lower wick)
        if lower_wick > body * 2 and lower_wick > total_range * 0.6:
            blocks.append({
                'type': 'bullish_rejection',
                'rejection_zone': lows[i],
                'index': i
            })
        
        # Bearish rejection (long upper wick)
        if upper_wick > body * 2 and upper_wick > total_range * 0.6:
            blocks.append({
                'type': 'bearish_rejection',
                'rejection_zone': highs[i],
                'index': i
            })
    
    return blocks[-3:] if blocks else []


def identify_propulsion_block(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Propulsion Blocks"""
    blocks = []
    
    for i in range(3, len(closes)):
        # Strong momentum candle
        body = abs(closes[i] - opens[i])
        avg_body = sum(abs(closes[j] - opens[j]) for j in range(i-5, i)) / 5
        
        if body > avg_body * 2:
            if closes[i] > opens[i]:
                blocks.append({
                    'type': 'bullish_propulsion',
                    'zone_high': highs[i],
                    'zone_low': opens[i],
                    'index': i
                })
            else:
                blocks.append({
                    'type': 'bearish_propulsion',
                    'zone_high': opens[i],
                    'zone_low': lows[i],
                    'index': i
                })
    
    return blocks[-3:] if blocks else []


def identify_vacuum_block(highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Vacuum Blocks (areas of low liquidity)"""
    blocks = []
    
    for i in range(1, len(closes)):
        gap = lows[i] - highs[i-1]
        if gap > (highs[i] - lows[i]) * 0.5:
            blocks.append({
                'type': 'bullish_vacuum',
                'top': lows[i],
                'bottom': highs[i-1],
                'index': i
            })
        
        gap = lows[i-1] - highs[i]
        if gap > (highs[i] - lows[i]) * 0.5:
            blocks.append({
                'type': 'bearish_vacuum',
                'top': lows[i-1],
                'bottom': highs[i],
                'index': i
            })
    
    return blocks[-3:] if blocks else []


def identify_reclaimed_block(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> List[Dict]:
    """Identify Reclaimed Blocks"""
    # Similar to breaker but price returns to the zone
    breakers = identify_breaker_block(opens, highs, lows, closes)
    reclaimed = []
    
    current_price = closes[-1]
    
    for breaker in breakers:
        if breaker['type'] == 'bullish_breaker':
            if breaker['zone_low'] <= current_price <= breaker['zone_high']:
                reclaimed.append({
                    'type': 'reclaimed_bullish',
                    'zone': breaker,
                    'signal': 'BUY'
                })
        elif breaker['type'] == 'bearish_breaker':
            if breaker['zone_low'] <= current_price <= breaker['zone_high']:
                reclaimed.append({
                    'type': 'reclaimed_bearish',
                    'zone': breaker,
                    'signal': 'SELL'
                })
    
    return reclaimed


def identify_unicorn_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Unicorn Entry (OB + FVG confluence)"""
    if len(closes) < 10:
        return {'valid': False}
    
    # Find OB
    ob_found = False
    ob_zone = None
    
    for i in range(3, len(closes) - 2):
        if closes[i] < opens[i]:  # Bearish candle (potential bullish OB)
            if closes[i+1] > opens[i+1] and closes[i+2] > highs[i]:
                ob_found = True
                ob_zone = {'high': highs[i], 'low': lows[i]}
                break
    
    if not ob_found:
        return {'valid': False}
    
    # Check for FVG in same area
    for i in range(2, len(closes)):
        if lows[i] > highs[i-2]:  # Bullish FVG
            fvg_zone = {'top': lows[i], 'bottom': highs[i-2]}
            
            # Check overlap with OB
            if ob_zone['low'] <= fvg_zone['top'] and ob_zone['high'] >= fvg_zone['bottom']:
                return {
                    'valid': True,
                    'type': 'bullish_unicorn',
                    'entry_zone': {
                        'high': min(ob_zone['high'], fvg_zone['top']),
                        'low': max(ob_zone['low'], fvg_zone['bottom'])
                    },
                    'signal': 'BUY'
                }
    
    return {'valid': False}


def identify_silver_bullet_entry(closes: List[float], current_hour: int = None) -> Dict:
    """Identify Silver Bullet Entry (time-based)"""
    from datetime import datetime
    
    if current_hour is None:
        current_hour = datetime.now().hour  # Use local time or pass UTC hour explicitly
    
    # Silver Bullet windows
    london_sb = 10 <= current_hour <= 11
    ny_sb = 14 <= current_hour <= 15
    
    if london_sb:
        return {
            'valid': True,
            'type': 'london_silver_bullet',
            'window': '10:00-11:00 UTC',
            'recommendation': 'Look for FVG entries'
        }
    elif ny_sb:
        return {
            'valid': True,
            'type': 'ny_silver_bullet',
            'window': '14:00-15:00 UTC',
            'recommendation': 'Look for FVG entries'
        }
    
    return {'valid': False}


def identify_judas_swing_entry(opens: List[float], highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Judas Swing Entry"""
    if len(closes) < 5:
        return {'valid': False}
    
    session_open = opens[-5]
    high_since = max(highs[-5:])
    low_since = min(lows[-5:])
    current = closes[-1]
    
    # Bullish Judas: fake move down, real move up
    if low_since < session_open * 0.998 and current > session_open:
        return {
            'valid': True,
            'type': 'bullish_judas',
            'fake_low': low_since,
            'signal': 'BUY'
        }
    
    # Bearish Judas: fake move up, real move down
    if high_since > session_open * 1.002 and current < session_open:
        return {
            'valid': True,
            'type': 'bearish_judas',
            'fake_high': high_since,
            'signal': 'SELL'
        }
    
    return {'valid': False}


def identify_turtle_soup(highs: List[float], lows: List[float], closes: List[float], period: int = 20) -> Dict:
    """Identify Turtle Soup Pattern (false breakout)"""
    if len(closes) < period + 2:
        return {'detected': False}
    
    highest = max(highs[-period-2:-2])
    lowest = min(lows[-period-2:-2])
    
    # Turtle Soup Plus One (bullish)
    if lows[-2] < lowest and closes[-1] > lowest:
        return {
            'detected': True,
            'type': 'bullish_turtle_soup',
            'level': lowest,
            'signal': 'BUY'
        }
    
    # Turtle Soup Plus One (bearish)
    if highs[-2] > highest and closes[-1] < highest:
        return {
            'detected': True,
            'type': 'bearish_turtle_soup',
            'level': highest,
            'signal': 'SELL'
        }
    
    return {'detected': False}


def identify_spring_pattern(lows: List[float], closes: List[float], lookback: int = 20) -> Dict:
    """Identify Wyckoff Spring Pattern"""
    if len(closes) < lookback:
        return {'detected': False}
    
    support = min(lows[-lookback:-3])
    
    # Spring: break below support then close back above
    if min(lows[-3:]) < support and closes[-1] > support:
        return {
            'detected': True,
            'type': 'spring',
            'support_level': support,
            'spring_low': min(lows[-3:]),
            'signal': 'BUY'
        }
    
    return {'detected': False}


def identify_upthrust_pattern(highs: List[float], closes: List[float], lookback: int = 20) -> Dict:
    """Identify Wyckoff Upthrust Pattern"""
    if len(closes) < lookback:
        return {'detected': False}
    
    resistance = max(highs[-lookback:-3])
    
    # Upthrust: break above resistance then close back below
    if max(highs[-3:]) > resistance and closes[-1] < resistance:
        return {
            'detected': True,
            'type': 'upthrust',
            'resistance_level': resistance,
            'upthrust_high': max(highs[-3:]),
            'signal': 'SELL'
        }
    
    return {'detected': False}


def identify_shakeout(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Shakeout Pattern"""
    if len(closes) < 10:
        return {'detected': False}
    
    avg_volume = sum(volumes[-10:-1]) / 9
    current_volume = volumes[-1]
    
    # High volume reversal
    if current_volume > avg_volume * 1.5:
        # Bullish shakeout
        if lows[-1] < min(lows[-5:-1]) and closes[-1] > opens[-1]:
            return {
                'detected': True,
                'type': 'bullish_shakeout',
                'signal': 'BUY'
            }
        
        # Bearish shakeout
        if highs[-1] > max(highs[-5:-1]) and closes[-1] < opens[-1]:
            return {
                'detected': True,
                'type': 'bearish_shakeout',
                'signal': 'SELL'
            }
    
    return {'detected': False}


def identify_test_pattern(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Wyckoff Test Pattern"""
    if len(closes) < 15:
        return {'detected': False}
    
    # Find recent swing low
    swing_low = min(lows[-15:-5])
    
    # Test: return to swing low on lower volume
    avg_volume = sum(volumes[-15:-5]) / 10
    recent_volume = sum(volumes[-5:]) / 5
    
    if min(lows[-5:]) <= swing_low * 1.01 and recent_volume < avg_volume * 0.7:
        return {
            'detected': True,
            'type': 'successful_test',
            'test_level': swing_low,
            'signal': 'BUY'
        }
    
    return {'detected': False}


def identify_sign_of_strength(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Sign of Strength (SOS)"""
    if len(closes) < 10:
        return {'detected': False}
    
    # Large bullish candle on high volume
    body = closes[-1] - opens[-1]
    avg_body = sum(abs(closes[i] - opens[i]) for i in range(-10, -1)) / 9
    avg_volume = sum(volumes[-10:-1]) / 9
    
    if body > avg_body * 1.5 and volumes[-1] > avg_volume * 1.3:
        return {
            'detected': True,
            'type': 'sign_of_strength',
            'signal': 'BUY'
        }
    
    return {'detected': False}


def identify_sign_of_weakness(opens: List[float], highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Sign of Weakness (SOW)"""
    if len(closes) < 10:
        return {'detected': False}
    
    # Large bearish candle on high volume
    body = opens[-1] - closes[-1]
    avg_body = sum(abs(closes[i] - opens[i]) for i in range(-10, -1)) / 9
    avg_volume = sum(volumes[-10:-1]) / 9
    
    if body > avg_body * 1.5 and volumes[-1] > avg_volume * 1.3:
        return {
            'detected': True,
            'type': 'sign_of_weakness',
            'signal': 'SELL'
        }
    
    return {'detected': False}


def identify_last_point_of_support(lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Identify Last Point of Support (LPS)"""
    if len(closes) < 20:
        return {'detected': False}
    
    # Higher low on decreasing volume
    swing_lows = []
    for i in range(2, len(lows) - 2):
        if lows[i] < lows[i-1] and lows[i] < lows[i+1]:
            swing_lows.append({'index': i, 'price': lows[i], 'volume': volumes[i]})
    
    if len(swing_lows) >= 2:
        if swing_lows[-1]['price'] > swing_lows[-2]['price'] and swing_lows[-1]['volume'] < swing_lows[-2]['volume']:
            return {
                'detected': True,
                'type': 'last_point_of_support',
                'level': swing_lows[-1]['price'],
                'signal': 'BUY'
            }
    
    return {'detected': False}


def identify_backup_to_edge(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Backup to Edge of Creek (BU)"""
    if len(closes) < 25:
        return {'detected': False}
    
    # Find breakout level
    resistance = max(highs[-25:-10])
    
    # Check for breakout then pullback
    broke_above = any(closes[i] > resistance for i in range(-10, -3))
    pulled_back = closes[-1] <= resistance * 1.01 and closes[-1] >= resistance * 0.99
    
    if broke_above and pulled_back:
        return {
            'detected': True,
            'type': 'backup_to_edge',
            'level': resistance,
            'signal': 'BUY'
        }
    
    return {'detected': False}


def identify_creek(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Creek (resistance in accumulation)"""
    if len(closes) < 30:
        return {'level': None}
    
    # Creek is the resistance level in accumulation phase
    recent_highs = [max(highs[i:i+5]) for i in range(0, len(highs)-5, 5)]
    
    if len(recent_highs) >= 3:
        # Creek is around the average of recent highs
        creek_level = sum(recent_highs[-3:]) / 3
        return {
            'level': creek_level,
            'above_creek': closes[-1] > creek_level
        }
    
    return {'level': None}


def identify_ice(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Identify Ice (support in distribution)"""
    if len(closes) < 30:
        return {'level': None}
    
    # Ice is the support level in distribution phase
    recent_lows = [min(lows[i:i+5]) for i in range(0, len(lows)-5, 5)]
    
    if len(recent_lows) >= 3:
        ice_level = sum(recent_lows[-3:]) / 3
        return {
            'level': ice_level,
            'below_ice': closes[-1] < ice_level
        }
    
    return {'level': None}


def calculate_market_profile(highs: List[float], lows: List[float], closes: List[float], volumes: List[float]) -> Dict:
    """Calculate Market Profile (TPO)"""
    if len(closes) < 20:
        return {}
    
    # Price levels
    price_min = min(lows[-20:])
    price_max = max(highs[-20:])
    price_range = price_max - price_min
    
    if price_range == 0:
        return {}
    
    # Create price bins
    num_bins = 20
    bin_size = price_range / num_bins
    
    tpo_count = [0] * num_bins
    volume_at_price = [0] * num_bins
    
    for i in range(-20, 0):
        for price in [lows[i], highs[i], closes[i]]:
            bin_idx = min(int((price - price_min) / bin_size), num_bins - 1)
            tpo_count[bin_idx] += 1
            volume_at_price[bin_idx] += volumes[i] / 3
    
    # Find POC (Point of Control)
    poc_idx = tpo_count.index(max(tpo_count))
    poc = price_min + (poc_idx + 0.5) * bin_size
    
    # Value Area (70% of TPOs)
    total_tpo = sum(tpo_count)
    target_tpo = total_tpo * 0.7
    
    current_tpo = tpo_count[poc_idx]
    vah_idx = poc_idx
    val_idx = poc_idx
    
    while current_tpo < target_tpo:
        if vah_idx < num_bins - 1 and (val_idx == 0 or tpo_count[vah_idx + 1] >= tpo_count[val_idx - 1]):
            vah_idx += 1
            current_tpo += tpo_count[vah_idx]
        elif val_idx > 0:
            val_idx -= 1
            current_tpo += tpo_count[val_idx]
        else:
            break
    
    vah = price_min + (vah_idx + 1) * bin_size
    val = price_min + val_idx * bin_size
    
    return {
        'poc': poc,
        'vah': vah,
        'val': val,
        'price_in_value': val <= closes[-1] <= vah
    }
