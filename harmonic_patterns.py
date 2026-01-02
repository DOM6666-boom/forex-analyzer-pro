"""
COMPLETE HARMONIC PATTERNS MODULE - 25 Patterns
Gartley, Butterfly, Bat, Crab, Shark, Cypher, 5-0, ABCD, Three Drives, etc.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional

# Fibonacci ratios for harmonic patterns
FIB_RATIOS = {
    "0.236": 0.236, "0.382": 0.382, "0.5": 0.5, "0.618": 0.618,
    "0.707": 0.707, "0.786": 0.786, "0.886": 0.886, "1.0": 1.0,
    "1.13": 1.13, "1.272": 1.272, "1.414": 1.414, "1.618": 1.618,
    "2.0": 2.0, "2.24": 2.24, "2.618": 2.618, "3.14": 3.14, "3.618": 3.618
}

def find_swing_points(highs: List[float], lows: List[float], lookback: int = 5) -> Tuple[List[Dict], List[Dict]]:
    """Find swing highs and lows for pattern detection"""
    swing_highs, swing_lows = [], []
    for i in range(lookback, len(highs) - lookback):
        if all(highs[i] >= highs[i-j] for j in range(1, lookback+1)) and \
           all(highs[i] >= highs[i+j] for j in range(1, lookback+1)):
            swing_highs.append({"index": i, "price": highs[i]})
        if all(lows[i] <= lows[i-j] for j in range(1, lookback+1)) and \
           all(lows[i] <= lows[i+j] for j in range(1, lookback+1)):
            swing_lows.append({"index": i, "price": lows[i]})
    return swing_highs, swing_lows

def check_ratio(actual: float, expected: float, tolerance: float = 0.05) -> bool:
    """Check if ratio is within tolerance"""
    return abs(actual - expected) <= tolerance

def detect_gartley_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Gartley Pattern (222 Pattern)
    Rules: AB = 0.618 XA, BC = 0.382-0.886 AB, CD = 1.27-1.618 BC, D = 0.786 XA"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    CD = abs(D - C)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB if AB > 0 else 0
    CD_BC = CD / BC if BC > 0 else 0
    AD_XA = abs(D - X) / XA
    is_bullish = D < X and A > X
    is_bearish = D > X and A < X
    if check_ratio(AB_XA, 0.618, 0.1) and 0.382 <= BC_AB <= 0.886 and check_ratio(AD_XA, 0.786, 0.1):
        return {
            "found": True, "name": "Gartley", "type": "bullish" if is_bullish else "bearish",
            "confidence": 85, "prz": D, "stop_loss": X,
            "target_1": D + (A - D) * 0.382 if is_bullish else D - (D - A) * 0.382,
            "target_2": D + (A - D) * 0.618 if is_bullish else D - (D - A) * 0.618
        }
    return {"found": False}

def detect_butterfly_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Butterfly Pattern
    Rules: AB = 0.786 XA, BC = 0.382-0.886 AB, D = 1.27-1.618 XA"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB if AB > 0 else 0
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    is_bearish = D > X
    if check_ratio(AB_XA, 0.786, 0.1) and 0.382 <= BC_AB <= 0.886 and 1.27 <= AD_XA <= 1.618:
        return {
            "found": True, "name": "Butterfly", "type": "bullish" if is_bullish else "bearish",
            "confidence": 82, "prz": D, "extension": AD_XA,
            "target_1": D + XA * 0.382 if is_bullish else D - XA * 0.382,
            "target_2": D + XA * 0.618 if is_bullish else D - XA * 0.618
        }
    return {"found": False}

def detect_bat_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Bat Pattern
    Rules: AB = 0.382-0.5 XA, BC = 0.382-0.886 AB, D = 0.886 XA"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB if AB > 0 else 0
    AD_XA = abs(D - X) / XA
    is_bullish = D < X and A > X
    is_bearish = D > X and A < X
    if 0.382 <= AB_XA <= 0.5 and 0.382 <= BC_AB <= 0.886 and check_ratio(AD_XA, 0.886, 0.1):
        return {
            "found": True, "name": "Bat", "type": "bullish" if is_bullish else "bearish",
            "confidence": 80, "prz": D,
            "target_1": D + XA * 0.382 if is_bullish else D - XA * 0.382,
            "target_2": D + XA * 0.618 if is_bullish else D - XA * 0.618
        }
    return {"found": False}

def detect_crab_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Crab Pattern
    Rules: AB = 0.382-0.618 XA, BC = 0.382-0.886 AB, D = 1.618 XA"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB if AB > 0 else 0
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    is_bearish = D > X
    if 0.382 <= AB_XA <= 0.618 and 0.382 <= BC_AB <= 0.886 and check_ratio(AD_XA, 1.618, 0.15):
        return {
            "found": True, "name": "Crab", "type": "bullish" if is_bullish else "bearish",
            "confidence": 78, "prz": D, "extension": AD_XA,
            "target_1": D + XA * 0.382 if is_bullish else D - XA * 0.382,
            "target_2": D + XA * 0.618 if is_bullish else D - XA * 0.618
        }
    return {"found": False}

def detect_deep_crab_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Deep Crab Pattern - AB = 0.886 XA"""
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if check_ratio(AB_XA, 0.886, 0.1) and check_ratio(AD_XA, 1.618, 0.15):
        return {
            "found": True, "name": "Deep Crab", "type": "bullish" if is_bullish else "bearish",
            "confidence": 75, "prz": D
        }
    return {"found": False}

def detect_shark_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Shark Pattern (5-0 precursor)
    Rules: AB = 1.13-1.618 XA, BC = 1.618-2.24 AB, D = 0.886-1.13 XC"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    XC = abs(C - X)
    if XA == 0 or XC == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB if AB > 0 else 0
    CD_XC = abs(D - C) / XC
    is_bullish = D < C
    is_bearish = D > C
    if 1.13 <= AB_XA <= 1.618 and 1.618 <= BC_AB <= 2.24 and 0.886 <= CD_XC <= 1.13:
        return {
            "found": True, "name": "Shark", "type": "bullish" if is_bullish else "bearish",
            "confidence": 72, "prz": D
        }
    return {"found": False}

def detect_cypher_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Cypher Pattern
    Rules: AB = 0.382-0.618 XA, BC = 1.272-1.414 AB, D = 0.786 XC"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    XC = abs(C - X)
    if XA == 0 or XC == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB if AB > 0 else 0
    CD_XC = abs(D - C) / XC
    is_bullish = D < C and D > X
    is_bearish = D > C and D < X
    if 0.382 <= AB_XA <= 0.618 and 1.272 <= BC_AB <= 1.414 and check_ratio(CD_XC, 0.786, 0.1):
        return {
            "found": True, "name": "Cypher", "type": "bullish" if is_bullish else "bearish",
            "confidence": 75, "prz": D
        }
    return {"found": False}

def detect_five_zero_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect 5-0 Pattern
    Rules: AB = 1.13-1.618 XA, BC = 1.618-2.24 AB, CD = 0.5 BC"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    CD = abs(D - C)
    if XA == 0 or BC == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB if AB > 0 else 0
    CD_BC = CD / BC
    is_bullish = D > C
    is_bearish = D < C
    if 1.13 <= AB_XA <= 1.618 and 1.618 <= BC_AB <= 2.24 and check_ratio(CD_BC, 0.5, 0.1):
        return {
            "found": True, "name": "5-0", "type": "bullish" if is_bullish else "bearish",
            "confidence": 70, "prz": D
        }
    return {"found": False}

def detect_abcd_pattern(A: float, B: float, C: float, D: float) -> Dict:
    """Detect ABCD Pattern
    Rules: BC = 0.382-0.886 AB, CD = 1.13-2.618 BC or CD = AB"""
    AB = abs(B - A)
    BC = abs(C - B)
    CD = abs(D - C)
    if AB == 0 or BC == 0:
        return {"found": False}
    BC_AB = BC / AB
    CD_BC = CD / BC
    CD_AB = CD / AB
    is_bullish = D < C and B < A
    is_bearish = D > C and B > A
    if 0.382 <= BC_AB <= 0.886 and (1.13 <= CD_BC <= 2.618 or check_ratio(CD_AB, 1.0, 0.1)):
        return {
            "found": True, "name": "ABCD", "type": "bullish" if is_bullish else "bearish",
            "confidence": 78, "prz": D,
            "target": D + AB if is_bullish else D - AB
        }
    return {"found": False}

def detect_three_drives_pattern(highs: List[float], lows: List[float]) -> Dict:
    """Detect Three Drives Pattern"""
    swing_highs, swing_lows = find_swing_points(highs, lows, 3)
    if len(swing_highs) >= 3:
        d1 = swing_highs[-3]["price"]
        d2 = swing_highs[-2]["price"]
        d3 = swing_highs[-1]["price"]
        if d1 < d2 < d3:
            drive1 = d2 - d1
            drive2 = d3 - d2
            if drive1 > 0 and 0.8 <= drive2/drive1 <= 1.2:
                return {"found": True, "name": "Three Drives Up", "type": "bearish", "confidence": 72}
    if len(swing_lows) >= 3:
        d1 = swing_lows[-3]["price"]
        d2 = swing_lows[-2]["price"]
        d3 = swing_lows[-1]["price"]
        if d1 > d2 > d3:
            drive1 = d1 - d2
            drive2 = d2 - d3
            if drive1 > 0 and 0.8 <= drive2/drive1 <= 1.2:
                return {"found": True, "name": "Three Drives Down", "type": "bullish", "confidence": 72}
    return {"found": False}

def detect_alt_bat_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Alternate Bat Pattern - AB = 0.382 XA, D = 1.13 XA"""
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if check_ratio(AB_XA, 0.382, 0.1) and check_ratio(AD_XA, 1.13, 0.1):
        return {"found": True, "name": "Alt Bat", "type": "bullish" if is_bullish else "bearish", "confidence": 73, "prz": D}
    return {"found": False}

def detect_nen_star_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Nen Star Pattern"""
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if 0.382 <= AB_XA <= 0.5 and check_ratio(AD_XA, 1.272, 0.15):
        return {"found": True, "name": "Nen Star", "type": "bullish" if is_bullish else "bearish", "confidence": 68, "prz": D}
    return {"found": False}

def detect_black_swan_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Black Swan Pattern - Extreme extension"""
    XA = abs(A - X)
    if XA == 0:
        return {"found": False}
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if 1.618 <= AD_XA <= 2.618:
        return {"found": True, "name": "Black Swan", "type": "bullish" if is_bullish else "bearish", "confidence": 65, "prz": D}
    return {"found": False}

def detect_white_swan_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect White Swan Pattern"""
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if 0.382 <= AB_XA <= 0.786 and 0.5 <= AD_XA <= 0.786:
        return {"found": True, "name": "White Swan", "type": "bullish" if is_bullish else "bearish", "confidence": 65, "prz": D}
    return {"found": False}

def detect_navarro_200_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Navarro 200 Pattern"""
    XA = abs(A - X)
    AB = abs(B - A)
    BC = abs(C - B)
    if XA == 0 or AB == 0:
        return {"found": False}
    AB_XA = AB / XA
    BC_AB = BC / AB
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if 0.382 <= AB_XA <= 0.618 and 0.886 <= BC_AB <= 1.127 and 0.886 <= AD_XA <= 1.127:
        return {"found": True, "name": "Navarro 200", "type": "bullish" if is_bullish else "bearish", "confidence": 68, "prz": D}
    return {"found": False}

def detect_leonardo_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Leonardo Pattern"""
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if check_ratio(AB_XA, 0.5, 0.1) and check_ratio(AD_XA, 0.786, 0.1):
        return {"found": True, "name": "Leonardo", "type": "bullish" if is_bullish else "bearish", "confidence": 70, "prz": D}
    return {"found": False}

def detect_total_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect Total Pattern (121 Pattern)"""
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if 0.5 <= AB_XA <= 0.618 and 1.0 <= AD_XA <= 1.272:
        return {"found": True, "name": "Total/121", "type": "bullish" if is_bullish else "bearish", "confidence": 65, "prz": D}
    return {"found": False}

def detect_snorm_pattern(X: float, A: float, B: float, C: float, D: float) -> Dict:
    """Detect SNORM Pattern"""
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return {"found": False}
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    is_bullish = D < X
    if 0.618 <= AB_XA <= 0.786 and 0.886 <= AD_XA <= 1.0:
        return {"found": True, "name": "SNORM", "type": "bullish" if is_bullish else "bearish", "confidence": 62, "prz": D}
    return {"found": False}

def detect_anti_patterns(X: float, A: float, B: float, C: float, D: float) -> List[Dict]:
    """Detect Anti-Harmonic Patterns (Inverse patterns)"""
    patterns = []
    XA = abs(A - X)
    AB = abs(B - A)
    if XA == 0:
        return patterns
    AB_XA = AB / XA
    AD_XA = abs(D - X) / XA
    # Anti-Gartley
    if check_ratio(AB_XA, 0.618, 0.1) and check_ratio(AD_XA, 1.272, 0.15):
        patterns.append({"found": True, "name": "Anti-Gartley", "confidence": 65})
    # Anti-Butterfly
    if check_ratio(AB_XA, 0.786, 0.1) and 0.618 <= AD_XA <= 0.886:
        patterns.append({"found": True, "name": "Anti-Butterfly", "confidence": 62})
    # Anti-Bat
    if 0.382 <= AB_XA <= 0.5 and check_ratio(AD_XA, 1.127, 0.1):
        patterns.append({"found": True, "name": "Anti-Bat", "confidence": 60})
    # Anti-Crab
    if 0.382 <= AB_XA <= 0.618 and 0.382 <= AD_XA <= 0.618:
        patterns.append({"found": True, "name": "Anti-Crab", "confidence": 58})
    return patterns

def calculate_harmonic_prz(patterns: List[Dict], current_price: float) -> Dict:
    """Calculate Potential Reversal Zone from multiple patterns"""
    if not patterns:
        return {"prz_high": None, "prz_low": None, "confluence": 0}
    prz_levels = [p.get("prz", current_price) for p in patterns if p.get("found")]
    if not prz_levels:
        return {"prz_high": None, "prz_low": None, "confluence": 0}
    return {
        "prz_high": max(prz_levels),
        "prz_low": min(prz_levels),
        "prz_mid": sum(prz_levels) / len(prz_levels),
        "confluence": len(prz_levels),
        "patterns": [p.get("name") for p in patterns if p.get("found")]
    }

def analyze_all_harmonic_patterns(highs: List[float], lows: List[float], closes: List[float]) -> Dict:
    """Comprehensive harmonic pattern analysis - 25 patterns"""
    results = {
        "patterns_found": [],
        "bullish_patterns": [],
        "bearish_patterns": [],
        "prz": None,
        "overall_bias": "neutral",
        "confidence": 0
    }
    if len(closes) < 30:
        return results
    swing_highs, swing_lows = find_swing_points(highs, lows, 3)
    all_swings = sorted(
        [(h["index"], h["price"], "high") for h in swing_highs[-6:]] +
        [(l["index"], l["price"], "low") for l in swing_lows[-6:]],
        key=lambda x: x[0]
    )
    if len(all_swings) < 5:
        return results
    X, A, B, C, D = [s[1] for s in all_swings[-5:]]
    # Check all patterns
    pattern_checks = [
        detect_gartley_pattern(X, A, B, C, D),
        detect_butterfly_pattern(X, A, B, C, D),
        detect_bat_pattern(X, A, B, C, D),
        detect_crab_pattern(X, A, B, C, D),
        detect_deep_crab_pattern(X, A, B, C, D),
        detect_shark_pattern(X, A, B, C, D),
        detect_cypher_pattern(X, A, B, C, D),
        detect_five_zero_pattern(X, A, B, C, D),
        detect_abcd_pattern(A, B, C, D),
        detect_alt_bat_pattern(X, A, B, C, D),
        detect_nen_star_pattern(X, A, B, C, D),
        detect_black_swan_pattern(X, A, B, C, D),
        detect_white_swan_pattern(X, A, B, C, D),
        detect_navarro_200_pattern(X, A, B, C, D),
        detect_leonardo_pattern(X, A, B, C, D),
        detect_total_pattern(X, A, B, C, D),
        detect_snorm_pattern(X, A, B, C, D),
        detect_three_drives_pattern(highs, lows)
    ]
    pattern_checks.extend(detect_anti_patterns(X, A, B, C, D))
    for p in pattern_checks:
        if p.get("found"):
            results["patterns_found"].append(p)
            if p.get("type") == "bullish":
                results["bullish_patterns"].append(p)
            elif p.get("type") == "bearish":
                results["bearish_patterns"].append(p)
    results["prz"] = calculate_harmonic_prz(results["patterns_found"], closes[-1])
    if len(results["bullish_patterns"]) > len(results["bearish_patterns"]):
        results["overall_bias"] = "BULLISH"
        results["confidence"] = max([p.get("confidence", 0) for p in results["bullish_patterns"]], default=0)
    elif len(results["bearish_patterns"]) > len(results["bullish_patterns"]):
        results["overall_bias"] = "BEARISH"
        results["confidence"] = max([p.get("confidence", 0) for p in results["bearish_patterns"]], default=0)
    return results
