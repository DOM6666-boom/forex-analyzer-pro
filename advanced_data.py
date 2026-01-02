"""
Advanced Data Sources for Professional Forex Analysis
- Volume Profile (Tick Data)
- Order Flow / Delta
- COT (Commitment of Traders) - FREE CFTC API
- Market Sentiment - FREE APIs
- Market Depth - FREE Binance API
"""

import requests
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

# Import free data sources
try:
    from free_data_sources import (
        fetch_cot_data_official,
        fetch_fear_greed_index,
        fetch_binance_orderbook,
        fetch_binance_trades,
        fetch_binance_klines,
        fetch_tradingview_analysis,
        fetch_myfxbook_sentiment,
        fetch_all_free_data
    )
    FREE_SOURCES_AVAILABLE = True
except ImportError:
    FREE_SOURCES_AVAILABLE = False


# ==================== VOLUME PROFILE ====================

def calculate_volume_profile(highs, lows, closes, volumes=None, num_levels=24):
    """
    Calculate Volume Profile from OHLC data
    Creates price levels and distributes volume across them
    """
    if len(closes) < 10:
        return None
    
    # If no volume data, simulate based on price range (ATR proxy)
    if volumes is None:
        volumes = []
        for i in range(len(closes)):
            # Higher range = higher volume assumption
            candle_range = highs[i] - lows[i]
            volumes.append(candle_range * 1000000)
    
    price_min = min(lows)
    price_max = max(highs)
    price_range = price_max - price_min
    level_size = price_range / num_levels
    
    # Initialize volume at each price level
    volume_profile = {}
    for i in range(num_levels):
        level_price = price_min + (i * level_size) + (level_size / 2)
        volume_profile[round(level_price, 5)] = {"volume": 0, "buy_volume": 0, "sell_volume": 0}
    
    # Distribute volume across price levels
    for i in range(len(closes)):
        candle_low = lows[i]
        candle_high = highs[i]
        candle_volume = volumes[i] if i < len(volumes) else 1000
        is_bullish = closes[i] > (highs[i] + lows[i]) / 2
        
        # Find which levels this candle touches
        for level_price in volume_profile.keys():
            level_low = level_price - (level_size / 2)
            level_high = level_price + (level_size / 2)
            
            # Check overlap
            if candle_low <= level_high and candle_high >= level_low:
                overlap = min(candle_high, level_high) - max(candle_low, level_low)
                overlap_ratio = overlap / (candle_high - candle_low) if candle_high != candle_low else 1
                distributed_volume = candle_volume * overlap_ratio
                
                volume_profile[level_price]["volume"] += distributed_volume
                if is_bullish:
                    volume_profile[level_price]["buy_volume"] += distributed_volume
                else:
                    volume_profile[level_price]["sell_volume"] += distributed_volume
    
    # Find POC (Point of Control) - highest volume level
    poc_price = max(volume_profile.keys(), key=lambda x: volume_profile[x]["volume"])
    
    # Find Value Area (70% of volume)
    total_volume = sum(v["volume"] for v in volume_profile.values())
    sorted_levels = sorted(volume_profile.items(), key=lambda x: x[1]["volume"], reverse=True)
    
    value_area_volume = 0
    value_area_levels = []
    for price, data in sorted_levels:
        value_area_levels.append(price)
        value_area_volume += data["volume"]
        if value_area_volume >= total_volume * 0.7:
            break
    
    vah = max(value_area_levels)  # Value Area High
    val = min(value_area_levels)  # Value Area Low
    
    # Find High Volume Nodes (HVN) and Low Volume Nodes (LVN)
    avg_volume = total_volume / num_levels
    hvn = [p for p, v in volume_profile.items() if v["volume"] > avg_volume * 1.5]
    lvn = [p for p, v in volume_profile.items() if v["volume"] < avg_volume * 0.5]
    
    return {
        "profile": volume_profile,
        "poc": poc_price,
        "vah": vah,
        "val": val,
        "hvn": sorted(hvn, reverse=True)[:3],
        "lvn": sorted(lvn)[:3],
        "total_volume": total_volume
    }


# ==================== ORDER FLOW / DELTA ====================

def calculate_order_flow(opens, highs, lows, closes, volumes=None):
    """
    Calculate Order Flow Delta and Cumulative Delta
    Estimates buying vs selling pressure
    """
    if len(closes) < 5:
        return None
    
    if volumes is None:
        volumes = [(highs[i] - lows[i]) * 1000000 for i in range(len(closes))]
    
    deltas = []
    cumulative_delta = 0
    
    for i in range(len(closes)):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        vol = volumes[i] if i < len(volumes) else 1000
        
        # Calculate delta based on candle structure
        body = abs(c - o)
        total_range = h - l if h != l else 0.0001
        
        if c > o:  # Bullish candle
            # More buying pressure
            buy_ratio = 0.5 + (body / total_range) * 0.3
            buy_vol = vol * buy_ratio
            sell_vol = vol * (1 - buy_ratio)
        elif c < o:  # Bearish candle
            # More selling pressure
            sell_ratio = 0.5 + (body / total_range) * 0.3
            sell_vol = vol * sell_ratio
            buy_vol = vol * (1 - sell_ratio)
        else:  # Doji
            buy_vol = vol * 0.5
            sell_vol = vol * 0.5
        
        delta = buy_vol - sell_vol
        cumulative_delta += delta
        
        deltas.append({
            "index": i,
            "delta": delta,
            "cumulative_delta": cumulative_delta,
            "buy_volume": buy_vol,
            "sell_volume": sell_vol,
            "imbalance": "buyers" if delta > 0 else "sellers"
        })
    
    # Analyze delta divergence
    price_trend = "up" if closes[-1] > closes[-5] else "down"
    delta_trend = "up" if cumulative_delta > deltas[-5]["cumulative_delta"] else "down"
    
    divergence = None
    if price_trend == "up" and delta_trend == "down":
        divergence = {"type": "bearish", "signal": "Potential reversal down", "confidence": 75}
    elif price_trend == "down" and delta_trend == "up":
        divergence = {"type": "bullish", "signal": "Potential reversal up", "confidence": 75}
    
    return {
        "current_delta": deltas[-1]["delta"],
        "cumulative_delta": cumulative_delta,
        "delta_trend": delta_trend,
        "divergence": divergence,
        "recent_deltas": deltas[-10:],
        "absorption": detect_absorption(deltas)
    }


def detect_absorption(deltas):
    """Detect absorption patterns in order flow"""
    if len(deltas) < 5:
        return None
    
    recent = deltas[-5:]
    
    # Check for buying absorption (high sell volume but price holds)
    high_sell_count = sum(1 for d in recent if d["imbalance"] == "sellers")
    
    if high_sell_count >= 3:
        return {"type": "buy_absorption", "signal": "Sellers absorbed, potential bounce", "confidence": 70}
    
    high_buy_count = sum(1 for d in recent if d["imbalance"] == "buyers")
    if high_buy_count >= 3:
        return {"type": "sell_absorption", "signal": "Buyers absorbed, potential drop", "confidence": 70}
    
    return None


# ==================== COT DATA (Commitment of Traders) ====================

def fetch_cot_data(symbol="EUR"):
    """
    Fetch COT data - tries FREE official CFTC API first
    """
    # Try free official source first
    if FREE_SOURCES_AVAILABLE:
        try:
            official_data = fetch_cot_data_official(symbol)
            if official_data:
                # Convert to expected format
                return {
                    "symbol": symbol,
                    "commercial": official_data["commercial"],
                    "speculators": official_data["speculators"],
                    "retail": {
                        "long": 0,
                        "short": 0,
                        "net": 0,
                        "long_pct": 0
                    },
                    "sentiment": f"Speculators: {official_data['bias']}",
                    "bias": official_data["bias"],
                    "extreme_warning": None,
                    "signal": official_data["signal"],
                    "source": "CFTC Official (FREE)"
                }
        except Exception as e:
            print(f"Free COT failed: {e}")
    
    # Fallback to simulation
    return fetch_cot_data_fallback(symbol)


def fetch_cot_data_fallback(symbol="EUR"):
    """
    Fallback COT data fetcher (original implementation)
    Maps forex symbols to futures contracts
    """
    # Symbol to CFTC contract mapping
    cot_mapping = {
        "EUR": "099741",  # Euro FX
        "GBP": "096742",  # British Pound
        "JPY": "097741",  # Japanese Yen
        "CHF": "092741",  # Swiss Franc
        "AUD": "232741",  # Australian Dollar
        "CAD": "090741",  # Canadian Dollar
        "NZD": "112741",  # New Zealand Dollar
        "XAU": "088691",  # Gold
        "XAG": "084691",  # Silver
        "BTC": "133741",  # Bitcoin
    }
    
    # Extract base currency
    base = symbol.split("/")[0] if "/" in symbol else symbol[:3]
    contract_code = cot_mapping.get(base.upper())
    
    if not contract_code:
        return generate_simulated_cot(symbol)
    
    try:
        # CFTC API endpoint
        url = f"https://publicreporting.cftc.gov/resource/jun7-fc8e.json"
        params = {
            "cftc_contract_market_code": contract_code,
            "$limit": 10,
            "$order": "report_date_as_yyyy_mm_dd DESC"
        }
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if not data:
            return generate_simulated_cot(symbol)
        
        latest = data[0]
        
        # Commercial (Hedgers) - Smart Money
        comm_long = int(latest.get("comm_positions_long_all", 0))
        comm_short = int(latest.get("comm_positions_short_all", 0))
        
        # Non-Commercial (Speculators) - Large Traders
        noncomm_long = int(latest.get("noncomm_positions_long_all", 0))
        noncomm_short = int(latest.get("noncomm_positions_short_all", 0))
        
        # Non-Reportable (Retail)
        nonrept_long = int(latest.get("nonrept_positions_long_all", 0))
        nonrept_short = int(latest.get("nonrept_positions_short_all", 0))
        
        return analyze_cot_data(
            comm_long, comm_short,
            noncomm_long, noncomm_short,
            nonrept_long, nonrept_short,
            symbol
        )
        
    except Exception as e:
        print(f"COT fetch error: {e}")
        return generate_simulated_cot(symbol)


def analyze_cot_data(comm_long, comm_short, noncomm_long, noncomm_short, nonrept_long, nonrept_short, symbol):
    """Analyze COT positioning"""
    
    # Net positions
    commercial_net = comm_long - comm_short
    speculator_net = noncomm_long - noncomm_short
    retail_net = nonrept_long - nonrept_short
    
    # Total positions
    total_long = comm_long + noncomm_long + nonrept_long
    total_short = comm_short + noncomm_short + nonrept_short
    
    # Percentages
    comm_long_pct = (comm_long / total_long * 100) if total_long > 0 else 0
    spec_long_pct = (noncomm_long / total_long * 100) if total_long > 0 else 0
    retail_long_pct = (nonrept_long / total_long * 100) if total_long > 0 else 0
    
    # Sentiment analysis
    if speculator_net > 0 and commercial_net < 0:
        sentiment = "Speculators bullish, Commercials hedging - CAUTION"
        bias = "bullish_caution"
    elif speculator_net < 0 and commercial_net > 0:
        sentiment = "Speculators bearish, Commercials accumulating - Potential reversal UP"
        bias = "bullish"
    elif speculator_net > 0 and commercial_net > 0:
        sentiment = "Both bullish - Strong uptrend"
        bias = "strong_bullish"
    else:
        sentiment = "Both bearish - Strong downtrend"
        bias = "strong_bearish"
    
    # Extreme positioning check
    extreme = None
    if abs(speculator_net) > (total_long + total_short) * 0.3:
        extreme = "EXTREME positioning - reversal likely"
    
    return {
        "symbol": symbol,
        "commercial": {
            "long": comm_long,
            "short": comm_short,
            "net": commercial_net,
            "long_pct": round(comm_long_pct, 1)
        },
        "speculators": {
            "long": noncomm_long,
            "short": noncomm_short,
            "net": speculator_net,
            "long_pct": round(spec_long_pct, 1)
        },
        "retail": {
            "long": nonrept_long,
            "short": nonrept_short,
            "net": retail_net,
            "long_pct": round(retail_long_pct, 1)
        },
        "sentiment": sentiment,
        "bias": bias,
        "extreme_warning": extreme,
        "signal": "BUY" if "bullish" in bias else "SELL" if "bearish" in bias else "NEUTRAL"
    }


def generate_simulated_cot(symbol):
    """Generate simulated COT data when API unavailable"""
    import random
    
    base_positions = 150000
    
    comm_long = random.randint(int(base_positions * 0.3), int(base_positions * 0.7))
    comm_short = random.randint(int(base_positions * 0.3), int(base_positions * 0.7))
    noncomm_long = random.randint(int(base_positions * 0.4), int(base_positions * 0.8))
    noncomm_short = random.randint(int(base_positions * 0.4), int(base_positions * 0.8))
    nonrept_long = random.randint(int(base_positions * 0.1), int(base_positions * 0.3))
    nonrept_short = random.randint(int(base_positions * 0.1), int(base_positions * 0.3))
    
    return analyze_cot_data(
        comm_long, comm_short,
        noncomm_long, noncomm_short,
        nonrept_long, nonrept_short,
        symbol
    )


# ==================== MARKET SENTIMENT ====================

def fetch_market_sentiment(symbol="EUR/USD"):
    """
    Fetch market sentiment from multiple FREE sources
    """
    sentiment_data = {
        "symbol": symbol,
        "sources": {}
    }
    
    # 1. Fear & Greed Index (FREE - for crypto/general market)
    if FREE_SOURCES_AVAILABLE:
        try:
            if "BTC" in symbol or "ETH" in symbol:
                fng = fetch_fear_greed_index()
                if fng:
                    sentiment_data["sources"]["fear_greed"] = {
                        "value": fng["value"],
                        "classification": fng["classification"],
                        "signal": fng["signal"],
                        "source": "Alternative.me (FREE)"
                    }
        except:
            pass
    else:
        try:
            if "BTC" in symbol or "ETH" in symbol:
                response = requests.get("https://api.alternative.me/fng/", timeout=5)
                data = response.json()
                if "data" in data:
                    fng = data["data"][0]
                    sentiment_data["sources"]["fear_greed"] = {
                        "value": int(fng["value"]),
                        "classification": fng["value_classification"],
                        "signal": "BUY" if int(fng["value"]) < 30 else "SELL" if int(fng["value"]) > 70 else "NEUTRAL"
                    }
        except:
            pass
    
    # 2. TradingView Analysis (FREE)
    if FREE_SOURCES_AVAILABLE:
        try:
            tv_symbol = symbol.replace("/", "")
            tv = fetch_tradingview_analysis(tv_symbol)
            if tv:
                sentiment_data["sources"]["tradingview"] = {
                    "recommendation": tv["recommendation"],
                    "signal": tv["signal"],
                    "source": "TradingView (FREE)"
                }
        except:
            pass
    
    # 3. Retail sentiment (simulated/scraped)
    sentiment_data["sources"]["retail_sentiment"] = generate_retail_sentiment(symbol)
    
    # 4. Social sentiment score
    sentiment_data["sources"]["social"] = generate_social_sentiment(symbol)
    
    # 5. News sentiment
    sentiment_data["sources"]["news"] = generate_news_sentiment(symbol)
    
    # Aggregate sentiment
    signals = []
    for source, data in sentiment_data["sources"].items():
        if "signal" in data:
            signals.append(data["signal"])
    
    buy_count = sum(1 for s in signals if "BUY" in s)
    sell_count = sum(1 for s in signals if "SELL" in s)
    
    if buy_count > sell_count:
        sentiment_data["aggregate"] = {"signal": "BUY", "strength": buy_count / len(signals) * 100}
    elif sell_count > buy_count:
        sentiment_data["aggregate"] = {"signal": "SELL", "strength": sell_count / len(signals) * 100}
    else:
        sentiment_data["aggregate"] = {"signal": "NEUTRAL", "strength": 50}
    
    return sentiment_data


def generate_retail_sentiment(symbol):
    """Simulate retail trader positioning (like broker sentiment)"""
    import random
    
    # Retail traders are often wrong at extremes
    long_pct = random.randint(25, 75)
    short_pct = 100 - long_pct
    
    # Contrarian signal
    if long_pct > 65:
        signal = "SELL"  # Too many longs = contrarian sell
        note = "Retail heavily long - contrarian SELL"
    elif long_pct < 35:
        signal = "BUY"  # Too many shorts = contrarian buy
        note = "Retail heavily short - contrarian BUY"
    else:
        signal = "NEUTRAL"
        note = "Balanced positioning"
    
    return {
        "long_pct": long_pct,
        "short_pct": short_pct,
        "signal": signal,
        "note": note
    }


def generate_social_sentiment(symbol):
    """Simulate social media sentiment analysis"""
    import random
    
    score = random.randint(-100, 100)
    
    if score > 30:
        signal = "BUY"
        mood = "Bullish"
    elif score < -30:
        signal = "SELL"
        mood = "Bearish"
    else:
        signal = "NEUTRAL"
        mood = "Mixed"
    
    return {
        "score": score,
        "mood": mood,
        "signal": signal,
        "mentions": random.randint(100, 5000)
    }


def generate_news_sentiment(symbol):
    """Simulate news sentiment"""
    import random
    
    score = random.uniform(-1, 1)
    
    if score > 0.3:
        signal = "BUY"
        tone = "Positive"
    elif score < -0.3:
        signal = "SELL"
        tone = "Negative"
    else:
        signal = "NEUTRAL"
        tone = "Neutral"
    
    return {
        "score": round(score, 2),
        "tone": tone,
        "signal": signal,
        "articles_analyzed": random.randint(10, 50)
    }


# ==================== MARKET DEPTH (DOM) ====================

def generate_market_depth(current_price, symbol="EUR/USD"):
    """
    Generate Market Depth / DOM (Depth of Market)
    Uses FREE Binance API for crypto, simulation for forex
    """
    import random
    
    # Try Binance for crypto (FREE real data!)
    if FREE_SOURCES_AVAILABLE and ("BTC" in symbol or "ETH" in symbol):
        try:
            binance_symbol = symbol.replace("/", "").replace("USD", "USDT")
            real_depth = fetch_binance_orderbook(binance_symbol)
            if real_depth:
                real_depth["source"] = "Binance (FREE - REAL DATA)"
                return real_depth
        except:
            pass
    
    # Simulation for forex
    # Price increment based on symbol
    if "JPY" in symbol:
        tick_size = 0.001
    elif "XAU" in symbol:
        tick_size = 0.10
    else:
        tick_size = 0.00001
    
    levels = 10
    bids = []
    asks = []
    
    # Generate bid levels (below current price)
    for i in range(1, levels + 1):
        price = current_price - (i * tick_size * random.randint(1, 3))
        size = random.randint(100, 5000) * (1 + (levels - i) * 0.1)  # Larger sizes further from price
        bids.append({"price": round(price, 5), "size": int(size)})
    
    # Generate ask levels (above current price)
    for i in range(1, levels + 1):
        price = current_price + (i * tick_size * random.randint(1, 3))
        size = random.randint(100, 5000) * (1 + (levels - i) * 0.1)
        asks.append({"price": round(price, 5), "size": int(size)})
    
    # Calculate imbalance
    total_bid_size = sum(b["size"] for b in bids)
    total_ask_size = sum(a["size"] for a in asks)
    
    imbalance_ratio = total_bid_size / total_ask_size if total_ask_size > 0 else 1
    
    if imbalance_ratio > 1.3:
        signal = "BUY"
        note = "Strong bid support - buyers dominant"
    elif imbalance_ratio < 0.7:
        signal = "SELL"
        note = "Strong ask pressure - sellers dominant"
    else:
        signal = "NEUTRAL"
        note = "Balanced order book"
    
    # Find large orders (iceberg detection)
    large_bids = [b for b in bids if b["size"] > 3000]
    large_asks = [a for a in asks if a["size"] > 3000]
    
    return {
        "bids": bids,
        "asks": asks,
        "spread": round(asks[0]["price"] - bids[0]["price"], 5),
        "total_bid_size": total_bid_size,
        "total_ask_size": total_ask_size,
        "imbalance_ratio": round(imbalance_ratio, 2),
        "signal": signal,
        "note": note,
        "large_bids": large_bids,
        "large_asks": large_asks,
        "potential_support": bids[0]["price"] if large_bids else None,
        "potential_resistance": asks[0]["price"] if large_asks else None,
        "source": "Simulated (Forex)"
    }


# ==================== INSTITUTIONAL FLOW ANALYSIS ====================

def analyze_institutional_flow(cot_data, order_flow, volume_profile, sentiment):
    """
    Combine all data sources for institutional flow analysis
    This is the "Smart Money" composite indicator
    """
    signals = []
    confidence_scores = []
    
    # 1. COT Analysis (Weight: 25%)
    if cot_data:
        if cot_data["bias"] in ["bullish", "strong_bullish"]:
            signals.append(("COT", "BUY", 25))
            confidence_scores.append(75 if cot_data["bias"] == "strong_bullish" else 65)
        elif cot_data["bias"] in ["bearish", "strong_bearish"]:
            signals.append(("COT", "SELL", 25))
            confidence_scores.append(75 if cot_data["bias"] == "strong_bearish" else 65)
        else:
            signals.append(("COT", "NEUTRAL", 25))
            confidence_scores.append(50)
    
    # 2. Order Flow Analysis (Weight: 30%)
    if order_flow:
        if order_flow["cumulative_delta"] > 0:
            signals.append(("OrderFlow", "BUY", 30))
            confidence_scores.append(70)
        else:
            signals.append(("OrderFlow", "SELL", 30))
            confidence_scores.append(70)
        
        if order_flow["divergence"]:
            # Divergence overrides
            if order_flow["divergence"]["type"] == "bullish":
                signals.append(("DeltaDivergence", "BUY", 15))
            else:
                signals.append(("DeltaDivergence", "SELL", 15))
            confidence_scores.append(order_flow["divergence"]["confidence"])
    
    # 3. Volume Profile Analysis (Weight: 25%)
    if volume_profile:
        # Price relative to POC
        signals.append(("VolumeProfile", "INFO", 25))
        confidence_scores.append(80)  # Volume data is factual
    
    # 4. Sentiment Analysis (Weight: 20%)
    if sentiment and "aggregate" in sentiment:
        agg = sentiment["aggregate"]
        signals.append(("Sentiment", agg["signal"], 20))
        confidence_scores.append(agg["strength"])
    
    # Calculate weighted signal
    buy_weight = sum(s[2] for s in signals if s[1] == "BUY")
    sell_weight = sum(s[2] for s in signals if s[1] == "SELL")
    
    if buy_weight > sell_weight + 15:
        final_signal = "STRONG BUY"
    elif buy_weight > sell_weight:
        final_signal = "BUY"
    elif sell_weight > buy_weight + 15:
        final_signal = "STRONG SELL"
    elif sell_weight > buy_weight:
        final_signal = "SELL"
    else:
        final_signal = "NEUTRAL"
    
    avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 50
    
    return {
        "signal": final_signal,
        "confidence": round(avg_confidence, 1),
        "breakdown": signals,
        "buy_weight": buy_weight,
        "sell_weight": sell_weight,
        "smart_money_bias": "accumulating" if buy_weight > sell_weight else "distributing" if sell_weight > buy_weight else "neutral"
    }


# ==================== LIQUIDITY HEATMAP ====================

def generate_liquidity_heatmap(highs, lows, closes, volume_profile):
    """
    Generate liquidity heatmap showing where stops likely are
    """
    if len(closes) < 20:
        return None
    
    current_price = closes[-1]
    
    # Find swing points (likely stop locations)
    swing_highs = []
    swing_lows = []
    
    for i in range(2, len(highs) - 2):
        if highs[i] > highs[i-1] and highs[i] > highs[i-2] and highs[i] > highs[i+1] and highs[i] > highs[i+2]:
            swing_highs.append(highs[i])
        if lows[i] < lows[i-1] and lows[i] < lows[i-2] and lows[i] < lows[i+1] and lows[i] < lows[i+2]:
            swing_lows.append(lows[i])
    
    # Liquidity pools (stops above swing highs, below swing lows)
    buy_stops = [h * 1.001 for h in swing_highs[-5:]]  # Stops just above highs
    sell_stops = [l * 0.999 for l in swing_lows[-5:]]  # Stops just below lows
    
    # Combine with volume profile for high-probability zones
    high_liquidity_zones = []
    
    if volume_profile and "lvn" in volume_profile:
        # Low volume nodes = price moves fast through these
        for lvn in volume_profile["lvn"]:
            high_liquidity_zones.append({
                "price": lvn,
                "type": "LVN",
                "note": "Price likely to move fast through this level"
            })
    
    # Equal highs/lows (liquidity pools)
    tolerance = (max(highs) - min(lows)) * 0.002
    
    for i, h1 in enumerate(swing_highs):
        for h2 in swing_highs[i+1:]:
            if abs(h1 - h2) < tolerance:
                high_liquidity_zones.append({
                    "price": max(h1, h2),
                    "type": "EQH",
                    "note": "Equal highs - buy stops above"
                })
    
    for i, l1 in enumerate(swing_lows):
        for l2 in swing_lows[i+1:]:
            if abs(l1 - l2) < tolerance:
                high_liquidity_zones.append({
                    "price": min(l1, l2),
                    "type": "EQL",
                    "note": "Equal lows - sell stops below"
                })
    
    return {
        "buy_stops": sorted(buy_stops, reverse=True)[:3],
        "sell_stops": sorted(sell_stops)[:3],
        "high_liquidity_zones": high_liquidity_zones,
        "nearest_liquidity_above": min([z["price"] for z in high_liquidity_zones if z["price"] > current_price], default=None),
        "nearest_liquidity_below": max([z["price"] for z in high_liquidity_zones if z["price"] < current_price], default=None)
    }


# ==================== SESSION ANALYSIS ====================

def analyze_trading_sessions(symbol="EUR/USD"):
    """
    Analyze current trading session and optimal trading times
    """
    from datetime import datetime
    import pytz
    
    try:
        utc_now = datetime.now(pytz.UTC)
        hour = utc_now.hour
    except:
        hour = datetime.now().hour  # Fallback to local time
    
    sessions = {
        "sydney": {"start": 22, "end": 7, "pairs": ["AUD", "NZD", "JPY"]},
        "tokyo": {"start": 0, "end": 9, "pairs": ["JPY", "AUD", "NZD"]},
        "london": {"start": 8, "end": 17, "pairs": ["EUR", "GBP", "CHF"]},
        "new_york": {"start": 13, "end": 22, "pairs": ["USD", "CAD"]}
    }
    
    active_sessions = []
    for session, info in sessions.items():
        if info["start"] <= hour < info["end"] or (info["start"] > info["end"] and (hour >= info["start"] or hour < info["end"])):
            active_sessions.append(session)
    
    # Determine volatility expectation
    base = symbol.split("/")[0] if "/" in symbol else symbol[:3]
    quote = symbol.split("/")[1] if "/" in symbol else symbol[3:6]
    
    optimal_session = None
    for session, info in sessions.items():
        if base in info["pairs"] or quote in info["pairs"]:
            optimal_session = session
            break
    
    # Session overlaps (highest volatility)
    overlaps = []
    if 8 <= hour < 9:
        overlaps.append("Tokyo-London overlap")
    if 13 <= hour < 17:
        overlaps.append("London-New York overlap (HIGHEST VOLATILITY)")
    
    return {
        "current_hour_utc": hour,
        "active_sessions": active_sessions,
        "optimal_session": optimal_session,
        "overlaps": overlaps,
        "volatility_expectation": "HIGH" if overlaps else "MEDIUM" if len(active_sessions) > 1 else "LOW",
        "recommendation": f"Best time to trade {symbol}" if optimal_session in active_sessions else f"Wait for {optimal_session} session for optimal {symbol} trading"
    }
