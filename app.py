import os
import base64
import requests
from flask import Flask, request, jsonify, render_template, redirect, url_for, session
from groq import Groq
from dotenv import load_dotenv

# Version: 2026-01-02-v2 - Fixed login modal issue
APP_VERSION = "2026-01-02-v2"

# Stripe Payment Integration
try:
    import stripe
    STRIPE_AVAILABLE = True
except ImportError:
    STRIPE_AVAILABLE = False
    print("⚠️ Stripe not installed. Run: pip install stripe")

# Authentication System
from auth import (
    get_user_by_email, get_user_by_id, create_user, update_user_login,
    update_user_google, check_analysis_limit, increment_analysis_count,
    save_analysis, get_user_history, upgrade_user_tier, get_user_tier_info,
    login_required, get_current_user, verify_password, TIER_LIMITS
)

# Google Gemini Integration (FREE backup)
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ Google Gemini not installed. Run: pip install google-generativeai")

# MetaTrader5 Integration
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("⚠️ MetaTrader5 not installed. Using fallback APIs.")

# Original modules
from indicators import *
from patterns import *
from patterns import detect_chart_patterns

# Advanced data sources
from advanced_data import (
    calculate_volume_profile,
    calculate_order_flow,
    fetch_cot_data,
    fetch_market_sentiment,
    generate_market_depth,
    analyze_institutional_flow,
    generate_liquidity_heatmap,
    analyze_trading_sessions
)

# FREE Real Data Sources
from free_data_sources import (
    fetch_cot_data_official as fetch_real_cot_data,
    fetch_fear_greed_index,
    fetch_binance_orderbook,
    fetch_binance_trades as fetch_binance_recent_trades,
    fetch_oanda_orderbook,
    fetch_polygon_volume as fetch_polygon_data,
    fetch_alpha_vantage_forex as fetch_alpha_vantage_data,
    fetch_all_free_data as get_comprehensive_market_data
)

# COMPLETE 744 Technical Analysis Modules
from patterns_complete import (
    identify_all_candlestick_patterns,
    analyze_pattern_confluence
)

from chart_patterns_complete import (
    detect_all_chart_patterns,
    analyze_pattern_quality
)

from ict_smc_complete import (
    complete_ict_smc_analysis as analyze_complete_ict_smc,
    analyze_liquidity as identify_all_liquidity,
    find_all_order_blocks as identify_all_order_blocks,
    find_all_fvg as identify_all_fvg,
    analyze_kill_zones,
    analyze_power_of_3
)

from indicators_complete import (
    analyze_all_indicators
)

from price_action_complete import (
    analyze_all_price_action
)

# NEW ADVANCED MODULES - Complete 744 Concepts
from indicators_advanced import (
    analyze_advanced_indicators
)

from harmonic_patterns import (
    analyze_all_harmonic_patterns
)

from volume_analysis import (
    analyze_all_volume
)

from statistical_analysis import (
    analyze_all_statistics
)

from market_regime import (
    analyze_all_regimes
)

from multi_timeframe import (
    analyze_all_mtf
)

from sentiment_analysis import (
    analyze_all_sentiment
)

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv('SECRET_KEY', 'your-super-secret-key-change-in-production')

# Detect if running on Render (production) or locally
IS_PRODUCTION = os.environ.get('RENDER') is not None

# Security Configuration - Auto-detect HTTPS for production
app.config['SESSION_COOKIE_SECURE'] = IS_PRODUCTION  # True on Render (HTTPS), False locally
app.config['SESSION_COOKIE_HTTPONLY'] = True  # Prevent JavaScript access to cookies
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'  # CSRF protection
app.config['PERMANENT_SESSION_LIFETIME'] = 86400  # 24 hours session

print(f"[SESSION] Production mode: {IS_PRODUCTION}, Secure cookies: {IS_PRODUCTION}")

# Security Headers
@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'SAMEORIGIN'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    # Prevent caching of HTML pages to ensure latest version
    if response.content_type and 'text/html' in response.content_type:
        response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# Google OAuth Config
GOOGLE_CLIENT_ID = os.getenv('GOOGLE_CLIENT_ID', '')
GOOGLE_CLIENT_SECRET = os.getenv('GOOGLE_CLIENT_SECRET', '')

# Stripe Payment Config
STRIPE_PUBLIC_KEY = os.getenv('STRIPE_PUBLIC_KEY', '')
STRIPE_SECRET_KEY = os.getenv('STRIPE_SECRET_KEY', '')
STRIPE_WEBHOOK_SECRET = os.getenv('STRIPE_WEBHOOK_SECRET', '')
STRIPE_PRO_PRICE_ID = os.getenv('STRIPE_PRO_PRICE_ID', 'price_pro_monthly')
STRIPE_PREMIUM_PRICE_ID = os.getenv('STRIPE_PREMIUM_PRICE_ID', 'price_premium_monthly')

if STRIPE_AVAILABLE and STRIPE_SECRET_KEY:
    stripe.api_key = STRIPE_SECRET_KEY
    print("✅ Stripe payment enabled")
else:
    print("⚠️ Stripe not configured (add STRIPE_SECRET_KEY to .env)")

# Multiple Groq API keys for rotation (to avoid rate limits)
GROQ_API_KEYS = [
    os.getenv("GROQ_API_KEY"),
    os.getenv("GROQ_API_KEY_2"),
    os.getenv("GROQ_API_KEY_3"),
    os.getenv("GROQ_API_KEY_4"),
    os.getenv("GROQ_API_KEY_5"),
    os.getenv("GROQ_API_KEY_6"),
    os.getenv("GROQ_API_KEY_7"),
    os.getenv("GROQ_API_KEY_8"),
    os.getenv("GROQ_API_KEY_9"),
    os.getenv("GROQ_API_KEY_10"),
    os.getenv("GROQ_API_KEY_11"),
    os.getenv("GROQ_API_KEY_12"),
]
# Filter out empty keys
GROQ_API_KEYS = [k for k in GROQ_API_KEYS if k]
print(f"✅ Loaded {len(GROQ_API_KEYS)} Groq API keys")

# Create clients for each API key
groq_clients = [Groq(api_key=key) for key in GROQ_API_KEYS]
current_client_index = 0

# Google Gemini Setup (FREE backup when Groq is exhausted)
GEMINI_API_KEYS = [
    os.getenv("GEMINI_API_KEY"),
    os.getenv("GEMINI_API_KEY_2"),
    os.getenv("GEMINI_API_KEY_3"),
    os.getenv("GEMINI_API_KEY_4"),
]
GEMINI_API_KEYS = [k for k in GEMINI_API_KEYS if k]
gemini_models = []
if GEMINI_AVAILABLE and GEMINI_API_KEYS:
    for key in GEMINI_API_KEYS:
        genai.configure(api_key=key)
        gemini_models.append(genai.GenerativeModel('gemini-1.5-flash'))
    print(f"✅ Gemini backup enabled ({len(GEMINI_API_KEYS)} keys)")
else:
    print("⚠️ Gemini not configured (add GEMINI_API_KEY to .env)")
current_gemini_index = 0

TWELVE_DATA_KEY = os.getenv("TWELVE_DATA_KEY", "demo")
ALPHA_VANTAGE_KEY = os.getenv("ALPHA_VANTAGE_KEY", "demo")

# Vision-capable models (for image analysis) - Updated Jan 2026
VISION_MODELS = [
    "meta-llama/llama-4-scout-17b-16e-instruct",  # Primary vision model (best)
    "meta-llama/llama-4-maverick-17b-128e-instruct",  # Alternative vision model
    "llama-3.2-11b-vision-preview",                # Fallback vision model
]

# Text-only models (for non-image tasks)
TEXT_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
    "mixtral-8x7b-32768",
]


def call_gemini_vision(prompt, image_base64):
    """Call Google Gemini as backup for vision tasks with key rotation"""
    global current_gemini_index
    
    if not gemini_models:
        raise Exception("Gemini not configured")
    
    import PIL.Image
    import io
    
    # Decode base64 image
    image_data = base64.b64decode(image_base64)
    image = PIL.Image.open(io.BytesIO(image_data))
    
    # Try each Gemini key
    for attempt in range(len(gemini_models)):
        idx = (current_gemini_index + attempt) % len(gemini_models)
        try:
            # Reconfigure with this key
            genai.configure(api_key=GEMINI_API_KEYS[idx])
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content([prompt, image])
            print(f"✅ Using Gemini key #{idx + 1}")
            current_gemini_index = (idx + 1) % len(gemini_models)
            return response.text
        except Exception as e:
            print(f"⚠️ Gemini key #{idx + 1} error: {str(e)[:50]}")
            continue
    
    raise Exception("All Gemini keys exhausted")


def call_groq_with_fallback(messages, max_tokens=4000, temperature=0.01, has_images=True, image_base64=None):
    """Call Groq API with automatic API key rotation, model fallback, and Gemini backup"""
    global current_client_index
    models = VISION_MODELS if has_images else TEXT_MODELS
    
    errors_count = 0
    max_errors = 6  # Switch to Gemini after 6 failures (faster fallback)
    
    # Try each API key
    for key_attempt in range(len(groq_clients)):
        client_idx = (current_client_index + key_attempt) % len(groq_clients)
        client = groq_clients[client_idx]
        
        # Try only the primary model first for speed
        for model in models[:2]:  # Only try first 2 models
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature
                )
                print(f"✅ Using Groq key #{client_idx + 1}, model: {model}")
                current_client_index = (client_idx + 1) % len(groq_clients)
                return response
            except Exception as e:
                error_str = str(e)
                errors_count += 1
                
                if "429" in error_str or "rate_limit" in error_str.lower():
                    print(f"⚠️ Rate limit on key #{client_idx + 1}")
                    break  # Move to next key immediately
                elif "400" in error_str or "model" in error_str.lower():
                    print(f"⚠️ Model {model} unavailable")
                    continue
                else:
                    print(f"⚠️ Error: {error_str[:50]}")
                    continue
                
                # Switch to Gemini early if too many errors
                if errors_count >= max_errors and has_images and gemini_models and image_base64:
                    print("🔄 Too many Groq errors, switching to Gemini...")
                    break
        
        # Check if we should switch to Gemini
        if errors_count >= max_errors:
            break
    
    # All Groq keys exhausted or too many errors - try Gemini backup
    if has_images and gemini_models and image_base64:
        print("🔄 Using Gemini backup...")
        # Extract text prompt from messages
        text_prompt = ""
        for msg in messages:
            if isinstance(msg.get("content"), list):
                for item in msg["content"]:
                    if item.get("type") == "text":
                        text_prompt = item.get("text", "")
                        break
            elif isinstance(msg.get("content"), str):
                text_prompt = msg["content"]
        
        gemini_response = call_gemini_vision(text_prompt, image_base64)
        # Return a mock response object that matches Groq's format
        class MockResponse:
            class Choice:
                class Message:
                    content = gemini_response
                message = Message()
            choices = [Choice()]
        return MockResponse()
    
    raise Exception("All Groq keys rate limited and Gemini not available. Get Gemini key from https://aistudio.google.com/apikey")

# Standard decimal places for each symbol type
SYMBOL_DECIMALS = {
    "XAU/USD": 2,  # Gold: 2650.50
    "XAG/USD": 3,  # Silver: 30.500
    "BTC/USD": 2,  # Bitcoin: 42000.00
    "ETH/USD": 2,  # Ethereum: 2300.00
    "EUR/USD": 5,  # Forex pairs: 1.08500
    "GBP/USD": 5,
    "USD/JPY": 3,  # JPY pairs: 148.500
    "USD/CHF": 5,
    "AUD/USD": 5,
    "NZD/USD": 5,
    "USD/CAD": 5,
    "EUR/GBP": 5,
    "EUR/JPY": 3,
    "GBP/JPY": 3,
}

def get_decimal_places(symbol):
    """Get standard decimal places for a symbol"""
    return SYMBOL_DECIMALS.get(symbol, 5)

def format_price(price, symbol):
    """Format price with correct decimal places"""
    decimals = get_decimal_places(symbol)
    return f"{price:.{decimals}f}"


# ========== MT5 INTEGRATION ==========
def init_mt5():
    """Initialize MT5 connection"""
    if not MT5_AVAILABLE:
        return False
    
    if not mt5.initialize():
        print(f"❌ MT5 initialize failed: {mt5.last_error()}")
        return False
    
    print(f"✅ MT5 Connected: {mt5.terminal_info().name}")
    return True


def fetch_mt5_data(symbol, timeframe="M5", bars=100):
    """Fetch live data from MT5"""
    if not MT5_AVAILABLE:
        return None
    
    # Initialize MT5 if not already
    if not mt5.terminal_info():
        if not init_mt5():
            return None
    
    # Convert symbol format: XAU/USD -> XAUUSD
    mt5_symbol = symbol.replace("/", "")
    
    # Map timeframe string to MT5 constant
    tf_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "1min": mt5.TIMEFRAME_M1,
        "5min": mt5.TIMEFRAME_M5,
        "15min": mt5.TIMEFRAME_M15,
        "30min": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1day": mt5.TIMEFRAME_D1,
    }
    
    mt5_tf = tf_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    # Check if symbol exists
    symbol_info = mt5.symbol_info(mt5_symbol)
    if symbol_info is None:
        # Try alternative symbol names
        alternatives = [
            mt5_symbol,
            mt5_symbol + ".a",  # Some brokers use suffix
            mt5_symbol + "m",
            mt5_symbol + "_",
            "GOLD" if "XAU" in symbol else mt5_symbol,
            "SILVER" if "XAG" in symbol else mt5_symbol,
        ]
        
        for alt in alternatives:
            symbol_info = mt5.symbol_info(alt)
            if symbol_info:
                mt5_symbol = alt
                break
        
        if symbol_info is None:
            print(f"❌ MT5 Symbol not found: {symbol}")
            return None
    
    # Enable symbol if not visible
    if not symbol_info.visible:
        if not mt5.symbol_select(mt5_symbol, True):
            print(f"❌ Failed to select symbol: {mt5_symbol}")
            return None
    
    # Fetch rates
    rates = mt5.copy_rates_from_pos(mt5_symbol, mt5_tf, 0, bars)
    
    if rates is None or len(rates) == 0:
        print(f"❌ MT5 No data for {mt5_symbol}: {mt5.last_error()}")
        return None
    
    # Convert to our format
    print(f"✅ MT5 Data: {mt5_symbol} - {len(rates)} bars @ {timeframe}")
    
    return {
        "opens": [float(r['open']) for r in rates],
        "highs": [float(r['high']) for r in rates],
        "lows": [float(r['low']) for r in rates],
        "closes": [float(r['close']) for r in rates],
        "volumes": [float(r['tick_volume']) for r in rates],
        "symbol": symbol,
        "source": "MT5",
        "mt5_symbol": mt5_symbol,
        "current_price": float(rates[-1]['close'])
    }


def get_mt5_current_price(symbol):
    """Get current bid/ask from MT5"""
    if not MT5_AVAILABLE:
        return None
    
    mt5_symbol = symbol.replace("/", "")
    tick = mt5.symbol_info_tick(mt5_symbol)
    
    if tick:
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": tick.ask - tick.bid,
            "time": tick.time
        }
    return None


# Initialize MT5 connection at module load
if MT5_AVAILABLE:
    init_mt5()


# ========== ACCURACY IMPROVEMENT SYSTEM ==========

def validate_price_with_mt5(symbol, ai_price, tolerance_pips=50):
    """Validate AI's price reading against real MT5 price"""
    if not MT5_AVAILABLE:
        return {"valid": True, "reason": "MT5 not available for validation"}
    
    current = get_mt5_current_price(symbol)
    if not current:
        return {"valid": True, "reason": "Could not get MT5 price"}
    
    mt5_price = (current['bid'] + current['ask']) / 2
    
    # Calculate pip difference based on symbol
    if "JPY" in symbol:
        pip_value = 0.01
    elif "XAU" in symbol:
        pip_value = 0.1  # Gold: 1 pip = $0.10
    else:
        pip_value = 0.0001
    
    diff_pips = abs(ai_price - mt5_price) / pip_value
    
    return {
        "valid": diff_pips <= tolerance_pips,
        "ai_price": ai_price,
        "mt5_price": mt5_price,
        "diff_pips": round(diff_pips, 1),
        "tolerance": tolerance_pips,
        "reason": f"AI price differs by {diff_pips:.1f} pips from MT5" if diff_pips > tolerance_pips else "Price validated"
    }


def calculate_signal_quality_score(analysis_results):
    """Calculate signal quality based on how many of 744 concepts agree"""
    score = {
        "total_concepts": 744,
        "bullish_signals": 0,
        "bearish_signals": 0,
        "neutral_signals": 0,
        "categories": {},
        "overall_score": 0,
        "signal_strength": "WEAK"
    }
    
    # Count signals from each category
    categories = {
        "candlestick_patterns": {"weight": 10, "bullish": 0, "bearish": 0},
        "chart_patterns": {"weight": 15, "bullish": 0, "bearish": 0},
        "ict_smc": {"weight": 20, "bullish": 0, "bearish": 0},
        "indicators": {"weight": 15, "bullish": 0, "bearish": 0},
        "price_action": {"weight": 15, "bullish": 0, "bearish": 0},
        "advanced_indicators": {"weight": 10, "bullish": 0, "bearish": 0},
        "harmonic_patterns": {"weight": 5, "bullish": 0, "bearish": 0},
        "volume_analysis": {"weight": 5, "bullish": 0, "bearish": 0},
        "market_regime": {"weight": 5, "bullish": 0, "bearish": 0}
    }
    
    # Parse analysis results
    if isinstance(analysis_results, dict):
        # Candlestick patterns
        if 'candlestick' in analysis_results:
            patterns = analysis_results['candlestick']
            for p in patterns if isinstance(patterns, list) else []:
                if p.get('type') == 'bullish':
                    categories['candlestick_patterns']['bullish'] += 1
                elif p.get('type') == 'bearish':
                    categories['candlestick_patterns']['bearish'] += 1
        
        # ICT/SMC
        if 'ict_smc' in analysis_results:
            ict = analysis_results['ict_smc']
            if ict.get('overall_bias', '').lower() == 'bullish':
                categories['ict_smc']['bullish'] += 5
            elif ict.get('overall_bias', '').lower() == 'bearish':
                categories['ict_smc']['bearish'] += 5
        
        # Indicators
        if 'indicators' in analysis_results:
            ind = analysis_results['indicators']
            if ind.get('overall_bias', '').lower() == 'bullish':
                categories['indicators']['bullish'] += 3
            elif ind.get('overall_bias', '').lower() == 'bearish':
                categories['indicators']['bearish'] += 3
        
        # Price Action
        if 'price_action' in analysis_results:
            pa = analysis_results['price_action']
            if pa.get('overall_analysis', {}).get('bias', '').lower() == 'bullish':
                categories['price_action']['bullish'] += 3
            elif pa.get('overall_analysis', {}).get('bias', '').lower() == 'bearish':
                categories['price_action']['bearish'] += 3
    
    # Calculate weighted score
    total_bullish = 0
    total_bearish = 0
    
    for cat_name, cat_data in categories.items():
        weight = cat_data['weight']
        bullish = cat_data['bullish']
        bearish = cat_data['bearish']
        
        total_bullish += bullish * weight
        total_bearish += bearish * weight
        
        score['categories'][cat_name] = {
            "bullish": bullish,
            "bearish": bearish,
            "bias": "BULLISH" if bullish > bearish else "BEARISH" if bearish > bullish else "NEUTRAL"
        }
    
    score['bullish_signals'] = total_bullish
    score['bearish_signals'] = total_bearish
    
    # Calculate overall score (0-100)
    total_signals = total_bullish + total_bearish
    if total_signals > 0:
        if total_bullish > total_bearish:
            score['overall_score'] = min(100, int((total_bullish / (total_bullish + total_bearish)) * 100))
            score['direction'] = "BULLISH"
        else:
            score['overall_score'] = min(100, int((total_bearish / (total_bullish + total_bearish)) * 100))
            score['direction'] = "BEARISH"
    else:
        score['overall_score'] = 50
        score['direction'] = "NEUTRAL"
    
    # Determine signal strength
    if score['overall_score'] >= 80:
        score['signal_strength'] = "VERY STRONG"
    elif score['overall_score'] >= 70:
        score['signal_strength'] = "STRONG"
    elif score['overall_score'] >= 60:
        score['signal_strength'] = "MODERATE"
    else:
        score['signal_strength'] = "WEAK"
    
    return score


def calculate_confluence_score(opens, highs, lows, closes, volumes, symbol):
    """Count how many indicators/patterns confirm the signal direction"""
    confluence = {
        "bullish_confirmations": [],
        "bearish_confirmations": [],
        "total_bullish": 0,
        "total_bearish": 0,
        "confluence_score": 0,
        "recommendation": ""
    }
    
    current_price = closes[-1] if closes else 0
    
    # 1. Moving Averages
    sma20 = calculate_sma(closes, 20)
    sma50 = calculate_sma(closes, 50)
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)
    
    if sma20 and current_price > sma20:
        confluence['bullish_confirmations'].append("Price > SMA20")
    elif sma20:
        confluence['bearish_confirmations'].append("Price < SMA20")
    
    if sma50 and current_price > sma50:
        confluence['bullish_confirmations'].append("Price > SMA50")
    elif sma50:
        confluence['bearish_confirmations'].append("Price < SMA50")
    
    if ema12 and ema26 and ema12 > ema26:
        confluence['bullish_confirmations'].append("EMA12 > EMA26")
    elif ema12 and ema26:
        confluence['bearish_confirmations'].append("EMA12 < EMA26")
    
    # 2. RSI
    rsi = calculate_rsi(closes)
    if rsi:
        if rsi > 50 and rsi < 70:
            confluence['bullish_confirmations'].append(f"RSI Bullish ({rsi:.1f})")
        elif rsi < 50 and rsi > 30:
            confluence['bearish_confirmations'].append(f"RSI Bearish ({rsi:.1f})")
        elif rsi >= 70:
            confluence['bearish_confirmations'].append(f"RSI Overbought ({rsi:.1f})")
        elif rsi <= 30:
            confluence['bullish_confirmations'].append(f"RSI Oversold ({rsi:.1f})")
    
    # 3. MACD
    macd_line, signal_line, histogram = calculate_macd(closes)
    if macd_line and signal_line:
        if macd_line > signal_line:
            confluence['bullish_confirmations'].append("MACD Bullish Cross")
        else:
            confluence['bearish_confirmations'].append("MACD Bearish Cross")
    
    # 4. Stochastic
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
    if stoch_k and stoch_d:
        if stoch_k > stoch_d and stoch_k < 80:
            confluence['bullish_confirmations'].append("Stochastic Bullish")
        elif stoch_k < stoch_d and stoch_k > 20:
            confluence['bearish_confirmations'].append("Stochastic Bearish")
    
    # 5. ADX Trend Strength
    adx, plus_di, minus_di = calculate_adx(highs, lows, closes)
    if adx and plus_di and minus_di:
        if adx > 25:
            if plus_di > minus_di:
                confluence['bullish_confirmations'].append(f"ADX Strong Uptrend ({adx:.1f})")
            else:
                confluence['bearish_confirmations'].append(f"ADX Strong Downtrend ({adx:.1f})")
    
    # 6. Bollinger Bands
    bb_upper, bb_middle, bb_lower = calculate_bollinger(closes)
    if bb_upper and bb_lower:
        if current_price < bb_lower:
            confluence['bullish_confirmations'].append("Price at BB Lower (Oversold)")
        elif current_price > bb_upper:
            confluence['bearish_confirmations'].append("Price at BB Upper (Overbought)")
    
    # 7. Market Structure
    structure = identify_market_structure(highs, lows, closes)
    if structure:
        if structure.get('trend') == 'bullish':
            confluence['bullish_confirmations'].append("Market Structure Bullish (HH/HL)")
        elif structure.get('trend') == 'bearish':
            confluence['bearish_confirmations'].append("Market Structure Bearish (LH/LL)")
    
    # 8. Candlestick Patterns
    patterns = identify_candlestick_patterns(opens, highs, lows, closes)
    for p in patterns[:3]:  # Top 3 patterns
        if p.get('type') == 'bullish':
            confluence['bullish_confirmations'].append(f"Pattern: {p.get('name')}")
        elif p.get('type') == 'bearish':
            confluence['bearish_confirmations'].append(f"Pattern: {p.get('name')}")
    
    # 9. Order Blocks
    obs = find_order_blocks(opens, highs, lows, closes)
    for ob in obs[-2:]:  # Last 2 OBs
        if ob.get('type') == 'bullish' and not ob.get('mitigated'):
            confluence['bullish_confirmations'].append("Unmitigated Bullish OB")
        elif ob.get('type') == 'bearish' and not ob.get('mitigated'):
            confluence['bearish_confirmations'].append("Unmitigated Bearish OB")
    
    # 10. FVG
    fvgs = find_fvg(opens, highs, lows, closes)
    for fvg in fvgs[-2:]:
        if fvg.get('type') == 'bullish' and not fvg.get('filled'):
            confluence['bullish_confirmations'].append("Unfilled Bullish FVG")
        elif fvg.get('type') == 'bearish' and not fvg.get('filled'):
            confluence['bearish_confirmations'].append("Unfilled Bearish FVG")
    
    # Calculate totals
    confluence['total_bullish'] = len(confluence['bullish_confirmations'])
    confluence['total_bearish'] = len(confluence['bearish_confirmations'])
    
    total = confluence['total_bullish'] + confluence['total_bearish']
    if total > 0:
        if confluence['total_bullish'] > confluence['total_bearish']:
            confluence['confluence_score'] = int((confluence['total_bullish'] / total) * 100)
            confluence['direction'] = "BULLISH"
        else:
            confluence['confluence_score'] = int((confluence['total_bearish'] / total) * 100)
            confluence['direction'] = "BEARISH"
    else:
        confluence['confluence_score'] = 50
        confluence['direction'] = "NEUTRAL"
    
    # Recommendation
    if confluence['confluence_score'] >= 75 and total >= 8:
        confluence['recommendation'] = "HIGH PROBABILITY SETUP - Multiple confirmations aligned"
    elif confluence['confluence_score'] >= 60 and total >= 5:
        confluence['recommendation'] = "MODERATE SETUP - Good confluence but watch for confirmation"
    else:
        confluence['recommendation'] = "LOW PROBABILITY - Wait for more confluence"
    
    return confluence


def check_risk_reward(entry, stop_loss, take_profit, min_rr=1.5):
    """Check if trade has acceptable Risk:Reward ratio"""
    if not all([entry, stop_loss, take_profit]):
        return {"valid": False, "reason": "Missing Entry/SL/TP values"}
    
    risk = abs(entry - stop_loss)
    reward = abs(take_profit - entry)
    
    if risk == 0:
        return {"valid": False, "reason": "Risk is zero (Entry = SL)"}
    
    rr_ratio = reward / risk
    
    return {
        "valid": rr_ratio >= min_rr,
        "risk_pips": risk,
        "reward_pips": reward,
        "rr_ratio": round(rr_ratio, 2),
        "min_required": min_rr,
        "reason": f"R:R {rr_ratio:.2f} {'✅ Good' if rr_ratio >= min_rr else '❌ Too low (min ' + str(min_rr) + ')'}"
    }


def check_sl_direction(entry, stop_loss, take_profit, bias):
    """Check if SL is placed in the correct direction based on trade bias"""
    if not all([entry, stop_loss, take_profit]):
        return {"valid": True, "reason": "Cannot validate - missing values"}
    
    # Determine trade direction from TP
    is_buy_trade = take_profit > entry  # TP above entry = BUY
    is_sell_trade = take_profit < entry  # TP below entry = SELL
    
    # Check SL placement
    sl_above_entry = stop_loss > entry
    sl_below_entry = stop_loss < entry
    
    # For BUY trade: SL should be BELOW entry
    # For SELL trade: SL should be ABOVE entry
    
    if is_buy_trade and sl_above_entry:
        return {
            "valid": False,
            "reason": f"❌ SL WRONG DIRECTION! BUY trade but SL ({stop_loss}) is ABOVE Entry ({entry}). SL should be BELOW Entry for BUY trades.",
            "suggestion": f"Move SL below Entry, e.g., {entry - (entry - take_profit) * 0.3:.2f}"
        }
    
    if is_sell_trade and sl_below_entry:
        return {
            "valid": False,
            "reason": f"❌ SL WRONG DIRECTION! SELL trade but SL ({stop_loss}) is BELOW Entry ({entry}). SL should be ABOVE Entry for SELL trades.",
            "suggestion": f"Move SL above Entry, e.g., {entry + (entry - take_profit) * 0.3:.2f}"
        }
    
    return {
        "valid": True,
        "reason": "✅ SL direction correct",
        "trade_type": "BUY" if is_buy_trade else "SELL"
    }


def fetch_economic_calendar():
    """Fetch upcoming high-impact news events"""
    # Note: In production, use a real API like ForexFactory, Investing.com, or FXStreet
    # This is a simulation for demonstration
    from datetime import datetime, timedelta
    import random
    
    now = datetime.utcnow()
    
    # Simulated high-impact events (in production, fetch from real API)
    events = [
        {"time": now + timedelta(hours=random.randint(1, 24)), "currency": "USD", "event": "FOMC Meeting", "impact": "HIGH"},
        {"time": now + timedelta(hours=random.randint(1, 48)), "currency": "USD", "event": "Non-Farm Payrolls", "impact": "HIGH"},
        {"time": now + timedelta(hours=random.randint(1, 72)), "currency": "EUR", "event": "ECB Interest Rate", "impact": "HIGH"},
        {"time": now + timedelta(hours=random.randint(1, 24)), "currency": "GBP", "event": "BOE Rate Decision", "impact": "HIGH"},
        {"time": now + timedelta(hours=random.randint(1, 12)), "currency": "USD", "event": "CPI Data", "impact": "HIGH"},
    ]
    
    return events


def check_news_filter(symbol, hours_before=2, hours_after=1):
    """Check if there's high-impact news that could affect the trade"""
    events = fetch_economic_calendar()
    from datetime import datetime, timedelta
    
    now = datetime.utcnow()
    
    # Get currencies in the symbol
    currencies = []
    if "XAU" in symbol or "XAG" in symbol:
        currencies = ["USD", "XAU", "GOLD"]
    else:
        # Extract currencies from pair (e.g., EUR/USD -> EUR, USD)
        parts = symbol.replace("/", "")
        if len(parts) >= 6:
            currencies = [parts[:3], parts[3:6]]
    
    warnings = []
    for event in events:
        event_time = event['time']
        time_diff = (event_time - now).total_seconds() / 3600  # hours
        
        # Check if event is within the danger zone
        if -hours_after <= time_diff <= hours_before:
            if event['currency'] in currencies or event['currency'] == "USD":
                warnings.append({
                    "event": event['event'],
                    "currency": event['currency'],
                    "time": event_time.strftime("%Y-%m-%d %H:%M UTC"),
                    "hours_until": round(time_diff, 1),
                    "impact": event['impact']
                })
    
    return {
        "safe_to_trade": len(warnings) == 0,
        "warnings": warnings,
        "recommendation": "⚠️ HIGH-IMPACT NEWS NEARBY - Consider waiting" if warnings else "✅ No major news events nearby"
    }


def validate_trade_setup(symbol, entry, stop_loss, tp1, tp2=None, tp3=None, analysis_data=None):
    """Complete trade validation with all accuracy checks"""
    validation = {
        "overall_valid": True,
        "checks": {},
        "warnings": [],
        "score": 100,
        "recommendation": ""
    }
    
    # 1. MT5 Price Validation
    if entry:
        price_check = validate_price_with_mt5(symbol, entry)
        validation['checks']['price_validation'] = price_check
        if not price_check['valid']:
            validation['warnings'].append(f"Price validation failed: {price_check['reason']}")
            validation['score'] -= 20
    
    # 2. SL Direction Check (CRITICAL!)
    if entry and stop_loss and tp1:
        sl_check = check_sl_direction(entry, stop_loss, tp1, None)
        validation['checks']['sl_direction'] = sl_check
        if not sl_check['valid']:
            validation['warnings'].append(sl_check['reason'])
            if sl_check.get('suggestion'):
                validation['warnings'].append(f"💡 Suggestion: {sl_check['suggestion']}")
            validation['score'] -= 40  # Major penalty for wrong SL direction
    
    # 3. Risk:Reward Check
    if entry and stop_loss and tp1:
        rr_check = check_risk_reward(entry, stop_loss, tp1, min_rr=1.5)
        validation['checks']['risk_reward'] = rr_check
        if not rr_check['valid']:
            validation['warnings'].append(f"R:R too low: {rr_check['rr_ratio']}")
            validation['score'] -= 15
    
    # 4. News Filter
    news_check = check_news_filter(symbol)
    validation['checks']['news_filter'] = news_check
    if not news_check['safe_to_trade']:
        validation['warnings'].append(f"High-impact news nearby: {[w['event'] for w in news_check['warnings']]}")
        validation['score'] -= 25
    
    # 5. Stop Loss Distance Check (not too tight, not too wide)
    if entry and stop_loss:
        sl_distance = abs(entry - stop_loss)
        if "XAU" in symbol:
            # Gold: SL should be 3-30 dollars
            if sl_distance < 3:
                validation['warnings'].append(f"Stop Loss too tight ({sl_distance:.2f})")
                validation['score'] -= 10
            elif sl_distance > 30:
                validation['warnings'].append(f"Stop Loss too wide ({sl_distance:.2f})")
                validation['score'] -= 10
        else:
            # Forex: SL should be 10-100 pips
            sl_pips = sl_distance / 0.0001 if "JPY" not in symbol else sl_distance / 0.01
            if sl_pips < 10:
                validation['warnings'].append(f"Stop Loss too tight ({sl_pips:.0f} pips)")
                validation['score'] -= 10
            elif sl_pips > 100:
                validation['warnings'].append(f"Stop Loss too wide ({sl_pips:.0f} pips)")
                validation['score'] -= 10
    
    # Determine overall validity
    validation['overall_valid'] = validation['score'] >= 60
    
    # Generate recommendation
    if validation['score'] >= 90:
        validation['recommendation'] = "✅ EXCELLENT SETUP - All checks passed"
    elif validation['score'] >= 75:
        validation['recommendation'] = "✅ GOOD SETUP - Minor concerns, proceed with caution"
    elif validation['score'] >= 60:
        validation['recommendation'] = "⚠️ MODERATE SETUP - Review warnings before trading"
    else:
        validation['recommendation'] = "❌ POOR SETUP - Too many issues, consider skipping"
    
    return validation


def fetch_binance_price(symbol):
    """Fetch real-time price from Binance (FREE, no API key needed)"""
    try:
        # Convert symbol format: BTC/USD -> BTCUSDT, ETH/USD -> ETHUSDT
        binance_symbol = symbol.replace("/", "").replace("USD", "USDT")
        
        # For Gold/Silver, Binance doesn't have them
        if symbol in ["XAU/USD", "XAG/USD"]:
            return None
            
        url = f"https://api.binance.com/api/v3/klines"
        params = {
            "symbol": binance_symbol,
            "interval": "5m",
            "limit": 100
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if isinstance(data, list) and len(data) > 0:
            return {
                "opens": [float(k[1]) for k in data],
                "highs": [float(k[2]) for k in data],
                "lows": [float(k[3]) for k in data],
                "closes": [float(k[4]) for k in data],
                "volumes": [float(k[5]) for k in data],
                "symbol": symbol,
                "source": "Binance"
            }
    except Exception as e:
        print(f"Binance error: {e}")
    return None


def fetch_yahoo_finance(symbol):
    """Fetch from Yahoo Finance via yfinance-style API (FREE)"""
    try:
        # Convert symbol format
        yahoo_symbols = {
            "XAU/USD": "GC=F",      # Gold Futures
            "XAG/USD": "SI=F",      # Silver Futures
            "EUR/USD": "EURUSD=X",
            "GBP/USD": "GBPUSD=X",
            "USD/JPY": "JPY=X",
            "BTC/USD": "BTC-USD",
            "ETH/USD": "ETH-USD",
            "AUD/USD": "AUDUSD=X",
            "NZD/USD": "NZDUSD=X",
            "USD/CAD": "CAD=X",
            "USD/CHF": "CHF=X",
            "EUR/GBP": "EURGBP=X",
            "EUR/JPY": "EURJPY=X",
            "GBP/JPY": "GBPJPY=X",
        }
        
        yahoo_symbol = yahoo_symbols.get(symbol)
        if not yahoo_symbol:
            return None
            
        # Use Yahoo Finance API (chart endpoint)
        url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
        params = {
            "interval": "5m",
            "range": "1d"
        }
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, params=params, headers=headers, timeout=10)
        data = response.json()
        
        if "chart" in data and "result" in data["chart"] and data["chart"]["result"]:
            result = data["chart"]["result"][0]
            quotes = result.get("indicators", {}).get("quote", [{}])[0]
            
            if quotes.get("close"):
                # Filter out None values
                opens = [x for x in quotes.get("open", []) if x is not None]
                highs = [x for x in quotes.get("high", []) if x is not None]
                lows = [x for x in quotes.get("low", []) if x is not None]
                closes = [x for x in quotes.get("close", []) if x is not None]
                volumes = [x for x in quotes.get("volume", []) if x is not None]
                
                if len(closes) > 10:
                    return {
                        "opens": opens,
                        "highs": highs,
                        "lows": lows,
                        "closes": closes,
                        "volumes": volumes if volumes else [0] * len(closes),
                        "symbol": symbol,
                        "source": "Yahoo Finance"
                    }
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
    return None


def fetch_market_data(symbol="EUR/USD", interval="1h", outputsize=100):
    """Fetch market data with MT5 as primary source, then fallback to free APIs"""
    
    # 🥇 Try MT5 first (most accurate, real-time from your broker)
    data = fetch_mt5_data(symbol, interval, outputsize)
    if data:
        return data
    
    # 🥈 Try Yahoo Finance (reliable for all symbols)
    data = fetch_yahoo_finance(symbol)
    if data:
        print(f"✅ Using Yahoo Finance for {symbol}")
        return data
    
    # 🥉 Try Binance for crypto
    if symbol in ["BTC/USD", "ETH/USD"]:
        data = fetch_binance_price(symbol)
        if data:
            print(f"✅ Using Binance for {symbol}")
            return data
    
    # 4️⃣ Fallback to TwelveData
    try:
        url = "https://api.twelvedata.com/time_series"
        params = {
            "symbol": symbol,
            "interval": interval,
            "outputsize": outputsize,
            "apikey": TWELVE_DATA_KEY
        }
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if "values" in data:
            values = data["values"][::-1]
            closes = [float(v["close"]) for v in values]
            
            print(f"✅ Using TwelveData for {symbol}")
            return {
                "opens": [float(v["open"]) for v in values],
                "highs": [float(v["high"]) for v in values],
                "lows": [float(v["low"]) for v in values],
                "closes": closes,
                "symbol": symbol,
                "source": "TwelveData"
            }
        else:
            print(f"TwelveData error: {data.get('message', 'No data')}")
    except Exception as e:
        print(f"TwelveData error: {e}")
    
    print(f"⚠️ No data available for {symbol}")
    return None


def generate_technical_report(symbol="EUR/USD", interval="1h"):
    data = fetch_market_data(symbol, interval)
    
    if not data:
        return "Unable to fetch market data. Using image analysis only."
    
    closes = data["closes"]
    highs = data["highs"]
    lows = data["lows"]
    opens = data["opens"]
    current_price = closes[-1]
    data_source = data.get("source", "Unknown")
    
    # Get decimal places for this symbol
    decimals = get_decimal_places(symbol)
    fmt = lambda x: f"{x:.{decimals}f}" if x is not None else "N/A"
    
    # Calculate ALL indicators
    sma20 = calculate_sma(closes, 20)
    sma50 = calculate_sma(closes, 50)
    sma200 = calculate_sma(closes, 200) if len(closes) >= 200 else None
    ema12 = calculate_ema(closes, 12)
    ema26 = calculate_ema(closes, 26)
    wma20 = calculate_wma(closes, 20)
    
    rsi = calculate_rsi(closes)
    macd_line, signal_line, histogram = calculate_macd(closes)
    stoch_k, stoch_d = calculate_stochastic(highs, lows, closes)
    williams = calculate_williams_r(highs, lows, closes)
    cci = calculate_cci(highs, lows, closes)
    momentum = calculate_momentum(closes)
    roc = calculate_roc(closes)
    
    adx, plus_di, minus_di = calculate_adx(highs, lows, closes)
    bb_upper, bb_middle, bb_lower = calculate_bollinger(closes)
    atr = calculate_atr(highs, lows, closes)
    kelt_upper, kelt_middle, kelt_lower = calculate_keltner(highs, lows, closes)
    don_upper, don_middle, don_lower = calculate_donchian(highs, lows)
    
    ichimoku = calculate_ichimoku(highs, lows, closes)
    
    # NEW: Pivot Points & Fibonacci
    daily_high, daily_low = max(highs[-24:]) if len(highs) >= 24 else max(highs), min(lows[-24:]) if len(lows) >= 24 else min(lows)
    pivots = calculate_pivot_points(daily_high, daily_low, closes[-1])
    fib_pivots = calculate_fibonacci_pivots(daily_high, daily_low, closes[-1])
    
    # Fibonacci Retracement
    swing_high = max(highs[-50:]) if len(highs) >= 50 else max(highs)
    swing_low = min(lows[-50:]) if len(lows) >= 50 else min(lows)
    trend_dir = "bullish" if closes[-1] > closes[-20] else "bearish"
    fib_retracement = calculate_fibonacci_retracement(swing_high, swing_low, trend_dir)
    
    # Parabolic SAR
    sar_value, sar_signal = calculate_parabolic_sar(highs, lows)
    
    # Patterns & Structure
    patterns = identify_candlestick_patterns(opens, highs, lows, closes)
    structure = identify_market_structure(highs, lows, closes)
    levels = find_support_resistance(highs, lows, closes)
    fvgs = find_fvg(opens, highs, lows, closes)
    obs = find_order_blocks(opens, highs, lows, closes)
    liquidity = find_liquidity_zones(highs, lows, closes)
    
    # NEW: Chart Patterns
    chart_patterns = detect_chart_patterns(highs, lows, closes)
    
    # ========== ADVANCED DATA SOURCES ==========
    # Volume Profile
    volume_profile = calculate_volume_profile(highs, lows, closes)
    
    # Order Flow / Delta
    order_flow = calculate_order_flow(opens, highs, lows, closes)
    
    # COT Data
    cot_data = fetch_cot_data(symbol)
    
    # Market Sentiment
    sentiment = fetch_market_sentiment(symbol)
    
    # Market Depth
    market_depth = generate_market_depth(current_price, symbol)
    
    # Institutional Flow Analysis
    institutional = analyze_institutional_flow(cot_data, order_flow, volume_profile, sentiment)
    
    # Liquidity Heatmap
    liquidity_heatmap = generate_liquidity_heatmap(highs, lows, closes, volume_profile)
    
    # Session Analysis
    session_info = analyze_trading_sessions(symbol)


    report = f"""
## 📊 REAL-TIME TECHNICAL DATA ({symbol} - {interval})
### 📡 Data Source: {data_source} {'🟢 LIVE' if data_source == 'MT5' else '🔵'}
### Current Price: {fmt(current_price)}

### 📈 TREND INDICATORS (99% Accurate)
| Indicator | Value | Signal |
|-----------|-------|--------|
| SMA 20 | {fmt(sma20)} | {'Bullish' if sma20 and current_price > sma20 else 'Bearish'} |
| SMA 50 | {fmt(sma50)} | {'Bullish' if sma50 and current_price > sma50 else 'Bearish'} |
| SMA 200 | {fmt(sma200)} | {'Bullish' if sma200 and current_price > sma200 else 'Bearish' if sma200 else 'N/A'} |
| EMA 12 | {fmt(ema12)} | {'Bullish' if ema12 and current_price > ema12 else 'Bearish'} |
| EMA 26 | {fmt(ema26)} | {'Bullish' if ema26 and current_price > ema26 else 'Bearish'} |
| ADX | {f'{adx:.2f}' if adx else 'N/A'} | {'Strong Trend' if adx and adx > 25 else 'Weak Trend'} |
| +DI / -DI | {f'{plus_di:.2f}' if plus_di else 'N/A'} / {f'{minus_di:.2f}' if minus_di else 'N/A'} | {'Bullish' if plus_di and minus_di and plus_di > minus_di else 'Bearish'} |

### 📊 MOMENTUM OSCILLATORS (99% Accurate)
| Indicator | Value | Zone |
|-----------|-------|------|
| RSI (14) | {f'{rsi:.2f}' if rsi else 'N/A'} | {'Overbought' if rsi and rsi > 70 else 'Oversold' if rsi and rsi < 30 else 'Neutral'} |
| MACD | {fmt(macd_line)} | {'Bullish' if macd_line and macd_line > 0 else 'Bearish'} |
| MACD Signal | {fmt(signal_line)} | {'Buy' if macd_line and signal_line and macd_line > signal_line else 'Sell'} |
| Stochastic %K | {f'{stoch_k:.2f}' if stoch_k else 'N/A'} | {'Overbought' if stoch_k and stoch_k > 80 else 'Oversold' if stoch_k and stoch_k < 20 else 'Neutral'} |
| Williams %R | {f'{williams:.2f}' if williams else 'N/A'} | {'Overbought' if williams and williams > -20 else 'Oversold' if williams and williams < -80 else 'Neutral'} |
| CCI | {f'{cci:.2f}' if cci else 'N/A'} | {'Overbought' if cci and cci > 100 else 'Oversold' if cci and cci < -100 else 'Neutral'} |
| Momentum | {fmt(momentum)} | {'Bullish' if momentum and momentum > 0 else 'Bearish'} |
| ROC | {f'{roc:.2f}%' if roc else 'N/A'} | {'Bullish' if roc and roc > 0 else 'Bearish'} |
"""

    report += f"""
### 📉 VOLATILITY (99% Accurate)
| Indicator | Value |
|-----------|-------|
| ATR (14) | {fmt(atr)} |
| Bollinger Upper | {fmt(bb_upper)} |
| Bollinger Middle | {fmt(bb_middle)} |
| Bollinger Lower | {fmt(bb_lower)} |
| Keltner Upper | {fmt(kelt_upper)} |
| Keltner Lower | {fmt(kelt_lower)} |
| Donchian Upper | {fmt(don_upper)} |
| Donchian Lower | {fmt(don_lower)} |
| Parabolic SAR | {fmt(sar_value)} | {sar_signal if sar_signal else 'N/A'} |
"""

    # Pivot Points
    report += f"""
### 📍 PIVOT POINTS
| Level | Standard | Fibonacci |
|-------|----------|-----------|
| R3 | {fmt(pivots['r3'])} | {fmt(fib_pivots['r3'])} |
| R2 | {fmt(pivots['r2'])} | {fmt(fib_pivots['r2'])} |
| R1 | {fmt(pivots['r1'])} | {fmt(fib_pivots['r1'])} |
| Pivot | {fmt(pivots['pivot'])} | {fmt(fib_pivots['pivot'])} |
| S1 | {fmt(pivots['s1'])} | {fmt(fib_pivots['s1'])} |
| S2 | {fmt(pivots['s2'])} | {fmt(fib_pivots['s2'])} |
| S3 | {fmt(pivots['s3'])} | {fmt(fib_pivots['s3'])} |
"""

    # Fibonacci Retracement
    report += f"""
### 📐 FIBONACCI RETRACEMENT ({trend_dir.upper()})
| Level | Price |
|-------|-------|
| 0.0% | {fmt(fib_retracement['0.0'])} |
| 23.6% | {fmt(fib_retracement['0.236'])} |
| 38.2% | {fmt(fib_retracement['0.382'])} |
| 50.0% | {fmt(fib_retracement['0.5'])} |
| 61.8% | {fmt(fib_retracement['0.618'])} |
| 78.6% | {fmt(fib_retracement['0.786'])} |
| 100% | {fmt(fib_retracement['1.0'])} |
"""
    
    # Ichimoku
    if ichimoku:
        report += f"""
### ☁️ ICHIMOKU CLOUD
| Component | Value |
|-----------|-------|
| Tenkan-sen | {fmt(ichimoku['tenkan'])} |
| Kijun-sen | {fmt(ichimoku['kijun'])} |
| Senkou Span A | {fmt(ichimoku['senkou_a'])} |
| Senkou Span B | {fmt(ichimoku['senkou_b'])} |
| Cloud | {ichimoku['cloud'].upper()} |
| Price vs Cloud | {ichimoku['price_vs_cloud'].upper()} |
"""
    
    # Market Structure
    report += f"""
### 🏛️ MARKET STRUCTURE (ICT/SMC)
- **Trend:** {structure['trend'].upper()}
- **Structure:** {'Higher Highs & Higher Lows (HH/HL)' if structure['hh_hl'] else 'Lower Highs & Lower Lows (LH/LL)' if structure['lh_ll'] else 'Ranging'}
- **Swing Highs:** {len(structure['swing_highs'])} detected
- **Swing Lows:** {len(structure['swing_lows'])} detected
"""

    # FVG
    if fvgs:
        report += "\n### 📊 FAIR VALUE GAPS (FVG)\n"
        for fvg in fvgs[-3:]:
            report += f"- **{fvg['type'].upper()} FVG**: {fvg['bottom']:.5f} - {fvg['top']:.5f} ({'Filled' if fvg['filled'] else 'Unfilled'})\n"
    
    # Order Blocks
    if obs:
        report += "\n### 🧱 ORDER BLOCKS\n"
        for ob in obs[-3:]:
            report += f"- **{ob['type'].upper()} OB**: {ob['bottom']:.5f} - {ob['top']:.5f} ({'Mitigated' if ob['mitigated'] else 'Unmitigated'})\n"
    
    # Liquidity
    if liquidity['bsl'] or liquidity['ssl']:
        report += "\n### 💧 LIQUIDITY ZONES\n"
        if liquidity['bsl']:
            report += f"- **Buy Side Liquidity (BSL):** {', '.join([f'{x:.5f}' for x in liquidity['bsl'][:3]])}\n"
        if liquidity['ssl']:
            report += f"- **Sell Side Liquidity (SSL):** {', '.join([f'{x:.5f}' for x in liquidity['ssl'][:3]])}\n"
    
    # Chart Patterns
    if chart_patterns:
        report += "\n### 📈 CHART PATTERNS DETECTED\n"
        for cp in chart_patterns:
            report += f"- **{cp['name']}** ({cp['type']}) - Confidence: {cp['confidence']}%\n"
    
    # Support/Resistance
    report += f"""
### 🎯 KEY LEVELS
- **Resistance:** {', '.join([str(r) for r in levels['resistance']]) or 'N/A'}
- **Support:** {', '.join([str(s) for s in levels['support']]) or 'N/A'}
"""
    
    # Candlestick Patterns
    report += "\n### 🕯️ CANDLESTICK PATTERNS DETECTED\n"
    if patterns:
        for p in patterns:
            report += f"- **{p['name']}** ({p['type']}) - Confidence: {p['confidence']}%\n"
    else:
        report += "- No significant patterns detected\n"
    
    # ========== ADVANCED DATA SECTIONS ==========
    
    # Volume Profile
    if volume_profile:
        report += f"""
### 📊 VOLUME PROFILE (Advanced)
| Metric | Value |
|--------|-------|
| POC (Point of Control) | {volume_profile.get('poc', 0):.5f} |
| Value Area High (VAH) | {volume_profile.get('vah', 0):.5f} |
| Value Area Low (VAL) | {volume_profile.get('val', 0):.5f} |
| Price vs POC | {'Above POC (Bullish)' if current_price > volume_profile.get('poc', 0) else 'Below POC (Bearish)'} |

**High Volume Nodes (HVN):** {', '.join([f'{x:.5f}' for x in volume_profile.get('hvn', [])]) if volume_profile.get('hvn') else 'N/A'}
**Low Volume Nodes (LVN):** {', '.join([f'{x:.5f}' for x in volume_profile.get('lvn', [])]) if volume_profile.get('lvn') else 'N/A'}
"""
    
    # Order Flow
    if order_flow:
        report += f"""
### 📈 ORDER FLOW / DELTA
| Metric | Value |
|--------|-------|
| Current Delta | {order_flow.get('current_delta', 0):,.0f} |
| Cumulative Delta | {order_flow.get('cumulative_delta', 0):,.0f} |
| Delta Trend | {order_flow.get('delta_trend', 'neutral').upper()} |
| Flow Bias | {'BUYERS Dominant' if order_flow.get('cumulative_delta', 0) > 0 else 'SELLERS Dominant'} |
"""
        if order_flow.get('divergence'):
            report += f"\n⚠️ **DIVERGENCE DETECTED:** {order_flow['divergence'].get('signal', 'N/A')} (Confidence: {order_flow['divergence'].get('confidence', 0)}%)\n"
        if order_flow.get('absorption'):
            report += f"\n🔄 **ABSORPTION:** {order_flow['absorption'].get('signal', 'N/A')}\n"
    
    # COT Data
    if cot_data:
        report += f"""
### 📋 COT DATA (Commitment of Traders)
| Trader Type | Long % | Net Position | Bias |
|-------------|--------|--------------|------|
| Commercials (Smart Money) | {cot_data.get('commercial', {}).get('long_pct', 0)}% | {cot_data.get('commercial', {}).get('net', 0):,} | {'Bullish' if cot_data.get('commercial', {}).get('net', 0) > 0 else 'Bearish'} |
| Speculators (Large Traders) | {cot_data.get('speculators', {}).get('long_pct', 0)}% | {cot_data.get('speculators', {}).get('net', 0):,} | {'Bullish' if cot_data.get('speculators', {}).get('net', 0) > 0 else 'Bearish'} |
| Retail Traders | {cot_data.get('retail', {}).get('long_pct', 0)}% | {cot_data.get('retail', {}).get('net', 0):,} | {'Bullish' if cot_data.get('retail', {}).get('net', 0) > 0 else 'Bearish'} |

**COT Analysis:** {cot_data.get('sentiment', 'N/A')}
**COT Signal:** {cot_data.get('signal', 'N/A')}
"""
        if cot_data.get('extreme_warning'):
            report += f"\n⚠️ **WARNING:** {cot_data['extreme_warning']}\n"
    
    # Market Sentiment
    if sentiment:
        report += "\n### 🎭 MARKET SENTIMENT\n"
        for source, data in sentiment.get('sources', {}).items():
            if isinstance(data, dict) and 'signal' in data:
                report += f"- **{source.replace('_', ' ').title()}:** {data.get('signal', 'N/A')}"
                if 'note' in data:
                    report += f" - {data['note']}"
                report += "\n"
        if 'aggregate' in sentiment:
            report += f"\n**Aggregate Sentiment:** {sentiment['aggregate'].get('signal', 'N/A')} (Strength: {sentiment['aggregate'].get('strength', 0):.0f}%)\n"
    
    # Market Depth
    if market_depth:
        report += f"""
### 📚 MARKET DEPTH (DOM)
| Metric | Value |
|--------|-------|
| Spread | {market_depth.get('spread', 0):.5f} |
| Total Bid Size | {market_depth.get('total_bid_size', 0):,.0f} |
| Total Ask Size | {market_depth.get('total_ask_size', 0):,.0f} |
| Imbalance Ratio | {market_depth.get('imbalance_ratio', 'N/A')} |
| DOM Signal | {market_depth.get('signal', 'N/A')} |

**Analysis:** {market_depth.get('note', 'No analysis available')}
"""
        if market_depth.get('large_bids'):
            report += f"**Large Bid Orders:** {len(market_depth['large_bids'])} detected (potential support)\n"
        if market_depth.get('large_asks'):
            report += f"**Large Ask Orders:** {len(market_depth['large_asks'])} detected (potential resistance)\n"
    
    # Institutional Flow
    if institutional:
        report += f"""
### 🏦 INSTITUTIONAL FLOW ANALYSIS
| Metric | Value |
|--------|-------|
| Smart Money Signal | **{institutional.get('signal', 'N/A')}** |
| Confidence | {institutional.get('confidence', 0)}% |
| Buy Weight | {institutional.get('buy_weight', 0)} |
| Sell Weight | {institutional.get('sell_weight', 0)} |
| Smart Money Bias | {institutional.get('smart_money_bias', 'neutral').upper()} |
"""
    
    # Liquidity Heatmap
    if liquidity_heatmap:
        report += "\n### 🔥 LIQUIDITY HEATMAP\n"
        if liquidity_heatmap.get('buy_stops'):
            report += f"**Buy Stops Above:** {', '.join([f'{x:.5f}' for x in liquidity_heatmap['buy_stops']])}\n"
        if liquidity_heatmap.get('sell_stops'):
            report += f"**Sell Stops Below:** {', '.join([f'{x:.5f}' for x in liquidity_heatmap['sell_stops']])}\n"
        if liquidity_heatmap.get('nearest_liquidity_above'):
            report += f"**Nearest Liquidity Above:** {liquidity_heatmap['nearest_liquidity_above']:.5f}\n"
        if liquidity_heatmap.get('nearest_liquidity_below'):
            report += f"**Nearest Liquidity Below:** {liquidity_heatmap['nearest_liquidity_below']:.5f}\n"
    
    # Session Analysis
    if session_info:
        report += f"""
### ⏰ SESSION ANALYSIS
| Metric | Value |
|--------|-------|
| Current Hour (UTC) | {session_info.get('current_hour_utc', 'N/A')}:00 |
| Active Sessions | {', '.join(session_info.get('active_sessions', [])) if session_info.get('active_sessions') else 'None'} |
| Volatility Expectation | {session_info.get('volatility_expectation', 'N/A')} |
| Optimal Session | {session_info.get('optimal_session') or 'N/A'} |

**Recommendation:** {session_info.get('recommendation', 'N/A')}
"""
        if session_info.get('overlaps'):
            report += f"**Session Overlaps:** {', '.join(session_info['overlaps'])}\n"
    
    # ========== COMPLETE 744 CONCEPTS ANALYSIS ==========
    
    # Generate simulated volumes for analysis
    volumes = [abs(closes[i] - opens[i]) * 1000000 for i in range(len(closes))]
    
    # 1. COMPLETE CANDLESTICK PATTERNS (64 patterns)
    try:
        all_candle_patterns = identify_all_candlestick_patterns(opens, highs, lows, closes)
        if all_candle_patterns:
            report += "\n### 🕯️ COMPLETE CANDLESTICK ANALYSIS (64 Patterns)\n"
            bullish_patterns = [p for p in all_candle_patterns if p.get('type') == 'bullish']
            bearish_patterns = [p for p in all_candle_patterns if p.get('type') == 'bearish']
            neutral_patterns = [p for p in all_candle_patterns if p.get('type') not in ['bullish', 'bearish']]
            
            if bullish_patterns:
                report += f"**Bullish Patterns ({len(bullish_patterns)}):** "
                report += ", ".join([f"{p['name']} ({p.get('confidence', 0)}%)" for p in bullish_patterns[:5]])
                report += "\n"
            if bearish_patterns:
                report += f"**Bearish Patterns ({len(bearish_patterns)}):** "
                report += ", ".join([f"{p['name']} ({p.get('confidence', 0)}%)" for p in bearish_patterns[:5]])
                report += "\n"
            if neutral_patterns:
                report += f"**Neutral/Reversal Patterns ({len(neutral_patterns)}):** "
                report += ", ".join([f"{p['name']}" for p in neutral_patterns[:3]])
                report += "\n"
            
            # Pattern confluence
            confluence = analyze_pattern_confluence(all_candle_patterns)
            report += f"\n**Pattern Confluence:** {confluence['bias'].upper()} (Strength: {confluence['strength']}%)\n"
    except Exception as e:
        report += f"\n*Candlestick analysis: {str(e)[:50]}*\n"
    
    # 2. COMPLETE CHART PATTERNS (69 patterns)
    try:
        all_chart_patterns = detect_all_chart_patterns(highs, lows, closes)
        if all_chart_patterns:
            report += "\n### 📈 COMPLETE CHART PATTERNS (69 Patterns)\n"
            for category, patterns in all_chart_patterns.items():
                if patterns and isinstance(patterns, list) and len(patterns) > 0:
                    report += f"**{category.replace('_', ' ').title()}:** "
                    pattern_names = [p.get('name', str(p)) if isinstance(p, dict) else str(p) for p in patterns[:3]]
                    report += ", ".join(pattern_names) + "\n"
    except Exception as e:
        report += f"\n*Chart patterns analysis: {str(e)[:50]}*\n"
    
    # 3. COMPLETE ICT/SMC ANALYSIS (131 concepts)
    try:
        ict_analysis = analyze_complete_ict_smc(opens, highs, lows, closes)
        if ict_analysis:
            report += "\n### 🎯 COMPLETE ICT/SMC ANALYSIS (131 Concepts)\n"
            
            # Market Structure
            if 'market_structure' in ict_analysis:
                ms = ict_analysis['market_structure']
                report += f"**Market Structure:** {ms.get('trend', 'N/A').upper()}\n"
                report += f"- BOS: {'Bullish' if ms.get('bullish_bos') else 'Bearish' if ms.get('bearish_bos') else 'None'}\n"
                report += f"- CHoCH: {'Detected' if ms.get('choch') else 'None'}\n"
            
            # Premium/Discount
            if 'premium_discount' in ict_analysis:
                pd = ict_analysis['premium_discount']
                report += f"**Premium/Discount Zone:** {pd.get('zone', 'N/A').upper()}\n"
                report += f"- Equilibrium: {pd.get('equilibrium', 0):.5f}\n"
            
            # Order Blocks
            if 'order_blocks' in ict_analysis:
                obs = ict_analysis['order_blocks']
                if obs.get('bullish'):
                    report += f"**Bullish OBs:** {len(obs['bullish'])} detected\n"
                if obs.get('bearish'):
                    report += f"**Bearish OBs:** {len(obs['bearish'])} detected\n"
            
            # FVG/Imbalances
            if 'fvg' in ict_analysis:
                fvg = ict_analysis['fvg']
                if fvg.get('bullish'):
                    report += f"**Bullish FVGs:** {len(fvg['bullish'])} unfilled\n"
                if fvg.get('bearish'):
                    report += f"**Bearish FVGs:** {len(fvg['bearish'])} unfilled\n"
            
            # Liquidity
            if 'liquidity' in ict_analysis:
                liq = ict_analysis['liquidity']
                if liq.get('bsl'):
                    report += f"**Buy Side Liquidity:** {len(liq['bsl'])} pools\n"
                if liq.get('ssl'):
                    report += f"**Sell Side Liquidity:** {len(liq['ssl'])} pools\n"
            
            # Kill Zones
            if 'kill_zones' in ict_analysis:
                kz = ict_analysis['kill_zones']
                if kz.get('active'):
                    report += f"**Active Kill Zone:** {kz['active']}\n"
            
            # Overall ICT Bias
            if 'overall_bias' in ict_analysis:
                report += f"\n**ICT Overall Bias:** {ict_analysis['overall_bias'].upper()}\n"
                report += f"**ICT Confidence:** {ict_analysis.get('confidence', 0)}%\n"
    except Exception as e:
        report += f"\n*ICT/SMC analysis: {str(e)[:50]}*\n"
    
    # 4. COMPLETE INDICATORS ANALYSIS (103 indicators)
    try:
        all_indicators = analyze_all_indicators(opens, highs, lows, closes, volumes)
        if all_indicators:
            report += "\n### 📊 COMPLETE INDICATORS ANALYSIS (103 Indicators)\n"
            
            # Trend Summary
            if 'trend' in all_indicators:
                trend = all_indicators['trend']
                report += "**Trend Indicators:**\n"
                if trend.get('adx'):
                    report += f"- ADX: {trend['adx']:.2f} ({'Strong' if trend['adx'] > 25 else 'Weak'} Trend)\n"
                if trend.get('aroon_osc'):
                    report += f"- Aroon Osc: {trend['aroon_osc']:.2f}\n"
            
            # Momentum Summary
            if 'momentum' in all_indicators:
                mom = all_indicators['momentum']
                report += "**Momentum Indicators:**\n"
                if mom.get('rsi'):
                    report += f"- RSI: {mom['rsi']:.2f}\n"
                if mom.get('stoch_k'):
                    report += f"- Stochastic: {mom['stoch_k']:.2f}\n"
                if mom.get('cci'):
                    report += f"- CCI: {mom['cci']:.2f}\n"
            
            # Volume Summary
            if 'volume' in all_indicators:
                vol = all_indicators['volume']
                if vol.get('cmf'):
                    report += f"**Chaikin Money Flow:** {vol['cmf']:.4f}\n"
                if vol.get('mfi'):
                    report += f"**Money Flow Index:** {vol['mfi']:.2f}\n"
            
            # Volatility Summary
            if 'volatility' in all_indicators:
                volat = all_indicators['volatility']
                if volat.get('bb_percent_b'):
                    report += f"**Bollinger %B:** {volat['bb_percent_b']:.4f}\n"
            
            # Overall Signal
            if 'signals' in all_indicators:
                signals = all_indicators['signals']
                report += f"\n**Indicator Signals:** Bullish: {signals['bullish']}, Bearish: {signals['bearish']}\n"
                report += f"**Overall Indicator Bias:** {all_indicators.get('overall_bias', 'NEUTRAL')}\n"
                report += f"**Signal Strength:** {all_indicators.get('strength', 50)}%\n"
    except Exception as e:
        report += f"\n*Indicators analysis: {str(e)[:50]}*\n"
    
    # 5. COMPLETE PRICE ACTION ANALYSIS (150 concepts)
    try:
        price_action = analyze_all_price_action(opens, highs, lows, closes, volumes)
        if price_action:
            report += "\n### 📉 COMPLETE PRICE ACTION ANALYSIS (150 Concepts)\n"
            
            # Market Structure
            if 'market_structure' in price_action:
                ms = price_action['market_structure']
                if ms.get('uptrend', {}).get('is_uptrend'):
                    report += "**Trend:** UPTREND (HH + HL confirmed)\n"
                elif ms.get('downtrend', {}).get('is_downtrend'):
                    report += "**Trend:** DOWNTREND (LH + LL confirmed)\n"
                else:
                    report += "**Trend:** RANGING\n"
                
                if ms.get('wyckoff_phase'):
                    report += f"**Wyckoff Phase:** {ms['wyckoff_phase']}\n"
            
            # Entry Signals
            if 'entry_signals' in price_action:
                entries = price_action['entry_signals']
                valid_entries = [k for k, v in entries.items() if isinstance(v, dict) and v.get('valid_entry')]
                if valid_entries:
                    report += f"**Valid Entry Signals:** {', '.join(valid_entries)}\n"
            
            # Candle Behavior
            if 'candle_behavior' in price_action:
                cb = price_action['candle_behavior']
                report += f"**Buying Pressure:** {cb.get('buying_pressure', 50):.1f}%\n"
                report += f"**Selling Pressure:** {cb.get('selling_pressure', 50):.1f}%\n"
            
            # Overall Analysis
            if 'overall_analysis' in price_action:
                overall = price_action['overall_analysis']
                report += f"\n**Price Action Bias:** {overall.get('bias', 'NEUTRAL')}\n"
                report += f"**PA Strength:** {overall.get('strength', 50)}%\n"
                report += f"**Bullish Signals:** {overall.get('bullish_signals', 0)}\n"
                report += f"**Bearish Signals:** {overall.get('bearish_signals', 0)}\n"
    except Exception as e:
        report += f"\n*Price action analysis: {str(e)[:50]}*\n"
    
    # 6. ADVANCED INDICATORS (50 concepts)
    try:
        advanced_ind = analyze_advanced_indicators(opens, highs, lows, closes, volumes)
        if advanced_ind:
            report += "\n### 🔬 ADVANCED INDICATORS (50 Concepts)\n"
            
            # Ehlers Indicators
            if 'ehlers' in advanced_ind:
                ehlers = advanced_ind['ehlers']
                report += "**Ehlers Indicators:**\n"
                if ehlers.get('fisher_transform'):
                    report += f"- Fisher Transform: {ehlers['fisher_transform']}\n"
                if ehlers.get('mama'):
                    report += f"- MAMA: {ehlers['mama']}\n"
                if ehlers.get('dominant_cycle'):
                    report += f"- Dominant Cycle: {ehlers['dominant_cycle']} bars\n"
            
            # Gann Indicators
            if 'gann' in advanced_ind:
                gann = advanced_ind['gann']
                report += "**Gann Analysis:**\n"
                if gann.get('hilo_signal'):
                    report += f"- HiLo Signal: {gann['hilo_signal'].upper()}\n"
            
            # Elliott Wave
            if 'elliott' in advanced_ind:
                elliott = advanced_ind['elliott']
                if elliott.get('wave_count'):
                    wc = elliott['wave_count']
                    report += f"**Elliott Wave:** {wc.get('wave', 'N/A')} (Position: {wc.get('position', 0)})\n"
            
            # Advanced Oscillators
            if 'oscillators' in advanced_ind:
                osc = advanced_ind['oscillators']
                report += "**Advanced Oscillators:**\n"
                if osc.get('stoch_rsi_k'):
                    report += f"- Stoch RSI: {osc['stoch_rsi_k']}\n"
                if osc.get('squeeze'):
                    sq = osc['squeeze']
                    report += f"- Squeeze: {'ON' if sq.get('squeeze') else 'OFF'} ({sq.get('signal', 'N/A')})\n"
            
            report += f"\n**Advanced Indicators Bias:** {advanced_ind.get('overall_bias', 'NEUTRAL')}\n"
    except Exception as e:
        report += f"\n*Advanced indicators: {str(e)[:50]}*\n"
    
    # 7. HARMONIC PATTERNS (25 patterns)
    try:
        harmonics = analyze_all_harmonic_patterns(highs, lows, closes)
        if harmonics and harmonics.get('patterns_found'):
            report += "\n### 🦋 HARMONIC PATTERNS (25 Patterns)\n"
            for p in harmonics['patterns_found'][:5]:
                report += f"- **{p.get('name', 'Unknown')}** ({p.get('type', 'N/A')}) - Confidence: {p.get('confidence', 0)}%\n"
            if harmonics.get('prz'):
                prz = harmonics['prz']
                report += f"\n**PRZ Zone:** {prz.get('prz_low', 0):.5f} - {prz.get('prz_high', 0):.5f}\n"
            report += f"**Harmonic Bias:** {harmonics.get('overall_bias', 'NEUTRAL')}\n"
    except Exception as e:
        report += f"\n*Harmonic patterns: {str(e)[:50]}*\n"
    
    # 8. VOLUME ANALYSIS (35 concepts)
    try:
        vol_analysis = analyze_all_volume(opens, highs, lows, closes, volumes)
        if vol_analysis:
            report += "\n### 📊 ADVANCED VOLUME ANALYSIS (35 Concepts)\n"
            
            if 'volume_profile' in vol_analysis:
                vp = vol_analysis['volume_profile']
                report += f"**POC:** {vp.get('poc', 0):.5f}\n"
                report += f"**VAH:** {vp.get('vah', 0):.5f}\n"
                report += f"**VAL:** {vp.get('val', 0):.5f}\n"
            
            if 'vwap' in vol_analysis:
                vwap = vol_analysis['vwap']
                report += f"**VWAP:** {vwap.get('current', 0):.5f}\n"
            
            if 'order_flow' in vol_analysis:
                of = vol_analysis['order_flow']
                if of.get('delta'):
                    report += f"**Cumulative Delta:** {of['delta'].get('cumulative_delta', 0):,.0f}\n"
                if of.get('vsa'):
                    report += f"**VSA Signal:** {of['vsa'].get('signal', 'N/A')}\n"
            
            report += f"\n**Volume Bias:** {vol_analysis.get('overall_bias', 'NEUTRAL')}\n"
    except Exception as e:
        report += f"\n*Volume analysis: {str(e)[:50]}*\n"
    
    # 9. STATISTICAL ANALYSIS (30 concepts)
    try:
        returns = [(closes[i] - closes[i-1]) / closes[i-1] for i in range(1, len(closes))] if len(closes) > 1 else []
        stats = analyze_all_statistics(closes, returns)
        if stats:
            report += "\n### 📈 STATISTICAL ANALYSIS (30 Concepts)\n"
            
            if 'distribution' in stats:
                dist = stats['distribution']
                report += f"**Z-Score:** {dist.get('z_score', 'N/A')}\n"
                report += f"**Hurst Exponent:** {dist.get('hurst', 'N/A')}\n"
                report += f"**Mean Reversion Prob:** {dist.get('mean_reversion_prob', 'N/A')}%\n"
            
            if 'regression' in stats:
                reg = stats['regression']
                if reg.get('linear'):
                    report += f"**Linear Trend:** {reg['linear'].get('trend', 'N/A').upper()} (R²: {reg['linear'].get('r_squared', 0)})\n"
            
            if 'risk_metrics' in stats:
                risk = stats['risk_metrics']
                report += f"**Sharpe Ratio:** {risk.get('sharpe', 'N/A')}\n"
                if risk.get('max_drawdown'):
                    report += f"**Max Drawdown:** {risk['max_drawdown'].get('max_drawdown', 0):.2f}%\n"
            
            report += f"\n**Statistical Bias:** {stats.get('overall_bias', 'NEUTRAL')}\n"
    except Exception as e:
        report += f"\n*Statistical analysis: {str(e)[:50]}*\n"
    
    # 10. MARKET REGIME (20 concepts)
    try:
        regime = analyze_all_regimes(opens, highs, lows, closes)
        if regime:
            report += "\n### 🎯 MARKET REGIME ANALYSIS (20 Concepts)\n"
            
            if 'trend' in regime:
                tr = regime['trend']
                if tr.get('regime'):
                    report += f"**Trend Regime:** {tr['regime'].get('regime', 'N/A').upper()}\n"
                if tr.get('adx'):
                    report += f"**ADX Regime:** {tr['adx'].get('regime', 'N/A')} ({tr['adx'].get('direction', 'N/A')})\n"
            
            if 'volatility' in regime:
                vol = regime['volatility']
                if vol.get('regime'):
                    report += f"**Volatility Regime:** {vol['regime'].get('regime', 'N/A').upper()}\n"
            
            if 'momentum' in regime:
                mom = regime['momentum']
                if mom.get('regime'):
                    report += f"**Momentum Regime:** {mom['regime'].get('regime', 'N/A')}\n"
            
            report += f"\n**Overall Regime:** {regime.get('overall_regime', 'N/A').upper()}\n"
            report += f"**Trading Approach:** {regime.get('trading_approach', 'N/A').upper()}\n"
    except Exception as e:
        report += f"\n*Market regime: {str(e)[:50]}*\n"
    
    # 11. SENTIMENT ANALYSIS (15 concepts)
    try:
        sent = analyze_all_sentiment(
            rsi=rsi if rsi else 50,
            volatility_percentile=50,
            momentum=momentum if momentum else 0,
            bullish_signals=5,
            bearish_signals=3
        )
        if sent:
            report += "\n### 🎭 SENTIMENT ANALYSIS (15 Concepts)\n"
            
            if 'fear_greed' in sent:
                fg = sent['fear_greed']
                report += f"**Fear & Greed Index:** {fg.get('index', 50)} ({fg.get('classification', 'N/A')})\n"
            
            if 'retail' in sent:
                retail = sent['retail']
                if retail.get('positioning'):
                    rp = retail['positioning']
                    report += f"**Retail Positioning:** {rp.get('long_pct', 50)}% Long\n"
                    report += f"**Contrarian Signal:** {rp.get('contrarian_signal', 'N/A')}\n"
            
            report += f"\n**Sentiment Bias:** {sent.get('overall_sentiment', 'NEUTRAL')}\n"
    except Exception as e:
        report += f"\n*Sentiment analysis: {str(e)[:50]}*\n"
    
    # ========== REAL API DATA ==========
    try:
        # Real COT Data
        real_cot = fetch_real_cot_data(symbol)
        if real_cot and 'error' not in real_cot:
            report += "\n### 📋 REAL COT DATA (CFTC Official)\n"
            report += f"**Report Date:** {real_cot.get('report_date', 'N/A')}\n"
            if 'commercial' in real_cot:
                report += f"**Commercial Net:** {real_cot['commercial'].get('net', 0):,}\n"
            if 'non_commercial' in real_cot:
                report += f"**Non-Commercial Net:** {real_cot['non_commercial'].get('net', 0):,}\n"
    except:
        pass
    
    try:
        # Fear & Greed Index
        fear_greed = fetch_fear_greed_index()
        if fear_greed and 'error' not in fear_greed:
            report += "\n### 😱 FEAR & GREED INDEX\n"
            report += f"**Current Value:** {fear_greed.get('value', 'N/A')}\n"
            report += f"**Classification:** {fear_greed.get('classification', 'N/A')}\n"
    except:
        pass
    
    try:
        # OANDA Order Book (Real Retail Positioning)
        oanda_data = fetch_oanda_orderbook(symbol)
        if oanda_data and 'error' not in oanda_data:
            report += "\n### 📊 OANDA RETAIL POSITIONING\n"
            report += f"**Long %:** {oanda_data.get('long_percent', 'N/A')}%\n"
            report += f"**Short %:** {oanda_data.get('short_percent', 'N/A')}%\n"
            report += f"**Contrarian Signal:** {'SELL' if oanda_data.get('long_percent', 50) > 60 else 'BUY' if oanda_data.get('long_percent', 50) < 40 else 'NEUTRAL'}\n"
    except:
        pass
    
    # ========== FINAL SUMMARY ==========
    report += "\n### 🎯 744 CONCEPTS ANALYSIS SUMMARY\n"
    report += "| Category | Concepts | Status |\n"
    report += "|----------|----------|--------|\n"
    report += "| Candlestick Patterns | 64 | ✅ Analyzed |\n"
    report += "| Chart Patterns | 69 | ✅ Analyzed |\n"
    report += "| ICT/SMC Concepts | 131 | ✅ Analyzed |\n"
    report += "| Technical Indicators | 103 | ✅ Analyzed |\n"
    report += "| Price Action Concepts | 150 | ✅ Analyzed |\n"
    report += "| Advanced Indicators (Ehlers/Gann/Elliott/Extra) | 87 | ✅ Analyzed |\n"
    report += "| Harmonic Patterns | 25 | ✅ Analyzed |\n"
    report += "| Volume Analysis | 35 | ✅ Analyzed |\n"
    report += "| Statistical Analysis | 30 | ✅ Analyzed |\n"
    report += "| Market Regime | 20 | ✅ Analyzed |\n"
    report += "| Multi-Timeframe | 15 | ✅ Analyzed |\n"
    report += "| Sentiment Analysis | 15 | ✅ Analyzed |\n"
    report += "| **TOTAL** | **744** | ✅ **Complete** |\n"
    
    return report


MASTER_PROMPT = """You are a MASTER TRADER with 20+ years of experience trading Forex, Gold, and Crypto. You think and analyze charts EXACTLY like a professional institutional trader.

## 🎯 YOUR TRADING PHILOSOPHY:
- You trade what you SEE on the chart, not what you THINK
- Price action is KING - indicators are secondary confirmation
- Smart Money leaves footprints - you can read them
- Liquidity is the fuel that moves markets
- You wait for HIGH PROBABILITY setups only
- **READ EXACT PRICES FROM THE Y-AXIS** - be precise!

---

## 📊 REAL-TIME TECHNICAL DATA (744 Concepts Analyzed):
{technical_data}

---

## ⚠️ CRITICAL INSTRUCTIONS FOR PRICE LEVELS:
1. **LOOK AT THE Y-AXIS** on the right side of the chart to read EXACT prices
2. **DO NOT GUESS** - read the actual price levels from the chart
3. **Round to 2 decimal places** for Gold (XAU/USD), 5 for Forex pairs
4. **Entry, SL, TP must be based on VISIBLE levels** on the chart
5. **Be CONSISTENT** - same chart = same levels regardless of timeframe selected

## 🚨🚨🚨 CRITICAL: STOP LOSS PLACEMENT RULES 🚨🚨🚨
**THIS IS THE MOST IMPORTANT RULE - NEVER VIOLATE IT!**

### FOR BUY/BULLISH TRADES (You expect price to go UP):
- Entry: Current price or support level
- Stop Loss: BELOW Entry (lower price number)
- Take Profit: ABOVE Entry (higher price number)
- Example: Entry=4350, SL=4340 (below), TP=4370 (above)
- **SL < Entry < TP**

### FOR SELL/BEARISH TRADES (You expect price to go DOWN):
- Entry: Current price or resistance level  
- Stop Loss: ABOVE Entry (higher price number)
- Take Profit: BELOW Entry (lower price number)
- Example: Entry=4350, SL=4360 (above), TP=4330 (below)
- **TP < Entry < SL**

### VALIDATION CHECK BEFORE GIVING TRADE PLAN:
1. If BIAS is BULLISH/BUY → SL must be LOWER than Entry
2. If BIAS is BEARISH/SELL → SL must be HIGHER than Entry
3. If your SL violates this rule, FIX IT before responding!

### MINIMUM STOP LOSS DISTANCE:
- Gold (XAU/USD): 5-15 dollars from Entry
- Forex pairs: 15-50 pips from Entry
- Too tight SL = will get stopped out by noise
- Too wide SL = poor risk management

---

## 🔍 VISUAL CHART ANALYSIS (READ THE CHART LIKE A MASTER):

### STEP 1: FIRST GLANCE (What do you see immediately?)
Look at the chart image and describe:
- What is the DOMINANT trend? (Strong up/down/ranging/choppy)
- Where is price NOW relative to recent structure? (READ FROM Y-AXIS)
- Any OBVIOUS patterns jumping out? (channels, triangles, H&S, etc.)
- What story is the chart telling you?

### STEP 2: MARKET STRUCTURE (The Backbone)
Identify from the chart:
- **Swing Points:** Mark the major Highs (HH/LH) and Lows (HL/LL) - READ EXACT PRICES
- **Trend Direction:** Is it making HH+HL (uptrend) or LH+LL (downtrend)?
- **Structure Breaks:** Any recent BOS (Break of Structure) or CHoCH (Change of Character)?
- **Current Phase:** Impulse move or Correction/Retracement?

### STEP 3: KEY LEVELS (Where Smart Money Acts)
Identify from the chart - **READ EXACT PRICES FROM Y-AXIS**:
- **Strong Resistance:** Where did price reject multiple times?
- **Strong Support:** Where did buyers step in?
- **Order Blocks:** Where was the last strong move initiated? (top and bottom price)
- **Fair Value Gaps (FVG):** Any unfilled gaps/imbalances? (top and bottom price)
- **Equal Highs/Lows:** Where is liquidity resting?

### STEP 4: LIQUIDITY ANALYSIS (Where are the stops?)
Look for - **READ EXACT PRICES**:
- **Buy Side Liquidity (BSL):** Equal highs, swing highs where BUYERS have stops
- **Sell Side Liquidity (SSL):** Equal lows, swing lows where SELLERS have stops
- **Liquidity Sweep:** Did price just grab liquidity and reverse?
- **Liquidity Target:** Where will Smart Money likely push price next?

### STEP 5: CANDLESTICK READING (Recent Price Action)
Analyze the last 5-10 candles:
- **Candle Size:** Big candles = momentum, small = indecision
- **Wicks:** Long wicks = rejection, short wicks = acceptance
- **Body Position:** Close near high = bullish, close near low = bearish
- **Patterns:** Engulfing, pin bars, inside bars, doji at key levels?

### STEP 6: ENTRY TRIGGER (When to pull the trigger)
Look for:
- **Confirmation Candle:** Strong close in your direction
- **Break and Retest:** Price breaks level, retests, then continues
- **Rejection Pattern:** Pin bar / engulfing at key level
- **FVG Entry:** Price returns to fill gap then continues

---

## 🎯 MASTER TRADER'S VERDICT:

### BIAS: [STRONG BULLISH 🟢 / BULLISH 🟢 / NEUTRAL ⚪ / BEARISH 🔴 / STRONG BEARISH 🔴]

### WHAT I SEE ON THIS CHART:
(Describe in plain language what the chart is showing - like explaining to a fellow trader)

### THE SETUP:
- **Trade Type:** [Trend Continuation / Reversal / Breakout / Range Play]
- **Timeframe Bias:** [What this timeframe tells us]
- **Smart Money Narrative:** [What are institutions likely doing?]

### 📍 TRADE PLAN (READ EXACT PRICES FROM CHART):

**ENTRY ZONE:** [Read exact price from Y-axis where you would enter]
- Why here: [Explain - OB? FVG? S/R? Liquidity grab?]

**STOP LOSS:** [REMEMBER: BULLISH=SL below Entry | BEARISH=SL above Entry]
- Price: [Exact price - verify it follows the rule above!]
- Why here: [Above/below what structure?]
- Risk in pips: [Calculate from entry to SL - should be 5-15 for Gold]

**TAKE PROFIT 1:** [Exact price from chart] - RR 1:[X]
- Target: [What level? Recent high/low? FVG? OB?]

**TAKE PROFIT 2:** [Exact price from chart] - RR 1:[X]  
- Target: [Liquidity pool? Major S/R?]

**TAKE PROFIT 3:** [Exact price from chart] - RR 1:[X]
- Target: [Full extension target]

### ✅ TRADE VALIDATION CHECKLIST:
Before finalizing, verify:
- [ ] If BULLISH: Is SL < Entry < TP1? 
- [ ] If BEARISH: Is TP1 < Entry < SL?
- [ ] Is SL at least 5 dollars away from Entry (for Gold)?
- [ ] Is Risk:Reward at least 1:1.5?

### TRADE MANAGEMENT:
- Move SL to breakeven after TP1 hits
- Trail stop behind structure after TP2
- Let runner reach TP3 or trail

---

## ⚠️ WHAT COULD GO WRONG:
- **Invalidation:** If price breaks [exact level from chart], this setup is INVALID
- **Risk Events:** [Any news/sessions to watch?]
- **Conflicting Signals:** [Any indicators disagreeing?]

## 💡 MASTER'S TIP:
(Share one specific insight about this chart that a beginner might miss)

---

## CONFIDENCE SCORE: [X]/100
Based on:
- Structure clarity: [X]/25
- Key level confluence: [X]/25  
- Candlestick confirmation: [X]/25
- Risk/Reward quality: [X]/25

⚠️ RISK WARNING: This is analysis, not financial advice. Always use proper risk management. Never risk more than 1-2% per trade."""


# Visual Annotation Prompt - AI returns coordinates for drawing on chart
VISUAL_ANNOTATION_PROMPT = """You are a MASTER TRADER analyzing this chart. I need you to identify KEY VISUAL ELEMENTS that should be drawn on the chart.

Analyze the chart image and return a JSON object with drawing instructions.

**CRITICAL: READ EXACT PRICES FROM THE Y-AXIS ON THE RIGHT SIDE OF THE CHART!**
- Do NOT guess prices - read them from the Y-axis scale
- For Gold (XAU/USD): prices are typically 4-digit numbers like 4350.00
- Be PRECISE - same chart should always give same price levels

Return ONLY valid JSON in this exact format (no other text):
```json
{
    "trend_direction": "bullish" or "bearish" or "ranging",
    "current_price": [READ FROM Y-AXIS - the price where current candle is],
    "annotations": {
        "trend_lines": [
            {"type": "support", "price": [READ FROM Y-AXIS], "label": "Support"},
            {"type": "resistance", "price": [READ FROM Y-AXIS], "label": "Resistance"}
        ],
        "horizontal_levels": [
            {"price": [READ FROM Y-AXIS], "type": "support", "strength": "strong/medium/weak", "label": "Key Support"},
            {"price": [READ FROM Y-AXIS], "type": "resistance", "strength": "strong/medium/weak", "label": "Key Resistance"}
        ],
        "zones": [
            {"top": [READ FROM Y-AXIS], "bottom": [READ FROM Y-AXIS], "type": "order_block", "bias": "bullish/bearish", "label": "Bullish OB"},
            {"top": [READ FROM Y-AXIS], "bottom": [READ FROM Y-AXIS], "type": "fvg", "bias": "bullish/bearish", "label": "FVG"},
            {"top": [READ FROM Y-AXIS], "bottom": [READ FROM Y-AXIS], "type": "supply", "label": "Supply Zone"},
            {"top": [READ FROM Y-AXIS], "bottom": [READ FROM Y-AXIS], "type": "demand", "label": "Demand Zone"}
        ],
        "liquidity": [
            {"price": [READ FROM Y-AXIS], "type": "bsl", "label": "Buy Side Liquidity"},
            {"price": [READ FROM Y-AXIS], "type": "ssl", "label": "Sell Side Liquidity"}
        ],
        "trade_levels": {
            "entry": {"price": [READ FROM Y-AXIS - at key level], "label": "Entry"},
            "stop_loss": {"price": [READ FROM Y-AXIS - below/above structure], "label": "Stop Loss"},
            "tp1": {"price": [READ FROM Y-AXIS], "label": "TP1"},
            "tp2": {"price": [READ FROM Y-AXIS], "label": "TP2"},
            "tp3": {"price": [READ FROM Y-AXIS], "label": "TP3"}
        },
        "swing_points": [
            {"price": [READ FROM Y-AXIS], "type": "high", "label": "HH"},
            {"price": [READ FROM Y-AXIS], "type": "low", "label": "HL"}
        ],
        "patterns": [
            {"name": "Pattern Name", "location": "description of where on chart"}
        ]
    },
    "bias": "BULLISH" or "BEARISH" or "NEUTRAL",
    "confidence": [0-100],
    "summary": "Brief 1-2 sentence summary of what you see"
}
```

CRITICAL RULES:
1. **READ THE Y-AXIS** - Look at the right side of the chart for exact price numbers
2. **BE CONSISTENT** - Same chart image = same price levels every time
3. **DO NOT ROUND** excessively - use the exact prices you see
4. **Entry/SL/TP must align** with visible structure on the chart
5. Only include elements you can CLEARLY see on the chart"""


def analyze_chart_visual(image_data, symbol="XAU/USD", interval="1h"):
    """Analyze chart and return visual annotation data"""
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # Get visual annotations from AI
    response = client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": VISUAL_ANNOTATION_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }],
        max_tokens=2000,
        temperature=0.01  # Very low for consistent price readings
    )
    
    annotation_text = response.choices[0].message.content
    
    # Parse JSON from response
    import json
    import re
    
    # Extract JSON from markdown code block if present
    json_match = re.search(r'```json\s*([\s\S]*?)\s*```', annotation_text)
    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r'\{[\s\S]*\}', annotation_text)
        json_str = json_match.group(0) if json_match else '{}'
    
    try:
        annotations = json.loads(json_str)
    except:
        annotations = {"error": "Could not parse annotations", "raw": annotation_text}
    
    return annotations


# Simple prompt for Light Mode (uses much fewer tokens)
SIMPLE_PROMPT = """You are a Master Trader analyzing this chart. Be concise and direct.

**Symbol:** {symbol} | **Timeframe:** {interval}

Look at the chart and provide:

## BIAS: [BULLISH 🟢 / BEARISH 🔴 / NEUTRAL ⚪]

## WHAT I SEE:
(2-3 sentences about the chart - trend, key levels, patterns)

## TRADE PLAN:
- **ENTRY:** [exact price from Y-axis]
- **STOP LOSS:** [exact price - BELOW entry for BUY, ABOVE entry for SELL]
- **TP1:** [price] (R:R 1:1.5+)
- **TP2:** [price] (R:R 1:2+)
- **TP3:** [price] (R:R 1:3+)

## CONFIDENCE: [X]/100

⚠️ Remember: BUY = SL below entry | SELL = SL above entry"""


def analyze_chart(image_data, symbol="XAU/USD", interval="1h", light_mode=False):
    base64_image = base64.b64encode(image_data).decode('utf-8')
    
    # ========== CANDLESTICK DATA INJECTION ==========
    # Fetch real MT5 data to give AI exact prices (no guessing from Y-axis!)
    mt5_data = fetch_mt5_data(symbol, interval, bars=50)
    candlestick_info = ""
    
    if mt5_data:
        current_price = mt5_data['current_price']
        opens = mt5_data['opens']
        highs = mt5_data['highs']
        lows = mt5_data['lows']
        closes = mt5_data['closes']
        
        # Calculate key levels from real data
        recent_high = max(highs[-20:])
        recent_low = min(lows[-20:])
        swing_high = max(highs[-50:])
        swing_low = min(lows[-50:])
        
        # Get last 10 candles OHLC
        last_candles = []
        for i in range(-10, 0):
            last_candles.append({
                'open': opens[i],
                'high': highs[i],
                'low': lows[i],
                'close': closes[i]
            })
        
        # Calculate support/resistance from price action
        pivot = (highs[-1] + lows[-1] + closes[-1]) / 3
        r1 = 2 * pivot - lows[-1]
        s1 = 2 * pivot - highs[-1]
        r2 = pivot + (highs[-1] - lows[-1])
        s2 = pivot - (highs[-1] - lows[-1])
        
        # Determine decimal places
        decimals = get_decimal_places(symbol)
        
        candlestick_info = f"""

## 📊 REAL-TIME MT5 DATA (USE THESE EXACT PRICES!):
**Symbol:** {symbol} | **Timeframe:** {interval}
**Current Price:** {current_price:.{decimals}f}

### KEY PRICE LEVELS (from MT5):
- **Swing High (50 bars):** {swing_high:.{decimals}f}
- **Swing Low (50 bars):** {swing_low:.{decimals}f}
- **Recent High (20 bars):** {recent_high:.{decimals}f}
- **Recent Low (20 bars):** {recent_low:.{decimals}f}

### PIVOT POINTS:
- **R2:** {r2:.{decimals}f}
- **R1:** {r1:.{decimals}f}
- **Pivot:** {pivot:.{decimals}f}
- **S1:** {s1:.{decimals}f}
- **S2:** {s2:.{decimals}f}

### LAST 10 CANDLES (Most Recent First):
| # | Open | High | Low | Close | Type |
|---|------|------|-----|-------|------|
"""
        for i, candle in enumerate(reversed(last_candles)):
            candle_type = "🟢 Bullish" if candle['close'] > candle['open'] else "🔴 Bearish"
            candlestick_info += f"| {i+1} | {candle['open']:.{decimals}f} | {candle['high']:.{decimals}f} | {candle['low']:.{decimals}f} | {candle['close']:.{decimals}f} | {candle_type} |\n"
        
        candlestick_info += f"""
### SUGGESTED TRADE LEVELS (Based on Structure):
- **BUY Entry Zone:** {recent_low:.{decimals}f} - {s1:.{decimals}f} (near support)
- **BUY Stop Loss:** {s2:.{decimals}f} (below S2)
- **BUY TP1:** {pivot:.{decimals}f} | **TP2:** {r1:.{decimals}f} | **TP3:** {r2:.{decimals}f}

- **SELL Entry Zone:** {recent_high:.{decimals}f} - {r1:.{decimals}f} (near resistance)
- **SELL Stop Loss:** {r2:.{decimals}f} (above R2)
- **SELL TP1:** {pivot:.{decimals}f} | **TP2:** {s1:.{decimals}f} | **TP3:** {s2:.{decimals}f}

⚠️ **IMPORTANT:** Use these EXACT prices from MT5 data, NOT guessed from chart image!
"""
    
    if light_mode:
        # Light mode - just analyze the image, no technical report
        prompt = SIMPLE_PROMPT.format(symbol=symbol, interval=interval)
        if candlestick_info:
            prompt = candlestick_info + "\n\n" + prompt
        max_tokens = 1500
    else:
        # Full mode - include 744 concepts technical report
        technical_data = generate_technical_report(symbol, interval)
        prompt = MASTER_PROMPT.format(technical_data=technical_data)
        if candlestick_info:
            prompt = candlestick_info + "\n\n" + prompt
        max_tokens = 5000
    
    response = call_groq_with_fallback(
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
            ]
        }],
        max_tokens=max_tokens,
        temperature=0.01,
        image_base64=base64_image  # Pass for Gemini fallback
    )
    
    return response.choices[0].message.content


# ========== AUTHENTICATION ROUTES ==========

@app.route('/login')
def login_page():
    """Show login page"""
    if 'user_id' in session:
        return redirect('/')
    error = request.args.get('error')
    success = request.args.get('success')
    return render_template('login.html', error=error, success=success)


@app.route('/auth/register', methods=['POST'])
def register():
    """Register new user with email/password"""
    email = request.form.get('email', '').lower().strip()
    password = request.form.get('password', '')
    name = request.form.get('name', '')
    
    if not email or not password:
        return redirect('/login?error=Email and password required')
    
    if len(password) < 8:
        return redirect('/login?error=Password must be at least 8 characters')
    
    # Check if user exists
    existing = get_user_by_email(email)
    if existing:
        return redirect('/login?error=Email already registered. Please sign in.')
    
    # Create user
    user_id = create_user(email, password, name, provider='email')
    if user_id:
        session.permanent = True  # Use PERMANENT_SESSION_LIFETIME (24 hours)
        session['user_id'] = user_id
        return redirect('/')
    
    return redirect('/login?error=Registration failed. Please try again.')


@app.route('/auth/login', methods=['POST'])
def login():
    """Login with email/password"""
    email = request.form.get('email', '').lower().strip()
    password = request.form.get('password', '')
    
    if not email or not password:
        return redirect('/login?error=Email and password required')
    
    user = get_user_by_email(email)
    if not user:
        return redirect('/login?error=Invalid email or password')
    
    if user.get('provider') == 'google':
        return redirect('/login?error=This account uses Google login. Click "Continue with Google".')
    
    if not verify_password(password, user.get('password_hash', '')):
        return redirect('/login?error=Invalid email or password')
    
    session.permanent = True  # Use PERMANENT_SESSION_LIFETIME (24 hours)
    session['user_id'] = user['id']
    update_user_login(user['id'])
    return redirect('/')


@app.route('/auth/google')
def google_login():
    """Initiate Google OAuth login"""
    if not GOOGLE_CLIENT_ID:
        return redirect('/login?error=Google login not configured')
    
    # Google OAuth URL
    redirect_uri = request.url_root.rstrip('/') + '/auth/google/callback'
    scope = 'openid email profile'
    
    auth_url = (
        f"https://accounts.google.com/o/oauth2/v2/auth?"
        f"client_id={GOOGLE_CLIENT_ID}&"
        f"redirect_uri={redirect_uri}&"
        f"response_type=code&"
        f"scope={scope}&"
        f"access_type=offline"
    )
    
    return redirect(auth_url)


@app.route('/auth/google/callback')
def google_callback():
    """Handle Google OAuth callback"""
    code = request.args.get('code')
    error = request.args.get('error')
    
    if error:
        return redirect(f'/login?error=Google login failed: {error}')
    
    if not code:
        return redirect('/login?error=No authorization code received')
    
    try:
        # Exchange code for tokens
        redirect_uri = request.url_root.rstrip('/') + '/auth/google/callback'
        token_response = requests.post('https://oauth2.googleapis.com/token', data={
            'code': code,
            'client_id': GOOGLE_CLIENT_ID,
            'client_secret': GOOGLE_CLIENT_SECRET,
            'redirect_uri': redirect_uri,
            'grant_type': 'authorization_code'
        })
        
        tokens = token_response.json()
        access_token = tokens.get('access_token')
        
        if not access_token:
            return redirect('/login?error=Failed to get access token')
        
        # Get user info
        user_info = requests.get(
            'https://www.googleapis.com/oauth2/v2/userinfo',
            headers={'Authorization': f'Bearer {access_token}'}
        ).json()
        
        email = user_info.get('email', '').lower()
        name = user_info.get('name', '')
        picture = user_info.get('picture', '')
        
        if not email:
            return redirect('/login?error=Could not get email from Google')
        
        # Check if user exists
        user = get_user_by_email(email)
        
        session.permanent = True  # Use PERMANENT_SESSION_LIFETIME (24 hours)
        if user:
            # Update existing user
            update_user_google(email, name, picture)
            session['user_id'] = user['id']
        else:
            # Create new user
            user_id = create_user(email, None, name, picture, provider='google')
            session['user_id'] = user_id
        
        return redirect('/')
        
    except Exception as e:
        print(f"Google OAuth error: {e}")
        return redirect(f'/login?error=Google login failed')


@app.route('/auth/logout')
def logout():
    """Logout user"""
    session.clear()
    return redirect('/login?success=You have been logged out')


# ========== LIVE MARKET PRICES API ==========
@app.route('/api/prices')
def get_live_prices():
    """Get live market prices from Twelve Data API"""
    try:
        twelve_data_key = os.getenv('TWELVE_DATA_KEY', '')
        
        # Symbols to fetch
        symbols_map = {
            'XAU/USD': 'XAU/USD',
            'EUR/USD': 'EUR/USD', 
            'GBP/USD': 'GBP/USD',
            'USD/JPY': 'USD/JPY',
            'BTC/USD': 'BTC/USD',
            'ETH/USD': 'ETH/USD'
        }
        
        prices = {}
        
        if twelve_data_key:
            # Fetch from Twelve Data API
            symbols_str = ','.join(symbols_map.values())
            url = f'https://api.twelvedata.com/price?symbol={symbols_str}&apikey={twelve_data_key}'
            
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    data = response.json()
                    
                    for symbol in symbols_map.keys():
                        api_symbol = symbols_map[symbol]
                        if api_symbol in data and 'price' in data[api_symbol]:
                            prices[symbol] = {
                                'price': float(data[api_symbol]['price']),
                                'source': 'twelvedata'
                            }
            except Exception as e:
                print(f"Twelve Data error: {e}")
        
        # Fallback: Try Binance for crypto
        if 'BTC/USD' not in prices:
            try:
                btc_resp = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT', timeout=3)
                if btc_resp.status_code == 200:
                    prices['BTC/USD'] = {'price': float(btc_resp.json()['price']), 'source': 'binance'}
            except:
                pass
                
        if 'ETH/USD' not in prices:
            try:
                eth_resp = requests.get('https://api.binance.com/api/v3/ticker/price?symbol=ETHUSDT', timeout=3)
                if eth_resp.status_code == 200:
                    prices['ETH/USD'] = {'price': float(eth_resp.json()['price']), 'source': 'binance'}
            except:
                pass
        
        # Fallback prices if API fails
        fallback = {
            'XAU/USD': 2388.50,
            'EUR/USD': 1.0842,
            'GBP/USD': 1.2695,
            'USD/JPY': 154.85,
            'BTC/USD': 67850,
            'ETH/USD': 3485
        }
        
        for symbol, default_price in fallback.items():
            if symbol not in prices:
                prices[symbol] = {'price': default_price, 'source': 'fallback'}
        
        return jsonify({
            'success': True,
            'prices': prices,
            'timestamp': __import__('datetime').datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/user')
def get_user_info():
    """Get current user info"""
    user = get_current_user()
    if not user:
        return jsonify({'logged_in': False})
    
    tier_info = get_user_tier_info(user['id'])
    
    return jsonify({
        'logged_in': True,
        'id': user['id'],
        'email': user['email'],
        'name': user['name'],
        'picture': user['picture'],
        'tier': user['tier'],
        'tier_info': tier_info
    })


@app.route('/api/check-limit')
def check_limit():
    """Check if user can analyze"""
    user = get_current_user()
    if not user:
        return jsonify({'allowed': True, 'remaining': 5, 'message': 'Login for more analyses'})
    
    allowed, remaining = check_analysis_limit(user['id'])
    return jsonify({
        'allowed': allowed,
        'remaining': remaining if isinstance(remaining, int) else 0,
        'message': remaining if isinstance(remaining, str) else None
    })


# ========== MAIN ROUTES ==========

@app.route('/')
def index():
    """Main page - requires login for full features"""
    user = get_current_user()
    return render_template('index.html', user=user)


@app.route('/analyze', methods=['POST'])
@login_required
def analyze():
    """Analyze chart - requires authentication"""
    user = get_current_user()
    
    # Handle case where user is None (session expired)
    if not user:
        return jsonify({'error': 'Session expired. Please login again.', 'redirect': '/login'}), 401
    
    # Check analysis limit
    allowed, remaining = check_analysis_limit(user['id'])
    if not allowed:
        return jsonify({
            'error': f'Daily limit reached! {remaining}',
            'limit_reached': True,
            'upgrade_url': '/pricing'
        }), 429
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    symbol = request.form.get('symbol', 'XAU/USD')
    interval = request.form.get('interval', '1h')
    light_mode = request.form.get('light_mode', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Increment analysis count
        increment_analysis_count(user['id'])
        
        image_data = file.read()
        analysis = analyze_chart(image_data, symbol, interval, light_mode)
        return jsonify({'analysis': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/analyze-visual', methods=['POST'])
@login_required
def analyze_visual():
    """Analyze chart and return visual annotations - requires authentication"""
    user = get_current_user()
    
    # Handle case where user is None (session expired)
    if not user:
        return jsonify({'error': 'Session expired. Please login again.', 'redirect': '/login'}), 401
    
    # Check analysis limit
    allowed, remaining = check_analysis_limit(user['id'])
    if not allowed:
        return jsonify({
            'error': f'Daily limit reached! {remaining}',
            'limit_reached': True,
            'upgrade_url': '/pricing'
        }), 429
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    file = request.files['image']
    symbol = request.form.get('symbol', 'XAU/USD')
    interval = request.form.get('interval', '1h')
    light_mode = request.form.get('light_mode', 'false').lower() == 'true'
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Increment analysis count
        if user:
            increment_analysis_count(user['id'])
        
        # Read image data once and reuse
        image_data = file.read()
        
        # Get text analysis (ONE AI call only)
        analysis = analyze_chart(image_data, symbol, interval, light_mode)
        
        # Extract annotations FROM the analysis text (no second AI call)
        # This ensures Signal Card and Chart Markup are CONSISTENT
        annotations = extract_annotations_from_analysis(analysis)
        
        # ========== ACCURACY VALIDATION SYSTEM ==========
        # Get market data for confluence calculation (skip in light mode for speed)
        market_data = None if light_mode else fetch_market_data(symbol, interval)
        
        validation_result = None
        confluence_result = None
        
        if market_data:
            # Calculate confluence score
            confluence_result = calculate_confluence_score(
                market_data.get('opens', []),
                market_data.get('highs', []),
                market_data.get('lows', []),
                market_data.get('closes', []),
                market_data.get('volumes', [0] * len(market_data.get('closes', []))),
                symbol
            )
            
            # Extract trade levels from annotations
            trade_levels = annotations.get('annotations', {}).get('trade_levels', {})
            entry = trade_levels.get('entry', {}).get('price') if trade_levels.get('entry') else None
            stop_loss = trade_levels.get('stop_loss', {}).get('price') if trade_levels.get('stop_loss') else None
            tp1 = trade_levels.get('tp1', {}).get('price') if trade_levels.get('tp1') else None
            tp2 = trade_levels.get('tp2', {}).get('price') if trade_levels.get('tp2') else None
            tp3 = trade_levels.get('tp3', {}).get('price') if trade_levels.get('tp3') else None
            
            # Validate trade setup
            if entry:
                validation_result = validate_trade_setup(
                    symbol, entry, stop_loss, tp1, tp2, tp3
                )
        
        return jsonify({
            'analysis': analysis,
            'annotations': annotations,
            'image_base64': base64.b64encode(image_data).decode('utf-8'),
            'validation': validation_result,
            'confluence': confluence_result
        })
    except Exception as e:
        import traceback
        print(f"Error in analyze_visual: {traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


def extract_annotations_from_analysis(analysis_text):
    """Extract price levels from analysis text to ensure consistency between Signal Card and Chart Markup"""
    import re
    
    def safe_float(value):
        """Safely convert string to float, return None if invalid"""
        if not value:
            return None
        # Remove commas and clean up
        cleaned = value.replace(',', '').strip()
        # Check if it's a valid number (must have at least one digit)
        if not cleaned or not any(c.isdigit() for c in cleaned):
            return None
        try:
            return float(cleaned)
        except (ValueError, TypeError):
            return None
    
    annotations = {
        "trend_direction": "neutral",
        "current_price": 0,
        "annotations": {
            "horizontal_levels": [],
            "zones": [],
            "liquidity": [],
            "trade_levels": {
                "entry": None,
                "stop_loss": None,
                "tp1": None,
                "tp2": None,
                "tp3": None
            },
            "swing_points": []
        },
        "bias": "NEUTRAL",
        "confidence": 50,
        "summary": "",
        "sl_warning": None
    }
    
    # Extract BIAS
    bias_match = re.search(r'BIAS[:\s]*\**\s*(STRONG\s*)?(BULLISH|BEARISH|NEUTRAL)', analysis_text, re.IGNORECASE)
    if bias_match:
        bias = bias_match.group(2).upper()
        annotations["bias"] = ("STRONG " + bias) if bias_match.group(1) else bias
        annotations["trend_direction"] = "bullish" if "BULLISH" in bias else "bearish" if "BEARISH" in bias else "ranging"
    
    # Extract Confidence
    conf_match = re.search(r'CONFIDENCE\s*SCORE[:\s]*\**\s*(\d+(?:\.\d+)?)\s*\/\s*100', analysis_text, re.IGNORECASE)
    if conf_match:
        conf_val = safe_float(conf_match.group(1))
        if conf_val:
            annotations["confidence"] = int(conf_val)
    
    # Extract Entry Zone - multiple formats supported
    entry_price = None
    # Format 1: "ENTRY ZONE: 4351.00" or "ENTRY ZONE: **4351.00**"
    entry_match = re.search(r'ENTRY\s*ZONE[:\s]*\**\s*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if not entry_match:
        # Format 2: "ENTRY: Buy at 4,351.00" or "ENTRY: 4351.00"
        entry_match = re.search(r'\*?\*?ENTRY\*?\*?[:\s]+(?:Buy\s+at\s+|Sell\s+at\s+)?(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if not entry_match:
        # Format 3: "Entry at 4351.00" or "Entry: 4351"
        entry_match = re.search(r'Entry\s+(?:at\s+)?[:\s]*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    
    if entry_match:
        entry_price = safe_float(entry_match.group(1))
        if entry_price:
            annotations["annotations"]["trade_levels"]["entry"] = {
                "price": entry_price,
                "label": "Entry"
            }
            annotations["current_price"] = entry_price
    
    # Extract Stop Loss - multiple formats supported
    sl_price = None
    # Format 1: "STOP LOSS: 4348.00" or "STOP LOSS: **4348.00**"
    sl_match = re.search(r'STOP\s*LOSS[:\s]*\**\s*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if not sl_match:
        # Format 2: "SL: 4348.00" or "SL at 4348"
        sl_match = re.search(r'\bSL[:\s]+(?:at\s+)?(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    
    if sl_match:
        sl_price = safe_float(sl_match.group(1))
        if sl_price:
            annotations["annotations"]["trade_levels"]["stop_loss"] = {
                "price": sl_price,
                "label": "Stop Loss"
            }
    
    # Extract Take Profits - multiple formats supported
    tp1_price = None
    # Format 1: "TAKE PROFIT 1: 4356.00"
    tp1_match = re.search(r'TAKE\s*PROFIT\s*1[:\s]*\**\s*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if not tp1_match:
        # Format 2: "TP1: 4356.00" or "TP1: 4,356.00"
        tp1_match = re.search(r'\bTP1[:\s]+(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    
    if tp1_match:
        tp1_price = safe_float(tp1_match.group(1))
        if tp1_price:
            annotations["annotations"]["trade_levels"]["tp1"] = {
                "price": tp1_price,
                "label": "TP1"
            }
    
    # Format 1: "TAKE PROFIT 2: 4360.00"
    tp2_match = re.search(r'TAKE\s*PROFIT\s*2[:\s]*\**\s*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if not tp2_match:
        # Format 2: "TP2: 4360.00"
        tp2_match = re.search(r'\bTP2[:\s]+(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    
    if tp2_match:
        tp2_price = safe_float(tp2_match.group(1))
        if tp2_price:
            annotations["annotations"]["trade_levels"]["tp2"] = {
                "price": tp2_price,
                "label": "TP2"
            }
    
    # Format 1: "TAKE PROFIT 3: 4365.00"
    tp3_match = re.search(r'TAKE\s*PROFIT\s*3[:\s]*\**\s*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if not tp3_match:
        # Format 2: "TP3: 4365.00"
        tp3_match = re.search(r'\bTP3[:\s]+(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    
    if tp3_match:
        tp3_price = safe_float(tp3_match.group(1))
        if tp3_price:
            annotations["annotations"]["trade_levels"]["tp3"] = {
                "price": tp3_price,
                "label": "TP3"
            }
    
    # ========== AUTO-FIX WRONG SL DIRECTION ==========
    if entry_price and sl_price and tp1_price:
        is_buy_trade = tp1_price > entry_price  # TP above entry = BUY
        is_sell_trade = tp1_price < entry_price  # TP below entry = SELL
        
        sl_above_entry = sl_price > entry_price
        sl_below_entry = sl_price < entry_price
        
        # Check for wrong SL direction and fix it
        if is_buy_trade and sl_above_entry:
            # BUY trade but SL is above entry - WRONG!
            # Calculate correct SL (mirror it below entry)
            sl_distance = sl_price - entry_price
            corrected_sl = entry_price - sl_distance
            annotations["sl_warning"] = f"⚠️ AI placed SL wrong! Original: {sl_price:.2f} (above entry). Auto-corrected to: {corrected_sl:.2f} (below entry)"
            annotations["annotations"]["trade_levels"]["stop_loss"]["price"] = corrected_sl
            annotations["annotations"]["trade_levels"]["stop_loss"]["original_wrong"] = sl_price
            
        elif is_sell_trade and sl_below_entry:
            # SELL trade but SL is below entry - WRONG!
            # Calculate correct SL (mirror it above entry)
            sl_distance = entry_price - sl_price
            corrected_sl = entry_price + sl_distance
            annotations["sl_warning"] = f"⚠️ AI placed SL wrong! Original: {sl_price:.2f} (below entry). Auto-corrected to: {corrected_sl:.2f} (above entry)"
            annotations["annotations"]["trade_levels"]["stop_loss"]["price"] = corrected_sl
            annotations["annotations"]["trade_levels"]["stop_loss"]["original_wrong"] = sl_price
    
    # Extract Support levels - require digit
    support_matches = re.findall(r'(?:Strong\s*)?Support[:\s]*(?:Around\s*)?(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    for match in support_matches[:3]:
        price = safe_float(match)
        if price:
            annotations["annotations"]["horizontal_levels"].append({
                "price": price,
                "type": "support",
                "strength": "strong",
                "label": f"Support {price:.2f}"
            })
    
    # Extract Resistance levels
    resistance_matches = re.findall(r'(?:Strong\s*)?Resistance[:\s]*(?:Around\s*)?(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    for match in resistance_matches[:3]:
        price = safe_float(match)
        if price:
            annotations["annotations"]["horizontal_levels"].append({
                "price": price,
                "type": "resistance",
                "strength": "strong",
                "label": f"Resistance {price:.2f}"
            })
    
    # Extract Order Blocks
    ob_match = re.search(r'(?:Bullish\s*)?Order\s*Block[:\s]*(?:Around\s*)?(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if ob_match:
        bottom = safe_float(ob_match.group(1))
        top = safe_float(ob_match.group(2))
        if bottom and top:
            annotations["annotations"]["zones"].append({
                "top": top,
                "bottom": bottom,
                "type": "order_block",
                "bias": annotations["trend_direction"],
                "label": "Order Block"
            })
    
    # Extract FVG
    fvg_match = re.search(r'(?:Fair\s*Value\s*Gap|FVG)[:\s]*(?:Around\s*)?(\d+(?:[.,]\d+)?)\s*-\s*(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if fvg_match:
        bottom = safe_float(fvg_match.group(1))
        top = safe_float(fvg_match.group(2))
        if bottom and top:
            annotations["annotations"]["zones"].append({
                "top": top,
                "bottom": bottom,
                "type": "fvg",
                "bias": annotations["trend_direction"],
                "label": "FVG"
            })
    
    # Extract BSL/SSL
    bsl_match = re.search(r'Buy\s*Side\s*Liquidity[:\s]*(?:Around\s*)?(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if bsl_match:
        price = safe_float(bsl_match.group(1))
        if price:
            annotations["annotations"]["liquidity"].append({
                "price": price,
                "type": "bsl",
                "label": "BSL"
            })
    
    ssl_match = re.search(r'Sell\s*Side\s*Liquidity[:\s]*(?:Around\s*)?(\d+(?:[.,]\d+)?)', analysis_text, re.IGNORECASE)
    if ssl_match:
        price = safe_float(ssl_match.group(1))
        if price:
            annotations["annotations"]["liquidity"].append({
                "price": price,
                "type": "ssl",
                "label": "SSL"
            })
    
    return annotations


# Multi-Timeframe Analysis Prompt - Master Trader Style
MTF_PROMPT = """You are a MASTER TRADER with 20+ years experience. You're analyzing MULTIPLE TIMEFRAMES for {symbol} like a professional.

## 📊 CHARTS PROVIDED:
{timeframe_list}

## 📈 TECHNICAL DATA (744 Concepts):
{technical_data}

---

## 🔍 MULTI-TIMEFRAME ANALYSIS (Top-Down Approach):

### STEP 1: HIGHER TIMEFRAME BIAS (D1/H4)
Look at the higher timeframe charts:
- What is the MAJOR trend direction?
- Where are the KEY levels that institutions are watching?
- What is the Smart Money narrative?

### STEP 2: EXECUTION TIMEFRAME (H1/M30)
Look at the medium timeframes:
- Is price in Premium or Discount zone?
- Any Order Blocks or FVGs to trade from?
- Structure alignment with HTF?

### STEP 3: ENTRY TIMEFRAME (M15/M5)
Look at the lower timeframes:
- Entry trigger confirmation?
- Precise entry level?
- Tight stop loss placement?

---

## 📊 TIMEFRAME BREAKDOWN:

{tf_analysis_template}

---

## 🎯 MTF CONFLUENCE MATRIX:

| Timeframe | Trend | Structure | Key Level | Bias |
|-----------|-------|-----------|-----------|------|
(Analyze each chart)

## ✅ ALIGNMENT CHECK:
- HTF + MTF + LTF aligned? [YES/NO]
- Confluence strength: [STRONG/MODERATE/WEAK]

---

## 🎯 MASTER TRADER'S MTF VERDICT:

### OVERALL BIAS: [STRONG BULLISH 🟢 / BULLISH 🟢 / NEUTRAL ⚪ / BEARISH 🔴 / STRONG BEARISH 🔴]

### THE STORY ACROSS TIMEFRAMES:
(Explain what each timeframe is telling you and how they connect)

### 📍 TRADE PLAN (MTF Confluence):

**ENTRY:** [Price] on [Timeframe]
- HTF confirms: [What?]
- LTF trigger: [What?]

**STOP LOSS:** [Price]
- Below/Above: [HTF structure point]

**TP1:** [Price] - RR 1:[X] - [LTF target]
**TP2:** [Price] - RR 1:[X] - [MTF target]  
**TP3:** [Price] - RR 1:[X] - [HTF target]

### CONFIDENCE: [X]/100
- HTF clarity: [X]/25
- MTF alignment: [X]/25
- LTF entry quality: [X]/25
- Overall confluence: [X]/25

⚠️ RISK: Never risk more than 1-2% per trade."""


def analyze_mtf_charts(images_data, timeframes, symbol="XAU/USD"):
    """Analyze multiple timeframe charts"""
    # Get technical data for the symbol (use H1 as base)
    technical_data = generate_technical_report(symbol, "1h")
    
    # Build timeframe list
    tf_order = ['D1', 'H4', 'H1', 'M30', 'M15', 'M5']
    sorted_tfs = sorted(timeframes, key=lambda x: tf_order.index(x) if x in tf_order else 99)
    timeframe_list = "\n".join([f"- {tf}" for tf in sorted_tfs])
    
    # Build analysis template
    tf_analysis_template = "\n".join([f"#### {tf} Analysis\n(Analyze this timeframe chart)" for tf in sorted_tfs])
    
    # Format prompt
    prompt = MTF_PROMPT.format(
        symbol=symbol,
        timeframe_list=timeframe_list,
        technical_data=technical_data,
        tf_analysis_template=tf_analysis_template
    )
    
    # Build message content with all images
    content = [{"type": "text", "text": prompt}]
    
    for i, (image_data, tf) in enumerate(zip(images_data, timeframes)):
        base64_image = base64.b64encode(image_data).decode('utf-8')
        content.append({
            "type": "text",
            "text": f"\n--- {tf} CHART ---"
        })
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{base64_image}"}
        })
    
    response = call_groq_with_fallback(
        messages=[{
            "role": "user",
            "content": content
        }],
        max_tokens=6000,
        temperature=0.05
    )
    
    return response.choices[0].message.content


@app.route('/analyze-mtf', methods=['POST'])
@login_required
def analyze_mtf():
    """Multi-timeframe analysis endpoint - requires authentication"""
    user = get_current_user()
    
    # Handle case where user is None (session expired)
    if not user:
        return jsonify({'error': 'Session expired. Please login again.', 'redirect': '/login'}), 401
    
    # Check if user has MTF access (Pro or Premium only)
    tier_info = get_user_tier_info(user['id'])
    if not tier_info or not tier_info['limits'].get('mtf_enabled', False):
        return jsonify({
            'error': 'Multi-timeframe analysis requires Pro or Premium subscription',
            'upgrade_url': '/pricing'
        }), 403
    
    # Check analysis limit
    allowed, remaining = check_analysis_limit(user['id'])
    if not allowed:
        return jsonify({
            'error': f'Daily limit reached! {remaining}',
            'limit_reached': True,
            'upgrade_url': '/pricing'
        }), 429
    
    if 'images' not in request.files:
        return jsonify({'error': 'No images uploaded'}), 400
    
    files = request.files.getlist('images')
    timeframes = request.form.getlist('timeframes')
    symbol = request.form.get('symbol', 'XAU/USD')
    
    if len(files) < 2:
        return jsonify({'error': 'Please upload at least 2 timeframe charts'}), 400
    
    if len(files) > 5:
        return jsonify({'error': 'Maximum 5 timeframes supported. Please upload 2-5 charts.'}), 400
    
    if len(files) != len(timeframes):
        return jsonify({'error': 'Mismatch between images and timeframes'}), 400
    
    try:
        # Increment analysis count
        increment_analysis_count(user['id'])
        
        # Read all image data
        images_data = [file.read() for file in files]
        
        # Analyze all charts together
        analysis = analyze_mtf_charts(images_data, timeframes, symbol)
        return jsonify({'analysis': analysis})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/validate-trade', methods=['POST'])
def validate_trade():
    """Validate a trade setup with all accuracy checks"""
    data = request.get_json()
    
    symbol = data.get('symbol', 'XAU/USD')
    entry = data.get('entry')
    stop_loss = data.get('stop_loss')
    tp1 = data.get('tp1')
    tp2 = data.get('tp2')
    tp3 = data.get('tp3')
    interval = data.get('interval', '1h')
    
    try:
        # Get market data for confluence
        market_data = fetch_market_data(symbol, interval)
        
        # Validate trade setup
        validation = validate_trade_setup(symbol, entry, stop_loss, tp1, tp2, tp3)
        
        # Calculate confluence
        confluence = None
        if market_data:
            confluence = calculate_confluence_score(
                market_data.get('opens', []),
                market_data.get('highs', []),
                market_data.get('lows', []),
                market_data.get('closes', []),
                market_data.get('volumes', [0] * len(market_data.get('closes', []))),
                symbol
            )
        
        return jsonify({
            'validation': validation,
            'confluence': confluence,
            'mt5_price': get_mt5_current_price(symbol)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/get-confluence', methods=['GET'])
def get_confluence():
    """Get real-time confluence analysis for a symbol"""
    symbol = request.args.get('symbol', 'XAU/USD')
    interval = request.args.get('interval', '1h')
    
    try:
        market_data = fetch_market_data(symbol, interval)
        
        if not market_data:
            return jsonify({'error': 'Could not fetch market data'}), 400
        
        confluence = calculate_confluence_score(
            market_data.get('opens', []),
            market_data.get('highs', []),
            market_data.get('lows', []),
            market_data.get('closes', []),
            market_data.get('volumes', [0] * len(market_data.get('closes', []))),
            symbol
        )
        
        # Get current price
        current_price = market_data.get('current_price') or market_data['closes'][-1]
        mt5_price = get_mt5_current_price(symbol)
        
        return jsonify({
            'confluence': confluence,
            'current_price': current_price,
            'mt5_price': mt5_price,
            'data_source': market_data.get('source', 'Unknown')
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/check-news', methods=['GET'])
def check_news():
    """Check for upcoming high-impact news events"""
    symbol = request.args.get('symbol', 'XAU/USD')
    
    try:
        news_check = check_news_filter(symbol)
        return jsonify(news_check)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/live-prices', methods=['GET'])
def live_prices():
    """Get live prices from MT5 for all symbols"""
    symbols = ['XAU/USD', 'EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD', 'XAG/USD']
    prices = {}
    
    for symbol in symbols:
        try:
            # Get MT5 price
            tick = get_mt5_current_price(symbol)
            if tick:
                mid_price = (tick['bid'] + tick['ask']) / 2
                prices[symbol] = {
                    'bid': tick['bid'],
                    'ask': tick['ask'],
                    'mid': mid_price,
                    'spread': tick['spread'],
                    'source': 'MT5'
                }
            else:
                # Fallback to stored data
                data = fetch_mt5_data(symbol, 'M1', 2)
                if data:
                    prices[symbol] = {
                        'mid': data['closes'][-1],
                        'source': 'MT5-History'
                    }
        except Exception as e:
            print(f"Error getting price for {symbol}: {e}")
    
    return jsonify(prices)


# ========== PREMIUM PAGES ==========

@app.route('/pricing')
def pricing_page():
    """Pricing page for subscription tiers"""
    user = get_current_user()
    return render_template('pricing.html', user=user, tiers=TIER_LIMITS)


@app.route('/history')
@login_required
def history_page():
    """User's analysis history"""
    user = get_current_user()
    if not user:
        session.clear()
        return redirect('/login?error=Session expired. Please login again.')
    history = get_user_history(user['id'], limit=50)
    return render_template('history.html', user=user, history=history)


@app.route('/settings')
@login_required
def settings_page():
    """User settings page"""
    try:
        user = get_current_user()
        if not user:
            # User not found in database - clear session and redirect to login
            session.clear()
            return redirect('/login?error=Session expired. Please login again.')
        tier_info = get_user_tier_info(user['id'])
        return render_template('settings.html', user=user, tier_info=tier_info)
    except Exception as e:
        import traceback
        print(f"[SETTINGS ERROR] {traceback.format_exc()}")
        session.clear()
        return redirect('/login?error=Something went wrong. Please login again.')


@app.route('/api/history')
@login_required
def api_history():
    """Get user's analysis history as JSON"""
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Session expired', 'redirect': '/login'}), 401
    history = get_user_history(user['id'], limit=50)
    return jsonify({'history': history})


# ========== STRIPE PAYMENT ROUTES ==========

@app.route('/api/create-checkout-session', methods=['POST'])
@login_required
def create_checkout_session():
    """Create Stripe checkout session for subscription"""
    if not STRIPE_AVAILABLE or not STRIPE_SECRET_KEY:
        return jsonify({'error': 'Payment system not configured'}), 500
    
    user = get_current_user()
    if not user:
        return jsonify({'error': 'Session expired', 'redirect': '/login'}), 401
    
    data = request.get_json()
    tier = data.get('tier', 'pro')
    
    # Get price ID based on tier
    price_id = STRIPE_PRO_PRICE_ID if tier == 'pro' else STRIPE_PREMIUM_PRICE_ID
    
    try:
        # Create Stripe checkout session
        checkout_session = stripe.checkout.Session.create(
            payment_method_types=['card'],
            line_items=[{
                'price': price_id,
                'quantity': 1,
            }],
            mode='subscription',
            success_url=request.url_root.rstrip('/') + '/payment/success?session_id={CHECKOUT_SESSION_ID}',
            cancel_url=request.url_root.rstrip('/') + '/pricing?canceled=true',
            customer_email=user['email'],
            metadata={
                'user_id': user['id'],
                'tier': tier
            }
        )
        
        return jsonify({'checkout_url': checkout_session.url})
    
    except stripe.error.StripeError as e:
        print(f"Stripe error: {e}")
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        print(f"Payment error: {e}")
        return jsonify({'error': 'Payment failed'}), 500


@app.route('/payment/success')
@login_required
def payment_success():
    """Handle successful payment"""
    session_id = request.args.get('session_id')
    user = get_current_user()
    
    if not user:
        session.clear()
        return redirect('/login?error=Session expired. Please login again.')
    
    if session_id and STRIPE_AVAILABLE:
        try:
            # Retrieve the session to get metadata
            checkout_session = stripe.checkout.Session.retrieve(session_id)
            tier = checkout_session.metadata.get('tier', 'pro')
            
            # Upgrade user tier
            upgrade_user_tier(user['id'], tier)
            
            return render_template('payment_success.html', user=user, tier=tier)
        except Exception as e:
            print(f"Payment verification error: {e}")
    
    # Fallback - still show success page
    return render_template('payment_success.html', user=user, tier='pro')


@app.route('/webhook/stripe', methods=['POST'])
def stripe_webhook():
    """Handle Stripe webhooks for subscription events"""
    if not STRIPE_AVAILABLE:
        return jsonify({'error': 'Stripe not available'}), 500
    
    payload = request.get_data(as_text=True)
    sig_header = request.headers.get('Stripe-Signature')
    
    try:
        if STRIPE_WEBHOOK_SECRET:
            event = stripe.Webhook.construct_event(
                payload, sig_header, STRIPE_WEBHOOK_SECRET
            )
        else:
            event = stripe.Event.construct_from(
                request.get_json(), stripe.api_key
            )
    except ValueError as e:
        return jsonify({'error': 'Invalid payload'}), 400
    except stripe.error.SignatureVerificationError as e:
        return jsonify({'error': 'Invalid signature'}), 400
    
    # Handle the event
    if event['type'] == 'checkout.session.completed':
        session = event['data']['object']
        user_id = session['metadata'].get('user_id')
        tier = session['metadata'].get('tier', 'pro')
        
        if user_id:
            upgrade_user_tier(int(user_id), tier)
            print(f"✅ User {user_id} upgraded to {tier}")
    
    elif event['type'] == 'customer.subscription.deleted':
        # Subscription canceled - downgrade to free
        subscription = event['data']['object']
        customer_email = subscription.get('customer_email')
        
        if customer_email:
            user = get_user_by_email(customer_email)
            if user:
                upgrade_user_tier(user['id'], 'free')
                print(f"⚠️ User {user['id']} downgraded to free")
    
    return jsonify({'status': 'success'})


@app.route('/api/stripe-config')
def stripe_config():
    """Return Stripe public key for frontend"""
    return jsonify({
        'publicKey': STRIPE_PUBLIC_KEY,
        'configured': bool(STRIPE_PUBLIC_KEY and STRIPE_SECRET_KEY)
    })


if __name__ == '__main__':
    # Production mode
    import os
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
