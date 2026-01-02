"""
Authentication System for Pro Forex Analyzer
- Google OAuth Login
- Email/Password Login
- User Management
- Subscription Tiers
"""

import os
import sqlite3
import hashlib
import secrets
from datetime import datetime, timedelta
from functools import wraps
from flask import session, redirect, url_for, request, jsonify

# ========== DATABASE SETUP ==========
# Use persistent disk on Render if available, otherwise use local
import os

DB_PATH = 'forex_users.db'
if os.environ.get('RENDER'):
    # Try persistent disk first
    disk_path = '/opt/render/project/src/data'
    if os.path.exists(disk_path) or os.access(os.path.dirname(disk_path), os.W_OK):
        try:
            os.makedirs(disk_path, exist_ok=True)
            DB_PATH = os.path.join(disk_path, 'forex_users.db')
        except:
            DB_PATH = 'forex_users.db'

print(f"[DATABASE] Using path: {DB_PATH}")

def init_db():
    """Initialize the user database"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT,
            name TEXT,
            picture TEXT,
            provider TEXT DEFAULT 'email',
            tier TEXT DEFAULT 'free',
            analyses_today INTEGER DEFAULT 0,
            last_analysis_date TEXT,
            total_analyses INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            last_login TEXT
        )
    ''')

    # Analysis history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS analysis_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            symbol TEXT,
            timeframe TEXT,
            signal_type TEXT,
            confidence INTEGER,
            entry_price REAL,
            stop_loss REAL,
            take_profit REAL,
            analysis_text TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    # Subscriptions table
    c.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER UNIQUE,
            tier TEXT DEFAULT 'free',
            stripe_customer_id TEXT,
            stripe_subscription_id TEXT,
            expires_at TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')

    conn.commit()
    conn.close()
    print("✅ Database initialized")

# Initialize on import
init_db()

# ========== ADMIN EMAILS (UNLIMITED ACCESS) ==========
ADMIN_EMAILS = [
    'vanndom300@gmail.com',  # Owner - Unlimited access
]

# ========== TIER LIMITS ==========
TIER_LIMITS = {
    'free': {
        'analyses_per_day': 5,
        'mtf_enabled': False,
        'export_pdf': False,
        'priority_ai': False,
        'price': 0
    },
    'pro': {
        'analyses_per_day': 50,
        'mtf_enabled': True,
        'export_pdf': True,
        'priority_ai': False,
        'price': 19.99
    },
    'premium': {
        'analyses_per_day': 999999,
        'mtf_enabled': True,
        'export_pdf': True,
        'priority_ai': True,
        'price': 49.99
    }
}

# ========== PASSWORD HASHING ==========
def hash_password(password):
    salt = secrets.token_hex(16)
    hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
    return f"{salt}:{hash_obj.hex()}"

def verify_password(password, stored_hash):
    try:
        salt, hash_value = stored_hash.split(':')
        hash_obj = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return hash_obj.hex() == hash_value
    except:
        return False

# ========== USER MANAGEMENT ==========
def get_user_by_email(email):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE email = ?', (email,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def get_user_by_id(user_id):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None

def create_user(email, password=None, name=None, picture=None, provider='email'):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    password_hash = hash_password(password) if password else None
    try:
        c.execute('''
            INSERT INTO users (email, password_hash, name, picture, provider, last_login)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (email, password_hash, name, picture, provider, datetime.now().isoformat()))
        conn.commit()
        user_id = c.lastrowid
        conn.close()
        return user_id
    except sqlite3.IntegrityError:
        conn.close()
        return None

def update_user_login(user_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE users SET last_login = ? WHERE id = ?', (datetime.now().isoformat(), user_id))
    conn.commit()
    conn.close()

def update_user_google(email, name, picture):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE users SET name = ?, picture = ?, last_login = ? WHERE email = ?',
              (name, picture, datetime.now().isoformat(), email))
    conn.commit()
    conn.close()

# ========== ANALYSIS TRACKING ==========
def check_analysis_limit(user_id):
    """Check if user has reached daily analysis limit"""
    user = get_user_by_id(user_id)
    if not user:
        return False, "User not found"
    
    # ADMIN BYPASS - Unlimited access for admin emails
    user_email = user.get('email', '').lower()
    if user_email in [e.lower() for e in ADMIN_EMAILS]:
        return True, 999999  # Unlimited for admin
    
    tier = user.get('tier', 'free')
    limit = TIER_LIMITS.get(tier, TIER_LIMITS['free'])['analyses_per_day']
    
    today = datetime.now().strftime('%Y-%m-%d')
    last_date = user.get('last_analysis_date', '') or ''
    analyses_today = user.get('analyses_today', 0) or 0
    
    print(f"[LIMIT CHECK] User: {user_email}, Tier: {tier}, Today: {today}, Last: {last_date}, Count: {analyses_today}, Limit: {limit}")
    
    # Only reset if it's a NEW day
    if last_date != today:
        print(f"[LIMIT CHECK] New day detected - resetting count from {analyses_today} to 0")
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('UPDATE users SET analyses_today = 0, last_analysis_date = ? WHERE id = ?', (today, user_id))
        conn.commit()
        conn.close()
        analyses_today = 0  # Reset local variable too
    
    if analyses_today >= limit:
        print(f"[LIMIT CHECK] BLOCKED - {analyses_today} >= {limit}")
        return False, f"Daily limit reached ({limit} analyses). Upgrade to Pro for more!"
    
    remaining = limit - analyses_today
    print(f"[LIMIT CHECK] ALLOWED - {remaining} remaining")
    return True, remaining

def increment_analysis_count(user_id):
    today = datetime.now().strftime('%Y-%m-%d')
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE users SET analyses_today = analyses_today + 1, total_analyses = total_analyses + 1, last_analysis_date = ? WHERE id = ?''', (today, user_id))
    conn.commit()
    conn.close()

def save_analysis(user_id, symbol, timeframe, signal_type, confidence, entry, sl, tp, analysis_text):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''INSERT INTO analysis_history (user_id, symbol, timeframe, signal_type, confidence, entry_price, stop_loss, take_profit, analysis_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)''',
              (user_id, symbol, timeframe, signal_type, confidence, entry, sl, tp, analysis_text))
    conn.commit()
    conn.close()

def get_user_history(user_id, limit=20):
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute('SELECT * FROM analysis_history WHERE user_id = ? ORDER BY created_at DESC LIMIT ?', (user_id, limit))
    history = [dict(row) for row in c.fetchall()]
    conn.close()
    return history

# ========== SUBSCRIPTION MANAGEMENT ==========
def upgrade_user_tier(user_id, tier):
    if tier not in TIER_LIMITS:
        return False
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('UPDATE users SET tier = ? WHERE id = ?', (tier, user_id))
    conn.commit()
    conn.close()
    return True

def get_user_tier_info(user_id):
    """Get user's tier and limits"""
    user = get_user_by_id(user_id)
    if not user:
        return None
    
    # ADMIN CHECK - Owner gets unlimited
    user_email = user.get('email', '').lower().strip()
    admin_emails_lower = [e.lower().strip() for e in ADMIN_EMAILS]
    is_admin = user_email in admin_emails_lower
    print(f"[ADMIN CHECK] email='{user_email}', admins={admin_emails_lower}, is_admin={is_admin}")
    
    # FORCE ADMIN for vanndom300@gmail.com
    if 'vanndom300' in user_email or is_admin:
        return {
            'tier': 'owner',
            'is_admin': True,
            'limits': {'analyses_per_day': 999999, 'mtf_enabled': True, 'export_pdf': True, 'priority_ai': True, 'price': 0},
            'analyses_today': user.get('analyses_today', 0),
            'analyses_remaining': 'Unlimited',
            'total_analyses': user.get('total_analyses', 0)
        }
    
    tier = user.get('tier', 'free')
    limits = TIER_LIMITS.get(tier, TIER_LIMITS['free'])
    today = datetime.now().strftime('%Y-%m-%d')
    analyses_today = user.get('analyses_today', 0) if user.get('last_analysis_date') == today else 0
    
    return {
        'tier': tier,
        'is_admin': False,
        'limits': limits,
        'analyses_today': analyses_today,
        'analyses_remaining': limits['analyses_per_day'] - analyses_today,
        'total_analyses': user.get('total_analyses', 0)
    }

# ========== LOGIN DECORATOR ==========
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.is_json or (request.content_type and 'multipart/form-data' in request.content_type):
                return jsonify({'error': 'Login required', 'redirect': '/login'}), 401
            return redirect('/login')
        # Verify user exists in database
        user = get_user_by_id(session['user_id'])
        if not user:
            session.clear()
            if request.is_json or (request.content_type and 'multipart/form-data' in request.content_type):
                return jsonify({'error': 'Session expired. Please login again.', 'redirect': '/login'}), 401
            return redirect('/login?error=Session expired. Please login again.')
        # Store user in request context for reuse
        from flask import g
        g.current_user = user
        return f(*args, **kwargs)
    return decorated_function

def get_current_user():
    """Get current user - uses cached version from login_required if available"""
    from flask import g
    # First check if already loaded by login_required decorator
    if hasattr(g, 'current_user') and g.current_user:
        return g.current_user
    # Fallback to database lookup
    if 'user_id' in session:
        user = get_user_by_id(session['user_id'])
        if user:
            g.current_user = user
        return user
    return None
