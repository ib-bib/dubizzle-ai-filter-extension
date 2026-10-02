import sqlite3
import datetime
import os

DB_PATH = "usage.db"
FREE_TIER_DAILY_LIMIT = 20

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    # Keep old table for backwards compatibility or global tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            month_year TEXT PRIMARY KEY,
            cost_usd REAL
        )
    ''')
    
    # New table for per-user tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_usage (
            user_id TEXT,
            date_key TEXT,
            requests_count INTEGER DEFAULT 0,
            vision_requests_count INTEGER DEFAULT 0,
            PRIMARY KEY (user_id, date_key)
        )
    ''')
    
    # Ensure vision column exists for old databases
    try:
        c.execute("ALTER TABLE user_usage ADD COLUMN vision_requests_count INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass # Column already exists
    
    conn.commit()
    conn.close()

def get_current_date_key():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m-%d")

def get_current_month_key():
    now = datetime.datetime.now()
    return now.strftime("%Y-%m")

def get_reset_time_daily():
    now = datetime.datetime.now()
    next_day = now.replace(hour=0, minute=0, second=0, microsecond=0) + datetime.timedelta(days=1)
    return next_day.isoformat() + "Z"

TOTAL_DAILY_LIMIT = 15
VISION_DAILY_LIMIT = 5

def check_budget(user_id: str = None, is_vision: bool = False) -> dict:
    """Returns dict with 'allowed', 'reason', and 'reset_time' if disabled."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Anonymous users cannot use the extension
    if not user_id or user_id == "anonymous_guest":
        return {
            "allowed": False,
            "reason": "Please login with Google to use the AI Assistant.",
            "reset_time": get_reset_time_daily()
        }
        
    date_key = get_current_date_key()
    c.execute("SELECT requests_count, vision_requests_count FROM user_usage WHERE user_id = ? AND date_key = ?", (user_id, date_key))
    row = c.fetchone()
    
    current_requests = row[0] if row else 0
    current_vision = row[1] if row else 0
    
    if current_requests >= TOTAL_DAILY_LIMIT:
        conn.close()
        return {
            "allowed": False,
            "reason": f"Daily limit of {TOTAL_DAILY_LIMIT} searches reached. You are on cooldown until tomorrow.",
            "reset_time": get_reset_time_daily()
        }
        
    if is_vision and current_vision >= VISION_DAILY_LIMIT:
        conn.close()
        return {
            "allowed": False,
            "reason": f"Daily limit of {VISION_DAILY_LIMIT} vision-enabled searches reached. You can still do {TOTAL_DAILY_LIMIT - current_requests} normal searches today.",
            "reset_time": get_reset_time_daily()
        }

    # (Optional) You can still keep a global kill-switch budget here if you want
    # For now we'll rely on the user limits
    conn.close()
    return {
        "allowed": True,
        "current_requests": current_requests,
        "current_vision": current_vision,
        "total_limit": TOTAL_DAILY_LIMIT,
        "vision_limit": VISION_DAILY_LIMIT
    }

def add_usage(user_id: str, usd_cost: float, is_vision: bool = False):
    init_db()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    if user_id and user_id != "anonymous_guest":
        date_key = get_current_date_key()
        c.execute("SELECT requests_count, vision_requests_count FROM user_usage WHERE user_id = ? AND date_key = ?", (user_id, date_key))
        row = c.fetchone()
        if row:
            vision_add = 1 if is_vision else 0
            c.execute("UPDATE user_usage SET requests_count = requests_count + 1, vision_requests_count = vision_requests_count + ? WHERE user_id = ? AND date_key = ?", (vision_add, user_id, date_key))
        else:
            vision_val = 1 if is_vision else 0
            c.execute("INSERT INTO user_usage (user_id, date_key, requests_count, vision_requests_count) VALUES (?, ?, ?, ?)", (user_id, date_key, 1, vision_val))
            
    # Keep global cost tracking active
    if usd_cost > 0:
        month_key = get_current_month_key()
        c.execute("SELECT cost_usd FROM usage WHERE month_year = ?", (month_key,))
        row = c.fetchone()
        if row:
            new_cost = row[0] + usd_cost
            c.execute("UPDATE usage SET cost_usd = ? WHERE month_year = ?", (new_cost, month_key))
        else:
            c.execute("INSERT INTO usage (month_year, cost_usd) VALUES (?, ?)", (month_key, usd_cost))

    conn.commit()
    conn.close()
