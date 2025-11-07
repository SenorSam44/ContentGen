# ---------------------------
# Database Setup
# ---------------------------
import os
import sqlite3

from dotenv import load_dotenv

load_dotenv()

def init_database():
    """Initialize the database with required tables."""
    with sqlite3.connect(os.getenv("DATABASE_PATH", "social_media.db")) as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS campaigns (
                id TEXT PRIMARY KEY,
                business_type TEXT NOT NULL,
                platform TEXT NOT NULL,
                total_posts INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT
            );
            
            CREATE TABLE IF NOT EXISTS post_headlines (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                headline TEXT NOT NULL,
                post_order INTEGER NOT NULL,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
            );
            
            CREATE TABLE IF NOT EXISTS generated_posts (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                headline_id TEXT NOT NULL,
                content TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id),
                FOREIGN KEY (headline_id) REFERENCES post_headlines (id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_campaign_status ON campaigns(status);
            CREATE INDEX IF NOT EXISTS idx_headlines_campaign ON post_headlines(campaign_id);
            CREATE INDEX IF NOT EXISTS idx_posts_campaign ON generated_posts(campaign_id);
        ''')

