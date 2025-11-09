# ---------------------------
# Database Setup
# ---------------------------
import os
import sqlite3
import uuid
from contextlib import contextmanager

from dotenv import load_dotenv

load_dotenv()


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(os.getenv("DATABASE_PATH", "social_media.db"))
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


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
            CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                topic TEXT,
                summary TEXT NOT NULL,
                post_order INTEGER NOT NULL,
                status TEXT DEFAULT 'ready',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id)
            );

            CREATE TABLE IF NOT EXISTS generated_posts (
                id TEXT PRIMARY KEY,
                campaign_id TEXT NOT NULL,
                summary_id TEXT NOT NULL,
                content TEXT,
                status TEXT DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                completed_at TIMESTAMP,
                error_message TEXT,
                FOREIGN KEY (campaign_id) REFERENCES campaigns (id) ON DELETE CASCADE,
                FOREIGN KEY (summary_id) REFERENCES summaries (id) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_summary_campaign ON summaries(campaign_id);
            CREATE INDEX IF NOT EXISTS idx_post_summary ON generated_posts(summary_id);

            CREATE INDEX IF NOT EXISTS idx_campaign_status ON campaigns(status);
        ''')


# ---------------------------
# Database Operations
# ---------------------------
class DatabaseManager:
    @staticmethod
    def create_campaign(business_type: str, platform: str, total_posts: int) -> str:
        """Create a new campaign and return its ID."""
        campaign_id = str(uuid.uuid4())

        with get_db_connection() as conn:
            conn.execute('''
                INSERT INTO campaigns (id, business_type, platform, total_posts, status)
                VALUES (?, ?, ?, ?, 'headlines_generating')
            ''', (campaign_id, business_type, platform, total_posts))
            conn.commit()

        return campaign_id

    @staticmethod
    def get_campaign_summaries(campaign_id):
        with get_db_connection() as conn:
            rows = conn.execute(
                "SELECT id, topic, summary FROM summaries WHERE campaign_id = ? ORDER BY post_order",
                (campaign_id,)
            ).fetchall()
            return [{"id": r["id"], "topic": r["topic"], "summary": r["summary"]} for r in rows]


    @staticmethod
    def save_summaries(campaign_id: str, summaries: list[dict[str, str]]):
        """Save generated summaries to DB."""
        with get_db_connection() as conn:
            for i, item in enumerate(summaries):
                summary_text = item.get("summary", "").strip()
                topic = item.get("topic", "").strip()

                summary_id = str(uuid.uuid4())
                conn.execute('''
                    INSERT INTO summaries (id, campaign_id, topic, summary, post_order, status)
                    VALUES (?, ?, ?, ?, ?, 'ready')
                ''', (summary_id, campaign_id, topic, summary_text, i + 1))

                post_id = str(uuid.uuid4())
                conn.execute('''
                    INSERT INTO generated_posts (id, campaign_id, summary_id, status)
                    VALUES (?, ?, ?, 'pending')
                ''', (post_id, campaign_id, summary_id))

            conn.execute('UPDATE campaigns SET status = "posts_generating" WHERE id = ?', (campaign_id,))
            conn.commit()


    @staticmethod
    def update_campaign_status(campaign_id: str, status: str, error_message: str = None):
        """Update campaign status."""
        with get_db_connection() as conn:
            if status == 'completed':
                conn.execute('''
                    UPDATE campaigns SET status = ?, completed_at = CURRENT_TIMESTAMP, error_message = ?
                    WHERE id = ?
                ''', (status, error_message, campaign_id))
            else:
                conn.execute('''
                    UPDATE campaigns SET status = ?, error_message = ? WHERE id = ?
                ''', (status, error_message, campaign_id))
            conn.commit()

    @staticmethod
    def get_campaign_info(campaign_id: str) -> dict | None:
        """Get campaign information with progress."""
        with get_db_connection() as conn:
            # Get campaign details
            campaign = conn.execute('''
                SELECT * FROM campaigns WHERE id = ?
            ''', (campaign_id,)).fetchone()

            if not campaign:
                return None

            # Get progress information
            headlines_count = conn.execute('''
                SELECT COUNT(*) as count FROM summaries WHERE campaign_id = ?
            ''', (campaign_id,)).fetchone()['count']

            completed_posts = conn.execute('''
                SELECT COUNT(*) as count FROM generated_posts 
                WHERE campaign_id = ? AND status = 'completed'
            ''', (campaign_id,)).fetchone()['count']

            failed_posts = conn.execute('''
                SELECT COUNT(*) as count FROM generated_posts 
                WHERE campaign_id = ? AND status = 'failed'
            ''', (campaign_id,)).fetchone()['count']

            return {
                'id': campaign['id'],
                'business_type': campaign['business_type'],
                'platform': campaign['platform'],
                'total_posts': campaign['total_posts'],
                'status': campaign['status'],
                'created_at': campaign['created_at'],
                'completed_at': campaign['completed_at'],
                'error_message': campaign['error_message'],
                'progress': {
                    'headlines_generated': headlines_count,
                    'posts_completed': completed_posts,
                    'posts_failed': failed_posts,
                    'posts_pending': campaign['total_posts'] - completed_posts - failed_posts
                }
            }

    @staticmethod
    def get_campaign_posts(campaign_id: str) -> list[dict]:
        """Get all posts for a campaign."""
        with get_db_connection() as conn:
            posts = conn.execute('''
                SELECT p.id, h.headline, h.post_order, p.content, p.status, 
                       p.created_at, p.completed_at, p.error_message
                FROM generated_posts p
                JOIN summaries h ON p.headline_id = h.id
                WHERE p.campaign_id = ?
                ORDER BY h.post_order
            ''', (campaign_id,)).fetchall()

            return [dict(post) for post in posts]

