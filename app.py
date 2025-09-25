# app.py - Main Flask Application
import os

from flask import Flask, request, jsonify
import sqlite3
from contextlib import contextmanager
import uuid
from typing import List, Dict, Optional
import logging

from src.contentgen.celery import make_celery
from src.contentgen.config import Config
from src.contentgen.database import init_database
from src.contentgen.llm import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

config = Config()

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)
# app.config.update(
#     CELERY_BROKER_URL=config.CELERY_BROKER_URL,
#     CELERY_RESULT_BACKEND=config.CELERY_RESULT_BACKEND
# )
app.config.update(
    CELERY_BROKER_URL="memory://",
    CELERY_RESULT_BACKEND="rpc://",
    CELERY_TASK_ALWAYS_EAGER=True  # run tasks synchronously, no broker needed
)


# ---------------------------
# Application Initialization
# ---------------------------
def initialize_app():
    """Initialize the application."""
    init_database()
    logger.info("Application initialized successfully")


@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


celery = make_celery(app)
# Initialize LLM manager
llm_manager = LLMManager()


# ---------------------------
# Content Generation Functions
# ---------------------------
def generate_headlines_batch(business_type: str, platform: str, count: int) -> List[str]:
    """Generate multiple headlines at once."""
    headline_template = f"""Generate {count} engaging social media headlines for a {business_type} on {platform}.

Requirements:
- Each headline should be unique and engaging  
- Suitable for {platform} audience
- Professional tone for {business_type}
- Maximum 60 characters each
- One headline per line

Headlines:
1."""

    try:
        generated_text = llm_manager.generate_text(headline_template, max_new_tokens=200, temperature=0.8)

        # Parse headlines from generated text
        headlines = []
        lines = generated_text.split('\n')

        for line in lines:
            line = line.strip()
            if line and len(line) > 3:  # Skip very short lines
                # Clean up numbering and formatting
                headline = line.lstrip('0123456789.- ').strip()
                if headline and len(headline) <= 60 and len(headlines) < count:
                    headlines.append(headline)

        # Ensure we have enough headlines with fallbacks
        while len(headlines) < count:
            fallback_ideas = [
                f"Latest updates from your favorite {business_type}",
                f"What's new at {business_type} this week?",
                f"Behind the scenes at {business_type}",
                f"Special announcement from {business_type}",
                f"Customer favorites at {business_type}",
                f"{business_type} community spotlight",
                f"Tips and tricks from {business_type}",
                f"Celebrating milestones at {business_type}"
            ]
            idx = len(headlines) % len(fallback_ideas)
            headlines.append(fallback_ideas[idx])

        return headlines[:count]

    except Exception as e:
        logger.error(f"Error generating headlines: {str(e)}")
        # Return fallback headlines
        return [f"Professional {business_type} update #{i + 1}" for i in range(count)]


def generate_full_post(headline: str, business_type: str, platform: str) -> str:
    """Generate a full post from a headline."""
    post_template = f"""You are an expert social media marketer. Create an engaging {platform} post for a {business_type}.

Headline: "{headline}"

Create a complete social media post that:
- Expands on the headline naturally
- Includes relevant hashtags for {platform}
- Has appropriate length for {platform}
- Includes a call-to-action
- Maintains professional tone

Post:"""

    try:
        generated_post = llm_manager.generate_text(post_template, max_new_tokens=250, temperature=0.7)

        # Clean up the generated post
        post = generated_post.strip()

        # Ensure we have some basic content if generation fails
        if not post or len(post) < 20:
            post = f"🌟 {headline}\n\nWe're excited to share this update with our {platform} community! Stay tuned for more from {business_type}.\n\n#{business_type.replace(' ', '')} #{platform.lower()} #community"

        return post

    except Exception as e:
        logger.error(f"Error generating post: {str(e)}")
        return f"🌟 {headline}\n\nExciting updates from {business_type}! Follow us for more.\n\n#{business_type.replace(' ', '')} #{platform.lower()}"


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
    def save_headlines(campaign_id: str, headlines: List[str]):
        """Save generated headlines to database."""
        with get_db_connection() as conn:
            for i, headline in enumerate(headlines):
                headline_id = str(uuid.uuid4())
                conn.execute('''
                    INSERT INTO post_headlines (id, campaign_id, headline, post_order, status)
                    VALUES (?, ?, ?, ?, 'ready')
                ''', (headline_id, campaign_id, headline, i + 1))

                # Create corresponding post entry
                post_id = str(uuid.uuid4())
                conn.execute('''
                    INSERT INTO generated_posts (id, campaign_id, headline_id, status)
                    VALUES (?, ?, ?, 'pending')
                ''', (post_id, campaign_id, headline_id))

            # Update campaign status
            conn.execute('''
                UPDATE campaigns SET status = 'posts_generating' WHERE id = ?
            ''', (campaign_id,))
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
    def get_campaign_info(campaign_id: str) -> Optional[Dict]:
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
                SELECT COUNT(*) as count FROM post_headlines WHERE campaign_id = ?
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
    def get_campaign_posts(campaign_id: str) -> List[Dict]:
        """Get all posts for a campaign."""
        with get_db_connection() as conn:
            posts = conn.execute('''
                SELECT p.id, h.headline, h.post_order, p.content, p.status, 
                       p.created_at, p.completed_at, p.error_message
                FROM generated_posts p
                JOIN post_headlines h ON p.headline_id = h.id
                WHERE p.campaign_id = ?
                ORDER BY h.post_order
            ''', (campaign_id,)).fetchall()

            return [dict(post) for post in posts]


# ---------------------------
# Celery Tasks
# ---------------------------
@celery.task(bind=True)
def generate_posts_task(self, campaign_id: str):
    """Background task to generate all posts for a campaign."""
    try:
        # Get campaign info
        with get_db_connection() as conn:
            campaign = conn.execute('''
                SELECT business_type, platform FROM campaigns WHERE id = ?
            ''', (campaign_id,)).fetchone()

            if not campaign:
                raise ValueError(f"Campaign {campaign_id} not found")

            # Get pending posts
            pending_posts = conn.execute('''
                SELECT p.id, p.headline_id, h.headline
                FROM generated_posts p
                JOIN post_headlines h ON p.headline_id = h.id
                WHERE p.campaign_id = ? AND p.status = 'pending'
                ORDER BY h.post_order
            ''', (campaign_id,)).fetchall()

        total_posts = len(pending_posts)
        business_type = campaign['business_type']
        platform = campaign['platform']

        # Generate posts one by one
        for i, post_data in enumerate(pending_posts):
            try:
                # Update task progress
                self.update_state(
                    state='PROGRESS',
                    meta={'current': i, 'total': total_posts, 'status': f'Generating post {i + 1}/{total_posts}'}
                )

                # Generate full post
                full_content = generate_full_post(
                    post_data['headline'],
                    business_type,
                    platform
                )

                # Save to database
                with get_db_connection() as conn:
                    conn.execute('''
                        UPDATE generated_posts 
                        SET content = ?, status = 'completed', completed_at = CURRENT_TIMESTAMP
                        WHERE id = ?
                    ''', (full_content, post_data['id']))
                    conn.commit()

                logger.info(f"Generated post {i + 1}/{total_posts} for campaign {campaign_id}")

            except Exception as e:
                logger.error(f"Error generating post {post_data['id']}: {str(e)}")
                # Mark post as failed
                with get_db_connection() as conn:
                    conn.execute('''
                        UPDATE generated_posts 
                        SET status = 'failed', error_message = ?
                        WHERE id = ?
                    ''', (str(e), post_data['id']))
                    conn.commit()

        # Update campaign status to completed
        DatabaseManager.update_campaign_status(campaign_id, 'completed')

        return {'status': 'completed', 'campaign_id': campaign_id}

    except Exception as e:
        logger.error(f"Task failed for campaign {campaign_id}: {str(e)}")
        DatabaseManager.update_campaign_status(campaign_id, 'failed', str(e))
        raise


# ---------------------------
# Flask API Routes
# ---------------------------
@app.route('/api/campaigns/create', methods=['POST'])
def create_campaign():
    """Stage 1: Create campaign and generate headlines."""
    try:
        data = request.json
        business_type = data.get('business_type', '').strip()
        platform = data.get('platform', '').strip()
        total_posts = data.get('total_posts', 10)

        # Validation
        if not business_type or not platform:
            return jsonify({'error': 'business_type and platform are required'}), 400

        if not isinstance(total_posts, int) or total_posts < 1 or total_posts > config.MAX_POSTS_PER_CAMPAIGN:
            return jsonify({'error': f'total_posts must be between 1 and {config.MAX_POSTS_PER_CAMPAIGN}'}), 400

        # Create campaign
        campaign_id = DatabaseManager.create_campaign(business_type, platform, total_posts)

        # Generate headlines (Stage 1)
        logger.info(f"Generating {total_posts} headlines for campaign {campaign_id}")
        headlines = generate_headlines_batch(business_type, platform, total_posts)

        # Save headlines
        DatabaseManager.save_headlines(campaign_id, headlines)

        # Start background task for post generation (Stage 2)
        task = generate_posts_task.delay(campaign_id)

        return jsonify({
            'success': True,
            'campaign_id': campaign_id,
            'task_id': task.id,
            'headlines': headlines,
            'message': 'Headlines generated successfully. Posts are being generated in background.'
        })

    except Exception as e:
        logger.error(f"Error creating campaign: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/campaigns/<campaign_id>/status', methods=['GET'])
def get_campaign_status(campaign_id):
    """Get campaign status and progress."""
    try:
        campaign_info = DatabaseManager.get_campaign_info(campaign_id)
        if not campaign_info:
            return jsonify({'error': 'Campaign not found'}), 404

        return jsonify({
            'success': True,
            'campaign': campaign_info
        })

    except Exception as e:
        logger.error(f"Error getting campaign status: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/campaigns/<campaign_id>/posts', methods=['GET'])
def get_campaign_posts(campaign_id):
    """Get all posts for a campaign."""
    try:
        posts = DatabaseManager.get_campaign_posts(campaign_id)
        return jsonify({
            'success': True,
            'posts': posts
        })

    except Exception as e:
        logger.error(f"Error getting campaign posts: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/task/<task_id>/status', methods=['GET'])
def get_task_status(task_id):
    """Get background task status."""
    try:
        task = generate_posts_task.AsyncResult(task_id)

        if task.state == 'PENDING':
            response = {
                'state': task.state,
                'status': 'Task is waiting to start...'
            }
        elif task.state == 'PROGRESS':
            response = {
                'state': task.state,
                'current': task.info.get('current', 0),
                'total': task.info.get('total', 1),
                'status': task.info.get('status', '')
            }
        elif task.state == 'SUCCESS':
            response = {
                'state': task.state,
                'result': task.info
            }
        else:  # FAILURE
            response = {
                'state': task.state,
                'error': str(task.info)
            }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500



# ---------------------------
# Routes
# ---------------------------
@app.route('/')
def index():
    return jsonify({
        'message': 'Social Media Post Generator API',
        'version': '1.0.0',
        'endpoints': {
            'create_campaign': '/api/campaigns/create',
            'campaign_status': '/api/campaigns/<id>/status',
            'campaign_posts': '/api/campaigns/<id>/posts',
            'task_status': '/api/task/<id>/status'
        }
    })


# ---------------------------
# Main Application Runner
# ---------------------------
if __name__ == '__main__':
    # Initialize app
    initialize_app()

    # Run Flask app
    app.run(host="0.0.0.0", port=int(os.environ.get("APP_PORT", 8000)), debug=True)
