# app.py - Main Flask Application
import os
import json
from datetime import datetime
from flask import Flask, request, jsonify
import gradio as gr
from celery import Celery
import sqlite3
from contextlib import contextmanager
import uuid
from typing import List, Dict, Optional
import logging

from transformers import TFGPT2LMHeadModel, GPT2Tokenizer
import tensorflow as tf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------
# Configuration
# ---------------------------
class Config:
    DATABASE_PATH = 'social_media.db'
    CELERY_BROKER_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    LOCAL_MODEL_DIR = "./models/gpt2"
    MAX_POSTS_PER_CAMPAIGN = 30

config = Config()

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL=config.CELERY_BROKER_URL,
    CELERY_RESULT_BACKEND=config.CELERY_RESULT_BACKEND
)

# ---------------------------
# Celery Setup
# ---------------------------
def make_celery(app):
    celery = Celery(
        app.import_name,
        backend=app.config['CELERY_RESULT_BACKEND'],
        broker=app.config['CELERY_BROKER_URL']
    )
    celery.conf.update(app.config)
    return celery

celery = make_celery(app)

# ---------------------------
# Database Setup
# ---------------------------
def init_database():
    """Initialize the database with required tables."""
    with sqlite3.connect(config.DATABASE_PATH) as conn:
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

@contextmanager
def get_db_connection():
    """Context manager for database connections."""
    conn = sqlite3.connect(config.DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

# ---------------------------
# LLM Setup (Singleton Pattern)
# ---------------------------
class LLMManager:
    _instance = None
    _tokenizer = None
    _model = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LLMManager, cls).__new__(cls)
            cls._instance._initialize_model()
        return cls._instance
    
    def _initialize_model(self):
        """Initialize the LLM model once."""
        try:
            if os.path.exists(config.LOCAL_MODEL_DIR):
                logger.info(f"Loading GPT-2 from {config.LOCAL_MODEL_DIR}")
                self._tokenizer = GPT2Tokenizer.from_pretrained(config.LOCAL_MODEL_DIR)
                self._model = TFGPT2LMHeadModel.from_pretrained(config.LOCAL_MODEL_DIR)
            else:
                logger.info("Downloading and saving GPT-2...")
                self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self._model = TFGPT2LMHeadModel.from_pretrained("gpt2", from_pt=True)
                
                # Add pad token
                self._tokenizer.pad_token = self._tokenizer.eos_token
                
                os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)
                self._tokenizer.save_pretrained(config.LOCAL_MODEL_DIR)
                self._model.save_pretrained(config.LOCAL_MODEL_DIR)
            
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
        """Generate text using TensorFlow GPT-2."""
        try:
            inputs = self._tokenizer(prompt, return_tensors="tf", max_length=512, truncation=True)
            outputs = self._model.generate(
                inputs["input_ids"],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id
            )
            full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Remove the original prompt from the beginning
            generated_only = full_text[len(prompt):].strip()
            return generated_only if generated_only else full_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return f"Generated content for: {prompt[:50]}..."

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
        return [f"Professional {business_type} update #{i+1}" for i in range(count)]

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
                    meta={'current': i, 'total': total_posts, 'status': f'Generating post {i+1}/{total_posts}'}
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
                
                logger.info(f"Generated post {i+1}/{total_posts} for campaign {campaign_id}")
                
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
# Gradio Development UI
# ---------------------------
def create_gradio_interface():
    """Create Gradio interface for development."""
    
    def create_campaign_ui(business_type, platform, total_posts):
        """Gradio function to create campaign."""
        try:
            import requests
            response = requests.post(f'http://localhost:8000/api/campaigns/create', json={
                'business_type': business_type,
                'platform': platform,
                'total_posts': int(total_posts)
            })
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def check_campaign_status(campaign_id):
        """Check campaign status."""
        try:
            import requests
            response = requests.get(f'http://localhost:8000/api/campaigns/{campaign_id}/status')
            return json.dumps(response.json(), indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
    
    def get_campaign_posts_ui(campaign_id):
        """Get campaign posts."""
        try:
            import requests
            response = requests.get(f'http://localhost:8000/api/campaigns/{campaign_id}/posts')
            data = response.json()
            if data.get('success'):
                posts = data['posts']
                result = ""
                for post in posts:
                    result += f"=== Post #{post['post_order']}: {post['headline']} ===\n"
                    result += f"Status: {post['status']}\n"
                    if post['content']:
                        result += f"Content: {post['content']}\n"
                    result += "\n" + "="*50 + "\n\n"
                return result
            return json.dumps(data, indent=2)
        except Exception as e:
            return f"Error: {str(e)}"
    
    # Create Gradio blocks interface
    with gr.Blocks(title="Social Media Post Generator") as demo:
        gr.Markdown("# 🚀 Social Media Post Generator (Two-Stage Process)")
        
        with gr.Tab("Create Campaign"):
            gr.Markdown("## Stage 1: Create Campaign & Generate Headlines")
            with gr.Row():
                business_type = gr.Textbox(label="Business Type", value="Coffee Shop")
                platform = gr.Dropdown(["Facebook", "Instagram", "LinkedIn", "Twitter"], value="Instagram", label="Platform")
                total_posts = gr.Number(label="Total Posts", value=5, minimum=1, maximum=30)
            
            create_btn = gr.Button("Create Campaign", variant="primary")
            campaign_output = gr.Code(label="Campaign Result", language="json")
            
            create_btn.click(
                create_campaign_ui,
                inputs=[business_type, platform, total_posts],
                outputs=campaign_output
            )
        
        with gr.Tab("Check Status"):
            gr.Markdown("## Monitor Campaign Progress")
            campaign_id_input = gr.Textbox(label="Campaign ID")
            check_btn = gr.Button("Check Status")
            status_output = gr.Code(label="Status", language="json")
            
            check_btn.click(
                check_campaign_status,
                inputs=campaign_id_input,
                outputs=status_output
            )
        
        with gr.Tab("View Posts"):
            gr.Markdown("## View Generated Posts")
            campaign_id_posts = gr.Textbox(label="Campaign ID")
            view_btn = gr.Button("View Posts")
            posts_output = gr.Textbox(label="Generated Posts", lines=20)
            
            view_btn.click(
                get_campaign_posts_ui,
                inputs=campaign_id_posts,
                outputs=posts_output
            )
    
    return demo

# ---------------------------
# Application Initialization
# ---------------------------
@app.before_first_request
def initialize_app():
    """Initialize the application."""
    init_database()
    logger.info("Application initialized successfully")

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

@app.route('/ui')
def ui():
    """Launch Gradio interface."""
    demo = create_gradio_interface()
    return demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        prevent_thread_lock=True
    )

# ---------------------------
# Main Application Runner
# ---------------------------
if __name__ == '__main__':
    # Initialize database
    init_database()
    
    # Run Flask app
    app.run(host="0.0.0.0", port=8000, debug=True)