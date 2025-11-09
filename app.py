# app.py - Main Flask Application

from flask import Flask, request, jsonify
import logging
import os
from dotenv import load_dotenv

import json
import re
from flask import Response, stream_with_context


from src.contentgen.celery import make_celery
from src.contentgen.database import init_database, get_db_connection, DatabaseManager
from src.contentgen.llm import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# ---------------------------
# Flask App Setup
# ---------------------------
app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL=os.getenv("CELERY_BROKER_URL", "memory://"),
    CELERY_RESULT_BACKEND=os.getenv("CELERY_RESULT_BACKEND", "rpc://"),
    CELERY_TASK_ALWAYS_EAGER=os.getenv("CELERY_TASK_ALWAYS_EAGER", "True").lower() == "true"
)

# ---------------------------
# Application Initialization
# ---------------------------
def initialize_app():
    """Initialize the application."""
    init_database()
    logger.info("Application initialized successfully")


celery = make_celery(app)
# Initialize LLM manager
llm_manager = LLMManager()

@celery.task
def generate_summaries_task(campaign_id, business_type, platform, total_posts):
    summaries = llm_manager.generate_summaries(
        topics=[business_type],
        business_profile={"business_type": business_type, "platform": platform},
        n=total_posts
    )
    DatabaseManager.save_summaries(campaign_id, summaries)
    return {"summaries": summaries}
# ---------------------------
# Flask API Routes
# ---------------------------
@app.route('/api/campaigns/create', methods=['POST'])
def create_campaign():
    data = request.json
    business_type = data.get('business_type', '').strip()
    platform = data.get('platform', '').strip()
    total_posts = int(data.get('total_posts', 10))

    max_posts = int(os.getenv("MAX_POSTS_PER_CAMPAIGN", 20))
    if not business_type or not platform:
        return jsonify({'error': 'business_type and platform are required'}), 400
    if not (1 <= total_posts <= max_posts):
        return jsonify({'error': f'total_posts must be between 1 and {max_posts}'}), 400

    # Step 1: Create campaign entry
    campaign_id = DatabaseManager.create_campaign(business_type, platform, total_posts)

    # Step 2: Generate summaries immediately (synchronous, for debugging)
    # try:
    summaries = llm_manager.generate_summaries(
        topics=[business_type],
        business_profile={"business_type": business_type, "platform": platform},
        n=total_posts
    )

    # Step 3: Save summaries to database
    # DatabaseManager.save_summaries(campaign_id, summaries)

    return jsonify({
        "success": True,
        "campaign_id": campaign_id,
        "summaries": summaries,
        "message": "Campaign created and summaries generated successfully."
    })

    # except Exception as e:
        # If something fails, update campaign status and return error
    DatabaseManager.update_campaign_status(campaign_id, 'failed', str(e))
    return jsonify({
        "success": False,
        "campaign_id": campaign_id,
        "error": str(e),
        "message": "Failed to generate summaries."
    }), 500

@app.route('/api/campaigns/<campaign_id>/summaries', methods=['GET'])
def get_campaign_summaries(campaign_id):
    """Get all summaries for a specific campaign after checking status."""
    try:
        # Step 1: Get campaign info
        campaign_info = DatabaseManager.get_campaign_info(campaign_id)
        if not campaign_info:
            return jsonify({'error': 'Campaign not found'}), 404

        # Step 2: Check status
        status = campaign_info['status']
        if status in ['headlines_generating', 'posts_generating']:
            return jsonify({
                'success': False,
                'campaign_id': campaign_id,
                'status': status,
                'message': 'Summaries are still being generated. Please try again later.'
            }), 202  # 202 Accepted, still processing
        elif status == 'failed':
            return jsonify({
                'success': False,
                'campaign_id': campaign_id,
                'status': status,
                'message': campaign_info.get('error_message', 'Task failed')
            }), 500

        # Step 3: Fetch summaries
        summaries = DatabaseManager.get_campaign_summaries(campaign_id)
        if not summaries:
            return jsonify({
                'success': True,
                'campaign_id': campaign_id,
                'status': status,
                'summaries': [],
                'message': 'No summaries generated yet.'
            })

        return jsonify({
            'success': True,
            'campaign_id': campaign_id,
            'status': status,
            'summaries': summaries
        })

    except Exception as e:
        logger.error(f"Error fetching summaries: {str(e)}")
        return jsonify({'error': str(e)}), 500



@app.route('/api/campaigns/develop_posts', methods=["POST"])
def develop_posts():
    data = request.get_json()
    campaign_id = data.get("campaign_id")
    summaries = data.get("summaries", [])
    platform = data.get("platform", "Facebook").strip()
    business_profile = data.get("business_profile", {})

    if not summaries or not isinstance(summaries, list):
        return jsonify({"success": False, "error": "summaries must be a list"}), 400

    # Extract clean summary texts
    summaries_list = [
        {"summary": s.get("summary", "").strip(), "topic": s.get("topic", "General").strip()}
        for s in summaries if s.get("summary", "").strip()
    ]

    # Call generate_posts instead of generate_text
    posts = llm_manager.generate_posts(
        summaries=summaries_list,
        platform=platform,
        business_profile=business_profile
    )

    return jsonify({
        "success": True,
        "campaign_id": campaign_id,
        "posts": posts,
        "message": f"Generated {len(posts)} posts successfully."
    })
    




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
