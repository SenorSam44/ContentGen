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


@app.route('/api/campaigns/<campaign_id>/develop_posts', methods=['POST'])
def develop_posts_for_campaign(campaign_id):
    """
    Generate detailed posts for all summaries of a given campaign.
    """
    try:
        data = request.get_json() or {}
        platform = data.get("platform", "").strip()
        business_type = data.get("business_type", "").strip()

        if not platform or not business_type:
            return jsonify({"error": "platform and business_type are required"}), 400

        # 1. Fetch summaries from database (if already saved)
        summaries = DatabaseManager.get_campaign_summaries(campaign_id)
        if not summaries:
            return jsonify({
                "error": "No summaries found for this campaign. Create one first."
            }), 404

        # 2. Extract summary texts
        summary_texts = [
            s["summary"] if isinstance(s, dict) and "summary" in s else s
            for s in summaries
        ]

        # 3. Generate posts for each summary
        posts = llm_manager.generate_posts(
            summaries=summary_texts,
            platform=platform,
            business_profile={"business_type": business_type, "platform": platform},
        )

        # 4. Save posts in database
        DatabaseManager.save_posts(campaign_id, posts)

        # 5. Return structured output
        return jsonify({
            "success": True,
            "campaign_id": campaign_id,
            "platform": platform,
            "posts": posts,
            "message": f"Generated {len(posts)} posts successfully."
        })

    except Exception as e:
        logger.error(f"Error generating posts: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "message": "Failed to generate posts."
        }), 500



@app.route('/api/develop_post', methods=["POST"])
def develop_post():
    data = request.get_json()
    # headline = data.get("headline", "").strip()
    business_type = data.get("business_type", "").strip()
    platform = data.get("platform", "").strip()

    if not business_type or not platform:
        return jsonify({"error": "headline, business_type, and platform are required"}), 400

    @stream_with_context
    def generate_stream():
        try:
            # Step 1 — Generate 5 summaries
            summary_prompt = f"""You are an expert social media strategist.
                Generate 5 short summaries for {platform} posts about a {business_type}.

                Each summary should be:
                - 1–2 sentences long
                - Distinct in tone or angle
                - No hashtags or emojis
                Output format: numbered list (1. ... 2. ... etc.)"""

            summaries_text = llm_manager.generate_text(summary_prompt, max_new_tokens=300)
            summaries = [s.strip(" .") for s in summaries_text.split("\n") if s.strip() and s[0].isdigit()]

            # if not summaries:
            #     summaries = [f"Short update about {headline} for {platform}."]

            # Step 2 — Expand each summary into a full post
            for i, summary in enumerate(summaries, start=1):
                expand_prompt = f"""You are a professional social media content creator.
                Using the following summary, write a complete post for {platform} about a {business_type}.

                Summary: "{summary}"

                Guidelines:
                - Keep the tone consistent and professional
                - Include 2–3 relevant hashtags
                - Add a call-to-action
                - Avoid repetition

                Post:"""

                full_post = llm_manager.generate_text(expand_prompt, max_new_tokens=800, temperature=0.9)

                # Step 3 — Stream the partial result
                yield f"data: {json.dumps({'index': i, 'summary': summary, 'post': full_post.strip()})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(generate_stream(), mimetype="text/event-stream")



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
