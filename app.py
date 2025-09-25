from flask import Flask
from config import Config
from extensions import celery
from routes.campaigns import campaigns_bp
from routes.tasks import tasks_bp
from database import init_database

def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    # Init celery
    celery.conf.update(app.config)

    # Init DB
    init_database()

    # Register blueprints
    app.register_blueprint(campaigns_bp, url_prefix="/api/campaigns")
    app.register_blueprint(tasks_bp, url_prefix="/api/task")

    @app.route("/")
    def index():
        return {
            "message": "Social Media Post Generator API",
            "version": "1.0.0"
        }

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000, debug=True)
