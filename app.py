from flask import Flask
from routes.generate import generate_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(generate_bp, url_prefix="/generate")
    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=8000)
