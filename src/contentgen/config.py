import os

class Config:
    DATABASE_PATH = 'social_media.db'
    CELERY_BROKER_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    CELERY_RESULT_BACKEND = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    LOCAL_MODEL_DIR = "./models/gpt2"
    MAX_POSTS_PER_CAMPAIGN = 30

