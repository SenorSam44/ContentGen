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
