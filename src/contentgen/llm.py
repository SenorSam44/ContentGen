import logging
import os
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

from src.contentgen.config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()


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
                self._model = GPT2LMHeadModel.from_pretrained(config.LOCAL_MODEL_DIR)
            else:
                logger.info("Downloading and saving GPT-2...")
                self._tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
                self._model = GPT2LMHeadModel.from_pretrained("gpt2")

                # Add pad token if missing
                if self._tokenizer.pad_token is None:
                    self._tokenizer.pad_token = self._tokenizer.eos_token
                
                os.makedirs(config.LOCAL_MODEL_DIR, exist_ok=True)
                self._tokenizer.save_pretrained(config.LOCAL_MODEL_DIR)
                self._model.save_pretrained(config.LOCAL_MODEL_DIR)
            
            # Force model to use CPU
            self._model.to("cpu")
            logger.info("LLM initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def generate_text(self, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
        """Generate text using PyTorch GPT-2."""
        try:
            inputs = self._tokenizer(prompt, return_tensors="pt", max_length=512, truncation=True)
            with torch.no_grad():
                outputs = self._model.generate(
                    inputs["input_ids"],
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self._tokenizer.pad_token_id,
                    eos_token_id=self._tokenizer.eos_token_id
                )

            full_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Remove prompt from beginning
            generated_only = full_text[len(prompt):].strip()
            return generated_only if generated_only else full_text
            
        except Exception as e:
            logger.error(f"Error in text generation: {str(e)}")
            return f"Generated content for: {prompt[:50]}..."
