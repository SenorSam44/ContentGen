import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, snapshot_download
from src.contentgen.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()


class LLMManager:
    _instance = None
    _tokenizer = None
    _model = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize_model()
        return cls._instance

    # ------------------------------------------------------------------
    # Model initialization (Gemma-3-1B-IT)
    # ------------------------------------------------------------------
    def _initialize_model(self):
        try:
            token = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
            if not token:
                logger.warning("No Hugging Face token found. Continuing without authentication.")

            login(token=token)

            model_id = "google/gemma-3-1b-it"
            local_dir = "models/gemma-3-1b-it"

            # Check and download if not available
            if not os.path.exists(local_dir):
                logger.info(f"Model not found locally. Downloading {model_id}...")
                snapshot_download(repo_id=model_id, local_dir=local_dir, token=token)
            else:
                logger.info(f"Found local model at {local_dir}")

            # Load tokenizer and model
            logger.info("Loading tokenizer and model...")
            self._tokenizer = AutoTokenizer.from_pretrained(local_dir)
            self._model = AutoModelForCausalLM.from_pretrained(
                local_dir,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            logger.info("Gemma model loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            raise

    # ------------------------------------------------------------------
    # Structured business context
    # ------------------------------------------------------------------
    def _format_business_context(self, business_profile: dict) -> str:
        name = business_profile.get("business_name", "The brand")
        industry = business_profile.get("industry", "e-commerce")
        tone = business_profile.get("tone", "friendly and promotional")
        audience = business_profile.get("audience", "general consumers")
        products = business_profile.get("products", [])
        seasonal_focus = business_profile.get("seasonal_focus", "ongoing promotions")

        product_list = ", ".join([p.get("name", "") for p in products if p.get("name")])
        return (
            f"{name} operates in the {industry} sector. "
            f"The brand tone is {tone}, targeting {audience}. "
            f"Key products include {product_list}. "
            f"Current campaign focus: {seasonal_focus}."
        )

    # ------------------------------------------------------------------
    # Text generation
    # ------------------------------------------------------------------
    def generate_text(
        self,
        prompt: str,
        business_profile: dict = None,
        max_new_tokens: int = 350,
        temperature: float = 0.7
    ) -> str:
        try:
            context = self._format_business_context(business_profile or {})
            full_prompt = (
                f"You are a professional e-commerce content strategist.\n\n"
                f"Context:\n{context}\n\n"
                f"Task:\n{prompt.strip()}\n\n"
                f"Ensure the output sounds authentic, on-brand, and ready for publication."
            )

            inputs = self._tokenizer(full_prompt, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.to("cuda") for k, v in inputs.items()}

            with torch.no_grad():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True
                )

            text = self._tokenizer.decode(output[0], skip_special_tokens=True)
            return text.strip()

        except Exception as e:
            logger.error(f"Error in text generation: {e}")
            return "Error generating content."
