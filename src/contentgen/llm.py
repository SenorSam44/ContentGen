# src/contentgen/llm.py
import os
from typing import Dict, Any

import torch
import logging
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login, snapshot_download

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMManager:
    """Simplified and memory-efficient Gemma 3-1B LLM wrapper."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._setup()
        return cls._instance

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------
    def _setup(self):
        self.model_id = "google/gemma-3-1b-it"
        self.local_dir = "models/gemma-3-1b-it"
        self.token = os.getenv("HUGGINGFACE_HUB_TOKEN", "")
        self._authenticate_huggingface()
        self._ensure_model_files()
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.device = self._get_model_device()

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------
    def _authenticate_huggingface(self):
        if not self.token:
            logger.warning("HUGGINGFACE_HUB_TOKEN not found. Running without authentication.")
            return
        try:
            login(token=self.token)
            logger.info("Authenticated with Hugging Face.")
        except Exception as e:
            logger.warning(f"Failed to login to Hugging Face: {e}")

    def _ensure_model_files(self):
        if os.path.exists(self.local_dir):
            logger.info(f"Using local model at {self.local_dir}")
            return
        try:
            snapshot_download(repo_id=self.model_id, local_dir=self.local_dir, token=self.token)
            logger.info("Downloaded model snapshot.")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise

    def _load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.local_dir, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    def _load_model(self):
        """Load model with minimal memory footprint."""
        kwargs = self._build_load_config()
        try:
            model = AutoModelForCausalLM.from_pretrained(self.local_dir, **kwargs)
            logger.info("Model loaded successfully.")
            return model
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _get_model_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    from typing import Dict, Any

    def _build_load_config(self) -> Dict[str, Any]:
        """
        Build efficient model load configuration for transformers.
        Uses bitsandbytes quantization if CUDA + USE_BNB=1.
        """
        cuda: bool = torch.cuda.is_available()
        use_bnb: bool = os.getenv("USE_BNB", "1") == "1"
        load_4bit: bool = os.getenv("LOAD_4BIT", "0") == "1"
        offload_dir: str | None = os.getenv("OFFLOAD_DIR")

        load_cfg: Dict[str, Any] = {"low_cpu_mem_usage": True}

        # Try bitsandbytes quantization if available
        if cuda and use_bnb:
            try:
                from transformers import BitsAndBytesConfig

                quant_cfg: BitsAndBytesConfig
                if load_4bit:
                    quant_cfg = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    quant_cfg = BitsAndBytesConfig(load_in_8bit=True)

                load_cfg["quantization_config"] = quant_cfg
                load_cfg["device_map"] = "auto"
                if offload_dir:
                    load_cfg["offload_folder"] = offload_dir

                logger.info("Using bitsandbytes quantized model.")
                return load_cfg

            except ImportError:
                logger.warning("bitsandbytes not installed. Falling back to normal load.")
            except Exception as e:
                logger.warning(f"Quantization setup failed: {e}. Falling back to standard load.")

        # Fallback configuration
        if cuda:
            load_cfg.update({
                "torch_dtype": torch.float16,
                "device_map": "auto",
            })
        else:
            load_cfg["device_map"] = None
            logger.info("Using CPU-only low-memory load.")

        return load_cfg

    # ------------------------------------------------------------------
    # Context and generation
    # ------------------------------------------------------------------
    def _business_context(self, profile: dict) -> str:
        name = profile.get("business_name", "The brand")
        industry = profile.get("industry", "e-commerce")
        tone = profile.get("tone", "friendly and promotional")
        audience = profile.get("audience", "general consumers")
        products = ", ".join(p["name"] for p in profile.get("products", []) if p.get("name"))
        focus = profile.get("seasonal_focus", "ongoing promotions")
        return (
            f"{name} operates in the {industry} sector. "
            f"The brand tone is {tone}, targeting {audience}. "
            f"Key products include {products}. "
            f"Current campaign focus: {focus}."
        )

    def _resolve_token_limit(self, default: int = 150) -> int:
        """Resolve MAX_NEW_TOKENS from environment."""
        env_val = os.getenv("MAX_NEW_TOKENS")
        if not env_val:
            return default
        try:
            return int(env_val)
        except ValueError:
            return default

    def generate_text(
        self,
        prompt: str,
        business_profile: dict = None,
        max_new_tokens: int = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> str:
        """Generate structured marketing text efficiently."""
        max_new_tokens = max_new_tokens or self._resolve_token_limit()
        context = self._business_context(business_profile or {})

        full_prompt = (
            "You are a professional e-commerce content strategist.\n\n"
            f"Context:\n{context}\n\n"
            f"Task:\n{prompt.strip()}\n\n"
            "Ensure the output sounds authentic, on-brand, and ready for publication."
        )

        inputs = self.tokenizer(
            full_prompt, return_tensors="pt", padding=True, truncation=True, max_length=1024
        )
        if "attention_mask" not in inputs:
            inputs["attention_mask"] = torch.ones_like(inputs["input_ids"])

        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            try:
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                )
            except RuntimeError:
                return "OOM: Lower MAX_NEW_TOKENS or enable quantization."

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(full_prompt):].strip() if text.startswith(full_prompt) else text.strip()
