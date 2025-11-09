import json
import os
import re

import torch
import logging
from dotenv import load_dotenv
from typing import Any

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig
)
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain


from huggingface_hub import login, snapshot_download

# LangChain integration
from langchain_huggingface import HuggingFacePipeline

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LLMManager:
    """LangChain-adapted Gemma 3-1B LLM wrapper."""

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
        self.llm = self._build_langchain_pipeline()  # <— integrated here

        torch.set_grad_enabled(False)

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
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

        # Limit model precision and disable gradients globally
        if torch.cuda.is_available():
            model.to(torch.bfloat16)  # lower precision to reduce VRAM
        else:
            model.to(torch.float32)

        return model

    def _get_model_device(self):
        try:
            return next(self.model.parameters()).device
        except Exception:
            return torch.device("cpu")

    # ------------------------------------------------------------------
    # Config helpers
    # ------------------------------------------------------------------
    def _build_load_config(self) -> dict[str, Any]:
        """
        Build efficient model load configuration for transformers.
        Uses bitsandbytes quantization if CUDA + USE_BNB=1.
        """
        cuda: bool = torch.cuda.is_available()
        use_bnb: bool = os.getenv("USE_BNB", "1") == "1"
        load_4bit: bool = os.getenv("LOAD_4BIT", "0") == "1"
        offload_dir: str | None = os.getenv("OFFLOAD_DIR")

        load_cfg: dict[str, Any] = {"low_cpu_mem_usage": True}

        # Try bitsandbytes quantization if available
        if cuda and use_bnb:
            try:

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
            logger.info("Using CUDA load.")
        else:
            load_cfg["device_map"] = None
            logger.info("Using CPU-only low-memory load.")

        return load_cfg

    # ------------------------------------------------------------------
    # LangChain Integration
    # ------------------------------------------------------------------
    def _build_langchain_pipeline(self):
        """Wrap model in a LangChain-compatible pipeline."""
        gen_pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0 if torch.cuda.is_available() else -1,
            max_new_tokens=self._resolve_token_limit(),
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )
        return HuggingFacePipeline(pipeline=gen_pipe)

    # ------------------------------------------------------------------
    # Context + generation
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

    def _resolve_token_limit(self, default: int = 1500) -> int:
        """Resolve MAX_NEW_TOKENS from environment."""
        env_val = os.getenv("MAX_NEW_TOKENS")
        if not env_val:
            return default
        try:
            return int(env_val)
        except ValueError:
            return default

    # ------------------------------------------------------------------
    # Summaries
    # ------------------------------------------------------------------
    def generate_summaries(self, topics: list[str], business_profile: dict = None, n: int = 5) -> list[dict]:
        """Generate N short marketing summaries with LangChain but restricted context and precision."""
        context = self._business_context(business_profile or {})
        topics_str = "\n".join(topics)

        # # Truncate overly long context to save memory
        # if len(context) > 800:
        #     context = context[:800] + "..."

        print("generate_summaries...")
        # Cache the template to avoid re-creation overhead
        if not hasattr(self, "_summary_template"):
            self._summary_template = PromptTemplate(
                input_variables=["context", "topics", "n"],
                template=(
                    "You are a professional e-commerce marketing strategist.\n"
                    "Business context:\n{context}\n\n"
                    "Generate exactly {n} short and catchy campaign summaries "
                    "for these topics:\n{topics}\n\n"
                    "Each summary must be concise (under 40 words), emotionally engaging, "
                    "and relevant to the brand tone.\n\n"
                    "Return a single JSON array of objects, each with 'topic' and 'summary' fields.\n"
                    "Example:\n"
                    '[{{\"topic\": \"Example Topic\", \"summary\": \"A short summary.\"}}]'
                ),
            )

        # Build or reuse LLMChain with the existing pipeline
        if not hasattr(self, "_summary_chain"):
            self._summary_chain = self._summary_template | self.llm

        # Run inference through LangChain
        response = self._summary_chain.invoke({
            "context": context,
            "topics": topics_str,
            "n": n
        })

        prompt_str = self._summary_template.format(context=context, topics=topics_str, n=n)
        text = response["text"] if isinstance(response, dict) else response
        text = text[len(prompt_str)+9:-5]

        print(f"{text=}")

        # Extract the first valid JSON array
        items = json.loads(text)
        print(f"{items=}")

        return items

    # ------------------------------------------------------------------
    # Platform-specific post generation
    # ------------------------------------------------------------------
    def generate_posts(
        self,
        summaries: list[str],
        platform: str,
        business_profile: dict = None,
    ) -> list[dict]:
        """Generate detailed social media posts for a given platform."""
        context = self._business_context(business_profile or {})
        template = PromptTemplate(
            input_variables=["context", "platform", "summary"],
            template=(
                "You are a content creator crafting {platform} posts for a brand.\n\n"
                "Brand context:\n{context}\n\n"
                "Post requirement:\n{summary}\n\n"
                "Write one complete post suitable for {platform}. "
                "Ensure it fits the platform’s tone and format, includes an engaging hook, "
                "and ends with a relevant call to action."
            ),
        )
        chain = LLMChain(llm=self.llm, prompt=template)

        posts = []
        for s in summaries:
            result = chain.invoke({"context": context, "platform": platform, "summary": s})
            posts.append({
                "summary": s,
                "platform": platform,
                "content": result["text"].strip()
            })
        return posts
