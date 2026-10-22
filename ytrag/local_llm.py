"""
Local inference engine for fine-tuned Striver DSA Model on NVIDIA GeForce RTX 5050.

Loads Qwen2.5-Coder-7B-Instruct with fine-tuned LoRA weights in 4-bit NF4 (BFloat16)
to provide grounded, pedagogical answers with pointer dry-runs, verified code,
and lecture timestamp citations.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ADAPTER_DIR = Path(r"C:\Users\2006s\Documents\Striver_DSA_FineTuning\striver_dsa_lora_final")
BASE_MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

logger = logging.getLogger(__name__)


class LocalTeachingAssistant:
    _instance: Optional["LocalTeachingAssistant"] = None

    def __init__(self, adapter_dir: Path = DEFAULT_ADAPTER_DIR):
        self.adapter_dir = Path(adapter_dir)
        self.tokenizer = None
        self.model = None
        self.device = "cpu"
        self._loaded = False

    @classmethod
    def get_instance(cls, adapter_dir: Path = DEFAULT_ADAPTER_DIR) -> "LocalTeachingAssistant":
        if cls._instance is None:
            cls._instance = cls(adapter_dir)
        return cls._instance

    @property
    def is_available(self) -> bool:
        return (self.adapter_dir / "adapter_model.safetensors").exists()

    def load(self):
        if self._loaded:
            return

        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Loading LocalTeachingAssistant on %s in 4-bit NF4 from %s", self.device, self.adapter_dir)

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.adapter_dir) if self.adapter_dir.exists() else BASE_MODEL_NAME,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

        if self.is_available:
            self.model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
            logger.info("Injected fine-tuned Striver DSA LoRA weights.")
        else:
            self.model = base_model
            logger.warning("Fine-tuned weights not found at %s; running base model.", self.adapter_dir)

        self.model.eval()
        self._loaded = True
        logger.info("LocalTeachingAssistant successfully ready on RTX 5050.")

    def generate_answer(
        self,
        query: str,
        lecture_excerpts: list[dict],
        dsa_context: Optional[dict] = None,
        max_new_tokens: int = 768,
    ) -> dict:
        """
        Generates an evidence-grounded Striver-style answer with citations.
        """
        if not self._loaded:
            self.load()

        import torch

        # Build prompt grounding
        lecture_bullets = []
        for i, item in enumerate(lecture_excerpts[:4]):
            sid = i + 1
            t_disp = item.get("timestamp_display", "")
            title = item.get("video_title", "")
            snip = item.get("snippet", "")[:400].replace("\n", " ")
            lecture_bullets.append(f"[{sid}] Video: '{title}' at {t_disp} -> \"{snip}\"")

        context_str = "\n".join(lecture_bullets)
        dsa_info = ""
        if dsa_context:
            title = dsa_context.get("title", "")
            appr = dsa_context.get("approach", "")
            tc = dsa_context.get("time_complexity", "")
            sc = dsa_context.get("space_complexity", "")
            dsa_info = f"\nRelevant Striver Sheet Topic: {title}\nApproach: {appr}\nTime: {tc}, Space: {sc}\n"

        system_prompt = (
            "You are an expert Data Structures and Algorithms mentor trained strictly on Striver's "
            "(Take U Forward) A2Z DSA teaching methodology.\n"
            "Guidelines:\n"
            "1. Ground your explanation in the provided lecture moments and canonical approach.\n"
            "2. Explain the intuition (Brute -> Better -> Optimal) and mention key pointer movements.\n"
            "3. Reference valid source IDs like [1], [2] when discussing specific lecture moments.\n"
            "4. Return strictly a JSON object with this exact schema:\n"
            '{\n'
            '  "approach_summary": "1-2 sentence core intuition",\n'
            '  "bullets": [\n'
            '    {"text": "concise explanation", "sources": [1]}\n'
            '  ]\n'
            '}'
        )

        user_content = (
            f"Student Question: {query}\n"
            f"{dsa_info}\n"
            f"Verified Striver Lecture Moments:\n{context_str}\n\n"
            "Synthesize a clear, grounded answer following Striver's exact teaching style."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.2,
                top_p=0.9,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        raw_text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

        # Parse JSON
        try:
            cleaned = raw_text
            if "```json" in cleaned:
                cleaned = cleaned.split("```json", 1)[1].split("```", 1)[0].strip()
            elif "```" in cleaned:
                cleaned = cleaned.split("```", 1)[1].split("```", 1)[0].strip()

            parsed = json.loads(cleaned)
            if isinstance(parsed, dict) and isinstance(parsed.get("bullets"), list):
                return parsed
        except Exception:
            pass

        # Fallback: Extract natural bullets if model outputs prose
        bullets = []
        for line in raw_text.split("\n"):
            line = line.strip()
            if line.startswith(("-", "*", "•", "1.", "2.", "3.")):
                clean_line = re.sub(r"^[-*•\d.]+\s*", "", line)
                sources = [int(s) for s in re.findall(r"\[(\d+)\]", line) if s.isdigit()]
                if not sources and len(lecture_excerpts) > 0:
                    sources = [1]
                bullets.append({"text": clean_line, "sources": sources})

        if not bullets:
            bullets = [{"text": raw_text[:300], "sources": [1] if lecture_excerpts else []}]

        return {
            "approach_summary": "Striver's Optimal DSA Strategy",
            "bullets": bullets[:3],
        }

    def generate_chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ) -> tuple[str, int, int]:
        """Generates full text completion for OpenAI-compatible chat API."""
        if not self._loaded:
            self.load()

        import torch

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        input_len = inputs.input_ids.shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(0.01, temperature),
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        gen_tokens = outputs[0][input_len:]
        completion_len = len(gen_tokens)
        text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return text, input_len, completion_len

    def stream_chat_completion(
        self,
        messages: list[dict],
        max_tokens: int = 1024,
        temperature: float = 0.2,
        top_p: float = 0.9,
    ):
        """Streams tokens for OpenAI-compatible Server-Sent Events (SSE)."""
        if not self._loaded:
            self.load()

        from threading import Thread
        from transformers import TextIteratorStreamer

        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)
        gen_kwargs = dict(
            **inputs,
            streamer=streamer,
            max_new_tokens=max_tokens,
            temperature=max(0.01, temperature),
            top_p=top_p,
            do_sample=temperature > 0.0,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield chunk
