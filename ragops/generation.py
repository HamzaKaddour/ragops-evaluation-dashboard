from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalGenerator:
    def __init__(self, model_name: str | None = None, device_map: str = "auto") -> None:
        self.model_name = model_name or os.getenv(
            "RAGOPS_LLM_MODEL",
            "Qwen/Qwen2.5-1.5B-Instruct",
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
        )

    def generate(
        self,
        query: str,
        contexts: list[dict],
        max_new_tokens: int = 220,
    ) -> dict:
        context_text = "\n\n".join(
            f"[{item['id']}] {item['title']}\n{item['text']}" for item in contexts
        )

        system = (
            "You are a retrieval-grounded assistant. Retrieval confidence has already "
            "been checked before you are called, so answer the question using only the "
            "supplied context. Write one to three concise factual sentences. Omit any "
            "claim that is not supported by the context. Source citations may be included "
            "using the exact retrieved IDs, but a deterministic citation layer will also "
            "verify and attach citations after generation. Do not add a references section. "
            "Only abstain if the supplied context truly contains no information that can "
            "answer the question, using exactly: "
            "'The available evidence is insufficient to answer this question.'" 
        )
        user = (
            f"Question:\n{query}\n\n"
            f"Retrieved context:\n{context_text}\n\n"
            "Return only the grounded answer. Do not add a references section."
        )

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        answer = self.tokenizer.batch_decode(
            generated,
            skip_special_tokens=True,
        )[0].strip()
        return {"answer": answer, "model": self.model_name}
