from __future__ import annotations

import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class LocalGenerator:
    def __init__(self, model_name: str | None = None, device_map: str = "auto") -> None:
        self.model_name = model_name or os.getenv("RAGOPS_LLM_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=dtype,
            device_map=device_map,
        )

    def generate(self, query: str, contexts: list[dict], max_new_tokens: int = 260) -> dict:
        context_text = "\n\n".join(
            f"[{item['id']}] {item['title']}\n{item['text']}" for item in contexts
        )
        system = (
            "You are a retrieval-grounded assistant. Answer only from the supplied context. "
            "Every factual sentence in a normal answer must end with at least one source citation "
            "using the exact source ID format [doc_01]. Use only source IDs provided in the context. "
            "If the context is insufficient to answer the question, explicitly say that the available "
            "evidence is insufficient and do not invent facts or citations. Keep the answer concise."
        )
        user = f"Question:\n{query}\n\nContext:\n{context_text}"
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        generated = outputs[:, inputs.input_ids.shape[1] :]
        answer = self.tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        return {"answer": answer, "model": self.model_name}
