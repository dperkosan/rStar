# Licensed under the MIT license.

import torch
from transformers import (
    GenerationConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def load_HF_model(ckpt) -> tuple:
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    # tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        ckpt,
        load_in_8bit=False,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    return tokenizer, model


def generate_with_HF_model(
    tokenizer, model, input=None, temperature=0.8, top_p=0.95, top_k=40, num_beams=1, max_new_tokens=128, **kwargs
):
    try:
        print(f"🧠 Generating: {input[:80]}...")

        inputs = tokenizer(input, return_tensors="pt", padding=True, truncation=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        generation_config = GenerationConfig(
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            num_beams=num_beams,
            **kwargs,
        )
        with torch.no_grad():
            generation_output = model.generate(
                **inputs,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_scores=True,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        s = generation_output.sequences[0]
        output = tokenizer.decode(s)
    except Exception as e:
        print(f"⚠️ HuggingFace generation failed: {e}")
        return "[HF Error]"
    return output
