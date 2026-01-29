"""Run the LaTeX hallucination detection experiment using real model.

This script:
- Loads prompts from `data/prompts/latex_prompts_3_xhard.csv`.
- Loads `openai/gpt-oss-20b` via `transformers` (configurable).
- Uses `hallucination_tools.latex.compute_hallucination_frequencies` to compute and save hallucination frequencies.
- Loads the saved frequencies and runs `interp_tools.linear_probe.run_probes_from_hallucination_rows` to train probes.

Notes:
- This script expects a CPU/GPU environment where `transformers` and the model weights are available.
- If you don't want to actually run the model, you can inspect or run the earlier demo scripts which use dummy models.
"""
import os
from typing import List

import torch

from hallucination_tools import data_loader
from hallucination_tools import latex as ht
from interp_tools import linear_probe

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
except Exception:
    AutoTokenizer = None
    AutoModelForCausalLM = None


class TokenizerWrapper:
    """Wrap a huggingface tokenizer to provide `apply_chat_template` used by the probing code."""
    def __init__(self, tok):
        self.tok = tok

    def apply_chat_template(self, messages, add_generation_prompt=False, return_tensors='pt'):
        # Flatten messages into a single string
        if isinstance(messages, list):
            text = '\n'.join([m.get('content', '') for m in messages])
        elif isinstance(messages, dict):
            text = messages.get('content', '')
        else:
            text = str(messages)

        enc = self.tok(text, return_tensors='pt')
        return {'input_ids': enc['input_ids'], 'attention_mask': enc.get('attention_mask')}

    def decode(self, tokens, skip_special_tokens=True):
        return self.tok.decode(tokens, skip_special_tokens=skip_special_tokens)


def adapt_model_for_hooks(model):
    """Ensure `model.model.layers` points to an iterable of layer modules used by the probing hooks.

    This tries several common attribute names used by HF model classes.
    """
    # If model already has model.layers, assume compatible
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        return model
    # GPT-2 style
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        model.model = type('M', (), {})()
        model.model.layers = model.transformer.h
        return model
    # NeoX style
    if hasattr(model, 'gpt_neox') and hasattr(model.gpt_neox, 'blocks'):
        model.model = type('M', (), {})()
        model.model.layers = model.gpt_neox.blocks
        return model
    # Try to find any attribute that is a list of modules named 'layers' or 'blocks'
    for attr in ['layers', 'blocks', 'h']:
        if hasattr(model, attr):
            model.model = type('M', (), {})()
            model.model.layers = getattr(model, attr)
            return model
    # Fallback: attach an empty list to avoid crashes (probes will find no activations)
    model.model = type('M', (), {})()
    model.model.layers = []
    return model


def build_generate_fn(model, tokenizer_wrapper):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    def generate_fn(prompt: str, settings: dict) -> str:
        # Compose prompt with optional system prompt
        system = settings.get('system_prompt', '') or ''
        full = (system + '\n' + prompt).strip()
        inp = tokenizer_wrapper.apply_chat_template([{'role': 'user', 'content': full}], return_tensors='pt')
        input_ids = inp['input_ids'].to(device)

        gen_kwargs = dict(
            max_new_tokens=int(settings.get('max_new_tokens', 128)),
            temperature=float(settings.get('temperature', 0.0)),
        )
        num_beams = int(settings.get('num_beams', 1) or 1)
        if num_beams > 1:
            gen_kwargs['num_beams'] = num_beams

        with torch.no_grad():
            outputs = model.generate(input_ids, **gen_kwargs)
        # decode the whole sequence
        return tokenizer_wrapper.decode(outputs[0])

    return generate_fn


# Configuration: fixed experiment parameters (no CLI)
# Uses `openai/gpt-oss-20b` unconditionally
MODEL_NAME = 'openai/gpt-oss-20b'
N_SAMPLES = 1
RECOMPUTE = False
SAVE_ACTIVATIONS = False
SAVE_REGRESSION = False
# Comma-separated layer indices or empty to use first 6
LAYERS = ''


def main():
    # Ensure prompts file exists
    prompts_filename = 'latex_prompts_3_xhard.csv'
    prompts_path = os.path.join('data', 'prompts', prompts_filename)
    if not os.path.exists(prompts_path):
        raise FileNotFoundError(f"Prompts file not found: {prompts_path}")

    if AutoTokenizer is None or AutoModelForCausalLM is None:
        raise RuntimeError('transformers is required to run the real experiment; install requirements.txt')

    print('Loading model', MODEL_NAME)
    # LLM setup exactly like in the notebook: provide `model` and `tokenizer` variables
    hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
    # Wrap tokenizer only for generate_fn and probing code which expects apply_chat_template
    tokenizer = TokenizerWrapper(hf_tokenizer)

    # adapt model so interp_tools can register hooks
    model = adapt_model_for_hooks(model)

    generate_fn = build_generate_fn(model, tokenizer)

    # Step 1: compute hallucination frequencies for the selected prompt file
    print('Computing hallucination frequencies (this may be slow)')
    ht.compute_hallucination_frequencies(prompts_filename, generate_fn, n_samples=N_SAMPLES, recompute=RECOMPUTE)

    # Step 2: load hallucination rows and run probes
    rows = data_loader.load_hallucination_frequencies('latex')
    if not rows:
        print('No hallucination rows found; aborting')
        return

    probing_layers = [int(x) for x in LAYERS.split(',')] if LAYERS else list(range(0, min(6, len(model.model.layers))))
    print('Running linear probes on layers:', probing_layers)
    linear_probe.run_probes_from_hallucination_rows(rows=rows, model=model, tokenizer=tokenizer, probing_layers=probing_layers, save_activations=SAVE_ACTIVATIONS, save_regression=SAVE_REGRESSION, recompute=RECOMPUTE)

    print('Experiment complete. Results in data/hallucination_frequencies and data/linear_probe_results')


if __name__ == '__main__':
    main()
