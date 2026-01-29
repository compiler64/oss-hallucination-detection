import torch
from hallucination_tools import typst as ht
from hallucination_tools import data_loader
from interp_tools import linear_probe

# Dummy tokenizer and model (same pattern as previous demo)
class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, return_tensors='pt'):
        return {'input_ids': torch.randint(0, 100, (1, 5))}

class DummyLayer:
    def __init__(self):
        self._hooks = []
    def register_forward_hook(self, fn):
        self._hooks.append(fn)
        class Handle:
            def __init__(self, hooks, fn):
                self.hooks = hooks
                self.fn = fn
            def remove(self):
                try:
                    self.hooks.remove(self.fn)
                except Exception:
                    pass
        return Handle(self._hooks, fn)

class DummyModel:
    def __init__(self, n_layers=2, hidden_size=16):
        self.device = torch.device('cpu')
        self.model = type('M', (), {})()
        self.model.layers = [DummyLayer() for _ in range(n_layers)]
        self.hidden_size = hidden_size

    def __call__(self, **kwargs):
        seq_len = kwargs.get('input_ids').shape[1]
        batch = kwargs.get('input_ids').shape[0]
        for layer in self.model.layers:
            out = torch.randn(batch, seq_len, self.hidden_size)
            for h in list(getattr(layer, '_hooks', [])):
                try:
                    h(layer, None, out)
                except Exception:
                    pass
        return None

# Dummy generator returns a typst code block (checker will be no-op if typst not installed)
def dummy_generate(prompt, settings):
    return '```typst\n# Dummy typst code for: "' + prompt.replace('"', "'") + '\n\n# end\n```'


def main():
    # Compute hallucination frequencies for typst prompts
    ht.compute_hallucination_frequencies('typst_prompts_1.csv', generate_fn=dummy_generate, n_samples=2, recompute=True)

    # Load computed hallucination rows
    rows = data_loader.load_hallucination_frequencies('typst')
    if not rows:
        print('No hallucination rows produced')
        return

    # Run probes using dummy model/tokenizer
    model = DummyModel(n_layers=2, hidden_size=24)
    tokenizer = DummyTokenizer()
    linear_probe.run_probes_from_hallucination_rows(rows=rows, model=model, tokenizer=tokenizer, probing_layers=[0,1], save_activations=True, save_regression=True, recompute=True)

    # List outputs
    import os
    for root, dirs, files in os.walk('data/linear_probe_results'):
        print(root)
        for f in files:
            print('  ', f)

if __name__ == '__main__':
    main()
