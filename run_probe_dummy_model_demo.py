import torch
import numpy as np
from hallucination_tools import data_loader
from interp_tools import linear_probe

# Dummy tokenizer and model that produce consistent random activations
class DummyTokenizer:
    def apply_chat_template(self, messages, add_generation_prompt=False, return_tensors='pt'):
        # return a dict with input_ids tensor
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
    def __init__(self, n_layers=3, hidden_size=16):
        self.device = torch.device('cpu')
        # replicate structure used in code: model.model.layers
        self.model = type('M', (), {})()
        self.model.layers = [DummyLayer() for _ in range(n_layers)]
        self.hidden_size = hidden_size

    def __call__(self, **kwargs):
        # Simulate forward pass: for each layer, call registered hooks with an output tensor
        # output expected shape: (batch, seq_len, hidden_size)
        seq_len = kwargs.get('input_ids').shape[1]
        batch = kwargs.get('input_ids').shape[0]
        for layer in self.model.layers:
            out = torch.randn(batch, seq_len, self.hidden_size)
            # call each hook with (module, input, output)
            for h in list(getattr(layer, '_hooks', [])):
                try:
                    h(layer, None, out)
                except Exception:
                    pass
        # return dummy
        return None


def main():
    prompts_rows = data_loader.load_prompts('python_prompts_1.csv')
    # take first 6 prompts
    prompts_rows = prompts_rows[:6]
    # add index and hallucination_rate alternating 0/1
    rows = []
    for i, r in enumerate(prompts_rows, start=1):
        row = dict(r)
        row['index'] = str(i)
        row['hallucination_rate'] = str(1.0 if (i % 2 == 0) else 0.0)
        rows.append(row)

    model = DummyModel(n_layers=2, hidden_size=32)
    tokenizer = DummyTokenizer()

    linear_probe.run_probes_from_hallucination_rows(
        rows=rows,
        model=model,
        tokenizer=tokenizer,
        probing_layers=[0, 1],
        threshold=0.5,
        save_activations=True,
        save_regression=True,
        recompute=True,
    )

    # list results
    import os
    base = os.path.join('data', 'linear_probe_results')
    for root, dirs, files in os.walk(base):
        print(root)
        for f in files:
            print('  ', f)

if __name__ == '__main__':
    main()
