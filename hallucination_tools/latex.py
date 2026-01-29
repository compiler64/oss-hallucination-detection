import csv
import os
import shutil
import subprocess
from typing import List
from datetime import datetime

from . import data_loader

def extract_latex_code(response: str) -> str:
    last_block_end = response.rfind('```')
    if last_block_end == -1:
        return ""
    last_latex_block_start = response.rfind('```latex', 0, last_block_end)
    last_general_block_start = response.rfind('```', 0, last_block_end)
    if last_latex_block_start != -1 and last_latex_block_start >= last_general_block_start:
        start = last_latex_block_start + len('```latex')
        return response[start:last_block_end].strip()
    elif last_general_block_start != -1:
        start = last_general_block_start + len('```')
        return response[start:last_block_end].strip()
    else:
        return ""

def check_latex_for_hallucination(code_str: str) -> bool:
    tectonic_bin = shutil.which('tectonic')
    if tectonic_bin is None:
        return False
    if '\\begin{document}' not in code_str:
        wrapped = """\\documentclass{article}
\\usepackage[utf8]{inputenc}
\\begin{document}
""" + code_str + "\n\\end{document}\n"
    else:
        wrapped = code_str
    tmp_tex = None
    try:
        import tempfile
        fd, tmp_tex = tempfile.mkstemp(suffix='.tex')
        os.close(fd)
        with open(tmp_tex, 'w', encoding='utf-8') as f:
            f.write(wrapped)
        proc = subprocess.run([tectonic_bin, tmp_tex], capture_output=True, text=True)
        if proc.returncode == 0:
            base = os.path.splitext(tmp_tex)[0]
            for ext in ['.pdf', '.log', '.aux', '.tex']:
                try:
                    p = base + ext
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            return False
        else:
            return True
    except Exception:
        return True
    finally:
        try:
            if tmp_tex and os.path.exists(tmp_tex):
                os.remove(tmp_tex)
        except Exception:
            pass

def make_latex_prompts(n: int = 50) -> List[str]:
    templates = [
        'write a minimal LaTeX snippet that {task}.',
        'provide a short LaTeX example that {task}.',
        'implement a tiny LaTeX macro which {task} and show a one-line demo.'
    ]
    tasks = [
        'defines a macro \\boldcaps{} that uppercases and bolds its argument (making sure to show an example)',
        'renders a centered 3x3 matrix with parentheses and highlights the diagonal',
        'creates a minimal TikZ resistor labeled R1',
        'provides a Beamer slide with title, subtitle, and three centered bullets',
        'uses siunitx to typeset 12.3 m/s^2 with correct spacing',
        'defines a theorem environment named mythm and shows one theorem',
        'creates a 2-column multicol layout with a short paragraph and a bullet list',
        'draws a tiny commutative diagram A -> B -> C using tikz-cd',
        'shows an align environment with two aligned equations and number the second',
        'places two side-by-side images with separate captions (no files required)',
        'wraps a small equation inline and references it with a manual label',
        'creates a minipage placing a figure placeholder at left and caption at right',
        'defines a simple counter examplecnt and a command \\ex to use it once (making sure to show an example)',
        'writes a short thebibliography entry and cites it inline',
        'makes a framed note box titled Note with a one-line body',
        'draws a small TikZ timeline with three labeled events',
        'typesets a small table with multirow-like merged cells',
        'shows a captioned figure with a short caption and centered layout',
        'uses subcaption-like layout to show two placeholders side-by-side',
        'creates a small numbered list with custom label formatting',
        'typesets a short matrix and highlights the (1,3) element with color',
        'demonstrates bold italic small caps combination on a short phrase',
        'shows how to set page margins for a simple article',
        'creates a short resume header with name and contact centered',
        'renders a simple calendar month with three event titles',
        'places a small decorative rule and a centered header',
        'demonstrates using \\verb|...| inline and escaping a backslash',
        'writes a tiny LaTeX command to uppercase the first letter of an argument',
        'typesets a short chemical formula using math mode with subscripts',
        'shows a tiny boxed equation with a surrounding frame',
        'provides a short example of using \\texttt for inline code and monospace',
        'creates a short footnote-heavy paragraph demonstrating footnote usage',
        'uses \\parbox or \\fbox to create a small captioned block',
        'renders a short commutative triangle diagram with tikz-cd',
        'provides a minimal package usage example for amsmath with an align',
        'shows how to place a small rotated text using \\rotatebox',
        'typesets a short bibliography entry in plain thebibliography style',
        'draws a small labeled axis with two ticks and numbers',
        'uses \\raisebox to slightly lift a short inline symbol',
        'typesets a short algorithm environment with two steps',
        'renders a short table of contents snippet with one entry',
        'creates a two-column flyer-like layout with short items',
        'shows a tiny example of \\lstlisting-like code block formatting',
        'demonstrates \\includegraphics with a placeholder box for an image',
        'provides a small example of using color to highlight text',
        'creates a short captioned math display with an equation and label',
        'shows a tiny example of using \\centering inside a figure environment',
        'typesets a short signature block with name and date right-aligned',
        'renders an inline fraction and then a displayed fraction example'
    ]
    prompts = []
    i = 0
    while len(prompts) < n:
        tmpl = templates[i % len(templates)]
        task = tasks[i % len(tasks)]
        prompts.append(tmpl.format(task=task))
        i += 1
    return prompts

def save_prompts_csv(prompts: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['prompt'])
        for p in prompts:
            writer.writerow([p])

def load_prompts_csv(path: str) -> List[str]:
    out = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            out.append(row['prompt'])
    return out


def compute_hallucination_frequencies(
    filename: str,
    generate_fn,
    n_samples: int = 1,
    recompute: bool = False,
) -> None:
    """Load prompts from data/prompts/<filename>, generate responses using generate_fn,
    check LaTeX compilation for hallucinations, and save results to data/hallucination_frequencies/latex.csv.

    generate_fn should be a callable: generate_fn(prompt: str, settings: dict) -> str
    """
    language = 'latex'
    rows = data_loader.load_prompts(filename)
    new_results = []
    for r in rows:
        prompt = r.get('prompt', '')
        settings = {
            'max_new_tokens': int(r.get('max_new_tokens', 128) or 128),
            'temperature': float(r.get('temperature', 0.0) or 0.0),
            'reasoning_level': r.get('reasoning_level', ''),
            'num_beams': int(r.get('num_beams', 1) or 1),
            'system_prompt': r.get('system_prompt', ''),
        }

        cached = data_loader.find_cached_for_row(language, {**r, **{k: str(v) for k, v in settings.items()}})
        if cached is not None and not recompute:
            continue

        trials = int(n_samples)
        hits = 0
        for _ in range(trials):
            resp = generate_fn(prompt, settings)
            code = extract_latex_code(resp)
            hallucinated = check_latex_for_hallucination(code)
            if hallucinated:
                hits += 1

        rate = hits / trials if trials > 0 else 0.0
        new_results.append({
            'prompt': prompt,
            'max_new_tokens': str(settings['max_new_tokens']),
            'temperature': str(settings['temperature']),
            'reasoning_level': settings['reasoning_level'],
            'num_beams': str(settings['num_beams']),
            'system_prompt': settings['system_prompt'],
            'trials': str(trials),
            'hits': str(hits),
            'hallucination_rate': str(rate),
            'last_updated': datetime.utcnow().isoformat(),
        })

    if new_results:
        data_loader.save_hallucination_frequencies(language, new_results)
