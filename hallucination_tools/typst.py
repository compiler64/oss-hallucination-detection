import csv
import os
import shutil
import subprocess
from typing import List
from datetime import datetime

from . import data_loader

def extract_typst_code(response: str) -> str:
    last_block_end = response.rfind('```')
    if last_block_end == -1:
        return ""
    last_typst_block_start = response.rfind('```typst', 0, last_block_end)
    last_general_block_start = response.rfind('```', 0, last_block_end)
    if last_typst_block_start != -1 and last_typst_block_start >= last_general_block_start:
        start = last_typst_block_start + len('```typst')
        return response[start:last_block_end].strip()
    elif last_general_block_start != -1:
        start = last_general_block_start + len('```')
        return response[start:last_block_end].strip()
    else:
        return ""

def check_typst_for_hallucination(code_str: str) -> bool:
    """Try to compile Typst code with `typst` CLI. Return True on failure (hallucination).

    If `typst` is unavailable, return False (cannot verify).
    """
    typst_bin = shutil.which('typst')
    if typst_bin is None:
        return False
    tmp_typ = None
    tmp_pdf = None
    try:
        import tempfile
        fd, tmp_typ = tempfile.mkstemp(suffix='.typ')
        os.close(fd)
        with open(tmp_typ, 'w', encoding='utf-8') as f:
            f.write(code_str)
        fd_pdf, tmp_pdf = tempfile.mkstemp(suffix='.pdf')
        os.close(fd_pdf)
        proc = subprocess.run([typst_bin, 'compile', tmp_typ, tmp_pdf], capture_output=True, text=True)
        if proc.returncode == 0:
            try:
                os.remove(tmp_pdf)
            except Exception:
                pass
            return False
        else:
            return True
    except Exception:
        return True
    finally:
        try:
            if tmp_typ and os.path.exists(tmp_typ):
                os.remove(tmp_typ)
        except Exception:
            pass

def make_typst_prompts(n: int = 50) -> List[str]:
    templates = [
        'typeset a {topic} using Typst.',
        'create a Typst document that demonstrates {topic}.',
        'implement a Typst macro to {topic} and show an example.',
        'produce a Typst template for {topic}.',
        'write Typst code that draws {topic} using Typst drawing primitives.'
    ]
    topics = [
        'a two-page academic CV with sections for Education, Research, and Publications',
        'a labeled electrical circuit diagram with resistors and capacitors',
        'a bold-and-capitalizing macro for text',
        'a complex table with merged rows and columns showing a weekly schedule',
        'a beamer-like slide with title, subtitle, and centered bullets',
        'a matrix with custom brackets and highlighted diagonal',
        'a timeline with labeled events and date ticks',
        'a bibliography entry style and example reference',
        'two figures side-by-side with independent captions',
        'an inline SVG-like vector drawing using primitives',
        'a resume header with name, contact, and links',
        'a poster layout with title, authors, and columns',
        'a custom enumerated list style with icons',
        'a multi-column magazine layout with images',
        'a recipe card with ingredients and steps',
        'a calendar month view with events',
        'a chemical molecule diagram using simple shapes',
        'a chessboard diagram with pieces placed',
        'a musical staff with notes for a short melody',
        'a poster citation block with DOI and authors',
        'a footnote-heavy article layout demonstrating footnote linking',
        'a form layout with labeled input boxes',
        'a greeting card with centered poem and decorative border',
        'a botanical illustration with labeled parts',
        'a flowchart of a simple algorithm',
        'a mindmap with central node and branches',
        'a set of flashcards with question and answer layout',
        'a table of contents with dotted leaders',
        'a simple invoice template with line items and totals',
        'a book title page with author and publisher',
        'a captioned figure demonstrating caption styling',
        'a poster with grid-aligned images and captions',
        'a resume with skills bar visualization',
        'a list of definitions with hanging indent formatting',
        'a glossary with terms and descriptions',
        'a sports scoreboard layout',
        'a travel itinerary with dates and times',
        'a price list table with currency alignment',
        'a certificate template with decorative frame',
        'a technical specification section with code blocks',
        'a comparison table with highlighted best values',
        'a newsletter header with issue number and date',
        'a meeting agenda with time slots and owners',
        'a research poster with sections and figures',
        'a cookbook layout with multiple recipes per page',
        'a greeting note with drop cap and ornamental initials',
        'a photo gallery grid with captions',
        'a diploma-style layout for a certificate'
    ]
    prompts = []
    i = 0
    while len(prompts) < n:
        tmpl = templates[i % len(templates)]
        topic = topics[i % len(topics)]
        prompts.append(tmpl.format(topic=topic))
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
    check Typst compilation for hallucinations, and save results to data/hallucination_frequencies/typst.csv.

    generate_fn should be a callable: generate_fn(prompt: str, settings: dict) -> str
    """
    language = 'typst'
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
            code = extract_typst_code(resp)
            hallucinated = check_typst_for_hallucination(code)
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
