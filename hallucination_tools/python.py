import csv
import importlib
import os
from typing import List
import ast
from datetime import datetime

from . import data_loader

try:
    from RestrictedPython import compile_restricted
    from RestrictedPython.Guards import safe_builtins
except Exception:
    compile_restricted = None
    safe_builtins = {}

def extract_code(response: str) -> str:
    last_block_end = response.rfind("```")
    if last_block_end == -1:
        return ""
    last_python_block_start = response.rfind("```python", 0, last_block_end)
    last_general_block_start = response.rfind("```", 0, last_block_end)
    if last_python_block_start != -1 and last_python_block_start >= last_general_block_start:
        start = last_python_block_start + len("```python")
        return response[start:last_block_end].strip()
    elif last_general_block_start != -1:
        start = last_general_block_start + len("```")
        return response[start:last_block_end].strip()
    else:
        return ""

def check_code_for_hallucination(code_str: str, library_name: str) -> bool:
    """Return True if hallucination detected (syntax error, NameError, AttributeError).

    If the target library cannot be imported, this function returns False (cannot verify).
    """
    try:
        library_module = importlib.import_module(library_name)
    except ImportError:
        return False
    except Exception:
        return False

    if compile_restricted is None:
        # Fallback: parse AST for syntax errors only
        try:
            import ast
            ast.parse(code_str)
        except SyntaxError:
            return True
        return False

    def restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == library_name:
            return library_module
        raise ImportError(f"Import of module '{name}' not allowed in restricted env")

    safe_builtins_extended = safe_builtins.copy()
    safe_builtins_extended['__import__'] = restricted_import

    try:
        byte_code = compile_restricted(code_str, '<string>', 'exec')
        restricted_globals = {'__builtins__': safe_builtins_extended, library_name: library_module}
        restricted_locals = {}
        exec(byte_code, restricted_globals, restricted_locals)
    except SyntaxError:
        return True
    except NameError:
        return True
    except AttributeError:
        return True
    except Exception:
        return True

    return False

def make_python_prompts(n: int = 50) -> List[str]:
    templates = [
        'write a Python function to {task} using the {library} library.',
        'implement a short Python script that {task} and show example usage.',
        'provide a minimal Python module that {task} with clear function signatures.'
    ]
    tasks = [
        'parse a TOML configuration file',
        'perform real-time character reading from a serial port',
        'generate a unique cryptocurrency name and symbol',
        'compress a folder to a zstandard archive',
        'predict simple linear trend from a CSV file',
        'read and write Excel files',
        'download a file over HTTP and save it',
        'send an email with attachment',
        'create a progress bar for a loop',
        'compute TF-IDF vectors for a list of documents',
        'connect to a serial GPS module and parse NMEA',
        'resize and crop an image',
        'perform sentiment analysis on text',
        'train a small scikit-learn classifier',
        'create and write to a SQLite database',
        'control a USB HID device',
        'generate MIDI events from a list of notes',
        'implement a CLI with click that echoes args',
        'watch a directory for file changes',
        'parse and pretty-print JSON',
        'convert CSV to Excel',
        'download and parse HTML to extract links',
        'implement a decorator that caches results',
        'stream audio from microphone and save to WAV',
        'generate secure random passwords',
        'send an SMS using an API client',
        'scrape a web page and extract headings',
        'implement exponential smoothing forecast',
        'plot a histogram from numeric data',
        'perform OCR on an image',
        'encode and decode base64 files',
        'compute pairwise distances between vectors',
        'merge multiple PDFs into one',
        'generate thumbnails for images in a folder',
        'read a CSV and compute group statistics',
        'send an email via SMTP with TLS',
        'download images concurrently',
        'monitor CPU and memory usage',
        'upload a file to an S3-compatible service',
        'parse command-line arguments and validate them',
        'render Markdown to HTML',
        'validate email addresses with regex',
        'hash a password securely',
        'perform k-means clustering on numeric data',
        'convert audio to spectrogram',
        'read EXIF metadata from images',
        'implement a binary search over a sorted list'
    ]
    libraries = [
        'tomli','pyserial','faker','zstandard','pandas','openpyxl','requests','smtplib','tqdm','scikit-learn',
        'pynmea2','Pillow','nltk','sklearn','sqlite3','hid','mido','click','watchdog','json',
        'csv','beautifulsoup4','statsmodels','matplotlib','pytesseract','base64','scipy','PyPDF2','concurrent.futures','psutil',
        'boto3','argparse','markdown','re','bcrypt','sklearn','librosa','piexif','numpy'
    ]
    prompts = []
    i = 0
    while len(prompts) < n:
        tmpl = templates[i % len(templates)]
        task = tasks[i % len(tasks)]
        lib = libraries[i % len(libraries)]
        prompts.append(tmpl.format(task=task, library=lib))
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
    library_name: str = None,
) -> None:
    """Load prompts from data/prompts/<filename>, generate responses using generate_fn,
    compute hallucination frequencies and save aggregated results to data/hallucination_frequencies/python.csv.

    generate_fn should be a callable: generate_fn(prompt: str, settings: dict) -> str
    """
    language = 'python'
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
            code = extract_code(resp)
            hallucinated = False
            if library_name:
                hallucinated = check_code_for_hallucination(code, library_name)
            else:
                # fallback: syntax check
                if not code.strip():
                    hallucinated = False
                else:
                    try:
                        ast.parse(code)
                        hallucinated = False
                    except SyntaxError:
                        hallucinated = True
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
