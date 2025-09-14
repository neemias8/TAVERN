import os
import sys
from pathlib import Path
import torch
from transformers import (
    LEDForConditionalGeneration,
    LEDTokenizer,
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)


def ensure_cache(base: Path) -> None:
    os.environ.setdefault('HF_HOME', str(base))
    os.environ.setdefault('TRANSFORMERS_CACHE', str(base / 'transformers'))
    os.environ.setdefault('HF_DATASETS_CACHE', str(base / 'datasets'))
    base.mkdir(parents=True, exist_ok=True)
    (base / 'transformers').mkdir(parents=True, exist_ok=True)
    (base / 'datasets').mkdir(parents=True, exist_ok=True)


def prefetch_bart(model_id: str = 'sshleifer/distilbart-cnn-12-6') -> None:
    device = 0 if torch.cuda.is_available() else -1
    # Use pipeline to ensure both model and tokenizer are cached
    _ = pipeline('summarization', model=model_id, device=device)


def prefetch_led(model_id: str) -> None:
    _ = LEDForConditionalGeneration.from_pretrained(model_id)
    _ = LEDTokenizer.from_pretrained(model_id)


def main():
    base = Path(os.environ.get('HF_HOME') or 'hf_cache').resolve()
    print(f'Using cache dir: {base}')
    ensure_cache(base)
    try:
        print('Prefetching BART (distilbart-cnn-12-6)...')
        prefetch_bart('sshleifer/distilbart-cnn-12-6')
        print('OK: BART cached')
    except Exception as e:
        print(f'WARN: Failed to prefetch BART: {e}')
    # PRIMERA + LED base
    for mid in ['allenai/PRIMERA', 'allenai/led-base-16384']:
        try:
            print(f'Prefetching {mid} ...')
            prefetch_led(mid)
            print(f'OK: {mid} cached')
        except Exception as e:
            print(f'WARN: Failed to prefetch {mid}: {e}')


if __name__ == '__main__':
    main()
