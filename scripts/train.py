# -*- coding: utf-8 -*-
"""
Локальное обучение MoonNeuro на своём ПК (GPU NVIDIA или CPU).

Полное дообучение Qwen2.5-0.5B на законах. Без bitsandbytes/trl — только
transformers + datasets, поэтому легко ставится на Windows.

Запуск:
    python scripts/train.py
    python scripts/train.py --epochs 2 --batch 4
    python scripts/train.py --model Qwen/Qwen2.5-0.5B-Instruct --out outputs/merged

После обучения модель -> outputs/merged. Чат: python scripts/inference.py --model outputs/merged
"""
import argparse
import os
import subprocess
import sys

DEF_MODEL = 'Qwen/Qwen2.5-0.5B-Instruct'


def ensure_data():
    """Если нет data/laws.json или data/train.jsonl — собираем их."""
    here = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists('data/laws.json'):
        print('Парсю законы...')
        subprocess.run([sys.executable, os.path.join(here, 'parse_laws.py'),
                        '--laws-dir', 'laws', '--out', 'data/laws.json'], check=True)
    if not os.path.exists('data/train.jsonl'):
        print('Собираю датасет...')
        subprocess.run([sys.executable, os.path.join(here, 'build_dataset.py'),
                        '--in', 'data/laws.json', '--out', 'data/train.jsonl'], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default=DEF_MODEL)
    ap.add_argument('--data', default='data/train.jsonl')
    ap.add_argument('--out', default='outputs/merged')
    ap.add_argument('--epochs', type=float, default=3)
    ap.add_argument('--batch', type=int, default=0, help='0 = авто')
    ap.add_argument('--seq', type=int, default=1024)
    ap.add_argument('--lr', type=float, default=1e-5)
    args = ap.parse_args()

    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    ensure_data()

    import torch
    from datasets import load_dataset
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              TrainingArguments, Trainer,
                              DataCollatorForLanguageModeling)

    cuda = torch.cuda.is_available()
    device = 'cuda' if cuda else 'cpu'
    if args.batch == 0:
        args.batch = 8 if cuda else 1
    print(f'Устройство: {device.upper()} | модель: {args.model} | '
          f'batch={args.batch} epochs={args.epochs}')
    if not cuda:
        print('ВНИМАНИЕ: обучение на CPU идёт медленно (часы). '
              'Для скорости поставь PyTorch с CUDA или уменьши --epochs.')

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype = torch.bfloat16 if (cuda and torch.cuda.is_bf16_supported()) else \
            (torch.float16 if cuda else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    # данные -> текст по chat-шаблону -> токены
    raw = load_dataset('json', data_files=args.data, split='train')
    print('Примеров:', len(raw))

    def tokenize(ex):
        text = tok.apply_chat_template(ex['messages'], tokenize=False,
                                       add_generation_prompt=False)
        out = tok(text, truncation=True, max_length=args.seq)
        return out

    ds = raw.map(tokenize, remove_columns=raw.column_names)
    collator = DataCollatorForLanguageModeling(tok, mlm=False)

    targs = TrainingArguments(
        output_dir='outputs/ckpt',
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        lr_scheduler_type='cosine',
        warmup_ratio=0.03,
        logging_steps=25,
        save_strategy='no',
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        report_to='none',
        dataloader_num_workers=0,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=collator)
    trainer.train()

    model.config.use_cache = True
    os.makedirs(args.out, exist_ok=True)
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(f'\nГотово! Модель сохранена в {args.out}')
    print(f'Запусти чат:  python scripts/inference.py --model {args.out}')


if __name__ == '__main__':
    main()
