# -*- coding: utf-8 -*-
"""
MoonNeuro — консольный чат (cmd) с обученной моделью по законам GTA 5 RP.

Запуск:
    python scripts/inference.py --model outputs/merged
    python scripts/inference.py --model Qwen/Qwen2.5-3B-Instruct --adapter outputs/lora

Нужны: transformers, torch, peft (если используется --adapter).
"""
import argparse
import sys

SYSTEM = (
    "Тебя зовут MoonNeuro. Ты — юридический ассистент по законам GTA 5 RP штата San-Andreas. "
    "Отвечай кратко и по делу. ★ — это уровень розыска. "
    "Указывай номер статьи, части, суть нарушения и наказание. "
    "Понимай сленг игроков: госсник — полицейский, крайм — бандит, тулиться — стрелять. "
    "Основной язык — русский, но отвечай на языке вопроса."
)

BANNER = r"""
  __  __                   _   _
 |  \/  | ___   ___  _ __ | \ | | ___ _   _ _ __ ___
 | |\/| |/ _ \ / _ \| '_ \|  \| |/ _ \ | | | '__/ _ \
 | |  | | (_) | (_) | | | | |\  |  __/ |_| | | | (_) |
 |_|  |_|\___/ \___/|_| |_|_| \_|\___|\__,_|_|  \___/

   MoonNeuro — нейро-памятка по законам San-Andreas
   Спрашивай: номер статьи (12.8), "наказание за 12.8 ч2", сленг (госсник).
   Выход: напиши "выход" или нажми Ctrl+C.
"""


def main():
    # корректный вывод кириллицы в cmd
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='outputs/merged', help='Путь к модели или имя на HF')
    ap.add_argument('--adapter', default=None, help='Путь к LoRA-адаптеру (опц.)')
    ap.add_argument('--max-new-tokens', type=int, default=256)
    args = ap.parse_args()

    print(BANNER)
    print('Загружаю модель, подожди...\n')

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map='auto' if torch.cuda.is_available() else None,
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)

    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    device = 'GPU' if torch.cuda.is_available() else 'CPU'
    print(f'Модель загружена ({device}). Можно спрашивать!\n')

    while True:
        try:
            q = input('Ты > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nПока!')
            break
        if q.lower() in ('выход', 'exit', 'quit', ''):
            print('Пока!')
            break

        msgs = [{'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': q}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors='pt').to(model.device)
        out = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        text = tok.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        print(f'\nMoonNeuro > {text.strip()}\n')


if __name__ == '__main__':
    main()
