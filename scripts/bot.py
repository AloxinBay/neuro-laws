# -*- coding: utf-8 -*-
"""
MoonNeuro Lite — поиск по законам GTA 5 RP БЕЗ нейросети.

Лёгкий, мгновенный, оффлайн. Работает на любом ПК (нужен только Python).
Понимает:
  - номер статьи:           12.8   |   ст 12.8   |   статья 12.8
  - конкретную часть:       12.8 ч1   |   12.8 часть 2
  - подробный режим:        подробно 12.8
  - поиск по словам:        убийство   |   превышение скорости
  - сленг:                  госсник, крайм, тулиться

Запуск:
    python scripts/bot.py
    python scripts/bot.py --laws data/laws.json
"""
import argparse
import json
import os
import re
import sys

# переиспользуем форматирование из build_dataset.py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_dataset import (
    concise_answer, detailed_answer, part_answer, SLANG,
)

BANNER = r"""
   MoonNeuro — памятка по законам San-Andreas (лёгкий режим, без ИИ)

   Примеры запросов:
     12.8            — кратко по статье
     12.8 ч1         — конкретная часть
     подробно 12.8   — полный текст
     убийство        — поиск по словам
     госсник         — сленг
   Выход: "выход" или Ctrl+C.
"""

RE_PART = re.compile(r'(\d+(?:\.\d+)*)\s*(?:ч\.?|часть)\s*(\d+)', re.IGNORECASE)
RE_ARTNUM = re.compile(r'\b(\d+(?:\.\d+){0,3})\b')
STOP = {'статья', 'ст', 'что', 'за', 'это', 'какая', 'грозит', 'наказание',
        'про', 'расскажи', 'часть', 'дай', 'покажи'}


def load_index(path):
    laws = json.load(open(path, encoding='utf-8'))
    by_num = {}          # "12.8" -> [(law_name, article), ...]
    all_articles = []    # [(law_name, article), ...]
    for law in laws:
        for art in law['articles']:
            all_articles.append((law['law'], art))
            by_num.setdefault(art['number'], []).append((law['law'], art))
    return by_num, all_articles


def find_part(article, pnum):
    for p in article['parts']:
        if p['num'] == str(pnum):
            return p
    return None


def keyword_search(query, all_articles, limit=3):
    words = [w for w in re.findall(r'[\w\-]+', query.lower())
             if len(w) > 2 and w not in STOP]
    if not words:
        return []
    scored = []
    for law_name, art in all_articles:
        haystack = (art['title'] + ' ' +
                    ' '.join(p['text'] for p in art['parts'])).lower()
        score = 0
        for w in words:
            stem = w[:5]                 # грубый стемминг по префиксу
            if stem in haystack:
                score += 1
        if score:
            scored.append((score, law_name, art))
    scored.sort(key=lambda x: (-x[0], len(x[2]['title'])))
    return scored[:limit]


def answer(query, by_num, all_articles):
    q = query.strip()
    ql = q.lower()

    # 1) сленг
    for term, meaning in SLANG.items():
        if term in ql:
            return f"{term.capitalize()} — это {meaning}."

    # 2) конкретная часть: "12.8 ч1"
    m = RE_PART.search(q)
    if m:
        num, pnum = m.group(1), m.group(2)
        if num in by_num:
            law_name, art = by_num[num][0]
            part = find_part(art, pnum)
            if part:
                return part_answer(art, part)
            return f"У статьи {num} нет части {pnum}.\n" + concise_answer(art)

    # 3) номер статьи (кратко / подробно)
    detailed = any(w in ql for w in ('подробно', 'полностью', 'детально'))
    m = RE_ARTNUM.search(q)
    if m:
        num = m.group(1)
        if num in by_num:
            outs = []
            for law_name, art in by_num[num]:
                body = detailed_answer(art) if detailed else concise_answer(art)
                if len(by_num[num]) > 1:
                    body = f"[{law_name}]\n{body}"
                outs.append(body)
            return '\n\n'.join(outs)

    # 4) поиск по ключевым словам
    hits = keyword_search(q, all_articles)
    if hits:
        parts = []
        for score, law_name, art in hits:
            parts.append(concise_answer(art))
        head = 'Нашёл по запросу:' if len(parts) > 1 else ''
        return (head + '\n\n' if head else '') + '\n\n'.join(parts)

    return ('Не нашёл. Попробуй номер статьи (например 12.8), '
            'часть (12.8 ч1) или ключевое слово (убийство).')


def main():
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

    ap = argparse.ArgumentParser()
    ap.add_argument('--laws', default='data/laws.json')
    args = ap.parse_args()

    if not os.path.exists(args.laws):
        print(f'Нет файла {args.laws}. Сначала выполни:\n'
              f'  python scripts/parse_laws.py --laws-dir laws --out data/laws.json')
        return

    by_num, all_articles = load_index(args.laws)
    print(BANNER)
    print(f'Загружено статей: {len(all_articles)}. Спрашивай!\n')

    while True:
        try:
            q = input('Ты > ').strip()
        except (EOFError, KeyboardInterrupt):
            print('\nПока!')
            break
        if q.lower() in ('выход', 'exit', 'quit', ''):
            print('Пока!')
            break
        print(f'\nMoonNeuro > {answer(q, by_num, all_articles)}\n')


if __name__ == '__main__':
    main()
