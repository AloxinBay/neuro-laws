# -*- coding: utf-8 -*-
"""
Генерация обучающего датасета (chat-формат) из data/laws.json.

Создаёт пары "вопрос-ответ":
  - КРАТКИЙ ответ по номеру статьи (основной режим, как просил пользователь):
        12.8  ->  ч1 — незаконный сбыт/перевозка оружия (4 года л/с или штраф $10k–30k)
                  ч2 — незаконное хранение/ношение оружия (4 года л/с или штраф $10k–30k)
  - подробный ответ ("подробно 12.8")
  - наказание по конкретной части ("наказание за 12.8 ч1")
  - обратный поиск по названию ("какая статья за убийство")
  - вопросы и ответы на английском (через переводчик, опционально)

Запуск:
    python scripts/build_dataset.py --in data/laws.json --out data/train.jsonl
    python scripts/build_dataset.py --translate           # + английские пары
"""
import argparse
import json
import os
import random
import re

random.seed(42)

SYSTEM_PROMPT = (
    "Тебя зовут MoonNeuro. Ты — юридический ассистент по законам GTA 5 RP штата San-Andreas. "
    "Отвечай кратко и по делу. ★ — это уровень розыска. "
    "Указывай номер статьи, части, суть нарушения и наказание. "
    "Понимай сленг игроков: госсник — полицейский, крайм — бандит, тулиться — стрелять. "
    "Основной язык — русский, но отвечай на языке вопроса."
)

# Сленг игроков: термин -> объяснение. Список легко расширять.
SLANG = {
    "госсник": "полицейский (сотрудник правоохранительных органов)",
    "тулиться": "стрелять, применять оружие",
    "крайм": "бандит, преступник",
}

# ─── Сжатие наказания ─────────────────────────────────────────────────────────
def compress_money(text: str) -> str:
    """$10000 / 60.000$ -> $10k и т.п."""
    def repl(m):
        digits = re.sub(r'[^\d]', '', m.group('num'))
        if not digits:
            return m.group(0)
        val = int(digits)
        if val >= 1000 and val % 1000 == 0:
            return f'${val // 1000}k'
        return f'${val}'
    # $10000  |  10.000$  |  $30.000
    text = re.sub(r'\$\s?(?P<num>\d[\d.,]*\d|\d)', repl, text)
    text = re.sub(r'(?P<num>\d[\d.,]*\d|\d)\s?\$', repl, text)
    return text


def short_punishment(pun: str) -> str:
    if not pun:
        return ''
    p = pun
    p = re.sub(r'лишени[ея]\s+свободы', 'л/с', p, flags=re.IGNORECASE)
    p = re.sub(r'(уголовн(ый|ого)\s+)?штраф(а)?\s+в\s+размере', 'штраф', p, flags=re.IGNORECASE)
    p = compress_money(p)
    p = re.sub(r'\s+', ' ', p).strip().rstrip('.')
    return p


# ─── Сжатие описания нарушения ────────────────────────────────────────────────
FILLER = [
    r'в автомобиле или ином транспортном средстве',
    r'то есть .*?(?=,|$)',
    r'любых видов',
    r'и иные.*?(?=,|$)',
]

def short_desc(text: str, max_words: int = 11) -> str:
    if not text:
        return ''
    t = text.strip()
    for f in FILLER:
        t = re.sub(f, '', t, flags=re.IGNORECASE)
    t = re.sub(r'\s+', ' ', t).strip().strip(',').strip()
    words = t.split()
    if len(words) > max_words:
        t = ' '.join(words[:max_words]) + '…'
    # первая буква строчная для аккуратного списка
    return (t[:1].lower() + t[1:]) if t else t


def stars_str(n: int) -> str:
    return '★' * n if n else ''


# ─── Формирование ответов ─────────────────────────────────────────────────────
def article_ref(article) -> str:
    s = stars_str(article['stars'])
    head = f"Ст. {article['number']}"
    if s:
        head += f" {s}"
    title = article['title'].rstrip(',. ')
    return f"{head} — {title}" if title else head


def concise_answer(article) -> str:
    """Краткий ответ — основной формат."""
    lines = [article_ref(article)]
    parts = article['parts']
    if parts:
        for p in parts:
            desc = short_desc(p['text'])
            pun = short_punishment(p['punishment'])
            st = stars_str(p['stars']) if p['stars'] != article['stars'] else ''
            chunk = f"ч{p['num']}"
            if st:
                chunk += f" {st}"
            chunk += f" — {desc}"
            if pun:
                chunk += f" ({pun})"
            lines.append(chunk)
    else:
        pun = short_punishment(article['punishment'])
        if pun:
            lines.append(f"Наказание: {pun}")
    return '\n'.join(lines)


def detailed_answer(article) -> str:
    """Подробный ответ — полный текст."""
    lines = [article_ref(article)]
    if article['tags']:
        lines.append('Тип: ' + ', '.join(article['tags']))
    parts = article['parts']
    if parts:
        for p in parts:
            st = stars_str(p['stars'])
            head = f"ч.{p['num']}" + (f" {st}" if st else '')
            lines.append(f"{head} {p['text'].rstrip()}")
            if p['punishment']:
                lines.append(f"  Наказание: {p['punishment'].rstrip()}")
    elif article['punishment']:
        lines.append(f"Наказание: {article['punishment'].rstrip()}")
    return '\n'.join(lines)


def part_answer(article, part) -> str:
    head = f"Ст. {article['number']} ч.{part['num']}"
    st = stars_str(part['stars'])
    if st:
        head += f" {st}"
    desc = short_desc(part['text'], max_words=16)
    pun = short_punishment(part['punishment'])
    out = f"{head} — {desc}"
    if pun:
        out += f"\nНаказание: {pun}"
    return out


# ─── Шаблоны вопросов ─────────────────────────────────────────────────────────
def q_article_ru(num):
    return [
        num,
        f"ст {num}",
        f"ст. {num}",
        f"статья {num}",
        f"что за {num}",
        f"что грозит за {num}",
        f"наказание за {num}",
        f"расскажи про {num}",
        f"{num} что это",
    ]

def q_article_en(num):
    return [
        f"article {num}",
        f"art {num}",
        f"what is {num}",
        f"punishment for {num}",
        f"what is article {num}",
    ]

def q_part_ru(num, pnum):
    return [
        f"{num} ч{pnum}",
        f"{num} ч.{pnum}",
        f"статья {num} часть {pnum}",
        f"наказание за {num} ч{pnum}",
    ]

def q_keyword_ru(title):
    t = title.rstrip(',. ').lower()
    return [
        f"какая статья за {t}",
        f"статья за {t}",
        f"что грозит за {t}",
    ]


def make_example(question, answer):
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
    }


def slang_examples():
    """Обучающие пары, объясняющие сленг игроков."""
    out = []
    for term, meaning in SLANG.items():
        cap = term.capitalize()
        answer = f"{cap} — это {meaning}."
        for q in [
            term,
            f"кто такой {term}",
            f"что такое {term}",
            f"что значит {term}",
            f"{term} это кто",
            f"{term} это что",
            f"объясни слово {term}",
        ]:
            out.append(make_example(q, answer))
    return out


# ─── Перевод (опционально) ────────────────────────────────────────────────────
_translator = None
_cache = {}

def translate_en(text: str) -> str:
    global _translator
    if not text.strip():
        return text
    if text in _cache:
        return _cache[text]
    try:
        if _translator is None:
            from deep_translator import GoogleTranslator
            _translator = GoogleTranslator(source='ru', target='en')
        out = _translator.translate(text[:4900])
        _cache[text] = out
        return out
    except Exception as e:
        print(f"[warn] перевод не удался: {e}")
        return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--in', dest='inp', default='data/laws.json')
    ap.add_argument('--out', default='data/train.jsonl')
    ap.add_argument('--translate', action='store_true',
                    help='Создавать английские пары через онлайн-переводчик')
    args = ap.parse_args()

    laws = json.load(open(args.inp, encoding='utf-8'))
    examples = []

    # сленг игроков
    examples.extend(slang_examples())

    for law in laws:
        for art in law['articles']:
            num = art['number']
            # пропускаем пустые/утратившие силу
            if 'утратил' in art['title'].lower() or 'утратила' in art['title'].lower():
                continue
            has_content = art['parts'] or art['punishment'] or art['title']
            if not has_content:
                continue

            concise = concise_answer(art)
            detailed = detailed_answer(art)

            # краткие ответы по номеру
            for q in q_article_ru(num):
                examples.append(make_example(q, concise))
            # подробные
            examples.append(make_example(f"подробно {num}", detailed))
            examples.append(make_example(f"статья {num} подробно", detailed))

            # по частям
            for p in art['parts']:
                for q in q_part_ru(num, p['num']):
                    examples.append(make_example(q, part_answer(art, p)))

            # обратный поиск по названию (только для коротких понятных заголовков)
            title_words = art['title'].rstrip(',. ').split()
            if 1 < len(title_words) <= 7:
                for q in q_keyword_ru(art['title']):
                    examples.append(make_example(q, concise))

            # ── английские пары ──
            if args.translate:
                concise_en = translate_en(concise)
                for q in q_article_en(num):
                    examples.append(make_example(q, concise_en))

    random.shuffle(examples)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for ex in examples:
            f.write(json.dumps(ex, ensure_ascii=False) + '\n')

    print(f'Готово: {len(examples)} обучающих примеров -> {args.out}')


if __name__ == '__main__':
    main()
