# -*- coding: utf-8 -*-
"""
Парсер законов GTA 5 RP (проекты Memphis / Orlando) в структурированный JSON.

Понимает форматы:
  Статья 12.8 ★★★★ [Федеральная/Региональная] Незаконный оборот оружия...
    ч. 1 Незаконное приобретение, передача, сбыт...
    Наказание: 4 года лишения свободы или штраф в размере от $10000 до $30000.
  Статья 1. Водитель должен предъявить ...
    Наказание: штраф до $5.000.

Запуск:
    python scripts/parse_laws.py --laws-dir laws --out data/laws.json
"""
import argparse
import json
import os
import re

# ─── Регулярки ────────────────────────────────────────────────────────────────
# Заголовок статьи: "Статья 12.8.1 ★★★ [теги] Название"
RE_ARTICLE = re.compile(r'^\s*Статья\s+(\d+(?:\.\d+)*)\.?\s*(.*)$', re.IGNORECASE)
# Часть: "ч. 1 ★★★ текст"  /  "ч.3 текст"
RE_PART = re.compile(r'^\s*ч\.?\s*(\d+)\b\.?\s*(.*)$', re.IGNORECASE)
# Наказание
RE_PUNISH = re.compile(r'^\s*Наказание:?\s*(.*)$', re.IGNORECASE)
# Подпункты а) б) в) / a) b)
RE_SUBPOINT = re.compile(r'^\s*[а-яёa-z]\)\s+', re.IGNORECASE)
# Главы / разделы — пропускаем, но запоминаем как контекст
RE_CHAPTER = re.compile(r'^\s*(ГЛАВА|РАЗДЕЛ|ОБЩАЯ ЧАСТЬ|ОСОБЕННАЯ ЧАСТЬ)\b', re.IGNORECASE)
# Звёзды розыска
RE_STARS = re.compile(r'★+')
# Теги в квадратных скобках
RE_TAGS = re.compile(r'\[([^\]]+)\]')

ZW = '\u200b'  # zero-width space, встречается в концах строк


def clean(s: str) -> str:
    """Чистим строку от мусорных символов."""
    return s.replace(ZW, '').replace('\u00a0', ' ').strip()


def extract_stars(text: str):
    """Возвращает (кол-во звёзд, текст без звёзд)."""
    m = RE_STARS.search(text)
    if not m:
        return 0, text
    stars = len(max(RE_STARS.findall(text), key=len))
    text = RE_STARS.sub('', text).strip()
    return stars, text


def extract_tags(text: str):
    """Возвращает (список тегов, текст без тегов)."""
    tags = RE_TAGS.findall(text)
    text = RE_TAGS.sub('', text).strip()
    return [t.strip() for t in tags], text


def parse_file(path: str, server: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        raw_lines = f.readlines()

    law_name = os.path.splitext(os.path.basename(path))[0]
    articles = []
    cur_article = None
    cur_part = None      # текущая часть (dict) или None
    cur_chapter = ''

    def push_text(target_key, line):
        """Дописываем текст в текущую часть или в статью."""
        if cur_part is not None:
            cur_part['text'] = (cur_part['text'] + ' ' + line).strip()
        elif cur_article is not None:
            cur_article['title'] = (cur_article['title'] + ' ' + line).strip()

    for raw in raw_lines:
        line = clean(raw)
        if not line:
            continue

        # Глава/раздел
        if RE_CHAPTER.match(line):
            cur_chapter = line
            continue

        # Новая статья
        m = RE_ARTICLE.match(line)
        if m:
            number = m.group(1)
            rest = m.group(2)
            stars, rest = extract_stars(rest)
            tags, rest = extract_tags(rest)
            cur_article = {
                'number': number,
                'title': clean(rest),
                'stars': stars,
                'tags': tags,
                'chapter': cur_chapter,
                'parts': [],
                'punishment': '',  # наказание уровня статьи (когда нет частей)
            }
            articles.append(cur_article)
            cur_part = None
            continue

        # Часть статьи
        m = RE_PART.match(line)
        if m and cur_article is not None:
            pnum = m.group(1)
            ptext = m.group(2)
            pstars, ptext = extract_stars(ptext)
            cur_part = {
                'num': pnum,
                'stars': pstars or cur_article['stars'],
                'text': clean(ptext),
                'punishment': '',
            }
            cur_article['parts'].append(cur_part)
            continue

        # Наказание
        m = RE_PUNISH.match(line)
        if m and cur_article is not None:
            pun = clean(m.group(1))
            if cur_part is not None:
                cur_part['punishment'] = pun
            else:
                cur_article['punishment'] = pun
            continue

        # Прочий текст (подпункты, продолжения) — дописываем
        if cur_article is not None:
            push_text(None, line)

    return {
        'law': law_name,
        'server': server,
        'source_file': os.path.relpath(path).replace('\\', '/'),
        'articles': articles,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--laws-dir', default='laws')
    ap.add_argument('--out', default='data/laws.json')
    args = ap.parse_args()

    laws = []
    for root, _dirs, files in os.walk(args.laws_dir):
        server = os.path.basename(root)
        for fn in sorted(files):
            if fn.lower().endswith('.txt'):
                full = os.path.join(root, fn)
                parsed = parse_file(full, server)
                laws.append(parsed)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        json.dump(laws, f, ensure_ascii=False, indent=2)

    n_articles = sum(len(l['articles']) for l in laws)
    n_parts = sum(len(a['parts']) for l in laws for a in l['articles'])
    print(f'Готово: {len(laws)} файлов, {n_articles} статей, {n_parts} частей -> {args.out}')


if __name__ == '__main__':
    main()
