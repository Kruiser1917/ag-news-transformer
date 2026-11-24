# Attention и Transformers: классификация новостей AG News (`PyTorch`)

Практическая работа по **Модулю 06 (Deep Learning), Тема 07 «Attention и Transformers»**: классификация новостей на 4 темы (World, Sports, Business, Sci/Tech) с помощью Transformer Encoder. Механизм attention сначала собирается вручную, затем Transformer сравнивается с baseline (masked mean embeddings, GRU).

## Данные и подготовка

- Источник: **AG News Classification Dataset** (Kaggle `amananandrai/ag-news-classification-dataset`, `train.csv` 120 000 / `test.csv` 7 600). Файлы скачиваются из публичной копии оригинального `ag_news_csv` (Zhang et al., 2015), запасной вариант — зеркало Hugging Face `sh0416/ag_news`. Ноутбук сам создаёт `data/train.csv` и `data/test.csv`, если их нет.
- Текст = `Title + Description`; восстановлены HTML-сущности (`#39;`, `quot;`, `&amp;lt;`), удалены `\` и теги.
- **До сплита** удалены 103 дубликата, 27 текстов с конфликтующими метками и 6 текстов, совпадающих с test.
- Стратифицированный сплит официального train (`seed = 42`): **train 107 864 / val 12 000**; официальный **test 7 600** использовался один раз в конце.
- Токенизация: lowercase + regex; словарь только по train: `<pad>=0`, `<unk>=1`, `<cls>=2`, `min_freq = 3`, `max_size = 30 000`; доля `<unk>` — 1.3 % (train), 1.7 % (val/test).
- `MAX_LEN = 80` (p99 = 71, обрезано ≈ 0.5 %), padding mask `True = PAD`, бакетирование по длине, ширина батча кратна 16 (меньше перекомпиляций на `mps`).

## Attention вручную

- Своя `scaled_dot_product_attention` с маской до softmax совпадает с `F.scaled_dot_product_attention` и `nn.MultiheadAttention` (`allclose`, разница ~2e-7); веса на PAD ровно 0, строки суммируются в 1.
- Без деления на √d_k при d_k = 512 softmax почти one-hot: энтропия 0.14 против 2.36 с масштабированием, средний максимальный вес 0.95 против 0.25.
- Causal mask даёт нижнетреугольную матрицу весов; без PE слой энкодера перестановочно эквивариантен, с PE — нет.

## Модели

Во всех моделях `Embedding(30 000, 128, padding_idx=0)`, `AdamW(lr=1e-3, wd=0.01)`, warmup 5 % + косинусное затухание, `CrossEntropyLoss`, `clip_grad_norm_(1.0)`, до 6 эпох, Early Stopping по val loss (`patience = 2`).

- `MeanEmb` — masked mean embeddings → `Linear`.
- `GRU` — GRU 128, состояние после последнего реального токена.
- `Transformer-mean` / `Transformer-CLS` — синусоидальный PE + 2 × `TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, norm_first=True)` + финальный `LayerNorm`, `src_key_padding_mask`; pooling — masked mean или токен `<cls>`. Pre-LN выбран за более стабильное обучение (residual-путь без нормализации).

## Сравнение моделей

| Модель | Обучаемых параметров | Эпох | Время, с | с/эпоху | Val Loss | Val Acc | Val macro F1 | Test Acc* | Test macro F1* | Выбрана по Val |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `MeanEmb` | `3 840 516` | 6 | 52.5 | 8.5 | 0.2733 | 0.9132 | 0.9131 | 0.9120 | 0.9119 |  |
| `GRU` | `3 939 588` | 5 | 238.9 | 47.2 | 0.2675 | 0.9095 | 0.9094 | 0.9087 | 0.9086 |  |
| `Transformer-mean` | `4 105 732` | 6 | 113.0 | 18.4 | **0.2623** | **0.9161** | **0.9159** | **0.9132** | **0.9129** | да |
| `Transformer-CLS` | `4 105 732` | 6 | 119.1 | 19.4 | 0.2633 | 0.9157 | 0.9156 | 0.9097 | 0.9097 |  |

\* Test для невыбранных моделей — только для справки, выбор сделан по validation до открытия test.

## Итог на test (Transformer-mean)

- **Accuracy 0.9132, macro F1 0.9129**; F1 по классам: World 0.920, Sports 0.967, Business 0.880, Sci/Tech 0.885.
- Чаще всего путаются **Business ↔ Sci/Tech** (163 + 125 ошибок), затем World → Business (71), Sci/Tech → World (64). Часть «уверенных» ошибок — шум разметки датасета.
- Padding mask: с маской logits не зависят от количества PAD (Δ ≈ 5e-7), без маски меняются на 2.5–2.8.
- Веса attention получены ручным вызовом `layer.self_attn(norm1(x), ..., need_weights=True, average_attn_weights=False)`; собранный вручную проход совпадает с `layer(...)` (~1e-6), веса на PAD = 0.
- Чекпоинт `checkpoints/best_model.pt` (16 МБ: веса, словарь, `max_len`) загружается с `weights_only=True`, logits совпадают точно.

## Выводы

Transformer дал лучшее качество, но всего на ~0.3 п.п. выше простого mean-embedding baseline и в ~2 раза медленнее на эпоху: темы новостей в основном определяются ключевыми словами, а контекст помогает лишь в неоднозначных случаях. Почти все параметры (94–100 %) — это `Embedding`; энкодер добавляет ~0.26 M. GRU на `mps` самая медленная и быстрее всех переобучается. Обучение на `mps` недетерминировано (разброс между запусками ~0.1–0.3 п.п.), поэтому разница между mean- и CLS-pooling незначима.

## Структура

- `ag_news_transformer.ipynb` — ноутбук с выводами (весь прогон ≈ 9 мин на `mps`).
- `results_summary.csv` — таблица сравнения.
- `checkpoints/best_model.pt` — лучшая модель.
- `requirements.txt` — зависимости.

## Запуск

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace ag_news_transformer.ipynb
```
