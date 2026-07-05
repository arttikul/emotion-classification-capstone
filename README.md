# Класифікація емоцій — капстон з Deep Learning

Класифікатор емоцій тексту на 6 класів, побудований як повний цикл розробки:
EDA → класичний baseline → модель глибокого навчання з нуля → fine-tuned
трансформер → порівняння та аналіз.

## Задача та датасет

**Задача:** класифікувати коротке англомовне речення в один з шести
класів емоцій: `sadness` (сум), `joy` (радість), `love` (любов),
`anger` (гнів), `fear` (страх), `surprise` (здивування).

**Датасет:** [dair-ai/emotion](https://huggingface.co/datasets/dair-ai/emotion)
(unsplit-версія), ~417 000 розмічених речень із твітів. Дві колонки:
`text` (сире речення) і `label` (ціле число 0–5). Розподіл класів
незбалансований — `joy` і `sadness` переважають, `surprise` і `love`
трапляються рідко — це враховано через стратифікований поділ і
зважену функцію втрат (class-weighted loss).

Файл `emotion-dataset.csv` треба покласти в `data/` перед запуском (не
закомічений у git через розмір — див. `.gitignore`).

## Підхід

Три моделі навчаються й порівнюються на одному й тому ж стратифікованому
поділі train/val/test (70/15/15), щоб порівняння було коректним:

| # | Модель | Що демонструє |
|---|-------|----------------------|
| 1 | **TF-IDF + Logistic Regression** | Класичний ML-baseline / перевірка на адекватність |
| 2 | **BiLSTM** (PyTorch, навчання з нуля) | Основи deep learning: ембединги, рекурентне моделювання послідовностей, регуляризація |
| 3 | **Fine-tuned DistilBERT** (Hugging Face `transformers`) | Transfer learning на основі попередньо навченого трансформера |

Кожна модель оцінюється за accuracy і **macro-F1** (саме macro, а не micro,
через дисбаланс класів — модель, яка добре вгадує лише `joy`/`sadness`,
не повинна виглядати штучно сильною).

## Структура репозиторію

```
emotion-classification-capstone/
├── README.md
├── requirements.txt
├── build_notebook.py            # генерує ноутбук нижче
├── data/
│   └── emotion-dataset.csv      # покласти датасет сюди (не в git)
├── notebook/
│   └── emotion_classification.ipynb   # головний артефакт — запускати цей файл
├── src/
│   ├── data.py                  # завантаження, очищення, стратифікований поділ
│   ├── vocab.py                 # проста токенізація + словник (для BiLSTM)
│   ├── models.py                # архітектура BiLSTM (PyTorch)
│   ├── train_baseline.py        # TF-IDF + LogisticRegression
│   ├── train_lstm.py            # цикл навчання BiLSTM
│   ├── train_transformer.py     # fine-tuning DistilBERT (HF Trainer)
│   └── viz.py                   # графіки confusion matrix / кривих навчання
└── artifacts/                   # збережені моделі + метрики (генеруються під час запуску)
```

## Як запустити (Google Colab, GPU)

1. Відкрий репозиторій `https://github.com/arttikul/emotion-classification-capstone` та `notebook/emotion_classification.ipynb` у Colab.
2. Runtime → Change runtime type → **GPU** (T4 достатньо).
4. Запусти всі клітинки зверху вниз. Ноутбук:
   - завантажує й досліджує дані (EDA),
   - навчає й оцінює TF-IDF baseline,
   - навчає й оцінює BiLSTM,
   - робить fine-tuning й оцінює DistilBERT,
   - будує підсумкову таблицю й графік порівняння моделей,
   - робить inference на кількох нових прикладах речень.

Для швидкого smoke-тесту перед повним запуском постав `SAMPLE_FRAC`
(на початку ноутбука) в маленьке значення, напр. `0.05`.

### Запуск локально замість Colab

```bash
pip install -r requirements.txt
python src/train_baseline.py --csv_path data/emotion-dataset.csv
python src/train_lstm.py --csv_path data/emotion-dataset.csv --epochs 6
python src/train_transformer.py --csv_path data/emotion-dataset.csv --epochs 2
```
Для останніх двох кроків настійно рекомендується GPU; fine-tuning DistilBERT
на повному датасеті лише на CPU практично неможливий.

## Результати

Повний запуск на Google Colab (GPU), повний датасет, тестова вибірка:

| Модель | Accuracy | Macro-F1 |
|---|---|---|
| TF-IDF + Logistic Regression | 0.9423 | 0.9179 |
| BiLSTM (з нуля) | 0.9604 | 0.9414 |
| DistilBERT (fine-tuned) | **0.9670** | **0.9480** |

Кожен крок ускладнення моделі покращує обидві метрики — очікуваний
результат: transfer learning (DistilBERT) > deep learning з нуля (BiLSTM) >
класичний ML (TF-IDF + LogReg).

### Аналіз confusion matrix

Дві помилки повторюються **у всіх трьох моделях**, що свідчить про
справжню неоднозначність у даних/розмітці, а не про слабкість моделі:

- **`fear` → `surprise`**: найстійкіша плутанина (7% у baseline, 5% у
  BiLSTM, 5% у DistilBERT). Речення на кшталт "I did not see that coming
  and I'm scared" правдоподібно перебувають десь між цими двома класами.
- **`joy` → `love`**: стабільна витік ~4% у всіх трьох моделях — багато
  речень про `love` ("I love spending time with my family") важко
  відрізнити від захоплених речень про `joy` лише за лексикою.

Де глибокі моделі явно виграють: recall для **`love`** зростає з
0.95 (baseline) → 0.99 (BiLSTM) → 1.00 (DistilBERT) — попередньо навчені
контекстні репрезентації розв'язують цей клас майже ідеально, тоді як
TF-IDF все ще плутає 3% з `joy`.

Неочевидний результат: recall DistilBERT для **`fear` (0.89) насправді
нижчий за BiLSTM (0.93)**, попри перемогу DistilBERT загалом. DistilBERT
додає новий витік 3% `fear`→`sadness`, якого немає в BiLSTM, поверх
спільного витоку `fear`→`surprise`. Тобто DistilBERT виграє сукупно, але не
є рівномірно кращим на кожному класі — варто явно проговорити це на
захисті як нюанс "найкраща модель не завжди найкраща в усьому".

## Що було перевірено перед здачею

Кожен модуль був написаний і протестований end-to-end на підвибірці даних
у CPU-only sandbox (без GPU, з обмеженим доступом до Hugging Face Hub):

- `src/data.py` — завантаження, очищення, стратифікований поділ:
  перевірено на повному CSV із 417 тис. рядків.
- `src/vocab.py` — побудова словника й кодування: юніт-тест пройдено ізольовано.
- `src/train_baseline.py` — повний прогін на 5% стратифікованій вибірці:
  **92% accuracy / 0.89 macro-F1**.
- `src/models.py`, `src/train_lstm.py` — архітектура й цикл навчання
  перевірені на синтаксис; сам PyTorch не вдалося встановити в sandbox
  (CPU-збірка недоступна там — лише CUDA-збірка, яка не влазить у дискову
  квоту sandbox), тому forward/training pass треба було запускати на Colab.
- `src/train_transformer.py` — перевірено на синтаксис; завантаження
  токенізатора/моделі з `huggingface.co` заблоковане мережевою політикою
  sandbox, тож це також вимагало Colab (де є повний доступ до інтернету)
  для першого реального запуску.
- `src/viz.py` — усі три функції побудови графіків успішно відпрацювали на тестових даних.

Коротко: усе, що можна було перевірити без GPU чи доступу до Hub, було
перевірено в sandbox, а потім увесь ноутбук був прогнаний end-to-end на
Google Colab (GPU) — результати вище й аналіз confusion matrix отримані
саме з цього запуску.

## Відповідність критеріям оцінювання

Проєкт побудований під критерії з опису курсу: власний датасет
(`dair-ai/emotion`, обраний і завантажений самостійно), дослідження задачі
(класифікація, з прогресією через 3 родини моделей) і повний цикл розробки
(EDA → baseline → deep learning → transfer learning → оцінка → порівняння),
оформлений як repo, готовий для GitHub, з цим README.
