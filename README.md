# House Prices: ML-пайплайн

Соревнование Kaggle [House Prices – Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

**Задача:** регрессия — предсказать `SalePrice` дома по 79 признакам (Ames, Iowa).
**Метрика:** RMSE между логарифмами предсказанной и реальной цены.
**Лучший результат:** _TODO_

---

## Результаты

_TODO: таблица моделей (Train / CV / CV std), график CV, лидерборд._

| # | Модель | Тип | Train RMSE | CV RMSE | CV std |
|---|--------|-----|-----------:|--------:|-------:|
| — | — | — | — | — | — |

### Лидерборд Kaggle

| Сабмит | Public LB |
|--------|----------:|
| — | — |

---

## Быстрый старт

```bash
git clone https://github.com/chasexchase/kaggle_housing_prices.git
cd kaggle_housing_prices
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Данные (в репозиторий не коммитятся) — скачать в `data/raw/`:

```bash
kaggle competitions download -c house-prices-advanced-regression-techniques -p data/raw
unzip data/raw/house-prices-advanced-regression-techniques.zip -d data/raw
```

Или вручную со [страницы соревнования](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data):
`train.csv`, `test.csv`, `data_description.txt`, `sample_submission.csv`.

---

## Структура репозитория

```
├── config.py             # все параметры эксперимента
├── eda.ipynb             # EDA и визуализации
├── EXPERIMENTS.md        # журнал решения: To-do, Notes, Questions, Solution, идеи
├── data/raw/             # train.csv, test.csv, data_description.txt (не в git)
├── submissions/          # сабмиты
├── experiments/          # runs.csv и графики прогонов
├── reports/figures/      # финальные графики для README
├── drafts/               # черновики
├── telegram_credits.example.json  # шаблон токенов Telegram-бота
└── requirements.txt
```

---

## Пайплайн

### 1. EDA ([`eda.ipynb`](eda.ipynb))
_TODO_

### 2. Признаки
_TODO_

### 3. Валидация
_TODO_

### 4. Модели
_TODO_

### 5. Логирование и воспроизводимость
_TODO_

---

## Что сработало и что нет

**Сработало**
- _TODO_

**Нейтрально**
- _TODO_

**Не сработало**
- _TODO_

---

## Ограничения и планы

- [ ] _TODO_

---

## Вопросы к ревьюеру

1. _TODO_
