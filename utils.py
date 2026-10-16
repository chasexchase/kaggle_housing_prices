import numpy as np
import pandas as pd

import random
from pathlib import Path


def set_seed(seed: int) -> None:

    random.seed(seed)
    np.random.seed(seed)


def check_dataset(X: pd.DataFrame, y, X_test: pd.DataFrame) -> None:
    """Проверяет данные перед обучением: нет NaN, все колонки числовые,
    колонки train и test совпадают, таргет конечный и похож на log-цену."""

    for name, df in [('X', X), ('X_test', X_test)]:
        nan_cols = df.columns[df.isna().any()].tolist()
        assert not nan_cols, f'NaN в {name}: {nan_cols}'

        non_numeric = df.select_dtypes(exclude='number').columns.tolist()
        assert not non_numeric, f'Нечисловые колонки в {name}: {non_numeric}'

    assert list(X.columns) == list(X_test.columns), 'Колонки X и X_test не совпадают'
    assert len(X) == len(y), f'Размеры X ({len(X)}) и y ({len(y)}) не совпадают'
    assert np.isfinite(y).all(), 'В таргете NaN или inf'

    print(f'Данные в порядке: X {X.shape}, X_test {X_test.shape}, y: {np.min(y):.2f}…{np.max(y):.2f}')


def save_submission(config, name: str, ids: pd.Series, preds) -> None:
    """Сохраняет предсказания модели name в submissions/submission_<name>.csv.
    Модель учится на log1p(SalePrice) — перед записью возвращаем доллары."""
    out_dir = Path(config.paths.submission_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds = np.asarray(preds, dtype=float).reshape(-1)
    if config.general.target_transform == 'log1p':
        preds = np.expm1(preds)
    pd.DataFrame({ids.name: ids, config.general.target: preds}).to_csv(out_dir / f'submission_{name}.csv', index=False)
    print(f'Сабмит сохранён: {name}')
