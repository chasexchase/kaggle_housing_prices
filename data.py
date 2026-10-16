import numpy as np
import pandas as pd
from omegaconf import OmegaConf


def remove_outliers(train, pp):
    """Удаляет из train дома с GrLivArea > 4000 и ценой < 300k (уникальные продажи - выброс)."""
    o = pp.outliers
    mask = (train['GrLivArea'] > o.GrLivArea_min) & (train['SalePrice'] < o.SalePrice_max)
    return train.loc[~mask].reset_index(drop=True)


def preprocess(train, test, cfg):

    pp, target = cfg.preprocessing, cfg.general.target

    train = remove_outliers(train, pp)
    y = train[target]
    if cfg.general.target_transform == 'log1p':
        y = np.log1p(y)

    X = pd.concat([train.drop(columns=target), test], keys=['train', 'test'])
    X = X.drop(columns=list(pp.drop_columns))
    for c in pp.as_category:
        X[c] = X[c].astype(str)
    tr = X.loc['train']

    # 1. пропуски
    X[list(pp.fill_none_cat)] = X[list(pp.fill_none_cat)].fillna('None')
    X[list(pp.fill_zero_num)] = X[list(pp.fill_zero_num)].fillna(0)
    for c in pp.fill_mode:
        X[c] = X[c].fillna(tr[c].mode()[0])
    lf_median = tr.groupby('Neighborhood')['LotFrontage'].median()
    X['LotFrontage'] = X['LotFrontage'].fillna(X['Neighborhood'].map(lf_median))
    X['GarageYrBlt'] = X['GarageYrBlt'].fillna(X['YearBuilt'])

    # 2. порядковые шкалы
    maps = OmegaConf.to_container(pp.ordinal_maps)
    for col, scale in pp.ordinal.items():
        X[col] = X[col].map(maps[scale])

    # 3. one-hot (train + test)
    X = pd.get_dummies(X, columns=list(pp.one_hot), dtype=int)

    assert X.isna().sum().sum() == 0, f'NaN: {X.columns[X.isna().any()].tolist()}'
    assert X.select_dtypes('object').empty, 'остались текстовые колонки'
    return X.loc['train'], X.loc['test'], y.values
