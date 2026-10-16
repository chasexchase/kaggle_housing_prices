from omegaconf import OmegaConf

import pandas as pd

import importlib

from sklearn.model_selection import KFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import VotingRegressor, StackingRegressor

from utils import save_submission


def import_class(path: str):
    """'sklearn.linear_model.Ridge' -> класс Ridge."""
    module_name, class_name = path.rsplit('.', 1)
    return getattr(importlib.import_module(module_name), class_name)


def get_model(model_cfg):
    """Создаёт модель по записи конфига; при scale=True добавляет RobustScaler
    (медиана и IQR — устойчив к длинным хвостам площадей)."""
    model_class = import_class(model_cfg.name)
    model = model_class(**OmegaConf.to_container(model_cfg.params, resolve=True))

    if model_cfg.scale:
        return make_pipeline(RobustScaler(), model)
    return model


def get_ensemble(config, name: str):
    """Собирает voting (усреднение прогнозов) или stacking ансамбль из моделей конфига."""
    ens = config.ml.ensembles[name]
    members = [(m, get_model(config.ml.models[m])) for m in ens.members]
    if ens.method == 'stacking':
        return StackingRegressor(members, final_estimator=import_class(ens.final)(), cv=config.split.n_splits)
    return VotingRegressor(members)


class MLTrainer:
    """Прогоняет модели и ансамбли через одну схему KFold и обучает лучшую."""

    def __init__(self, config) -> None:
        self.config = config
        self.cv = KFold(n_splits=config.split.n_splits, shuffle=True,
                        random_state=config.general.seed)
        self.best_name = None
        self.best_model = None

    def build(self, name: str):
        """Модель или ансамбль по имени из конфига."""
        if name in self.config.ml.ensembles:
            return get_ensemble(self.config, name)
        return get_model(self.config.ml.models[name])

    def evaluate(self, X: pd.DataFrame, y) -> pd.DataFrame:
        """Кросс-валидация моделей из ml.run_all.
        RMSE по log1p(SalePrice) = метрика соревнования. Чем меньше, тем лучше — лучшая сверху."""
        results = []

        for name in self.config.ml.run_all:
            model = self.build(name)
            res = cross_validate(model, X, y, cv=self.cv, scoring='neg_root_mean_squared_error',
                                 return_train_score=True, n_jobs=-1)
            results.append({
                'model': name,
                'train': -res['train_score'].mean(),   # sklearn отдаёт RMSE со знаком минус
                'cv_mean': -res['test_score'].mean(),
                'cv_std': res['test_score'].std(),
            })
            print(f"{name}: train {results[-1]['train']:.4f} | "
                  f"CV {results[-1]['cv_mean']:.4f} ± {results[-1]['cv_std']:.4f}")

        table = pd.DataFrame(results).sort_values('cv_mean', ascending=True).reset_index(drop=True)
        print('\n' + table.to_string(index=False, float_format='%.4f'))

        self.best_name = table.loc[0, 'model']
        return table

    def fit_best(self, X: pd.DataFrame, y) -> None:
        """Обучает лучшую по CV модель на всём train."""
        self.best_model = self.build(self.best_name)
        self.best_model.fit(X, y)

    def save_submission(self, X_test: pd.DataFrame, ids: pd.Series) -> None:
        """Сохраняет предсказания лучшей модели в submissions/submission_<model>.csv."""
        save_submission(self.config, self.best_name, ids, self.best_model.predict(X_test))
