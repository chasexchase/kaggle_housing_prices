import json
import logging
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import telebot
import tg_logger
from omegaconf import OmegaConf

matplotlib.use('Agg')


def _read_tokens(config) -> dict | None:
    """Токены бота из telegram_credentials; None, если бот не привязан."""
    try:
        with open(config.paths.telegram_credentials, 'r') as file:
            tokens = json.load(file)
        if not tokens.get('bot') or not tokens.get('user'):
            raise ValueError('пустые bot / user')
        telebot.TeleBot(tokens['bot']).get_me()
        return tokens
    except Exception as e:
        print(f'Telegram-бот не привязан ({e}), уведомления отключены. '
              f'Шаблон credentials: telegram_credits.example.json')
        return None


def get_tg_logger(config):
    """Логгер с отправкой в Telegram; None, если уведомления выключены или бот не привязан."""
    if not config.logging.telegram:
        return None

    tokens = _read_tokens(config)
    if tokens is None:
        return None

    logger = logging.getLogger('house_prices')
    logger.setLevel(logging.INFO)
    tg_logger.setup(logger, token=tokens['bot'], users=[tokens['user']])
    return logger


def send_plot(config, caption: str = '') -> None:
    """Шлёт график эксперимента в Telegram."""
    if not config.logging.telegram:
        return

    tokens = _read_tokens(config)
    if tokens is None:
        return
    try:
        with open(plot_path(config), 'rb') as photo:
            telebot.TeleBot(tokens['bot']).send_photo(tokens['user'], photo, caption=caption)
    except Exception as e:
        print(f'График не отправлен в Telegram: {e}')


def plot_path(config) -> Path:
    """Путь к графику текущего эксперимента."""
    path = Path(config.paths.experiments_dir) / 'plots' / f'{config.general.experiment_name}.png'
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_cv(config, table: pd.DataFrame) -> None:
    """CV RMSE ± std по моделям + train-точки. Меньше — лучше;
    чем дальше train-точка левее столбца, тем сильнее переобучение."""
    t = table.iloc[::-1]  # лучшая модель сверху
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(t) + 1))
    ax.barh(t['model'], t['cv_mean'], xerr=t['cv_std'], label='CV')
    ax.scatter(t['train'], t['model'], color='red', zorder=3, label='train')
    ax.set_xlabel('RMSE по log1p(SalePrice) — меньше лучше')
    ax.set_title(config.general.experiment_name)
    ax.legend()
    fig.savefig(plot_path(config), bbox_inches='tight')
    plt.close(fig)


def save_report_figure(config) -> None:
    """Копирует график прогона в reports/figures, откуда его берёт README."""
    dest = Path(config.paths.figures_dir) / 'cv_results.png'
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(plot_path(config), dest)


def get_params(config, model_name: str):
    """Что писать в лог: гиперпараметры модели или состав ансамбля."""
    if model_name in config.ml.models:
        return config.ml.models[model_name].params
    return config.ml.ensembles[model_name]  # метод + состав


def save_run(config, table: pd.DataFrame, time_s: int) -> None:
    """Дописывает в runs.csv строку на каждую модель: метрики, параметры, коммит, время."""
    try:
        commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    except Exception:
        commit = 'unknown'

    now = datetime.now()
    rows = table.copy()
    rows.insert(0, 'run_id', now.strftime('%Y%m%d_%H%M%S'))
    rows.insert(1, 'date', now.strftime('%Y-%m-%d %H:%M'))
    rows.insert(2, 'experiment', config.general.experiment_name)
    rows.insert(3, 'commit', commit)
    rows['is_best'] = rows.index == 0
    rows['seed'] = config.general.seed
    rows['n_splits'] = config.split.n_splits
    rows['time_s'] = time_s
    rows['params'] = [json.dumps(OmegaConf.to_container(get_params(config, m), resolve=True))
                      for m in rows['model']]

    path = Path(config.paths.experiments_dir) / 'runs.csv'
    path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(path, mode='a', header=not path.exists(), index=False, float_format='%.4f')
