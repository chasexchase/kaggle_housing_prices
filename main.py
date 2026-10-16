from config import config

import time

import pandas as pd

from data import preprocess
from logger import get_tg_logger, plot_cv, save_report_figure, save_run, send_plot
from ml_model import MLTrainer
from utils import check_dataset, set_seed


def main() -> None:

    set_seed(config.general.seed)
    name = config.general.experiment_name
    start_time = time.time()

    logger = get_tg_logger(config)
    if logger:
        logger.info(f'Старт: {name}')

    train = pd.read_csv(config.paths.train_csv)
    test = pd.read_csv(config.paths.test_csv)
    ids = test[config.general.id_column]  # Id удаляется в preprocess — забираем заранее

    X, X_test, y = preprocess(train, test, config)
    check_dataset(X, y, X_test)

    model = MLTrainer(config)
    table = model.evaluate(X, y)
    model.fit_best(X, y)
    model.save_submission(X_test, ids)

    t = int(time.time() - start_time)
    if config.logging.runs_csv:
        save_run(config, table, t)

    plot_cv(config, table)
    if config.logging.report_figure:
        save_report_figure(config)

    if logger:
        lines = '\n'.join(f'{r.model}: {r.cv_mean:.4f} ± {r.cv_std:.4f}' for r in table.itertuples())
        logger.info(f'{name} | {t} s | лучшая: {model.best_name}\n\n{lines}')
        send_plot(config, caption=f'{name} | лучшая: {model.best_name}')


if __name__ == '__main__':
    main()
