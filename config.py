from omegaconf import OmegaConf

config = {
    'general': {
        'experiment_name': 'baseline v1',
        'seed': 42,
        'id_column': 'Id',
        'target': 'SalePrice',
        'target_transform': 'log1p',  # метрика — RMSE по логарифмам; обратно — np.expm1
    },
    'paths': {
        'data_dir': 'data/raw',
        'train_csv': '${paths.data_dir}/train.csv',
        'test_csv': '${paths.data_dir}/test.csv',
        'sample_submission_csv': '${paths.data_dir}/sample_submission.csv',
        'data_description': '${paths.data_dir}/data_description.txt',
        'submission_dir': 'submissions',
        'experiments_dir': 'experiments',
        'figures_dir': 'reports/figures',
        'telegram_credentials': 'telegram_credits.json',
    },
    'logging': {
        'telegram': True,
        'runs_csv': True,
        'report_figure': False,  # True — скопировать график прогона в reports/figures для README
    },
    'preprocessing': {
        'drop_columns': ['Id'],
        'outliers': {'GrLivArea_min': 4000, 'SalePrice_max': 300000},
        'as_category': ['MSSubClass'],  # код типа здания, а не число

        'fill_none_cat': ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'MasVnrType',
                          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
                          'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'],

        'fill_zero_num': ['MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF',
                          'BsmtFullBath', 'BsmtHalfBath', 'GarageCars', 'GarageArea'],

        'fill_mode': ['MSZoning', 'Electrical', 'KitchenQual', 'Functional', 'SaleType',
                      'Exterior1st', 'Exterior2nd', 'Utilities'],
        # LotFrontage -> медиана по Neighborhood (train); GarageYrBlt -> YearBuilt

        'ordinal_maps': {
            'quality': {'None': 0, 'Po': 1, 'Fa': 2, 'TA': 3, 'Gd': 4, 'Ex': 5},
            'exposure': {'None': 0, 'No': 1, 'Mn': 2, 'Av': 3, 'Gd': 4},
            'fin_type': {'None': 0, 'Unf': 1, 'LwQ': 2, 'Rec': 3, 'BLQ': 4, 'ALQ': 5, 'GLQ': 6},
            'garage_finish': {'None': 0, 'Unf': 1, 'RFn': 2, 'Fin': 3},
            'functional': {'Sal': 0, 'Sev': 1, 'Maj2': 2, 'Maj1': 3, 'Mod': 4, 'Min2': 5, 'Min1': 6, 'Typ': 7},
            'lot_shape': {'IR3': 0, 'IR2': 1, 'IR1': 2, 'Reg': 3},
            'land_slope': {'Sev': 0, 'Mod': 1, 'Gtl': 2},
            'paved_drive': {'N': 0, 'P': 1, 'Y': 2},
        },
        'ordinal': {
            'ExterQual': 'quality', 'ExterCond': 'quality', 'BsmtQual': 'quality', 'BsmtCond': 'quality',
            'HeatingQC': 'quality', 'KitchenQual': 'quality', 'FireplaceQu': 'quality',
            'GarageQual': 'quality', 'GarageCond': 'quality', 'PoolQC': 'quality',
            'BsmtExposure': 'exposure', 'BsmtFinType1': 'fin_type', 'BsmtFinType2': 'fin_type',
            'GarageFinish': 'garage_finish', 'Functional': 'functional',
            'LotShape': 'lot_shape', 'LandSlope': 'land_slope', 'PavedDrive': 'paved_drive',
        },
        'one_hot': ['MSSubClass', 'MSZoning', 'Street', 'Alley', 'LandContour', 'Utilities', 'LotConfig',
                    'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                    'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Foundation', 'Heating',
                    'CentralAir', 'Electrical', 'GarageType', 'Fence', 'MiscFeature', 'SaleType',
                    'SaleCondition'],
    },
    'split': {
        'n_splits': 5,
    },
    'ml': {
        'run_all': ['ridge', 'lasso', 'elastic_net', 'svr', 'random_forest',
                    'gbr', 'xgboost', 'lightgbm', 'catboost', 'voting', 'stacking'],
        'ensembles': {
            'voting': {'method': 'voting', 'members': ['lasso', 'svr', 'gbr', 'lightgbm']},
            'stacking': {'method': 'stacking', 'final': 'sklearn.linear_model.Ridge',
                         'members': ['lasso', 'svr', 'gbr', 'lightgbm']},
        },
        'models': {
            'ridge': {
                'name': 'sklearn.linear_model.Ridge',
                'scale': True,
                'params': {'alpha': 10.0, 'random_state': '${general.seed}'},
            },
            'lasso': {
                'name': 'sklearn.linear_model.Lasso',
                'scale': True,
                'params': {'alpha': 0.0005, 'max_iter': 50000, 'random_state': '${general.seed}'},
            },
            'elastic_net': {
                'name': 'sklearn.linear_model.ElasticNet',
                'scale': True,
                'params': {'alpha': 0.0005, 'l1_ratio': 0.9, 'max_iter': 50000, 'random_state': '${general.seed}'},
            },
            'svr': {
                'name': 'sklearn.svm.SVR',
                'scale': True,
                'params': {'C': 20, 'epsilon': 0.008, 'gamma': 0.0003},
            },
            'random_forest': {
                'name': 'sklearn.ensemble.RandomForestRegressor',
                'scale': False,
                'params': {'n_estimators': 300, 'max_features': 0.3, 'n_jobs': -1, 'random_state': '${general.seed}'},
            },
            'gbr': {
                'name': 'sklearn.ensemble.GradientBoostingRegressor',
                'scale': False,
                'params': {'n_estimators': 1000, 'learning_rate': 0.05, 'max_depth': 4, 'max_features': 'sqrt',
                           'min_samples_leaf': 15, 'loss': 'huber', 'random_state': '${general.seed}'},
            },
            'xgboost': {
                'name': 'xgboost.XGBRegressor',
                'scale': False,
                'params': {'n_estimators': 1500, 'learning_rate': 0.03, 'max_depth': 3, 'subsample': 0.7,
                           'colsample_bytree': 0.5, 'random_state': '${general.seed}'},
            },
            'lightgbm': {
                'name': 'lightgbm.LGBMRegressor',
                'scale': False,
                'params': {'n_estimators': 1500, 'learning_rate': 0.03, 'num_leaves': 8, 'subsample': 0.7,
                           'subsample_freq': 1, 'colsample_bytree': 0.5, 'verbose': -1,
                           'random_state': '${general.seed}'},
            },
            'catboost': {
                'name': 'catboost.CatBoostRegressor',
                'scale': False,
                'params': {'iterations': 1500, 'learning_rate': 0.03, 'depth': 6,
                           'verbose': 0, 'random_seed': '${general.seed}'},
            },
        },
    },
}

config = OmegaConf.create(config)
