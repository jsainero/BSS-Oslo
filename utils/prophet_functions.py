import sys
from sklearn.metrics import mean_squared_error
from prophet import Prophet
import random
import numpy as np

random.seed(27)


if './utils' not in sys.path:
    sys.path.insert(0, './utils')
from data_functions import series_to_supervised

def prepare_data(df, n_hours, n_features):

    reframed = series_to_supervised(df, n_hours, 1)
    reframed = reframed.iloc[:, :-n_features+1]
    reframed.reset_index(inplace=True)
    reframed = reframed.rename(columns={'day_hour': 'ds', 'var1(t)': 'y'})

    return reframed


def data_split(df, test_size, validation=False):

    if test_size < 0.0 or test_size > 1.0:
        raise Exception('test_size must be between 0.0 and 1.0')

    if validation:

        TRAIN_SIZE = int(len(df)*(1.0 - 2*test_size))
        TRAIN_VALIDATION_SIZE = int(len(df)*(1.0 - test_size))

        train = df[:TRAIN_SIZE]
        validation = df[TRAIN_SIZE:TRAIN_VALIDATION_SIZE]
        test = df[TRAIN_VALIDATION_SIZE:]

        validation_X, validation_y = validation.iloc[:,
                                                     :-1], validation.iloc[:, -1]

        test_X, test_y = test.iloc[:, :-1], test.iloc[:, -1]

        return train, validation_X, validation_y, test_X, test_y

    else:
        TRAIN_SIZE = int(len(df)*(1.0 - test_size))

        train = df[:TRAIN_SIZE]
        test = df[TRAIN_SIZE:]

        test_X, test_y = test.iloc[:, :-1], test.iloc[:, -1]

        return train, None, None, test_X, test_y


def create_model(regressors, changepoint_prior_scale, seasonality_prior_scale):

    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        seasonality_prior_scale=seasonality_prior_scale
    )

    for reg in regressors:
        model.add_regressor(reg)

    return model


def train_model(model, train):

    model.fit(train)


def predict(model, test_X, test_y):

    predicciones = model.predict(test_X)['yhat']

    rmse = mean_squared_error(
        y_true=test_y,
        y_pred=predicciones,
        squared=False
    )

    rmse_rounded = mean_squared_error(
        y_true=test_y,
        y_pred=np.round(predicciones),
        squared=False
    )

    return predicciones, rmse, rmse_rounded


def run_model(df, n_hours, n_features, test_size, changepoint_prior_scale,
              seasonality_prior_scale, hyperparam_search=False):

    reframed = prepare_data(df, n_hours, n_features)

    train, validation_X, validation_y, test_X, test_y = data_split(
        reframed, test_size, validation=hyperparam_search)

    regressors = reframed.columns[1:-1]
    model = create_model(
        regressors, changepoint_prior_scale, seasonality_prior_scale)

    train_model(model, train)

    if hyperparam_search:
        y_pred, rmse, rmse_rounded = predict(model, validation_X, validation_y)
    else:
        y_pred, rmse, rmse_rounded = predict(model, test_X, test_y)

    output = {
        'data': {
            'df': reframed,
            'train': train,
            'validation_X': validation_X,
            'validation_y': validation_y,
            'test_X': test_X,
            'test_y': test_y
        },
        'model': model,
        'test_y_pred': y_pred,
        'test_rmse': rmse,
        'test_rmse_rounded': rmse_rounded
    }

    return output
