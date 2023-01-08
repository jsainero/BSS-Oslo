import sys
from sklearn.metrics import mean_squared_error
from greykite.algo.forecast.silverkite.forecast_silverkite import SilverkiteForecast
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
    reframed.columns = [col.replace('(', '_').replace(
        '-', '').replace(')', '') for col in reframed.columns]

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


def create_model():

    model = SilverkiteForecast()

    return model


def train_model(model, train, regressors, min_admissible_value, changepoint_method):

    changepoints_dict = {
        'method': changepoint_method
    }if changepoint_method is not None else None

    history = model.forecast(
        df=train,
        time_col="ds",  # name of the time column
        value_col="y",  # name of the value column
        extra_pred_cols=list(regressors),
        min_admissible_value=min_admissible_value,
        changepoints_dict=changepoints_dict
    )

    return history


def predict(model, history, test_X, test_y):

    predicciones = model.predict(
        fut_df=test_X,
        trained_model=history
    )['fut_df']['y']

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


def run_model(df, n_hours, n_features, test_size, min_admissible_value,
              changepoint_method, hyperparam_search=False):

    reframed = prepare_data(df, n_hours, n_features)

    train, validation_X, validation_y, test_X, test_y = data_split(
        reframed, test_size, validation=hyperparam_search)

    regressors = reframed.columns[1:-1]
    model = create_model()

    history = train_model(model, train, regressors,
                          min_admissible_value, changepoint_method)

    if hyperparam_search:
        y_pred, rmse, rmse_rounded = predict(
            model, history, validation_X, validation_y)
    else:
        y_pred, rmse, rmse_rounded = predict(model, history, test_X, test_y)

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
        'history': history,
        'test_y_pred': y_pred,
        'test_rmse': rmse,
        'test_rmse_rounded': rmse_rounded
    }

    return output
