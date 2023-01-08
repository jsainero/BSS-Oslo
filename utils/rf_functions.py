import sys
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np

if './utils' not in sys.path:
    sys.path.insert(0, './utils')
from data_functions import series_to_supervised

RANDOM_STATE = 27


def prepare_data(df, n_hours, n_features):

    values = df.values
    values = values.astype('float32')
    reframed = series_to_supervised(values, n_hours, 1).values
    reframed = reframed[:, :-n_features+1]

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

        train_X, train_y = train[:, :-1], train[:, -1]
        validation_X, validation_y = validation[:, :-1], validation[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        return train_X, train_y, validation_X, validation_y, test_X, test_y

    else:
        TRAIN_SIZE = int(len(df)*(1.0 - test_size))

        train = df[:TRAIN_SIZE]
        test = df[TRAIN_SIZE:]

        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        return train_X, train_y, None, None, test_X, test_y


def create_model(n_estimators, max_depth, max_features):

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion='squared_error',
        max_depth=max_depth,
        max_features=max_features,
        n_jobs=-1,
        random_state=RANDOM_STATE
    )

    return model


def train_model(model, train_X, train_y):

    model.fit(train_X, train_y)


def predict(model, test_X, test_y):

    predicciones = model.predict(X=test_X)

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


def run_model(df, n_hours, n_features, test_size, n_estimators, max_depth,
              max_features, hyperparam_search=False):

    reframed = prepare_data(df, n_hours, n_features)

    train_X, train_y, validation_X, validation_y, test_X, test_y = data_split(
        reframed, test_size, validation=hyperparam_search)

    model = create_model(n_estimators, max_depth, max_features)

    train_model(model, train_X, train_y)

    if hyperparam_search:
        y_pred, rmse, rmse_rounded = predict(model, validation_X, validation_y)
    else:
        y_pred, rmse, rmse_rounded = predict(model, test_X, test_y)

    output = {
        'data': {
            'df': reframed,
            'train_X': train_X,
            'train_y': train_y,
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
