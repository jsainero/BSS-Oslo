import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM, GRU
from keras.callbacks import EarlyStopping
from keras.utils import set_random_seed
import matplotlib.pyplot as plt
import numpy as np


if './utils' not in sys.path:
    sys.path.insert(0, './utils')
from data_functions import series_to_supervised

np.random.seed(27)
set_random_seed(27)


def prepare_data(df, n_hours, n_features):

    values = df.values
    values = values.astype('float32')
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(values)
    reframed = series_to_supervised(scaled, n_hours, 1).values
    reframed = reframed[:, :-n_features+1]

    return reframed, scaler


def data_split(df, n_hours, n_features, test_size, validation=False):

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

        train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
        validation_X = validation_X.reshape(
            (validation_X.shape[0], n_hours, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

        return train_X, train_y, validation_X, validation_y, test_X, test_y

    else:
        TRAIN_SIZE = int(len(df)*(1.0 - test_size))

        train = df[:TRAIN_SIZE]
        test = df[TRAIN_SIZE:]

        train_X, train_y = train[:, :-1], train[:, -1]
        test_X, test_y = test[:, :-1], test[:, -1]

        train_X = train_X.reshape((train_X.shape[0], n_hours, n_features))
        test_X = test_X.reshape((test_X.shape[0], n_hours, n_features))

        return train_X, train_y, None, None, test_X, test_y


def create_model(rnn_type, n_hours, n_features, units, optimizer):

    if rnn_type not in ('LSTM', 'GRU'):
        raise Exception('Invalid value for type')

    model = Sequential()
    if rnn_type == 'LSTM':
        model.add(LSTM(units, return_sequences=True,
                       input_shape=(n_hours, n_features)))
        model.add(LSTM(units))
    else:
        model.add(GRU(units, return_sequences=True,
                      input_shape=(n_hours, n_features)))
        model.add(GRU(units))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(loss='mae', optimizer=optimizer)

    return model


def train_model(model, epochs, batch_size, early_stopping,
                train_X, train_y, validation_X, validation_y):

    callback = EarlyStopping(monitor='loss', patience=early_stopping)

    history = model.fit(train_X, train_y, epochs=epochs, batch_size=batch_size, validation_data=(
        validation_X, validation_y), verbose=0, shuffle=False, callbacks=[callback])

    return history


def plot_history(history):

    plt.figure(figsize=(15, 5))
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.legend()
    plt.show()


def predict(model, scaler, n_hours, n_features, test_X, test_y):

    # make a prediction
    yhat = model.predict(test_X)
    test_X = test_X.reshape((test_X.shape[0], n_features*n_hours))
    # invert scaling for forecast
    inv_yhat = np.concatenate((yhat, test_X[:, -n_features+1:]), axis=1)
    inv_yhat = scaler.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:, 0]
    # invert scaling for actual
    test_y = test_y.reshape((len(test_y), 1))
    inv_y = np.concatenate((test_y, test_X[:, -n_features+1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]
    # calculate RMSE
    rmse = mean_squared_error(
        y_true=inv_y,
        y_pred=inv_yhat,
        squared=False
    )

    rmse_rounded = mean_squared_error(
        y_true=inv_y,
        y_pred=np.round(inv_yhat),
        squared=False
    )

    return inv_yhat, rmse, rmse_rounded


def run_model(df, n_hours, n_features, test_size, rnn_type, units,
              optimizer, epochs, batch_size, early_stopping,
              hyperparam_search=False):

    reframed, scaler = prepare_data(df, n_hours, n_features)

    train_X, train_y, validation_X, validation_y, test_X, test_y = data_split(
        reframed, n_hours, n_features, test_size, validation=hyperparam_search)

    model = create_model(rnn_type, n_hours, n_features, units, optimizer)

    if hyperparam_search:
        history = train_model(model, epochs, batch_size, early_stopping,
                              train_X, train_y, validation_X, validation_y)

        y_pred, rmse, rmse_rounded = predict(model, scaler, n_hours,
                                             n_features, validation_X, validation_y)
    else:
        history = train_model(model, epochs, batch_size, early_stopping,
                              train_X, train_y, test_X, test_y)

        y_pred, rmse, rmse_rounded = predict(model, scaler, n_hours,
                                             n_features, test_X, test_y)
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
        'history': history,
        'test_y_pred': y_pred,
        'test_rmse': rmse,
        'test_rmse_rounded': rmse_rounded
    }

    return output
