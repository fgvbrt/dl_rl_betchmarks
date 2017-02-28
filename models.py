from keras.layers import Conv2D, Dense, Flatten, Activation, LSTM, GRU
from keras.models import Sequential
from keras.layers.wrappers import TimeDistributed
import keras.backend as K


def change_input_dim(input_dim):
    if K.image_dim_ordering() == 'tf':
        input_dim = list(input_dim[1:]) + [input_dim[0]]

    return list(input_dim)


def get_rnn(rnn_type):
    if rnn_type == 'LSTM':
        return LSTM
    elif rnn_type == 'GRU':
        return GRU
    else:
        raise ValueError('wrong rnn type')


def build_fc(input_dim=100, out_size=10):
    model = Sequential([
        Dense(100, input_dim=input_dim),
        Activation('relu'),
        Dense(out_size),
        Activation('softmax'),
    ])

    return model


def build_cnn_dqn(input_dim=(4, 84, 84),  out_size=10):
    assert len(input_dim) == 3

    input_dim = change_input_dim(input_dim)

    model = Sequential([
        Conv2D(32, 8, 8, subsample=(4, 4), activation='relu', input_shape=input_dim),
        Conv2D(64, 4, 4, subsample=(2, 2), activation='relu'),
        Conv2D(64, 3, 3, subsample=(1, 1), activation='relu'),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(out_size),
        Activation('softmax')
    ])

    return model


def build_cnn_openai(input_dim=(4, 42, 42), out_size=10):
    assert len(input_dim) == 3

    input_dim = change_input_dim(input_dim)

    model = Sequential([
        Conv2D(32, 3, 3, subsample=(2, 2), activation='elu', border_mode='same', input_shape=input_dim),
        Conv2D(32, 3, 3, subsample=(2, 2), activation='elu'),
        Conv2D(32, 3, 3, subsample=(2, 2), activation='elu'),
        Conv2D(32, 3, 3, subsample=(2, 2), activation='elu'),
        Flatten(),
        Dense(256, activation='relu'),
        Dense(out_size),
        Activation('softmax')
    ])

    return model


def build_rnn_fc(input_dim=100, seq_len=None, out_size=10, consume_less='gpu', rnn_type='LSTM',):
    rnn = get_rnn(rnn_type)

    input_shape = [seq_len, input_dim]
    unroll = seq_len > 0

    model = Sequential([
        rnn(256, return_sequences=True, unroll=unroll, input_shape=input_shape, consume_less=consume_less),
        TimeDistributed(Dense(out_size)),
        TimeDistributed(Activation('softmax')),
    ])

    return model


def build_rnn_dqn(input_dim=(1, 84, 84),  seq_len=None, out_size=10, consume_less='gpu', rnn_type='LSTM',):
    assert len(input_dim) == 3

    rnn = get_rnn(rnn_type)

    input_dim = change_input_dim(input_dim)
    input_dim = [seq_len] + input_dim
    unroll = seq_len > 0

    model = Sequential([
        TimeDistributed(Conv2D(32, 8, 8, subsample=(4, 4), activation='relu'), input_shape=input_dim),
        TimeDistributed(Conv2D(64, 4, 4, subsample=(2, 2), activation='relu')),
        TimeDistributed(Conv2D(64, 3, 3, subsample=(1, 1), activation='relu')),
        TimeDistributed(Flatten()),
        rnn(256, unroll=unroll, return_sequences=True, consume_less=consume_less),
        TimeDistributed(Dense(out_size)),
        TimeDistributed(Activation('softmax')),
    ])

    return model


def build_rnn_openai(input_dim=(1, 42, 42),  seq_len=None, out_size=10, consume_less='gpu', rnn_type='LSTM',):
    assert len(input_dim) == 3

    rnn = get_rnn(rnn_type)

    input_dim = change_input_dim(input_dim)
    input_dim = [seq_len] + input_dim
    unroll = seq_len > 0

    model = Sequential([
        TimeDistributed(Conv2D(32, 3, 3, subsample=(2, 2), activation='elu'), input_shape=input_dim),
        TimeDistributed(Conv2D(32, 3, 3, subsample=(2, 2), activation='elu')),
        TimeDistributed(Conv2D(32, 3, 3, subsample=(2, 2), activation='elu')),
        TimeDistributed(Conv2D(32, 3, 3, subsample=(2, 2), activation='elu')),
        TimeDistributed(Flatten()),
        rnn(256, unroll=unroll, return_sequences=True, consume_less=consume_less),
        TimeDistributed(Dense(out_size)),
        TimeDistributed(Activation('softmax')),
    ])

    return model
