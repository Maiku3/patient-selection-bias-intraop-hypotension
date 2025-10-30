import os
import pickle
import numpy as np
import csv

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, concatenate, multiply, dot
from tensorflow.keras.layers import Layer, Dense, Flatten, Dropout, MultiHeadAttention, LayerNormalization, LSTM
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Add, ReLU, BatchNormalization
from tensorflow.keras.layers import GlobalMaxPool1D, GlobalAveragePooling1D


DATASET_PATH = "./dataset/"


def load_mbp_or_wave_dataset_with_sampling(data_selection_type, testset_path, pred_win, sampling_rate,
                                                             return_type,
                                                             remove_under65_in_train = False,
                                                             filter_nan = False, seed = 1234):

    supporting_selection_type = ['biased_selection', 'random_selection']
    if not data_selection_type in supporting_selection_type:
        raise ValueError('data_setection_type must be one of {}'.format(str(supporting_selection_type)))

    supporting_prediction_window = [5, 10, 15]
    if not pred_win in supporting_prediction_window:
        raise ValueError('prediction_window must be one of {}'.format(str(supporting_prediction_window)))

    supporting_return_type = ['mbp', 'wave']
    if not return_type in supporting_return_type:
        raise ValueError('return type must be one of {}'.format(str(supporting_return_type)))


    if data_selection_type == 'biased_selection':
        x = pickle.load(
            open(os.path.join(DATASET_PATH, 'x_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
        mbp = pickle.load(
            open(os.path.join(DATASET_PATH, 'mbp_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
        y = pickle.load(
            open(os.path.join(DATASET_PATH, 'y_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
        c = pickle.load(
            open(os.path.join(DATASET_PATH, 'c_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))

    elif data_selection_type == 'random_selection':
        x = pickle.load(
            open(os.path.join(DATASET_PATH, 'x_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
        mbp = pickle.load(
            open(os.path.join(DATASET_PATH, 'mbp_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
        y = pickle.load(
            open(os.path.join(DATASET_PATH, 'y_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
        c = pickle.load(
            open(os.path.join(DATASET_PATH, 'c_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))

    else:
        raise ValueError('data_setection_type must be one of {}'.format(str(supporting_selection_type)))

    if filter_nan:
        # remove wave with nan
        filter_index_mbp = ~np.isnan(mbp).any(axis=1)
        filter_index_wave = ~np.isnan(x).any(axis=1)
        filter_index = filter_index_mbp & filter_index_wave
        x = x[filter_index]
        mbp = mbp[filter_index]
        y = y[filter_index]
        c = c[filter_index]

    # wave data normalization
    wav_min = 0.
    wav_max = 200.
    x_ = (x - wav_min) / (wav_max - wav_min)

    # split train and test
    caseids = list(np.unique(c))

    with open(testset_path, 'r') as f:
        caseids_test = list(csv.reader(f, delimiter=','))
        caseids_test = caseids_test[0]

    caseids_test = np.array(caseids_test).astype(float).astype(int)

    test_mask = np.isin(c, caseids_test)
    train_mask = np.invert(test_mask)

    x_train = x[train_mask]
    mbp_train = mbp[train_mask]
    y_train = y[train_mask]
    c_train = y[train_mask]

    if remove_under65_in_train:
        # remove train segment if mean of wave under 65mmHg
        over_65_filter_mbp = np.nanmean(mbp_train, axis=1) >= 65
        over_65_filter_wave = np.nanmean(x_train, axis=1) >= (65 - wav_min) / (wav_max - wav_min)
        over_65_filter = over_65_filter_mbp & over_65_filter_wave
        x_train = x_train[over_65_filter]
        mbp_train = mbp_train[over_65_filter]
        y_train = y_train[over_65_filter]
        c_train = c_train[over_65_filter]

    x_test = x[test_mask]
    mbp_test = mbp[test_mask]
    y_test = y[test_mask]
    c_test = c[test_mask]

    if (sampling_rate is None) or (sampling_rate == 1):
        print('====================================================')
        print('total: {} cases {} samples'.format(len(caseids), len(y)))
        print('train: {} cases {} samples'.format(len(np.unique(c_train)), len(y_train)))
        print('test: {} cases {} samples'.format(len(np.unique(c_test)), len(y_test)))
        print('====================================================')

        if return_type == 'mbp':
            return mbp_train, y_train, c_train, mbp_test, y_test, c_test
        elif return_type == 'wave':
            return x_train, y_train, c_train, x_test, y_test, c_test
        else:
            raise (ValueError('supporting_return_type must be one of {}'.format(str(supporting_return_type))))

    if (sampling_rate > 0) and (sampling_rate < 1):
        np.random.seed(seed)
        rand_idx_train = np.random.randint(len(mbp_train), size=int(len(mbp_train) * sampling_rate))
        x_train = x_train[rand_idx_train, :]
        mbp_train = mbp_train[rand_idx_train, :]
        y_train = y_train[rand_idx_train]
        c_train = c_train[rand_idx_train]

    print('====================================================')
    print('total: {} cases {} samples'.format(len(caseids), len(y)))
    print('train: {} cases {} samples'.format(len(np.unique(c_train)), len(y_train)))
    print('test: {} cases {} samples'.format(len(np.unique(c_test)), len(y_test)))
    print('====================================================')

    if return_type == 'mbp':
        return mbp_train, y_train, c_train, mbp_test, y_test, c_test
    elif return_type == 'wave':
        return x_train, y_train, c_train, x_test, y_test, c_test
    else:
        raise(ValueError('supporting_return_type must be one of {}'.format(str(supporting_return_type))))


def load_mbp_or_wave_testing_dataset(testset_path, pred_win, return_type,
                                     remove_under65_in_test=False, filter_nan=False):

    supporting_return_type = ['mbp', 'wave']
    if not return_type in supporting_return_type:
        raise ValueError('return_type must be one of {}'.format(str(supporting_return_type)))

    x = pickle.load(
        open(os.path.join(DATASET_PATH, 'x_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
    mbp = pickle.load(
        open(os.path.join(DATASET_PATH, 'mbp_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
    y = pickle.load(
        open(os.path.join(DATASET_PATH, 'y_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))
    c = pickle.load(
        open(os.path.join(DATASET_PATH, 'c_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'rb'))

    if filter_nan:
        filter_index_mbp = ~np.isnan(mbp).any(axis=1)
        filter_index_wave = ~np.isnan(x).any(axis=1)
        filter_index = filter_index_mbp & filter_index_wave
        x = x[filter_index]
        mbp = mbp[filter_index]
        y = y[filter_index]
        c = c[filter_index]

    # wave data normalization
    wav_min = 0.
    wav_max = 200.
    x = (x - wav_min) / (wav_max - wav_min)

    # load test_caseids
    with open(testset_path, 'r') as f:
        caseids_test = list(csv.reader(f, delimiter=','))
        caseids_test = caseids_test[0]
    caseids_test = np.array(caseids_test).astype(float).astype(int)
    test_mask = np.isin(c, caseids_test)

    x_test = x[test_mask]
    mbp_test = mbp[test_mask]
    y_test = y[test_mask]
    c_test = c[test_mask]

    if remove_under65_in_test:
        # remove test segment if mean of wave under 65mmHg
        over_65_filter_mbp = np.nanmean(mbp_test, axis=1) >= 65
        over_65_filter_wave = np.nanmean(x_test, axis=1) >= (65 - wav_min) / (wav_max - wav_min)
        over_65_filter = over_65_filter_mbp & over_65_filter_wave
        x_test = x_test[over_65_filter]
        mbp_test = mbp_test[over_65_filter]
        y_test = y_test[over_65_filter]
        c_test = c_test[over_65_filter]

    if return_type == 'mbp':
        return mbp_test, y_test, c_test
    elif return_type == 'wave':
        return x_test, y_test, c_test
    else:
        raise(ValueError('supporting_return_type must be one of {}'.format(str(supporting_return_type))))



def load_cnn_model_architecture(sample_x):

    inp = Input(shape=(sample_x.shape[1], 1))

    # stem
    x = Conv1D(filters=32, kernel_size=5, activation='relu')(inp)

    # block 1
    fx = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(x)
    fx = Conv1D(filters=32, kernel_size=5, padding='same')(fx)
    out = Add()([x, fx])
    out = ReLU()(out)
    out = MaxPooling1D(pool_size=5, strides=2)(out)

    # block 2
    fx = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(out)
    fx = Conv1D(filters=32, kernel_size=5, padding='same')(fx)
    out = Add()([out, fx])
    out = ReLU()(out)
    out = MaxPooling1D(pool_size=5, strides=2)(out)

    # block 3
    fx = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(out)
    fx = Conv1D(filters=32, kernel_size=5, padding='same')(fx)
    out = Add()([out, fx])
    out = ReLU()(out)
    out = MaxPooling1D(pool_size=5, strides=2)(out)

    # block 4
    fx = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(out)
    fx = Conv1D(filters=32, kernel_size=5, padding='same')(fx)
    out = Add()([out, fx])
    out = ReLU()(out)
    out = MaxPooling1D(pool_size=5, strides=2)(out)

    # block 5
    fx = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(out)
    fx = Conv1D(filters=32, kernel_size=5, padding='same')(fx)
    out = Add()([out, fx])
    out = ReLU()(out)
    out = MaxPooling1D(pool_size=5, strides=2)(out)

    # block 6
    fx = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(out)
    fx = Conv1D(filters=32, kernel_size=5, padding='same')(fx)
    out = Add()([out, fx])
    out = ReLU()(out)
    out = MaxPooling1D(pool_size=5, strides=2)(out)

    # block 7
    fx = Conv1D(filters=32, kernel_size=5, padding='same', activation='relu')(out)
    fx = Conv1D(filters=32, kernel_size=5, padding='same')(fx)
    out = Add()([out, fx])
    out = ReLU()(out)
    out = MaxPooling1D(pool_size=5, strides=2)(out)

    # MLP Layers
    out = Flatten()(out)
    out = Dense(128, activation='relu')(out)
    out = Dense(32, activation='relu')(out)
    out = Dense(1, activation='sigmoid')(out)

    cnn_model = Model(inputs=[inp], outputs=[out])
    adam_optimizer = tf.keras.optimizers.Adam()
    cnn_model.compile(loss='binary_crossentropy', optimizer=adam_optimizer, metrics=['accuracy'])

    return cnn_model


def load_transformer_model_architecture(sample_x):

    # predefined hyperparameters
    hyperparameters = {
        "nfilt": [32],
        'nhead': [2],
        'kdim': [32],
        "fnode": [32],
        "clayer": [3],
        "tlayer": [4],
        "droprate": [0.1],
        "filtsize": [9],
        'poolsize': [2]
    }

    inp = Input(shape=(sample_x.shape[1], 1))
    for i in range(hyperparameters['clayer'][0]):
        if i == 0:
            out = Conv1D(filters=hyperparameters['nfilt'][0], kernel_size=hyperparameters['filtsize'][0],
                         padding='same', activation='relu')(inp)
            out = MaxPooling1D(hyperparameters['poolsize'][0], padding='same')(out)
        else:
            out = Conv1D(filters=hyperparameters['nfilt'][0], kernel_size=hyperparameters['filtsize'][0],
                         padding='same', activation='relu')(out)
            out = MaxPooling1D(hyperparameters['poolsize'][0], padding='same')(out)
    out = Dense(hyperparameters['kdim'][0])(out)  # 마지막 차원이 nfilt 인데 kdim 으로 바꿔야 transformer block을 쌓을 수 있다.
    for i in range(hyperparameters['tlayer'][0]):  # transformer
        attn_output = MultiHeadAttention(num_heads=hyperparameters['nhead'][0],
                                         key_dim=hyperparameters['kdim'][0], attention_axes=[1, ])(out, out)
        attn_output = Dropout(hyperparameters['droprate'][0])(attn_output)
        out1 = LayerNormalization(epsilon=1e-6)(out + attn_output)  # sum and norm
        ffn_output = Dense(hyperparameters['fnode'][0], activation='relu')(out1)
        ffn_output = Dense(hyperparameters['kdim'][0])(ffn_output)
        # keras.Sequential([Dense(fnode, activation="relu"), Dense(kdim)])(out1)
        out2 = Dropout(hyperparameters['droprate'][0])(ffn_output)
        out = LayerNormalization(epsilon=1e-6)(out1 + out2)  # sum and norm
    out = GlobalAveragePooling1D()(out)
    out = Dropout(hyperparameters['droprate'][0])(out)
    out = Dense(hyperparameters['fnode'][0], activation='relu')(out)
    out = Dropout(hyperparameters['droprate'][0])(out)
    out = Dense(1, activation='sigmoid')(out)

    transformer = Model(inputs=[inp], outputs=[out])
    transformer.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])

    return transformer


def load_lstm_model_architecture_3channel(sample_x):

    lstm_model = Sequential()
    lstm_model.add(LSTM(16, input_shape=(sample_x.shape[1], 3)))
    lstm_model.add(BatchNormalization())
    lstm_model.add(Dense(1, activation = 'sigmoid'))
    lstm_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC()])

    return lstm_model
