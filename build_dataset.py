import vitaldb
import numpy as np
import pandas as pd
import tqdm
import pickle
import os
import gc
import multiprocessing
import scipy.signal as sig
from pyvital import arr


# Define saving path
SAVE_PATH = "./dataset/"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)


# Define hyperparameters
PREDICTION_WINDOW = [5, 10, 15]
BATCH_SIZE = 512
SRATE=100
SEGMENT_SIZE = 1
CASE_DURATION = 1
CONTROL_DURATION = 5
CONTROL_APART = 20
SKIP_INTERVAL = 1


# Read case and track information from vitaldb
track_name_pd = pd.read_csv("https://api.vitaldb.net/trks")
case_info_pd = pd.read_csv("https://api.vitaldb.net/cases")

# revmove eligible cases
eligible_caseids = list(
    set(track_name_pd[track_name_pd['tname'] == 'SNUADC/ART']['caseid']) &
    set(case_info_pd[case_info_pd['age'] > 18]['caseid']) &
    set(case_info_pd[case_info_pd['age'] >= 18]['caseid']) &
    set(case_info_pd[case_info_pd['weight'] >= 30]['caseid']) &
    set(case_info_pd[case_info_pd['weight'] < 140]['caseid']) &
    set(case_info_pd[case_info_pd['height'] >= 135]['caseid']) &
    set(case_info_pd[case_info_pd['height'] < 200]['caseid']) &
    set(case_info_pd[~case_info_pd['opname'].str.contains("transplant", case=False)]['caseid']) &
    set(case_info_pd[~case_info_pd['opname'].str.contains("aneurysm", case=False)]['caseid']) &
    set(case_info_pd[~case_info_pd['opname'].str.contains("aorto", case=False)]['caseid'])&
    set(case_info_pd[case_info_pd['ane_type'] == 'General']['caseid'])
)
print('Total {} cases found'.format(len(eligible_caseids)))


######################################
# 00. Define utility functions #
######################################

# abp preprocessing
def abp_process_beat(seg):
    """
    :param seg:
    :return:
    """
    # return: mean_std, avg_beat
    minlist, maxlist = arr.detect_peaks(seg, 100)

    if (minlist is None) and (maxlist is None):
        return 0, []

    # beat lengths
    beatlens = []
    beats = []
    beats_128 = []
    for i in range(1, len(maxlist) - 1):
        beatlen = maxlist[i] - maxlist[i - 1]  # in samps
        pp = seg[maxlist[i]] - seg[minlist[i - 1]]  # pulse pressure

        # allow hr 20 - 200
        if pp < 20:
            return 0, []
        elif beatlen < 30:  # print('{} too fast rhythm {}'.format(id, beatlen))
            return 0, []
        elif beatlen > 300 or (i == 1 and maxlist[0] > 300) or (i == len(maxlist) - 1 and len(seg) - maxlist[i] > 300):
            # print ('{} too slow rhythm {}', format(id, beatlen))
            return 0, []
        else:
            beatlens.append(beatlen)
            beat = seg[minlist[i - 1]: minlist[i]]
            beats.append(beat)
            resampled = sig.resample(beat, 128)
            beats_128.append(resampled)

    if not beats_128:
        return 0, []

    avgbeat = np.array(beats_128).mean(axis=0)

    nucase_mbeats = len(beats)
    if nucase_mbeats < 30:  # print('{} too small # of rhythm {}'.format(id, nucase_mbeats))
        return 0, []
    else:
        meanlen = np.mean(beatlens)
        stdlen = np.std(beatlens)
        if stdlen > meanlen * 0.2:  # print('{} irregular thythm', format(id))
            return 0, []

    # select wave with beat correlation > 0.9
    beatstds = []
    for i in range(len(beats_128)):
        if np.corrcoef(avgbeat, beats_128[i])[0, 1] > 0.9:
            beatstds.append(np.std(beats[i]))

    if len(beatstds) * 2 < len(beats):
        return 0, []

    return np.mean(beatstds), avgbeat


def filter_abps(segx, SRATE=100):
    range_filter = True if ((segx > 20).all() & (segx < 200).all()) else False

    mstd_seg, avg_beat = abp_process_beat(segx)
    mstds_filter = True if mstd_seg > 0 else False

    return (range_filter & mstds_filter)


######################################
# 01. Build random selection dataset #
######################################

print('Start building random selection datasets')

for pred_win in PREDICTION_WINDOW:
    print("Building for prediction window of {}".format(str(int(pred_win))))

    x = np.empty((0, 6000), float)
    y = np.array([])
    m = np.empty((0, 30), float)
    c = np.array([])

    for caseid in tqdm.tqdm(eligible_caseids):

        print('Start processing {}'.format(caseid))

        if not os.path.exists(os.path.join(SAVE_PATH, 'random_selection_tmp')):
            os.makedirs(os.path.join(SAVE_PATH, 'random_selection_tmp'))

        tmp_filename = os.path.join(SAVE_PATH, 'random_selection_tmp', '{}min_pred_{}_np.tmp'.format(str(int(pred_win)), str(caseid).zfill(4)))

        if not os.path.exists(tmp_filename):

            vf = vitaldb.VitalFile(caseid, ['SNUADC/ART', 'Solar8000/ART_MBP'])
            vf_arts = vf.get_samples(['SNUADC/ART', 'Solar8000/ART_MBP'], interval=1 / SRATE)
            art_wav = vf_arts[0][0]
            art_mbp = vf_arts[0][1]

            # INPUT_SEGMENT_SIZE=1, SKIP_INTERVAL=1 / Get extraction index
            selection_arange = np.arange(SRATE * (SEGMENT_SIZE + pred_win) * 60,
                                         len(art_wav), 60 * (SKIP_INTERVAL) * SRATE)

            # select eligible wave
            eligible_wave_position = []
            for index in selection_arange:
                # wave is from  1min length segment of 5-min ahead waveform
                wave = art_wav[index - (SRATE * (SEGMENT_SIZE + pred_win) * 60):index - (SRATE * pred_win * 60)]

                valid = True

                if np.isnan(wave).mean() > 0.1:
                    valid = False
                elif (wave > 200).any():
                    valid = False
                elif (wave < 20).any():
                    valid = False
                elif np.max(wave) - np.min(wave) < 30:
                    valid = False
                elif (np.abs(np.diff(wave)) > 30).any():  # abrupt change -> noise
                    valid = False

                if valid:
                    eligible_wave_position.append(index)

            if len(eligible_wave_position) == 0:
                continue

            eligible_case_position = []
            eligible_control_position = []

            for index in eligible_wave_position:

                # select mbp segment
                mbp = art_mbp[index - (SRATE * CASE_DURATION * 60):index]
                mbp_notnull = mbp[~np.isnan(mbp)]

                # select eligible mbp
                valid = True
                if (mbp_notnull > 200).any():
                    valid = False
                elif (mbp_notnull < 30).any():
                    valid = False
                elif (np.abs(np.diff(mbp_notnull)) > 30).any():  # abrupt change -> noise
                    valid = False

                # assign case/control
                mbp_is_case = np.mean(mbp_notnull) <= 65.

                if valid & mbp_is_case:
                    eligible_case_position.append(index)

                if valid & (not mbp_is_case):
                    eligible_control_position.append(index)

            if len(eligible_case_position) == 0 and len(eligible_control_position) == 0:
                print('case dose not have eligible points, skip this case')
                continue

            # select case segments
            seg_x, seg_y, seg_mbp = [], [], []
            for case_idx in eligible_case_position:
                sample_x = art_wav[
                           case_idx - (SRATE * (SEGMENT_SIZE + pred_win) * 60):case_idx - (SRATE * pred_win * 60)]
                sample_y = 1.
                sample_mbp = art_mbp[
                             case_idx - (SRATE * (SEGMENT_SIZE + pred_win) * 60):case_idx - (SRATE * pred_win * 60)]
                sample_mbp_notnull = sample_mbp[~np.isnan(sample_mbp)]

                if len(sample_x) != 6000:
                    sample_x = np.array([np.nan] * 6000)
                if len(sample_mbp_notnull) != 30:
                    sample_mbp_notnull = np.array([np.nan] * 30)

                seg_x.append(sample_x)
                seg_y.append(sample_y)
                seg_mbp.append(sample_mbp_notnull)

            # select control segments
            for control_idx in eligible_control_position:
                sample_x = art_wav[
                           control_idx - (SRATE * (SEGMENT_SIZE + pred_win) * 60):control_idx - (SRATE * pred_win * 60)]
                sample_y = 0.
                sample_mbp = art_mbp[control_idx - (SRATE * (SEGMENT_SIZE + pred_win) * 60):control_idx - (
                            SRATE * pred_win * 60)]
                sample_mbp_notnull = sample_mbp[~np.isnan(sample_mbp)]

                if len(sample_x) != 6000:
                    sample_x = np.array([np.nan] * 6000)
                if len(sample_mbp_notnull) != 30:
                    sample_mbp_notnull = np.array([np.nan] * 30)

                seg_x.append(sample_x)
                seg_y.append(sample_y)
                seg_mbp.append(sample_mbp_notnull)

            seg_x_np = np.array(seg_x)
            seg_y_np = np.array(seg_y)
            seg_mbp_np = np.array(seg_mbp)
            seg_c_np = np.array([caseid] * len(seg_y))

            pickle.dump((seg_x_np, seg_y_np, seg_mbp_np, seg_c_np,), open(tmp_filename, 'wb'), protocol=4)

            if len(seg_x_np) == 0 or len(seg_y_np) == 0 or len(seg_mbp_np) == 0:
                continue
        else:
            seg_x_np, seg_y_np, seg_mbp_np, seg_c_np = pickle.load(open(tmp_filename, 'rb'))

            if len(seg_x_np) == 0 or len(seg_y_np) == 0 or len(seg_mbp_np) == 0:
                continue

        x = np.append(x, seg_x_np, axis=0)
        y = np.concatenate([y, seg_y_np])
        m = np.append(m, seg_mbp_np, axis=0)
        c = np.concatenate([c, seg_c_np])


    print("Start preprocessing of waveform")
    n_process = 20
    pool = multiprocessing.Pool(processes=n_process)
    filter_random_sample = pool.map(filter_abps, list(x))
    pool.close()
    pool.join()

    x = x[filter_random_sample]
    y = y[filter_random_sample]
    c = c[filter_random_sample]
    m = m[filter_random_sample]

    pickle.dump(x, open(
        os.path.join(SAVE_PATH, 'x_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)
    pickle.dump(y, open(
        os.path.join(SAVE_PATH, 'y_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)
    pickle.dump(m, open(
        os.path.join(SAVE_PATH, 'mbp_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)
    pickle.dump(c, open(
        os.path.join(SAVE_PATH, 'c_random_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)

    del (x)
    del (y)
    del (m)
    del (c)
    gc.collect()

    print('end building prediction window {}min dataset'.format(str(int(pred_win))))
    print(' ')

print('end building random sample dataset')
print(' ')
print(' ')



######################################
# 02. Build Biased selection dataset #
######################################

print('Start building hpi_style datasets')

for pred_win in PREDICTION_WINDOW:
    print("Building for prediction window of {}".format(str(int(pred_win))))

    x_hpi_style = np.empty((0, 6000), float)
    y_hpi_style = np.array([])
    mbp_hpi_style = np.empty((0, 30), float)
    c_hpi_style = np.array([])

    for caseid in tqdm.tqdm(eligible_caseids):

        print('Start processing {}'.format(caseid))

        if not os.path.join(SAVE_PATH, 'biased_selection_tmp'):
            os.makedirs(os.path.join(SAVE_PATH, 'biased_selection_tmp'))

        tmp_filename = os.path.join(SAVE_PATH, 'biased_selection_tmp', '{}min_pred_{}'.format(str(int(pred_win)), str(caseid).zfill(4)))

        if not os.path.exists(tmp_filename):

            vf = vitaldb.VitalFile(caseid, ['SNUADC/ART', 'Solar8000/ART_MBP'])
            vf_arts = vf.get_samples(['SNUADC/ART', 'Solar8000/ART_MBP'], interval=1 / SRATE)
            art_wav = vf_arts[0][0]
            art_mbp = vf_arts[0][1]

            # INPUT_SEGMENT_SIZE=1, SKIP_INTERVAL=1 / Get extraction index
            selection_arange = np.arange(SRATE * (SEGMENT_SIZE + pred_win) * 60,
                                         len(art_wav) - (SRATE * CONTROL_DURATION * 60), 60 * SKIP_INTERVAL * SRATE)

            # select eligible wave
            eligible_wave_position = []
            for index in selection_arange:
                # wave is from  1min length segment of 5-min ahead waveform
                wave = art_wav[index - (SRATE * (SEGMENT_SIZE + pred_win) * 60):index - (
                            SRATE * pred_win * 60)]

                valid = True

                if np.isnan(wave).mean() > 0.1:
                    valid = False
                elif (wave > 200).any():
                    valid = False
                elif (wave < 30).any():
                    valid = False
                elif np.max(wave) - np.min(wave) < 30:
                    valid = False
                elif (np.abs(np.diff(wave)) > 30).any():  # abrupt change -> noise
                    valid = False

                if valid:
                    eligible_wave_position.append(index)

            if len(eligible_wave_position) == 0:
                continue

            # select mbp segment of CASE event
            eligible_case_position = []
            for index in eligible_wave_position:

                # select recent CASE_DURATION of mbp for CASE
                mbp = art_mbp[index - (SRATE * CASE_DURATION * 60):index]
                mbp_notnull = mbp[~np.isnan(mbp)]

                # eligible mbp?
                valid = True
                if (mbp_notnull > 200).any():
                    valid = False
                elif (mbp_notnull < 30).any():
                    valid = False
                elif (np.abs(np.diff(mbp_notnull)) > 30).any():  # abrupt change -> noise
                    valid = False

                # assign case?
                mbp_is_case = np.mean(mbp_notnull) <= 65.

                if valid & mbp_is_case:
                    eligible_case_position.append(index)


            # select mbp segment of control, considering CASE event
            # mbp > 75mmHg lasting for 5min, 20min apart from CASE event
            eligible_control_position = []

            for index in eligible_wave_position:
                # select recent CONTROL_DURATION of mbp for CONTROL
                mbp = art_mbp[index:index + (SRATE * CONTROL_DURATION * 60)]
                mbp_notnull = mbp[~np.isnan(mbp)]

                # elibile mbp?
                valid = True
                if (mbp_notnull > 200).any():
                    valid = False
                elif (mbp_notnull < 30).any():
                    valid = False
                elif (np.abs(np.diff(mbp_notnull)) > 30).any():  # abrupt change -> noise
                    valid = False

                # assign control?
                mbp_is_normotension = np.array(mbp_notnull >= 75).all()

                if len(eligible_case_position) == 0:
                    mbp_is_control = True
                else:
                    gap_with_case = np.array(eligible_case_position) - index
                    # consider 21min ahead, and 25min behind, which considering 20min apart rule and length of input segment
                    mbp_is_apart_20min = ((gap_with_case > (SRATE * (CONTROL_DURATION + CONTROL_APART) * 60)) | (
                                gap_with_case < -(SRATE * (CASE_DURATION + CONTROL_APART) * 60))).all()

                    mbp_is_control = mbp_is_normotension & mbp_is_apart_20min

                if valid & mbp_is_control:
                    eligible_control_position.append(index)


            # select case segments
            seg_x, seg_y, seg_mbp = [], [], []
            for case_idx in eligible_case_position:
                sample_x = art_wav[case_idx - (SRATE * (SEGMENT_SIZE + pred_win) * 60):case_idx - (
                            SRATE * pred_win * 60)]
                sample_y = 1.
                sample_mbp = art_mbp[case_idx - (SRATE * (SEGMENT_SIZE + pred_win) * 60):case_idx - (
                            SRATE * pred_win * 60)]
                sample_mbp_notnull = sample_mbp[~np.isnan(sample_mbp)]

                if len(sample_x) != 6000:
                    sample_x = np.array([np.nan] * 6000)
                if len(sample_mbp_notnull) != 30:
                    sample_mbp_notnull = np.array([np.nan] * 30)

                seg_x.append(sample_x)
                seg_y.append(sample_y)
                seg_mbp.append(sample_mbp_notnull)

            # select control segment
            for control_idx in eligible_control_position:
                # extract the center point of 5-min control segment, index + 2.5min - 0.5min ~ index + 2.5min + 0.5min
                sample_x = art_wav[
                           control_idx + int(SRATE * (pred_win / 2 - SEGMENT_SIZE / 2) * 60):control_idx + int(
                               SRATE * (pred_win / 2 + SEGMENT_SIZE / 2) * 60)]
                sample_y = 0.
                sample_mbp = art_mbp[control_idx + int(
                    SRATE * (pred_win / 2 - SEGMENT_SIZE / 2) * 60):control_idx + int(
                    SRATE * (pred_win / 2 + SEGMENT_SIZE / 2) * 60)]
                sample_mbp_notnull = sample_mbp[~np.isnan(sample_mbp)]

                if len(sample_x) != 6000:
                    sample_x = np.array([np.nan] * 6000)
                if len(sample_mbp_notnull) != 30:
                    sample_mbp_notnull = np.array([np.nan] * 30)

                seg_x.append(sample_x)
                seg_y.append(sample_y)
                seg_mbp.append(sample_mbp_notnull)

            seg_x_np = np.array(seg_x)
            seg_y_np = np.array(seg_y)
            seg_mbp_np = np.array(seg_mbp)
            seg_c_np = np.array([caseid] * len(seg_y))

            pickle.dump((seg_x_np, seg_y_np, seg_mbp_np, seg_c_np,), open(tmp_filename, 'wb'), protocol=4)

            if len(seg_x_np) == 0 or len(seg_y_np) == 0 or len(seg_mbp_np) == 0:
                continue

        else:
            seg_x_np, seg_y_np, seg_mbp_np, seg_c_np = pickle.load(open(tmp_filename, 'rb'))

            if len(seg_x_np) == 0 or len(seg_y_np) == 0 or len(seg_mbp_np) == 0:
                continue

        x_hpi_style = np.append(x_hpi_style, seg_x_np, axis=0)
        y_hpi_style = np.concatenate([y_hpi_style, seg_y_np])
        mbp_hpi_style = np.append(mbp_hpi_style, seg_mbp_np, axis=0)
        c_hpi_style = np.concatenate([c_hpi_style, seg_c_np])

    print("Start preprocessing of waveform")
    n_process = 20
    pool = multiprocessing.Pool(processes=n_process)
    filter_random_sample = pool.map(filter_abps, list(x_hpi_style))
    pool.close()
    pool.join()

    x_hpi_style = x_hpi_style[filter_random_sample]
    y_hpi_style = y_hpi_style[filter_random_sample]
    c_hpi_style = c_hpi_style[filter_random_sample]
    mbp_hpi_style = mbp_hpi_style[filter_random_sample]

    pickle.dump(x_hpi_style, open(
        os.path.join(SAVE_PATH, 'x_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)
    pickle.dump(y_hpi_style, open(
        os.path.join(SAVE_PATH, 'y_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)
    pickle.dump(mbp_hpi_style, open(
        os.path.join(SAVE_PATH, 'mbp_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)
    pickle.dump(c_hpi_style, open(
        os.path.join(SAVE_PATH, 'c_biased_selection_{}min_pred.np'.format(str(int(pred_win)))), 'wb'), protocol=4)

    del (x_hpi_style)
    del (y_hpi_style)
    del (mbp_hpi_style)
    del (c_hpi_style)
    gc.collect()

    print('end building prediction window {}min dataset'.format(str(int(pred_win))))
    print(' ')

print('end building hpi style dataset')
print(' ')
print(' ')

