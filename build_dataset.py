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

from typing import Dict, Tuple, Optional
import glob, re


# Define saving path
SAVE_PATH = "./dataset/"
if not os.path.exists(SAVE_PATH):
    os.mkdir(SAVE_PATH)

# Subfolders
UNBIASED_TMP = os.path.join(SAVE_PATH, "random_selection_tmp")
BIASED_TMP   = os.path.join(SAVE_PATH, "biased_selection_tmp")
os.makedirs(UNBIASED_TMP, exist_ok=True)
os.makedirs(BIASED_TMP,   exist_ok=True)

# Define hyperparameters
PREDICTION_WINDOW = [5]
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

#=====Added Helpers============

# Check already done cases
def already_done_cases(tmp_dir, pred_min):
    done = set()
    for f in glob.glob(os.path.join(tmp_dir, f"{pred_min}min_*.pkl")):
        m = re.search(rf"{pred_min}min_(\d+)\.pkl$", os.path.basename(f))
        if m: done.add(int(m.group(1)))
    return done

# Cohort assignment helpers
def _get_col(row: pd.Series, names) -> Optional[float]:
    for n in names:
        if n in row and pd.notna(row[n]):
            return row[n]
    return None

def get_case_meta(case_id: int) -> Dict:
    row = case_info_pd.loc[case_info_pd["caseid"] == case_id]
    row = row.iloc[0] if len(row) else pd.Series(dtype=object)

    asa = _get_col(row, ["asa", "ASA", "asa_class", "asaclass"])
    # In VitalDB, emergency often appears as 'emop' (0/1). Fall back to other aliases if needed.
    emergency = _get_col(row, ["emop", "emergency", "is_emergency"])
    # Duration in minutes: if not in metadata, we’ll compute from waveform length lazily (see below)
    duration_meta = _get_col(row, ["opdur", "duration_min", "duration"])

    return {"asa": asa, "emergency": emergency, "duration_meta": duration_meta}

def assign_cohort(asa: Optional[float], emergency: Optional[int], duration_min: float) -> str:
    """A: clean (ASA<=2, non-emergency, dur>=60); B: inclusive (adds ASA>=3/emergency); C: real-world else."""
    em = int(emergency) if emergency is not None and not pd.isna(emergency) else 0
    asai = int(asa) if asa is not None and not pd.isna(asa) else 3  # if missing, treat as higher risk
    if (asai <= 2) and (em == 0) and (duration_min >= 60.0):
        return "A"  # selected/clean
    if (asai >= 3) or (em == 1):
        return "B"  # inclusive (higher risk and/or emergency)
    return "C"      # remaining adult cases (real-world)

def _append_buffers(buffers: Dict[str, Dict[str, list]],
                    cohort: str,
                    x, y, mbp, case_ids, asa_tags, emop_tags):
    # Accumulate into per-cohort and pooled buffers
    for key, arr_ in [("x", x), ("y", y), ("mbp", mbp), ("case", case_ids), ("asa", asa_tags), ("emop", emop_tags)]:
        buffers["pooled"][key].append(arr_)
        buffers[cohort][key].append(arr_)

def _init_buffers() -> Dict[str, Dict[str, list]]:
    def empty():
        return {"x": [], "y": [], "mbp": [], "case": [], "asa": [], "emop": []}
    return {"pooled": empty(), "A": empty(), "B": empty(), "C": empty()}

def _finalize_and_save(buffers: Dict[str, Dict[str, list]], pred_min: int, tag: str):
    # Stack numpy arrays and save per cohort + pooled
    for cohort in ["pooled", "A", "B", "C"]:
        if len(buffers[cohort]["y"]) == 0:
            continue
        X = np.vstack(buffers[cohort]["x"])
        Y = np.concatenate(buffers[cohort]["y"])
        M = np.vstack(buffers[cohort]["mbp"])
        C = np.concatenate(buffers[cohort]["case"])
        ASA = np.concatenate(buffers[cohort]["asa"])
        EM  = np.concatenate(buffers[cohort]["emop"])

        suffix = f"{tag}_{pred_min}min_{cohort}"
        pickle.dump(X,   open(os.path.join(SAVE_PATH, f"x_{suffix}.np"),   "wb"), protocol=4)
        pickle.dump(Y,   open(os.path.join(SAVE_PATH, f"y_{suffix}.np"),   "wb"), protocol=4)
        pickle.dump(M,   open(os.path.join(SAVE_PATH, f"mbp_{suffix}.np"), "wb"), protocol=4)
        pickle.dump(C,   open(os.path.join(SAVE_PATH, f"c_{suffix}.np"),   "wb"), protocol=4)
        pickle.dump(ASA, open(os.path.join(SAVE_PATH, f"asa_{suffix}.np"), "wb"), protocol=4)
        pickle.dump(EM,  open(os.path.join(SAVE_PATH, f"emop_{suffix}.np"),"wb"), protocol=4)

def _case_duration_minutes_from_wave(art_wav_len: int) -> float:
    # duration of whole waveform in minutes (at SRATE Hz)
    return art_wav_len / SRATE / 60.0


######################################
# 01. Build random selection dataset #
######################################

# Modified into functions with added helpers
def build_unbiased_random(pred_min: int, case_summary_rows: list):
    print(f"[UNBIASED] Building for prediction window = {pred_min} min")
    buffers = _init_buffers()

    done = already_done_cases(UNBIASED_TMP, pred_min)   # or BIASED_TMP
    for caseid in tqdm.tqdm([c for c in eligible_caseids if c not in done]):
        tmp_filename = os.path.join(UNBIASED_TMP, f"{pred_min}min_{str(caseid).zfill(4)}.pkl")
        if not os.path.exists(tmp_filename):
            try:
                vf = vitaldb.VitalFile(caseid, ["SNUADC/ART", "Solar8000/ART_MBP"])
                wav, mbp = vf.get_samples(["SNUADC/ART", "Solar8000/ART_MBP"], interval=1/SRATE)[0]
            except Exception:
                continue

            # unbiased selection: scan timeline with stride SKIP_INTERVAL, take 1-minute window ending pred_min before index
            selection_arange = np.arange(SRATE * (SEGMENT_SIZE + pred_min) * 60,
                                         len(wav),
                                         SRATE * 60 * SKIP_INTERVAL)

            # eligible wave positions after basic QC
            eligible_positions = []
            for idx in selection_arange:
                wave = wav[idx - (SRATE * (SEGMENT_SIZE + pred_min) * 60) : idx - (SRATE * pred_min * 60)]
                valid = True
                if np.isnan(wave).mean() > 0.1: valid = False
                elif (wave > 200).any():        valid = False
                elif (wave < 20).any():         valid = False
                elif np.max(wave) - np.min(wave) < 30: valid = False
                elif (np.abs(np.diff(wave)) > 30).any(): valid = False
                if valid: eligible_positions.append(idx)
            if len(eligible_positions) == 0:
                continue

            # case/control labeling by recent 1-min MBP mean
            case_pos, ctrl_pos = [], []
            for idx in eligible_positions:
                seg_mbp = mbp[idx - (SRATE * CASE_DURATION * 60): idx]
                seg_mbp = seg_mbp[~np.isnan(seg_mbp)]
                if len(seg_mbp) == 0:  # skip empty
                    continue
                if (seg_mbp > 200).any() or (seg_mbp < 30).any() or (np.abs(np.diff(seg_mbp)) > 30).any():
                    continue
                is_case = (np.mean(seg_mbp) <= 65.0)
                (case_pos if is_case else ctrl_pos).append(idx)

            seg_x, seg_y, seg_m = [], [], []
            for idx in case_pos + ctrl_pos:
                x = wav[idx - (SRATE * (SEGMENT_SIZE + pred_min) * 60) : idx - (SRATE * pred_min * 60)]
                y = 1.0 if idx in case_pos else 0.0
                m = mbp[idx - (SRATE * (SEGMENT_SIZE + pred_min) * 60) : idx - (SRATE * pred_min * 60)]
                m = m[~np.isnan(m)]
                if len(x) != 6000: x = np.full(6000, np.nan)
                if len(m) != 30:   m = np.full(30,   np.nan)
                seg_x.append(x); seg_y.append(y); seg_m.append(m)

            if len(seg_x) == 0:
                continue

            # cohort metadata
            meta = get_case_meta(caseid)
            duration_min = meta["duration_meta"] if meta["duration_meta"] is not None else _case_duration_minutes_from_wave(len(wav))
            cohort = assign_cohort(meta["asa"], meta["emergency"], float(duration_min))
            asa_tag = int(meta["asa"]) if meta["asa"] is not None and not pd.isna(meta["asa"]) else -1
            em_tag  = int(meta["emergency"]) if meta["emergency"] is not None and not pd.isna(meta["emergency"]) else 0

            # persist tmp for restartability
            payload = (np.array(seg_x), np.array(seg_y), np.array(seg_m), np.array([caseid]*len(seg_y)),
                       np.full(len(seg_y), asa_tag), np.full(len(seg_y), em_tag), cohort, duration_min)
            pickle.dump(payload, open(tmp_filename, "wb"), protocol=4)
        else:
            try:
                payload = pickle.load(open(tmp_filename, "rb"))
            except Exception:
                continue

        seg_x_np, seg_y_np, seg_m_np, seg_c_np, seg_asa_np, seg_em_np, cohort, duration_min = payload

        # waveform QC by beats
        nproc = min(20, os.cpu_count() or 4)
        with multiprocessing.Pool(processes=nproc) as pool:
            keep = pool.map(filter_abps, list(seg_x_np))
        keep = np.array(keep, dtype=bool)
        if not keep.any():
            continue

        X = seg_x_np[keep]; Y = seg_y_np[keep]; M = seg_m_np[keep]
        C = seg_c_np[keep];  ASA = seg_asa_np[keep]; EM = seg_em_np[keep]

        # append to buffers (pooled + cohort)
        _append_buffers(buffers, cohort, X, Y, M, C, ASA, EM)

        # add 1 case-level summary row
        case_summary_rows.append({
            "caseid": int(C[0]),
            "cohort": cohort,
            "asa":    int(ASA[0]),
            "emop":   int(EM[0]),
            "duration_min": float(duration_min),
            "n_samples": int(len(Y)),
            "pos_rate": float(Y.mean()) if len(Y) else np.nan,
            "builder": "unbiased",
            "pred_min": pred_min
        })

    # save per-cohort + pooled
    _finalize_and_save(buffers, pred_min, tag="unbiased")


######################################
# 02. Build Biased selection dataset #
######################################

# Modified into functions with added helpers
def build_biased_hpi_style(pred_min: int, case_summary_rows: list):
    print(f"[BIASED] Building for prediction window = {pred_min} min")
    buffers = _init_buffers()

    done = already_done_cases(BIASED_TMP, pred_min)
    for caseid in tqdm.tqdm([c for c in eligible_caseids if c not in done]):
        tmp_filename = os.path.join(BIASED_TMP, f"{pred_min}min_{str(caseid).zfill(4)}.pkl")
        if not os.path.exists(tmp_filename):
            try:
                vf = vitaldb.VitalFile(caseid, ["SNUADC/ART", "Solar8000/ART_MBP"])
                wav, mbp = vf.get_samples(["SNUADC/ART", "Solar8000/ART_MBP"], interval=1/SRATE)[0]
            except Exception:
                continue

            selection_arange = np.arange(SRATE * (SEGMENT_SIZE + pred_min) * 60,
                                         len(wav) - (SRATE * CONTROL_DURATION * 60),
                                         SRATE * 60 * SKIP_INTERVAL)

            eligible_positions = []
            for idx in selection_arange:
                wave = wav[idx - (SRATE * (SEGMENT_SIZE + pred_min) * 60) : idx - (SRATE * pred_min * 60)]
                valid = True
                if np.isnan(wave).mean() > 0.1: valid = False
                elif (wave > 200).any():        valid = False
                elif (wave < 30).any():         valid = False
                elif np.max(wave) - np.min(wave) < 30: valid = False
                elif (np.abs(np.diff(wave)) > 30).any(): valid = False
                if valid: eligible_positions.append(idx)
            if len(eligible_positions) == 0:
                continue

            # CASE positions by recent 1-min mbp
            case_pos = []
            for idx in eligible_positions:
                seg_mbp = mbp[idx - (SRATE * CASE_DURATION * 60): idx]
                seg_mbp = seg_mbp[~np.isnan(seg_mbp)]
                if len(seg_mbp) == 0: continue
                if (seg_mbp > 200).any() or (seg_mbp < 30).any() or (np.abs(np.diff(seg_mbp)) > 30).any():
                    continue
                if (np.mean(seg_mbp) <= 65.0):
                    case_pos.append(idx)

            # CONTROL positions: MBP >= 75 for CONTROL_DURATION and apart >= 20 min from any case
            ctrl_pos = []
            for idx in eligible_positions:
                seg = mbp[idx: idx + (SRATE * CONTROL_DURATION * 60)]
                seg = seg[~np.isnan(seg)]
                if len(seg) == 0: continue
                valid = True
                if (seg > 200).any() or (seg < 30).any() or (np.abs(np.diff(seg)) > 30).any():
                    valid = False
                is_normo = bool((seg >= 75).all())
                if len(case_pos) == 0:
                    is_apart = True
                else:
                    gaps = np.array(case_pos) - idx
                    is_apart = ((gaps > (SRATE * (CONTROL_DURATION + CONTROL_APART) * 60)) |
                                (gaps < -(SRATE * (CASE_DURATION + CONTROL_APART) * 60))).all()
                is_ctrl = valid and is_normo and is_apart
                if is_ctrl:
                    ctrl_pos.append(idx)

            seg_x, seg_y, seg_m = [], [], []
            # cases
            for idx in case_pos:
                x = wav[idx - (SRATE * (SEGMENT_SIZE + pred_min) * 60) : idx - (SRATE * pred_min * 60)]
                y = 1.0
                m = mbp[idx - (SRATE * (SEGMENT_SIZE + pred_min) * 60) : idx - (SRATE * pred_min * 60)]
                m = m[~np.isnan(m)]
                if len(x) != 6000: x = np.full(6000, np.nan)
                if len(m) != 30:   m = np.full(30,   np.nan)
                seg_x.append(x); seg_y.append(y); seg_m.append(m)
            # controls (centered)
            for idx in ctrl_pos:
                lo = idx + int(SRATE * (pred_min/2 - SEGMENT_SIZE/2) * 60)
                hi = idx + int(SRATE * (pred_min/2 + SEGMENT_SIZE/2) * 60)
                x = wav[lo:hi]
                y = 0.0
                m = mbp[lo:hi]
                m = m[~np.isnan(m)]
                if len(x) != 6000: x = np.full(6000, np.nan)
                if len(m) != 30:   m = np.full(30,   np.nan)
                seg_x.append(x); seg_y.append(y); seg_m.append(m)

            if len(seg_x) == 0:
                continue

            # cohort meta
            meta = get_case_meta(caseid)
            duration_min = meta["duration_meta"] if meta["duration_meta"] is not None else _case_duration_minutes_from_wave(len(wav))
            cohort = assign_cohort(meta["asa"], meta["emergency"], float(duration_min))
            asa_tag = int(meta["asa"]) if meta["asa"] is not None and not pd.isna(meta["asa"]) else -1
            em_tag  = int(meta["emergency"]) if meta["emergency"] is not None and not pd.isna(meta["emergency"]) else 0

            payload = (np.array(seg_x), np.array(seg_y), np.array(seg_m), np.array([caseid]*len(seg_y)),
                       np.full(len(seg_y), asa_tag), np.full(len(seg_y), em_tag), cohort, duration_min)
            pickle.dump(payload, open(tmp_filename, "wb"), protocol=4)
        else:
            try:
                payload = pickle.load(open(tmp_filename, "rb"))
            except Exception:
                continue

        seg_x_np, seg_y_np, seg_m_np, seg_c_np, seg_asa_np, seg_em_np, cohort, duration_min = payload

        # waveform QC
        nproc = min(20, os.cpu_count() or 4)
        with multiprocessing.Pool(processes=nproc) as pool:
            keep = pool.map(filter_abps, list(seg_x_np))
        keep = np.array(keep, dtype=bool)
        if not keep.any():
            continue

        X = seg_x_np[keep]; Y = seg_y_np[keep]; M = seg_m_np[keep]
        C = seg_c_np[keep];  ASA = seg_asa_np[keep]; EM = seg_em_np[keep]

        _append_buffers(buffers, cohort, X, Y, M, C, ASA, EM)

        case_summary_rows.append({
            "caseid": int(C[0]),
            "cohort": cohort,
            "asa":    int(ASA[0]),
            "emop":   int(EM[0]),
            "duration_min": float(duration_min),
            "n_samples": int(len(Y)),
            "pos_rate": float(Y.mean()) if len(Y) else np.nan,
            "builder": "biased",
            "pred_min": pred_min
        })

    _finalize_and_save(buffers, pred_min, tag="biased")

if __name__ == "__main__":
    case_summary_rows = []
    for pred in PREDICTION_WINDOW:
        build_unbiased_random(pred, case_summary_rows)
        gc.collect()
        build_biased_hpi_style(pred, case_summary_rows)
        gc.collect()

    # Write a case-level summary for reporting bias/imbalance
    if len(case_summary_rows):
        summary_df = pd.DataFrame(case_summary_rows)
        summary_df.to_csv(os.path.join(SAVE_PATH, "cohort_case_summary.csv"), index=False)
        print("Wrote per-case summary to dataset/cohort_case_summary.csv")

    print("Done.")