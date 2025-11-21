import os
import pickle
import numpy as np
import matplotlib.pylab as plt
import pandas as pd

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    classification_report,
    auc,
    accuracy_score,
    precision_recall_curve,
)
from sklearn.utils import class_weight

import utils as ut
import csv
import itertools


def evaluate(model_predict, y_test, plot_directory):
    """
    Compute AUROC, AUPRC (+ bootstrap CIs) and save ROC/PR curves and a
    sklearn classification_report to plot_directory.
    """
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    model_result_inclass = np.greater(model_predict, 0.5)

    with open(os.path.join(plot_directory, "classification_report.txt"), "w") as f:
        print(
            classification_report(
                y_test, model_result_inclass, target_names=["normal", "lowbp"]
            ),
            file=f,
        )

    # AUROC
    fpr, tpr, thresh = roc_curve(y_test, model_predict)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.text(0.8, 0.1, str(round(roc_auc, 4)), fontsize=12)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))

    plt.savefig(os.path.join(plot_directory, "roc_curve.png"))
    plt.close()

    # AUPRC
    precision, recall, thresh_pr = precision_recall_curve(y_test, model_predict)
    prc_auc = auc(recall, precision)

    plt.plot(recall, precision)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.text(0.1, 0.1, str(round(prc_auc, 4)), fontsize=12)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
    plt.savefig(os.path.join(plot_directory, "prc_curve.png"))
    plt.close()

    # Bootstrap confidence intervals
    n_bootstraps = 1000
    rng_seed = 77
    bootstrapped_auroc_scores = []
    bootstrapped_auprc_scores = []

    rng = np.random.RandomState(rng_seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(model_predict), len(model_predict))
        if len(np.unique(y_test[indices])) < 2:
            # need both classes for ROC/AUPRC
            continue

        fpr_b, tpr_b, _ = roc_curve(y_test[indices], model_predict[indices])
        auroc_score = auc(fpr_b, tpr_b)

        precision_b, recall_b, _ = precision_recall_curve(
            y_test[indices], model_predict[indices]
        )
        auprc_score = auc(recall_b, precision_b)

        bootstrapped_auroc_scores.append(auroc_score)
        bootstrapped_auprc_scores.append(auprc_score)

    sorted_auroc_scores = np.array(bootstrapped_auroc_scores)
    sorted_auroc_scores.sort()
    sorted_auprc_scores = np.array(bootstrapped_auprc_scores)
    sorted_auprc_scores.sort()

    auroc_confidence_lower = sorted_auroc_scores[
        int(0.05 * len(sorted_auroc_scores))
    ]
    auroc_confidence_upper = sorted_auroc_scores[
        int(0.95 * len(sorted_auroc_scores))
    ]

    auprc_confidence_lower = sorted_auprc_scores[
        int(0.05 * len(sorted_auprc_scores))
    ]
    auprc_confidence_upper = sorted_auprc_scores[
        int(0.95 * len(sorted_auprc_scores))
    ]

    with open(os.path.join(plot_directory, "scores_with_ci.txt"), "w") as f:
        f.write("auroc score : {:0.4f}".format(roc_auc))
        f.write("\n")
        f.write(
            "Confidence interval for the auroc score: [{:0.4f} - {:0.4f}]".format(
                auroc_confidence_lower, auroc_confidence_upper
            )
        )
        f.write("\n\n")
        f.write("auprc score : {:0.4f}".format(prc_auc))
        f.write("\n")
        f.write(
            "Confidence interval for the auprc score: [{:0.4f} - {:0.4f}]".format(
                auprc_confidence_lower, auprc_confidence_upper
            )
        )

    return None


def training_and_testing(
    data_selection_method,
    fixed_test_case_path,
    pred_window,
    sampling_rate,
    return_type,
    model_type,
    output_dir,
    cohort="pooled",
    remove_under65_in_train=True,
    filter_nan=True,
):
    """
    Train and evaluate a model for a given:
      - data_selection_method: 'unbiased' or 'biased' (or legacy names if utils maps them)
      - cohort: 'A', 'B', 'C', or 'pooled'
      - pred_window: prediction horizon in minutes (e.g. 5)
      - sampling_rate: subsampling fraction of train set (1.0 = all)
      - return_type: 'wave' (CNN/Transformer) or 'mbp' (LR/LSTM)
    """

    # subdir name: e.g. 'unbiased_A_predwin_5_sampling_1'
    subdirectory = "{}_{}_predwin_{}_sampling_{}".format(
        data_selection_method,
        cohort,
        int(pred_window),
        str(sampling_rate).replace(".", ""),
    )

    ########################################
    # test_on_same_dataset_used_in_training
    ########################################

    # data load (uses your modified utils.py with cohort support)
    x_train, y_train, c_train, x_test, y_test, c_test = (
        ut.load_mbp_or_wave_dataset_with_sampling(
            data_selection_type=data_selection_method,
            testset_path=fixed_test_case_path,
            pred_win=pred_window,
            sampling_rate=sampling_rate,
            return_type=return_type,
            cohort=cohort,
            remove_under65_in_train=remove_under65_in_train,
            filter_nan=filter_nan,
        )
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ------------------------------
    # Deep models: CNN / Transformer
    # ------------------------------
    if model_type in ["cnn", "transformer"]:
        if return_type != "wave":
            raise ValueError(
                "return_type should be 'wave' when model_type is cnn or transformer"
            )

        BATCH_SIZE = 256
        EPOCH = 100

        if model_type == "cnn":
            model = ut.load_cnn_model_architecture(sample_x=x_test)
        elif model_type == "transformer":
            model = ut.load_transformer_model_architecture(sample_x=x_test)
        else:
            raise ValueError("model_type should be 'cnn' or 'transformer'")

        run_dir = os.path.join(output_dir, subdirectory)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        output_tmp = os.path.join(run_dir, "model.hdf5")

        class_weights_arr = class_weight.compute_class_weight(
            "balanced", classes=np.unique(y_train), y=y_train
        )
        class_weights_dict = {i: class_weights_arr[i] for i in range(2)}

        callback_functions = [
            ModelCheckpoint(
                monitor="val_loss",
                filepath=output_tmp,
                verbose=1,
                save_best_only=True,
            ),
            EarlyStopping(
                monitor="val_loss", patience=50, verbose=1, mode="auto"
            ),
        ]

        train_history = model.fit(
            x_train,
            y_train,
            epochs=EPOCH,
            batch_size=BATCH_SIZE,
            verbose=1,
            validation_split=0.2,
            callbacks=callback_functions,
            class_weight=class_weights_dict,
        )

        # training curves
        plt.plot(train_history.history["accuracy"])
        plt.plot(train_history.history["val_accuracy"])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epoch")
        plt.legend(["train", "valid"], loc="upper left")
        plt.savefig(os.path.join(run_dir, "model_accuracy_history.png"))
        plt.close()

        plt.plot(train_history.history["loss"])
        plt.plot(train_history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "valid"], loc="upper left")
        plt.savefig(os.path.join(run_dir, "model_loss_history.png"))
        plt.close()

        # eval on same dataset
        model.load_weights(output_tmp)
        model_result = model.predict(x_test)

        pickle.dump(
            y_test, open(os.path.join(run_dir, "ground_truth.np"), "wb"), protocol=4
        )
        pickle.dump(
            model_result,
            open(os.path.join(run_dir, "inference.np"), "wb"),
            protocol=4,
        )

        evaluate(
            model_predict=model_result,
            y_test=y_test,
            plot_directory=os.path.join(
                run_dir, "test_on_same_dataset_used_in_train"
            ),
        )

        ########################################
        # test_on_unbiased_pooled_dataset
        # (generalisation to unbiased real-world)
        ########################################
        dir_gt_test = "test_on_unbiased_pooled"
        ext_test_dir = os.path.join(run_dir, dir_gt_test)
        if not os.path.exists(ext_test_dir):
            os.makedirs(ext_test_dir)

        # ALWAYS test on unbiased, pooled set for comparability
        x_test_gt, y_test_gt, c_test_gt = ut.load_mbp_or_wave_testing_dataset(
            testset_path=fixed_test_case_path,
            pred_win=pred_window,
            return_type=return_type,
            data_selection_type="unbiased",
            cohort="pooled",
            filter_nan=True,
        )

        model_gt_result = model.predict(x_test_gt)
        pickle.dump(
            model_gt_result,
            open(os.path.join(ext_test_dir, "inference.np"), "wb"),
            protocol=4,
        )

        evaluate(
            model_predict=model_gt_result,
            y_test=y_test_gt,
            plot_directory=ext_test_dir,
        )

    # ------------------------------
    # Simpler models: LR / LSTM (MBP)
    # ------------------------------
    elif model_type in ["lr", "lstm"]:
        if return_type != "mbp":
            raise ValueError(
                "return_type should be 'mbp' when model_type is LR or LSTM"
            )

        # we treat x_* as MBP series here (as in Yang)
        delta_x_train = np.diff(x_train)
        delta_2_x_train = np.diff(delta_x_train)

        delta_x_test = np.diff(x_test)
        delta_2_x_test = np.diff(delta_x_test)

        run_dir = os.path.join(output_dir, subdirectory)
        if not os.path.exists(run_dir):
            os.makedirs(run_dir)

        if model_type == "lr":
            # simple feature engineering: means of MBP and its first/second differences
            mean_x_train = np.mean(x_train, axis=1)
            mean_delta_x_train = np.mean(delta_x_train, axis=1)
            mean_delta_2_x_train = np.mean(delta_2_x_train, axis=1)

            mean_x_test = np.mean(x_test, axis=1)
            mean_delta_x_test = np.mean(delta_x_test, axis=1)
            mean_delta_2_x_test = np.mean(delta_2_x_test, axis=1)

            features_train = pd.DataFrame(
                {
                    "mbp": mean_x_train,
                    "delta_mbp": mean_delta_x_train,
                    "delta2_mbp": mean_delta_2_x_train,
                }
            )
            target_train = y_train

            features_test = pd.DataFrame(
                {
                    "mbp": mean_x_test,
                    "delta_mbp": mean_delta_x_test,
                    "delta2_mbp": mean_delta_2_x_test,
                }
            )

            scaler = StandardScaler()
            features_train = scaler.fit_transform(features_train)
            features_test = scaler.transform(features_test)

            model = LogisticRegression()
            model.fit(features_train, target_train)

            model_result = model.predict_proba(features_test)[:, 1]

            pickle.dump(
                model,
                open(os.path.join(run_dir, "lr_model.sav"), "wb"),
                protocol=4,
            )
            pickle.dump(
                y_test,
                open(os.path.join(run_dir, "ground_truth.np"), "wb"),
                protocol=4,
            )
            pickle.dump(
                model_result,
                open(os.path.join(run_dir, "lr_inference.np"), "wb"),
                protocol=4,
            )
            pickle.dump(
                scaler,
                open(os.path.join(run_dir, "scaler.pkl"), "wb"),
                protocol=4,
            )

            ########################################
            # test_on_unbiased_pooled_dataset (LR)
            ########################################
            dir_gt_test = "test_on_unbiased_pooled"
            ext_test_dir = os.path.join(run_dir, dir_gt_test)
            if not os.path.exists(ext_test_dir):
                os.makedirs(ext_test_dir)

            x_test_gt, y_test_gt, c_test_gt = ut.load_mbp_or_wave_testing_dataset(
                testset_path=fixed_test_case_path,
                pred_win=pred_window,
                return_type="mbp",
                data_selection_type="unbiased",
                cohort="pooled",
                remove_under65_in_test=True,
                filter_nan=True,
            )

            delta_x_test_gt = np.diff(x_test_gt)
            delta_2_x_test_gt = np.diff(delta_x_test_gt)

            mean_x_test_gt = np.mean(x_test_gt, axis=1)
            mean_delta_x_test_gt = np.mean(delta_x_test_gt, axis=1)
            mean_delta_2_x_test_gt = np.mean(delta_2_x_test_gt, axis=1)

            features_test_gt = pd.DataFrame(
                {
                    "mbp": mean_x_test_gt,
                    "delta_mbp": mean_delta_x_test_gt,
                    "delta2_mbp": mean_delta_2_x_test_gt,
                }
            )
            features_test_gt = scaler.transform(features_test_gt)

            model_gt_result = model.predict_proba(features_test_gt)[:, 1]

            pickle.dump(
                model_gt_result,
                open(os.path.join(ext_test_dir, "inference.np"), "wb"),
                protocol=4,
            )

            evaluate(
                model_predict=model_gt_result,
                y_test=y_test_gt,
                plot_directory=ext_test_dir,
            )

        elif model_type == "lstm":
            # build 3-channel time series: MBP + 1st diff + 2nd diff
            delta_x_train_reshape = np.insert(delta_x_train, 0, 0, axis=1)
            delta_2_x_train_tmp = np.insert(delta_2_x_train, 0, 0, axis=1)
            delta_2_x_train_reshape = np.insert(delta_2_x_train_tmp, 0, 0, axis=1)

            delta_x_test_reshape = np.insert(delta_x_test, 0, 0, axis=1)
            delta_2_x_test_tmp = np.insert(delta_2_x_test, 0, 0, axis=1)
            delta_2_x_test_reshape = np.insert(delta_2_x_test_tmp, 0, 0, axis=1)

            x_train_total = np.stack(
                [x_train, delta_x_train_reshape, delta_2_x_train_reshape], 2
            )
            x_test_total = np.stack(
                [x_test, delta_x_test_reshape, delta_2_x_test_reshape], 2
            )

            BATCH_SIZE = 256
            EPOCH = 100

            model = ut.load_lstm_model_architecture_3channel(
                sample_x=x_test_total
            )
            output_tmp = os.path.join(run_dir, "weights.hdf5")

            class_weights_arr = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            class_weights_dict = {i: class_weights_arr[i] for i in range(2)}

            callback_functions = [
                ModelCheckpoint(
                    monitor="val_loss",
                    filepath=output_tmp,
                    verbose=1,
                    save_best_only=True,
                ),
                EarlyStopping(
                    monitor="val_loss", patience=50, verbose=1, mode="auto"
                ),
            ]

            train_history = model.fit(
                x_train_total,
                y_train,
                epochs=EPOCH,
                batch_size=BATCH_SIZE,
                verbose=1,
                validation_split=0.2,
                callbacks=callback_functions,
                class_weight=class_weights_dict,
            )

            # training curves
            plt.plot(train_history.history["accuracy"])
            plt.plot(train_history.history["val_accuracy"])
            plt.title("model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "valid"], loc="upper left")
            plt.savefig(os.path.join(run_dir, "model_accuracy_history.png"))
            plt.close()

            plt.plot(train_history.history["loss"])
            plt.plot(train_history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "valid"], loc="upper left")
            plt.savefig(os.path.join(run_dir, "model_loss_history.png"))
            plt.close()

            # test on same cohort/dataset
            model.load_weights(output_tmp)
            model_result = model.predict(x_test_total)

            pickle.dump(
                y_test,
                open(os.path.join(run_dir, "ground_truth.np"), "wb"),
                protocol=4,
            )
            pickle.dump(
                model_result,
                open(os.path.join(run_dir, "lstm_inference.np"), "wb"),
                protocol=4,
            )

            ########################################
            # test_on_unbiased_pooled_dataset (LSTM)
            ########################################
            dir_gt_test = "test_on_unbiased_pooled"
            ext_test_dir = os.path.join(run_dir, dir_gt_test)
            if not os.path.exists(ext_test_dir):
                os.makedirs(ext_test_dir)

            x_test_gt, y_test_gt, c_test_gt = ut.load_mbp_or_wave_testing_dataset(
                testset_path=fixed_test_case_path,
                pred_win=pred_window,
                return_type="mbp",
                data_selection_type="unbiased",
                cohort="pooled",
                filter_nan=True,
            )

            delta_x_test_gt = np.diff(x_test_gt)
            delta_x_test_reshape_gt = np.insert(delta_x_test_gt, 0, 0, axis=1)
            delta_2_x_test_gt = np.diff(delta_x_test_gt)
            delta_x_test_tmp_gt = np.insert(delta_2_x_test_gt, 0, 0, axis=1)
            delta_2_x_test_reshape_gt = np.insert(delta_x_test_tmp_gt, 0, 0, axis=1)

            x_test_total_gt = np.stack(
                [x_test_gt, delta_x_test_reshape_gt, delta_2_x_test_reshape_gt], 2
            )

            model_gt_result = model.predict(x_test_total_gt)
            pickle.dump(
                model_gt_result,
                open(os.path.join(ext_test_dir, "inference.np"), "wb"),
                protocol=4,
            )

            evaluate(
                model_predict=model_gt_result,
                y_test=y_test_gt,
                plot_directory=ext_test_dir,
            )

        else:
            raise ValueError("model_type must be 'lr' or 'lstm'")

    else:
        raise NotImplementedError("Unsupported model_type")

    return None


####################################
# training codes with fixed dataset
####################################

fixed_testcase = "./test_caseids/fixed_test_caseid.csv"

MODEL_TYPE = ["cnn", "transformer", "lr", "lstm"]
PREDICTION_WINDOW = [5]           # match data builder (currently only 5-min)
SAMP_RATE_LIST = [1]
SELECTION_TYPE_LIST = ["unbiased", "biased"]  # new tags
COHORT_LIST = ["A", "B", "C", "pooled"]      # clean vs non-clean etc.

list_combination = list(
    itertools.product(
        MODEL_TYPE, PREDICTION_WINDOW, SAMP_RATE_LIST, SELECTION_TYPE_LIST, COHORT_LIST
    )
)

for model_type, pred_win, samp_rate, selection_type, cohort in list_combination:
    print(
        f"model_type: {model_type}, pred_window: {pred_win}, "
        f"samp_rate: {samp_rate}, selection_type: {selection_type}, cohort: {cohort}"
    )

    if model_type in ["cnn", "transformer"]:
        ret_type = "wave"
    elif model_type in ["lr", "lstm"]:
        ret_type = "mbp"
    else:
        raise ValueError("model_type must be cnn, transformer, lr, or lstm")

    result_saving_dir = f"./outputs/{model_type}"
    if not os.path.exists(result_saving_dir):
        os.makedirs(result_saving_dir)

    # ---------- NEW: resume/skip logic ----------
    subdirectory = "{}_{}_predwin_{}_sampling_{}".format(
        selection_type,
        cohort,
        int(pred_win),
        str(samp_rate).replace(".", ""),
    )
    run_dir = os.path.join(result_saving_dir, subdirectory)
    scores_path = os.path.join(
        run_dir,
        "test_on_same_dataset_used_in_train",
        "scores_with_ci.txt",
    )

    if os.path.exists(scores_path):
        print(f"[SKIP] Already finished {model_type} / {subdirectory}")
        continue

    training_and_testing(
        data_selection_method=selection_type,
        fixed_test_case_path=fixed_testcase,
        pred_window=pred_win,
        sampling_rate=samp_rate,
        return_type=ret_type,
        model_type=model_type,
        output_dir=result_saving_dir,
        cohort=cohort,
        remove_under65_in_train=True,
        filter_nan=True,
    )

    print(
        f"model_type: {model_type}, pred_window: {pred_win}, "
        f"samp_rate: {samp_rate}, selection_type: {selection_type}, cohort: {cohort} Done !"
    )
    print("\n")


####################################
# Training codes with KFOLD dataset
####################################

MODEL_TYPE = ["cnn", "transformer", "lr", "lstm"]
PREDICTION_WINDOW = [5]
SAMP_RATE_LIST = [1]
SELECTION_TYPE_LIST = ["unbiased", "biased"]
COHORT_LIST = ["A", "B", "C", "pooled"]

list_combination = list(
    itertools.product(
        MODEL_TYPE, PREDICTION_WINDOW, SAMP_RATE_LIST, SELECTION_TYPE_LIST, COHORT_LIST
    )
)

for kf in range(10):
    print("\n")
    print("=====================")
    print(f"Start FOLD {kf}")
    print("=====================")
    print("\n")

    current_test_case_path = "./test_caseids/test_caseid_kf_{}.csv".format(int(kf))

    for model_type, pred_win, samp_rate, selection_type, cohort in list_combination:
        print(
            "model_type: {}, pred_window: {}, samp_rate: {}, selection_type: {}, cohort: {}".format(
                model_type, pred_win, samp_rate, selection_type, cohort
            )
        )

        if model_type in ["cnn", "transformer"]:
            ret_type = "wave"
        elif model_type in ["lr", "lstm"]:
            ret_type = "mbp"
        else:
            raise ValueError("model_type must be cnn, transformer, lr, or lstm")

        result_saving_dir = "./outputs/{}".format(model_type)
        if not os.path.exists(result_saving_dir):
            os.makedirs(result_saving_dir)

        # ---------- NEW: resume/skip logic for k-fold ----------
        subdirectory = "{}_{}_predwin_{}_sampling_{}_kf{}".format(
            selection_type,
            cohort,
            int(pred_win),
            str(samp_rate).replace(".", ""),
            int(kf),
        )

        run_dir = os.path.join(result_saving_dir, subdirectory)
        scores_path = os.path.join(
            run_dir,
            "test_on_same_dataset_used_in_train",
            "scores_with_ci.txt",
        )

        if os.path.exists(scores_path):
            print(f"[SKIP] Already finished fold {kf} / {model_type} / {subdirectory}")
            continue

        training_and_testing(
            data_selection_method=selection_type,
            fixed_test_case_path=current_test_case_path,
            pred_window=pred_win,
            sampling_rate=samp_rate,
            return_type=ret_type,
            model_type=model_type,
            output_dir=result_saving_dir,
            cohort=cohort,
            remove_under65_in_train=True,
            filter_nan=True,
        )

        print(
            "model_type: {}, pred_window: {}, samp_rate: {}, selection_type: {}, cohort: {} Done !".format(
                model_type, pred_win, samp_rate, selection_type, cohort
            )
        )
        print("\n")
