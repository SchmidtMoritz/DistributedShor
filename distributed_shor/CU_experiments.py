from datetime import datetime
import pickle
from distributed_shor.U_ops_from_qrisp import (
    get_avg_CU_duration,
    get_avg_CU_durations_dlog,
)
from distributed_shor.weight_utils import (
    weights_heron,
    weights_neutral_atom,
    weights_ionq_forte,
)  #
from distributed_shor.data_utils import merge_data_dict
import os
import sys
import numpy as np


def average_CU_experiments(N_list, weight_list, result_path_root):

    for weights in weight_list:

        print(f"{datetime.now()}: Starting average CU duration: {weights.label}")
        avg_CU_durations = {}

        for N in N_list:
            print(
                f"{datetime.now()}: Computing average CU duration for N={N} ({weights.label})"
            )
            count = int(np.ceil(np.log2(N)))
            avg_d, work_reg_size, instructions_per_length = get_avg_CU_duration(
                N=N, a=2, count=count, weights=weights
            )
            avg_CU_durations[N] = {
                "average_duration": avg_d,
                "work_reg_size": work_reg_size,
                "inst_per_length": instructions_per_length,
            }

        result_path = result_path_root + f"/depth_{weights.label}/"

        os.makedirs(result_path, exist_ok=True)
        if os.path.isfile(result_path + "avg_CU.pkl"):
            with open(result_path + "avg_CU.pkl", "rb") as fp:  # Unpickling
                old_results = pickle.load(fp)
                print(f"Merging results for {weights.label}:")
                avg_CU_durations = merge_data_dict(old_results, avg_CU_durations)

        with open(result_path + "avg_CU.pkl", "wb") as fp:
            pickle.dump(avg_CU_durations, fp)

        with open(result_path + "weights.pkl", "wb") as fp:
            pickle.dump(weights, fp)

        sys.stdout.flush()


def average_CU_experiments_dlog(N_list, weight_list, result_path_root):

    for weights in weight_list:

        print(f"{datetime.now()}: Starting average CU duration: {weights.label}")
        avg_CU_durations = {}

        for N in N_list:
            print(
                f"{datetime.now()}: Computing average CU duration for N={N} ({weights.label})"
            )
            sys.stdout.flush()

            count = int(np.ceil(np.ceil(np.log2(N)) * 0.5))
            avg_d, work_reg_size, instructions_per_length = get_avg_CU_durations_dlog(
                N=N, a=2, count=count, weights=weights
            )
            avg_CU_durations[N] = {
                "average_duration": avg_d,
                "work_reg_size": work_reg_size,
                "inst_per_length": instructions_per_length,
            }

        result_path = result_path_root + f"/depth_dlog_{weights.label}/"

        os.makedirs(result_path, exist_ok=True)
        if os.path.isfile(result_path + "avg_CU.pkl"):
            with open(result_path + "avg_CU.pkl", "rb") as fp:  # Unpickling
                old_results = pickle.load(fp)
                print(f"Merging results for {weights.label}:")
                avg_CU_durations = merge_data_dict(old_results, avg_CU_durations)

        with open(result_path + "avg_CU.pkl", "wb") as fp:
            pickle.dump(avg_CU_durations, fp)

        with open(result_path + "weights.pkl", "wb") as fp:
            pickle.dump(weights, fp)

        sys.stdout.flush()


def run_average_CU_experiments():

    result_path_root = f"./out/results"
    weight_list = [
        weights_neutral_atom
    ]  # [weights_heron, weights_neutral_atom, weights_ionq_forte]

    # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
    N_list = [
        15,
        21,
        25,
        # 33,
        # 35,
        # 39,
        # 45,
        # 49,
        # 71,
        # 711,
        # 1311,
        # 3111,
        # 7111,
        # 13111,
        # 31111,
        # 41111,
        # 71111,
        # 141111,
        # 411111,
        # 711111,
        # 1311111,
        # 13111111,
        # 141111111,
        # 1411111111,
        # 13111111111,
        # 131111111111,
        # 711111111111,
        # 33111111111111,
        # 711111111111111,
        # 711111111111111111,
        # 13111111111111111111,
    ]
    average_CU_experiments(
        N_list=N_list, weight_list=weight_list, result_path_root=result_path_root
    )


def run_average_CU_experiments_dlog():

    result_path_root = f"./out/results_dlog"
    weight_list = [weights_heron, weights_neutral_atom, weights_ionq_forte]

    # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
    N_list = [
        15,
        21,
        25,
        33,
        35,
        39,
        45,
        49,
        71,
        711,
        1311,
        3111,
        7111,
        13111,
        31111,
        41111,
        71111,
        141111,
        411111,
        711111,
        1311111,
        13111111,
        141111111,
        1411111111,
        13111111111,
        131111111111,
        711111111111,
        33111111111111,
        711111111111111,
        711111111111111111,
        13111111111111111111,
    ]
    average_CU_experiments_dlog(
        N_list=N_list, weight_list=weight_list, result_path_root=result_path_root
    )


if __name__ == "__main__":
    run_average_CU_experiments()
