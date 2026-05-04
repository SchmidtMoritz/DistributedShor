import pickle
import os


def check_configuration(
    circ_names, EJPP_list, approach_list, ebit_list, hw_label, path_short, mono=False
):

    if not mono:
        for ebit_time in ebit_list:
            result_path_top_level = f"{path_short}/{hw_label}_{ebit_time}/results.pkl"
            if os.path.isfile(result_path_top_level):
                with open(result_path_top_level, "rb") as fp:  # Unpickling
                    results = pickle.load(fp)
                    for circ_name in circ_names:
                        for approach in approach_list:
                            for EJPP in EJPP_list:
                                try:
                                    d = results[circ_name][approach][EJPP]["depth"]
                                    if d == 0:
                                        print(
                                            f"depth 0 for {ebit_time}, {circ_name}, {approach}, {EJPP} in {result_path_top_level}"
                                        )
                                except:
                                    print(
                                        f"no depth for {ebit_time}, {circ_name}, {approach}, {EJPP} in {result_path_top_level}"
                                    )
            else:
                print(f"file {result_path_top_level} does not exist!")

            for circ_name in circ_names:
                result_path_circ_level = (
                    f"{path_short}/{hw_label}_{ebit_time}/{circ_name}/results.pkl"
                )
                if os.path.isfile(result_path_circ_level):
                    with open(result_path_circ_level, "rb") as fp:  # Unpickling
                        results = pickle.load(fp)
                        for approach in approach_list:
                            for EJPP in EJPP_list:
                                try:
                                    d = results[approach][EJPP]["depth"]
                                    if d == 0:
                                        print(
                                            f"depth 0 for {ebit_time}, {circ_name}, {approach}, {EJPP} in {result_path_circ_level}"
                                        )
                                except:
                                    print(
                                        f"no depth for {ebit_time}, {circ_name}, {approach}, {EJPP} in {result_path_circ_level}"
                                    )
                else:
                    print(f"file {result_path_top_level} does not exist!")

    # check monolithic results
    result_path_top_level = f"{path_short}/{hw_label}/results.pkl"
    if os.path.isfile(result_path_top_level):
        with open(result_path_top_level, "rb") as fp:  # Unpickling
            results = pickle.load(fp)
            for circ_name in circ_names:
                for approach in approach_list:
                    try:
                        d = results[circ_name][approach][0]["depth"]
                        if d == 0:
                            print(
                                f"depth 0 for {circ_name}, {approach}, {0} in {result_path_top_level}"
                            )
                    except:
                        print(
                            f"no depth for {circ_name}, {approach}, {0} in {result_path_top_level}"
                        )
    else:
        print(f"file {result_path_top_level} does not exist!")

    for circ_name in circ_names:
        result_path_circ_level = f"{path_short}/{hw_label}/{circ_name}/results.pkl"
        if os.path.isfile(result_path_circ_level):
            with open(result_path_circ_level, "rb") as fp:  # Unpickling
                results = pickle.load(fp)
                for approach in approach_list:
                    try:
                        d = results[approach][0]["depth"]
                        if d == 0:
                            print(
                                f"depth 0 for {circ_name}, {approach}, {0} in {result_path_circ_level}"
                            )
                    except:
                        print(
                            f"no depth for {circ_name}, {approach}, {0} in {result_path_circ_level}"
                        )
        else:
            print(f"file {result_path_top_level} does not exist!")


if __name__ == "__main__":

    EJPP_list = [i for i in range(1, 5)] + [6, 8, 10]

    circ_names = [
        "N15_QRISP",
        "N21_QRISP",
        "N25_QRISP",
        "N33_QRISP",
        "N35_QRISP",
        "N39_QRISP",
        "N45_QRISP",
        "N49_QRISP",
        "N71_QRISP",
        "N711_QRISP",
        "N1311_QRISP",
        "N3111_QRISP",
        "N7111_QRISP",
        "N13111_QRISP",
        "N31111_QRISP",
        "N41111_QRISP",
        "N71111_QRISP",
        "N141111_QRISP",
        "N411111_QRISP",
        "N711111_QRISP",
        "N1311111_QRISP",
        "N13111111_QRISP",
        "N141111111_QRISP",
        "N1411111111_QRISP",
        "N13111111111_QRISP",
        "N131111111111_QRISP",
        "N711111111111_QRISP",
        "N33111111111111_QRISP",
        "N711111111111111_QRISP",
        "N711111111111111111_QRISP",
        "N13111111111111111111_QRISP",
    ]

    approach_list = [
        # "regular",
        # "iterative",
        "alternating",
        # "double_iterative",
        # "three_cyclic",
    ]

    ebit_list_ion = [
        (ebit_d_ms * 10**6) for ebit_d_ms in range(5, 20000, 500)
    ]  # 5ms to 20s

    ebit_list_sc = [(ebit_d_mus * 10**3) for ebit_d_mus in range(10, 1000, 10)]
    ebit_list_natom = [(ebit_d_ms * 10**6) for ebit_d_ms in range(5, 120, 5)]

    ebit_dict = {
        "depth_dlog_neutral_atom": ebit_list_natom,
        "depth_dlog_heron_ebit": ebit_list_sc,
        "depth_dlog_ionq_forte": ebit_list_ion,
    }
    hw_label = "depth_dlog_neutral_atom"
    hw_label = "depth_dlog_heron_ebit"
    hw_label = "depth_dlog_ionq_forte"

    # hw_label = "depth_heron"
    ebit_list = ebit_dict[hw_label]

    result_path_root = f"./src/semi_iterative_comparison/results_dlog"

    check_configuration(
        circ_names=circ_names,
        EJPP_list=EJPP_list,
        approach_list=approach_list,
        ebit_list=ebit_list,
        hw_label=hw_label,
        path_short=result_path_root,
        # mono=True,
    )

    approach_list = [
        # "regular",
        # "iterative",
        "alternating",
        # "double_iterative",
        # "three_cyclic",
    ]

    ebit_dict = {
        "depth_neutral_atom": ebit_list_natom,
        "depth_heron_ebit": ebit_list_sc,
        "depth_ionq_forte": ebit_list_ion,
    }
    hw_label = "depth_neutral_atom"
    hw_label = "depth_heron_ebit"
    # hw_label = "depth_ionq_forte"
    ebit_list = ebit_dict[hw_label]
    result_path_root = f"./src/semi_iterative_comparison/results_CU"

    check_configuration(
        circ_names=circ_names,
        EJPP_list=EJPP_list,
        approach_list=approach_list,
        ebit_list=ebit_list,
        hw_label=hw_label,
        path_short=result_path_root,
        # mono=True,
    )
