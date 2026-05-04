import argparse

from distributed_shor.circ_stat_utils import (
    depths_from_circ,
    count_circuit_ops,
    get_transpiled_circuit,
    circuit_depths_top_sort,
)
from distributed_shor.semi_iterative_comparison import (
    regular_circ,
    alternating_iterative_QPE_circ,
    iterative_QPE_circ,
    iterative_dlog_circ,
    alternating_seq_dlog_circ,
    double_iterative_dlog_circ,
    three_cycle_dlog_circ,
    regular_dlog_circ,
)
from distributed_shor.utils import *
import csv
from qiskit.visualization import plot_histogram, plot_distribution
from matplotlib import pyplot as plt
import numpy as np
from distributed_shor.basis_gates import get_basis_gates
import pickle
from distributed_shor.U_ops_from_qrisp import (
    get_QRISP_circ_data_function,
    get_CU_list_gates,
    get_dlog_QRISP_circ_data_function,
)
import os
from distributed_shor.graph_utils import CircuitGraph
from distributed_shor.weight_utils import (
    get_weights_heron_ebit_duration,
    get_weights_neutral_atom_ebit_duration,
    get_weights_ionq_forte_ebit_duration,
    instruction_weights,
    weights_heron_ebit,
    weights_aspdac,
    weights_neutral_atom,
    weights_heron,
    weights_ionq_forte,
)
from qiskit import QuantumRegister, QuantumCircuit
from distributed_shor.data_utils import merge_data_dict

"""
!!! WEIGHTS ARE ONLY USED IN THE GRAPHS, NOT IN THE CIRCUITS !!!    
"""
from copy import deepcopy
import sys
import os.path
import psutil
import time
from datetime import datetime

import multiprocessing as mp
from distributed_shor.U_ops_from_qrisp import get_avg_CU_duration
from distributed_shor.weight_utils import get_start_stop_duration


def get_start_stop_duration_heron(reset=True, ebit=True):
    return get_start_stop_duration(weights=weights_heron_ebit, reset=reset, ebit=ebit)


def experiments(pkl_circuits=False):
    """

    dict structure:

    results["circ_name"]["design"][ejpp][setup/ejpp/depth]

    circ_results["design"][ejpp][setup/ejpp/depth]




    """

    weight_list = [
        get_weights_heron_ebit_duration(ebit_d_mus * 10**3)
        for ebit_d_mus in range(100, 1000, 100)
    ]
    for weights in weight_list:
        result_path = (
            f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
        )
        circuit_specs = [
            ("N15", get_N15_circ_data),
            ("N21B2", get_N21B2_circ_data),
            ("N21B4", get_N21B4_circ_data),
            ("N35", get_N35_circ_data),
            ("N15_QRISP", get_QRISP_circ_data_function(15, 2, 2 * 4)),
            ("N21_QRISP", get_QRISP_circ_data_function(21, 2, 2 * 5)),
            ("N25_QRISP", get_QRISP_circ_data_function(25, 2, 2 * 5)),
            ("N33_QRISP", get_QRISP_circ_data_function(33, 2, 2 * 6)),
            ("N35_QRISP", get_QRISP_circ_data_function(35, 2, 2 * 6)),
            ("N39_QRISP", get_QRISP_circ_data_function(39, 2, 2 * 6)),
            ("N45_QRISP", get_QRISP_circ_data_function(45, 2, 2 * 6)),
            ("N49_QRISP", get_QRISP_circ_data_function(49, 2, 2 * 6)),
            ("N71_QRISP", get_QRISP_circ_data_function(71, 2, 2 * 7)),
            ("N711_QRISP", get_QRISP_circ_data_function(711, 2, 2 * 10)),
            ("N1311_QRISP", get_QRISP_circ_data_function(1311, 2, 2 * 11)),
            ("N3111_QRISP", get_QRISP_circ_data_function(3111, 2, 2 * 12)),
            ("N7111_QRISP", get_QRISP_circ_data_function(7111, 2, 2 * 13)),
            ("N13111_QRISP", get_QRISP_circ_data_function(13111, 2, 2 * 14)),
            ("N31111_QRISP", get_QRISP_circ_data_function(31111, 2, 2 * 15)),
        ]

        # [15, 21, 25, 33, 35, 39, 45, 49]

        # list to store all results which shall get stored

        all_results = {}
        try:

            with open(result_path + "results.pkl", "rb") as fp:  # Unpickling
                all_results = pickle.load(fp)
        except:
            print("file not found")

        for circ_spec in circuit_specs:
            U_list, top_c, bot_c, init_ind, init_one = circ_spec[1]()
            circ_results = {}
            os.makedirs(result_path + f"{circ_spec[0]}/", exist_ok=True)
            regular_results = []
            iterative_results = []
            alternating_results = []

            for EJPP in [0, 1, 2, 3]:

                print(f"Starting regular {circ_spec[0]}, EJPP={EJPP}")
                ##########
                # regular
                ##########
                circ = regular_circ(
                    U_list=U_list,
                    top_c=top_c,
                    bot_c=bot_c,
                    init_one=init_one,
                    init_ind=init_ind,
                    EJPP=EJPP,
                )
                if pkl_circuits:
                    with open(
                        result_path + f"{circ_spec[0]}/circuit_regular_EJPP{EJPP}.pkl",
                        "wb",
                    ) as fp:
                        pickle.dump(circ, fp)

                circ_transpiled = get_transpiled_circuit(circ)
                del circ
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_reg = g.get_weighted_depth()

                del circ_transpiled
                del g

                regular_results.append(
                    {
                        "setup": f"Setup: Regular {circ_spec[0]}",
                        "EJPP": EJPP,
                        "depth": d_reg,
                    }
                )

                ############
                # iterative
                ############

                print(f"Starting iterative {circ_spec[0]}, EJPP={EJPP}")

                circ = iterative_QPE_circ(
                    U_list=U_list,
                    top_c=top_c,
                    bot_c=bot_c,
                    init_one=init_one,
                    init_ind=init_ind,
                    EJPP=EJPP,
                )
                if pkl_circuits:
                    with open(
                        result_path
                        + f"{circ_spec[0]}/circuit_iterative_EJPP{EJPP}.pkl",
                        "wb",
                    ) as fp:
                        pickle.dump(circ, fp)

                circ_transpiled = get_transpiled_circuit(circ)
                del circ
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_reg = g.get_weighted_depth()

                del circ_transpiled
                del g

                iterative_results.append(
                    {
                        "setup": f"Setup: Regular {circ_spec[0]}",
                        "EJPP": EJPP,
                        "depth": d_reg,
                    }
                )

                ##############
                # alternating
                ##############

                print(f"Starting alternating {circ_spec[0]}, EJPP={EJPP}")

                circ = alternating_iterative_QPE_circ(
                    U_list=U_list,
                    top_c=top_c,
                    bot_c=bot_c,
                    init_one=init_one,
                    init_ind=init_ind,
                    EJPP=EJPP,
                )
                if pkl_circuits:
                    with open(
                        result_path
                        + f"{circ_spec[0]}/circuit_alternating_EJPP{EJPP}.pkl",
                        "wb",
                    ) as fp:
                        pickle.dump(circ, fp)

                circ_transpiled = get_transpiled_circuit(circ)
                del circ
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_reg = g.get_weighted_depth()

                del circ_transpiled
                del g

                alternating_results.append(
                    {
                        "setup": f"Setup: Regular {circ_spec[0]}",
                        "EJPP": EJPP,
                        "depth": d_reg,
                    }
                )

            circ_results["regular"] = regular_results
            circ_results["iterative"] = iterative_results
            circ_results["alternating"] = alternating_results

            with open(result_path + f"{circ_spec[0]}/" + "results.pkl", "wb") as fp:
                pickle.dump(circ_results, fp)

            if f"{circ_spec[0]}" in all_results.keys():
                print(f"Overriding {circ_spec[0]} results.")

            all_results[f"{circ_spec[0]}"] = circ_results

            with open(result_path + "results.pkl", "wb") as fp:
                pickle.dump(all_results, fp)

            with open(result_path + "weights.pkl", "wb") as fp:
                pickle.dump(weights, fp)


def print_memory_usage():
    process = psutil.Process()
    mem_usage = process.memory_info().rss
    print(f"Currently using {mem_usage*10**-6} MB RAM")


def circuit_exp_task(
    weight_list,
    approach,
    circ_spec_name,
    EJPP,
    queue,
    result_path_root,
    pkl_circuits=False,
    use_graph=False,
):
    circ_spec = None
    for cs in circuit_specs:
        if cs[0] == circ_spec_name:
            circ_spec = cs
            break

    regular = False
    iterative = False
    alternating = False

    if approach == "regular":
        regular = True
    elif approach == "iterative":
        iterative = True
    elif approach == "alternating":
        alternating = True

    U_list, top_c, bot_c, init_ind, init_one = circ_spec[1]()

    if regular:
        print(f"{datetime.now()}: Building regular {circ_spec[0]} circuit, EJPP={EJPP}")
        ##########
        # regular
        ##########
        circ = regular_circ(
            U_list=U_list,
            top_c=top_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )
        if pkl_circuits:
            with open(
                result_path_root + f"{circ_spec[0]}/circuit_regular_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)
        del circ
        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating regular {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph
                del g

            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Regular {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }
            queue.put([weights, circ_spec[0], "regular", EJPP, res_dict])

        del circ_transpiled

    if iterative:
        ############
        # iterative
        ############

        print(
            f"{datetime.now()}: Building iterative {circ_spec[0]} circuit, EJPP={EJPP}"
        )

        circ = iterative_QPE_circ(
            U_list=U_list,
            top_c=top_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )
        if pkl_circuits:
            with open(
                result_path_root + f"{circ_spec[0]}/circuit_iterative_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)

        del circ
        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating iterative {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph
                del g

            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Iterative {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }

            queue.put([weights, circ_spec[0], "iterative", EJPP, res_dict])

        del circ_transpiled

    if alternating:
        ##############
        # alternating
        ##############

        print(
            f"{datetime.now()}: Building alternating {circ_spec[0]} circuit, EJPP={EJPP}"
        )

        circ = alternating_iterative_QPE_circ(
            U_list=U_list,
            top_c=top_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )
        if pkl_circuits:
            with open(
                result_path_root + f"{circ_spec[0]}/circuit_alternating_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)

        del circ

        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating alternating {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph

            # print_memory_usage()

            # del g
            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Alternating {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }

            queue.put([weights, circ_spec[0], "alternating", EJPP, res_dict])

        del circ_transpiled
    sys.stdout.flush()


def data_collector(
    queue,
    result_path_root,
    EJPP_list,
    regular=False,
    alternating=True,
    iterative=False,
):

    approach_dict = {}
    if regular:
        approach_dict["regular"] = {EJPP: "missing" for EJPP in EJPP_list}
    if iterative:
        approach_dict["iterative"] = {EJPP: "missing" for EJPP in EJPP_list}
    if alternating:
        approach_dict["alternating"] = {EJPP: "missing" for EJPP in EJPP_list}

    results_ebit_d = {
        weights.label: {
            circ_spec[0]: deepcopy(approach_dict) for circ_spec in circuit_specs
        }
        for weights in weight_list
    }

    circuit_names = [circ_spec[0] for circ_spec in circuit_specs]

    while True:
        item = queue.get()
        if item == "kill":
            print("Data collector stopped.")
            sys.stdout.flush()
            break

        else:
            item_weights, item_circ_spec_name, item_approach, item_EJPP, res_dict = item
        print(f"{datetime.now()}: Saving results: circuit {item_circ_spec_name}")

        # save results in dictionary
        results_ebit_d[item_weights.label][item_circ_spec_name][item_approach][
            item_EJPP
        ] = res_dict

        # check which circuits (for the current items weight) are complete
        circ_complete = {circ_name: True for circ_name in circuit_names}

        for circ_spec_name_i in circuit_names:
            for approach in approach_dict.keys():
                for EJPP in EJPP_list:
                    if (
                        results_ebit_d[item_weights.label][circ_spec_name_i][approach][
                            EJPP
                        ]
                        == "missing"
                    ):
                        circ_complete[circ_spec_name_i] = False

        for circ_spec_name_i in circuit_names:
            if circ_complete[circ_spec_name_i]:

                result_path = result_path_root + f"/depth_{item_weights.label}/"

                all_results = results_ebit_d[item_weights.label]

                os.makedirs(result_path + f"{circ_spec_name_i}/", exist_ok=True)

                circ_results = all_results[circ_spec_name_i]

                result_path_circ = result_path + f"{circ_spec_name_i}/" + "results.pkl"
                if os.path.isfile(result_path_circ):

                    with open(result_path_circ, "rb") as fp:  # Unpickling
                        old_results = pickle.load(fp)
                        print(
                            f"Merging results for {item_weights.label}, {circ_spec_name_i}:"
                        )
                        circ_results = merge_data_dict(old_results, circ_results)

                with open(
                    result_path + f"{circ_spec_name_i}/" + "results.pkl", "wb"
                ) as fp:
                    pickle.dump(circ_results, fp)

                if all(
                    v == True for v in circ_complete.values()
                ):  # only save all_results if all circuits are completed for a weight
                    if os.path.isfile(result_path + "results.pkl"):

                        with open(
                            result_path + "results.pkl", "rb"
                        ) as fp:  # Unpickling
                            old_results = pickle.load(fp)
                            print(f"Merging results for {item_weights.label}:")
                            all_results = merge_data_dict(old_results, all_results)

                    with open(result_path + "results.pkl", "wb") as fp:
                        pickle.dump(all_results, fp)

                    with open(result_path + "weights.pkl", "wb") as fp:
                        pickle.dump(item_weights, fp)

            sys.stdout.flush()


def run_experiments_mp(
    weight_list,
    circuit_specs,
    result_path_root,
    n_processes,
    EJPP_list=[i for i in range(5)],
    pkl_circuits=False,
    regular=False,
    alternating=True,
    iterative=False,
    use_graph=False,
):
    assert n_processes >= 2

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(n_processes)

    data_collect = pool.apply_async(
        data_collector,
        (
            queue,
            result_path_root,
            EJPP_list,
            regular,
            alternating,
            iterative,
        ),
    )
    jobs = []
    approaches = []
    if regular:
        approaches.append("regular")
    if iterative:
        approaches.append("iterative")
    if alternating:
        approaches.append("alternating")

    for circ_spec in circuit_specs:
        circ_spec_name = circ_spec[0]
        for approach in approaches:
            for EJPP in EJPP_list:
                job = pool.apply_async(
                    circuit_exp_task,
                    (
                        weight_list,
                        approach,
                        circ_spec_name,
                        EJPP,
                        queue,
                        result_path_root,
                        pkl_circuits,
                        use_graph,
                    ),
                )
                jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()


def experiments_weights(
    weight_list,
    circuit_specs,
    result_path_root,
    EJPP_list=[i for i in range(5)],
    pkl_circuits=False,
    regular=False,
    alternating=True,
    iterative=False,
):
    """

    dict structure:

    results["circ_name"]["design"][ejpp][setup/ejpp/depth]

    circ_results["design"][ejpp][setup/ejpp/depth]




    """

    # [15, 21, 25, 33, 35, 39, 45, 49]

    # list to store all results which shall get stored
    approach_dict = {}
    if regular:
        approach_dict["regular"] = {EJPP: "missing" for EJPP in EJPP_list}
    if iterative:
        approach_dict["iterative"] = {EJPP: "missing" for EJPP in EJPP_list}
    if alternating:
        approach_dict["alternating"] = {EJPP: "missing" for EJPP in EJPP_list}

    results_ebit_d = {
        weights.label: {
            circ_spec[0]: deepcopy(approach_dict) for circ_spec in circuit_specs
        }
        for weights in weight_list
    }
    # [ebit_duration][circ_name]["alternating"][ejpp]

    for circ_spec in circuit_specs:
        U_list, top_c, bot_c, init_ind, init_one = circ_spec[1]()

        for EJPP in EJPP_list:

            if regular:
                print(
                    f"{datetime.now()}: Building regular {circ_spec[0]} circuit, EJPP={EJPP}"
                )
                ##########
                # regular
                ##########
                circ = regular_circ(
                    U_list=U_list,
                    top_c=top_c,
                    bot_c=bot_c,
                    init_one=init_one,
                    init_ind=init_ind,
                    EJPP=EJPP,
                )
                if pkl_circuits:
                    with open(
                        result_path_root
                        + f"{circ_spec[0]}/circuit_regular_EJPP{EJPP}.pkl",
                        "wb",
                    ) as fp:
                        pickle.dump(circ, fp)

                circ_transpiled = get_transpiled_circuit(circ)
                del circ
                for weights in weight_list:
                    print(
                        f"{datetime.now()}: Calculating regular {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
                    )
                    g = CircuitGraph(
                        circ_transpiled,
                        weight_dict=weights.weight_dict,
                        weight_one_qubit=weights.weight_one_qubit,
                        weight_two_qubit=weights.weight_two_qubit,
                    )

                    d_graph = g.get_weighted_depth()

                    del g

                    results_ebit_d[weights.label][circ_spec[0]]["regular"][EJPP] = {
                        "setup": f"Setup: Regular {circ_spec[0]}",
                        "EJPP": EJPP,
                        "depth": d_graph,
                    }

                del circ_transpiled

            if iterative:
                ############
                # iterative
                ############

                print(
                    f"{datetime.now()}: Building iterative {circ_spec[0]} circuit, EJPP={EJPP}"
                )

                circ = iterative_QPE_circ(
                    U_list=U_list,
                    top_c=top_c,
                    bot_c=bot_c,
                    init_one=init_one,
                    init_ind=init_ind,
                    EJPP=EJPP,
                )
                if pkl_circuits:
                    with open(
                        result_path_root
                        + f"{circ_spec[0]}/circuit_iterative_EJPP{EJPP}.pkl",
                        "wb",
                    ) as fp:
                        pickle.dump(circ, fp)

                circ_transpiled = get_transpiled_circuit(circ)

                del circ
                for weights in weight_list:
                    print(
                        f"{datetime.now()}: Calculating iterative {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
                    )
                    g = CircuitGraph(
                        circ_transpiled,
                        weight_dict=weights.weight_dict,
                        weight_one_qubit=weights.weight_one_qubit,
                        weight_two_qubit=weights.weight_two_qubit,
                    )

                    d_graph = g.get_weighted_depth()
                    del g

                    results_ebit_d[weights.label][circ_spec[0]]["iterative"][EJPP] = {
                        "setup": f"Setup: Iterative {circ_spec[0]}",
                        "EJPP": EJPP,
                        "depth": d_graph,
                    }

                del circ_transpiled

            if alternating:
                ##############
                # alternating
                ##############

                print(
                    f"{datetime.now()}: Building alternating {circ_spec[0]} circuit, EJPP={EJPP}"
                )

                circ = alternating_iterative_QPE_circ(
                    U_list=U_list,
                    top_c=top_c,
                    bot_c=bot_c,
                    init_one=init_one,
                    init_ind=init_ind,
                    EJPP=EJPP,
                )
                if pkl_circuits:
                    with open(
                        result_path_root
                        + f"{circ_spec[0]}/circuit_alternating_EJPP{EJPP}.pkl",
                        "wb",
                    ) as fp:
                        pickle.dump(circ, fp)

                circ_transpiled = get_transpiled_circuit(circ)

                del circ

                for weights in weight_list:
                    print(
                        f"{datetime.now()}: Calculating alternating {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
                    )

                    # g = CircuitGraph(
                    #     circ_transpiled,
                    #     weight_dict=weights.weight_dict,
                    #     weight_one_qubit=weights.weight_one_qubit,
                    #     weight_two_qubit=weights.weight_two_qubit,
                    # )

                    # d_graph = g.get_weighted_depth()

                    # print_memory_usage()

                    # del g

                    print_memory_usage()

                    d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                    d = d_topo

                    results_ebit_d[weights.label][circ_spec[0]]["alternating"][EJPP] = {
                        "setup": f"Setup: Alternating {circ_spec[0]}",
                        "EJPP": EJPP,
                        "depth": d,
                    }

                del circ_transpiled
            sys.stdout.flush()

        print(f"{datetime.now()}: Saving results: circuit {circ_spec[0]}")

        for weights in weight_list:

            result_path = result_path_root + f"/depth_{weights.label}/"

            all_results = results_ebit_d[weights.label]

            os.makedirs(result_path + f"{circ_spec[0]}/", exist_ok=True)

            circ_results = all_results[circ_spec[0]]

            with open(result_path + f"{circ_spec[0]}/" + "results.pkl", "wb") as fp:
                pickle.dump(circ_results, fp)

            if os.path.isfile(result_path + "results.pkl"):

                with open(result_path + "results.pkl", "rb") as fp:  # Unpickling
                    old_results = pickle.load(fp)
                    print(f"Merging results for {weights.label}:")
                    all_results = merge_data_dict(old_results, all_results)

            with open(result_path + "results.pkl", "wb") as fp:
                pickle.dump(all_results, fp)

            with open(result_path + "weights.pkl", "wb") as fp:
                pickle.dump(weights, fp)

            sys.stdout.flush()


def parse_args():
    # Parse command line arguments. Only one is supported here to specify the number
    # of parallel processes.
    parser = argparse.ArgumentParser("Run Experiment")

    parser.add_argument(
        "-p",
        type=int,
        default=4,
        dest="n_processes",
        help="Number of processes to launch",
    )

    parser.add_argument(
        "-e",
        type=int,
        default=1,
        dest="experiment",
        help="1=testing, 2=monolithic, 3=ionq_distributed",
    )

    return parser.parse_args()


# ------------------------------------------------
# dlog experiments


#
# 4 EJPP * 5 approaches * 14 large circs = 280
# 4 EJPP * 3 approaches * 14 large circs = 168
# 4 EJPP * 3 approaches * 14 large circs = 168
#
# 1500gb / 60gb = 25
# 280/25 = 11.2 sequential jobs
# 11.2*14h = 156.8 h = 6.533 days
# 168/25 = 6.72 sequential jobs
# 6.72 * 14h = 94.08h = 3.92 days


def circuit_exp_task_dlog(
    weight_list,
    approach,
    circ_spec_name,
    EJPP,
    queue,
    result_path_root,
    pkl_circuits=False,
    use_graph=False,
):
    circ_spec = None
    for cs in circuit_specs:
        if cs[0] == circ_spec_name:
            circ_spec = cs
            break

    regular = False
    iterative = False
    alternating = False
    double_iterative = False
    three_cyclic = False

    if approach == "regular":
        regular = True
    elif approach == "iterative":
        iterative = True
    elif approach == "alternating":
        alternating = True
    elif approach == "double_iterative":
        double_iterative = True
    elif approach == "three_cyclic":
        three_cyclic = True

    U_e1_list, U_e2_list, top_e1_c, top_e2_c, bot_c, init_ind, init_one = circ_spec[1]()

    if regular:
        print(
            f"{datetime.now()}: Building regular discrete log {circ_spec[0]} circuit, EJPP={EJPP}"
        )
        ##########
        # regular
        ##########
        circ = regular_dlog_circ(
            U_e1_list=U_e1_list,
            U_e2_list=U_e2_list,
            top_e1_c=top_e1_c,
            top_e2_c=top_e2_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )
        if pkl_circuits:
            with open(
                result_path_root
                + f"{circ_spec[0]}/circuit_regular_dlog_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)
        del circ
        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating regular discrete log {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph
                del g

            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Regular dlog {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }
            queue.put([weights, circ_spec[0], "regular", EJPP, res_dict])

        del circ_transpiled

    if iterative:
        ############
        # iterative
        ############

        print(
            f"{datetime.now()}: Building iterative discrete log {circ_spec[0]} circuit, EJPP={EJPP}"
        )

        circ = iterative_dlog_circ(
            U_e1_list=U_e1_list,
            U_e2_list=U_e2_list,
            top_e1_c=top_e1_c,
            top_e2_c=top_e2_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )
        if pkl_circuits:
            with open(
                result_path_root
                + f"{circ_spec[0]}/circuit_iterative_dlog_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)

        del circ
        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating iterative discrete log {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph
                del g

            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Iterative dlog {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }

            queue.put([weights, circ_spec[0], "iterative", EJPP, res_dict])

        del circ_transpiled

    if alternating:
        ##############
        # alternating
        ##############

        print(
            f"{datetime.now()}: Building alternating discrete log {circ_spec[0]} circuit, EJPP={EJPP}"
        )

        circ = alternating_seq_dlog_circ(
            U_e1_list=U_e1_list,
            U_e2_list=U_e2_list,
            top_e1_c=top_e1_c,
            top_e2_c=top_e2_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )

        if pkl_circuits:
            with open(
                result_path_root
                + f"{circ_spec[0]}/circuit_alternating_dlog_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)

        del circ

        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating alternating discrete log {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph

            # print_memory_usage()

            # del g
            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Alternating dlog {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }

            queue.put([weights, circ_spec[0], "alternating", EJPP, res_dict])

        del circ_transpiled

    if double_iterative:
        print(
            f"{datetime.now()}: Building double_iterative discrete log {circ_spec[0]} circuit, EJPP={EJPP}"
        )
        ##########
        # double_iterative
        ##########
        circ = double_iterative_dlog_circ(
            U_e1_list=U_e1_list,
            U_e2_list=U_e2_list,
            top_e1_c=top_e1_c,
            top_e2_c=top_e2_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )
        if pkl_circuits:
            with open(
                result_path_root
                + f"{circ_spec[0]}/circuit_double_iterative_dlog_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)
        del circ
        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating double_iterative discrete log {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph
                del g

            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Double Iterative dlog {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }
            queue.put([weights, circ_spec[0], "double_iterative", EJPP, res_dict])

        del circ_transpiled

    if three_cyclic:
        print(
            f"{datetime.now()}: Building three_cyclic discrete log {circ_spec[0]} circuit, EJPP={EJPP}"
        )
        ##########
        # three_cyclic
        ##########
        circ = three_cycle_dlog_circ(
            U_e1_list=U_e1_list,
            U_e2_list=U_e2_list,
            top_e1_c=top_e1_c,
            top_e2_c=top_e2_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            EJPP=EJPP,
        )
        if pkl_circuits:
            with open(
                result_path_root
                + f"{circ_spec[0]}/circuit_three_cyclic_dlog_EJPP{EJPP}.pkl",
                "wb",
            ) as fp:
                pickle.dump(circ, fp)

        circ_transpiled = get_transpiled_circuit(circ)
        del circ
        for weights in weight_list:
            print(
                f"{datetime.now()}: Calculating three_cyclic discrete log {circ_spec[0]} duration, EJPP={EJPP}, T={weights.label}"
            )
            d = 0
            if use_graph:
                g = CircuitGraph(
                    circ_transpiled,
                    weight_dict=weights.weight_dict,
                    weight_one_qubit=weights.weight_one_qubit,
                    weight_two_qubit=weights.weight_two_qubit,
                )

                d_graph = g.get_weighted_depth()
                d = d_graph
                del g

            else:
                d_topo = circuit_depths_top_sort(circ_transpiled, weights)
                d = d_topo

            print_memory_usage()

            res_dict = {
                "setup": f"Setup: Three Cyclic dlog {circ_spec[0]}",
                "EJPP": EJPP,
                "depth": d,
            }
            queue.put([weights, circ_spec[0], "three_cyclic", EJPP, res_dict])

        del circ_transpiled

    sys.stdout.flush()


def data_collector_dlog(
    queue,
    result_path_root,
    EJPP_list,
    regular=False,
    iterative=False,
    alternating=True,
    double_iterative=False,
    three_cyclic=False,
):

    approach_dict = {}
    if regular:
        approach_dict["regular"] = {EJPP: "missing" for EJPP in EJPP_list}
    if iterative:
        approach_dict["iterative"] = {EJPP: "missing" for EJPP in EJPP_list}
    if alternating:
        approach_dict["alternating"] = {EJPP: "missing" for EJPP in EJPP_list}
    if double_iterative:
        approach_dict["double_iterative"] = {EJPP: "missing" for EJPP in EJPP_list}
    if three_cyclic:
        approach_dict["three_cyclic"] = {EJPP: "missing" for EJPP in EJPP_list}

    results_ebit_d = {
        weights.label: {
            circ_spec[0]: deepcopy(approach_dict) for circ_spec in circuit_specs
        }
        for weights in weight_list
    }

    circuit_names = [circ_spec[0] for circ_spec in circuit_specs]

    while True:
        item = queue.get()
        if item == "kill":
            print("Data collector stopped.")
            sys.stdout.flush()
            break

        else:
            item_weights, item_circ_spec_name, item_approach, item_EJPP, res_dict = item
        print(f"{datetime.now()}: Saving results: circuit {item_circ_spec_name}")

        # save results in dictionary
        results_ebit_d[item_weights.label][item_circ_spec_name][item_approach][
            item_EJPP
        ] = res_dict

        # check which circuits (for the current items weight) are complete
        circ_complete = {circ_name: True for circ_name in circuit_names}

        for circ_spec_name_i in circuit_names:
            for approach in approach_dict.keys():
                for EJPP in EJPP_list:
                    if (
                        results_ebit_d[item_weights.label][circ_spec_name_i][approach][
                            EJPP
                        ]
                        == "missing"
                    ):
                        circ_complete[circ_spec_name_i] = False

        for circ_spec_name_i in circuit_names:
            if circ_complete[circ_spec_name_i]:

                result_path = result_path_root + f"/depth_dlog_{item_weights.label}/"

                all_results = results_ebit_d[item_weights.label]

                os.makedirs(result_path + f"{circ_spec_name_i}/", exist_ok=True)

                circ_results = all_results[circ_spec_name_i]

                result_path_circ = result_path + f"{circ_spec_name_i}/" + "results.pkl"
                if os.path.isfile(result_path_circ):

                    with open(result_path_circ, "rb") as fp:  # Unpickling
                        old_results = pickle.load(fp)
                        print(
                            f"Merging results for {item_weights.label}, {circ_spec_name_i}:"
                        )
                        circ_results = merge_data_dict(old_results, circ_results)

                with open(
                    result_path + f"{circ_spec_name_i}/" + "results.pkl", "wb"
                ) as fp:
                    pickle.dump(circ_results, fp)

                if all(
                    v == True for v in circ_complete.values()
                ):  # only save all_results if all circuits are completed for a weight
                    if os.path.isfile(result_path + "results.pkl"):

                        with open(
                            result_path + "results.pkl", "rb"
                        ) as fp:  # Unpickling
                            old_results = pickle.load(fp)
                            print(f"Merging results for {item_weights.label}:")
                            all_results = merge_data_dict(old_results, all_results)

                    with open(result_path + "results.pkl", "wb") as fp:
                        pickle.dump(all_results, fp)

                    with open(result_path + "weights.pkl", "wb") as fp:
                        pickle.dump(item_weights, fp)

            sys.stdout.flush()


def run_experiments_mp_dlog(
    weight_list,
    circuit_specs,
    result_path_root,
    n_processes,
    EJPP_list=[i for i in range(5)],
    pkl_circuits=False,
    regular=False,
    iterative=False,
    alternating=True,
    double_iterative=False,
    three_cyclic=False,
    use_graph=False,
):
    assert n_processes >= 2

    manager = mp.Manager()
    queue = manager.Queue()
    pool = mp.Pool(n_processes)

    data_collect = pool.apply_async(
        data_collector_dlog,
        (
            queue,
            result_path_root,
            EJPP_list,
            regular,
            iterative,
            alternating,
            double_iterative,
            three_cyclic,
        ),
    )
    jobs = []
    approaches = []
    if regular:
        approaches.append("regular")
    if iterative:
        approaches.append("iterative")
    if alternating:
        approaches.append("alternating")
    if double_iterative:
        approaches.append("double_iterative")
    if three_cyclic:
        approaches.append("three_cyclic")

    for circ_spec in circuit_specs:
        circ_spec_name = circ_spec[0]
        for approach in approaches:
            for EJPP in EJPP_list:
                job = pool.apply_async(
                    circuit_exp_task_dlog,
                    (
                        weight_list,
                        approach,
                        circ_spec_name,
                        EJPP,
                        queue,
                        result_path_root,
                        pkl_circuits,
                        use_graph,
                    ),
                )
                jobs.append(job)

    for job in jobs:
        job.get()

    queue.put("kill")
    pool.close()
    pool.join()


if __name__ == "__main__":

    args = parse_args()

    experiment = args.experiment

    # mp test
    """
    result_path_root = f"./src/semi_iterative_comparison/results_mp_test"
    weight_list = [
        get_weights_neutral_atom_ebit_duration(ebit_d_ms * 10**6)
        for ebit_d_ms in range(5, 20, 5)
    ]

    # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
    circuit_specs = [
        # ("N15", get_N15_circ_data),
        # ("N21B2", get_N21B2_circ_data),
        # ("N21B4", get_N21B4_circ_data),
        # ("N35", get_N35_circ_data),
        ("N15_QRISP", get_QRISP_circ_data_function(15, 2, 2 * 4)),
        ("N21_QRISP", get_QRISP_circ_data_function(21, 2, 2 * 5)),
        ("N25_QRISP", get_QRISP_circ_data_function(25, 2, 2 * 5)),
        # ("N33_QRISP", get_QRISP_circ_data_function(33, 2, 2 * 6)),
        # ("N35_QRISP", get_QRISP_circ_data_function(35, 2, 2 * 6)),
        # ("N39_QRISP", get_QRISP_circ_data_function(39, 2, 2 * 6)),
        # ("N45_QRISP", get_QRISP_circ_data_function(45, 2, 2 * 6)),
        # ("N49_QRISP", get_QRISP_circ_data_function(49, 2, 2 * 6)),
        # ("N71_QRISP", get_QRISP_circ_data_function(71, 2, 2 * 7)),
        # ("N711_QRISP", get_QRISP_circ_data_function(711, 2, 2 * 10)),
    ]
    """

    if experiment == 1:

        # neutral atom 1,2 ebit experiments

        result_path_root = f"./out/results"
        weight_list = [
            get_weights_neutral_atom_ebit_duration(ebit_d_ms * 10**6)
            for ebit_d_ms in range(5, 120, 20)
        ]

        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
        circuit_specs = [
            # ("N15", get_N15_circ_data),
            # ("N21B2", get_N21B2_circ_data),
            # ("N21B4", get_N21B4_circ_data),
            # ("N35", get_N35_circ_data),
            ("N15_QRISP", get_QRISP_circ_data_function(15, 2, 2 * 4)),
            ("N21_QRISP", get_QRISP_circ_data_function(21, 2, 2 * 5)),
            ("N25_QRISP", get_QRISP_circ_data_function(25, 2, 2 * 5)),
            # ("N33_QRISP", get_QRISP_circ_data_function(33, 2, 2 * 6)),
            # ("N35_QRISP", get_QRISP_circ_data_function(35, 2, 2 * 6)),
            # ("N39_QRISP", get_QRISP_circ_data_function(39, 2, 2 * 6)),
            # ("N45_QRISP", get_QRISP_circ_data_function(45, 2, 2 * 6)),
            # ("N49_QRISP", get_QRISP_circ_data_function(49, 2, 2 * 6)),
            # ("N71_QRISP", get_QRISP_circ_data_function(71, 2, 2 * 7)),
            # ("N711_QRISP", get_QRISP_circ_data_function(711, 2, 2 * 10)),
            # ("N1311_QRISP", get_QRISP_circ_data_function(1311, 2, 2 * 11)),
            # ("N3111_QRISP", get_QRISP_circ_data_function(3111, 2, 2 * 12)),
            # ("N7111_QRISP", get_QRISP_circ_data_function(7111, 2, 2 * 13)),
            # ("N13111_QRISP", get_QRISP_circ_data_function(13111, 2, 2 * 14)),
            # ("N31111_QRISP", get_QRISP_circ_data_function(31111, 2, 2 * 15)),
            # ("N41111_QRISP", get_QRISP_circ_data_function(41111, 2, 2 * 16)),
            # ("N71111_QRISP", get_QRISP_circ_data_function(71111, 2, 2 * 17)),
            # ("N141111_QRISP", get_QRISP_circ_data_function(141111, 2, 2 * 18)),
            # ("N411111_QRISP", get_QRISP_circ_data_function(411111, 2, 2 * 19)),
            # ("N711111_QRISP", get_QRISP_circ_data_function(711111, 2, 2 * 20)),
            # ("N1311111_QRISP", get_QRISP_circ_data_function(1311111, 2, 2 * 21)),
            # ("N13111111_QRISP", get_QRISP_circ_data_function(13111111, 2, 2 * 24)),
            # ("N141111111_QRISP", get_QRISP_circ_data_function(141111111, 2, 2 * 28)),
            # ("N1411111111_QRISP", get_QRISP_circ_data_function(1411111111, 2, 2 * 31)),
            # (
            #     "N13111111111_QRISP",
            #     get_QRISP_circ_data_function(13111111111, 2, 2 * 34),
            # ),
            # (
            #     "N131111111111_QRISP",
            #     get_QRISP_circ_data_function(131111111111, 2, 2 * 37),
            # ),
            # (
            #     "N711111111111_QRISP",
            #     get_QRISP_circ_data_function(711111111111, 2, 2 * 40),
            # ),
            # (
            #     "N33111111111111_QRISP",
            #     get_QRISP_circ_data_function(33111111111111, 2, 2 * 45),
            # ),
            # (
            #     "N711111111111111_QRISP",
            #     get_QRISP_circ_data_function(711111111111111, 2, 2 * 50),
            # ),
            # largest circuits later with less processes
            # (
            #     "N711111111111111111_QRISP",
            #     get_QRISP_circ_data_function(711111111111111111, 2, 2 * 60),
            # ),
            # (
            #     "N13111111111111111111_QRISP",
            #     get_QRISP_circ_data_function(13111111111111111111, 2, 2 * 64),
            # ),
            # (
            #     "N811111111111111111111_QRISP",
            #     get_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ),
        ]
        EJPP_list = [1, 2, 4]

        n_processes = args.n_processes
        # n_processes = n_processes - 7

        run_experiments_mp(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=True,
            iterative=True,
            alternating=True,
            EJPP_list=EJPP_list,
            n_processes=n_processes,
        )

    if experiment == 2:
        # monolithic experiments

        result_path_root = f"./out/results"
        weight_list = [weights_heron, weights_ionq_forte, weights_neutral_atom]

        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
        circuit_specs = [
            ("N15", get_N15_circ_data),
            ("N21B2", get_N21B2_circ_data),
            ("N21B4", get_N21B4_circ_data),
            ("N35", get_N35_circ_data),
            ("N15_QRISP", get_QRISP_circ_data_function(15, 2, 2 * 4)),
            ("N21_QRISP", get_QRISP_circ_data_function(21, 2, 2 * 5)),
            ("N25_QRISP", get_QRISP_circ_data_function(25, 2, 2 * 5)),
            ("N33_QRISP", get_QRISP_circ_data_function(33, 2, 2 * 6)),
            ("N35_QRISP", get_QRISP_circ_data_function(35, 2, 2 * 6)),
            ("N39_QRISP", get_QRISP_circ_data_function(39, 2, 2 * 6)),
            ("N45_QRISP", get_QRISP_circ_data_function(45, 2, 2 * 6)),
            ("N49_QRISP", get_QRISP_circ_data_function(49, 2, 2 * 6)),
            ("N71_QRISP", get_QRISP_circ_data_function(71, 2, 2 * 7)),
            ("N711_QRISP", get_QRISP_circ_data_function(711, 2, 2 * 10)),
            ("N1311_QRISP", get_QRISP_circ_data_function(1311, 2, 2 * 11)),
            ("N3111_QRISP", get_QRISP_circ_data_function(3111, 2, 2 * 12)),
            ("N7111_QRISP", get_QRISP_circ_data_function(7111, 2, 2 * 13)),
            ("N13111_QRISP", get_QRISP_circ_data_function(13111, 2, 2 * 14)),
            ("N31111_QRISP", get_QRISP_circ_data_function(31111, 2, 2 * 15)),
            ("N41111_QRISP", get_QRISP_circ_data_function(41111, 2, 2 * 16)),
            ("N71111_QRISP", get_QRISP_circ_data_function(71111, 2, 2 * 17)),
            ("N141111_QRISP", get_QRISP_circ_data_function(141111, 2, 2 * 18)),
            ("N411111_QRISP", get_QRISP_circ_data_function(411111, 2, 2 * 19)),
            ("N711111_QRISP", get_QRISP_circ_data_function(711111, 2, 2 * 20)),
            ("N1311111_QRISP", get_QRISP_circ_data_function(1311111, 2, 2 * 21)),
            ("N13111111_QRISP", get_QRISP_circ_data_function(13111111, 2, 2 * 24)),
            ("N141111111_QRISP", get_QRISP_circ_data_function(141111111, 2, 2 * 28)),
            ("N1411111111_QRISP", get_QRISP_circ_data_function(1411111111, 2, 2 * 31)),
            (
                "N13111111111_QRISP",
                get_QRISP_circ_data_function(13111111111, 2, 2 * 34),
            ),
            (
                "N131111111111_QRISP",
                get_QRISP_circ_data_function(131111111111, 2, 2 * 37),
            ),
            (
                "N711111111111_QRISP",
                get_QRISP_circ_data_function(711111111111, 2, 2 * 40),
            ),
            (
                "N33111111111111_QRISP",
                get_QRISP_circ_data_function(33111111111111, 2, 2 * 45),
            ),
            (
                "N711111111111111_QRISP",
                get_QRISP_circ_data_function(711111111111111, 2, 2 * 50),
            ),
            (
                "N711111111111111111_QRISP",
                get_QRISP_circ_data_function(711111111111111111, 2, 2 * 60),
            ),
            (
                "N13111111111111111111_QRISP",
                get_QRISP_circ_data_function(13111111111111111111, 2, 2 * 64),
            ),
            # (
            #     "N811111111111111111111_QRISP",
            #     get_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ),
        ]
        n_processes = min(args.n_processes, len(circuit_specs) * 3)

        run_experiments_mp(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=True,
            iterative=True,
            EJPP_list=[0],
            n_processes=n_processes,
        )

    if experiment == 4:
        # heron_ebit 1,2,3,4 ebit experiments

        result_path_root = f"./out/results"
        weight_list = [
            get_weights_heron_ebit_duration(ebit_d_mus * 10**3)
            for ebit_d_mus in range(10, 1000, 10)
        ]

        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
        circuit_specs = [
            # ("N15", get_N15_circ_data),
            # ("N21B2", get_N21B2_circ_data),
            # ("N21B4", get_N21B4_circ_data),
            # ("N35", get_N35_circ_data),
            # ("N15_QRISP", get_QRISP_circ_data_function(15, 2, 2 * 4)),
            # ("N21_QRISP", get_QRISP_circ_data_function(21, 2, 2 * 5)),
            # ("N25_QRISP", get_QRISP_circ_data_function(25, 2, 2 * 5)),
            # ("N33_QRISP", get_QRISP_circ_data_function(33, 2, 2 * 6)),
            # ("N35_QRISP", get_QRISP_circ_data_function(35, 2, 2 * 6)),
            # ("N39_QRISP", get_QRISP_circ_data_function(39, 2, 2 * 6)),
            # ("N45_QRISP", get_QRISP_circ_data_function(45, 2, 2 * 6)),
            # ("N49_QRISP", get_QRISP_circ_data_function(49, 2, 2 * 6)),
            # ("N71_QRISP", get_QRISP_circ_data_function(71, 2, 2 * 7)),
            # ("N711_QRISP", get_QRISP_circ_data_function(711, 2, 2 * 10)),
            # ("N1311_QRISP", get_QRISP_circ_data_function(1311, 2, 2 * 11)),
            # ("N3111_QRISP", get_QRISP_circ_data_function(3111, 2, 2 * 12)),
            # ("N7111_QRISP", get_QRISP_circ_data_function(7111, 2, 2 * 13)),
            # ("N13111_QRISP", get_QRISP_circ_data_function(13111, 2, 2 * 14)),
            # ("N31111_QRISP", get_QRISP_circ_data_function(31111, 2, 2 * 15)),
            # ("N41111_QRISP", get_QRISP_circ_data_function(41111, 2, 2 * 16)),
            # ("N71111_QRISP", get_QRISP_circ_data_function(71111, 2, 2 * 17)),
            # ("N141111_QRISP", get_QRISP_circ_data_function(141111, 2, 2 * 18)),
            # ("N411111_QRISP", get_QRISP_circ_data_function(411111, 2, 2 * 19)),
            # ("N711111_QRISP", get_QRISP_circ_data_function(711111, 2, 2 * 20)),
            # ("N1311111_QRISP", get_QRISP_circ_data_function(1311111, 2, 2 * 21)),
            # ("N13111111_QRISP", get_QRISP_circ_data_function(13111111, 2, 2 * 24)),
            # ("N141111111_QRISP", get_QRISP_circ_data_function(141111111, 2, 2 * 28)),
            # ("N1411111111_QRISP", get_QRISP_circ_data_function(1411111111, 2, 2 * 31)),
            # (
            #     "N13111111111_QRISP",
            #     get_QRISP_circ_data_function(13111111111, 2, 2 * 34),
            # ),
            # (
            #     "N131111111111_QRISP",
            #     get_QRISP_circ_data_function(131111111111, 2, 2 * 37),
            # ),
            # (
            #     "N711111111111_QRISP",
            #     get_QRISP_circ_data_function(711111111111, 2, 2 * 40),
            # ),
            # (
            #     "N33111111111111_QRISP",
            #     get_QRISP_circ_data_function(33111111111111, 2, 2 * 45),
            # ),
            # (
            #     "N711111111111111_QRISP",
            #     get_QRISP_circ_data_function(711111111111111, 2, 2 * 50),
            # ),
            # largest circuits later with less processes
            (
                "N711111111111111111_QRISP",
                get_QRISP_circ_data_function(711111111111111111, 2, 2 * 60),
            ),
            (
                "N13111111111111111111_QRISP",
                get_QRISP_circ_data_function(13111111111111111111, 2, 2 * 64),
            ),
            # (
            #     "N811111111111111111111_QRISP",
            #     get_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ),
        ]

        EJPP_list = [6, 8, 10]

        # n_processes = min(args.n_processes, len(circuit_specs) * len(EJPP_list) * 3)
        n_processes = args.n_processes

        n_processes = n_processes - 8
        run_experiments_mp(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=False,
            iterative=False,
            alternating=True,
            EJPP_list=EJPP_list,
            n_processes=n_processes,
        )

    if experiment == 5:
        # ionq_forte 1,2,3,4 ebit experiments

        result_path_root = f"./out/results"
        weight_list = [
            get_weights_ionq_forte_ebit_duration(ebit_d_ms * 10**6)
            for ebit_d_ms in range(5, 20000, 500)  # 5ms to 20s
        ]

        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
        circuit_specs = [
            # ("N15", get_N15_circ_data),
            # ("N21B2", get_N21B2_circ_data),
            # ("N21B4", get_N21B4_circ_data),
            # ("N35", get_N35_circ_data),
            # ("N15_QRISP", get_QRISP_circ_data_function(15, 2, 2 * 4)),
            # ("N21_QRISP", get_QRISP_circ_data_function(21, 2, 2 * 5)),
            # ("N25_QRISP", get_QRISP_circ_data_function(25, 2, 2 * 5)),
            # ("N33_QRISP", get_QRISP_circ_data_function(33, 2, 2 * 6)),
            # ("N35_QRISP", get_QRISP_circ_data_function(35, 2, 2 * 6)),
            # ("N39_QRISP", get_QRISP_circ_data_function(39, 2, 2 * 6)),
            # ("N45_QRISP", get_QRISP_circ_data_function(45, 2, 2 * 6)),
            # ("N49_QRISP", get_QRISP_circ_data_function(49, 2, 2 * 6)),
            # ("N71_QRISP", get_QRISP_circ_data_function(71, 2, 2 * 7)),
            # ("N711_QRISP", get_QRISP_circ_data_function(711, 2, 2 * 10)),
            # ("N1311_QRISP", get_QRISP_circ_data_function(1311, 2, 2 * 11)),
            # ("N3111_QRISP", get_QRISP_circ_data_function(3111, 2, 2 * 12)),
            # ("N7111_QRISP", get_QRISP_circ_data_function(7111, 2, 2 * 13)),
            # ("N13111_QRISP", get_QRISP_circ_data_function(13111, 2, 2 * 14)),
            # ("N31111_QRISP", get_QRISP_circ_data_function(31111, 2, 2 * 15)),
            # ("N41111_QRISP", get_QRISP_circ_data_function(41111, 2, 2 * 16)),
            # ("N71111_QRISP", get_QRISP_circ_data_function(71111, 2, 2 * 17)),
            # ("N141111_QRISP", get_QRISP_circ_data_function(141111, 2, 2 * 18)),
            # ("N411111_QRISP", get_QRISP_circ_data_function(411111, 2, 2 * 19)),
            # ("N711111_QRISP", get_QRISP_circ_data_function(711111, 2, 2 * 20)),
            # ("N1311111_QRISP", get_QRISP_circ_data_function(1311111, 2, 2 * 21)),
            # ("N13111111_QRISP", get_QRISP_circ_data_function(13111111, 2, 2 * 24)),
            # ("N141111111_QRISP", get_QRISP_circ_data_function(141111111, 2, 2 * 28)),
            # ("N1411111111_QRISP", get_QRISP_circ_data_function(1411111111, 2, 2 * 31)),
            # (
            #     "N13111111111_QRISP",
            #     get_QRISP_circ_data_function(13111111111, 2, 2 * 34),
            # ),
            # (
            #     "N131111111111_QRISP",
            #     get_QRISP_circ_data_function(131111111111, 2, 2 * 37),
            # ),
            # (
            #     "N711111111111_QRISP",
            #     get_QRISP_circ_data_function(711111111111, 2, 2 * 40),
            # ),
            # (
            #     "N33111111111111_QRISP",
            #     get_QRISP_circ_data_function(33111111111111, 2, 2 * 45),
            # ),
            # (
            #     "N711111111111111_QRISP",
            #     get_QRISP_circ_data_function(711111111111111, 2, 2 * 50),
            # ),
            # largest circuits later with less processes
            (
                "N711111111111111111_QRISP",
                get_QRISP_circ_data_function(711111111111111111, 2, 2 * 60),
            ),
            (
                "N13111111111111111111_QRISP",
                get_QRISP_circ_data_function(13111111111111111111, 2, 2 * 64),
            ),
            # (
            #     "N811111111111111111111_QRISP",
            #     get_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ),
        ]

        EJPP_list = [6, 8, 10]

        # n_processes = min(args.n_processes, len(circuit_specs) * len(EJPP_list) * 3)
        n_processes = args.n_processes

        n_processes = n_processes - 8
        run_experiments_mp(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=False,
            iterative=False,
            alternating=True,
            EJPP_list=EJPP_list,
            n_processes=n_processes,
        )

    if experiment == 6:
        # monolithic dlog experiments

        result_path_root = f"./out/results_log"
        weight_list = [weights_heron, weights_ionq_forte, weights_neutral_atom]

        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"
        circuit_specs = [
            # ("N15", get_N15_circ_data),
            # ("N21B2", get_N21B2_circ_data),
            # ("N21B4", get_N21B4_circ_data),
            # ("N35", get_N35_circ_data),
            ("N15_QRISP", get_dlog_QRISP_circ_data_function(15, 2, 2)),  # 4bit
            ("N21_QRISP", get_dlog_QRISP_circ_data_function(21, 2, 3)),  # 5bit
            ("N25_QRISP", get_dlog_QRISP_circ_data_function(25, 2, 3)),  # 5bit
            ("N33_QRISP", get_dlog_QRISP_circ_data_function(33, 2, 3)),  # 6 bit
            ("N35_QRISP", get_dlog_QRISP_circ_data_function(35, 2, 3)),  # 6 bit
            ("N39_QRISP", get_dlog_QRISP_circ_data_function(39, 2, 3)),  # 6 bit
            ("N45_QRISP", get_dlog_QRISP_circ_data_function(45, 2, 3)),  # 6 bit
            ("N49_QRISP", get_dlog_QRISP_circ_data_function(49, 2, 3)),  # 6 bit
            ("N71_QRISP", get_dlog_QRISP_circ_data_function(71, 2, 4)),  # 7 bit
            ("N711_QRISP", get_dlog_QRISP_circ_data_function(711, 2, 5)),  # 10 bit
            ("N1311_QRISP", get_dlog_QRISP_circ_data_function(1311, 2, 6)),  # 11 bit
            ("N3111_QRISP", get_dlog_QRISP_circ_data_function(3111, 2, 6)),  # 12 bit
            ("N7111_QRISP", get_dlog_QRISP_circ_data_function(7111, 2, 7)),  # 13 bit
            ("N13111_QRISP", get_dlog_QRISP_circ_data_function(13111, 2, 7)),  # 14 bit
            ("N31111_QRISP", get_dlog_QRISP_circ_data_function(31111, 2, 8)),  # 15 bit
            ("N41111_QRISP", get_dlog_QRISP_circ_data_function(41111, 2, 8)),  # 16 bit
            ("N71111_QRISP", get_dlog_QRISP_circ_data_function(71111, 2, 9)),  # 17 bit
            (
                "N141111_QRISP",
                get_dlog_QRISP_circ_data_function(141111, 2, 9),
            ),  # 18 bit
            (
                "N411111_QRISP",
                get_dlog_QRISP_circ_data_function(411111, 2, 10),
            ),  # 19 bit
            (
                "N711111_QRISP",
                get_dlog_QRISP_circ_data_function(711111, 2, 10),
            ),  # 20 bit
            (
                "N1311111_QRISP",
                get_dlog_QRISP_circ_data_function(1311111, 2, 11),
            ),  # 21 bit
            (
                "N13111111_QRISP",
                get_dlog_QRISP_circ_data_function(13111111, 2, 12),
            ),  # 24 bit
            (
                "N141111111_QRISP",
                get_dlog_QRISP_circ_data_function(141111111, 2, 14),
            ),  # 28 bit
            (
                "N1411111111_QRISP",
                get_dlog_QRISP_circ_data_function(1411111111, 2, 16),
            ),  # 31 bit
            (
                "N13111111111_QRISP",
                get_dlog_QRISP_circ_data_function(13111111111, 2, 17),
            ),  # 34 bit
            (
                "N131111111111_QRISP",
                get_dlog_QRISP_circ_data_function(131111111111, 2, 19),
            ),  # 37 bit
            (
                "N711111111111_QRISP",
                get_dlog_QRISP_circ_data_function(711111111111, 2, 20),
            ),  # 40 bit
            (
                "N33111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(33111111111111, 2, 23),
            ),  # 45 bit
            (
                "N711111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(711111111111111, 2, 25),
            ),  # 50 bit
            (
                "N711111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(711111111111111111, 2, 30),
            ),  # 60 bit
            (
                "N13111111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(13111111111111111111, 2, 32),
            ),  # 64 bit
            # (
            #     "N811111111111111111111_QRISP",
            #     get_dlog_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ), # 10 bit
        ]
        n_processes = min(args.n_processes, len(circuit_specs) * 3)

        run_experiments_mp_dlog(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=True,
            iterative=True,
            alternating=True,
            double_iterative=True,
            three_cyclic=True,
            EJPP_list=[0],
            n_processes=n_processes,
        )

    if experiment == 7:
        # distributed dlog experiments neutral atom

        result_path_root = f"./out/results_dlog"
        # weight_list = [weights_heron, weights_ionq_forte, weights_neutral_atom]
        weight_list = [
            get_weights_neutral_atom_ebit_duration(ebit_d_ms * 10**6)
            for ebit_d_ms in range(5, 120, 5)
        ]

        n_processes = args.n_processes

        EJPP_list = [6, 8, 10]

        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"

        # circuit_specs = [
        #     # ("N15", get_N15_circ_data),
        #     # ("N21B2", get_N21B2_circ_data),
        #     # ("N21B4", get_N21B4_circ_data),
        #     # ("N35", get_N35_circ_data),
        #     ("N15_QRISP", get_dlog_QRISP_circ_data_function(15, 2, 2)),  # 4bit
        #     ("N21_QRISP", get_dlog_QRISP_circ_data_function(21, 2, 3)),  # 5bit
        #     ("N25_QRISP", get_dlog_QRISP_circ_data_function(25, 2, 3)),  # 5bit
        #     ("N33_QRISP", get_dlog_QRISP_circ_data_function(33, 2, 3)),  # 6 bit
        #     ("N35_QRISP", get_dlog_QRISP_circ_data_function(35, 2, 3)),  # 6 bit
        #     ("N39_QRISP", get_dlog_QRISP_circ_data_function(39, 2, 3)),  # 6 bit
        #     ("N45_QRISP", get_dlog_QRISP_circ_data_function(45, 2, 3)),  # 6 bit
        #     ("N49_QRISP", get_dlog_QRISP_circ_data_function(49, 2, 3)),  # 6 bit
        #     ("N71_QRISP", get_dlog_QRISP_circ_data_function(71, 2, 4)),  # 7 bit
        #     ("N711_QRISP", get_dlog_QRISP_circ_data_function(711, 2, 5)),  # 10 bit
        #     ("N1311_QRISP", get_dlog_QRISP_circ_data_function(1311, 2, 6)),  # 11 bit
        #     ("N3111_QRISP", get_dlog_QRISP_circ_data_function(3111, 2, 6)),  # 12 bit
        #     ("N7111_QRISP", get_dlog_QRISP_circ_data_function(7111, 2, 7)),  # 13 bit
        #     ("N13111_QRISP", get_dlog_QRISP_circ_data_function(13111, 2, 7)),  # 14 bit
        #     ("N31111_QRISP", get_dlog_QRISP_circ_data_function(31111, 2, 8)),  # 15 bit
        #     ("N41111_QRISP", get_dlog_QRISP_circ_data_function(41111, 2, 8)),  # 16 bit
        #     ("N71111_QRISP", get_dlog_QRISP_circ_data_function(71111, 2, 9)),  # 17 bit
        #     (
        #         "N141111_QRISP",
        #         get_dlog_QRISP_circ_data_function(141111, 2, 9),
        #     ),  # 18 bit
        #     (
        #         "N411111_QRISP",
        #         get_dlog_QRISP_circ_data_function(411111, 2, 10),
        #     ),  # 19 bit
        #     (
        #         "N711111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111, 2, 10),
        #     ),  # 20 bit
        #     (
        #         "N1311111_QRISP",
        #         get_dlog_QRISP_circ_data_function(1311111, 2, 11),
        #     ),  # 21 bit
        #     (
        #         "N13111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(13111111, 2, 12),
        #     ),  # 24 bit
        #     (
        #         "N141111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(141111111, 2, 14),
        #     ),  # 28 bit
        #     (
        #         "N1411111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(1411111111, 2, 16),
        #     ),  # 31 bit
        #     (
        #         "N13111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(13111111111, 2, 17),
        #     ),  # 34 bit
        #     (
        #         "N131111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(131111111111, 2, 19),
        #     ),  # 37 bit
        #     (
        #         "N711111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111111111, 2, 20),
        #     ),  # 40 bit
        #     (
        #         "N33111111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(33111111111111, 2, 23),
        #     ),  # 45 bit
        #     (
        #         "N711111111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111111111111, 2, 25),
        #     ),  # 50 bit
        # ]

        # # first run smaller circuits with more processes
        # run_experiments_mp_dlog(
        #     weight_list=weight_list,
        #     result_path_root=result_path_root,
        #     circuit_specs=circuit_specs,
        #     regular=True,
        #     iterative=True,
        #     alternating=True,
        #     double_iterative=True,
        #     three_cyclic=True,
        #     EJPP_list=EJPP_list,
        #     n_processes=n_processes,
        # )

        circuit_specs = [
            (
                "N711111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(711111111111111111, 2, 30),
            ),  # 60 bit
            (
                "N13111111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(13111111111111111111, 2, 32),
            ),  # 64 bit
            # (
            #     "N811111111111111111111_QRISP",
            #     get_dlog_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ),  # 10 bit
        ]
        # then larger circuits with less because of memory constraints
        run_experiments_mp_dlog(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=True,
            iterative=True,
            alternating=True,
            double_iterative=True,
            three_cyclic=True,
            EJPP_list=EJPP_list,
            n_processes=n_processes - 7,
        )

    if experiment == 8:
        # distributed dlog experiments ion trap

        result_path_root = f"./out/results_dlog"
        # weight_list = [weights_heron, weights_ionq_forte, weights_neutral_atom]
        weight_list = [
            get_weights_ionq_forte_ebit_duration(ebit_d_ms * 10**6)
            for ebit_d_ms in range(5, 20000, 500)  # 5ms to 20s
        ]

        n_processes = args.n_processes

        EJPP_list = [6, 8, 10]

        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"

        # circuit_specs = [
        #     # ("N15", get_N15_circ_data),
        #     # ("N21B2", get_N21B2_circ_data),
        #     # ("N21B4", get_N21B4_circ_data),
        #     # ("N35", get_N35_circ_data),
        #     ("N15_QRISP", get_dlog_QRISP_circ_data_function(15, 2, 2)),  # 4bit
        #     ("N21_QRISP", get_dlog_QRISP_circ_data_function(21, 2, 3)),  # 5bit
        #     ("N25_QRISP", get_dlog_QRISP_circ_data_function(25, 2, 3)),  # 5bit
        #     ("N33_QRISP", get_dlog_QRISP_circ_data_function(33, 2, 3)),  # 6 bit
        #     ("N35_QRISP", get_dlog_QRISP_circ_data_function(35, 2, 3)),  # 6 bit
        #     ("N39_QRISP", get_dlog_QRISP_circ_data_function(39, 2, 3)),  # 6 bit
        #     ("N45_QRISP", get_dlog_QRISP_circ_data_function(45, 2, 3)),  # 6 bit
        #     ("N49_QRISP", get_dlog_QRISP_circ_data_function(49, 2, 3)),  # 6 bit
        #     ("N71_QRISP", get_dlog_QRISP_circ_data_function(71, 2, 4)),  # 7 bit
        #     ("N711_QRISP", get_dlog_QRISP_circ_data_function(711, 2, 5)),  # 10 bit
        #     ("N1311_QRISP", get_dlog_QRISP_circ_data_function(1311, 2, 6)),  # 11 bit
        #     ("N3111_QRISP", get_dlog_QRISP_circ_data_function(3111, 2, 6)),  # 12 bit
        #     ("N7111_QRISP", get_dlog_QRISP_circ_data_function(7111, 2, 7)),  # 13 bit
        #     ("N13111_QRISP", get_dlog_QRISP_circ_data_function(13111, 2, 7)),  # 14 bit
        #     ("N31111_QRISP", get_dlog_QRISP_circ_data_function(31111, 2, 8)),  # 15 bit
        #     ("N41111_QRISP", get_dlog_QRISP_circ_data_function(41111, 2, 8)),  # 16 bit
        #     ("N71111_QRISP", get_dlog_QRISP_circ_data_function(71111, 2, 9)),  # 17 bit
        #     (
        #         "N141111_QRISP",
        #         get_dlog_QRISP_circ_data_function(141111, 2, 9),
        #     ),  # 18 bit
        #     (
        #         "N411111_QRISP",
        #         get_dlog_QRISP_circ_data_function(411111, 2, 10),
        #     ),  # 19 bit
        #     (
        #         "N711111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111, 2, 10),
        #     ),  # 20 bit
        #     (
        #         "N1311111_QRISP",
        #         get_dlog_QRISP_circ_data_function(1311111, 2, 11),
        #     ),  # 21 bit
        #     (
        #         "N13111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(13111111, 2, 12),
        #     ),  # 24 bit
        #     (
        #         "N141111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(141111111, 2, 14),
        #     ),  # 28 bit
        #     (
        #         "N1411111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(1411111111, 2, 16),
        #     ),  # 31 bit
        #     (
        #         "N13111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(13111111111, 2, 17),
        #     ),  # 34 bit
        #     (
        #         "N131111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(131111111111, 2, 19),
        #     ),  # 37 bit
        #     (
        #         "N711111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111111111, 2, 20),
        #     ),  # 40 bit
        #     (
        #         "N33111111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(33111111111111, 2, 23),
        #     ),  # 45 bit
        #     (
        #         "N711111111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111111111111, 2, 25),
        #     ),  # 50 bit
        # ]

        # # first run smaller circuits with more processes
        # run_experiments_mp_dlog(
        #     weight_list=weight_list,
        #     result_path_root=result_path_root,
        #     circuit_specs=circuit_specs,
        #     regular=False,
        #     iterative=False,
        #     alternating=True,
        #     double_iterative=False,
        #     three_cyclic=False,
        #     EJPP_list=EJPP_list,
        #     n_processes=n_processes,
        # )

        circuit_specs = [
            (
                "N711111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(711111111111111111, 2, 30),
            ),  # 60 bit
            (
                "N13111111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(13111111111111111111, 2, 32),
            ),  # 64 bit
            # (
            #     "N811111111111111111111_QRISP",
            #     get_dlog_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ), # 10 bit
        ]

        # then larger circuits with less because of memory constraints
        run_experiments_mp_dlog(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=False,
            iterative=False,
            alternating=True,
            double_iterative=False,
            three_cyclic=False,
            EJPP_list=EJPP_list,
            n_processes=n_processes - 8,
        )

    if experiment == 9:
        # distributed dlog experiments super conducting ibm

        result_path_root = f"./out/results_dlog"
        # weight_list = [weights_heron, weights_ionq_forte, weights_neutral_atom]
        weight_list = [
            get_weights_heron_ebit_duration(ebit_d_mus * 10**3)
            for ebit_d_mus in range(10, 1000, 10)
        ]

        n_processes = args.n_processes

        EJPP_list = [6, 8, 10]
        # result_path = f"./src/semi_iterative_comparison/results_CU/depth_{weights.label}/"

        # circuit_specs = [
        #     # ("N15", get_N15_circ_data),
        #     # ("N21B2", get_N21B2_circ_data),
        #     # ("N21B4", get_N21B4_circ_data),
        #     # ("N35", get_N35_circ_data),
        #     ("N15_QRISP", get_dlog_QRISP_circ_data_function(15, 2, 2)),  # 4bit
        #     ("N21_QRISP", get_dlog_QRISP_circ_data_function(21, 2, 3)),  # 5bit
        #     ("N25_QRISP", get_dlog_QRISP_circ_data_function(25, 2, 3)),  # 5bit
        #     ("N33_QRISP", get_dlog_QRISP_circ_data_function(33, 2, 3)),  # 6 bit
        #     ("N35_QRISP", get_dlog_QRISP_circ_data_function(35, 2, 3)),  # 6 bit
        #     ("N39_QRISP", get_dlog_QRISP_circ_data_function(39, 2, 3)),  # 6 bit
        #     ("N45_QRISP", get_dlog_QRISP_circ_data_function(45, 2, 3)),  # 6 bit
        #     ("N49_QRISP", get_dlog_QRISP_circ_data_function(49, 2, 3)),  # 6 bit
        #     ("N71_QRISP", get_dlog_QRISP_circ_data_function(71, 2, 4)),  # 7 bit
        #     ("N711_QRISP", get_dlog_QRISP_circ_data_function(711, 2, 5)),  # 10 bit
        #     ("N1311_QRISP", get_dlog_QRISP_circ_data_function(1311, 2, 6)),  # 11 bit
        #     ("N3111_QRISP", get_dlog_QRISP_circ_data_function(3111, 2, 6)),  # 12 bit
        #     ("N7111_QRISP", get_dlog_QRISP_circ_data_function(7111, 2, 7)),  # 13 bit
        #     ("N13111_QRISP", get_dlog_QRISP_circ_data_function(13111, 2, 7)),  # 14 bit
        #     ("N31111_QRISP", get_dlog_QRISP_circ_data_function(31111, 2, 8)),  # 15 bit
        #     ("N41111_QRISP", get_dlog_QRISP_circ_data_function(41111, 2, 8)),  # 16 bit
        #     ("N71111_QRISP", get_dlog_QRISP_circ_data_function(71111, 2, 9)),  # 17 bit
        #     (
        #         "N141111_QRISP",
        #         get_dlog_QRISP_circ_data_function(141111, 2, 9),
        #     ),  # 18 bit
        #     (
        #         "N411111_QRISP",
        #         get_dlog_QRISP_circ_data_function(411111, 2, 10),
        #     ),  # 19 bit
        #     (
        #         "N711111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111, 2, 10),
        #     ),  # 20 bit
        #     (
        #         "N1311111_QRISP",
        #         get_dlog_QRISP_circ_data_function(1311111, 2, 11),
        #     ),  # 21 bit
        #     (
        #         "N13111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(13111111, 2, 12),
        #     ),  # 24 bit
        #     (
        #         "N141111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(141111111, 2, 14),
        #     ),  # 28 bit
        #     (
        #         "N1411111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(1411111111, 2, 16),
        #     ),  # 31 bit
        #     (
        #         "N13111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(13111111111, 2, 17),
        #     ),  # 34 bit
        #     (
        #         "N131111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(131111111111, 2, 19),
        #     ),  # 37 bit
        #     (
        #         "N711111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111111111, 2, 20),
        #     ),  # 40 bit
        #     (
        #         "N33111111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(33111111111111, 2, 23),
        #     ),  # 45 bit
        #     (
        #         "N711111111111111_QRISP",
        #         get_dlog_QRISP_circ_data_function(711111111111111, 2, 25),
        #     ),  # 50 bit
        # ]

        # # first run smaller circuits with more processes
        # run_experiments_mp_dlog(
        #     weight_list=weight_list,
        #     result_path_root=result_path_root,
        #     circuit_specs=circuit_specs,
        #     regular=False,
        #     iterative=False,
        #     alternating=True,
        #     double_iterative=False,
        #     three_cyclic=False,
        #     EJPP_list=EJPP_list,
        #     n_processes=n_processes,
        # )

        circuit_specs = [
            (
                "N711111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(711111111111111111, 2, 30),
            ),  # 60 bit
            (
                "N13111111111111111111_QRISP",
                get_dlog_QRISP_circ_data_function(13111111111111111111, 2, 32),
            ),  # 64 bit
            # (
            #     "N811111111111111111111_QRISP",
            #     get_dlog_QRISP_circ_data_function(811111111111111111111, 2, 2 * 70),
            # ), # 10 bit
        ]

        # then larger circuits with less because of memory constraints
        run_experiments_mp_dlog(
            weight_list=weight_list,
            result_path_root=result_path_root,
            circuit_specs=circuit_specs,
            regular=False,
            iterative=False,
            alternating=True,
            double_iterative=False,
            three_cyclic=False,
            EJPP_list=EJPP_list,
            n_processes=n_processes - 8,
        )
