from qiskit import *
from distributed_shor.utils import *
from matplotlib import pyplot as plt
from qiskit.visualization import plot_histogram
from distributed_shor.noise_utils import homogenous_noise_model, BackendSnapshot
from qiskit_aer import AerSimulator
from distributed_shor.circ_stat_utils import (
    get_moment_circ,
    get_transpiled_circuit,
)
from qiskit.transpiler.passes import RemoveBarriers
from distributed_shor.U_ops_from_qrisp import (
    get_U_list,
    get_QRISP_circ_data_function,
    get_dlog_QRISP_circ_data_function,
)
from distributed_shor.ibmq_experiments import start_experiment, read_results
from distributed_shor.circ_stat_utils import (
    depths_from_circ,
    count_circuit_ops,
    get_transpiled_circuit,
)
from distributed_shor.noise_utils import get_basis_gates
from distributed_shor.graph_utils import CircuitGraph
import time
import random
from qiskit.circuit.library import XGate
import sys

"""
Structure this script in classes of factoring N=21 circuits from https://www.nature.com/articles/s41598-021-95973-w by varying QFT implementations (non iterative -> iterative QFT). Each section has a version w/ and wo/ EJPP horizontal cut.


"""

# ----------------------------------------
# This section investigates standard monolithic Shor circuit


def regular_circ(
    U_list: list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    EJPP: int = 0,
    swap: bool = False,
    quantum_qft: bool = False,
) -> QuantumCircuit:
    """
    builds standard monolithic Shor for given list of U operations
    """
    control_reg = QuantumRegister(top_c)
    root_communication = QuantumRegister(EJPP)
    target_communication = QuantumRegister(EJPP)
    target_register = QuantumRegister(bot_c)
    classical_memory = ClassicalRegister(EJPP)
    measure_register = ClassicalRegister(top_c)
    circ = (
        QuantumCircuit(
            control_reg[:]
            + root_communication[:]
            + target_communication[:]
            + target_register[:],
            classical_memory[:] + measure_register[:],
        )
        if EJPP != 0
        else QuantumCircuit(control_reg[:] + target_register[:], measure_register[:])
    )
    if init_one:
        circ.x(target_register[init_ind])
    circ.h(control_reg)
    circ.barrier()
    for i in range(top_c):
        if EJPP != 0:
            append_EJPP_start(
                circ=circ,
                root=[control_reg[i]],
                root_communication=[root_communication[i % EJPP]],
                target_communication=[target_communication[i % EJPP]],
                classical_memory=[classical_memory[i % EJPP]],
            )
            circ.barrier()
        U_list[i](
            circ=circ,
            control=target_communication[i % EJPP] if EJPP != 0 else control_reg[i],
            target=target_register[:],
        )
        circ.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=circ,
                root=[control_reg[i]],
                target_communication=[target_communication[i % EJPP]],
                classical_memory=[classical_memory[i % EJPP]],
            )
            circ.barrier()
        if quantum_qft == False:
            for k in range(i):
                circ.p(m.pi / 2 ** (k + 1), control_reg[i]).c_if(
                    measure_register[i - k - 1], 1
                )
            circ.h(control_reg[i])
            append_measure(
                circ=circ,
                qubit=[control_reg[i]],
                clbit=[measure_register[i]],
                duration=DURATION_MEASURE,
            )
            circ.barrier()
    if quantum_qft:
        circ.append(qft_gate(n=top_c, draw=False, swap=swap), control_reg[::-1])
        circ.barrier()
        append_measure(
            circ=circ,
            qubit=control_reg[:],
            clbit=measure_register[:],
            duration=DURATION_MEASURE,
        )
    if draw:
        print(circ.decompose())
    return circ


def run_monolithic(
    U_list: list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    noise: bool = False,
    shots: int = 10000,
    EJPP: int = 0,
    quantum_qft: bool = False,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    control_reg = QuantumRegister(top_c)
    root_communication = QuantumRegister(EJPP)
    target_communication = QuantumRegister(EJPP)
    target_register = QuantumRegister(bot_c)
    classical_memory = ClassicalRegister(EJPP)
    measure_register = ClassicalRegister(top_c)
    qr = (
        control_reg[:]
        + root_communication[:]
        + target_communication[:]
        + target_register[:]
        if EJPP != 0
        else control_reg[:] + target_register[:]
    )
    cr = classical_memory[:] + measure_register[:] if EJPP != 0 else measure_register[:]
    circ = QuantumCircuit(qr, cr)
    circ.append(
        regular_circ(
            U_list=U_list,
            top_c=top_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            draw=draw,
            EJPP=EJPP,
            quantum_qft=quantum_qft,
        ),
        qr,
        cr,
    )
    circ = get_moment_circ(get_transpiled_circuit(circ))
    results = run_circuit(
        circ,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP != 0:
        for key in results.keys():
            # removing additional 1 here from key as whitespace was included
            cut_results[key[0 : -EJPP - 1]] = (
                cut_results.get(key[0 : -EJPP - 1], 0) + results[key]
            )
    else:
        cut_results = results
    return circ, cut_results


# ---------------------------------------------------------------
# section for iterative QFT which uses only single qubit on control register at each time instance.
def iterative_IQFT_circ(
    qc: QuantumCircuit,
    qr,
    cr,
    i: int,
    measuring_and_reset: bool = True,
    phase_minus: bool = True,
    reset=True,
) -> QuantumCircuit:
    """
    one block of iterative inverse QFT:

    at iteration i we assume that i previous bits have already been measured
    they are stored in cr + 1 additional classical bit to save this iterations measurement
    """
    cr_read = cr[:-1]
    cr_write = cr[-1]

    # do the phase corrections, conditioned on the classical measurement results
    if phase_minus:
        for k in range(i):
            qc.p(-m.pi / 2 ** (k + 1), qr).c_if(cr_read[i - k - 1], 1)
    else:
        for k in range(i):
            qc.p(m.pi / 2 ** (k + 1), qr).c_if(cr_read[i - k - 1], 1)

    qc.h(qr)
    if measuring_and_reset:
        append_measure(circ=qc, qubit=qr, clbit=[cr_write], duration=DURATION_MEASURE)
        if reset:
            append_reset(circ=qc, qubit=qr[:], duration=DURATION_RESET)
    return qc


def iterative_QPE_circ(
    U_list: list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    swap: bool = False,
    EJPP: int = 0,
) -> QuantumCircuit:
    qr = QuantumRegister(bot_c + 1 + 2 * EJPP)
    cr = ClassicalRegister(top_c + EJPP)

    top_r = qr[0]
    bot_r = qr[1 + 2 * EJPP :]
    if EJPP != 0:
        cl_mem = cr[-EJPP:]
        com_root = qr[1 : EJPP + 1]
        com_targ = qr[EJPP + 1 : 2 * EJPP + 1]
    qc = QuantumCircuit(qr[:], cr)
    if swap:
        cr = cr[::-1]
    if init_one:
        qc.x(bot_r[init_ind])
    for i in range(top_c):
        qc.h(top_r)
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r],
                root_communication=[com_root[i % EJPP]],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()
        U_list[i](
            circ=qc, control=com_targ[i % EJPP] if EJPP != 0 else top_r, target=bot_r[:]
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()

        if i < top_c - 1:
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(qc, [top_r], cr[: (i + 1)], i, reset=reset)
        qc.barrier()
    return qc


def run_IQFT(
    U_list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    noise: bool = False,
    shots: int = 10000,
    EJPP: int = 0,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    qc = iterative_QPE_circ(
        U_list=U_list,
        top_c=top_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
    )
    qc = get_moment_circ(get_transpiled_circuit(qc))
    results = run_circuit(
        qc,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    if draw:
        print(qc)
    return qc, cut_results


# --------------------------------------------------------------
# Add section here for top_c - 1 control register qubits by seperating QFT in one iterative step and a top_c -1 qubit QFT.
# do not have multiple ebit option here so far


def semi_iterative_1_n_circuit(
    U_list: list,
    top_c: int,
    bot_c: int,
    swap=False,
    EJPP=False,
    init_one: bool = False,
    init_ind: int = -1,
) -> QuantumCircuit:
    qr = (
        QuantumRegister(bot_c + top_c + 1)
        if EJPP
        else QuantumRegister(bot_c + top_c - 1)
    )
    cr = ClassicalRegister(top_c + 1) if EJPP else ClassicalRegister(top_c)
    top_r = qr[0 : top_c - 1]
    bot_r = qr[top_c + 1 :] if EJPP else qr[top_c - 1 :]
    if EJPP:
        cl_mem = cr[-1]
        com_root = qr[top_c - 1]
        com_targ = qr[top_c]
    qc = QuantumCircuit(qr[:], cr)
    if swap:
        cr = cr[::-1]
    qc.h(top_r[0])
    if init_one:
        qc.x(bot_r[init_ind])
    qc.barrier()
    if EJPP:
        append_EJPP_start(
            circ=qc,
            root=[top_r[0]],
            root_communication=[com_root],
            target_communication=[com_targ],
            classical_memory=[cl_mem],
        )
        qc.barrier()
    U_list[0](circ=qc, control=com_targ if EJPP else top_r[0], target=bot_r[:])
    qc.barrier()
    if EJPP:
        append_EJPP_end(
            circ=qc,
            root=[top_r[0]],
            target_communication=[com_targ],
            classical_memory=[cl_mem],
        )
        qc.barrier()
    qc = iterative_IQFT_circ(qc, [top_r[0]], cr[:1], 0)
    qc.barrier()
    qc.h(top_r)
    qc.barrier()
    for i in range(top_c - 1):
        if EJPP:
            append_EJPP_start(
                circ=qc,
                root=[top_r[i]],
                root_communication=[com_root],
                target_communication=[com_targ],
                classical_memory=[cl_mem],
            )
            qc.barrier()
        U_list[i + 1](circ=qc, control=com_targ if EJPP else top_r[i], target=bot_r[:])
        qc.barrier()
        if EJPP:
            append_EJPP_end(
                circ=qc,
                root=[top_r[i]],
                target_communication=[com_targ],
                classical_memory=[cl_mem],
            )
            qc.barrier()
        qc.p(m.pi / pow(2, i + 1), top_r[i]).c_if(cr[0], 1)
        qc.barrier()
    qc.append(qft_gate(n=top_c - 1, draw=False, swap=False), top_r[::-1])
    qc.barrier()
    append_measure(
        circ=qc,
        qubit=top_r,
        clbit=cr[1:-1] if EJPP else cr[1:],
        duration=DURATION_MEASURE,
    )
    return qc


def semi_iterative_n_1_circuit(
    U_list: list,
    top_c: int,
    bot_c: int,
    swap=False,
    EJPP=False,
    init_one: bool = False,
    init_ind: int = -1,
) -> QuantumCircuit:
    qr = (
        QuantumRegister(bot_c + top_c + 1)
        if EJPP
        else QuantumRegister(bot_c + top_c - 1)
    )
    cr = ClassicalRegister(top_c + 1) if EJPP else ClassicalRegister(top_c)
    top_r = qr[: top_c - 1]
    bot_r = qr[top_c + 1 :] if EJPP else qr[top_c - 1 :]
    if EJPP:
        cl_mem = cr[-1]
        com_root = qr[top_c - 1]
        com_targ = qr[top_c]
    qc = QuantumCircuit(qr[:], cr)
    qc.h(top_r[:])
    if init_one:
        qc.x(bot_r[init_ind])
    qc.barrier()
    for i in range(top_c - 1):
        if EJPP:
            append_EJPP_start(
                circ=qc,
                root=[top_r[i]],
                root_communication=[com_root],
                target_communication=[com_targ],
                classical_memory=[cl_mem],
            )
            qc.barrier()
        U_list[i](circ=qc, control=com_targ if EJPP else top_r[i], target=bot_r[:])
        qc.barrier()
        if EJPP:
            append_EJPP_end(
                circ=qc,
                root=[top_r[i]],
                target_communication=[com_targ],
                classical_memory=[cl_mem],
            )
            qc.barrier()
    qc.append(qft_gate(n=top_c - 1, draw=False, swap=False), top_r[::-1])
    qc.barrier()
    append_measure(
        circ=qc,
        qubit=top_r,
        clbit=cr[0:-2] if EJPP else cr[0:-1],
        duration=DURATION_MEASURE,
    )
    append_reset(circ=qc, qubit=[top_r[0]], duration=DURATION_RESET)
    qc.h(top_r[0])
    qc.barrier()
    if EJPP:
        append_EJPP_start(
            circ=qc,
            root=[top_r[0]],
            root_communication=[com_root],
            target_communication=[com_targ],
            classical_memory=[cl_mem],
        )
        qc.barrier()
    U_list[-1](circ=qc, control=com_targ if EJPP else top_r[0], target=bot_r[:])
    qc.barrier()
    if EJPP:
        append_EJPP_end(
            circ=qc,
            root=[top_r[0]],
            target_communication=[com_targ],
            classical_memory=[cl_mem],
        )
        qc.barrier()
    qc = iterative_IQFT_circ(
        qc=qc,
        qr=[top_r[0]],
        cr=cr,
        i=top_c - 1,
        measuring_and_reset=False,
        phase_minus=False,
    )
    append_measure(
        circ=qc,
        qubit=[top_r[0]],
        clbit=[cr[-2]] if EJPP else [cr[-1]],
        duration=DURATION_MEASURE,
    )
    return qc


def run_semi_iterative(
    U_list,
    top_c: int,
    bot_c: int,
    draw: bool = False,
    noise: bool = False,
    shots: int = 10000,
    EJPP: bool = False,
    init_one: bool = False,
    init_ind: int = -1,
    ordering: int = 0,
):
    if ordering == 0:
        qc = semi_iterative_n_1_circuit(
            U_list=U_list,
            top_c=top_c,
            bot_c=bot_c,
            EJPP=EJPP,
            init_one=init_one,
            init_ind=init_ind,
        )
    elif ordering == 1:
        qc = semi_iterative_1_n_circuit(
            U_list=U_list,
            top_c=top_c,
            bot_c=bot_c,
            EJPP=EJPP,
            init_one=init_one,
            init_ind=init_ind,
        )
    else:
        print("specifier for ordering is not valid")
    if draw:
        print(qc.decompose())
    results = run_circuit(qc, shots=shots, noise_model=noise)
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[1:]] = cut_results.get(key[1:], 0) + results[key]
    else:
        cut_results = results
    return qc, cut_results


# --------------------------------------------
# this section investigates the iterative setting with an additional available qubit on the control register which can be used to alternate the control sequence to improve the scheduling of gate execution


def alternating_iterative_QPE_circ(
    U_list: list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    swap: bool = False,
    EJPP: int = 0,
    draw: bool = False,
) -> QuantumCircuit:
    qr = QuantumRegister(bot_c + 2 + 2 * EJPP)
    cr = ClassicalRegister(top_c + EJPP, name="meas")

    top_r = qr[0:2]
    bot_r = qr[2 + 2 * EJPP :]
    if EJPP:
        cl_mem = cr[-EJPP:]
        com_root = qr[2 : EJPP + 2]
        com_targ = qr[2 + EJPP : 2 + 2 * EJPP]

    qc = QuantumCircuit(qr[:], cr)

    if swap:
        cr = cr[::-1]
    if init_one:
        qc.x(bot_r[init_ind])
    for i in range(top_c):
        qc.h(top_r[i % 2])
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r[i % 2]],
                root_communication=[com_root[i % EJPP]],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()
        U_list[i](
            circ=qc,
            control=com_targ[i % EJPP] if EJPP != 0 else top_r[i % 2],
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r[i % 2]],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()

        if i < top_c - 2:
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(qc, [top_r[i % 2]], cr[: (i + 1)], i, reset=reset)
        qc.barrier()
    if draw:
        print(qc.decompose())
    return qc


def run_alternating_IQFT(
    U_list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    noise: bool = False,
    shots: int = 10000,
    EJPP: int = 0,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    qc = alternating_iterative_QPE_circ(
        U_list=U_list,
        top_c=top_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
        draw=draw,
    )

    qc = get_moment_circ(get_transpiled_circuit(qc), noise=noise)

    results = run_circuit(
        qc,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    return qc, cut_results


def start_IQFT_ibmq(
    U_list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    shots: int = 10000,
    EJPP: int = 0,
):
    qc = iterative_QPE_circ(
        U_list=U_list,
        top_c=top_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
    )
    # qc = get_moment_circ(get_transpiled_circuit(qc), noise=False)

    start_experiment(qc, shots)
    return qc


def get_results_IQFT_ibmq(path: str, EJPP):
    results = read_results(path)
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results

    return cut_results


def start_alternating_IQFT_ibmq(
    U_list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    shots: int = 10000,
    EJPP: int = 0,
):
    qc = alternating_iterative_QPE_circ(
        U_list=U_list,
        top_c=top_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
        draw=draw,
    )
    print(qc)

    # qc = get_moment_circ(get_transpiled_circuit(qc), noise=False)

    start_experiment(qc, shots)
    return qc


def get_results_alternating_IQFT_ibmq(path: str, EJPP):
    results = read_results(path)
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    return cut_results


def start_monolithic_ibmq(
    U_list: list,
    top_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    shots: int = 10000,
    EJPP: int = 0,
    quantum_qft: bool = False,
):
    control_reg = QuantumRegister(top_c)
    root_communication = QuantumRegister(EJPP)
    target_communication = QuantumRegister(EJPP)
    target_register = QuantumRegister(bot_c)
    classical_memory = ClassicalRegister(EJPP)
    measure_register = ClassicalRegister(top_c, name="meas")
    qr = (
        control_reg[:]
        + root_communication[:]
        + target_communication[:]
        + target_register[:]
        if EJPP != 0
        else control_reg[:] + target_register[:]
    )
    cr = classical_memory[:] + measure_register[:] if EJPP != 0 else measure_register[:]
    cr.name = "meas"
    circ = QuantumCircuit(qr, cr)
    circ.append(
        regular_circ(
            U_list=U_list,
            top_c=top_c,
            bot_c=bot_c,
            init_one=init_one,
            init_ind=init_ind,
            draw=draw,
            EJPP=EJPP,
            quantum_qft=quantum_qft,
        ),
        qr,
        cr,
    )
    # circ = get_moment_circ(get_transpiled_circuit(circ), noise=False)
    start_experiment(circ, shots)
    return circ


def get_results_monolithic_ibmq(path: str, EJPP):
    results = read_results(path)
    cut_results = {}
    if EJPP != 0:
        for key in results.keys():
            # removing additional 1 here from key as whitespace was included
            cut_results[key[0 : -EJPP - 1]] = (
                cut_results.get(key[0 : -EJPP - 1], 0) + results[key]
            )
    else:
        cut_results = results
    return cut_results


# ----------------------------------------------------
# dlog circuits


def iterative_dlog_circ(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    swap: bool = False,
    EJPP: int = 0,
    draw: bool = False,
) -> QuantumCircuit:
    qr = QuantumRegister(bot_c + 1 + 2 * EJPP)
    cr = ClassicalRegister(top_e1_c + top_e2_c + EJPP)

    top_r = qr[0]
    bot_r = qr[1 + 2 * EJPP :]

    cr_e1 = cr[:top_e1_c]  # classical register for measurements of e1 reg
    cr_e2 = cr[
        top_e1_c : top_e1_c + top_e2_c
    ]  # classical register for measurements of e2 reg

    if EJPP != 0:
        cl_mem = cr[-EJPP:]  # last classical bits are reserved for dqc protocols
        com_root = qr[1 : EJPP + 1]  # root and target qubits reserved for ebit channels
        com_targ = qr[EJPP + 1 : 2 * EJPP + 1]

    qc = QuantumCircuit(qr[:], cr)

    if swap:
        cr_e1 = cr_e1[::-1]
        cr_e2 = cr_e2[::-1]

    if init_one:
        qc.x(bot_r[init_ind])

    # first handle e1 register
    for i in range(top_e1_c):
        qc.h(top_r)
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r],
                root_communication=[com_root[i % EJPP]],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()
        U_e1_list[i](
            circ=qc, control=com_targ[i % EJPP] if EJPP != 0 else top_r, target=bot_r[:]
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()

        if (i < top_e1_c - 1) or (top_e2_c > 0):
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(qc, [top_r], cr_e1[: (i + 1)], i, reset=reset)
        qc.barrier()

    # start on correct next ejpp channel
    if EJPP > 0:
        e2_ejpp_shift = top_e1_c % EJPP
    else:
        e2_ejpp_shift = 0
    # then e2 register
    for i in range(top_e2_c):
        i_ejpp = i + e2_ejpp_shift
        qc.h(top_r)
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r],
                root_communication=[com_root[i_ejpp % EJPP]],
                target_communication=[com_targ[i_ejpp % EJPP]],
                classical_memory=[cl_mem[i_ejpp % EJPP]],
            )
            qc.barrier()
        U_e2_list[i](
            circ=qc,
            control=com_targ[i_ejpp % EJPP] if EJPP != 0 else top_r,
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r],
                target_communication=[com_targ[i_ejpp % EJPP]],
                classical_memory=[cl_mem[i_ejpp % EJPP]],
            )
            qc.barrier()

        if i < top_e2_c - 1:
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(qc, [top_r], cr_e2[: (i + 1)], i, reset=reset)
        qc.barrier()

    if draw:
        print(qc.decompose())
    return qc


def alternating_seq_dlog_circ(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    swap: bool = False,
    EJPP: int = 0,
    draw: bool = False,
) -> QuantumCircuit:
    qr = QuantumRegister(bot_c + 2 + 2 * EJPP)
    cr = ClassicalRegister(top_e1_c + top_e2_c + EJPP, name="meas")

    top_r = qr[0:2]
    bot_r = qr[2 + 2 * EJPP :]
    if EJPP:
        cl_mem = cr[-EJPP:]
        com_root = qr[2 : EJPP + 2]
        com_targ = qr[2 + EJPP : 2 + 2 * EJPP]

    qc = QuantumCircuit(qr[:], cr)

    cr_e1 = cr[:top_e1_c]  # classical register for measurements of e1 reg
    cr_e2 = cr[
        top_e1_c : top_e1_c + top_e2_c
    ]  # classical register for measurements of e2 reg

    if swap:
        cr_e1 = cr_e1[::-1]
        cr_e2 = cr_e2[::-1]

    if init_one:
        qc.x(bot_r[init_ind])

    # first handle all of e1
    for i in range(top_e1_c):
        qc.h(top_r[i % 2])
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r[i % 2]],
                root_communication=[com_root[i % EJPP]],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()
        U_e1_list[i](
            circ=qc,
            control=com_targ[i % EJPP] if EJPP != 0 else top_r[i % 2],
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r[i % 2]],
                target_communication=[com_targ[i % EJPP]],
                classical_memory=[cl_mem[i % EJPP]],
            )
            qc.barrier()

        if i - (top_e1_c - 1) > top_e2_c and (
            top_e2_c < 2
        ):  # if there are more remaining e1 U operations than there are e2 U operations, dont reset
            reset = False
        else:
            reset = True

        qc = iterative_IQFT_circ(qc, [top_r[i % 2]], cr_e1[: (i + 1)], i, reset=reset)
        qc.barrier()

    # depending on if e1 is even or odd,
    # e2 has to start on the top or bottom of the two top register qubits
    # also the appropriate first ebit channel has to be chosen
    e2_top_shift = top_e1_c % 2
    if EJPP > 0:
        e2_ejpp_shift = top_e1_c % EJPP
    else:
        e2_ejpp_shift = 0
    # then handle all of e2
    for i in range(top_e2_c):
        i_top = i + e2_top_shift
        i_ejpp = i + e2_ejpp_shift

        qc.h(top_r[i_top % 2])
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r[i_top % 2]],
                root_communication=[com_root[i_ejpp % EJPP]],
                target_communication=[com_targ[i_ejpp % EJPP]],
                classical_memory=[cl_mem[i_ejpp % EJPP]],
            )
            qc.barrier()
        U_e2_list[i](
            circ=qc,
            control=(com_targ[i_ejpp % EJPP] if EJPP != 0 else top_r[i_top % 2]),
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r[i_top % 2]],
                target_communication=[com_targ[i_ejpp % EJPP]],
                classical_memory=[cl_mem[i_ejpp % EJPP]],
            )
            qc.barrier()

        if i < top_e2_c - 2:
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(
            qc, [top_r[i_top % 2]], cr_e2[: (i + 1)], i, reset=reset
        )
        qc.barrier()

    if draw:
        print(qc.decompose())
    return qc


def double_iterative_dlog_circ(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    swap: bool = False,
    EJPP: int = 0,
    draw: bool = False,
) -> QuantumCircuit:

    qr = QuantumRegister(bot_c + 2 + 2 * EJPP)
    cr = ClassicalRegister(top_e1_c + top_e2_c + EJPP, name="meas")

    assert top_e1_c >= top_e2_c

    e1_remaining = top_e1_c - top_e2_c

    top_r = qr[0:2]
    bot_r = qr[2 + 2 * EJPP :]
    if EJPP:
        cl_mem = cr[-EJPP:]
        com_root = qr[2 : EJPP + 2]
        com_targ = qr[2 + EJPP : 2 + 2 * EJPP]

    qc = QuantumCircuit(qr[:], cr)

    cr_e1 = cr[:top_e1_c]  # classical register for measurements of e1 reg
    cr_e2 = cr[
        top_e1_c : top_e1_c + top_e2_c
    ]  # classical register for measurements of e2 reg

    if swap:
        cr_e1 = cr_e1[::-1]
        cr_e2 = cr_e2[::-1]

    if init_one:
        qc.x(bot_r[init_ind])

    i_ejpp = 0
    for i in range(top_e2_c):
        # e1 on first top qubit
        qc.h(top_r[0])
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r[0]],
                root_communication=[com_root[i_ejpp]],
                target_communication=[com_targ[i_ejpp]],
                classical_memory=[cl_mem[i_ejpp]],
            )
            qc.barrier()
        U_e1_list[i](
            circ=qc,
            control=com_targ[i_ejpp] if EJPP != 0 else top_r[0],
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r[0]],
                target_communication=[com_targ[i_ejpp]],
                classical_memory=[cl_mem[i_ejpp]],
            )
            i_ejpp = (i_ejpp + 1) % EJPP
            qc.barrier()

        if (i < top_e1_c - 1) or (e1_remaining > 0):
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(qc, [top_r[0]], cr_e1[: (i + 1)], i, reset=reset)
        qc.barrier()

        # e2 on lower top_r qubit
        qc.h(top_r[1])
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r[1]],
                root_communication=[com_root[i_ejpp]],
                target_communication=[com_targ[i_ejpp]],
                classical_memory=[cl_mem[i_ejpp]],
            )
            qc.barrier()
        U_e2_list[i](
            circ=qc,
            control=com_targ[i_ejpp] if EJPP != 0 else top_r[1],
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r[1]],
                target_communication=[com_targ[i_ejpp]],
                classical_memory=[cl_mem[i_ejpp]],
            )
            i_ejpp = (i_ejpp + 1) % EJPP
            qc.barrier()

        if (i < top_e2_c - 1) or (e1_remaining > 2):
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(qc, [top_r[1]], cr_e2[: (i + 1)], i, reset=reset)
        qc.barrier()

    # the appropriate first ebit channel has to be chosen

    if EJPP > 0:
        ejpp_shift = i_ejpp
    else:
        ejpp_shift = 0
    # handle remaining parts of e1 in alternating fashion
    for i in range(top_e2_c, top_e1_c):
        i_top = i  # no shift needed as we will always start with the top qubit
        i_ejpp = i + ejpp_shift

        qc.h(top_r[i_top % 2])
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r[i_top % 2]],
                root_communication=[com_root[i_ejpp % EJPP]],
                target_communication=[com_targ[i_ejpp % EJPP]],
                classical_memory=[cl_mem[i_ejpp % EJPP]],
            )
            qc.barrier()
        U_e1_list[i](
            circ=qc,
            control=(com_targ[i_ejpp % EJPP] if EJPP != 0 else top_r[i_top % 2]),
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r[i_top % 2]],
                target_communication=[com_targ[i_ejpp % EJPP]],
                classical_memory=[cl_mem[i_ejpp % EJPP]],
            )
            qc.barrier()

        if i < top_e1_c - 2:
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(
            qc, [top_r[i_top % 2]], cr_e1[: (i + 1)], i, reset=reset
        )
        qc.barrier()

    if draw:
        print(qc.decompose())
    return qc


def three_cycle_dlog_circ(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    swap: bool = False,
    EJPP: int = 0,
    draw: bool = False,
) -> QuantumCircuit:

    qr = QuantumRegister(bot_c + 3 + 2 * EJPP)
    cr = ClassicalRegister(top_e1_c + top_e2_c + EJPP, name="meas")

    assert top_e1_c >= top_e2_c and top_e1_c <= 2 * top_e2_c

    top_r = qr[0:3]
    bot_r = qr[3 + 2 * EJPP :]
    if EJPP:
        cl_mem = cr[-EJPP:]
        com_root = qr[3 : EJPP + 3]
        com_targ = qr[3 + EJPP : 3 + 2 * EJPP]

    qc = QuantumCircuit(qr[:], cr)

    cr_e1 = cr[:top_e1_c]  # classical register for measurements of e1 reg
    cr_e2 = cr[
        top_e1_c : top_e1_c + top_e2_c
    ]  # classical register for measurements of e2 reg

    if swap:
        cr_e1 = cr_e1[::-1]
        cr_e2 = cr_e2[::-1]

    if init_one:
        qc.x(bot_r[init_ind])

    i_ejpp = 0
    for i in range(top_e2_c):

        if 2 * i < top_e1_c:
            # first e1 on first top qubit
            qc.h(top_r[0])
            qc.barrier()
            if EJPP != 0:
                append_EJPP_start(
                    circ=qc,
                    root=[top_r[0]],
                    root_communication=[com_root[i_ejpp]],
                    target_communication=[com_targ[i_ejpp]],
                    classical_memory=[cl_mem[i_ejpp]],
                )
                qc.barrier()
            U_e1_list[2 * i](
                circ=qc,
                control=com_targ[i_ejpp] if EJPP != 0 else top_r[0],
                target=bot_r[:],
            )
            qc.barrier()
            if EJPP != 0:
                append_EJPP_end(
                    circ=qc,
                    root=[top_r[0]],
                    target_communication=[com_targ[i_ejpp]],
                    classical_memory=[cl_mem[i_ejpp]],
                )
                i_ejpp = (i_ejpp + 1) % EJPP
                qc.barrier()

            if 2 * (i + 1) > top_e1_c - 1:
                reset = False

            else:
                reset = True

            qc = iterative_IQFT_circ(
                qc, [top_r[0]], cr_e1[: (2 * i + 1)], 2 * i, reset=reset
            )
            qc.barrier()
        if 2 * i + 1 < top_e1_c:
            # second e1 on second top qubit
            qc.h(top_r[1])
            qc.barrier()
            if EJPP != 0:
                append_EJPP_start(
                    circ=qc,
                    root=[top_r[1]],
                    root_communication=[com_root[i_ejpp]],
                    target_communication=[com_targ[i_ejpp]],
                    classical_memory=[cl_mem[i_ejpp]],
                )
                qc.barrier()
            U_e1_list[2 * i + 1](
                circ=qc,
                control=com_targ[i_ejpp] if EJPP != 0 else top_r[1],
                target=bot_r[:],
            )
            qc.barrier()
            if EJPP != 0:
                append_EJPP_end(
                    circ=qc,
                    root=[top_r[1]],
                    target_communication=[com_targ[i_ejpp]],
                    classical_memory=[cl_mem[i_ejpp]],
                )
                i_ejpp = (i_ejpp + 1) % EJPP
                qc.barrier()

            if 2 * (i + 1) + 1 > top_e1_c - 1:
                reset = False

            else:
                reset = True

            qc = iterative_IQFT_circ(
                qc, [top_r[1]], cr_e1[: ((2 * i + 1) + 1)], 2 * i + 1, reset=reset
            )
            qc.barrier()

        # e2 on third top_r qubit
        qc.h(top_r[2])
        qc.barrier()
        if EJPP != 0:
            append_EJPP_start(
                circ=qc,
                root=[top_r[2]],
                root_communication=[com_root[i_ejpp]],
                target_communication=[com_targ[i_ejpp]],
                classical_memory=[cl_mem[i_ejpp]],
            )
            qc.barrier()
        U_e2_list[i](
            circ=qc,
            control=com_targ[i_ejpp] if EJPP != 0 else top_r[2],
            target=bot_r[:],
        )
        qc.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=qc,
                root=[top_r[2]],
                target_communication=[com_targ[i_ejpp]],
                classical_memory=[cl_mem[i_ejpp]],
            )
            i_ejpp = (i_ejpp + 1) % EJPP
            qc.barrier()

        if i < top_e2_c - 1:
            reset = True

        else:
            reset = False

        qc = iterative_IQFT_circ(qc, [top_r[2]], cr_e2[: (i + 1)], i, reset=reset)
        qc.barrier()

    return qc


def regular_dlog_circ(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    EJPP: int = 0,
    swap: bool = False,
    quantum_qft: bool = False,
) -> QuantumCircuit:
    """
    builds standard monolithic Shor for given list of U operations
    """
    control_e1_reg = QuantumRegister(top_e1_c)
    control_e2_reg = QuantumRegister(top_e2_c)
    root_communication = QuantumRegister(EJPP)
    target_communication = QuantumRegister(EJPP)
    target_register = QuantumRegister(bot_c)
    classical_memory = ClassicalRegister(EJPP)
    measure_e1_register = ClassicalRegister(top_e1_c)
    measure_e2_register = ClassicalRegister(top_e2_c)

    # usually e1_reg is size 2m, e2_reg size m, we allow flexibility
    # but at least e1 should be as larger or larger than e2
    assert top_e1_c >= top_e2_c

    circ = (
        QuantumCircuit(
            control_e1_reg[:]
            + control_e2_reg[:]
            + root_communication[:]
            + target_communication[:]
            + target_register[:],
            classical_memory[:] + measure_e1_register[:] + measure_e2_register[:],
        )
        if EJPP != 0
        else QuantumCircuit(
            control_e1_reg[:] + control_e2_reg[:] + target_register[:],
            measure_e1_register[:] + measure_e2_register[:],
        )
    )
    if init_one:
        circ.x(target_register[init_ind])

    # init both top regs
    circ.h(control_e1_reg)
    circ.h(control_e2_reg)

    circ.barrier()

    for i in range(top_e1_c):
        if EJPP != 0:
            append_EJPP_start(
                circ=circ,
                root=[control_e1_reg[i]],
                root_communication=[root_communication[i % EJPP]],
                target_communication=[target_communication[i % EJPP]],
                classical_memory=[classical_memory[i % EJPP]],
            )
            circ.barrier()
        U_e1_list[i](
            circ=circ,
            control=target_communication[i % EJPP] if EJPP != 0 else control_e1_reg[i],
            target=target_register[:],
        )
        circ.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=circ,
                root=[control_e1_reg[i]],
                target_communication=[target_communication[i % EJPP]],
                classical_memory=[classical_memory[i % EJPP]],
            )
            circ.barrier()
        if quantum_qft == False:
            for k in range(i):
                circ.p(m.pi / 2 ** (k + 1), control_e1_reg[i]).c_if(
                    measure_e1_register[i - k - 1], 1
                )
            circ.h(control_e1_reg[i])
            append_measure(
                circ=circ,
                qubit=[control_e1_reg[i]],
                clbit=[measure_e1_register[i]],
                duration=DURATION_MEASURE,
            )
            circ.barrier()

    for i in range(top_e2_c):
        if EJPP != 0:
            append_EJPP_start(
                circ=circ,
                root=[control_e2_reg[i]],
                root_communication=[root_communication[i % EJPP]],
                target_communication=[target_communication[i % EJPP]],
                classical_memory=[classical_memory[i % EJPP]],
            )
            circ.barrier()
        U_e2_list[i](
            circ=circ,
            control=target_communication[i % EJPP] if EJPP != 0 else control_e2_reg[i],
            target=target_register[:],
        )
        circ.barrier()
        if EJPP != 0:
            append_EJPP_end(
                circ=circ,
                root=[control_e2_reg[i]],
                target_communication=[target_communication[i % EJPP]],
                classical_memory=[classical_memory[i % EJPP]],
            )
            circ.barrier()
        if quantum_qft == False:
            for k in range(i):
                circ.p(m.pi / 2 ** (k + 1), control_e2_reg[i]).c_if(
                    measure_e2_register[i - k - 1], 1
                )
            circ.h(control_e2_reg[i])
            append_measure(
                circ=circ,
                qubit=[control_e2_reg[i]],
                clbit=[measure_e2_register[i]],
                duration=DURATION_MEASURE,
            )
            circ.barrier()

    # apply two separate qfts on the two top regs
    if quantum_qft:
        circ.append(qft_gate(n=top_e1_c, draw=False, swap=swap), control_e1_reg[::-1])
        circ.append(qft_gate(n=top_e2_c, draw=False, swap=swap), control_e2_reg[::-1])
        circ.barrier()
        append_measure(
            circ=circ,
            qubit=control_e1_reg[:],
            clbit=measure_e1_register[:],
            duration=DURATION_MEASURE,
        )
        append_measure(
            circ=circ,
            qubit=control_e2_reg[:],
            clbit=measure_e2_register[:],
            duration=DURATION_MEASURE,
        )
    if draw:
        print(circ.decompose())
    return circ


def run_regular_dlog(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    EJPP: int = 0,
    swap: bool = False,
    noise: bool = False,
    shots: int = 10000,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    qc = regular_dlog_circ(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
        draw=draw,
    )

    qc = get_moment_circ(get_transpiled_circuit(qc), noise=noise)

    print("Running circuit regular..")
    results = run_circuit(
        qc,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    return qc, cut_results


def run_iterative_dlog(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    EJPP: int = 0,
    swap: bool = False,
    noise: bool = False,
    shots: int = 10000,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    qc = iterative_dlog_circ(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
        draw=draw,
    )

    qc = get_moment_circ(get_transpiled_circuit(qc), noise=noise)

    print("Running circuit iterative..")
    results = run_circuit(
        qc,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    return qc, cut_results


def run_alternating_dlog(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    EJPP: int = 0,
    swap: bool = False,
    noise: bool = False,
    shots: int = 10000,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    qc = alternating_seq_dlog_circ(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
        draw=draw,
    )

    qc = get_moment_circ(get_transpiled_circuit(qc), noise=noise)

    print("Running circuit alternating..")
    results = run_circuit(
        qc,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    return qc, cut_results


def run_double_iterative_dlog(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    EJPP: int = 0,
    swap: bool = False,
    noise: bool = False,
    shots: int = 10000,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    qc = double_iterative_dlog_circ(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
        draw=draw,
    )

    qc = get_moment_circ(get_transpiled_circuit(qc), noise=noise)

    print("Running circuit double iterative..")
    results = run_circuit(
        qc,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    return qc, cut_results


def run_three_cyclic_dlog(
    U_e1_list: list,
    U_e2_list: list,
    top_e1_c: int,
    top_e2_c: int,
    bot_c: int,
    init_one: bool,
    init_ind: int = -1,
    draw: bool = False,
    EJPP: int = 0,
    swap: bool = False,
    noise: bool = False,
    shots: int = 10000,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    qc = three_cycle_dlog_circ(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        EJPP=EJPP,
        draw=draw,
    )

    qc = get_moment_circ(get_transpiled_circuit(qc), noise=noise)

    print("Running circuit three cyclic..")
    results = run_circuit(
        qc,
        shots=shots,
        noise=noise,
        noise_scale=noise_scale,
        relaxation_noise_scale=relaxation_noise_scale,
    )
    cut_results = {}
    if EJPP:
        for key in results.keys():
            cut_results[key[EJPP:]] = cut_results.get(key[EJPP:], 0) + results[key]
    else:
        cut_results = results
    return qc, cut_results


# --------------------------------------------
"""
Possible tasks:
    - include a four controls example to maybe include 2 qubit blocks in alternating iterative way and split in two two qubit QFT blocks //
    - include multiple ebit option in semi iterative set up //
"""


def get_random_uneven(k):
    p = 2
    while p % 2 == 0 or p <= 1:
        p = random.getrandbits(k)
    return p


def main():
    # Build noise model from backend info
    bs = BackendSnapshot.from_snapshot("backend_snapshots/test.json")
    noise_model = homogenous_noise_model(bs)

    # Lists of unitary gate calls to execute circuits with:
    # work register: 5
    # init one: True

    # p = get_random_uneven(5)
    # q = get_random_uneven(5)

    # N = 15
    # a = 2
    # n = np.ceil(np.log2(N)).astype(np.int64)
    # c = n

    N = 21
    a = 2
    c = 3

    # print(p, q, N, n)
    # U_list, top_c, bot_c, init_one, init_ind = get_QRISP_circ_data_function(N, a, c)()

    # U_list = [
    #     create_append_controlled_U(XGate()),
    #     create_append_controlled_U(XGate()),
    #     create_append_controlled_U(XGate()),
    # ]
    # top_c = 3
    # bot_c = 1
    # init_one = True
    # init_ind = 0

    U_list = [
        create_append_controlled_U(U4_n21_base4()),
        create_append_controlled_U(U2_n21_base4()),
        create_append_controlled_U(U1_n21_base4()),
    ]
    top_c = 3
    bot_c = 2
    init_ind = 0
    init_one = False

    # work register: 6
    # init one: True
    # U_list = [create_append_controlled_U(U4_n35()), create_append_controlled_U(U2_n35()), create_append_controlled_U(U1_n35())] # order is important in these optimized circuits we are investigating so do not reorder!

    # work register: 2
    # init one: False
    # U_list = [create_append_controlled_U(U1_n21_base4()), create_append_controlled_U(U2_n21_base4()), create_append_controlled_U(U4_n21_base4())] # order is important in these optimized circuits we are investigating so do not reorder!

    # work register: 4
    # init one: True
    # U_list = [create_append_controlled_U(U4_n15_base2()), create_append_controlled_U(U2_n15_base2()), create_append_controlled_U(U1_n15_base2())] # order is important in these optimized circuits we are investigating so do not reorder!

    # circuit executions in different settings:
    # qc, results = run_alternating_IQFT(
    #     U_list=U_list, top_c=3, bot_c=5, init_one=True, draw=True, noise=False, EJPP=3
    # )

    # run_IQFT(
    #     U_list=U_list, top_c=3, bot_c=5, init_one=True, draw=True, noise=False, EJPP=2
    # )

    # qc = iterative_QPE_circ(
    #     U_list=U_list,
    #     top_c=top_c,
    #     bot_c=bot_c,
    #     init_one=init_one,
    #     init_ind=init_ind,
    #     EJPP=2,
    # )
    # print(qc)
    # # print(qc.depth())
    # start = time.time()
    # # rb = RemoveBarriers()
    # # circ_transpiled = rb(qc).decompose()
    # # print(circ_transpiled)
    # circ_transpiled = get_transpiled_circuit(qc)
    # print(circ_transpiled)
    # end = time.time()
    # print(end - start)

    # start = time.time()
    # g = CircuitGraph(circ_transpiled)
    # g.plot_graph()
    # d_graph = g.get_weighted_depth()
    # print(d_graph)
    # end = time.time()
    # print(end - start)
    # # print(N, a, n, c, p, q)
    # # print(depths_from_circ(get_transpiled_circuit(qc)))
    # # print(g.get_depth())

    # start = time.time()
    # # print(get_moment_circ(circ_transpiled, squeezed=False, noise=False))
    # d_tetris = depths_from_circ(circ_transpiled, squeeze=False)
    # print(d_tetris)
    # end = time.time()
    # print(end - start)

    # circuit_reg, results_reg = run_alternating_IQFT(
    #     U_list=U_list,
    #     shots=10000,
    #     top_c=top_c,
    #     bot_c=bot_c,
    #     init_one=init_one,
    #     init_ind=init_ind,
    #     draw=True,
    #     noise=False,
    #     EJPP=0,
    #     relaxation_noise_scale=1,
    #     noise_scale=0.01,
    # )

    # # run_semi_iterative(U_list=U_list, top_c=3, bot_c=5, draw=True, noise=noise_model, EJPP=True, init_one=True, ordering=0)
    # print(circuit_reg)
    # print(circuit_reg.depth())
    # plot_histogram(results_reg)
    # plt.show()
    # plt.show()

    U_e1_list, U_e2_list, top_e1_c, top_e2_c, bot_c, init_ind, init_one = (
        get_dlog_QRISP_circ_data_function(15, 2, 2)()
    )

    # U_e1_list = [
    #     create_append_controlled_U(U4_n21_base4()),
    #     create_append_controlled_U(U2_n21_base4()),
    #     create_append_controlled_U(U1_n21_base4()),
    # ]

    # U_e2_list = [
    #     # create_append_controlled_U(U4_n21_base4()),
    #     create_append_controlled_U(U2_n21_base4()),
    #     create_append_controlled_U(U1_n21_base4()),
    # ]
    # top_e1_c = 3
    # top_e2_c = 2
    # bot_c = 2
    # init_ind = 0
    # init_one = False

    # circuit = three_cycle_dlog_circ(
    #     U_e1_list=U_e1_list,
    #     U_e2_list=U_e2_list,
    #     top_e1_c=top_e1_c,
    #     top_e2_c=top_e2_c,
    #     bot_c=bot_c,
    #     init_one=init_one,
    #     init_ind=init_ind,
    #     draw=True,
    #     EJPP=0,
    #     swap=False,
    # )

    circuit = regular_dlog_circ(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
        draw=True,
        EJPP=4,
        swap=False,
    )
    circuit.draw("mpl").savefig("test.png")

    # qc, cut_results = run_alternating_dlog(
    #     U_e1_list=U_e1_list,
    #     U_e2_list=U_e2_list,
    #     top_e1_c=top_e1_c,
    #     top_e2_c=top_e2_c,
    #     bot_c=bot_c,
    #     init_one=init_one,
    #     init_ind=init_ind,
    # )
    # print(cut_results)


def design_validation():
    # bitstrings are ordered top_e1, top_e2, ejpp
    # qiskit execution gives reverse order
    # and the ejpp bits (i.e. the first bits of the string) are cut
    # so the remaining strings are e_2, e_1 with little-endian notation
    # i.e. so if e_2 = abcd then a represents a higher power of 2 than d

    U_e1_list, U_e2_list, top_e1_c, top_e2_c, bot_c, init_ind, init_one = (
        get_dlog_QRISP_circ_data_function(15, 2, 2)()
    )
    qc, cut_results = run_iterative_dlog(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
    )
    print(cut_results)
    sys.stdout.flush()

    qc, cut_results = run_alternating_dlog(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
    )
    print(cut_results)
    sys.stdout.flush()

    qc, cut_results = run_double_iterative_dlog(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
    )
    print(cut_results)
    sys.stdout.flush()

    qc, cut_results = run_three_cyclic_dlog(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
    )
    print(cut_results)
    sys.stdout.flush()

    qc, cut_results = run_regular_dlog(
        U_e1_list=U_e1_list,
        U_e2_list=U_e2_list,
        top_e1_c=top_e1_c,
        top_e2_c=top_e2_c,
        bot_c=bot_c,
        init_one=init_one,
        init_ind=init_ind,
    )
    print(cut_results)
    sys.stdout.flush()
    # print("")
    # print("N25 iterative")
    # U_e1_list, U_e2_list, top_e1_c, top_e2_c, bot_c, init_ind, init_one = (
    #     get_dlog_QRISP_circ_data_function(25, 2, 3)()
    # )

    # qc, cut_results = run_iterative_dlog(
    #     U_e1_list=U_e1_list,
    #     U_e2_list=U_e2_list,
    #     top_e1_c=top_e1_c,
    #     top_e2_c=top_e2_c,
    #     bot_c=bot_c,
    #     init_one=init_one,
    #     init_ind=init_ind,
    # )
    # print(cut_results)
    # sys.stdout.flush()


if __name__ == "__main__":
    # design_validation()
    main()
    # results_reg = get_results_alternating_IQFT_ibmq(
    #     "./src/semi_iterative_comparison/results/ibm_hw/2024_10_21_11_31_7", 0
    # )
    # print(results_reg)

    # plot_histogram(results_reg)
    # plt.show()

    """
    #############################################################
    c = circ
    rb = RemoveBarriers()
    qc_nob = rb(c.decompose(reps=2))  # qc.decompose())
    print("----------------")
    print(qc_nob)
    moments = circuit_moments_cond(qc_nob)
    circuit_depths_simple(qc_nob, moments)
    circuit_depths_reset(qc_nob, moments)
    print(f"Qiskit depth: {qc_nob.depth()}")
    mom_sq = circuit_moments_squeezed_cond(qc_nob, moments)
    circuit_depths_reset_squeezed(qc_nob, moments)

    q_p = circuit_depths_reset_squeezed(qc_nob, mom_sq)
    depth_from_points(q_p)
    # QPU_depth(q_p, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    circ_sq = moments_to_circuit(qc_nob, mom_sq)
    circ_m = moments_to_circuit(qc_nob, moments)
    print("moments circ")
    print(circ_m)
    print("moments squeezed circ")
    print(circ_sq)
    circ_sq.draw(output="mpl")
    plt.show()

    ##################################################################
    """
