from qrisp import QuantumModulus
from qiskit_aer import AerSimulator
from qiskit import *
from qiskit.circuit import Gate
from qrisp import QuantumModulus
from qrisp import cuccaro_adder, gidney_adder, QuantumBool, control
from distributed_shor.basis_gates import get_basis_gates
import numpy as np
from qiskit.primitives import Sampler
from distributed_shor.utils import create_append_controlled_U, create_append_CU
from distributed_shor.circ_stat_utils import (
    get_transpiled_circuit,
    circuit_depths_top_sort,
    count_circuit_ops,
)

# TODO: save U ops to qasm and load them when needed so they dont have to be generated everytime?


# ----------------------


def get_U_circ(N: int, a: int, exp: int, inpl_adder=None) -> QuantumCircuit:
    qg = QuantumModulus(N, inpl_adder)
    qg[:] = 0
    a = pow(a, exp, N)
    qg *= a
    circ = qg.qs.compile().to_qiskit().decompose()
    return circ


def get_CU_circ(N: int, a: int, exp: int, inpl_adder=None) -> QuantumCircuit:
    qb = QuantumBool()
    qg = QuantumModulus(N, inpl_adder)
    qg[:] = 0
    with control(qb):
        a = pow(a, exp, N)
        qg *= a
    circ = qg.qs.compile().to_qiskit().decompose()
    return circ


def reverse_bits(qc: QuantumCircuit, n_data_qubits: int, ctrl=False):
    """
    Reverses order of

    Args:
        qc (QuantumCircuit): qrisp circuit
        n_data_qubits: amount of data qubits

    """
    reordered_qubits = []
    if ctrl:
        reordered_qubits = (
            [qc.qubits[0]]
            + list(reversed(qc.qubits[1 : n_data_qubits + 1]))
            + qc.qubits[n_data_qubits + 1 :]
        )

    else:
        reordered_qubits = (
            list(reversed(qc.qubits[1:n_data_qubits])) + qc.qubits[n_data_qubits:]
        )

    circ = QuantumCircuit(
        reordered_qubits,
        list(reversed(qc.clbits)),
        name=qc.name,
        global_phase=qc.global_phase,
    )
    if ctrl:
        new_qubit_map = (
            [circ.qubits[0]]
            + circ.qubits[1 : n_data_qubits + 1][::-1]
            + circ.qubits[n_data_qubits + 1 :]
        )
    else:
        new_qubit_map = circ.qubits[:n_data_qubits][::-1] + circ.qubits[n_data_qubits:]
    new_clbit_map = circ.clbits[::-1]
    for reg in reversed(qc.qregs):
        bits = [new_qubit_map[qc.find_bit(qubit).index] for qubit in reversed(reg)]
        circ.add_register(QuantumRegister(bits=bits, name=reg.name))
    for reg in reversed(qc.cregs):
        bits = [new_clbit_map[qc.find_bit(clbit).index] for clbit in reversed(reg)]
        circ.add_register(ClassicalRegister(bits=bits, name=reg.name))

    for instruction in qc.data:
        qubits = [
            new_qubit_map[qc.find_bit(qubit).index] for qubit in instruction.qubits
        ]
        clbits = [
            new_clbit_map[qc.find_bit(clbit).index] for clbit in instruction.clbits
        ]
        circ._append(instruction.replace(qubits=qubits, clbits=clbits))
    return circ


def fill_up_qubits_wrapper(gate: Gate, bot_c):
    gate_n_qubits = gate.num_qubits
    qr = QuantumRegister(bot_c)
    qc = QuantumCircuit(qr)
    qc.append(gate, qr[:gate_n_qubits])
    return qc.to_gate(label=gate.label)


def get_U_gate(N: int, a: int, exp: int, rev_order=True, inpl_adder=None):
    circ = get_U_circ(N, a, exp, inpl_adder)
    nq = circ.num_qubits
    n_data_qubits = np.ceil(np.log2(N)).astype(np.int64)
    if rev_order:
        circ = reverse_bits(circ, n_data_qubits)

    Ugate = circ.to_gate(label=f"U{exp}")
    return Ugate


def get_CU_gate(N: int, a: int, exp: int, rev_order=True, inpl_adder=None):
    circ = get_CU_circ(N, a, exp, inpl_adder)
    circ = get_transpiled_circuit(circ)
    nq = circ.num_qubits
    n_data_qubits = np.ceil(np.log2(N)).astype(np.int64)
    if rev_order:
        circ = reverse_bits(circ, n_data_qubits, ctrl=True)

    CUgate = circ.to_gate(label=f"CU{exp}")
    return CUgate


def get_U_list(N: int, a: int, count: int, rev_order=True, inpl_adder=None):

    U_list = []
    bot_c = 0
    for i in range(count - 1, -1, -1):

        Ugate = get_U_gate(N, a, 2**i, rev_order, inpl_adder)
        if Ugate.num_qubits > bot_c:
            bot_c = Ugate.num_qubits
        U_list.append(Ugate)

    U_list_adjusted = [
        create_append_controlled_U(fill_up_qubits_wrapper(gate, bot_c))
        for gate in U_list
    ]

    top_c = count

    return U_list_adjusted, count, bot_c


def get_CU_list(N: int, a: int, count: int, rev_order=True, inpl_adder=None):

    CU_list = []
    bot_c = 0
    for i in range(count - 1, -1, -1):

        CUgate = get_CU_gate(N, a, 2**i, rev_order, inpl_adder)
        if CUgate.num_qubits > bot_c:
            bot_c = CUgate.num_qubits
        CU_list.append(CUgate)

    CU_list_adjusted = [
        create_append_CU(fill_up_qubits_wrapper(gate, bot_c)) for gate in CU_list
    ]

    bot_c = bot_c - 1  # dont count control for bot_c size

    return CU_list_adjusted, count, bot_c


def get_CU_list_gates(N: int, a: int, count: int, rev_order=True, inpl_adder=None):

    CU_list = []
    bot_c = 0
    for i in range(count - 1, -1, -1):

        CUgate = get_CU_gate(N, a, 2**i, rev_order, inpl_adder)
        if CUgate.num_qubits > bot_c:
            bot_c = CUgate.num_qubits
        CU_list.append(CUgate)

    CU_list_adjusted = [fill_up_qubits_wrapper(gate, bot_c) for gate in CU_list]

    bot_c = bot_c - 1  # dont count control for bot_c size

    return CU_list_adjusted, bot_c


def get_QRISP_circ_data_function(N: int, a: int, count: int, CU=True):
    # for now not using rev_order, inpl_adder

    def get_QRISP_circ_data_CU():
        U_list, top_c, bot_c = get_CU_list(N, a, count)
        n = np.ceil(np.log2(N)).astype(np.int64)
        init_ind = n - 1
        init_one = True
        return U_list, top_c, bot_c, init_ind, init_one

    def get_QRISP_circ_data():
        U_list, top_c, bot_c = get_U_list(N, a, count)
        n = np.ceil(np.log2(N)).astype(np.int64)
        init_ind = n - 1
        init_one = True
        return U_list, top_c, bot_c, init_ind, init_one

    if CU:
        return get_QRISP_circ_data_CU
    else:
        return get_QRISP_circ_data


# --------------------
# dlog functions


# y = a^(N+1), y^-e2, -> y <- a^(-c*(N+1))


def get_e2_CU_circ(N: int, a: int, exp: int, inpl_adder=None) -> QuantumCircuit:
    qb = QuantumBool()
    qg = QuantumModulus(N, inpl_adder)
    qg[:] = 0
    with control(qb):
        a = pow(a, -exp * (N + 1), N)
        qg *= a
    circ = qg.qs.compile().to_qiskit().decompose()
    return circ


def get_e2_CU_gate(N: int, a: int, exp: int, rev_order=True, inpl_adder=None):
    circ = get_e2_CU_circ(N, a, exp, inpl_adder)
    circ = get_transpiled_circuit(circ)
    nq = circ.num_qubits
    n_data_qubits = np.ceil(np.log2(N)).astype(np.int64)
    if rev_order:
        circ = reverse_bits(circ, n_data_qubits, ctrl=True)

    CUgate = circ.to_gate(label=f"CU{-exp}(N+1)")
    return CUgate


def get_dlog_CU_lists(N: int, a: int, count: int, rev_order=True, inpl_adder=None):

    # count = m
    bot_c = 0
    CU_e2_list = []

    for i in range(count - 1, -1, -1):

        CUgate = get_e2_CU_gate(N, a, 2**i, rev_order, inpl_adder)
        if CUgate.num_qubits > bot_c:
            bot_c = CUgate.num_qubits
        CU_e2_list.append(CUgate)

    CU_e1_list = []

    for i in range(2 * count - 1, -1, -1):

        CUgate = get_CU_gate(N, a, 2**i, rev_order, inpl_adder)
        if CUgate.num_qubits > bot_c:
            bot_c = CUgate.num_qubits
        CU_e1_list.append(CUgate)

    CU_e1_list_adjusted = [
        create_append_CU(fill_up_qubits_wrapper(gate, bot_c)) for gate in CU_e1_list
    ]

    CU_e2_list_adjusted = [
        create_append_CU(fill_up_qubits_wrapper(gate, bot_c)) for gate in CU_e2_list
    ]

    bot_c = bot_c - 1  # dont count control for bot_c size

    return CU_e1_list_adjusted, CU_e2_list_adjusted, 2 * count, count, bot_c


def get_dlog_CU_lists_gates(
    N: int, a: int, count: int, rev_order=True, inpl_adder=None
):

    # count = m
    bot_c = 0
    CU_e2_list = []

    for i in range(count - 1, -1, -1):

        CUgate = get_e2_CU_gate(N, a, 2**i, rev_order, inpl_adder)
        if CUgate.num_qubits > bot_c:
            bot_c = CUgate.num_qubits
        CU_e2_list.append(CUgate)

    CU_e1_list = []

    for i in range(2 * count - 1, -1, -1):

        CUgate = get_CU_gate(N, a, 2**i, rev_order, inpl_adder)
        if CUgate.num_qubits > bot_c:
            bot_c = CUgate.num_qubits
        CU_e1_list.append(CUgate)

    CU_e1_list_adjusted = [fill_up_qubits_wrapper(gate, bot_c) for gate in CU_e1_list]

    CU_e2_list_adjusted = [fill_up_qubits_wrapper(gate, bot_c) for gate in CU_e2_list]

    bot_c = bot_c - 1  # dont count control for bot_c size

    return CU_e1_list_adjusted, CU_e2_list_adjusted, 2 * count, count, bot_c


def get_dlog_QRISP_circ_data_function(N: int, a: int, count: int, CU=True):
    # for now not using rev_order, inpl_adder
    # count = m

    def get_QRISP_circ_data_CU():
        U_e1_list, U_e2_list, top_e1_c, top_e2_c, bot_c = get_dlog_CU_lists(N, a, count)

        n = np.ceil(np.log2(N)).astype(np.int64)
        init_ind = n - 1
        init_one = True
        return U_e1_list, U_e2_list, top_e1_c, top_e2_c, bot_c, init_ind, init_one

    def get_QRISP_circ_data():
        raise NotImplementedError
        # U_list, top_c, bot_c = get_U_list(N, a, count)
        # n = np.ceil(np.log2(N)).astype(np.int64)
        # init_ind = n - 1
        # init_one = True
        # return U_list, top_c, bot_c, init_ind, init_one

    if CU:
        return get_QRISP_circ_data_CU
    else:
        return get_QRISP_circ_data


def get_avg_CU_durations_dlog(N, a, count, weights, both=False):
    CU_e1_list_adjusted, CU_e2_list_adjusted, top_e1_c, top_e2_c, bot_c = (
        get_dlog_CU_lists_gates(N, a, count)
    )
    avg_duration_e1 = 0

    for i, cu in enumerate(CU_e1_list_adjusted):
        qr = QuantumRegister(bot_c + 1)
        qc = QuantumCircuit(qr)
        qc.append(cu, qr)

        qc_transpiled = get_transpiled_circuit(qc)

        duration_cu = circuit_depths_top_sort(qc_transpiled, weights)

        avg_duration_e1 = avg_duration_e1 * (i) / (i + 1) + duration_cu / (i + 1)

        _, instructions_per_length = count_circuit_ops(qc_transpiled)

    avg_duration_e2 = 0

    for i, cu in enumerate(CU_e2_list_adjusted):
        qr = QuantumRegister(bot_c + 1)
        qc = QuantumCircuit(qr)
        qc.append(cu, qr)

        qc_transpiled = get_transpiled_circuit(qc)

        duration_cu = circuit_depths_top_sort(qc_transpiled, weights)

        avg_duration_e2 = avg_duration_e2 * (i) / (i + 1) + duration_cu / (i + 1)

        _, instructions_per_length = count_circuit_ops(qc_transpiled)
    if both:
        return avg_duration_e1, avg_duration_e2, bot_c, instructions_per_length
    else:
        return (avg_duration_e1 + avg_duration_e2) / 2.0, bot_c, instructions_per_length


# ---------------------------------------------------------------------------


def example_execution_test():
    N = 21
    a = 2
    exp = 4

    CUgate = get_CU_gate(N, a, exp)
    nq = CUgate.num_qubits
    n_data_qubits = np.ceil(np.log2(N)).astype(np.int64)
    qr = QuantumRegister(nq)
    cr = ClassicalRegister(n_data_qubits)
    circ = QuantumCircuit(qr, cr)
    circ.x(n_data_qubits - 1 + 1)
    circ.h(qr[0])
    circ.append(CUgate, qr)
    circ.measure(qr[1 : n_data_qubits + 1], cr)

    sampler = Sampler()
    job = sampler.run(circ, shots=1000)
    res = job.result()
    print(circ)
    print(res)
    print(pow(a, exp, N))


def get_avg_CU_duration(N, a, count, weights):
    CU_list, bot_c = get_CU_list_gates(N, a, count)
    avg_duration = 0
    for i, cu in enumerate(CU_list):
        qr = QuantumRegister(bot_c + 1)
        qc = QuantumCircuit(qr)
        qc.append(cu, qr)

        qc_transpiled = get_transpiled_circuit(qc)

        duration_cu = circuit_depths_top_sort(qc_transpiled, weights)

        avg_duration = avg_duration * (i) / (i + 1) + duration_cu / (i + 1)

        _, instructions_per_length = count_circuit_ops(qc_transpiled)

    return avg_duration, bot_c, instructions_per_length


if __name__ == "__main__":
    example_execution_test()
