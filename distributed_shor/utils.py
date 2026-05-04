from qiskit import *
from qiskit.circuit import Gate, Instruction, CircuitInstruction, Clbit
from qiskit_aer import AerSimulator
import qiskit_aer.noise as noise
import math as m
import numpy as np
from qiskit.quantum_info import Operator
from distributed_shor.circ_stat_utils import (
    depths_test,
    count_circuit_ops,
    depths_from_circ,
    max_depth,
    get_transpiled_circuit,
)
from distributed_shor.noise_utils import backend_noise_model, custom_noise_model
from distributed_shor.basis_gates import get_basis_gates

# Global Vars
DURATION_MEASURE = 10
DURATION_RESET = 10
DURATION_EBIT = 10


def U1_n21_base4() -> Gate:
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.x(qr[1])
    return qc.to_gate(label="U1")


def U2_n21_base4() -> Gate:
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.x(qr[1])
    qc.cx(qr[1], qr[0])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[0])
    return qc.to_gate(label="U2")


def U4_n21_base4() -> Gate:
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    qc.x(qr[1])
    qc.cx(qr[1], qr[0])
    qc.x(qr[1])
    qc.cx(qr[1], qr[0])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[0])
    return qc.to_gate(label="U4")


def U4_n21_base2() -> Gate:
    qr = QuantumRegister(5)
    qc = QuantumCircuit(qr)
    qc.x(qr[0])
    qc.x(qr[4])
    return qc.to_gate(label="U4")


def U4_n15_base2() -> Gate:
    qr = QuantumRegister(4)
    qc = QuantumCircuit(qr)
    return qc.to_gate(label="U4")


def U2_n15_base2() -> Gate:
    qr = QuantumRegister(4)
    qc = QuantumCircuit(qr)
    qc.x(qr[1])
    qc.x(qr[3])
    return qc.to_gate(label="U2")


def U1_n15_base2() -> Gate:
    qr = QuantumRegister(4)
    qc = QuantumCircuit(qr)
    qc.swap(qr[0], qr[1])
    qc.swap(qr[2], qr[3])
    return qc.to_gate(label="U1")


def U2_n21_base2() -> Gate:
    qr = QuantumRegister(5)
    qc = QuantumCircuit(qr)
    qc.swap(qr[2], qr[4])
    qc.swap(qr[0], qr[4])
    return qc.to_gate(label="U2")


def U1_n21_base2() -> Gate:
    qr = QuantumRegister(5)
    qc = QuantumCircuit(qr)
    qc.swap(qr[3], qr[4])
    qc.swap(qr[1], qr[2])
    qc.swap(qr[0], qr[4])
    qc.cx(qr[4], qr[3])
    qc.cx(qr[4], qr[1])
    return qc.to_gate(label="U1")


def U4_n35() -> Gate:
    """
    1->11
    """
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    qc.cx(qr[5], qr[4])
    qc.cx(qr[5], qr[2])

    return qc.to_gate(label="U4")


def U2_n35() -> Gate:
    """
    1->16
    11->1
    """
    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    qc.cx(qr[2], qr[1])
    qc.x(qr[1])
    qc.x(qr[2])
    qc.x(qr[4])
    qc.cx(qr[1], qr[2])
    qc.cx(qr[1], qr[4])  # 1 lower target as in paper
    qc.cx(qr[1], qr[5])  # 1 lower target as in paper

    return qc.to_gate(label="U2")


def U1_n35() -> Gate:
    """
    1->4
    11->9
    16->29
    """

    qr = QuantumRegister(6)
    qc = QuantumCircuit(qr)

    qc.swap(qr[5], qr[3])  # 1 higher target as in paper
    qc.cx(qr[2], qr[3])
    qc.cx(qr[2], qr[4])
    qc.cx(qr[2], qr[5])

    qc.cx(qr[1], qr[2])
    qc.cx(qr[1], qr[3])
    qc.cx(qr[1], qr[5])  # 1 lower target as in paper

    return qc.to_gate(label="U1")


def append_controlled_U(U: Gate, circ: QuantumCircuit, control, target):
    circ.append(U.control(1), [control] + target)
    return


def create_append_controlled_U(U: Gate):
    def new_function(circ: QuantumCircuit, control, target):
        return append_controlled_U(U, circ, control, target)

    return new_function


def append_CU(CU: Gate, circ: QuantumCircuit, control, target):
    circ.append(CU, [control] + target)


def create_append_CU(CU: Gate):

    return lambda circ, control, target: append_CU(CU, circ, control, target)


def run_circuit(
    qc: QuantumCircuit,
    shots: int = 20000,
    noise: bool = False,
    return_res: bool = False,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
):
    """
    Call to run QuantumCircuit object on aer backend.

    return_res: flag for looking into Statevector or Densitymatrix objects in results

    Args:
        qc (QuantumCircuit): Which circuit shall get executed.
        shots (int): Amount of shots to run the simulation for.
        noisy

    Returns:
        Counts of results
    """
    basis_gates_1q = [
        "id",
        "x",
        "p",
        "h",
        "z",
        "t",
        "s",
        "rz",
        "reset",  # reset gets measure + one qubit error
    ]

    basis_gates_2q = ["cx", "cp", "cz"]

    basis_gates = basis_gates_1q + basis_gates_2q + ["unitary"]

    if noise:
        sim = AerSimulator(
            noise_model=backend_noise_model(
                noise_scale=noise_scale, relaxation_noise_scale=relaxation_noise_scale
            ),
            basis_gates=basis_gates,
        )

    else:
        sim = AerSimulator(basis_gates=basis_gates)

    qc_transpiled = transpile(qc, sim, basis_gates=basis_gates)

    # print(qc_transpiled)
    print(f"Depth: {qc_transpiled.depth()}, Width: {qc_transpiled.num_qubits}")

    result = sim.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts(qc_transpiled)
    if return_res:
        return counts, result
    return counts


def old_run_circuit(
    qc: QuantumCircuit, shots: int = 10000, noisy: bool = False, return_res=False
):
    """
    Call to run QuantumCircuit object on aer backend.

    return_res: flag for looking into Statevector or Densitymatrix objects in results

    Args:
        qc (QuantumCircuit): Which circuit shall get executed.
        shots (int): Amount of shots to run the simulation for.
        noisy

    Returns:
        Counts of results
    """
    noise_model = noise.NoiseModel()
    error_1 = noise.depolarizing_error(0.001, 1)
    error_2 = noise.depolarizing_error(0.01, 2)
    error_3 = noise.depolarizing_error(0.1, 3)

    noise_model.add_all_qubit_quantum_error(error_1, ["h", "x", "u2", "p", "rz", "id"])
    noise_model.add_all_qubit_quantum_error(
        error_2, ["cx", "crz", "cp", "swap"]
    )  # prob dont include swap as its just used for convenience
    noise_model.add_all_qubit_quantum_error(error_3, ["ccz", "ccx"])

    basis_gates = noise_model.basis_gates
    if noisy:
        sim = AerSimulator(noise_model=noise_model, basis_gates=basis_gates)
    else:
        sim = AerSimulator(basis_gates=basis_gates)

    qc_transpiled = transpile(qc, sim)

    result = sim.run(qc_transpiled, shots=shots).result()
    counts = result.get_counts(qc_transpiled)
    if return_res:
        return counts, result
    return counts


def append_ccx(circ: QuantumCircuit, control_1, control_2, target):
    """
    Applies the expanded ccx gate to the handed circuit.
    Cutting advice for EJPP: ctrl1 (root) | ctrl2 + target such that no corrections must get applied.

    Args:
        circ (QuantumCircuit): Quantum circuit the Toffoli shall get applied to.
        control_1 (qubit): Control qubit number 1 which only acts as control within the expansion.
        control_2 (qubit): Control qubit number 2 which is acted up on with controlled not operations within the expansion.
        target (qubit): The target of the ccx operation.

    Returns:
        Void
    """
    circ.h(target)
    circ.cx(control_2, target)
    circ.p(-m.pi / 4, target)
    circ.cx(control_1, target)
    circ.p(m.pi / 4, target)
    circ.cx(control_2, target)
    circ.p(-m.pi / 4, target)
    circ.cx(control_1, target)
    circ.p(m.pi / 4, control_2)
    circ.p(m.pi / 4, target)
    circ.h(target)
    circ.cx(control_1, control_2)
    circ.p(-m.pi / 4, control_2)
    circ.p(m.pi / 4, control_1)
    circ.cx(control_1, control_2)
    circ.name = "ccx EJPP"
    return


def qft_gate(n: int, draw=False, swap=True) -> Gate:
    """
    |x> = |x_{n-1}...x_{0}>
    |x_i> -> (1/sqrt(2))[|0>+exp(2*pi*i*x/2^(i+1)))|1>]

    Args:
        n (int): number of bits

    Returns:
        Gate: QFT Gate
    """
    qr = QuantumRegister(n, name="x")
    qc = QuantumCircuit(qr)
    for qubit in reversed(range(n)):
        qc.h(qr[qubit])  # apply Hadamard on each qubit
        for control_qubit in reversed(range(qubit)):
            # apply controlled phase gate on each qubit with varying control bits. The phase is directly related to the distance of target to control qubit.
            qc.cp(
                m.pi / 2 ** (qubit - control_qubit),
                control_qubit=qr[control_qubit],
                target_qubit=qr[qubit],
            )
    if swap:
        for qubit in range(n // 2):
            qc.swap(
                qr[qubit], qr[n - qubit - 1]
            )  # reorder results on qubits to comply with order on qubits in mathematical representation of fourier transform.
    if draw:
        print(qc.draw())
    return qc.to_gate(label="QFT{}".format(n))


def EJPP_start(draw: bool = False) -> QuantumCircuit:
    """
    Generates staring process of EJPP protocol with qubits specified in order: root, communication root, communication target. Additionally one classical bit is needed to feed forward information.
    The total bit count is 3 qubits + 1 clbit.

    Args:
        draw (bool): Signaler whether circuit shall get printed.

    Return:
        Quantum circuit representing the EJPP starting process.
    """
    root = QuantumRegister(1)
    communication_root = QuantumRegister(1)
    communication_target = QuantumRegister(1)
    classical_memory = ClassicalRegister(1, name="classical bus")
    circ = QuantumCircuit(
        root[:] + communication_root[:] + communication_target[:], classical_memory
    )
    append_reset(circ=circ, qubit=[communication_root[0]], duration=DURATION_RESET)
    append_reset(circ=circ, qubit=[communication_target[0]], duration=DURATION_RESET)
    circ.append(
        ebit_gate(DURATION_EBIT), [communication_root[0], communication_target[0]]
    )
    circ.cx(root[0], communication_root[0])
    append_measure(
        circ=circ,
        qubit=[communication_root[0]],
        clbit=[classical_memory[0]],
        duration=DURATION_MEASURE,
    )
    # with circ.if_test((classical_memory[0], 1)):
    #     circ.x(communication_target)
    circ.x(communication_target).c_if(classical_memory[0], 1)
    circ.name = "EJPP start"
    if draw:
        print(circ)
    return circ


def append_EJPP_start(
    circ: QuantumCircuit,
    root,
    root_communication,
    target_communication,
    classical_memory,
    draw: bool = False,
):
    """
    This function appends the EJPP starting process to a handed quantum circuit.

    Args:
        circ (QuantumCircuit): The quantum circuit the EJPP starting process shall get applied to.
        root (qubit): The root qubit for the EJPP protocol.
        root_communication (qubit): The communication qubit which shall get used on the root QPU for the EJPP process.
        target_communication (qubit): The communication qubit which shall get used on the target QPU for the EJPP process.
        classical_memory (clbit): A classical bit which will be used for classical communication within the EJPP protocol.
        draw (bool): Specifier whether the circuit shall be displayed.

    Returns:
        Void
    """
    circ.append(
        EJPP_start(draw=draw),
        root + root_communication + target_communication,
        classical_memory,
    )


def EJPP_end(draw: bool = False) -> QuantumCircuit:
    """
    Generates end process of EJPP protocol with qubits specified in order: root, communication target. Additionally one classical bit is needed to feed forward information.
    The total bit count is 2 qubits + 1 clbit.

    Args:
        draw (bool): Signaler whether circuit shall get printed.

    Return:
        Quantum circuit representing the EJPP end process.
    """
    root = QuantumRegister(1)
    communication_target = QuantumRegister(1)
    classical_memory = ClassicalRegister(1, name="classical bus")
    circ = QuantumCircuit(root[:] + communication_target[:], classical_memory)
    circ.h(communication_target[0])
    append_measure(
        circ=circ,
        qubit=[communication_target[0]],
        clbit=[classical_memory[0]],
        duration=DURATION_MEASURE,
    )
    # with circ.if_test((classical_memory[0], 1)):
    #     circ.z(root[0])
    circ.z(root[0]).c_if(classical_memory[0], 1)
    circ.name = "EJPP end"
    if draw:
        print(circ)
    return circ


def append_EJPP_end(
    circ: QuantumCircuit,
    root,
    target_communication,
    classical_memory,
    draw: bool = False,
) -> QuantumCircuit:
    """
    This function appends the EJPP end process to a handed quantum circuit.

    Args:
        circ (QuantumCircuit): The quantum circuit the EJPP starting process shall get applied to.
        root (qubit): The root qubit for the EJPP protocol.
        target_communication (qubit): The communication qubit which shall get used on the target QPU for the EJPP process.
        classical_memory (clbit): A classical bit which will be used for classical communication within the EJPP protocol.
        draw (bool): Specifier whether the circuit shall be displayed.

    Returns:
        Void
    """
    circ.append(EJPP_end(draw=draw), root + target_communication, classical_memory)


def id_noiseless() -> Gate:
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    id_op = Operator([[1, 0], [0, 1]])
    qc.unitary(id_op, qr[0], label="id_unitary")

    return qc.to_gate(label="id noiseless")


def id_noiseless_2q() -> Gate:
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr)
    id_op = Operator([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    qc.unitary(id_op, [qr[0], qr[1]], label="id_unitary_2q")

    return qc.to_gate(label="id noiseless_2q")


def reset(duration: int = 0) -> Instruction:
    """
    This function returns a reset instruction which may be prolonged by noiseless identity gates to model longer execution times.

    Args:
        duration (int): Specifier of amount of noiseless identity gates by which the reset gate shall get prolonged.
    """
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    for i in range(duration):
        qc.append(id_noiseless(), qr[:])
    qc.reset(qr[0])

    return qc.to_instruction(label="reset prolonged")


def measure(duration: int = 0) -> Instruction:
    qr = QuantumRegister(1)
    cr = ClassicalRegister(1)
    qc = QuantumCircuit(qr, cr)
    for i in range(duration):
        qc.append(id_noiseless(), qr[:])
    qc.measure(qr, cr)

    return qc.to_instruction(label="measure prolonged")


def ebit_gate(duration: int = 0) -> Gate:
    qr = QuantumRegister(2)
    qc = QuantumCircuit(qr, name="ebit")

    cx_op = Operator([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]])

    h_op = Operator(
        [[1 / np.sqrt(2), 1 / np.sqrt(2)], [1 / np.sqrt(2), -1 / np.sqrt(2)]]
    )
    for _ in range(duration):
        qc.append(id_noiseless_2q(), [qr[0], qr[1]])
    qc.append(id_noiseless(), [qr[1]])
    qc.unitary(h_op, qr[0], label="ebit_h")
    qc.unitary(cx_op, [qr[0], qr[1]], label="ebit_cx")

    return qc.to_gate(label="ebit")


def append_ebit(circ: QuantumCircuit, qubits, duration=0):
    circ.append(ebit_gate(duration), qubits)


def append_reset(circ: QuantumCircuit, qubit, duration=0):
    for q in qubit:
        circ.append(reset(duration), [q])


def append_measure(circ: QuantumCircuit, qubit, clbit, duration=0):
    for i, q in enumerate(qubit):
        circ.append(measure(duration), [q], [clbit[i]])


def get_N15_circ_data():
    U_list = [
        create_append_controlled_U(U4_n15_base2()),
        create_append_controlled_U(U2_n15_base2()),
        create_append_controlled_U(U1_n15_base2()),
    ]
    top_c = 3
    bot_c = 4
    init_ind = 3
    init_one = True
    return U_list, top_c, bot_c, init_one, init_ind


def get_N21B2_circ_data():
    U_list = [
        create_append_controlled_U(U4_n21_base2()),
        create_append_controlled_U(U2_n21_base2()),
        create_append_controlled_U(U1_n21_base2()),
    ]
    top_c = 3
    bot_c = 5
    init_ind = 4
    init_one = True
    return U_list, top_c, bot_c, init_one, init_ind


def get_N21B4_circ_data():
    U_list = [
        create_append_controlled_U(U1_n21_base4()),
        create_append_controlled_U(U2_n21_base4()),
        create_append_controlled_U(U4_n21_base4()),
    ]
    top_c = 3
    bot_c = 2
    init_ind = 4
    init_one = False
    return U_list, top_c, bot_c, init_one, init_ind


def get_N35_circ_data():
    U_list = [
        create_append_controlled_U(U4_n35()),
        create_append_controlled_U(U2_n35()),
        create_append_controlled_U(U1_n35()),
    ]
    top_c = 3
    bot_c = 6
    init_ind = 5
    init_one = False
    return U_list, top_c, bot_c, init_one, init_ind


from qiskit.circuit.library import CXGate, XGate

if __name__ == "__main__":

    qr = QuantumRegister(3)

    qc = QuantumCircuit(qr)

    create_append_CU(XGate().control())(qc, qr[1], [qr[0]])
    print(qc)
    create_append_CU(XGate().control())(qc, qr[0], [qr[2]])
    print(qc)

    qr = QuantumRegister(3)

    qc = QuantumCircuit(qr)

    create_append_controlled_U(XGate())(qc, qr[1], [qr[0]])
    print(qc)
    create_append_controlled_U(XGate())(qc, qr[0], [qr[2]])
    print(qc)
