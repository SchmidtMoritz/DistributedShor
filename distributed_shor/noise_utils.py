from __future__ import annotations
from qiskit_aer.noise import (
    NoiseModel,
    thermal_relaxation_error,
    depolarizing_error,
    pauli_error,
)
from dataclasses import dataclass
from dataclasses_json import dataclass_json

from qiskit.circuit import Instruction
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit.quantum_info import average_gate_fidelity
from typing import Union
from distributed_shor.basis_gates import (
    get_basis_gates,
    get_basis_gates_1q,
    get_basis_gates_2q,
)
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino

"""

-   change measurements to include (variable) identities in front
    to model longer readout times

    -   make custom "identity" gates as they should not be noisy since they are just modeling time of the readout

-   replace ebits with custom "h","cx"

-   homogenous noise (equal for all qubits)

-   heterogeneous noise

-   pick values according to

    -   IBM specifications

    -   papers (circuit level noise etc)


"""


@dataclass_json
@dataclass
class BackendSnapshot:
    avg_t1: float
    avg_t2: float
    avg_1q_error: float
    avg_1q_duration: float
    avg_2q_error: float
    avg_2q_duration: float
    avg_ro_error: float
    avg_ro_duration: float

    def save_snapshot(self, path):
        with open(path, "w") as f:
            f.write(self.to_json())

    @classmethod
    def from_snapshot(cls, path):
        with open(path, "r") as f:
            return cls.from_json(f.read())


def homogenous_noise_model(
    bs: BackendSnapshot,
    custom_ebit_error: Union[list[float], bool] = False,
    noise_scale: float = 1,
    relaxation_noise_scale: float = 1,
) -> NoiseModel:
    """
    Noise:

        -1q gate: TR + DP if TR doesnt already match avg 1q error of backend

        -2q gate; DP based on avg 2q error, TR doesnt really work here as it is a 1q error

        -measure: for now bit flip based on ro error of backend

        -reset: 1q error + measure error

        -ebit gates: simple DP error channel with variable error rate

        -no noise ids: part of basis_gates for transpilation but no noise

    """

    # for the current snapshots id has the same error as any other one qubit gate so we dont treat it differently for now

    noise_model = NoiseModel(get_basis_gates())

    tr_1q = thermal_relaxation_error(
        bs.avg_t1 / relaxation_noise_scale,
        bs.avg_t2 / relaxation_noise_scale,
        bs.avg_1q_duration,
    )
    tr_2q = thermal_relaxation_error(
        bs.avg_t1 / relaxation_noise_scale,
        bs.avg_t2 / relaxation_noise_scale,
        bs.avg_2q_duration,
    ) ^ thermal_relaxation_error(
        bs.avg_t1 / relaxation_noise_scale,
        bs.avg_t2 / relaxation_noise_scale,
        bs.avg_2q_duration,
    )

    # noise_model.add_all_qubit_quantum_error(tr_2q, basis_gates_2q)

    avg_fid_1q = average_gate_fidelity(tr_1q.to_quantumchannel())
    avg_fid_2q = average_gate_fidelity(tr_2q.to_quantumchannel())

    # follows from F = (1-p_dp)F_TR + p_dp F_D
    # where F_D = 1/d (completely depolarizing channel) and d is the dim of the hilbert space
    # our target error is 1-F

    error_1q = tr_1q
    er_2q = tr_2q

    if 1 - avg_fid_1q < noise_scale * bs.avg_1q_error:

        # hilbert space of dim 2 for one qubit gate
        p_dp_q1 = (noise_scale * bs.avg_1q_error - (1 - avg_fid_1q)) / (
            avg_fid_1q - 1 / 2.0
        )
        dp_q1 = depolarizing_error(p_dp_q1, 1)
        error_1q = error_1q.compose(dp_q1)

    noise_model.add_all_qubit_quantum_error(error_1q, get_basis_gates_1q())

    if 1 - avg_fid_2q < noise_scale * bs.avg_2q_error:

        # hilbert space of dim 4 for two qubit gate
        p_dp_q2 = (noise_scale * bs.avg_2q_error - (1 - avg_fid_2q)) / (
            avg_fid_2q - 1 / 4.0
        )
        dp_q2 = depolarizing_error(p_dp_q2, 2)
        er_2q = er_2q.compose(dp_q2)

    noise_model.add_all_qubit_quantum_error(er_2q, get_basis_gates_2q())

    # p_dp_q2 = bs.avg_2q_error / (1 - (1 / 4))
    # dp_q2 = depolarizing_error(noise_scale * p_dp_q2, 2)
    # noise_model.add_all_qubit_quantum_error(dp_q2, get_basis_gates_2q())

    bf_ro = pauli_error(
        [("X", noise_scale * bs.avg_ro_error), ("I", 1 - noise_scale * bs.avg_ro_error)]
    )

    noise_model.add_all_qubit_quantum_error(
        bf_ro,
        ["measure"],
    )

    error_reset = error_1q.compose(bf_ro)
    noise_model.add_all_qubit_quantum_error(
        error_reset,
        ["reset"],
    )
    if not custom_ebit_error:
        ebit1 = depolarizing_error(noise_scale * bs.avg_1q_error, 1)
        ebit2 = depolarizing_error(noise_scale * bs.avg_2q_error, 2)
    else:
        ebit1 = depolarizing_error(custom_ebit_error[0], 1)
        ebit2 = depolarizing_error(custom_ebit_error[1], 2)

    noise_model.add_all_qubit_quantum_error(ebit1, ["ebit_h"])
    noise_model.add_all_qubit_quantum_error(ebit2, ["ebit_cx"])

    # id_noise_err = depolarizing_error(noise_scale*bs.avg_1q_error / 2, 1)
    noise_model.add_all_qubit_quantum_error(tr_1q, ["id_noise"])
    return noise_model


def custom_noise_model() -> NoiseModel:
    """
    Noise:

        -1q gate: TR + DP if TR doesnt already match avg 1q error of backend

        -2q gate; DP based on avg 2q error, TR doesnt really work here as it is a 1q error

        -measure: for now bit flip based on ro error of backend

        -reset: 1q error + measure error

        -ebit gates: simple DP error channel with variable error rate

        -no noise ids: part of basis_gates for transpilation but no noise

    """

    # for the current snapshots id has the same error as any other one qubit gate so we dont treat it differently for now

    noise_model = NoiseModel(get_basis_gates())

    p_1q = 0.002
    p_2q = 0.004
    p_r = 0.01
    p_ebit1 = 0.05
    p_ebit2 = 0.1
    p_id = 0.003

    error_1q = depolarizing_error(p_1q, 1)
    error_2q = depolarizing_error(p_2q, 2)

    noise_model.add_all_qubit_quantum_error(error_1q, get_basis_gates_1q())
    noise_model.add_all_qubit_quantum_error(error_2q, get_basis_gates_2q())

    bf_ro = pauli_error([("X", p_r), ("I", 1 - p_r)])

    noise_model.add_all_qubit_quantum_error(
        bf_ro,
        ["measure"],
    )

    error_reset = error_1q.compose(bf_ro)
    noise_model.add_all_qubit_quantum_error(
        error_reset,
        ["reset"],
    )

    ebit1 = depolarizing_error(p_ebit1, 1)
    ebit2 = depolarizing_error(p_ebit2, 2)

    noise_model.add_all_qubit_quantum_error(ebit1, ["ebit_h"], [0])
    noise_model.add_all_qubit_quantum_error(ebit2, ["ebit_cx"], [0, 1])

    id_noise_err = depolarizing_error(p_id, 1)
    noise_model.add_all_qubit_quantum_error(id_noise_err, ["id_noise"])
    return noise_model


def get_backend_snapshot():
    # should replace this with
    # https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.device.readout_error_values.html
    # https://qiskit.github.io/qiskit-aer/stubs/qiskit_aer.noise.device.basic_device_gate_errors.html#qiskit_aer.noise.device.basic_device_gate_errors

    # requires saved ibmq account
    service = QiskitRuntimeService()

    print(service.backends())
    backend = service.backend("ibm_sherbrooke")
    # "ibm_marrakesh"
    # "ibm_fez"
    # fake_sherbrooke = FakeSherbrooke()
    # fake_torino = FakeTorino()
    # backend = fake_torino

    n = backend.num_qubits
    avg_t1 = 0
    avg_t2 = 0
    for i in range(n):
        props = backend.qubit_properties(i)
        avg_t1 += 1 / n * props.t1
        avg_t2 += 1 / n * props.t2

    print(avg_t1)
    print(avg_t2)

    inst_list = [
        (op.name, op.num_qubits)
        for op in backend.operations
        if (
            isinstance(op, Instruction)
            and op.name != "reset"
            and op.name != "measure"
            and op.name != "delay"
        )
    ]
    print(backend.operations)
    print(inst_list)
    avg_1q_error = 0
    avg_1q_d = 0

    avg_2q_error = 0
    avg_2q_d = 0

    avg_readout_error = 0
    avg_readout_d = 0

    readout_list = []
    reset_list = []
    q1_list = []
    q2_list = []
    q2_m_list = []
    # avg_reset_error just gets modeled through measure + cond x

    avg_reset_d = 0

    q1_c = 0
    q2_c = 0
    for inst, n_q in inst_list:
        avg_d = 0
        avg_e = 0
        d_list = []
        for inst_prop in backend.target[inst].values():
            d_list.append(inst_prop.duration)
            avg_d += 1 / n * inst_prop.duration
            avg_e += 1 / n * inst_prop.error

        print(f"{inst}: duration {avg_d}, error {avg_e}")
        if n_q == 1 and (avg_e != 0.0):  # ignore free rz for now
            q1_list.append(avg_d)
            avg_1q_error = (avg_e + avg_1q_error * q1_c) / (q1_c + 1)
            avg_1q_d = (avg_d + avg_1q_d * q1_c) / (q1_c + 1)
            q1_c += 1
        if n_q == 2:
            q2_m_list.append(sorted(d_list)[int(len(d_list) / 2)])
            q2_list.append(avg_d)
            avg_2q_error = (avg_e + avg_2q_error * q2_c) / (q2_c + 1)
            avg_2q_d = (avg_d + avg_2q_d * q2_c) / (q2_c + 1)
            q2_c += 1

    for inst_prop in backend.target["measure"].values():
        readout_list.append(inst_prop.duration)
        avg_readout_d += 1 / n * inst_prop.duration
        avg_readout_error += 1 / n * inst_prop.error

    for inst_prop in backend.target["reset"].values():
        reset_list.append(inst_prop.duration)
        avg_reset_d += 1 / n * inst_prop.duration

    median_1q = sorted(q1_list)[int(len(q1_list) / 2)]
    median_2q = sorted(q2_list)[int(len(q2_list) / 2)]
    median_readout = sorted(readout_list)[int(len(readout_list) / 2)]
    median_reset = sorted(reset_list)[int(len(reset_list) / 2)]

    print(avg_1q_error, avg_1q_d, avg_2q_error, avg_2q_d)
    print(avg_readout_d, avg_readout_error)
    print("--------------------------------")
    print(avg_1q_d, q2_m_list, median_readout, median_reset)
    print("--------------------------------")
    return BackendSnapshot(
        avg_t1=avg_t1,
        avg_t2=avg_t2,
        avg_1q_error=avg_1q_error,
        avg_1q_duration=avg_1q_d,
        avg_2q_error=avg_2q_error,
        avg_2q_duration=avg_2q_d,
        avg_ro_error=avg_readout_error,
        avg_ro_duration=avg_readout_d,
    )


def backend_noise_model(noise_scale: float = 1, relaxation_noise_scale: float = 1):
    bs = BackendSnapshot.from_snapshot("backend_snapshots/test.json")
    return homogenous_noise_model(
        bs, noise_scale=noise_scale, relaxation_noise_scale=relaxation_noise_scale
    )


if __name__ == "__main__":
    bs = get_backend_snapshot()
    print(bs.avg_1q_duration)
    print(bs.avg_2q_duration)
    print(bs.avg_ro_duration)
