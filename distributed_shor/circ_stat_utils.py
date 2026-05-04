from __future__ import annotations
from qiskit import *
from qiskit.circuit import Gate, Instruction, CircuitInstruction, Clbit
from qiskit.transpiler.passes import RemoveBarriers
import matplotlib.pyplot as plt
from distributed_shor.basis_gates import get_basis_gates
from qiskit.quantum_info import Operator
from distributed_shor.graph_utils import CircuitGraph
from distributed_shor.weight_utils import instruction_weights


def a_to_bin(a: int, n: int) -> list[bool]:
    string_list = [int(s) for s in f"{a:0{n}b}"[::-1]]
    return string_list


def count_circuit_ops_QPU(qc: QuantumCircuit, qpu_qubits: list[QuantumRegister]):
    """
    TODO: for now just gate counts, think about where this might not be enough or
          if this is sufficient with correct barrier placements
    """
    qubit_instructions = [[] for _ in qc.qubits]
    QPU_qubit_indices = [[qubit._index for qubit in QPU_l] for QPU_l in qpu_qubits]

    for inst in qc.decompose():
        for q_i in inst[1]:
            if inst[0].name != "barrier":
                qubit_instructions[q_i._index].append(inst)

    qubit_ops = [len(l) for l in qubit_instructions]
    max_ops_qpu = [max([qubit_ops[i] for i in QPU_q]) for QPU_q in QPU_qubit_indices]
    print(f"Ops per Qubit: {qubit_ops}")
    print(f"Max QPU  Qubit ops: {max_ops_qpu}")


def circuit_moments(qc: QuantumCircuit) -> list[list[CircuitInstruction]]:
    """
    assumes transpiled circuit
    """

    moments = []

    for instruction in qc:
        if len(moments) == 0:  # add first moment if first instruction
            moments.append([instruction])
        else:
            new_moment = True
            lowest_add_layer = 0
            for i, moment in reversed(list(enumerate(moments))):

                can_add = True
                for moment_inst in moment:
                    qubits_inst = instruction[1]
                    for q in qubits_inst:
                        if q in moment_inst[1]:
                            can_add = False
                            break

                    if not can_add:
                        break

                    clbits_inst = instruction[2]
                    clbits_inst += instruction.operation.condition_bits
                    for c in clbits_inst:
                        if (
                            c in moment_inst[2]
                            or c in moment_inst.operation.condition_bits
                        ):
                            can_add = False
                            break

                    if not can_add:
                        break

                if can_add:
                    lowest_add_layer = i
                    new_moment = False
                else:
                    break

            if new_moment:
                moments.append([instruction])
            else:
                moments[lowest_add_layer].append(instruction)

    for i, moment in enumerate(moments):
        print(f"Moment {i + 1}:")
        for op in moment:
            print(f"  {op.operation.name} , {op.operation.label}")

    return moments


def circuit_depths_simple(
    qc: QuantumCircuit,
    moments: list[list[CircuitInstruction]],
    # qpu_qubits: list[QuantumRegister],
):

    # for i, moment in enumerate(moments):
    #     print(f"Moment {i + 1}:")
    #     for op in moment:
    #         print(f"  {op.operation.name} , {op.operation.label}")

    # QPU_qubit_indices = [[qubit._index for qubit in QPU_l] for QPU_l in qpu_qubits]

    qubit_depths = {q: 0 for q in qc.qubits}

    qubit_hit = {q: False for q in qc.qubits}

    for i, moment in reversed(list(enumerate(moments))):

        for instruction in moment:
            for q in instruction[1]:
                if not qubit_hit[q]:
                    qubit_depths[q] = i + 1
                    qubit_hit[q] = True
        if all(qubit_hit.values()):
            break

    return qubit_depths


def circuit_depths_reset(
    qc: QuantumCircuit,
    moments: list[list[CircuitInstruction]],
    # qpu_qubits: list[QuantumRegister],
):

    # QPU_qubit_indices = [[qubit._index for qubit in QPU_l] for QPU_l in qpu_qubits]

    qubit_depths = {q: [] for q in qc.qubits}

    current_qubit_depths = {q: 0 for q in qc.qubits}
    for i, moment in enumerate(moments):

        for instruction in moment:
            for q in instruction[1]:
                if instruction.operation.name == "reset":
                    if len(qubit_depths[q]) > 0:
                        qubit_depths[q].append(i - sum(qubit_depths[q]))
                    else:
                        qubit_depths[q].append(
                            current_qubit_depths[q]
                        )  # TODO: clarify here if the last gate or the moment before the reset should be relevant for depth, probably this is always the same anyways?

                current_qubit_depths[q] = i + 1

    for q in current_qubit_depths.keys():
        if len(qubit_depths[q]) > 0:
            qubit_depths[q].append(current_qubit_depths[q] - sum(qubit_depths[q]))
        else:
            qubit_depths[q].append(current_qubit_depths[q])
    return qubit_depths


def circuit_moments_squeezed(
    qc: QuantumCircuit,
    moments: list[list[CircuitInstruction]],
) -> list[list[CircuitInstruction]]:
    """
    squeeze moments to see better individual qubit depth
    """

    qubit_solid_ind = [0 for _ in qc.qubits]
    clbit_solid_ind = [0 for _ in qc.clbits]

    for i, moment in reversed(list(enumerate(moments))):
        new_moment = copy.deepcopy(moment)
        for inst in moment:
            first_inst = False
            for q in inst[1]:
                q_i = q._index
                if not qubit_solid_ind[q_i]:
                    first_inst = True  # if first instruction on bit, no squeezing

            clbits_inst = inst[2]
            clbits_inst += inst.operation.condition_bits
            for c in clbits_inst:
                c_i = c._index
                if not clbit_solid_ind[c_i]:
                    first_inst = True  # if first instruction on bit, no squeezing

            squeezed = False
            if not first_inst:
                nearest_border = len(moments)
                for q in inst[1]:
                    q_i = q._index
                    if qubit_solid_ind[q_i] < nearest_border:
                        nearest_border = qubit_solid_ind[q_i]

                for c in clbits_inst:
                    c_i = c._index
                    if clbit_solid_ind[c_i] < nearest_border:
                        nearest_border = clbit_solid_ind[c_i]

                if nearest_border - i - 1 > 0:  # if instruction not directly at border
                    moments[nearest_border - 1].append(inst)
                    new_moment.remove(inst)
                    squeezed = True

            for q in inst[1]:
                q_i = q._index
                if squeezed:
                    qubit_solid_ind[q_i] = nearest_border - 1
                else:
                    qubit_solid_ind[q_i] = i
            for c in clbits_inst:
                c_i = c._index
                if squeezed:
                    clbit_solid_ind[c_i] = nearest_border - 1
                else:
                    clbit_solid_ind[c_i] = i

        moments[i] = new_moment
    print(qubit_solid_ind)
    print(clbit_solid_ind)
    for i, moment in enumerate(moments):
        print(f"Moment {i + 1}:")
        for op in moment:
            print(f"  {op}")
    return moments


def circuit_moments_reset_squeeze(
    qc: QuantumCircuit,
    sq_moments: list[list[CircuitInstruction]],
) -> list[list[CircuitInstruction]]:
    # squeeze operatations away from resets

    bound_ind = len(sq_moments)

    qubit_solid_ind = [bound_ind for _ in qc.qubits]
    clbit_solid_ind = [bound_ind for _ in qc.clbits]

    for i, moment in list(enumerate(sq_moments)):
        new_moment = copy.deepcopy(moment)
        for inst in moment:
            first_inst = False
            for q in inst[1]:
                q_i = q._index
                if qubit_solid_ind[q_i] == bound_ind:
                    first_inst = True  # if first instruction on bit, no squeezing

            clbits_inst = inst[2]
            clbits_inst += inst.operation.condition_bits
            for c in clbits_inst:
                c_i = c._index
                if clbit_solid_ind[c_i] == bound_ind:
                    first_inst = True  # if first instruction on bit, no squeezing

            squeezed = False
            if not first_inst and inst.operation.name != "reset":
                nearest_border = 0
                for q in inst[1]:
                    q_i = q._index
                    if qubit_solid_ind[q_i] > nearest_border:
                        nearest_border = qubit_solid_ind[q_i]

                for c in clbits_inst:
                    c_i = c._index
                    if clbit_solid_ind[c_i] > nearest_border:
                        nearest_border = clbit_solid_ind[c_i]

                if i + 1 - nearest_border > 0:  # if instruction not directly at border
                    sq_moments[nearest_border + 1].append(inst)
                    new_moment.remove(inst)
                    squeezed = True

            for q in inst[1]:
                q_i = q._index
                if squeezed:
                    qubit_solid_ind[q_i] = nearest_border + 1
                else:
                    qubit_solid_ind[q_i] = i
            for c in clbits_inst:
                c_i = c._index
                if squeezed:
                    clbit_solid_ind[c_i] = nearest_border + 1
                else:
                    clbit_solid_ind[c_i] = i

        sq_moments[i] = new_moment
    print(qubit_solid_ind)
    print(clbit_solid_ind)
    for i, moment in enumerate(sq_moments):
        print(f"Moment {i + 1}:")
        for op in moment:
            print(f"  {op}")
    return sq_moments


def moments_to_circuit(
    qc_old: QuantumCircuit, moments: list[list[CircuitInstruction]]
) -> QuantumCircuit:

    qrs_old = []
    for q in qc_old.qubits:
        if q._register not in qrs_old:
            qrs_old.append(q._register)

    qrs = []
    qr_old_new_map = {}

    for qr_o in qrs_old:
        qr_new = QuantumRegister(size=qr_o.size)
        qr_old_new_map[qr_o.name] = qr_new
        qrs.append(qr_new)

    crs_old = []
    for c in qc_old.clbits:
        if c._register not in crs_old:
            crs_old.append(c._register)

    crs = []
    cr_old_new_map = {}
    for cr_o in crs_old:
        cr_new = ClassicalRegister(cr_o.size)
        cr_old_new_map[cr_o.name] = cr_new
        crs.append(cr_new)

    qc = QuantumCircuit()
    for qr in qrs:
        qc.add_register(qr)
    for cr in crs:
        qc.add_register(cr)

    for moment in moments:
        for circ_inst in moment:
            cl_indices = [(c._register.name, c._index) for c in circ_inst[2]]
            q_indices = [(q._register.name, q._index) for q in circ_inst[1]]
            cond_indices = [
                (c._register.name, c._index) for c in circ_inst.operation.condition_bits
            ]

            # op = copy.deepcopy(circ_inst.operation)

            # TODO: more aligned with qiskit but check if this breaks!!
            op = circ_inst.operation.to_mutable()
            if len(cond_indices) > 0:
                cond = op.condition[1]
                # cond_reg = ClassicalRegister(bits=[cr_old_new_map[c_r][c_i] for c_r, c_i in cond_indices])
                op.c_if(cr_old_new_map[cond_indices[0][0]][cond_indices[0][1]], cond)
            qc.append(
                op,
                [qr_old_new_map[q_r][q_i] for q_r, q_i in q_indices],
                [cr_old_new_map[c_r][c_i] for c_r, c_i in cl_indices],
            )

        qc.barrier(qc.qubits)

    return qc


def id_noise() -> Gate:
    qr = QuantumRegister(1)
    qc = QuantumCircuit(qr)
    id_op = Operator([[1, 0], [0, 1]])
    qc.unitary(id_op, qr[0], label="id_noise")

    return qc.to_gate(label="id noise")


def first_moments_per_qubit(qc, moments):
    first_moments = {q: -1 for q in qc.qubits}

    for i, moment in enumerate(moments):
        for inst in moment:
            for q in inst[1]:
                if first_moments[q] == -1:
                    first_moments[q] = i

        if all(v > -1 for v in first_moments.values()):
            break

    return first_moments


def last_moments_per_qubit(qc, moments):
    bound = len(moments)
    last_moments = {q: bound for q in qc.qubits}

    for i, moment in reversed(list(enumerate(moments))):
        for inst in moment:
            for q in inst[1]:
                if last_moments[q] == bound:
                    last_moments[q] = i

        if all(v < bound for v in last_moments.values()):
            break

    return last_moments


def moments_to_circuit_noise(
    qc_old: QuantumCircuit, moments: list[list[CircuitInstruction]], squeezed=True
) -> QuantumCircuit:

    qrs_old = []
    for q in qc_old.qubits:
        if q._register not in qrs_old:
            qrs_old.append(q._register)

    qrs = []
    qr_old_new_map = {}

    for qr_o in qrs_old:
        qr_new = QuantumRegister(size=qr_o.size)
        qr_old_new_map[qr_o.name] = qr_new
        qrs.append(qr_new)

    crs_old = []
    for c in qc_old.clbits:
        if c._register not in crs_old:
            crs_old.append(c._register)

    crs = []
    cr_old_new_map = {}
    for cr_o in crs_old:
        cr_new = ClassicalRegister(cr_o.size)
        cr_old_new_map[cr_o.name] = cr_new
        crs.append(cr_new)

    qc = QuantumCircuit()
    for qr in qrs:
        qc.add_register(qr)
    for cr in crs:
        qc.add_register(cr)

    first = first_moments_per_qubit(qc_old, moments)
    last = last_moments_per_qubit(qc_old, moments)

    for i, moment in enumerate(moments):

        qubit_used = {q: False for q in qc.qubits}

        for circ_inst in moment:
            cl_indices = [(c._register.name, c._index) for c in circ_inst[2]]
            q_indices = [(q._register.name, q._index) for q in circ_inst[1]]
            cond_indices = [
                (c._register.name, c._index) for c in circ_inst.operation.condition_bits
            ]

            # op = copy.deepcopy(circ_inst.operation)

            # TODO: more aligned with qiskit but check if this breaks!!
            op = circ_inst.operation.to_mutable()
            if len(cond_indices) > 0:
                cond = op.condition[1]
                # cond_reg = ClassicalRegister(bits=[cr_old_new_map[c_r][c_i] for c_r, c_i in cond_indices])
                op.c_if(cr_old_new_map[cond_indices[0][0]][cond_indices[0][1]], cond)
            qc.append(
                op,
                [qr_old_new_map[q_r][q_i] for q_r, q_i in q_indices],
                [cr_old_new_map[c_r][c_i] for c_r, c_i in cl_indices],
            )

            for q_r, q_i in q_indices:
                qubit_used[qr_old_new_map[q_r][q_i]] = True

        for q_old in qc_old.qubits:
            q_new = qr_old_new_map[q_old._register.name][q_old._index]
            if not qubit_used[q_new] and i < last[q_old]:
                if (not squeezed) or i > first[q_old]:
                    qc.append(id_noise(), [q_new])

        qc.barrier(qc.qubits)

    return qc


def get_transpiled_circuit(circ: QuantumCircuit):
    """
    does it make sense to remove barriers later?
    """
    rb = RemoveBarriers()

    qc_transpiled = transpile(circ, basis_gates=get_basis_gates())

    qc_nob = rb(qc_transpiled)

    return qc_nob


def get_moment_circ(circ: QuantumCircuit, squeezed=True, noise=True):

    moments = circuit_moments_cond(circ)
    if squeezed:
        mom_sq = circuit_moments_squeezed_cond(circ, moments)
        moments = circuit_moments_reset_squeeze_cond(circ, mom_sq)

    if noise:
        circ_new = moments_to_circuit_noise(
            circ, moments, squeezed=False
        )  # why squeezed=False here?
    else:
        circ_new = moments_to_circuit(circ, moments)
    return circ_new


def circuit_depths_reset_squeezed(
    qc: QuantumCircuit,
    moments: list[list[CircuitInstruction]],
    # qpu_qubits: list[QuantumRegister],
):

    # QPU_qubit_indices = [[qubit._index for qubit in QPU_l] for QPU_l in qpu_qubits]
    qubit_points = {q: [] for q in qc.qubits}
    qubit_depths = {c: [] for c in qc.qubits}

    current_qubit_depths = {q: 0 for q in qc.qubits}
    first_inst = {q: -1 for q in qc.qubits}

    for i, moment in enumerate(moments):

        for instruction in moment:
            for q in instruction[1]:
                if first_inst[q] == -1:  # if first op on qubit
                    first_inst[q] = i
                    qubit_points[q].append(i)

                if instruction.operation.name == "reset":
                    qubit_points[q].append(i)
                    if len(qubit_depths[q]) > 0:
                        qubit_depths[q].append(i - sum(qubit_depths[q]) - first_inst[q])
                    else:
                        qubit_depths[q].append(
                            current_qubit_depths[q] - first_inst[q]
                        )  # TODO: clarify here if the last gate or the moment before the reset should be relevant for depth, probably this is always the same anyways?

                current_qubit_depths[q] = i + 1

    for q in qc.qubits:
        qubit_points[q].append(current_qubit_depths[q])

    for q in qc.qubits:
        if len(qubit_depths[q]) > 0:
            qubit_depths[q].append(
                current_qubit_depths[q] - sum(qubit_depths[q]) - first_inst[q]
            )
        else:
            if first_inst[q] == -1:
                qubit_depths[q].append(current_qubit_depths[q])
            else:
                qubit_depths[q].append(current_qubit_depths[q] - first_inst[q])
    # print(qubit_depths)
    # print(f"Points: {qubit_points}")
    return qubit_points


def cl_val_dictionary(conbits: list[Clbit], cvals: list[bool]) -> dict[Clbit, bool]:
    return {conbits[i]: cvals[i] for i in range(len(conbits))}


def is_inst_addable(instruction, moment):
    can_add = True

    can_skip = True
    if instruction.operation.name != "p":
        can_skip = False

    conbits_inst = instruction.operation.condition_bits
    is_cond_gate = len(conbits_inst) > 0
    if is_cond_gate:
        cond = instruction.operation.condition[1]

    for moment_inst in moment:

        cond_overlap = False
        conbits_inst_m = moment_inst.operation.condition_bits
        m_is_cond_gate = len(conbits_inst_m) > 0

        if is_cond_gate and m_is_cond_gate:
            cond_overlap = True
            cond_m = moment_inst.operation.condition[1]
            # TODO build test case that checks if this works when cond bits have non empty intersection and are not subsets of each other
            c_vals_inst = a_to_bin(cond, len(conbits_inst))
            c_vals_m_inst = a_to_bin(cond_m, len(conbits_inst_m))

            c_inst_dict = cl_val_dictionary(conbits_inst, c_vals_inst)
            c_inst_m_dict = cl_val_dictionary(conbits_inst_m, c_vals_m_inst)

            for c in conbits_inst:
                if c in conbits_inst_m:
                    if c_inst_dict[c] != c_inst_m_dict[c]:
                        cond_overlap = False
                        break

        if (
            not m_is_cond_gate or not is_cond_gate or cond_overlap
        ):  # only skip comparing if both are conditional and there is no overlap in cond bit values

            qubits_inst = instruction[1]
            for q in qubits_inst:
                if q in moment_inst[1]:
                    can_add = False
                    if moment_inst.operation.name not in ["p", "z"]:
                        can_skip = False
                    break

            if not can_add and not can_skip:
                break

            if not cond_overlap:
                # for both cond we only have to check if the qubits overlap
                # in all other cases using the same classical bits is not allowed
                clbits_inst = instruction[2]

                clbits_inst += instruction.operation.condition_bits
                for c in clbits_inst:

                    if c in moment_inst[2] or c in moment_inst.operation.condition_bits:
                        can_add = False
                        if moment_inst.operation.name not in ["p", "z"]:
                            can_skip = False
                        break

            if not can_add and not can_skip:
                break

    return can_add, can_skip


def circuit_moments_cond(qc: QuantumCircuit) -> list[list[CircuitInstruction]]:
    """
    assumes transpiled circuit
    """

    moments = []

    for instruction in qc:
        if len(moments) == 0:  # add first moment if first instruction
            moments.append([instruction])
        else:
            new_moment = True
            lowest_add_layer = 0
            for i, moment in reversed(list(enumerate(moments))):

                can_add, can_skip = is_inst_addable(instruction, moment)
                if can_add:
                    lowest_add_layer = i
                    new_moment = False
                else:
                    if not can_skip:
                        break

            if new_moment:
                moments.append([instruction])
            else:
                moments[lowest_add_layer].append(instruction)
    """
    for i, moment in enumerate(moments):
        print(f"Moment {i + 1}:")
        for op in moment:
            print(f"  {op}")
    """
    return moments


def circuit_moments_squeezed_cond(
    qc: QuantumCircuit,
    moments: list[list[CircuitInstruction]],
) -> list[list[CircuitInstruction]]:
    """
    squeeze moments to see better individual qubit depth

    TODO!! same approach prob wont work because of cond

    Do same as moments before but with reverse order and first instruction fixed as foundation (think about what if first instruction is also conditional!!)
    """

    qubit_bound_ind = {q: 0 for q in qc.qubits}
    clbit_bound_ind = {c: 0 for c in qc.clbits}

    new_moments = [[] for i in range(len(moments))]

    for i, moment in reversed(list(enumerate(moments))):

        for inst in moment:

            # this is just depth so just use that
            first_inst = False
            for q in inst[1]:
                if not qubit_bound_ind[q]:
                    first_inst = True  # if first instruction on bit, no squeezing
                    qubit_bound_ind[q] = i

            clbits_inst = inst[2]
            clbits_inst += inst.operation.condition_bits
            for c in clbits_inst:
                if not clbit_bound_ind[c]:
                    first_inst = True  # if first instruction on bit, no squeezing
                    clbit_bound_ind[c] = i

            highest_layer = i
            if not first_inst:
                nearest_border = len(moments)
                for q in inst[1]:
                    if qubit_bound_ind[q] < nearest_border:
                        nearest_border = qubit_bound_ind[q]

                for c in clbits_inst:
                    if clbit_bound_ind[c] < nearest_border:
                        nearest_border = clbit_bound_ind[c]

                for l in range(i + 1, nearest_border + 1):  # go up to border
                    moment = new_moments[l]
                    can_add, can_skip = is_inst_addable(inst, moment)
                    if can_add:
                        highest_layer = l
                    else:
                        if not can_skip:
                            break

            new_moments[highest_layer].append(inst)

    # print(qubit_bound_ind)
    # print(clbit_bound_ind)
    """
    for i, moment in enumerate(new_moments):
        print(f"Moment {i + 1}:")
        for op in moment:
            print(f"  {op}")
    """
    return new_moments


def circuit_moments_reset_squeeze_cond(
    qc: QuantumCircuit,
    sq_moments: list[list[CircuitInstruction]],
) -> list[list[CircuitInstruction]]:
    """
    squeeze moments to see better individual qubit depth

    """

    bound_ind = len(sq_moments)

    qubit_bound_ind = {q: bound_ind for q in qc.qubits}
    clbit_bound_ind = {c: bound_ind for c in qc.clbits}

    new_moments = [[] for i in range(len(sq_moments))]

    for i, moment in enumerate(sq_moments):

        for inst in moment:

            if inst.operation.name == "measure":
                pass
            # this is just depth so just use that
            first_inst = False
            for q in inst[1]:
                if qubit_bound_ind[q] == bound_ind:
                    first_inst = True  # if first instruction on bit, no squeezing
                    qubit_bound_ind[q] = i

            clbits_inst = inst[2]
            clbits_inst += inst.operation.condition_bits
            for c in clbits_inst:
                if clbit_bound_ind[c] == bound_ind:
                    # (not here!) if first instruction on bit, no squeezing
                    clbit_bound_ind[c] = i

            lowest_layer = i

            if not first_inst and inst.operation.name != "reset":
                nearest_border = 0
                for q in inst[1]:
                    if qubit_bound_ind[q] > nearest_border:
                        nearest_border = qubit_bound_ind[q]

                # dont block measure from being moved, but block cond from being moved over measures
                if inst.operation.name != "measure":
                    for c in clbits_inst:
                        if clbit_bound_ind[c] > nearest_border:
                            nearest_border = clbit_bound_ind[c]

                for l in range(i - 1, nearest_border + 1, -1):  # go up to border
                    moment = new_moments[l]
                    can_add, can_skip = is_inst_addable(inst, moment)
                    if can_add:
                        lowest_layer = l
                    else:
                        if not can_skip:
                            break

            new_moments[lowest_layer].append(inst)

    # print(qubit_bound_ind)
    # print(clbit_bound_ind)
    """
    for i, moment in enumerate(new_moments):
        print(f"Moment {i + 1}:")
        for op in moment:
            print(f"  {op}")
    """
    return new_moments


def depth_from_points(qubit_points):

    depths = {q: 0 for q in qubit_points.keys()}

    for q in qubit_points.keys():
        ps = qubit_points[q]
        depths[q] = [ps[i + 1] - ps[i] for i in range(len(ps) - 1)]

    return depths


def QPU_depth(qubit_points, qpu_mapping: list[list[int]]):
    # not adjusted yet as probably not used!
    num_qubits = len(qubit_points)
    assert len(qubit_points) == sum([len(q_list) for q_list in qpu_mapping])
    depths = []
    for q_list in qpu_mapping:
        start = -1
        stop = 0
        for q_i in q_list:
            if len(qubit_points[q_i]) > 0:
                last = qubit_points[q_i][-1]
                first = qubit_points[q_i][0]
                if last > stop:
                    stop = last
                if start < 0 or first < start:
                    start = first

        depths.append(stop - start)
    print(f"QPUs: {depths}")


def print_depths(depths):
    for q in depths.keys():
        print(f"{q._register.name} {q._index}: {depths[q]}")


def depths_dict(depths):
    depths_d = {}
    for q in depths.keys():
        depths_d[f"{q._register.name} {q._index}"] = depths[q]

    return depths_d


def max_depth(depths):
    max_d = 0
    for q in depths.keys():
        if isinstance(depths[q], list):
            if len(depths[q]) > 0:
                max_i = max(depths[q])
            else:
                max_i = 0
        else:
            max_i = depths[q]
        if max_i > max_d:
            max_d = max_i

    return max_d


def depths_test(c):
    #############################################################
    qc_nob = get_transpiled_circuit(c)
    print("----------------")
    print(qc_nob)
    moments = circuit_moments_cond(qc_nob)
    mom_sq = circuit_moments_squeezed_cond(qc_nob, moments)

    print()
    print("Circuit depths simple:")
    print_depths(circuit_depths_simple(qc_nob, moments))

    print()
    print("Circuit depths reset:")
    print_depths(circuit_depths_reset(qc_nob, moments))

    print()
    print(f"Qiskit depth: {qc_nob.depth()}")
    q_p = circuit_depths_reset_squeezed(qc_nob, mom_sq)

    print()
    print("circuit depths reset squeezed:")
    print_depths(q_p)

    print()
    print("Depth from points:")
    print_depths(depth_from_points(q_p))

    # QPU_depth(q_p, [[0, 1, 2], [3, 4, 5], [6, 7, 8]])

    circ_sq = moments_to_circuit(qc_nob, mom_sq)
    circ_m = moments_to_circuit(qc_nob, moments)
    print("moments circ")
    print(circ_m)
    print("moments squeezed circ")
    print(circ_sq)
    # circ_sq.draw(output="mpl")
    # plt.show()

    ##################################################################


def depths_from_circ(circ, squeeze=True):

    moments = circuit_moments_cond(circ)

    depths_simple = depths_dict(
        circuit_depths_simple(circ, moments)
    )  # for now we stay with simple moments
    if squeeze:
        mom_sq = circuit_moments_squeezed_cond(circ, moments)
        q_p = circuit_depths_reset_squeezed(circ, mom_sq)
        depths_squeezed = depths_dict(depth_from_points(q_p))
        return depths_simple, depths_squeezed
    else:
        return depths_simple


def count_circuit_ops(qc: QuantumCircuit):

    qubit_instructions = {q: [] for q in qc.qubits}
    instructions_per_length = [0, 0, 0, 0]
    for inst in qc:
        if (
            inst[0].name != "barrier"
            and inst[0].label != "id_unitary"
            and inst[0].label != "id_unitary_2q"
            and inst[0].label != "id_noise"
        ):
            for q_i in inst[1]:
                qubit_instructions[q_i].append(inst)

            instructions_per_length[len(inst[1])] += 1

    qubit_ops = [len(qubit_instructions[q]) for q in qc.qubits]

    # print(f"Ops per Qubit: {qubit_ops}")
    # print(f"instrucitons per length: {instructions_per_length}")
    return qubit_ops, instructions_per_length


def circuit_depths_top_sort(qc: QuantumCircuit, weights: instruction_weights):

    bits = qc.qubits + qc.clbits
    bit_depth = {bit: 0 for bit in bits}

    for instruction in qc:

        assert instruction.operation.name != "barrier"
        # skip those instructions
        if (
            instruction.operation.name != "barrier"
            and instruction.operation.label != "id_unitary"
            and instruction.operation.label != "id_unitary_2q"
            and instruction.operation.label != "id_noise"
            and instruction.operation.label != "id noiseless"
            and instruction.operation.label != "id noiseless_2q"
        ):

            clbits_inst = [c for c in instruction.clbits]
            clbits_inst += instruction.operation.condition_bits

            qubits_inst = [q for q in instruction.qubits]

            bits_inst = clbits_inst + qubits_inst

            weight = 1
            if instruction.operation.label in weights.weight_dict:
                weight = weights.weight_dict[instruction.operation.label]
            elif instruction.operation.name in weights.weight_dict:
                weight = weights.weight_dict[instruction.operation.name]
            elif len(qubits_inst) == 2:
                weight = weights.weight_two_qubit
            elif len(qubits_inst) == 1:
                weight = weights.weight_one_qubit

            max_duration = max([bit_depth[bit_inst] for bit_inst in bits_inst])

            for bit_inst in bits_inst:
                bit_depth[bit_inst] = max_duration + weight

    return max(bit_depth.values())


import pickle


def get_avg_CU_durations(result_path, weight_label, x_vals):

    avg_CU_durations = []

    results = {}

    with open(
        result_path + f"depth_{weight_label}/" + "avg_CU.pkl", "rb"
    ) as fp:  # Unpickling
        results = pickle.load(fp)

    avg_CU_durations = [results[x]["average_duration"] for x in x_vals]

    return avg_CU_durations
