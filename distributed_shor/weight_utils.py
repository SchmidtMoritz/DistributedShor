from dataclasses import dataclass


@dataclass
class instruction_weights:
    weight_dict: dict
    weight_one_qubit: int
    weight_two_qubit: int
    label: str


weights_eagle = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 1216,
        "measure": 1216,
        "reset": 1216,
    },
    weight_one_qubit=57,
    weight_two_qubit=533,
    label="eagle",
)


weights_sherbrooke = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 1216,
        "measure": 1216,
        "reset": 1276,
    },
    weight_one_qubit=57,
    weight_two_qubit=533,
    label="sherbrooke",
)


weights_heron = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 1560,
        "measure": 1560,
        "reset": 1560,
    },
    weight_one_qubit=32,
    weight_two_qubit=68,
    label="heron",
)

weights_torino = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 1560,
        "measure": 1560,
        "reset": 1708,
    },
    weight_one_qubit=32,
    weight_two_qubit=68,
    label="torino",
)

weights_fez = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 1560,
        "measure": 1560,
        "reset": 1584,
    },
    weight_one_qubit=24,
    weight_two_qubit=84,
    label="fez",
)

weights_marrakesh = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 2100,
        "measure": 2100,
        "reset": 2236,
    },
    weight_one_qubit=36,
    weight_two_qubit=68,
    label="marrakesh",
)

weights_heron_ebit = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 156000,  # 156 micro seconds
        "measure": 1560,
        "reset": 1560,
    },
    weight_one_qubit=32,
    weight_two_qubit=68,
    label="heron_ebit",
)

weights_aspdac = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 12,
        "measure": 11,
        "reset": 11,
    },
    weight_one_qubit=1,
    weight_two_qubit=1,
    label="aspdac",
)

weights_uniform = instruction_weights(
    weight_dict={},
    weight_one_qubit=1,
    weight_two_qubit=1,
    label="uniform",
)

weights_neutral_atom = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 10000000,  # 10ms
        "measure": 10000000,  # 10ms
        "reset": 10002000,  # 10 ms measure + single qubit gate
    },
    weight_one_qubit=2000,  # 2 mus
    weight_two_qubit=400,  # 400 ns
    label="neutral_atom",
)

weights_infleqtion = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 6000000,  # 6ms
        "measure": 6000000,  # 6ms
        "reset": 6004100,  # 6ms measure + single qubit gate
    },
    weight_one_qubit=4100,  # assume 4.1 mus global rot gate, not 250ns phase gate
    weight_two_qubit=416,  # 416ns
    label="infleqtion",
)

weights_ionq_forte = instruction_weights(
    weight_dict={
        "ebit_h": 0,
        "ebit_cx": 5500000,  # 5.5ms
        "measure": 150000,  # 6mus
        "reset": 50000,  # 50mus
    },
    weight_one_qubit=130000,  # 130mus
    weight_two_qubit=970000,  # 970mus
    label="ionq_forte",
)


def get_weights_heron_ebit_duration(ebit_duration=1560):
    weights_heron_ebit_custom_duration = instruction_weights(
        weight_dict={
            "ebit_h": 0,
            "ebit_cx": ebit_duration,  # 156 micro seconds
            "measure": 1560,
            "reset": 1560,
        },
        weight_one_qubit=32,
        weight_two_qubit=68,
        label=f"heron_ebit_{ebit_duration}",
    )
    return weights_heron_ebit_custom_duration


def get_weights_neutral_atom_ebit_duration(ebit_duration=10000000):
    weights_neutral_atom_custom_duration = instruction_weights(
        weight_dict={
            "ebit_h": 0,
            "ebit_cx": ebit_duration,
            "measure": 10000000,  # 10ms
            "reset": 10002000,  # 10 ms measure + single qubit gate
        },
        weight_one_qubit=2000,  # 2 mus
        weight_two_qubit=400,  # 400 ns
        label=f"neutral_atom_{ebit_duration}",
    )
    return weights_neutral_atom_custom_duration


def get_weights_ionq_forte_ebit_duration(ebit_duration=5500000):
    weights_ionq_forte_custom_duration = instruction_weights(
        weight_dict={
            "ebit_h": 0,
            "ebit_cx": ebit_duration,  # 5.5ms
            "measure": 150000,  # 6mus
            "reset": 50000,  # 50mus
        },
        weight_one_qubit=130000,  # 130mus
        weight_two_qubit=970000,  # 970mus
        label=f"ionq_forte_{ebit_duration}",
    )
    return weights_ionq_forte_custom_duration


def get_mono_upper_bound(weights: instruction_weights, m: int):
    t_bound = (m - 1) * (
        (1 + m) * weights.weight_one_qubit  # 2 hadamards + (m-1) phases
        + weights.weight_dict["measure"]
        + weights.weight_dict["reset"]
    )
    return t_bound


def get_start_stop_duration(weights: instruction_weights, reset=True, ebit=True):
    start_d = (
        +weights.weight_two_qubit
        + weights.weight_dict["measure"]
        + weights.weight_one_qubit
    )
    stop_d = weights.weight_dict["measure"] + 2 * weights.weight_one_qubit

    sum_d = start_d + stop_d

    if ebit:
        ebit_d = weights.weight_dict["ebit_cx"]
        sum_d += ebit_d
    if reset:
        sum_d += weights.weight_dict["reset"]

    return sum_d
