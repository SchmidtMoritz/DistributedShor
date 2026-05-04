def get_basis_gates_1q():
    basis_gates_1q = [
        "id",
        "x",
        "p",
        "h",
        "z",
        "t",
        "s",
        "rz",
    ]
    return basis_gates_1q


def get_basis_gates_2q():
    basis_gates_2q = ["cx", "cp", "cz"]
    return basis_gates_2q


def get_basis_gates():

    basis_gates = (
        get_basis_gates_1q()
        + get_basis_gates_2q()
        + ["unitary"]
        + ["reset", "measure"]
        # + ["ccx"]
    )

    return basis_gates
