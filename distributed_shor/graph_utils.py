import networkx as nx
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


class CircuitGraph:

    def __init__(
        self,
        circuit: QuantumCircuit,
        weight_dict: dict[str, int] = {
            "ebit_h": 0,
            "ebit_cx": 12,  # 1560,  # 12
            "measure": 11,  # 1560,  # 11
            "reset": 11,  # 1560,  # 11
        },
        weight_one_qubit: int = 1,
        weight_two_qubit: int = 1,  # 68
    ) -> None:
        self.graph = nx.DiGraph()
        self.weight_dict = weight_dict
        self.weight_one_qubit = weight_one_qubit
        self.weight_two_qubit = weight_two_qubit
        self.qubits = circuit.qubits
        self.clbits = circuit.clbits

        # maybe not needed just use index of instruction or instruction itself as defining object for node

        # self.topological_sort = []
        self.graph.add_node("source", y_pos=0)
        i = 0

        last_node_on_qubit = {qubit: "source" for qubit in circuit.qubits}
        last_node_on_clbit = {clbit: "source" for clbit in circuit.clbits}

        # go through all instructions in circuit and add edge between node and predecessor on same qubit/clbit
        for instruction in circuit:

            vert = i
            i += 1
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

                if len(clbits_inst) > 0:
                    y_pos = -self.clbits.index(clbits_inst[0]) - 1
                else:
                    y_pos = (len(self.qubits) - 1) - self.qubits.index(qubits_inst[0])

                # y_pos *= 2.0 / (len(self.clbits) + len(self.qubits) - 1)

                self.graph.add_node(vert, y_pos=y_pos)

                weight = 1
                if instruction.operation.label in self.weight_dict:
                    weight = self.weight_dict[instruction.operation.label]
                elif instruction.operation.name in self.weight_dict:
                    weight = self.weight_dict[instruction.operation.name]
                elif len(qubits_inst) == 2:
                    weight = self.weight_two_qubit
                elif len(qubits_inst) == 1:
                    weight = self.weight_one_qubit

                for qubit in qubits_inst:
                    last = last_node_on_qubit[qubit]
                    self.graph.add_edge(last, vert, weight=weight)
                    last_node_on_qubit[qubit] = vert

                for clbit in clbits_inst:
                    last = last_node_on_clbit[clbit]
                    self.graph.add_edge(last, vert, weight=weight)
                    last_node_on_clbit[clbit] = vert

    def plot_graph(self):

        pos = {}
        for layer, nodes in enumerate(nx.topological_generations(self.graph)):
            # `multipartite_layout` expects the layer as a node attribute, so add the
            # numeric layer value as a node attribute
            for node in nodes:
                # self.graph.nodes[node]["layer"] = layer
                pos[node] = (layer, self.graph.nodes[node]["y_pos"])
        # Compute the multipartite_layout using the "layer" node attribute

        # pos = nx.multipartite_layout(self.graph, subset_key="layer")

        fig, ax = plt.subplots()
        edge_colors = [self.graph.edges[u, v]["weight"] for u, v in self.graph.edges()]
        nx.draw_networkx_nodes(self.graph, pos=pos, ax=ax, alpha=0.2)
        nx.draw_networkx_edges(
            self.graph,
            pos=pos,
            ax=ax,
            edge_color=edge_colors,
            alpha=0.7,
            width=[e / 3.0 for e in edge_colors],
        )
        ax.set_title("DAG layout in topological order")
        fig.tight_layout()
        plt.show()

    def get_depth(self):
        topological_gens = [gen for gen in nx.topological_generations(self.graph)]
        return len(topological_gens) - 1  # ignore the source layer

    def get_weighted_depth(self):
        return nx.dag_longest_path_length(self.graph, weight="weight")


if __name__ == "__main__":
    cr = ClassicalRegister(2)
    qr = QuantumRegister(3)
    qc = QuantumCircuit(qr, cr)

    qc.h(qr)
    qc.measure(qr[2], cr[1])
    qc.cx(qr[0], qr[1])
    qc.cx(qr[1], qr[2]).c_if(cr[1], 1)
    qc.measure(qr[1], cr[0])

    G = CircuitGraph(qc)
    G.plot_graph()
