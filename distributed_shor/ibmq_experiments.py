from qiskit_ibm_runtime import SamplerV2 as Sampler, Session, QiskitRuntimeService
from qiskit_aer import AerSimulator
from datetime import datetime
import pickle
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime.fake_provider import FakeSherbrooke, FakeTorino
from qiskit import QuantumCircuit
from qiskit.transpiler.passes import RemoveBarriers
import os
from qiskit.visualization import plot_histogram
from matplotlib import pyplot as plt
from distributed_shor.graph_utils import CircuitGraph

""" 
TODO:
https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/runtime_service

 - do local testing first with sherbrooke fake backend then go from there

 - transpile for backend

 - sampler result is now digit for whatever reason.. so adjust postprocessing for distributions

 - plug into kl divergence postprocessing pipeline and compare
"""


def test_qc():

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    return qc


def run_circ_local(transpiled_qc, backend, shots):
    with Session(backend=backend) as session:

        sampler = Sampler(mode=session)
        job = sampler.run([transpiled_qc], shots=shots)
        result = job.result()
        return result


def run_circ_cloud(transpiled_qc, backend, shots):

    with Session(backend=backend) as session:

        sampler = Sampler(mode=session)
        job = sampler.run([transpiled_qc], shots=shots)

        t = datetime.now()
        folder_path = f"./src/semi_iterative_comparison/results/ibm_hw/{t.year}_{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}"
        os.mkdir(folder_path)
        with open(folder_path + "/job_id.txt", "w") as file:
            file.write(job.job_id())
        return job.job_id()


def retrieve_cloud_results(path):
    service = QiskitRuntimeService()
    job_id = None
    with open(path + "/job_id.txt", "r") as file:
        job_id = file.read()

    job = service.job(job_id)
    job_circuit = job.inputs["pubs"][0][0]
    result = job.result()
    t = datetime.now()
    filename = path + "/data.pkl"
    print(f"Results of job {job_id} stored in {filename}")
    with open(
        filename,
        "wb",
    ) as file:
        pickle.dump([job_circuit, result], file)
    return result


def start_experiment(qc, shots, optimization_level=3, rm_barriers=True):
    # token = ""
    # with open("./src/api_token.txt", "r") as file:
    #     token = file.read()

    service = QiskitRuntimeService()
    # print(service.backends())

    sherbrooke = service.backend("ibm_sherbrooke")
    aer_sim = AerSimulator()
    fake_sherbrooke = FakeSherbrooke()
    fake_torino = FakeTorino()

    backend = aer_sim

    pm = generate_preset_pass_manager(
        target=backend.target, optimization_level=optimization_level
    )
    if rm_barriers:
        qc = RemoveBarriers()(qc)
    transpiled_qc = pm.run(qc)
    CircuitGraph(transpiled_qc).plot_graph()
    print(qc)
    print(transpiled_qc)
    print(transpiled_qc.depth())
    if "ibm" in backend.name:
        job_id = run_circ_cloud(transpiled_qc, backend, shots)
        print(job_id)
    else:
        result = run_circ_local(transpiled_qc, backend, shots)

        counts = result[0].data.meas.get_counts()
        plot_histogram(counts)
        plt.show()
        print(counts, optimization_level, rm_barriers)


def read_results(path):
    result = retrieve_cloud_results(path)

    counts = result[0].data.meas.get_counts()
    print(counts)
    return counts


if __name__ == "__main__":
    # start_experiment()
    read_results("./src/semi_iterative_comparison/results/ibm_hw/2024_10_15_16_30_14")


# def get_results(job_id):
#     print(job.job_id(), job.done())
#     result = job.result()[0].data.meas.get_counts()

#     print(result)
#     t = datetime.now()
#     filename = f"./src/semi_iterative_comparison/results/ibm_hw/{t.year}_{t.month}_{t.day}_{t.hour}_{t.minute}_{t.second}"
#     print(filename)
#     with open(
#         filename,
#         "wb",
#     ) as file:
#         pickle.dump(result, file)

#     res_loaded = None
#     with open(filename, "rb") as file:
#         res_loaded = pickle.load(file)

#     print(res_loaded)
