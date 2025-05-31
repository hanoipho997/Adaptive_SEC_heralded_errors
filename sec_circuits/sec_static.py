import stim as stim
from graphs.tanner_graph import tanner_graph as tanner_graph
import networkx as nx
from graphs.edge_coloring import edge_color_bipartite as edge_color_bipartite
import numpy as np

def syndrome_extraction_round(ancillas, qubit_index, colorings_x, colorings_z, p1,  p2,  with_noise=True):
    """
    Buiding 1 round of sydrome extraction circuit. Here we use simple coloring: Edges E_X before Edges E_Z
    """
    circuit = stim.Circuit()
    # Intialize ancillas
    circuit.append("R", ancillas)
    circuit.append("H", ancillas)
    if with_noise:
        circuit.append("DEPOLARIZE1", ancillas, p1)

    # X checks
    for edges in colorings_x:
        for c, q in edges:
            c_idx = qubit_index[c]
            q_idx = qubit_index[q]
            circuit.append("H", [q_idx])
            if with_noise:
                circuit.append("DEPOLARIZE1", [q_idx], p1)
            circuit.append("CZ", [c_idx, q_idx])
            if with_noise:
                circuit.append("DEPOLARIZE2", [c_idx, q_idx], p2)
            circuit.append("H", [q_idx])
            if with_noise:
                circuit.append("DEPOLARIZE1", [q_idx], p1)

    # Z checks
    for edges in colorings_z:
        for c, q in edges:
            c_idx = qubit_index[c]
            q_idx = qubit_index[q]
            circuit.append("CZ", [c_idx, q_idx])
            if with_noise:
                circuit.append("DEPOLARIZE2", [c_idx, q_idx], p2)

    circuit.append("H", ancillas)
    if with_noise:
        circuit.append("DEPOLARIZE1", ancillas, p1)
    circuit.append("MR", ancillas)
    
    return circuit

def stim_circuit_from_checks(Hx, Hz, logical_obs_qubits, num_rounds=3,
                             p1=0.001, p2=0.005):  # noise parameters
    """
    Function that generates Stim circuit from parity check matrices H_X, H_Z
    """
    circuit = stim.Circuit()

    Gx = tanner_graph(Hx, "cx")
    Gz = tanner_graph(Hz, "cz")
    colorings_x = edge_color_bipartite(Gx)
    colorings_z = edge_color_bipartite(Gz)

    cx, n = Hx.shape
    cz, _ = Hz.shape

    qubit_index = {f"q{i}": i for i in range(n)}
    qubit_index.update({f"cx{i}": n + i for i in range(cx)})
    qubit_index.update({f"cz{i}": n + cx + i for i in range(cz)})
   
    data = [qubit_index[f"q{i}"] for i in range(n)]
    check_x = [qubit_index[f"cx{i}"] for i in range(cx)]
    check_z = [qubit_index[f"cz{i}"] for i in range(cz)]
    ancillas = check_x + check_z
    total_ancillas = len(ancillas)

    # Init
    circuit.append("R", data)
    # One initial noiseless measurement round (baseline).
    circuit += syndrome_extraction_round(ancillas, qubit_index, colorings_x, colorings_z, p1,  p2,  with_noise=False)
    # num_rounds noisy rounds to detect syndromes.    
    for _ in range(num_rounds):
        circuit += syndrome_extraction_round(ancillas, qubit_index, colorings_x, colorings_z, p1,  p2,  with_noise=True)
    # One final noiseless measurement round    
    circuit += syndrome_extraction_round(ancillas, qubit_index, colorings_x, colorings_z, p1,  p2,  with_noise=False)

    # DETECTORS: compare ancilla measurements between rounds m_{i,r} and m{i,r+1}
    for r in range(num_rounds + 1):
        for i in range(total_ancillas):
            m_i_r = -(num_rounds+2-r) * total_ancillas + i
            m_i_r_plus_1 = m_i_r + total_ancillas
            circuit.append("DETECTOR", [stim.target_rec(m_i_r), stim.target_rec(m_i_r_plus_1)])

    # LOGICAL observable (Z or X)
    l_qb = list(np.where(logical_obs_qubits == 1)[1])
  
    circuit.append("M", l_qb)
    obs_targets = [stim.target_rec(-i - 1) for i in range(len(l_qb))]
    circuit.append("OBSERVABLE_INCLUDE", obs_targets, 0)

    return circuit
