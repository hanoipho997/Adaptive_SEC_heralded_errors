import networkx as nx

def tanner_graph(H, check_prefix):
    G = nx.Graph()
    n_checks, n_qubits = H.shape
    for i in range(n_checks):
        for j in range(n_qubits):
            if H[i, j] == 1:
                G.add_edge(f"{check_prefix}{i}", f"q{j}")
    return G
