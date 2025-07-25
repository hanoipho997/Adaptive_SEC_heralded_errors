from css import *
import networkx as nx
import numpy as np
import stim
from collections import defaultdict


class CircuitBuilder():
    """Class to build syndrome extraction circuit.
    In the perfect, noiseless case.

    Args:
        code (CSS): Input is a CSS code
    """

    def __init__(self, code: CSS):
        self.code = code
        self.n = self.code.n
        self.c_x = self.code.H_X.shape[0]
        self.c_z = self.code.H_Z.shape[0]

    @property
    def data(self) -> list:
        """Indices of the data qubits

        Returns:
            list: index list
        """
        return [i for i in range(self.n)]

    @property
    def ancilla_x(self) -> list:
        """Indices of the X ancilla qubits

        Returns:
            list: index list
        """
        return [self.n + i for i in range(self.c_x)]

    @property
    def ancilla_z(self) -> list:
        """Indices of the Z ancilla qubits

        Returns:
            list: index list
        """
        return [self.n + self.c_x + i for i in range(self.c_z)]
    pass

    def edge_coloring(self, biadj_matrix: np.array, offset: int = 0) -> dict:
        """Edge coloring of a bipartite graph given by its biadjacency matrix.

        Args:
            biadj_matrix (np.array): Bi-adjacency matrix
            offset (int, optional): Offset for different check qubits. Defaults to 0.

        Returns:
            dict: edge coloring dictionnary
        """
        rows, cols = biadj_matrix.shape
        G = nx.Graph()

        # Create bipartite graph: left = 0..rows-1, right = rows..rows+cols-1
        for i in range(rows):
            for j in range(cols):
                if biadj_matrix[i, j]:
                    G.add_edge(i + cols, j)

        # Create line graph: nodes = edges of G, edges = shared nodes in G
        L = nx.line_graph(G)

        # Node coloring on line graph => edge coloring on G
        coloring = nx.coloring.greedy_color(L, strategy="largest_first")

        # Map color -> original (row, column) edges
        color_to_edges = defaultdict(list)
        for (u, v), color in coloring.items():
            if u < cols:
                row, col = v - cols, u
            else:
                row, col = u - cols, v
            color_to_edges[color].append((row + offset, col))

        return dict(color_to_edges)

    def x_coloring(self,) -> dict:
        """Edge coloring of the X-Tanner graph

        Returns:
            dict: edge coloring dictionary
        """
        return self.edge_coloring(self.code.H_X, offset=self.n)

    def z_coloring(self,) -> dict:
        """Edge coloring of the Z-Tanner graph

        Returns:
            dict: edge coloring dictionary
        """
        return self.edge_coloring(self.code.H_Z, offset=self.n + self.c_x)

    def syndrome_extraction_cz(self, coloring: dict) -> stim.Circuit:
        """CZ part in a syndrome extraction circuits given by its coloring.

        Args:
            coloring (dict): edge coloring dictionnary

        Returns:
            stim.Circuit: output circuit
        """
        circ = stim.Circuit()
        for color in coloring.keys():
            cz_inds = [x for edge in coloring[color] for x in edge]
            circ.append("CZ", cz_inds)
        return circ

    def syndrome_extraction_circuit(self) -> stim.Circuit:
        """Full syndrome extraction circuit
        We do first all the X checks then all the Z checks

        Returns:
            stim.Circuit: output circuit
        """
        circ = stim.Circuit()

        # Initialize ancilla qubits
        circ.append("RX", self.ancilla_x + self.ancilla_z)

        # Do X QEC cycle CZ
        self.x_color = self.x_coloring()
        circ.append("H", self.data)
        circ += self.syndrome_extraction_cz(self.x_color)
        circ.append("H", self.data)

        # Do Z QEC cycle CZ
        self.z_color = self.z_coloring()
        circ += self.syndrome_extraction_cz(self.z_color)

        # Measure qubits in the X basis
        circ.append("MX", self.ancilla_x + self.ancilla_z)
        return circ

    def add_logical_operators(self, basis: str) -> stim.Circuit:
        """Add (perfect) logical observable measurements

        Args:
            basis (str): X or Z basis

        Returns:
            stim.Circuit: output circuit
        """
        circ = stim.Circuit()
        if basis == "X":
            L = self.code.L_X
        elif basis == "Z":
            L = self.code.L_Z
        for ind, l in enumerate(L):
            a = "*".join([f"{basis}{ind}" for ind, val in enumerate(l) if val])
            circ.append_from_stim_program_text("MPP " + a)
        for ind, l in enumerate(L):
            circ.append("OBSERVABLE_INCLUDE", [
                        stim.target_rec(- len(L) + ind)], ind)
        return circ

    def initialization(self, basis: str) -> stim.Circuit:
        """Initialization of a memory experiment

        Args:
            basis (str): initialization basis "X" or "Z"

        Returns:
            stim.Circuit: output circuit
        """
        circ = stim.Circuit()
        if basis == "X":
            circ.append("RX", self.data)
        elif basis == "Z":
            circ.append("RZ", self.data)
        else:
            raise ValueError(
                f"basis={basis} not supported by CircuitBuilder.memory_exp... Either 'X' or 'Z'  expected")
        circ += self.add_logical_operators(basis)
        return circ

    def memory_exp(self, n_rounds: int, basis: str) -> stim.Circuit:
        """Do a memory experiment.

        With perfect initialization and final measurement.
        And n (potentially noisy) rounds 

        Args:
            n_rounds (int): number of noisy rounds  
            basis (str): "X" or "Z" type experiment

        Returns:
            stim.Circuit: full memory experiment circuit
        """
        # Initialize qubits and logicals
        circ = self.initialization(basis)

        # Noiseless syndrome extraction. Project in code's subspace
        circ += self.syndrome_extraction_circuit()

        # Potentially noisy syndrome extraction
        circ += self.noisy_rounds(n_rounds)

        # Noiseless syndrome extraction. Project back in the code subspace
        circ += self.syndrome_extraction_circuit()

        # Get the deterministic detectors
        circ += self.add_detectors(n_rounds)

        # Measure logical operator
        circ += self.add_logical_operators(basis)
        return circ

    def add_detectors(self, n_rounds: int) -> stim.Circuit:
        """Add detector to the circuit.
        We do that at the end to make it simpler

        Args:
            n_rounds (int): number of noisy QEC cycles

        Returns:
            stim.Circuit: output circuit with detectors
        """
        n_checks = self.c_x + self.c_z
        n_tot_measurements = n_checks * (n_rounds + 2)
        circ = stim.Circuit()
        for n_round in range(n_rounds + 1):
            for k in range(n_checks):
                rec_1 = stim.target_rec(- n_tot_measurements +
                                        n_round * n_checks + k)
                rec_2 = stim.target_rec(- n_tot_measurements +
                                        (n_round + 1) * n_checks + k)
                circ.append("DETECTOR", [rec_1, rec_2])
        return circ

    def noisy_rounds(self, n_rounds: int) -> stim.Circuit:
        """Do n_rounds of (potentially noisy) syndrome extraction.

        Args:
            n_rounds (int): number of qec_cycles

        Returns:
            stim.Circuit: output circuit
        """
        return self.syndrome_extraction_circuit() * n_rounds


class NoisyCircuitBuilder(CircuitBuilder):
    """Noisy syndrome extraction circuit builder

    Args:
        CircuitBuilder: child class of the perfect builder
    """

    def __init__(self, code: CSS, p_1: float = 0, p_2: float = None):
        """

        Args:
            code (CSS): CSS code
            p_1 (float, optional): single-qubit error probability. Defaults to 0.
            p_2 (float, optional): two-qubit error probability. Defaults to None (same as p_1).
        """        
        self.p_1 = p_1
        if p_2 is None:
            p_2 = p_1
        self.p_2 = p_2
        super(NoisyCircuitBuilder, self).__init__(code)

    def syndrome_extraction_cz(self, coloring: dict, with_noise: bool=False) -> stim.Circuit:
        """CZ part of the syndrome extraction circuit.

        Args:
            coloring (dict): edge coloring dict 
            with_noise (bool, optional): add noise to the system. Defaults to False.

        Returns:
            stim.Circuit: output circuit
        """        
        circ = stim.Circuit()
        for color in coloring.keys():
            cz_inds = [x for edge in coloring[color] for x in edge]
            circ.append("CZ", cz_inds)
            if with_noise:
                circ.append("DEPOLARIZE2", cz_inds, self.p_2)
        return circ

    def syndrome_extraction_circuit(self, with_noise: bool=False) -> stim.Circuit:
        """Total syndrome extraction circuit

        Args:
            with_noise (bool, optional): add noise to the system. Defaults to False.

        Returns:
            stim.Circuit: output circuit
        """        
        circ = stim.Circuit()

        # Initialize ancilla qubits
        circ.append("RX", self.ancilla_x + self.ancilla_z)
        if with_noise:
            circ.append("Z_ERROR", self.ancilla_x + self.ancilla_z, self.p_1)

        # Do X QEC cycle CZ
        self.x_color = self.x_coloring()
        circ.append("H", self.data)
        if with_noise:
            circ.append("DEPOLARIZE1", self.data, self.p_1)

        circ += self.syndrome_extraction_cz(self.x_color, with_noise)

        circ.append("H", self.data)
        if with_noise:
            circ.append("DEPOLARIZE1", self.data, self.p_1)

        # Do Z QEC cycle CZ
        self.z_color = self.z_coloring()
        circ += self.syndrome_extraction_cz(self.z_color, with_noise)

        # Measure qubits in the X basis
        if with_noise:
            circ.append("Z_ERROR", self.ancilla_x + self.ancilla_z, self.p_1)
        circ.append("MX", self.ancilla_x + self.ancilla_z)
        return circ

    def noisy_rounds(self, n_rounds: int) -> stim.Circuit:
        """Noisy rounds (put the noise to True)

        Args:
            n_rounds (int): number of QEC cycles

        Returns:
            stim.Circuit: output circuit
        """        
        return self.syndrome_extraction_circuit(with_noise=True) * n_rounds


class HeraldedNoiseCircuitBuilder(NoisyCircuitBuilder):
    """Circuit Builder for heralded errors

    Args:
        NoisyCircuitBuilder: child of noisy circuit builder class
    """    
    def __init__(self, code: CSS, p_erasure: float, p_1: float=0, p_2: float=0):
        self.p_erasure = p_erasure
        super(HeraldedNoiseCircuitBuilder, self).__init__(code, p_1, p_2)

    def noisy_rounds(self, n_rounds: int) -> stim.Circuit:
        """Add noisy rounds.
        Since they are probabilistic we cannot just do circ * n_rounds as in the parent class.

        Args:
            n_rounds (int): number of noisy QEC cycles

        Returns:
            stim.Circuit: output circuit
        """        
        circ = stim.Circuit()
        for _ in range(n_rounds):
            circ += self.syndrome_extraction_circuit(with_noise=True)
        return circ

    def syndrome_extraction_cz(self, coloring: dict, with_noise: bool=False) -> stim.Circuit:
        """CZ part of the syndrome extraction circuit.

        Args:
            coloring (dict): edge coloring
            with_noise (bool, optional): add noise or not. Defaults to False.

        Returns:
            stim.Circuit: output circuit
        """        
        circ = stim.Circuit()
        for color in coloring.keys():
            cz_inds = [x for edge in coloring[color] for x in edge]
            circ.append("CZ", cz_inds)
            if with_noise:
                circ.append("DEPOLARIZE2", cz_inds, self.p_2)

                erasures = np.random.choice([0, 1], size=len(coloring[color]), p=[
                                            1 - self.p_erasure, self.p_erasure])
                erased_edges = np.where(erasures == 1)[0]
                for edge_ind in erased_edges:
                    data, check = coloring[color][edge_ind]
                    circ.append("Z_ERROR", [data, check], 1/2)

        return circ

    # pass

class DynamicCircuitBuilder(CircuitBuilder):
    """
    Dynamic version where gates are done or not depending on other's success or failure.
    For the moment, we just change ancilla measurement basis

    TODO: Add detectors. Make sure it is working.

    """    
    def __init__(self, code: CSS, p_fail: float):
        """p_fail is the gate failure probability.
        So far no errors are added to the system yet. Objective is just to recover the detectors.
        """        
        self.p_fail = p_fail

        self.total_checks_X = []
        self.total_checks_Z = []
        self.total_completed_X = []
        self.total_completed_Z = []
        super(DynamicCircuitBuilder, self).__init__(code)

    def syndrome_extraction_cz(self, coloring: dict) -> (stim.Circuit, list):
        """Modified CZ circuit where we have now failed gates

        Args:
            coloring (dict): edge coloring

        Returns:
            stim.Circuit: output circuit
            list: list of all the successful CZ gates
        """        
        circ = stim.Circuit()
        successful_edges = []
        for color in coloring.keys():

            # Draw at random gates that are not done at this timestep
            failed = np.random.choice([0, 1], size=len(coloring[color]), p=[
                                      1 - self.p_fail, self.p_fail])
            success_edges = np.where(failed == 0)[0]

            if len(success_edges) == 0:
                continue


            # Gates that are actually done
            success_CZ = np.array(coloring[color])[success_edges]
            
            cz_inds = [x for edge in success_CZ for x in edge]
            circ.append("CZ", cz_inds)
            successful_edges += list(success_CZ)
        return circ, successful_edges

    def syndrome_extraction_circuit(self, ):
        circ = stim.Circuit()

        # Initialize ancilla qubits
        circ.append("RX", self.ancilla_x + self.ancilla_z)

        # Do X QEC cycle CZ
        self.x_color = self.x_coloring()
        circ.append("H", self.data)
        circ_cz_x, success_edge_X = self.syndrome_extraction_cz(self.x_color)
        circ += circ_cz_x
        circ.append("H", self.data)

        check_X = self.dynamic_check_matrix(success_edge_X, "X")
        completed_x_checks = self.completed_checks(check_X, self.code.H_X)
        self.total_checks_X.append(check_X)
        self.total_completed_X.append(completed_x_checks)
        
        # Do Z QEC cycle CZ
        self.z_color = self.z_coloring()

        circ_cz_z, success_edge_Z = self.syndrome_extraction_cz(self.z_color)
        circ += circ_cz_z

        check_Z = self.dynamic_check_matrix(success_edge_Z, "Z")
        completed_z_checks = self.completed_checks(check_Z, self.code.H_Z)
        self.total_checks_Z.append(check_Z)
        self.total_completed_Z.append(completed_z_checks)
        
        # Measure qubits in the X basis if completed, in the Z basis otherwise

        circ.append(
            "H",
            [val for ind, val in enumerate(self.ancilla_x) if completed_x_checks[ind]] +
            [val for ind, val in enumerate(
                self.ancilla_z) if completed_z_checks[ind]]
        )

        circ.append("MZ", self.ancilla_x + self.ancilla_z)
        return circ

    def dynamic_check_matrix(self, edges: list, basis: str) -> np.array:
        """Create a matrix of all the successful CZ gates between data and check qubit.
        Resemble the parity check matrix if the gates are successful.

        Args:
            edges (list): List of successful edge
            basis (str): basis of check matrix

        Returns:
            np.array: output dynamic matrix
        """        
        if basis == "X":
            check = np.zeros(self.code.H_X.shape)
            offset = self.n
        elif basis == "Z":
            check = np.zeros(self.code.H_Z.shape)
            offset = self.n + self.c_x
        for pair in edges:
            i, j = pair
            check[i - offset, j] = 1
        return check

    def completed_checks(self, cz_check: np.array, parity_check: np.array) -> np.array:
        """Find checks that are successfully measured.

        Args:
            cz_check (np.array): dynamic CZ check matrix
            parity_check (np.array): actual corresponding parity check matrix  

        Returns:
            _type_: vector of all successful check measurements
        """        
        return np.all(cz_check == parity_check, axis=1)

    def noisy_rounds(self, n_rounds):

        circ = stim.Circuit()
        for i in range(n_rounds):
            circ += self.syndrome_extraction_circuit()
        return circ

    def memory_exp(self, n_rounds, basis):
        circ = stim.Circuit()
        circ += self.initialization(basis)

        # Reinitialize the check matrices
        self.total_checks_X = []
        self.total_checks_Z = []
        self.total_completed_X = []
        self.total_completed_Z = []

        circ += self.syndrome_extraction_circuit()

        circ += self.noisy_rounds(
            n_rounds)

        circ += self.syndrome_extraction_circuit()

        circ += self.add_detectors(n_rounds)

        circ += self.add_logical_operators(basis)
        return circ

    def add_detectors(self, n_rounds):
        n_checks = self.c_x + self.c_z
        n_tot_measurements = n_checks * (n_rounds + 2)
        circ = stim.Circuit()

        self.comm_X, self.comm_Z = self.commutation_matrices()

        
        # Noiseless measurement at the beginning
        previous_X = np.array(np.zeros((self.c_x,)), dtype=int)
        previous_Z = np.array(np.zeros((self.c_z,)), dtype=int)


        # Go through the rounds
        for round_current in range(1, n_rounds + 2):
            # Go through the check qubits:
            for ind_check in range(self.c_x):
                # If a measurement is completed:
                if self.total_completed_X[round_current][ind_check]:
                    # make a detector with previous_X[ind_check] round and this one round_current
                    rec_list = self.get_detector(previous_X[ind_check], round_current, ind_check, "X", n_rounds)
                    previous_X[ind_check] = round_current
                    circ.append("DETECTOR", rec_list)
            for ind_check in range(self.c_z):
                if self.total_completed_Z[round_current][ind_check]:
                    # make a detector with previous_X[ind_check] round and this one round_current
                    rec_list = self.get_detector(previous_Z[ind_check], round_current, ind_check, "Z", n_rounds)
                    previous_Z[ind_check] = round_current
                    circ.append("DETECTOR", rec_list)
        return circ

    def get_measurement_rec(self, i_round, j_check, basis, n_rounds):
        n_checks = self.c_x + self.c_z
        n_tot_measurements = n_checks * (n_rounds + 2)
        if basis == "X":
            k = j_check
        else:
            k = j_check + self.c_x
        ind_rec = - n_tot_measurements + i_round * n_checks + k
        return stim.target_rec(int(ind_rec))

    def commutation_matrices(self):
        self.comm_X = []
        for mat in self.total_checks_X:
            self.comm_X.append(self.commutation_matrix(self.code.H_Z, mat))
        self.comm_Z = []
        for mat in self.total_checks_Z:
            self.comm_Z.append(self.commutation_matrix(self.code.H_X, mat))
        return self.comm_X, self.comm_Z

    def commutation_matrix(self, H, check):
        return np.matmul(H, check.T) % 2

    def get_detector(self, round_prev, round_current, ind_check, basis, n_rounds):
        rec_list = []
        rec_list.append(self.get_measurement_rec(round_prev, ind_check, basis, n_rounds))
        rec_list.append(self.get_measurement_rec(round_current, ind_check, basis, n_rounds))
        if basis == "X":
            other_check_range = range(round_prev, round_current)
            comm_mat = self.comm_Z
            other_basis = "Z"
        else:
            other_check_range = range(round_prev + 1, round_current + 1)
            comm_mat = self.comm_X
            other_basis = "X"
        for round_other_checks in other_check_range:
            for ind_other_check in range(comm_mat[round_other_checks].shape[1]):
                if comm_mat[round_other_checks][ind_check, ind_other_check]:
                    rec_list.append(self.get_measurement_rec(round_other_checks, ind_other_check, other_basis, n_rounds))
        return rec_list

    pass

class AdaptiveCircuitBuilder(NoisyCircuitBuilder):
    """
    Dynamic version where gates are done or not depending on other's success or failure.
    For the moment, we just change ancilla measurement basis

    TODO: Add detectors. Make sure it is working.

    """    
    def __init__(self, code: CSS, p_fail: float):
        """p_fail is the gate failure probability.
        So far no errors are added to the system yet. Objective is just to recover the detectors.
        """        
        self.p_fail = p_fail

        self.total_checks_X = []
        self.total_checks_Z = []
        self.total_completed_X = []
        self.total_completed_Z = []
        super(DynamicCircuitBuilder, self).__init__(code)

    def syndrome_extraction_cz(self, coloring: dict) -> (stim.Circuit, list):
        """Modified CZ circuit where we have now failed gates

        Args:
            coloring (dict): edge coloring

        Returns:
            stim.Circuit: output circuit
            list: list of all the successful CZ gates
        """        
        circ = stim.Circuit()
        successful_edges = []
        for color in coloring.keys():

            # Draw at random gates that are not done at this timestep
            failed = np.random.choice([0, 1], size=len(coloring[color]), p=[
                                      1 - self.p_fail, self.p_fail])
            success_edges = np.where(failed == 0)[0]

            if len(success_edges) == 0:
                continue


            # Gates that are actually done
            success_CZ = np.array(coloring[color])[success_edges]
            
            cz_inds = [x for edge in success_CZ for x in edge]
            circ.append("CZ", cz_inds)
            successful_edges += list(success_CZ)
        return circ, successful_edges

    def syndrome_extraction_circuit(self, ):
        circ = stim.Circuit()

        # Initialize ancilla qubits
        circ.append("RX", self.ancilla_x + self.ancilla_z)

        # Do X QEC cycle CZ
        self.x_color = self.x_coloring()
        circ.append("H", self.data)
        circ_cz_x, success_edge_X = self.syndrome_extraction_cz(self.x_color)
        circ += circ_cz_x
        circ.append("H", self.data)

        check_X = self.dynamic_check_matrix(success_edge_X, "X")
        completed_x_checks = self.completed_checks(check_X, self.code.H_X)
        self.total_checks_X.append(check_X)
        self.total_completed_X.append(completed_x_checks)
        
        # Do Z QEC cycle CZ
        self.z_color = self.z_coloring()

        circ_cz_z, success_edge_Z = self.syndrome_extraction_cz(self.z_color)
        circ += circ_cz_z

        check_Z = self.dynamic_check_matrix(success_edge_Z, "Z")
        completed_z_checks = self.completed_checks(check_Z, self.code.H_Z)
        self.total_checks_Z.append(check_Z)
        self.total_completed_Z.append(completed_z_checks)
        
        # Measure qubits in the X basis if completed, in the Z basis otherwise

        circ.append(
            "H",
            [val for ind, val in enumerate(self.ancilla_x) if completed_x_checks[ind]] +
            [val for ind, val in enumerate(
                self.ancilla_z) if completed_z_checks[ind]]
        )

        circ.append("MZ", self.ancilla_x + self.ancilla_z)
        return circ

    def dynamic_check_matrix(self, edges: list, basis: str) -> np.array:
        """Create a matrix of all the successful CZ gates between data and check qubit.
        Resemble the parity check matrix if the gates are successful.

        Args:
            edges (list): List of successful edge
            basis (str): basis of check matrix

        Returns:
            np.array: output dynamic matrix
        """        
        if basis == "X":
            check = np.zeros(self.code.H_X.shape)
            offset = self.n
        elif basis == "Z":
            check = np.zeros(self.code.H_Z.shape)
            offset = self.n + self.c_x
        for pair in edges:
            i, j = pair
            check[i - offset, j] = 1
        return check

    def completed_checks(self, cz_check: np.array, parity_check: np.array) -> np.array:
        """Find checks that are successfully measured.

        Args:
            cz_check (np.array): dynamic CZ check matrix
            parity_check (np.array): actual corresponding parity check matrix  

        Returns:
            _type_: vector of all successful check measurements
        """        
        return np.all(cz_check == parity_check, axis=1)

    def noisy_rounds(self, n_rounds):

        circ = stim.Circuit()
        for i in range(n_rounds):
            circ += self.syndrome_extraction_circuit()
        return circ

    def memory_exp(self, n_rounds, basis):
        circ = stim.Circuit()
        circ += self.initialization(basis)

        # Reinitialize the check matrices
        self.total_checks_X = []
        self.total_checks_Z = []
        self.total_completed_X = []
        self.total_completed_Z = []

        circ += self.syndrome_extraction_circuit()

        circ += self.noisy_rounds(
            n_rounds)

        circ += self.syndrome_extraction_circuit()

        circ += self.add_detectors(n_rounds)

        circ += self.add_logical_operators(basis)
        return circ

    def add_detectors(self, n_rounds):
        n_checks = self.c_x + self.c_z
        n_tot_measurements = n_checks * (n_rounds + 2)
        circ = stim.Circuit()

        self.comm_X, self.comm_Z = self.commutation_matrices()

        
        # Noiseless measurement at the beginning
        previous_X = np.array(np.zeros((self.c_x,)), dtype=int)
        previous_Z = np.array(np.zeros((self.c_z,)), dtype=int)


        # Go through the rounds
        for round_current in range(1, n_rounds + 2):
            # Go through the check qubits:
            for ind_check in range(self.c_x):
                # If a measurement is completed:
                if self.total_completed_X[round_current][ind_check]:
                    # make a detector with previous_X[ind_check] round and this one round_current
                    rec_list = self.get_detector(previous_X[ind_check], round_current, ind_check, "X", n_rounds)
                    previous_X[ind_check] = round_current
                    circ.append("DETECTOR", rec_list)
            for ind_check in range(self.c_z):
                if self.total_completed_Z[round_current][ind_check]:
                    # make a detector with previous_X[ind_check] round and this one round_current
                    rec_list = self.get_detector(previous_Z[ind_check], round_current, ind_check, "Z", n_rounds)
                    previous_Z[ind_check] = round_current
                    circ.append("DETECTOR", rec_list)
        return circ

    def get_measurement_rec(self, i_round, j_check, basis, n_rounds):
        n_checks = self.c_x + self.c_z
        n_tot_measurements = n_checks * (n_rounds + 2)
        if basis == "X":
            k = j_check
        else:
            k = j_check + self.c_x
        ind_rec = - n_tot_measurements + i_round * n_checks + k
        return stim.target_rec(int(ind_rec))

    def commutation_matrices(self):
        self.comm_X = []
        for mat in self.total_checks_X:
            self.comm_X.append(self.commutation_matrix(self.code.H_Z, mat))
        self.comm_Z = []
        for mat in self.total_checks_Z:
            self.comm_Z.append(self.commutation_matrix(self.code.H_X, mat))
        return self.comm_X, self.comm_Z

    def commutation_matrix(self, H, check):
        return np.matmul(H, check.T) % 2

    def get_detector(self, round_prev, round_current, ind_check, basis, n_rounds):
        rec_list = []
        rec_list.append(self.get_measurement_rec(round_prev, ind_check, basis, n_rounds))
        rec_list.append(self.get_measurement_rec(round_current, ind_check, basis, n_rounds))
        if basis == "X":
            other_check_range = range(round_prev, round_current)
            comm_mat = self.comm_Z
            other_basis = "Z"
        else:
            other_check_range = range(round_prev + 1, round_current + 1)
            comm_mat = self.comm_X
            other_basis = "X"
        for round_other_checks in other_check_range:
            for ind_other_check in range(comm_mat[round_other_checks].shape[1]):
                if comm_mat[round_other_checks][ind_check, ind_other_check]:
                    rec_list.append(self.get_measurement_rec(round_other_checks, ind_other_check, other_basis, n_rounds))
        return rec_list
# class DynamicCircuitBuilder(CircuitBuilder):
#     """
#     Dynamic version where gates are done or not depending on other's success or failure.
#     For the moment, we just change ancilla measurement basis

#     TODO: Add detectors. Make sure it is working.

#     """    
#     def __init__(self, code: CSS, p_fail: float):
#         """p_fail is the gate failure probability.
#         So far no errors are added to the system yet. Objective is just to recover the detectors.
#         """        
#         self.p_fail = p_fail

#         self.total_checks_X = []
#         self.total_checks_Z = []
#         self.total_completed_X = []
#         self.total_completed_Z = []
#         super(DynamicCircuitBuilder, self).__init__(code)

#     def syndrome_extraction_cz(self, coloring: dict) -> (stim.Circuit, list):
#         """Modified CZ circuit where we have now failed gates

#         Args:
#             coloring (dict): edge coloring

#         Returns:
#             stim.Circuit: output circuit
#             list: list of all the successful CZ gates
#         """        
#         circ = stim.Circuit()
#         successful_edges = []
#         for color in coloring.keys():

#             # Draw at random gates that are not done at this timestep
#             failed = np.random.choice([0, 1], size=len(coloring[color]), p=[
#                                       1 - self.p_fail, self.p_fail])
#             success_edges = np.where(failed == 0)[0]

#             if len(success_edges) == 0:
#                 continue


#             # Gates that are actually done
#             success_CZ = np.array(coloring[color])[success_edges]
            
#             cz_inds = [x for edge in success_CZ for x in edge]
#             circ.append("CZ", cz_inds)
#             successful_edges += list(success_CZ)
#         return circ, successful_edges

#     def syndrome_extraction_circuit(self, ):
#         circ = stim.Circuit()

#         # Initialize ancilla qubits
#         circ.append("RX", self.ancilla_x + self.ancilla_z)

#         # Do X QEC cycle CZ
#         self.x_color = self.x_coloring()
#         circ.append("H", self.data)
#         circ_cz_x, success_edge_X = self.syndrome_extraction_cz(self.x_color)
#         circ += circ_cz_x
#         circ.append("H", self.data)

#         check_X = self.dynamic_check_matrix(success_edge_X, "X")
#         completed_x_checks = self.completed_checks(check_X, self.code.H_X)
#         self.total_checks_X.append(check_X)
#         self.total_completed_X.append(completed_x_checks)
        
#         # Do Z QEC cycle CZ
#         self.z_color = self.z_coloring()

#         circ_cz_z, success_edge_Z = self.syndrome_extraction_cz(self.z_color)
#         circ += circ_cz_z

#         check_Z = self.dynamic_check_matrix(success_edge_Z, "Z")
#         completed_z_checks = self.completed_checks(check_Z, self.code.H_Z)
#         self.total_checks_Z.append(check_Z)
#         self.total_completed_Z.append(completed_z_checks)
        
#         # Measure qubits in the X basis if completed, in the Z basis otherwise

#         circ.append(
#             "H",
#             [val for ind, val in enumerate(self.ancilla_x) if completed_x_checks[ind]] +
#             [val for ind, val in enumerate(
#                 self.ancilla_z) if completed_z_checks[ind]]
#         )

#         circ.append("MZ", self.ancilla_x + self.ancilla_z)
#         return circ

#     def dynamic_check_matrix(self, edges: list, basis: str) -> np.array:
#         """Create a matrix of all the successful CZ gates between data and check qubit.
#         Resemble the parity check matrix if the gates are successful.

#         Args:
#             edges (list): List of successful edge
#             basis (str): basis of check matrix

#         Returns:
#             np.array: output dynamic matrix
#         """        
#         if basis == "X":
#             check = np.zeros(self.code.H_X.shape)
#             offset = self.n
#         elif basis == "Z":
#             check = np.zeros(self.code.H_Z.shape)
#             offset = self.n + self.c_x
#         for pair in edges:
#             i, j = pair
#             check[i - offset, j] = 1
#         return check

#     def completed_checks(self, cz_check: np.array, parity_check: np.array) -> np.array:
#         """Find checks that are successfully measured.

#         Args:
#             cz_check (np.array): dynamic CZ check matrix
#             parity_check (np.array): actual corresponding parity check matrix  

#         Returns:
#             _type_: vector of all successful check measurements
#         """        
#         return np.all(cz_check == parity_check, axis=1)

#     def noisy_rounds(self, n_rounds):

#         circ = stim.Circuit()
#         for i in range(n_rounds):
#             circ += self.syndrome_extraction_circuit()
#         return circ

#     def memory_exp(self, n_rounds, basis):
#         circ = stim.Circuit()
#         circ += self.initialization(basis)

#         # Reinitialize the check matrices
#         self.total_checks_X = []
#         self.total_checks_Z = []
#         self.total_completed_X = []
#         self.total_completed_Z = []

#         circ += self.syndrome_extraction_circuit()

#         circ += self.noisy_rounds(
#             n_rounds)

#         circ += self.syndrome_extraction_circuit()

#         circ += self.add_detectors(n_rounds)

#         circ += self.add_logical_operators(basis)
#         return circ

#     def add_detectors(self, n_rounds):
#         n_checks = self.c_x + self.c_z
#         n_tot_measurements = n_checks * (n_rounds + 2)
#         circ = stim.Circuit()

#         self.comm_X, self.comm_Z = self.commutation_matrices()

        
#         # Noiseless measurement at the beginning
#         previous_X = np.array(np.zeros((self.c_x,)), dtype=int)
#         previous_Z = np.array(np.zeros((self.c_z,)), dtype=int)


#         # Go through the rounds
#         for round_current in range(1, n_rounds + 2):
#             # Go through the check qubits:
#             for ind_check in range(self.c_x):
#                 # If a measurement is completed:
#                 if self.total_completed_X[round_current][ind_check]:
#                     # make a detector with previous_X[ind_check] round and this one round_current
#                     rec_list = self.get_detector(previous_X[ind_check], round_current, ind_check, "X", n_rounds)
#                     previous_X[ind_check] = round_current
#                     circ.append("DETECTOR", rec_list)
#             for ind_check in range(self.c_z):
#                 if self.total_completed_Z[round_current][ind_check]:
#                     # make a detector with previous_X[ind_check] round and this one round_current
#                     rec_list = self.get_detector(previous_Z[ind_check], round_current, ind_check, "Z", n_rounds)
#                     previous_Z[ind_check] = round_current
#                     circ.append("DETECTOR", rec_list)
#         return circ

#     def get_measurement_rec(self, i_round, j_check, basis, n_rounds):
#         n_checks = self.c_x + self.c_z
#         n_tot_measurements = n_checks * (n_rounds + 2)
#         if basis == "X":
#             k = j_check
#         else:
#             k = j_check + self.c_x
#         ind_rec = - n_tot_measurements + i_round * n_checks + k
#         return stim.target_rec(int(ind_rec))

#     def commutation_matrices(self):
#         self.comm_X = []
#         for mat in self.total_checks_X:
#             self.comm_X.append(self.commutation_matrix(self.code.H_Z, mat))
#         self.comm_Z = []
#         for mat in self.total_checks_Z:
#             self.comm_Z.append(self.commutation_matrix(self.code.H_X, mat))
#         return self.comm_X, self.comm_Z

#     def commutation_matrix(self, H, check):
#         return np.matmul(H, check.T) % 2

#     def get_detector(self, round_prev, round_current, ind_check, basis, n_rounds):
#         rec_list = []
#         rec_list.append(self.get_measurement_rec(round_prev, ind_check, basis, n_rounds))
#         rec_list.append(self.get_measurement_rec(round_current, ind_check, basis, n_rounds))
#         if basis == "X":
#             other_check_range = range(round_prev, round_current)
#             comm_mat = self.comm_Z
#             other_basis = "Z"
#         else:
#             other_check_range = range(round_prev + 1, round_current + 1)
#             comm_mat = self.comm_X
#             other_basis = "X"
#         for round_other_checks in other_check_range:
#             for ind_other_check in range(comm_mat[round_other_checks].shape[1]):
#                 if comm_mat[round_other_checks][ind_check, ind_other_check]:
#                     rec_list.append(self.get_measurement_rec(round_other_checks, ind_other_check, other_basis, n_rounds))
#         return rec_list

#     pass
