import numpy as np
from numpy.linalg import LinAlgError

from scipy.sparse.linalg import lsqr
import pymatching
from ldpc.bplsd_decoder import BpLsdDecoder
from beliefmatching import detector_error_model_to_check_matrices
import galois

import stim

GF = galois.GF(2)



class Decoder():

    def get_erasure_locations(self, n: int, p_erasure: float=0) -> np.array:
        """Sample erasures for n qubits with probability p_erasure.

        Args:
            n (int): number of elements
            p_erasure (float, optional): erasure probability. Defaults to 0.

        Returns:
            np.array: Sampled erasure locations
        """        
        erased = np.random.rand(n)
        erasure_location = np.where(erased <p_erasure)[0]
        return erasure_location
    
    def get_error_channel(self, n:int, p_erasure:float=0, p_comp:float=0, p_flip: float=1/2) -> tuple:
        """Get the error channel (error probability + erasures).
        This is the information about errors that a decoder could access

        Args:
            n (int): number of qubits
            p_erasure (float, optional): Probability of erasure. Defaults to 0.
            p_comp (float, optional): computational error (bitflip) probability. Defaults to 0.
            p_flip (float, optional): Probability of bitflip for an erasure. Defaults to 1/2.

        Returns:
            tuple: error channel + erasure locations
        """        
        error_channel = np.ones((n,)) * p_comp 
        erasure_location = self.get_erasure_locations(n, p_erasure)
        erasure_channel = np.zeros((n,))
        erasure_channel[erasure_location] = np.ones((len(erasure_location),)) * p_flip
        error_channel = self.add_flip_error_probabilities(error_channel, erasure_channel)
        return error_channel, erasure_location

    def sample_errors(self, error_channel: np.array) -> np.array:
        """Sample errors from the error channel

        Args:
            error_channel (np.array): error channel (bitflip probability per qubit)

        Returns:
            np.array: Sampled error bitstring.
        """        
        rand_vals = np.random.rand(len(error_channel))
        error = (rand_vals < error_channel)
        return error.astype(bool)

    def add_flip_error_probabilities(self, error_channel_1: np.array, error_channel_2: np.array) -> np.array:
        """Add two error channels

        Args:
            error_channel_1 (np.array): first error channel
            error_channel_2 (np.array): second error channel

        Returns:
            np.array: output error channel
        """        
        assert error_channel_1.shape == error_channel_2.shape
        return error_channel_1 * (1 - error_channel_2) + error_channel_2 * (1 - error_channel_1)

    def decode(self, H: np.array, L: np.array, error: np.array, error_channel: np.array, erasure_location: np.array=[]) -> np.array:
        """Decode a sample and output whether or not there is a logical error per logical qubit.

        Args:
            H (np.array): Parity check matrix
            L (np.array): Logical observable matrix
            error (np.array): error bitstring
            error_channel (np.array): error probabilities
            erasure_location (np.array, optional): location of errors. Defaults to [].

        Returns:
            np.array: _description_
        """        
        syndrome = (H @ error) % 2
        correction = self.decoder(H, syndrome, error_channel, erasure_location)
        return (L @ (error + correction)) % 2

    def decoder(self):
        raise ValueError("Decoder().decoder should be overwritten")
        pass

    def do_simulations(self, H: np.array, L: np.array, n_shots:int, p_erasure: float=0, p_comp: float=0, p_flip: float = 1/2, per_qubit=False):
        c, n = H.shape
        k, n = L.shape

        if per_qubit:
            ler = np.zeros((k,))
        else:
            ler = 0
        for _ in range(n_shots):
            error_channel, erasure_location = self.get_error_channel(n, p_erasure, p_comp, p_flip)
            error = self.sample_errors(error_channel)
            logical_errors = self.decode(H, L, error, error_channel, erasure_location)
            if per_qubit:
                ler += logical_errors * 1
            else:
                ler += np.any(logical_errors)
        return ler / n_shots


    def simulations_with_stim(self, circuit: stim.Circuit):
        dem = circuit.detector_error_model()
        checks = detector_error_model_to_check_matrices(dem)
        H = checks.check_matrix.toarray()
        L = checks.observables_matrix.toarray()
        probs = checks.priors
        # print(H)
        # print(f"L={L}")
        # print(probs)
        
        syndrome, logicals = circuit.compile_detector_sampler().sample(1, separate_observables=True)
        # print(syndrome)
        logicals = logicals[0]
        correction = self.decoder(H, syndrome[0], probs, erasure_location=[])
        # print(f"correction = {correction}")
        # print(f"Le = {L @ correction}")
        # print(f"logicals = {logicals}")
        # print(f"res = {(logicals + L @ correction) % 2}")
        return (logicals + L @ correction) % 2


class GaussianEliminationDecoder(Decoder):

    @staticmethod
    # def gf2_gaussian_elimination(A, b):
    #     """
    #     Solve A x = b over GF(2) using Gaussian elimination with the galois package.

    #     Parameters:
    #         A (ndarray): Binary matrix (m x n)
    #         b (ndarray): Binary vector (length m)

    #     Returns:
    #         x (ndarray): Binary solution vector of length n

    #     Raises:
    #         ValueError: If no solution exists
    #     """
    #     A = GF(A)
    #     b = GF(b).reshape(-1, 1)
    #     m, n = A.shape

    #     # Augmented matrix
    #     Ab = np.hstack((A, b))

    #     row = 0
    #     for col in range(n):
    #         # Find pivot
    #         pivot_rows = np.where(Ab[row:, col] == 1)[0]
    #         if len(pivot_rows) == 0:
    #             continue
    #         pivot_row = pivot_rows[0] + row

    #         # Swap current row with pivot row
    #         if pivot_row != row:
    #             Ab[[row, pivot_row], :] = Ab[[pivot_row, row], :]

    #         # Eliminate other rows
    #         for r in range(m):
    #             if r != row and Ab[r, col] == 1:
    #                 Ab[r, :] += Ab[row, :]  # XOR over GF(2)
    #         row += 1

    #     # Check for inconsistency
    #     for r in range(row, m):
    #         if Ab[r, -1] == 1 and np.all(Ab[r, :-1] == 0):
    #             raise ValueError("No solution exists")

    #     # Back-substitution (already in RREF)
    #     x = GF.Zeros(n)
    #     for r in range(row):
    #         pivot_col = np.where(Ab[r, :-1] == 1)[0]
    #         if len(pivot_col) > 0:
    #             x[pivot_col[0]] = Ab[r, -1]

    #     return np.array(x, dtype=int)

    def gf2_gaussian_elimination(A, b):
        """
        Solve A x = b over GF(2) using Gaussian elimination.
        A: binary matrix (shape m x n)
        b: binary vector (length m)
        Returns: binary solution vector x (length n)
        Raises: ValueError if no solution exists
        """
        A = A.copy() % 2
        b = b.copy() % 2
        m, n = A.shape

        # Augmented matrix
        Ab = np.concatenate((A, b.reshape(-1, 1)), axis=1)

        row = 0
        for col in range(n):
            # Find pivot
            pivot_row = None
            for r in range(row, m):
                if Ab[r, col] == 1:
                    pivot_row = r
                    break
            if pivot_row is None:
                continue  # move to next column (free variable)
            # Swap rows
            Ab[[row, pivot_row]] = Ab[[pivot_row, row]]
            # Eliminate below and above
            for r in range(m):
                if r != row and Ab[r, col] == 1:
                    Ab[r] ^= Ab[row]
            row += 1

        # Check for inconsistency
        for r in range(row, m):
            if Ab[r, -1] == 1:
                raise ValueError("No solution exists")

        # Back-substitution (trivial since matrix is already in RREF)
        x = np.zeros(n, dtype=int)
        for r in range(row):
            pivot_col = np.argmax(Ab[r, :-1])
            x[pivot_col] = Ab[r, -1]

        return x



    def decoder(self, H, syndrome, error_channel, erasure_location):
        """
        Decode erasures using GF(2) Gaussian elimination.

        Parameters:
            H (ndarray): Parity-check matrix (m x n)
            syndrome (ndarray): Syndrome vector (length m)
            erasure_location (list[int]): List of indices of erased bits

        Returns:
            error_estimate (ndarray): Binary vector of length n (errors in erasures)
        """
        m, n = H.shape
        e = np.zeros(n, dtype=int)

        # Submatrix of H for erased bits
        H_e = H[:, erasure_location]

        try:
            e_e = GaussianEliminationDecoder.gf2_gaussian_elimination(H_e, syndrome)
            e[erasure_location] = e_e
            return e
        except ValueError:
            print("Decoding failed: no solution.")
            return None




class PymatchingDecoder(Decoder):
    cap_errors = 10 ** (-20)
    def decoder(self, H, syndrome, error_channel, erasure_location):
        error_channel_new = error_channel[:]
        error_channel_new[np.where(error_channel_new == 0)[0]] = np.ones((len(np.where(error_channel_new == 0)[0]))) * self.cap_errors
        
        weights = np.log2((1 - error_channel_new) / error_channel_new)
        matching = pymatching.Matching.from_check_matrix(H, weights=weights)
        return matching.decode(syndrome)


class BPDecoder(Decoder):
    pass

class BPLSDDecoder(BPDecoder):
    param = {
        'bp_method': 'product_sum',
        'max_iter': 10,
        'schedule': 'serial',
        'lsd_method': 'lsd_cs',
        'lsd_order': 0
    }
    def decoder(self, H, syndrome, error_channel, erasure_location, **args):
        if args == {}:
            args = self.param

        bp_lsd = BpLsdDecoder(
            H,
            error_channel = error_channel,
            **args
        )

        return bp_lsd.decode(syndrome)


