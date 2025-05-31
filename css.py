import numpy as np
import qldpc

class CSS:
    """
    Represents a CSS quantum error-correcting code using check matrices (H_X, H_Z)
    and logical operators (L_X, L_Z), based on the qldpc package.

    """
    def __init__(self, H_Z, H_X, L_Z, L_X):
        self.H_X = H_X
        self.H_Z = H_Z
        self.L_X = L_X
        self.L_Z = L_Z
        self.k, self.n = self.L_X.shape
        self.c_x = self.H_X.shape[0]
        self.c_z = self.H_Z.shape[0]

    @classmethod
    def from_qldpc(cls, code):
        """
        Create a new instance out of the qldpc package.
        """
        H_Z = np.asarray(code.matrix_z)
        H_X = np.asarray(code.matrix_x)
        L_X = np.asarray(code.get_logical_ops(qldpc.objects.Pauli.X))
        L_Z = np.asarray(code.get_logical_ops(qldpc.objects.Pauli.Z))
        return cls(H_Z, H_X, L_Z, L_X)
    pass

    def __repr__(self):
        return f"CSS code: [[{self.n}, {self.k}, d]]"

class SurfaceCode(CSS):
    def __init__(self, d):
        code = qldpc.codes.SurfaceCode(d)
        H_Z = np.asarray(code.matrix_z)
        H_X = np.asarray(code.matrix_x)
        L_X = np.asarray(code.get_logical_ops(qldpc.objects.Pauli.X))
        L_Z = np.asarray(code.get_logical_ops(qldpc.objects.Pauli.Z))
        super(SurfaceCode, self).__init__(H_Z, H_X, L_Z, L_X)

class ToricCode(CSS):
    def __init__(self, d):
        rep_code = qldpc.codes.RingCode(d)
        code = qldpc.codes.HGPCode(rep_code, rep_code)
        H_Z = np.asarray(code.matrix_z)
        H_X = np.asarray(code.matrix_x)
        L_X = np.asarray(code.get_logical_ops(qldpc.objects.Pauli.X))
        L_Z = np.asarray(code.get_logical_ops(qldpc.objects.Pauli.Z))
        super(ToricCode, self).__init__(H_Z, H_X, L_Z, L_X)

