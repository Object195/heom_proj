import numpy as np
import scipy.sparse as sp
'''
quantum operations on ADO
'''
def state_to_real(v):
    #convert vectorized dm from complex to real representation
    v = np.asarray(v)
    return np.concatenate([v.real, v.imag], axis=0)
def sup_op_to_real(L):
    #convert superoperators from complex to real representation
    #while preserving the multiplication rule
    # L is expected to be a scipy.sparse matrix
    return sp.bmat([[L.real, -L.imag],
                    [L.imag, L.real]])
def left_sup_op(op):
    #construct the superoperator that corresponds to A*\rho
    op = np.asarray(op)
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError("op must be a square matrix")
    return sp.kron(sp.identity(op.shape[0], dtype=op.dtype), op)
def right_sup_op(op):
    #construct the superoperator that corresponds to \rho*A
    op = np.asarray(op)
    if op.ndim != 2 or op.shape[0] != op.shape[1]:
        raise ValueError("op must be a square matrix")
    return sp.kron(op.transpose(), sp.identity(op.shape[0], dtype=op.dtype))
def commutator_sup_op(op):
    return left_sup_op(op) - right_sup_op(op)
