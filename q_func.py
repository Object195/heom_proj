import torch
'''
quantum operations on torch tensor
'''
def state_to_real(v):
    #convert vectorized dm from complex to real representation
    return torch.cat([v.real,v.imag],dim=-1)
def sup_op_to_real(L):
    #convert superoperators from complex to real representation
    #while preserving the multiplication rule
    return torch.cat([torch.cat([L.real, -L.imag], dim=1),
                      torch.cat([L.imag, L.real], dim=1)],
                      dim=0)
def left_sup_op(op):
    #construct the superoperator that corresponds to A*\rho
    return sup_op_to_real(torch.kron(torch.eye(op.size(0)).to(op),op))
def right_sup_op(op):
    #construct the superoperator that corresponds to \rho*A
    return sup_op_to_real(torch.kron(op.t(), torch.eye(op.size(0)).to(op)))