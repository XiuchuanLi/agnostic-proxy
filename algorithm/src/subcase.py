import numpy as np
from src.utils import candidates, ind_constraint, cum31, cum12


def TestCase(T, O, Z):
    if ind_constraint(T, O, Z)[0]:
        return 'a'
    
    root1, root2 = candidates(T, O)
    flag1, _ = ind_constraint(Z, T, O - root1 * T)
    flag2, _ = ind_constraint(Z, T, O - root2 * T)
    if flag1 ^ flag2:
        return 'b'
    
    root1, root2 = candidates(T, O)
    flag1, _ = ind_constraint(O - root1 * T, Z, T)
    flag2, _ = ind_constraint(O - root2 * T, Z, T)
    if flag1 or flag2:
        return 'c'
    
    root1, root2 = candidates(O, Z)
    flag1, _ = ind_constraint(T, O, Z - root1 * O)
    flag2, _ = ind_constraint(T, O, Z - root2 * O)
    if flag1 or flag2:
        return 'd'
    
    root_t1, root_t2 = candidates(Z, T)
    root_o1, root_o2 = candidates(Z, O)
    flag1, _ = ind_constraint(T - root_t1 * Z, O - root_o1 * Z, Z)
    flag2, _ = ind_constraint(T - root_t1 * Z, O - root_o2 * Z, Z)
    flag3, _ = ind_constraint(T - root_t2 * Z, O - root_o1 * Z, Z)
    flag4, _ = ind_constraint(T - root_t2 * Z, O - root_o2 * Z, Z)
    if flag1 or flag2 or flag3 or flag4:
        return 'e'

    return 'f,g,h'


def CalCase(T, O, Z, case, robust=True):
    
    def perfect(T, O, Z, robust):
        ratio = np.sign(np.cov(T,Z)[0,1]) * np.sqrt(np.abs(cum31(T, Z) / cum31(Z, T)))
        if robust:
            if np.abs(np.corrcoef(T, Z**2)[0,1]) > 0.1:
                ratio = cum12(Z,T) / cum12(T,Z)
        result = (np.cov(T,O)[0,1] - ratio * np.cov(O,Z)[0,1]) / (np.var(T) - ratio * np.cov(T,Z)[0,1])
        return result

    if case == 'a':
        return perfect(T, O, Z, robust)
    if case == 'b':
        root1, root2 = candidates(T, O)
        _, p1 = ind_constraint(Z, T, O - root1 * T)
        _, p2 = ind_constraint(Z, T, O - root2 * T)
        if p1 > p2:
            return root1
        else:
            return root2
    if case == 'c':
        root1, root2 = candidates(T, O)
        _, p1 = ind_constraint(O - root1 * T, Z, T)
        _, p2 = ind_constraint(O - root2 * T, Z, T)
        if p1 > p2:
            return root1
        else:
            return root2
    if case == 'd':
        root1, root2 = candidates(O, Z)
        _, p1 = ind_constraint(T, O, Z - root1 * O)
        _, p2 = ind_constraint(T, O, Z - root2 * O)
        if p1 > p2:
            return perfect(T, O, Z - root1 * O, robust)
        else:
            return perfect(T, O, Z - root2 * O, robust)
    if case == 'e':
        root_t1, root_t2 = candidates(Z, T)
        root_o1, root_o2 = candidates(Z, O)
        _, p1 = ind_constraint(T - root_t1 * Z, O - root_o1 * Z, Z)
        _, p2 = ind_constraint(T - root_t1 * Z, O - root_o2 * Z, Z)
        _, p3 = ind_constraint(T - root_t2 * Z, O - root_o1 * Z, Z)
        _, p4 = ind_constraint(T - root_t2 * Z, O - root_o2 * Z, Z)
        if p1 > max([p2, p3, p4]):
            return perfect(T - root_t1 * Z, O - root_o1 * Z, Z, robust)
        elif p2 > max([p1, p3, p4]):
            return perfect(T - root_t1 * Z, O - root_o2 * Z, Z, robust)
        elif p3 > max([p1, p2, p4]):
            return perfect(T - root_t2 * Z, O - root_o1 * Z, Z, robust)
        else:
            return perfect(T - root_t2 * Z, O - root_o2 * Z, Z, robust)
    else:
        raise ValueError
