''' Operators w.r.t. the model '''
import numpy as np
import cv2
from matplotlib.ticker import IndexLocator
from matplotlib import patches


def Vec(M):
    return M.reshape(-1, 1)


def Vec_inv(M, shape):
    return M.reshape(shape)


def Vectorize(T):
    n, p, q = T.shape
    return T.reshape((n, p*q))


''' R operator for matrix '''


def R_opt(M, idctshape):
    m, n = M.shape
    p1, p2 = idctshape
    assert m % p1 == 0 and n % p2 == 0, "Dimension wrong"
    d1, d2 = m // p1, n // p2
    RM = []
    for i in range(p1):
        for j in range(p2):
            Mij = M[d1*i: d1*(i + 1), d2*j: d2*(j + 1)]
            RM.append(Vec(Mij))
    return np.concatenate(RM, axis=1).T


def R_opt_pro(A, idctshape):
    m, n = A.shape
    p1, p2 = idctshape
    assert m % p1 == 0 and n % p2 == 0, "Dimension wrong"
    d1, d2 = m // p1, n // p2
    strides = A.itemsize * np.array([p2*d2*d1, d2, p2*d2, 1])
    A_blocked = np.lib.stride_tricks.as_strided(A, shape=(p1, p2, d1, d2), strides=strides)
    RA = A_blocked.reshape(-1, d1*d2)
    return A_blocked, RA


def R_inv(RM, blockshape, idctshape):
    m, n = RM.shape
    d1, d2 = blockshape
    p1, p2 = idctshape
    assert m == p1 * p2 and n == d1 * d2, "Dimension wrong"
    M = np.zeros([d1 * p1, d2 * p2])
    for i in range(m):
        Block = Vec_inv(RM[i, :], blockshape)
        ith = i // p2  # quotient
        jth = i % p2  # remainder
        M[d1*ith: d1*(ith+1), d2*jth: d2*(jth+1)] = Block

    return M


def soft_threshold(x, th):
    return np.sign(x) * np.maximum(np.abs(x)-th, 0)


''' Resize the sample '''


def Resize(data, shape):
    new = []
    shape = list(shape)
    shape.reverse()
    for pic in data:
        pic_new = cv2.resize(pic, shape)
        new.append(pic_new)
    return np.array(new)


def visualize(theax, array, coordinates=[], num_yx=(8, 8), grid=False, title=''):
    shape = array.shape
    theax.imshow(array)
    theax.xaxis.set_major_locator(IndexLocator(offset=-0.5, base=4))
    theax.xaxis.set_minor_locator(IndexLocator(offset=0, base=4))
    theax.yaxis.set_major_locator(IndexLocator(offset=-0.5, base=4))
    theax.yaxis.set_minor_locator(IndexLocator(offset=0, base=4))
    xticks = np.arange(shape[1], step=int(shape[1] / num_yx[1]))
    yticks = np.arange(shape[0], step=int(shape[0] / num_yx[0]))
    theax.set_xticks(xticks, xticks + 1)
    theax.set_yticks(yticks, yticks + 1)
    if grid:
        theax.grid(which='minor', axis='both', linewidth=0.75, linestyle='-', color='orange', zorder=2)
    if len(coordinates) > 0:
        for coor in coordinates:
            coordinate = [x * 4 - 0.5 for x in coor]
            coordinate.reverse()
            theax.add_patch(patches.Rectangle(coordinate, 4, 4, fill=False, color='red', zorder=2))
    theax.set_title(title, fontweight='bold')


