import copy
import numpy.linalg as la
from utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale


def circle(img, x0, y0, r):
    temp = copy.deepcopy(img)
    m, n = temp.shape
    for i in range(m):
        for j in range(n):
            dist = np.round(la.norm(np.array([i - x0, j - y0]), 2))
            if dist <= r:
                temp[i, j] = 1
    return temp


def Gnrt_circle(img_size, center, radius):
    canvas = np.zeros(img_size)
    x, y = center
    C = circle(canvas, x, y, radius)
    return C


# Circles
def Gnrt_circles(img_size, center_list, radius_list):
    canvas = np.zeros(img_size)
    C = canvas.copy()
    for i in range(len(radius_list)):
        x, y = center_list[i]
        r = radius_list[i]
        C = circle(C, x, y, r)
    return C


def random_location(shape, num, random_state=0):
    rng = np.random.RandomState(random_state)
    total_num = np.prod(shape)
    map = np.arange(total_num).reshape(shape)
    roi_coor = rng.choice(np.arange(total_num), size=num, replace=False)
    roi_coor = np.array([np.where(map == ind) for ind in roi_coor]).squeeze(axis=-1)
    return roi_coor


def Correlated_array(shape=(4, 4), corr=0.9, random_state=0):
    rng = np.random.RandomState(random_state)
    array_1 = rng.normal(0, 1, size=shape)
    array_2 = corr * array_1 + np.sqrt(1 - corr ** 2) * rng.normal(0, 1, size=shape)
    return array_1, array_2


def Generate_W(shape, method='low-rank', symmetric=True, normalize=True, random_state=0):
    rng = np.random.RandomState(random_state)
    if method == 'low-rank':
        R = 2
        U = rng.normal(size=(shape[0], R))
        Sig = np.diag([2, 1])
        V = rng.normal(size=(shape[1], R))
        W = U.dot(Sig).dot(U.T) if symmetric else U.dot(Sig).dot(V.T)
    elif method == 'identity':
        W = np.eye(shape[0])
    elif method == 'gradient':
        W = np.zeros(shape)
        for i in range(shape[0]):
            for j in range(shape[1]):
                W[i, j] = 1 / (np.abs(i - j) + 1)
    else:
        W = rng.normal(shape)
    W = (W + W.T) / 2 if symmetric else W

    if normalize:
        W = W / np.linalg.norm(W)

    return W


def Generate_byW(W, sample_size=1000, shape=(40, 40), blockshape=(4, 4), ROI_NUM=1, random_state=0):
    rng = np.random.RandomState(random_state)
    n = sample_size
    data = np.zeros(shape=(n, *shape))
    idctshape = tuple((np.array(shape) / np.array(blockshape)).astype(int))
    ROI_IND = []
    U, Sig, Vt = np.linalg.svd(W)
    vl, vr = U[:, :1].reshape(blockshape), Vt[:1, :].reshape(blockshape)
    # scale = np.sqrt(2 * np.prod(blockshape)) / 1
    scale = np.sqrt(2 * np.prod(blockshape)) / 2
    sigma_ns = 0.5
    for i in range(n):
        x = rng.normal(0, 1, size=shape)
        x_blocked, Rx = R_opt_pro(x, idctshape)
        roi_num = ROI_NUM if type(ROI_NUM) == int else rng.randint(2, 5)
        roi_coor = random_location(idctshape, roi_num, random_state=i)
        for ind in roi_coor:
            arr = vl * scale + rng.normal(scale=sigma_ns, size=blockshape)
            x_blocked[ind[0], ind[1]] = arr
        x = x_blocked.reshape(Rx.shape)
        x = R_inv(x, blockshape, idctshape)
        data[i] = x

        ROI_IND.append(roi_coor)

    return data, ROI_IND


def Generate_byCorr(sample_size=1000, shape=(40, 40), blockshape=(4, 4), ROI_NUM=1, random_state=0):
    rng = np.random.RandomState(random_state)
    n = sample_size
    data = np.zeros(shape=(n, *shape))
    idctshape = tuple((np.array(shape) / np.array(blockshape)).astype(int))
    ROI_IND = []
    sigma_ns = 0.1
    for i in range(n):
        x = rng.normal(0, 1, size=shape)
        x_blocked, Rx = R_opt_pro(x, idctshape)
        roi_num = ROI_NUM if type(ROI_NUM) == int else rng.randint(2, 5)
        roi_coor = random_location(idctshape, roi_num, random_state=i)
        base = rng.normal(size=blockshape)
        for ind in roi_coor:
            arr = base + rng.normal(scale=sigma_ns, size=blockshape)
            x_blocked[ind[0], ind[1]] = arr
        x = x_blocked.reshape(Rx.shape)
        x = R_inv(x, blockshape, idctshape)
        data[i] = x

        ROI_IND.append(roi_coor)

    return data, ROI_IND


def Generate_XnY(method='ByW', N=1000, shape=(28, 28), blockshape=(4, 4), DI_EXP=1.0, noise_level=1, random_state=0):
    rng = np.random.RandomState(random_state)

    C = Gnrt_circle(shape, (16, 12), 10)
    D = Gnrt_circle(shape, (14, 14), 5)
    idctshape = tuple((np.array(shape) / np.array(blockshape)).astype(int))
    _, RC = R_opt_pro(C, idctshape)
    _, RD = R_opt_pro(D, idctshape)

    params = {'sample_size': N, 'shape': shape, 'blockshape': blockshape, 'ROI_NUM': 2, 'random_state': random_state}

    if method == 'ByW':
        W = Generate_W([np.prod(blockshape)] * 2, method='low-rank', symmetric=False, random_state=random_state)
        X, ROI_COOR = Generate_byW(W=W, **params)
        SA_fun = lambda RXi: RXi.dot(W).dot(RXi.T)
    else:  # method == 'ByCorr'
        W = None
        X, ROI_COOR = Generate_byCorr(**params)
        SA_fun = np.corrcoef

    RX = np.array([R_opt_pro(Xi, idctshape)[1] for Xi in X])
    Y_homo = np.array([np.vdot(Xi, C) for Xi in X])
    Y_hete = np.array([np.vdot(SA_fun(RXi).dot(RXi), RD) for RXi in RX])
    L2_homo, L2_hete = np.linalg.norm(Y_homo), np.linalg.norm(Y_hete)
    RATE = (L2_hete / L2_homo) / DI_EXP
    D, Y_hete = D / RATE, Y_hete / RATE
    Y = Y_homo + Y_hete + rng.normal(size=N) * noise_level

    return (X, Y), (C, D, W), ROI_COOR


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from utils import visualize

    n = 4000
    r = 0.8
    N = int(n / r)
    shape = (28, 28)
    blockshape = (4, 4)
    idctshape = tuple((np.array(shape) / np.array(blockshape)).astype(int))
    DI_EXP = 1.0
    noise_level = 1
    print(f'Degree of individuation: {DI_EXP:.1f}')

    method = 'ByW'
    (X, Y), (C, D, W), ROI_COOR = Generate_XnY(method, N, shape, blockshape, DI_EXP, noise_level)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=r, random_state=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i in [0, 1, 2]:
        Xi = X[i]
        roi_coor = ROI_COOR[i]
        visualize(axes[i], Xi, coordinates=roi_coor, num_yx=idctshape, grid=True,
              title=f'The original sample {i} \n ROI: {", ".join([str(yx) for yx in roi_coor])}')
    plt.show()
