import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import root_mean_squared_error, accuracy_score
from utils import *
import time
rng = np.random.RandomState(666)


class AIR():
    def __init__(self, task='reg'):
        self.task = task
        self.metric = 'RMSE' if self.task == 'reg' else 'Accuracy'
        self.metric_fun = root_mean_squared_error if self.task == 'reg' else accuracy_score
        self.C_hat = None
        self.D_hat = None
        self.W_hat = None

    def update_CD(self, X, Y, W, lam1, lam2):
        p, d = X[0].shape
        X_trans = np.array([Xi.dot(W).dot(Xi.T).dot(Xi) for Xi in X])
        X_stack = np.concatenate((X, X_trans), axis=1)
        M = Vectorize(X_stack)
        # print(M.shape)
        A = np.diag(np.concatenate([lam1 * np.ones(p*d), lam2 * np.ones(p*d)]))
        CD_hat = np.linalg.inv(M.T.dot(M) + A.T.dot(A)).dot(M.T).dot(Y).reshape(2*p, d)
        # print(CD_hat.shape)
        # model = LinearRegression(fit_intercept=False)
        # model.fit(M, Y)
        # CD_hat = model.coef_.reshape(X_stack[0].shape)
        C_hat = CD_hat[:p, :]
        D_hat = CD_hat[p:, :]

        return C_hat, D_hat

    def update_W(self, X, Y, C, D, lam3, normalize):
        Y_homo = np.array([np.vdot(Xi, C) for Xi in X])
        Y_hete = Y - Y_homo
        X_trans = np.array([Xi.T.dot(D).dot(Xi.T).dot(Xi) for Xi in X])
        model = Ridge(alpha=lam3, fit_intercept=False) if lam3 > 0 else LinearRegression(fit_intercept=False)
        model.fit(Vectorize(X_trans), Y_hete)
        shape_W = np.repeat(C.shape[1], 2)
        W_hat = model.coef_.reshape(shape_W)
        if normalize:
            W_hat = W_hat / np.linalg.norm(W_hat)

        return W_hat

    def Init(self, X, Y, method='Init_CD', C_init=None, D_init=None, W_init=None):
        shape = X[0].shape
        if method == 'Init_CD':
            print('Initializing C and D...', end=' | ')
            model = LinearRegression(fit_intercept=False)
            model.fit(Vectorize(X), Y)
            C_hat = model.coef_.reshape(shape)
            Y_res = Y - model.predict(Vectorize(X))
            model = LinearRegression(fit_intercept=False)
            model.fit(Vectorize(X), Y_res)
            D_hat = model.coef_.reshape(shape)
        elif method == 'Init_W':
            print('Initializing W...', end=' | ')
            M = np.zeros((shape[1] ** 2, np.prod(shape)))
            for i in range(len(Y)):
                M += Y[i] * np.kron(X[i].T.dot(X[i]), X[i].T)
            U, S, Vt = np.linalg.svd(M)
            W_init = U[:, 0].reshape([shape[1]] * 2)
            C_hat, D_hat = self.update_CD(X, Y, W_init, lam1=1, lam2=1)
        else:
            C_hat, D_hat = C_init, D_init

        print('Finished.')
        return C_hat, D_hat

    def predict(self, X):
        Y_homo = np.array([np.vdot(Xi, self.C_hat) for Xi in X])
        Y_hete = np.array([np.vdot(Xi.dot(self.W_hat).dot(Xi.T).dot(Xi), self.D_hat) for Xi in X])
        Y = Y_homo + Y_hete
        return Y

    def fit(self, train, valid, init='Init_CD', init_dict={}, lams=None, norm_W=True, max_itr=10, tol=1e-5, timing=False, fold=False):
        print('Start at', time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())) if timing else None
        t0 = time.time()

        lam1, lam2, lam3 = [1, 1, 1] if lams is None else lams
        print('Lambda | C: {}, D: {}, W: {}'.format(lam1, lam2, lam3))

        X_trn, Y_trn = train
        X_val, Y_val = valid
        # n, p, d = X_trn.shape

        ''' Initialization '''
        C_init, D_init, W_init = init_dict['C'], init_dict['D'], init_dict['W']
        C_hat, D_hat = self.Init(X_trn, Y_trn, init, C_init=C_init, D_init=D_init, W_init=W_init)

        ''' Iteration '''
        loss_former = 1e10
        Record_list = []
        for itr in range(max_itr):
            ''' Update W '''
            W_hat = self.update_W(X_trn, Y_trn, C_hat, D_hat, lam3, norm_W)
            ''' Update (C, D) '''
            C_hat, D_hat = self.update_CD(X_trn, Y_trn, W_hat, lam1, lam2)

            ''' store '''
            self.C_hat = C_hat
            self.D_hat = D_hat
            self.W_hat = W_hat

            ''' Evaluation '''
            Y_inhat = self.predict(X_trn)
            Y_outhat = self.predict(X_val)
            loss_inhat = self.metric_fun(Y_trn, Y_inhat)
            loss_outhat = self.metric_fun(Y_val, Y_outhat)
            text = f'Iteration: ' + str(itr + 1).zfill(
                len(str(max_itr))) + ' | ' + self.metric + f' | Train: {loss_inhat:.3f}, Valid: {loss_outhat:.3f}'
            print(text, end='\r') if fold else print(text)

            Record_list.append([C_hat, W_hat, loss_inhat, loss_outhat])

            ''' Early Stopping '''
            change_rate = np.abs(loss_former - loss_inhat) / loss_former
            loss_former = loss_inhat
            if change_rate < tol:
                print('Training loss converges at iteration {}.'.format(itr))
                break

            print('End at', time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())) if timing else None
            t1 = time.time()
            print(f'Cost time {round(t1 - t0)}s.') if timing else None
        print('\n')

        return (self.C_hat, self.D_hat, self.W_hat), Record_list


if __name__ == '__main__':
    from Generate_XnY import Generate_XnY
    from sklearn.model_selection import train_test_split

    n = 4000
    r = 0.8
    N = int(n / r)
    shape = (28, 28)
    blockshape = (4, 4)
    idctshape = tuple((np.array(shape) / np.array(blockshape)).astype(int))
    DI_EXP = 0.5
    noise_level = 1

    settings = {'method': 'ByW', 'N': N, 'shape': shape, 'blockshape': blockshape, 'DI_EXP': DI_EXP, 'noise_level': noise_level, 'random_state': 0}
    (X, Y), (C, D, W), ROI_COOR = Generate_XnY(**settings)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=r, random_state=0)
    RX = np.array([R_opt_pro(x, idctshape)[1] for x in X])
    RX_train = np.array([R_opt_pro(x, idctshape)[1] for x in X_train])
    RX_test = np.array([R_opt_pro(x, idctshape)[1] for x in X_test])
    train, valid = (RX_train, Y_train.ravel()), (RX_test, Y_test.ravel())

    _, RC = R_opt_pro(C, idctshape)
    _, RD = R_opt_pro(D, idctshape)
    p, d = RC.shape

    print(f'Degree of individuation: {DI_EXP:.1f}')
    print(f'Y | Mean: {Y.mean():.3f}, Variance: {Y.var():.3f} (s.d.: {Y.std():.3f}), Max abs: {np.abs(Y).max():.3f}')

    ''' AIR '''
    print('AIR...')
    print(f'Number of parameters: {2 * p * d + d * d} = {p * d} x {2} + {d * d}')
    air = AIR()
    Init_method = 'Init_W'
    Init_Dict = {'C': None, 'D': None, 'W': None}

    lams = [1e-3, 1, 0]
    max_itr = 50
    params, records = air.fit(train=train, valid=valid, lams=lams, max_itr=max_itr, init=Init_method,
                              init_dict=Init_Dict)
    RC_hat, RD_hat, W_hat = params