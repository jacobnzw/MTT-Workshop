import numpy as np
import scipy.stats

"""
GMM-PHD filter [1]_ for linear motion/measurement model assuming no target spawning.

Based on the MATLAB implementation in the RFS tracking toolbox http://ba-tuong.vo-au.com/codes.html.

References
----------
.. [1]: B.-N. Vo, and W. K. Ma, "The Gaussian mixture Probability Hypothesis Density Filter," 
        IEEE Trans Signal Processing, Vol. 54, No. 11, pp. 4091-4104, 2006.
"""


class Model:

    def __init__(self, dt=1):
        self.dim_state = 4
        self.dim_obs = 2
        self.dt = dt  # sampling period
        # transition matrix
        self.F = np.array([[1, dt, 0, 0],
                           [0, 1, 0, 0],
                           [0, 0, 1, dt],
                           [0, 0, 0, 1]])
        self.B = np.array([[dt**2/2, 0],
                           [dt, 0],
                           [0, dt**2/2],
                           [0, dt]])
        # process noise covariance
        self.Q = 5**2 * self.B.dot(self.B.T)
        # target survival probability
        self.P_S = .99
        self.Q_S = 1 - self.P_S

        # Poisson birth RFS parameters (multiple Gaussian components)
        self.L_birth = 4  # of Gaussian birth terms
        # weights, means, stds, covs of Gaussian birth terms
        self.w_birth = 0.03 * np.ones((self.L_birth, 1))
        self.m_birth = np.zeros((self.dim_state, self.L_birth))
        self.B_birth = np.zeros((self.dim_state, self.dim_state, self.L_birth))
        self.P_birth = np.zeros((self.dim_state, self.dim_state, self.L_birth))

        self.m_birth[:, 0] = np.array([0, 0, 0, 0]).astype(np.float)
        self.B_birth[..., 0] = np.diag([10, 10, 10, 10]).astype(np.float)
        self.P_birth[..., 0] = self.B_birth[..., 0].dot(self.B_birth[..., 0].T)

        self.m_birth[:, 1] = np.array([400, 0, -600, 0]).astype(np.float)
        self.B_birth[..., 1] = np.diag([10, 10, 10, 10]).astype(np.float)
        self.P_birth[..., 1] = self.B_birth[..., 1].dot(self.B_birth[..., 1].T)

        self.m_birth[:, 2] = np.array([-800, 0, -200, 0]).astype(np.float)
        self.B_birth[..., 2] = np.diag([10, 10, 10, 10]).astype(np.float)
        self.P_birth[..., 2] = self.B_birth[..., 2].dot(self.B_birth[..., 2].T)

        self.m_birth[:, 3] = np.array([-200, 0, 800, 0]).astype(np.float)
        self.B_birth[..., 3] = np.diag([10, 10, 10, 10]).astype(np.float)
        self.P_birth[..., 3] = self.B_birth[..., 3].dot(self.B_birth[..., 3].T)

        # observation model parameters
        self.H = np.array([[1, 0, 0, 0],
                           [0, 0, 1, 0]])
        self.D = np.diag([10, 10])
        # measurement noise covariance
        self.R = self.D.dot(self.D.T)
        # detection probability
        self.P_D = .98
        self.Q_D = 1 - self.P_D  # probability of missed detection of measurements
        # clutter parameters
        self.lambda_c = 60  # poisson average rate of uniform clutter
        self.range_c = np.array([[-1000, 1000],
                                 [-1000, 1000]])
        self.pdf_c = 1 / np.prod(self.range_c[:, 1] - self.range_c[:, 0])  # uniform clutter density

    def gen_truth(self):
        K = 100
        total_tracks = 12
        truth = {
            'K': K,  # length of data/number of scans
            'X': np.empty((self.dim_state, K, total_tracks)) * np.nan,  # ground-truth target states
            'N': np.zeros((K, ), dtype=np.int8),  # number of targets at each time step
            'track_list': np.empty((K, total_tracks)) * np.nan,  # absolute index target identities
            'total_tracks': total_tracks  # total number of appearing tracks
        }
        # initial states of targets
        xstart = np.zeros((self.dim_state, total_tracks), dtype=np.float)
        # xstart = np.array([0, 400, -800, 400, 400, 0, -800, -200, -800, -200, 0, -200])
        xstart[:, 0] = np.array([0, 0, 0, -10])
        xstart[:, 1] = np.array([400, -10, -600, 5])
        xstart[:, 2] = np.array([-800, 20, -200, -5])
        xstart[:, 3] = np.array([400, -7, -600, -4])
        xstart[:, 4] = np.array([400, -2.5, -600, 10])
        xstart[:, 5] = np.array([0, 7.5, 0, -5])
        xstart[:, 6] = np.array([-800, 12, -200, 7])
        xstart[:, 7] = np.array([-200, 15, 800, -10])
        xstart[:, 8] = np.array([-800, 3, -200, 15])
        xstart[:, 9] = np.array([-200, -3, 800, -15])
        xstart[:, 10] = np.array([0, -20, 0, -15])
        xstart[:, 11] = np.array([-200, 15, 800, -5])

        # i-th column == birth/death times of i-th target
        bd_times = np.array([[1, 1, 1, 20, 20, 20, 40, 40, 60, 60, 80, 80],
                             [70, K+1, 70, K+1, K+1, K+1, K+1, K+1, K+1, K+1, K+1, K+1]])

        for target_ind in range(total_tracks):  # for each target
            tstate = xstart[:, target_ind]
            for k in range(bd_times[0, target_ind], min(bd_times[1, target_ind], K)):
                # propagate state through constant-velocity (CV) model
                truth['X'][:, k, target_ind] = self.F.dot(tstate)  # noiseless for now
                truth['track_list'][k, target_ind] = target_ind
                truth['N'][k] += 1
        return truth

    def gen_meas(self, truth):
        meas = {
            'K': truth['K'],
            'Z': []  # np.empty((self.dim_obs, truth['K'], truth['total_tracks'])) * np.nan,
        }
        for k in range(truth['K']):
            Z_k = None
            if truth['N'][k] > 0:  # if there are some targets in the scene
                # determine which targets in the scene were detected (based on detection probability P_D)
                detected = np.random.rand(truth['N'][k], ) <= self.P_D
                x = truth['X'][:, k, :]
                present_and_detected = ~np.isnan(x.sum(axis=0))
                present_and_detected[present_and_detected == True] &= detected
                x = x[:, present_and_detected]

                # generate measurement
                r = np.random.multivariate_normal(np.zeros((self.dim_obs,)), self.R, size=x.shape[1]).T
                # meas['Z'][:, k, present_and_detected] = self.H.dot(x) + r
                Z_k = self.H.dot(x) + r

            # generate clutter
            N_c = np.random.poisson(self.lambda_c)  # number of clutter points
            bounds = np.diag(self.range_c.dot(np.array([-1, 1])))
            clutter = -1000.0 + bounds.dot(np.random.rand(self.dim_obs, N_c))
            # measurement set = union of target measurements and clutter
            Z_k = np.hstack((Z_k, clutter)) if Z_k is not None else clutter
            meas['Z'].append(Z_k)
        return meas


class GMPHDFilter:
    """
    Gaussian mixture Probability Hypothesis Density Filter (GM-PHD) for a linear model.

    Parameters
    ----------
    model : Model
    """

    def __init__(self, model, diagnostics=True):
        self.model = model
        self.L_max = 100
        self.elim_threshold = 1e-5
        self.merge_threshold = 4
        self.P_G = 0.999  # gate size in percentage
        self.gamma = scipy.stats.gamma.ppf(self.P_G, 0.5 * self.model.dim_obs, scale=2)
        self.gate_flag = True
        self.diagnostics = diagnostics
        self.F_EPS = np.finfo(float).eps

    def filter(self, data):
        est = {
            'X': [],  # np.empty(data['X'].shape) * np.nan
            'N': np.zeros((data['K'], )),
        }
        # TODO: w_, m_ and P_ are probably better off being ndarrays
        # initial prior
        w_update = [self.F_EPS]
        m_update = [np.array([0.1, 0, 0.1, 0])]
        P_update = [np.diag([1, 1, 1, 1]) ** 2]
        L_update = 1

        w_predict = []
        m_predict = []
        P_predict = []

        for k in range(data['K']):
            # PREDICTION
            for i in range(len(w_update)):
                # surviving weights
                w_predict.append(self.model.P_S * w_update[i])
                # Kalman prediction
                m_predict.append(self.model.F.dot(m_update[i]))
                P_predict.append(self.model.F.dot(P_update[i]).dot(self.model.F.T) + self.model.Q)

            # append birth components to weights
            w_predict.append(self.model.w_birth)
            m_predict.append(self.model.m_birth)
            P_predict.append(self.model.P_birth)
            # number of predicted components
            L_predict = self.model.L_birth + L_update

            # GATING
            if self.gate_flag:
                # do gating of measurement set
                data['Z'][k] = self._gate_meas_gms(data['Z'][k], m_predict, P_predict)

            # UPDATE
            num_obs = len(data['Z'][k])  # number of measurements after gating
            # missed detection term
            w_update = [self.model.Q_D*w_predict[i] for i in range(len(w_predict))]
            m_update = m_predict
            P_update = P_predict

            if num_obs > 0:  # if some measurements were detected
                # num_obs detection terms
                qz_temp, m_temp, P_temp = self._kalman_update(data['Z'][k], m_predict, P_predict)
                for i in range(num_obs):
                    w_temp = [self.model.P_D * w_predict[j] * qz_temp[i, j] for j in range(len(w_predict))]
                    denom = self.model.lambda_c*self.model.pdf_c + sum(w_temp)
                    w_temp = [w / denom for w in w_temp]

                    # updated mixture component weights, means and covariances
                    w_update.append(w_temp)
                    m_update.append(m_temp[:, i, :])
                    P_update.append(P_temp)

            # MANAGEMENT OF MIXTURE COMPONENTS
            L_posterior = len(w_update)

            # prunning
            w_update, m_update, P_update = self._gauss_prune(w_update, m_update, P_update)
            L_prune = len(w_update)

            # merging
            w_update, m_update, P_update = self._gauss_merge(w_update, m_update, P_update)
            L_merge = len(w_update)

            # capping the number of mixture components at given max (self.L_max)
            w_update, m_update, P_update = self._gauss_cap(w_update, m_update, P_update)
            L_cap = len(w_update)

            # STATE ESTIMATE EXTRACTION
            idx = [index for index in range(len(w_update)) if w_update[index] > 0.5]
            for index in idx:
                num_targets = np.round(w_update[index])
                est['X'].append(np.tile(m_update[index], num_targets))
                est['N'][k] += num_targets

            # DIAGNOSTICS
            if self.diagnostics:
                print('time= {:3d} | '
                      'est_mean= {:4.2f} | '
                      'est_card= {:3d} | '
                      'gm_orig= {:3d} | '
                      'gm_elim= {:3d} | '
                      'gm_merg= {:3d}'.format(k, sum(w_update), est['N'][k], L_posterior, L_prune, L_merge))

    def _kalman_update(self, z, m_predict, P_predict):
        num_obs, num_pred = z.shape[1], len(m_predict)
        # space allocation
        qz = np.empty((num_obs, num_pred))
        m = np.empty((self.model.dim_state, num_obs, num_pred))
        P = np.empty((self.model.dim_state, self.model.dim_state, num_pred))
        I = np.eye(self.model.dim_state)

        for i in range(len(m_predict)):
            # predicted measurement mean, covariance
            mz = self.model.H.dot(m_predict[i])
            Pz = self.model.H.dot(P_predict[i]).dot(self.model.H.T) + self.model.R

            # Kalman gain
            iPz = np.linalg.inv(Pz)  # FIXME replace this atrocity with cho_solve
            K_gain = P_predict[i].dot(self.model.H.T).dot(iPz)

            for j in range(z.shape[1]):
                qz[j, i] = scipy.stats.multivariate_normal.pdf(z[j], mz, Pz)
            m[..., i] = m_predict[i][:, None] + K_gain.dot(z - mz[:, None])
            P[..., i] = (I - K_gain.dot(self.model.H)).dot(P_predict)

        return qz, m, P

    @staticmethod
    def _cho_inv_dot(A, b=None):
        if A.ndim != 2:
            raise ValueError('A must be 2-D array.')
        if A.shape[0] != A.shape[1]:
            raise ValueError('A must be square.')
        if b is None:
            b = np.eye(A.shape)
        return scipy.linalg.cho_solve(scipy.linalg.cho_factor(A, lower=True), b)

    def _gate_meas_gms(self, z, m, P):
        # pass through empty measurement sets (arrays)
        if len(z) == 0:
            return z

        gated = np.array([])
        for i in range(len(m)):
            mz = self.model.H.dot(m[i])
            Pz = self.model.H.dot(P[i]).dot(self.model.H.T) + self.model.R
            dz = self._cho_inv_dot(Pz, z - mz[:, None])
            gated = np.union1d(gated, np.where(dz.T.dot(dz) < self.gamma))
        return z[:, gated]

    def _gauss_prune(self, w, m, P):
        """
        Pruning of Gaussian mixture components based on threshold.

        Parameters
        ----------
        w
        m
        P

        Returns
        -------

        """

        idx = np.where(np.asarray(w) > self.elim_threshold)
        w = [w[i] for i in idx]
        m = [m[i] for i in idx]
        P = [P[i] for i in idx]
        return w, m, P

    def _gauss_merge(self, w, m, P):
        """
        Merging of Gaussian mixture components based on threshold.

        Parameters
        ----------
        w
        m
        P

        Returns
        -------

        """

        idx = np.arange(len(w))
        el = 0
        w_merged, m_merged, P_merged = [], [], []
        while len(idx) != 0:
            w_i = [w[i] for i in idx]
            m_i = [w[i] for i in idx]
            P_i = [w[i] for i in idx]

            # indices of mixture components too close to component with highest weight
            j = np.argmax(w_i)
            too_close_idx = set([i for i in idx if (m[i] - m[j]).T.dot(np.linalg.inv(P[i])).dot(m[i] - m[j])])

            w_el = [w_i[i] for i in too_close_idx]
            m_el = [m_i[i] for i in too_close_idx]
            P_el = [P_i[i] for i in too_close_idx]

            w_merged.append(sum(w_el))
            m_merged.append(sum([w_el[i]*m_el[i] for i in range(len(w_el))]) / w_merged[el])
            P_merged.append(sum([w_el[i]*(P_el[i] + np.outer(m_merged[el] - m_el[i], m_merged[el] - m_el[i]))
                                for i in range(len(w_el))]) / w_merged[el])
            # update index list by removing indices of merged components
            idx = np.setdiff1d(idx, too_close_idx)
            # idx = [index for index in idx if index not in too_close_idx]
            el += 1

        return w_merged, m_merged, P_merged

    def _gauss_cap(self, w, m, P):
        """
        Capping of the number of Gaussian mixture components at given threshold.

        Parameters
        ----------
        w
        m
        P

        Returns
        -------

        """

        if len(w) > self.L_max:
            # L_max components in descending magnitude of weights
            idx = np.flip(np.argsort(np.asarray(w)))[:self.L_max]
            w = [w[i] for i in idx]
            m = [m[i] for i in idx]
            P = [P[i] for i in idx]
        return w, m, P


mod = Model()
true_state = mod.gen_truth()
meas = mod.gen_meas(true_state)
filt = GMPHDFilter(mod)
# est_state = filt.forward_pass(meas)
