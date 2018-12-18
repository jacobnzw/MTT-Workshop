import numpy as np
from scipy.stats import invgamma

"""
GMM-PHD filter for linear motion/measurement model.

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

    def __init__(self, model):
        self.model = model
        self.L_max = 100
        self.elim_threshold = 1e-5
        self.merge_threshold = 4
        self.P_G = 0.999  # gate size in percentage
        self.gamma = invgamma.pdf(self.P_G, 0.0, 0.5 * self.model.dim_obs)  # FIXME
        self.gate_flag = True

    def filter(self, data):
        est = {
            'X': [],  # np.empty(data['X'].shape) * np.nan
            'N': np.zeros((data['K'], )),
        }


mod = Model()
true_state = mod.gen_truth()
meas = mod.gen_meas(true_state)
filt = GMPHDFilter(mod)
# est_state = filt.forward_pass(meas)