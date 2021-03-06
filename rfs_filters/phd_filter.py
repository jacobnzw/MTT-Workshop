import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from numpy import newaxis as na
from sklearn.externals import joblib
from scipy.optimize import linear_sum_assignment

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
        self.w_birth = 0.03 * np.ones((self.L_birth, ))
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
                tstate = self.F.dot(tstate)  # noiseless for now
                truth['X'][:, k, target_ind] = tstate
                truth['track_list'][k, target_ind] = target_ind
                truth['N'][k] += 1
        return truth

    def gen_meas(self, truth):
        meas = {
            'K': truth['K'],
            'Z': []  # np.empty((self.dim_obs, truth['K'], truth['total_tracks'])) * np.nan,
        }
        zero_mean = np.zeros((self.dim_obs, ))
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
                r = np.random.multivariate_normal(zero_mean, self.R, size=x.shape[1]).T
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

    Notes
    -----
    GM-PHD produces target estimates for each step, but can't tell which of the estimates form a track.
    The filters based on the labeled random finite sets can do this.

    Parameters
    ----------
    model : Model
    """

    def __init__(self, model, diagnostics=True):
        self.model = model
        self.L_max = 100  # maximum number of mixture components
        self.elim_threshold = 1e-5  # pruning threshold
        self.merge_threshold = 4.0  # merging threshold
        self.P_G = 0.999  # gate size in percentage
        self.gamma = scipy.stats.gamma.ppf(self.P_G, 0.5 * self.model.dim_obs, scale=2)
        self.gate_flag = True
        self.diagnostics = diagnostics
        self.F_EPS = np.finfo(float).eps

    def filter(self, data):
        est = {
            'X': np.frompyfunc(list, 0, 1)(np.empty((data['K'],), dtype=object)),
            'N': np.zeros((data['K'], ), dtype=np.int16),
        }

        # list for gated measurements
        data['Z_gated'] = list()

        # initial prior
        w_update = np.array([self.F_EPS])
        m_update = np.array([[0.1, 0, 0.1, 0]]).T
        P_update = np.expand_dims(np.diag([1, 1, 1, 1]) ** 2, axis=-1)  # add axis to the end -> (4,4,1)

        for k in range(data['K']):
            # PREDICTION
            # TODO: can be done without the loop, and thus w/o pre-allocs and the concatenation!
            num_update = len(w_update)
            w_predict = np.empty((num_update, ))
            m_predict = np.empty((self.model.dim_state, num_update))
            P_predict = np.empty((self.model.dim_state, self.model.dim_state, num_update))
            for i in range(num_update):
                # surviving weights
                w_predict[i] = self.model.P_S * w_update[i]
                # Kalman prediction
                m_predict[..., i] = self.model.F.dot(m_update[..., i])
                P_predict[..., i] = self.model.F.dot(P_update[..., i]).dot(self.model.F.T) + self.model.Q

            # append birth components to weights
            w_predict = np.concatenate((self.model.w_birth, w_predict))
            m_predict = np.concatenate((self.model.m_birth, m_predict), axis=-1)
            P_predict = np.concatenate((self.model.P_birth, P_predict), axis=-1)
            # number of predicted components
            L_predict = self.model.L_birth + num_update

            # GATING
            if self.gate_flag:
                # do gating of measurement set
                z_gated = self._gate_meas_gms(data['Z'][k], m_predict, P_predict)
            data['Z_gated'].append(z_gated)

            # UPDATE
            num_obs = z_gated.shape[1]  # number of measurements after gating
            # missed detection term
            w_update = self.model.Q_D * w_predict.copy()
            m_update = m_predict.copy()
            P_update = P_predict.copy()

            if num_obs > 0:  # if some measurements were detected
                # num_obs detection terms
                qz_temp, m_temp, P_temp = self._kalman_update(z_gated, m_predict, P_predict)
                for i in range(num_obs):  # TODO: can be done w/o the loop!
                    w_temp = self.model.P_D * w_predict * qz_temp[i, :]
                    denom = self.model.lambda_c*self.model.pdf_c + sum(w_temp)
                    w_temp /= denom

                    # updated mixture component weights, means and covariances
                    w_update = np.concatenate((w_update, w_temp))
                    m_update = np.concatenate((m_update, m_temp[:, i, :]), axis=-1)
                    P_update = np.concatenate((P_update, P_temp), axis=-1)

            # MANAGEMENT OF MIXTURE COMPONENTS
            L_posterior = len(w_update)

            # prunning
            w_update, m_update, P_update = gauss_prune(w_update, m_update, P_update, self.elim_threshold)
            L_prune = len(w_update)

            # merging
            w_update, m_update, P_update = gauss_merge(w_update, m_update, P_update, self.merge_threshold)
            L_merge = len(w_update)

            # capping the number of mixture components at given max (self.L_max)
            w_update, m_update, P_update = gauss_cap(w_update, m_update, P_update, self.L_max)
            L_cap = len(w_update)

            # STATE ESTIMATE EXTRACTION
            for i in (w_update > 0.5).nonzero()[0]:
                num_targets = int(np.round(w_update[i]))
                # TODO: right now, appending arrays with varying # columns to a list => hard to plot
                est['X'][k].append(np.tile(m_update[..., i, na], num_targets))
                est['N'][k] += num_targets

            # DIAGNOSTICS
            if self.diagnostics:
                print('time = {:3d} | '
                      'est_mean = {:4.2f} | '  # integral of the PHD == expected # targets in the whole state space
                      'est_card = {:3d} | '  # targets estimated by the filter (estimated cardinality of the state RFS)
                      'gm_orig = {:3d} | '
                      'gm_elim = {:3d} | '
                      'gm_merg = {:3d}'.format(k, sum(w_update), est['N'][k], L_posterior, L_prune, L_merge))

        # convert list of arrays to one 2D array
        # (each such array has varying # columns at each position in est['X'][k])
        for k in range(data['K']):
            est['X'][k] = np.concatenate(est['X'][k], axis=1) if len(est['X'][k]) > 0 else None

        return est

    def _kalman_update(self, z, m_predict, P_predict):
        num_obs, num_pred = z.shape[1], m_predict.shape[1]
        # space allocation
        qz = np.empty((num_obs, num_pred))
        m = np.empty((self.model.dim_state, num_obs, num_pred))
        P = np.empty((self.model.dim_state, self.model.dim_state, num_pred))
        I = np.eye(self.model.dim_state)

        for i in range(num_pred):
            # predicted measurement mean, covariance
            mz = self.model.H.dot(m_predict[..., i])
            Pz = self.model.H.dot(P_predict[..., i]).dot(self.model.H.T) + self.model.R

            for j in range(z.shape[1]):
                qz[j, i] = scipy.stats.multivariate_normal.pdf(z[:, j], mz, Pz)

            # Kalman gain NOTE: doesn't cover semi-definite Pz
            # iPz = np.linalg.inv(Pz)
            # K_gain = P_predict[..., i].dot(self.model.H.T).dot(iPz)
            # K_gain = scipy.solve(Pz, self.model.H.dot(P_predict[..., i])).T
            K_gain = scipy.linalg.cho_solve(scipy.linalg.cho_factor(Pz), self.model.H.dot(P_predict[..., i])).T

            # updated state mean and covariance
            m[..., i] = m_predict[..., i, None] + K_gain.dot(z - mz[:, None])
            P[..., i] = (I - K_gain.dot(self.model.H)).dot(P_predict[..., i])

        return qz, m, P

    def _gate_meas_gms(self, z, m, P):
        # pass through empty measurement sets (arrays)
        if z.shape[1] == 0:
            return z

        gated = np.array([], dtype=np.int)
        for i in range(m.shape[1]):
            mz = self.model.H.dot(m[..., i])
            Pz = self.model.H.dot(P[..., i]).dot(self.model.H.T) + self.model.R
            dz = scipy.linalg.solve_triangular(scipy.linalg.cholesky(Pz), z - mz[:, None])
            gated = np.union1d(gated, np.where((dz**2).sum(axis=0) < self.gamma))
        return z[:, gated]


def gauss_prune(w, m, P, elimination_threshold=1e-5):
    """
    Pruning of Gaussian mixture components based on threshold.

    Parameters
    ----------
    w
    m
    P
    elimination_threshold

    Returns
    -------

    """

    idx = np.asarray(w) > elimination_threshold
    return w[idx], m[..., idx], P[..., idx]


def gauss_merge(w, m, P, merge_threshold=4.0):
    """
    Merging of Gaussian mixture components based on threshold.

    Parameters
    ----------
    w
    m
    P
    merge_threshold

    Returns
    -------

    """

    idx = np.arange(len(w))
    el = 0
    w_merged, m_merged, P_merged = [], [], []
    while len(idx) != 0:
        w_i = w[idx]

        # indices of mixture components too close to component with highest weight
        j = idx[np.argmax(w_i)]
        dm = scipy.linalg.solve_triangular(scipy.linalg.cholesky(P[..., j]), m[:, idx] - m[:, j, na])
        too_close_idx = idx[np.sum(dm ** 2, axis=0) <= merge_threshold]

        # components to be merged
        w_el = w[too_close_idx]
        m_el = m[..., too_close_idx]
        P_el = P[..., too_close_idx]

        w_merged.append(sum(w_el))
        m_merged.append((w_el[na, :]*m_el).sum(axis=1) / w_merged[el])
        # covariance merging according to MATLAB IMPLEMENTATION
        P_merged.append((w_el[na, na, :] * P_el).sum(axis=-1) / w_merged[el])

        # covariance merging according to PAPER
        # dm = np.asarray(m_merged[el])[:, None] - m_el
        # P_merged.append((w_el[None, None, :]*(P_el + np.einsum('ij,kj->ikj', dm, dm))).sum(axis=-1) / w_merged[el])

        # update index list by removing indices of merged components
        idx = np.setdiff1d(idx, too_close_idx)
        el += 1

    return np.asarray(w_merged), np.asarray(m_merged).T, np.asarray(P_merged).T


def gauss_cap(w, m, P, max_comp=100):
    """
    Capping of the number of Gaussian mixture components at given threshold.

    Parameters
    ----------
    w
    m
    P
    max_comp

    Returns
    -------

    """

    if len(w) > max_comp:
        # L_max components in descending magnitude of weights
        idx = np.flip(np.argsort(np.asarray(w)))[:max_comp]
        return w[idx], m[..., idx], P[..., idx]
    else:
        return w, m, P


def unpack_matfile(mat_filename):
    """
    Unpack multi-target state sequence and measurements from MAT-file for debugging purposes.
    Necessary because MATLAB implementation uses cell arrays, which are otherwise hard to work with in Python.

    Parameters
    ----------
    mat_filename : str

    Returns
    -------
    : dict

    """

    # TODO: extend to load state

    from scipy.io import loadmat
    dd = loadmat(mat_filename)

    # extract multi-target measurements Z and number of time steps K
    meas = {
        'K': dd['meas'][0, 0][0][0, 0],
        'Z': list(dd['meas'][0, 0][1][:, 0])
    }
    return meas


def ospa(x, y, cutoff=100.0, p=1.0):
    """
    Optimal Subpattern Assignment (OSPA) distance between two finite sets.

    Parameters
    ----------
    x : ndarray
    y : ndarray
        Finite sets as 2D arrays, where each column is an element.

    cutoff : float
        Cut off parameter.

    p : float
        p-parameter of the metric.

    Returns
    -------
    dist : float
    """

    num_x, num_y = x.shape[1], y.shape[1]

    # if both sets are empty
    if num_x == 0 and num_y == 0:
        return 0

    # if at least one is empty
    if num_x == 0 or num_y == 0:
        return cutoff

    # compute cost matrix
    d = np.sqrt(np.sum((np.tile(x, num_y) - np.repeat(y, num_x, axis=1)) ** 2, axis=0))
    d = d.reshape(num_x, num_y).T
    d = np.minimum(d, cutoff) ** p

    # solve optimal assignment using Hungarian algorithm
    row_ind, col_ind = linear_sum_assignment(d)
    # compute cost of the optimal assignment
    cost = d[row_ind, col_ind].sum()

    dist = ((cutoff ** p * np.abs(num_y - num_x) + cost) / np.max((num_y, num_x))) ** (1/p)

    return dist


def plot_cardinality(true, estimated):
    plt.figure()
    plt.subplot(title='State RFS Cardinality')
    plt.plot(true['N'], 'r-', label='true')
    plt.plot(estimated['N'], 'k.', label='estimated')
    plt.xlabel('time [k]')
    plt.ylabel('# targets')
    plt.legend()
    plt.show()


def plot_xy_time(true, estimated, measurement):
    fig, ax = plt.subplots(2, 1, sharex=True)
    # plot of the X-coordinate
    ax[0].set_title('X-Position in Time')
    not_nan = np.logical_not(np.isnan(true['X'][0]))
    for target in range(true['X'].shape[-1]):
        ax[0].plot(np.where(not_nan[:, target])[0], true['X'][0, not_nan[:, target], target],
                   'r--', lw=1, alpha=0.5, label='true position')
    for k in range(len(estimated['X'])):
        # estimated target x-position
        if estimated['X'][k] is not None:
            num_targets = estimated['X'][k].shape[1]
            ax[0].plot(num_targets*[k], estimated['X'][k][0, :],
                       'ko', ms=4, label='estimated position')
        # all measurements
        num_measurements = measurement['Z'][k].shape[1]
        if num_measurements > 0:
            ax[0].plot(num_measurements*[k], measurement['Z'][k][0, :],
                       'k.', alpha=0.05, label='measurements')
        # gated measurements
        num_gated = measurement['Z_gated'][k].shape[1]
        if num_gated > 0:
            ax[0].plot(num_gated*[k], measurement['Z_gated'][k][0, :],
                       color='blue', ls='', marker='.', alpha=0.4, label='gated measurements')

    label_list = ['true position', 'estimated position', 'measurements', 'gated measurements']
    ax[0].legend([next(line for line in ax[0].lines if line._label == label) for label in label_list], label_list)

    # plot of the Y-coordinate
    ax[1].set_title('Y-Position in Time')
    not_nan = np.logical_not(np.isnan(true['X'][2]))
    for target in range(true['X'].shape[-1]):
        ax[1].plot(np.where(not_nan[:, target])[0], true['X'][2, not_nan[:, target], target], 'r--', lw=1, alpha=0.5)
    for k in range(len(estimated['X'])):
        # estimated target x-position
        if estimated['X'][k] is not None:
            num_targets = estimated['X'][k].shape[1]
            ax[1].plot(num_targets*[k], estimated['X'][k][2, :], 'ko', ms=4)
        # all measurements
        num_measurements = measurement['Z'][k].shape[1]
        if num_measurements > 0:
            ax[1].plot(num_measurements*[k], measurement['Z'][k][1, :], 'k.', alpha=0.05)
        # gated measurements
        num_gated = measurement['Z_gated'][k].shape[1]
        if num_gated > 0:
            ax[1].plot(num_gated*[k], measurement['Z_gated'][k][1, :], color='blue', ls='', marker='.', alpha=0.4)
    ax[1].set_xlabel('Time [k]')
    plt.show()


if __name__ == '__main__':
    # mod = Model()
    # true_state = mod.gen_truth()
    # meas = mod.gen_meas(true_state)
    # # meas = unpack_matfile('gmphd_meas.mat')
    # filt = GMPHDFilter(mod)
    # est_state = filt.filter(meas)
    #
    # joblib.dump({'state_true': true_state, 'state_estimated': est_state, 'measurement': meas}, 'plot_dump.dat')

    dd = joblib.load('plot_dump.dat')
    # plot_cardinality(dd['state_true'], dd['state_estimated'])
    plot_xy_time(dd['state_true'], dd['state_estimated'], dd['measurement'])

    # TODO: x/y plot of ground truth tracks (multi-target states)
    # TODO: plot of OSPA metrics
