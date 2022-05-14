import numpy as np
from scipy.sparse import csr_matrix
from scipy.optimize import nnls
from tqdm import tqdm

import vvt.utils as utils

def get_solver(cube, proj_mat):
    '''Initializes a Solver for the given Cube and projection matrix.'''
    # Laplacian operator
    L = laplacian_matrix_3d(cube.nx, cube.ny, cube.nz)

    # voxel matrices
    n_voxels = len(cube.element_stiffness_mats)
    element_stiffness_mats, element_mass_mats = [], []
    for (Ke, Me) in tqdm(zip(cube.element_stiffness_mats,
                             cube.element_mass_mats), total=n_voxels,
                            desc='Gather element mats'):
        Ke_free = Ke.tocsr()[cube.nonbc_dofs].tocsc()[:, cube.nonbc_dofs].tocoo()
        Me_free = Me.tocsr()[cube.nonbc_dofs].tocsc()[:, cube.nonbc_dofs].tocoo()
        element_stiffness_mats.append(Ke_free.astype('float32'))
        element_mass_mats.append(Me_free.astype('float32'))

    # projection matrices
    n_vts = cube.n_dofs // 3
    proj_mat_normalized = proj_mat / abs(proj_mat).max()
    P = full_projection_matrix(n_vts, proj_mat_normalized)
    G = full_mask_matrix(cube.n_dofs, cube.image_space_dofs)
    P_free = P[cube.nonbc_dofs][:, cube.nonbc_dofs]
    G_free = G[cube.nonbc_dofs][:, cube.nonbc_dofs]

    # Initialize solver.
    s = Solver(
        element_stiffness_mats, element_mass_mats, 
        laplacian=L, P=P_free, G=G_free)
    
    return s

def laplacian_matrix_3d(nx, ny, nz):
    '''Return a Laplacian matrix (L) for a 3d array.
    L[i, j] = 1 iff i != j are adjacent in the 3d array of size
    (nx, ny, nz).
    L[i, i] = -deg(i).
    '''
    def _index(x, y, z):
        return (nx*ny)*z + nx*y + x
    nelems = nx * ny * nz
    L = np.zeros((nelems, nelems))
    for z in range(nz):
        for y in range(ny):
            for x in range(nx):
                curr = _index(x, y, z)
                if x > 0:
                    up = _index(x-1, y, z)
                    L[curr, up] = 1
                if x < nx - 1:
                    down = _index(x+1, y, z)
                    L[curr, down] = 1
                if y > 0:
                    left = _index(x, y-1, z)
                    L[curr, left] = 1
                if y < ny - 1:
                    right = _index(x, y+1, z)
                    L[curr, right] = 1
                if z > 0:
                    bottom = _index(x, y, z-1)
                    L[curr, bottom] = 1
                if z < nz - 1:
                    top = _index(x, y, z+1)
                    L[curr, top] = 1
                L[curr, curr]  = -np.count_nonzero(L[curr])
    return L

def full_projection_matrix(n_total_pts, proj_mat):
    '''Create a full projection matrix that maps all DOFs onto image-space.
    Inputs:
        n_total_pts -- total number of points that would be mapped from
            3d to 2d.
        proj_mat (2, 3) -- 3d-to-2d projection matrix.
    Output:
        P (N_DOFS, N_DOFS) -- full projection matrix, that maps DOFs in 3d
            to 2d image-space, where the 3rd dimension is always 0.
    '''
    P = np.kron(
        np.eye(n_total_pts, dtype=int),
        np.r_[proj_mat, np.zeros((1, 3))])
    return P

def full_mask_matrix(n_dofs, seen_dofs):
    '''Return a binary matrix with 1s for DOFs seen in image-space.'''
    G = np.zeros((n_dofs, n_dofs))
    G[seen_dofs[::3], seen_dofs[::3]] = 1
    G[seen_dofs[1::3], seen_dofs[1::3]] = 1
    return G

class Solver:
    def __init__(self, element_stiffness_mats, element_mass_mats,
                 laplacian, P, G):
        '''Initialize Solver instance.
        Params:
            element_stiffness_mats: element stiffness matrices (of non-bc dofs).
            element_mass_mats: element mass matrices (of non-bc dofs).
            laplacian: Laplacian operator matrix.
            P: projection matrix from 3D to 2D.
            G: masking matrix that extracts seen DOFs in 2D space.
        '''
        n_free_dofs = element_stiffness_mats[0].shape[0]
        self.n_free_dofs = n_free_dofs
        
        self.element_stiffness_mats = element_stiffness_mats
        self.element_mass_mats = element_mass_mats
        self.n_weights = len(element_stiffness_mats)
        self.L = laplacian
        self.P = csr_matrix(G @ P)

        assert len(element_stiffness_mats) == len(element_mass_mats)
        assert element_mass_mats[0].shape[0] == n_free_dofs
        assert laplacian.shape[0] == self.n_weights
        assert P.shape[0] == n_free_dofs
        assert G.shape[0] == n_free_dofs

        # Create dictionary to store hyperparams.
        self.params = { }

    def update_observations(self, U_observed, evals_observed):
        assert len(evals_observed) == U_observed.shape[1]
        self.evals_observed = evals_observed
        self.Uo = U_observed
        self.k = len(evals_observed)

    def _constraint_loss(self, Ut, evalst, Kt, Mt, dual_vars):
        '''Returns the unweighted and dual-var-weighted constraint
        losses.'''
        constr_loss_unw, constr_loss = 0., 0.
        for i in range(self.k):
            Di = Kt - evalst[i] * Mt
            loss_i = np.linalg.norm(Di @ Ut[:, i])**2
            constr_loss_unw += loss_i
            constr_loss += dual_vars[i] * loss_i
        constr_loss_unw *= (0.5 / self.k)
        constr_loss *= (0.5 / self.k)
        return constr_loss_unw, constr_loss

    def _data_loss(self, Ut):
        data_loss = 0.
        for i in range(self.k):
            loss_i = np.linalg.norm(
                self.P @ Ut[:, i] - self.Uo[:, i])**2
            data_loss += loss_i
        data_loss *= (0.5 / self.k)
        return data_loss

    def _eigval_data_loss(self, evalst):
        loss = np.linalg.norm(evalst - self.evals_observed)**2
        loss *= (0.5 / self.k)
        return loss

    def _tsv(self, x):
        loss = 0.5 * np.linalg.norm(self.L @ x)**2
        return loss

    def _scale_reg_loss(self, x, x_mean):
        loss = 0.5 * (np.mean(x) - x_mean)**2
        return loss

    def loss(self, Ut, evalst, Kt, Mt, wt, vt, dual_vars):
        alpha_u = self.params['alpha_u']
        alpha_e = self.params['alpha_e']
        alpha_w = self.params['alpha_w']
        alpha_v = self.params['alpha_v']
        alpha_w_mean = self.params['alpha_w_mean']
        alpha_v_mean = self.params['alpha_v_mean']
        w_mean = self.params['w_mean']
        v_mean = self.params['v_mean']

        # constraint loss
        constr_loss_unw, constr_loss = self._constraint_loss(
            Ut, evalst, Kt, Mt, dual_vars)

        # data-matching loss
        data_loss_unw = self._data_loss(Ut)

        # eigenvalue data-matching loss
        eigval_data_loss_unw = self._eigval_data_loss(evalst)

        # TSV regularization
        tsv_w_unw = self._tsv(wt)
        tsv_v_unw = self._tsv(vt)

        # scale regularization
        w_mean_loss_unw = self._scale_reg_loss(wt, w_mean)
        v_mean_loss_unw = self._scale_reg_loss(vt, v_mean)
        scale_reg_loss_unw = w_mean_loss_unw + v_mean_loss_unw

        # total unweighted loss
        total_loss_unw = (constr_loss_unw + data_loss_unw
            + eigval_data_loss_unw + tsv_w_unw + tsv_v_unw
            + scale_reg_loss_unw)

        unweighted_loss_dict = {
            'constr': constr_loss_unw,
            'data': data_loss_unw,
            'eigval_data': eigval_data_loss_unw,
            'tsv_w': tsv_w_unw,
            'tsv_v': tsv_v_unw,
            'scale_reg': scale_reg_loss_unw,
            'total': total_loss_unw
        }

        # weighted losses
        data_loss = alpha_u * data_loss_unw
        eigval_data_loss = alpha_e * eigval_data_loss_unw
        tsv_w = alpha_w * tsv_w_unw
        tsv_v = alpha_v * tsv_v_unw
        scale_reg_loss = (
            alpha_w_mean * w_mean_loss_unw + alpha_v_mean * v_mean_loss_unw)

        total_loss = (constr_loss + data_loss + eigval_data_loss
            + tsv_w + tsv_v + scale_reg_loss)
        
        loss_dict = {
            'constr': constr_loss,
            'data': data_loss,
            'eigval_data': eigval_data_loss,
            'tsv_w': tsv_w,
            'tsv_v': tsv_v,
            'scale_reg': scale_reg_loss,
            'total': total_loss
        }
        return unweighted_loss_dict, loss_dict
    
    def solve_w_v(self, Ut, evalst, dual_vars, enforce_pos=False):
        alpha_w = self.params['alpha_w']
        alpha_v = self.params['alpha_v']
        alpha_w_mean = self.params['alpha_w_mean']
        alpha_v_mean = self.params['alpha_v_mean']
        w_mean = self.params['w_mean']
        v_mean = self.params['v_mean']

        k = self.k
        m = self.n_weights
        n = self.n_free_dofs
        
        # constraint A
        A_constr = np.zeros((k*n, 2*m))
        for l in range(k):
            ul = Ut[:,l]
            A_l = np.zeros((n, 2*m))
            for (i, Ki) in enumerate(self.element_stiffness_mats):
                A_l[:, i] = Ki @ ul
            for (i, Mi) in enumerate(self.element_mass_mats):
                A_l[:, m+i] = -evalst[l] * Mi @ ul
            # Plug this A_l matrix into the full A_constr matrix.
            A_constr[(l*n):(l+1)*n] = np.sqrt(dual_vars[l] / k) * A_l

        # smoothness regularization A
        A_reg = np.block([
            [np.sqrt(alpha_w)*self.L,   np.zeros((m, m))],
            [np.zeros((m, m)),          np.sqrt(alpha_v)*self.L]
        ])

        # mean regularization A
        A_mean = np.block([
            [(np.sqrt(alpha_w_mean)/m) * np.ones(m), np.zeros(m)],
            [np.zeros(m), (np.sqrt(alpha_v_mean)/m) * np.ones(m)]
        ])
        # mean regularization b
        b_mean = np.array([np.sqrt(alpha_w_mean) * w_mean,
                           np.sqrt(alpha_v_mean) * v_mean])

        # Stack the A matrices into one big A matrix.
        A_full = np.concatenate((A_constr, A_reg, A_mean))

        # Solve linear system for x: Ap * x = bp
        Ap = A_full.T @ A_full
        bp = A_mean.T @ b_mean

        if enforce_pos:
            return nnls(Ap, bp)[0]
        else:
            return np.linalg.solve(Ap, bp)

    def solve_U(self, Kt, Mt, evalst, dual_vars):
        alpha_u = self.params['alpha_u']
        Ustar = np.zeros(self.Uo.shape, dtype=self.Uo.dtype)
        for i in range(self.k):
            D = Kt - evalst[i] * Mt
            A = dual_vars[i] * D.T @ D + alpha_u * self.P.T @ self.P
            b = alpha_u * self.P.T @ self.Uo[:,i]
            Ustar[:, i] = np.linalg.solve(A.todense(), b)
        return Ustar
    
    def solve_evals(self, Kt, Mt, Ut, dual_vars):
        alpha_e = self.params['alpha_e']
        evals_star = np.zeros(self.k)
        for i in range(self.k):
            ui = Ut[:,i]
            yi = dual_vars[i]
            num = yi * ui.T @ Kt.T @ Mt @ ui + alpha_e*self.evals_observed[i]
            den = yi * np.linalg.norm(Mt @ ui)**2 + alpha_e
            evals_star[i] = num / den
        return evals_star

    def update_dual_vars(self, Kt, Mt, Ut, evalst, dual_vars, eta):
        new_dual_vars = dual_vars.copy()
        for i in range(self.k):
            Di = Kt - evalst[i] * Mt
            new_dual_vars[i] += eta * np.linalg.norm(Di @ Ut[:, i])
        return new_dual_vars