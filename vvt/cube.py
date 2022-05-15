import multiprocessing as mp

import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse

from dolfin import *
from tqdm import tqdm

def get_cube_model(xlen, ylen, zlen, nx, ny, nz, nu, degree=1):
    """
    Initialize a Cube object, assuming E=1, rho=1. With these default
    Young's modulus and density values, the cube's element stiffness and mass
    matrices have unit material properties and thus can be used when solving
    for material properties.

    Parameters
    ----------
    xlen, ylen, zlen: float
        The dimensions of the box. Note that these do not have to be equal,
        so the cube can be a box of any aspect ratio.
    nx, ny, nz: int
        The number of voxels in the x, y, and z directions, respectively.
    nu: float, between 0.1 and 0.45
        The homogeneous Poisson's ratio of the cube.
    degree: int, default=1
        The element order of the cube mesh. E.g., `degree=1` corresponds to
        linear elements.

    Returns
    -------
    cube: Cube
        A Cube object that corresponds to the given parameters and has 
        homogeneous material properties E=1, rho=1, and nu=`nu`.
    """
    n_vox = nx * ny * nz
    if np.isscalar(nu):
        elem_nus = np.ones(n_vox) * nu
    else:
        elem_nus = nu
    cube = Cube(
        xlen, ylen, zlen, nx, ny, nz,
        elem_Es=np.ones(n_vox), elem_rhos=np.ones(n_vox),
        elem_nus=elem_nus, deg=degree)
    return cube

# Strain tensor
def epsilon(v):
    return sym(grad(v))

# Stress tensor
def sigma(v, E, nu):
    mu = E/2./(1+nu)
    lmbda = E*nu/(1+nu)/(1-2*nu)
    dim = v.geometric_dimension()
    return 2.0*mu*epsilon(v) + lmbda*tr(epsilon(v))*Identity(dim)

def element_stiffness_mat(cell_idx, xlen, ylen, zlen, nx, ny, nz, elem_nus, deg=1):
    mesh = BoxMesh.create(
        [Point(0., 0., 0.), Point(xlen, ylen, zlen)],
        [nx, ny, nz], CellType.Type.hexahedron)
    V = VectorFunctionSpace(mesh, 'Lagrange', degree=deg)
    u = TrialFunction(V)
    v = TestFunction(V)
    k_form = inner(sigma(v, 1, elem_nus[cell_idx]), epsilon(u))*dx
    K_local = assemble_local(k_form, Cell(mesh, cell_idx))
    
    # Map to a global (sparse) matrix.
    dofmap = V.dofmap()
    n_vts = len(V.tabulate_dof_coordinates())//3
    n_cell_dofs = len(K_local)
    cell_to_global_dof_map = dofmap.cell_dofs(cell_idx)
    global_i, global_j = [], []
    for i in range(n_cell_dofs):
        for j in range(n_cell_dofs):
            global_i.append(cell_to_global_dof_map[i])
            global_j.append(cell_to_global_dof_map[j])
    K_global = scipy.sparse.coo_matrix(
        (K_local.flatten(), (global_i, global_j)),
        shape=(n_vts*3, n_vts*3))
    return K_global, K_local.flatten(), global_i, global_j

def element_mass_mat(cell_idx, xlen, ylen, zlen, nx, ny, nz, deg=1):
    mesh = BoxMesh.create(
        [Point(0., 0., 0.), Point(xlen, ylen, zlen)],
        [nx, ny, nz], CellType.Type.hexahedron)
    V = VectorFunctionSpace(mesh, 'Lagrange', degree=deg)
    u = TrialFunction(V)
    v = TestFunction(V)
    
    m_form = 1 * dot(v, u) * dx
    M_local = assemble_local(m_form, Cell(mesh, cell_idx))
    
    # Map to a global (sparse) matrix.
    dofmap = V.dofmap()
    n_vts = len(V.tabulate_dof_coordinates())//3
    n_cell_dofs = len(M_local)
    cell_to_global_dof_map = dofmap.cell_dofs(cell_idx)
    global_i, global_j = [], []
    for i in range(n_cell_dofs):
        for j in range(n_cell_dofs):
            global_i.append(cell_to_global_dof_map[i])
            global_j.append(cell_to_global_dof_map[j])
    M_global = scipy.sparse.coo_matrix(
        (M_local.flatten(), (global_i, global_j)),
        shape=(n_vts*3, n_vts*3))
    return M_global, M_local.flatten(), global_i, global_j


class Cube:
    def __init__(self, xlen, ylen, zlen, nx, ny, nz,
                 elem_Es, elem_rhos, elem_nus, deg=1):
        """
        Initialize a Cube object, which is primarily defined by a
        FEniCS BoxMesh and mesh-element-wise material properties.

        Parameters
        ----------
        xlen, ylen, zlen: float
            The length of the box in the x, y, and z directions, respectively.
        nx, ny, nz: int
            The number of voxels in the x, y, and z directions, respectively.
        elem_Es: np.array or list
            The Young's modulus value of each voxel.
        elem_rhos: np.array or list
            The density value of each voxel.
        elem_nus: np.array or list
            The Poisson's ratio of each voxel.
        deg: int, default=1
            The mesh element order, i.e., degree of the Largrangian shape function.
        """
        n_elems = nx * ny * nz
        assert len(elem_Es) == n_elems
        assert len(elem_rhos) == n_elems
        assert len(elem_nus) == n_elems
        # Store properties.
        self.xlen, self.ylen, self.zlen = xlen, ylen, zlen
        self.nx, self.ny, self.nz = nx, ny, nz
        self.elem_Es, self.elem_rhos, self.elem_nus = elem_Es, elem_rhos, elem_nus
        self.degree = deg
        self.n_elems = n_elems
        
        # Mesh and vector function space.
        mesh = BoxMesh.create(
            [Point(0., 0., 0.), Point(xlen, ylen, zlen)],
            [nx, ny, nz], CellType.Type.hexahedron)
        V = VectorFunctionSpace(mesh, 'Lagrange', degree=deg)
        self.mesh = mesh
        self.V = V
        self.n_dofs = len(V.tabulate_dof_coordinates())
        self.n_vts = len(mesh.coordinates())
        
        # Boundary condition.
        def _bottom(x, on_boundary):
            return near(x[2], 0) and on_boundary
        bc = DirichletBC(V, Constant((0., 0., 0.)), _bottom)
        self.bc = bc
        self.bc_dofs = sorted(list(bc.get_boundary_values().keys()))
        self.nonbc_dofs = sorted(list(set(np.arange(self.n_dofs)) - set(self.bc_dofs)))
        
        # Element, voxel, and global stiffness matrices.
        self.element_stiffness_mats = self.get_stiffness_mats()
        # Mass matrix.
        self.element_mass_mats = self.get_mass_mats()
        
        # Exterior/interior DOFs.
        def _exterior(x, on_boundary):
            return on_boundary
        bc_ext = DirichletBC(V, Constant((0., 0., 0.)), _exterior)
        self.ext_dofs = sorted(list(bc_ext.get_boundary_values().keys()))
        self.int_dofs = sorted(list(set(np.arange(self.n_dofs)) - set(self.ext_dofs)))

        # Visible DOFs.
        def _inview(x):
            return (near(x[1], 0) or near(x[0], self.xlen)
                    or near(x[2], self.zlen))
        obs_dofs = [dof for (dof, coords) in enumerate(V.tabulate_dof_coordinates()) \
                    if _inview(coords)]
        obs_dofs = np.array(obs_dofs)
        self.obs_dofs = obs_dofs

        # DOFs in the order of the image-space mesh vertices.
        self.image_space_dofs = self.get_image_space_dofs()
        self.unseen_dofs = [dof for dof in range(self.n_dofs) \
            if dof not in self.image_space_dofs]

    def get_image_space_dofs(self):
        """
        Get DOFs in the order of the user-selected visible mesh vertices in
        image-space.

        Returns
        -------
        image_space_dofs: np.array
            A 1D numpy array of the visible DOFs, in the order in which they
            appear in image-space.
        """
        def _inview(x):
            return (near(x[1], 0) or near(x[0], self.xlen)
                    or near(x[2], self.zlen))

        mesh_points = self.mesh.coordinates()
        seen_mesh_points = np.array([pt for pt in mesh_points if _inview(pt)])

        # Sort visible mesh vertices in lexicographic order.
        sort_idxs = np.lexsort((
            seen_mesh_points[:,0], seen_mesh_points[:,1], 
            seen_mesh_points[:,2]))
        seen_mesh_points = seen_mesh_points[sort_idxs]

        # Order FEniCS DOFs in the order they appear in image-space.
        coords = self.V.tabulate_dof_coordinates()
        image_space_dofs = []
        for pt in seen_mesh_points:
            image_space_dofs += np.where(
                np.all(np.isclose(coords, pt), axis=1))[0].tolist()
        image_space_dofs = np.array(image_space_dofs)
        assert len(image_space_dofs) == len(seen_mesh_points) * 3
        return image_space_dofs

    def get_stiffness_mats(self):
        """
        Assemble and return the element stiffness matrices.

        Returns
        -------
        element_mats: list of scipy.sparse.coo_matrix's
            All of the element stiffness matrices. Each matrix, stored as a
            COO sparse matrix, is of size (N_DOFS x N_DOFS) and assembled
            according to the Young's modulus value of the voxel to which it
            belongs.
        """
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply(element_stiffness_mat,
                              args=(cell_idx, self.xlen, self.ylen, self.zlen,
                                    self.nx, self.ny, self.nz, 
                                    self.elem_nus, self.degree)) \
                   for cell_idx in tqdm(range(self.n_elems), desc='Element stiffness mats')]
        pool.close()
        
        element_mats = [res[0] for res in results]  # element stiffness mats
        return element_mats

    def get_mass_mats(self):
        """
        Assemble and return the element mass matrices.

        Returns
        -------
        element_mats: list of scipy.sparse.coo_matrix's
            All of the element mass matrices. Each matrix, stored as a
            COO sparse matrix, is of size (N_DOFS x N_DOFS) and assembled
            according to the density of the voxel to which it belongs.
        """
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply(element_mass_mat,
                              args=(cell_idx, self.xlen, self.ylen, self.zlen,
                                    self.nx, self.ny, self.nz, self.degree)) \
                   for cell_idx in tqdm(range(self.n_elems), desc='Element mass mats')]
        pool.close()
        
        element_mats = [res[0] for res in results]  # element mass mats
        return element_mats
    
    def solve_modes(self, K, M, apply_bc=True, neig=6):
        """
        Compute the eigensolutions of the generalized eigenvalue equation.

        Parameters
        ----------
        K: ndarray or scipy.sparse matrix
            The global stiffness matrix, of size (N_DOFS x N_DOFS).
        M: ndarray or scipy.sparse matrix
            The global mass matrix, of size (N_DOFS x N_DOFS).
        apply_bc: bool, default=True
            Whether to strictly apply the Dirichlet boundary condition by
            removing boundary DOFs from the equation.
        neig: int, default=6
            Number of eigensolutions to compute.

        Returns
        -------
        evals: np.array
            The list of eigenvalues, sorted in increasing order.
        evecs: np.ndarray of shape (N_DOFS, n_eig)
            The full-field eigenvectors, sorted in increasing eigenvalue order.
        """
        if apply_bc:
            K = K[self.nonbc_dofs, :][:, self.nonbc_dofs]
            M = M[self.nonbc_dofs, :][:, self.nonbc_dofs]
        w, v = scipy.sparse.linalg.eigsh(K, neig, M, sigma=1e-3)
        sort_idx = np.argsort(w)
        evals = w[sort_idx]
        evecs = v[:, sort_idx]
        if apply_bc:
            evecs_full = np.zeros((self.n_dofs, neig))
            evecs_full[self.nonbc_dofs] = evecs
            evecs = evecs_full
        return evals, evecs
        
    def layer_weights(self, weights, layer):
        """
        Retrieve the voxel values for a specific z-layer of the cube.

        Parameters
        ----------
        weights: np.ndarray
            A 1D numpy array containing the voxel values for the entire cube.
        layer: int
            The 0-indexed z-layer of which to retrieve the voxel values.

        Returns
        -------
        layer_weights: np.ndarray
            A 2D numpy array representing an image of the voxel values at the
            specified layer.
        """
        w3d = weights.reshape(self.nz, self.ny, self.nx)
        layer_weights = w3d[layer]
        return layer_weights
        
    def plot_2d(self, weights=None, vmin=None, vmax=None, cmap='viridis',
                cbar=True):
        """
        Plot a 2D visualization of the material-property values throughout
        the cube.

        Parameters
        ----------
        weights: np.ndarray
            A 1D numpy array containing the voxel-wise material-property values.
        vmin, vmax: float, optional
            The range of values to be covered by the colormap. By default,
            the colormap will cover the entire range of values in `weights`.
        cmap: str, default='viridis'
            A string identifier for the desired colormap.
        cbar: bool, default=True
            Whether to include the colorbar in the plot.
        """
        if weights is None:
            weights = self.elem_Es
        if vmin is None:
            vmin = np.min(weights)
        if vmax is None:
            vmax = np.max(weights)
        W = np.zeros((self.ny, self.nx * self.nz))
        for z in range(self.nz):
            layer = self.layer_weights(weights, z)
            W[:, z * self.nx : (z + 1) * self.nx] = layer
        norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)
        plt.imshow(W, origin='lower', cmap=cmap, norm=norm)
        if cbar:
            plt.colorbar(orientation='horizontal', ticks=[vmin, vmax])