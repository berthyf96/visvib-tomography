import multiprocessing as mp

import numpy as np
from dolfin import *
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


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
    K_global = coo_matrix((K_local.flatten(), (global_i, global_j)),
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
    M_global = coo_matrix((M_local.flatten(), (global_i, global_j)),
                           shape=(n_vts*3, n_vts*3))
    return M_global, M_local.flatten(), global_i, global_j


class Cube:
    def __init__(self, xlen, ylen, zlen, nx, ny, nz,
                 elem_Es, elem_rhos, elem_nus, deg=1):
        '''
        Params:
            xlen (float): Length of box along x-axis.
            ylen (float): Length of box along y-axis.
            zlen (float): Length of box along z-axis.
            nx (int): Number of cells in x direction.
            ny (int): Number of cells in y direction.
            nz (int): Number of cells in z direction.
            elem_Es (1d array): Young's modulus for each cell.
            elem_rhos (1d array): Density for each cell.
            elem_nus (1d array): Poisson's ratio for each cell.
            deg (int): Degree of Lagrangian shape function.
        '''
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
        '''Get DOFs in the order of the user-selected visible mesh
        vertices in image.space'''
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
        '''Assemble and return element stiffness matrices.'''
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
        '''Assemble and return element mass matrices.'''
        pool = mp.Pool(mp.cpu_count())
        results = [pool.apply(element_mass_mat,
                              args=(cell_idx, self.xlen, self.ylen, self.zlen,
                                    self.nx, self.ny, self.nz, self.degree)) \
                   for cell_idx in tqdm(range(self.n_elems), desc='Element mass mats')]
        pool.close()
        
        element_mats = [res[0] for res in results]  # element mass mats
        return element_mats
    
    def solve_modes(self, K, M, apply_bc=True, neig=6):
        if apply_bc:
            K = K[self.nonbc_dofs,:][:,self.nonbc_dofs]
            M = M[self.nonbc_dofs,:][:,self.nonbc_dofs]
        w, v = eigsh(K, neig, M, sigma=1e-3)
        sort_idx = np.argsort(w)
        evals = w[sort_idx]
        evecs = v[:, sort_idx]
        if apply_bc:
            evecs_full = np.zeros((self.n_dofs, neig))
            evecs_full[self.nonbc_dofs] = evecs
            evecs = evecs_full
        return evals, evecs
        
    def layer_weights(self, weights, layer):
        w3d = weights.reshape(self.nz, self.ny, self.nx)
        layer = w3d[layer]
        return layer
        
    def plot_2d(self, weights=None, wmin=None, wmax=None, cmap='viridis',
                cbar=True):
        if weights is None:
            weights = self.elem_Es
        if wmin is None:
            wmin = weights.min()
        if wmax is None:
            wmax = weights.max()
        W = np.zeros((self.ny, self.nx*self.nz))
        for z in range(self.nz):
            layer = self.layer_weights(weights, z)
            W[:, z*self.nx:(z+1)*self.nx] = layer
        norm = Normalize(vmin=wmin, vmax=wmax)
        plt.imshow(W, origin='lower', cmap=cmap, norm=norm)
        if cbar:
            plt.colorbar(orientation='horizontal', ticks=[wmin, wmax])