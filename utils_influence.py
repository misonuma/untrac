import torch
from tqdm import tqdm
import pdb


def hvp_fn(trainer, vectors, debug=False):
    hvps_sum = None
    data_size = 0

    for i in range(trainer.args.gradient_accumulation_steps):
        try:
            inputs = next(trainer.train_dataloader)
        except:
            trainer.train_dataloader = iter(trainer.get_train_dataloader())
            inputs = next(trainer.train_dataloader)

        if debug: pdb.set_trace()
        inputs = trainer._prepare_inputs(inputs)        
        loss = trainer.compute_loss(trainer.model, inputs)

        grads = torch.autograd.grad(loss, trainer.model.parameters(), create_graph=True)
        hvps = torch.autograd.grad(grads, trainer.model.parameters(), grad_outputs=vectors)
        del grads

        if hvps_sum is None:
            hvps_sum = hvps
        else:
            hvps_sum = add_scalar_multiple(hvps_sum, batch_size, hvps)
        del hvps

        batch_size = len(inputs["input_ids"])
        data_size += batch_size

    hvps_sum = multiply_by_scalar(1/data_size, hvps_sum)
    return hvps_sum


def lissa(trainer, vectors, debug=False):
    decay_rate = trainer.args.decay_rate
    inv_steps = trainer.args.inv_steps
    
    inv_hvps = tuple(vector.clone().detach() for vector in vectors)

    for _ in range(inv_steps):
        hvps = hvp_fn(trainer, vectors, debug)
        
        assert len(inv_hvps) == len(vectors) == len(hvps)
        for inv_hvp, vector, hvp in zip(inv_hvps, vectors, hvps):
            vector -= decay_rate*hvp
            inv_hvp += vector
        del hvps

    inv_hvps = tuple(decay_rate*inv_hvp for inv_hvp in inv_hvps)
    return inv_hvps


def arnoldi_iter(trainer, n_iters, norm_constant=1.0, stop_tol=1e-5):
    trainer.model.eval()
    start_vector = tuple(torch.randn_like(param) for param in trainer.model.parameters())
    
    proj = []
    appr_mat = torch.zeros((n_iters, n_iters-1), dtype=trainer.model.dtype, device=trainer.model.device)
    
    v0_norm = norm_fn(start_vector)
    vec0 = multiply_by_scalar(norm_constant / v0_norm, start_vector)
    if trainer.args.arnoldi_cpu: 
        vec_cpu = to_cpu(vec0)
        proj.append(vec_cpu)
    else:
        proj.append(vec0)
    del start_vector

    for n in tqdm(range(n_iters - 1)):
        vec = hvp_fn(trainer, vec0)
        
        for j, proj_vec in enumerate(proj):
            if trainer.args.arnoldi_cpu: proj_vec = to_gpu(proj_vec, trainer.model.device)
            appr_mat[j, n] = dot_product_fn(vec, proj_vec) / norm_constant**2
            vec = add_scalar_multiple(vec, -appr_mat[j, n], proj_vec)

        new_norm = norm_fn(vec)

        # Early termination if the Krylov subspace is invariant within the tolerance.
        if new_norm < stop_tol:
            appr_mat[n+1, n] = 0
            vec = tuple(torch.zeros_like(v) for v in vec)
            proj.append(vec)
            break

        appr_mat[n+1, n] = new_norm / norm_constant
        vec0 = multiply_by_scalar(1.0 / appr_mat[n+1, n], vec)

        if trainer.args.arnoldi_cpu: 
            vec_cpu = to_cpu(vec0)
            proj.append(vec_cpu)
        else:
            proj.append(vec0)
        
    if trainer.args.arnoldi_cpu: appr_mat = appr_mat.cpu()
    return appr_mat, proj


def arnoldi_distill(
    appr_mat,
    proj,
    top_k,
    init_fn=torch.zeros_like,
    force_hermitian=True,
    log_progress=False,
):
    appr_mat = appr_mat[:-1, :]
    n = appr_mat.shape[0]

    if force_hermitian:
        # Make appr_mat Hermitian and tridiagonal when force_hermitian=True.
        for i in range(n):
            for j in range(n):
                if i - j > 1 or j - i > 1:
                    appr_mat[i, j] = 0
        # Make Hermitian.
        appr_mat = .5 * (appr_mat + appr_mat.T)
        # Get eigenvalues / vectors for Hermitian matrix.
        eigvals, eigvecs = torch.linalg.eigh(appr_mat)
    else:
        eigvals, eigvecs = torch.linalg.eig(appr_mat)

    # Sort the eigvals by absolute value.
    idx = torch.argsort(torch.abs(eigvals))
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]

    # Note we need to discard the last projector as this is a correction term.
    reduced_projections = change_basis_of_projections(
            eigvecs[:, -top_k:],
            proj[:-1],
            init_fn=init_fn,
            log_progress=log_progress)

    return eigvals[-top_k:], reduced_projections


# +
def to_cpu(vec):
    return [v.cpu() for v in vec]

def to_gpu(vec, device):
    return [v.to(device) for v in vec]

def dot_product_fn(vec1, vec2):
    return sum(torch.sum(v1*v2) for v1, v2 in zip(vec1, vec2))

def norm_fn(vec):
    return torch.sqrt(sum(torch.sum(v**2) for v in vec))

def add_scalar_multiple(vec1, scalar, vec2):
    return tuple(v1+v2*scalar for v1, v2 in zip(vec1, vec2))

def multiply_by_scalar(scalar, vec):
    return tuple(v*scalar for v in vec)

def change_basis_of_projections(
        matrix,
        proj,
        init_fn = torch.zeros_like,
        log_progress=False
    ):
    
    if matrix.shape[0] != len(proj):
        raise ValueError('Incompatible composition')

    out = []
    for j in tqdm(range(matrix.shape[1])):
        if log_progress:
            logging.info('Compose projections: j=%d', j)
        element = tuple(init_fn(tensor) for tensor in proj[0])

        for i in range(matrix.shape[0]):
            if log_progress:
                logging.info('Compose projections: i,j=%d,%d', i, j)
            element = add_scalar_multiple(element, matrix[i, j], proj[i])

        out.append(element)
    return out
