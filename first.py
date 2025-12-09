import torch
import torch.nn as nn
from typing import List, Tuple


# ---------- time B-spline machinery ----------

def make_full_knots(knot_positions: torch.Tensor, degree: int) -> torch.Tensor:
    k0 = knot_positions[0]
    kN = knot_positions[-1]
    left = k0.repeat(degree)
    right = kN.repeat(degree)
    return torch.cat([left, knot_positions, right])


def bspline_basis(ages: torch.Tensor,
                  knots_full: torch.Tensor,
                  degree: int) -> torch.Tensor:
    ages = ages.view(-1)  # (B,)
    B = ages.shape[0]
    K = knots_full.shape[0]
    n_basis = K - degree - 1
    assert n_basis > 0

    N = torch.zeros(B, K - 1, dtype=ages.dtype, device=ages.device)
    tau = ages.view(B, 1)

    # N_{i,0}
    for i in range(K - 1):
        left = knots_full[i]
        right = knots_full[i + 1]
        if i < K - 2:
            mask = (tau >= left) & (tau < right)
        else:
            mask = (tau >= left) & (tau <= right)
        N[:, i] = mask.float().view(B)

    # Cox–de Boor recursion
    for p in range(1, degree + 1):
        N_new = torch.zeros(B, K - p - 1, dtype=ages.dtype, device=ages.device)
        for i in range(K - p - 1):
            left = knots_full[i]
            mid = knots_full[i + p]
            right = knots_full[i + 1]
            far = knots_full[i + p + 1]

            term1 = torch.zeros(B, 1, dtype=ages.dtype, device=ages.device)
            term2 = torch.zeros(B, 1, dtype=ages.dtype, device=ages.device)

            if mid > left:
                term1 = ((tau - left) / (mid - left)) * N[:, i].view(B, 1)
            if far > right:
                term2 = ((far - tau) / (far - right)) * N[:, i + 1].view(B, 1)

            N_new[:, i] = (term1 + term2).view(B)
        N = N_new

    return N  # (B, n_basis)


class TimeVaryingBSplineWeights(nn.Module):
    """
    Time-varying weights a_ℓ(τ) on an arbitrary lattice.

    grid_shape: shape of lattice indices (e.g. (K, Nx, Ny, Nz))
    knot_positions: interior time knots, strictly increasing
    """

    def __init__(self,
                 grid_shape: Tuple[int, ...],
                 knot_positions: List[float],
                 degree: int = 3,
                 init: str = "randn"):
        super().__init__()

        interior = torch.as_tensor(knot_positions, dtype=torch.float32)
        assert interior.ndim == 1
        assert torch.all(interior[1:] > interior[:-1])

        self.degree = degree
        self.grid_shape = tuple(grid_shape)

        knots_full = make_full_knots(interior, degree)
        self.register_buffer("knots_full", knots_full)

        self.num_basis = knots_full.shape[0] - degree - 1
        if self.num_basis <= 0:
            raise ValueError("Not enough knots for given degree")

        coeffs = torch.zeros(*self.grid_shape, self.num_basis, dtype=torch.float32)
        if init == "randn":
            nn.init.normal_(coeffs, mean=0.0, std=0.01)
        self.coeffs = nn.Parameter(coeffs)  # (*grid_shape, num_basis_time)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: scalar or (B,)
        returns: (B, *grid_shape) of a_ℓ(t)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=self.coeffs.dtype, device=self.coeffs.device)

        t = t.to(self.knots_full.device).view(-1)  # (B,)
        B = t.shape[0]

        basis = bspline_basis(t, self.knots_full, self.degree)    # (B, P)
        coeffs_flat = self.coeffs.view(-1, self.num_basis)        # (L, P)
        values_flat = basis @ coeffs_flat.t()                     # (B, L)
        values = values_flat.view(B, *self.grid_shape)            # (B, *grid_shape)
        return values

    def temporal_smoothing_loss(self, lam: float) -> torch.Tensor:
        c = self.coeffs.view(-1, self.num_basis)  # (L, P)
        if c.shape[1] < 3:
            return torch.tensor(0.0, device=c.device, dtype=c.dtype)
        d2 = c[:, :-2] - 2 * c[:, 1:-1] + c[:, 2:]
        return lam * (d2 ** 2).mean()

    def spatial_smoothing_loss(self, lam: float, spatial_ndims: int) -> torch.Tensor:
        """
        Bending-like penalty over spatial dimensions of coeffs; ignores last (time-basis) dim.

        spatial_ndims: how many last-but-one dims are spatial (excluding class / channels and time-basis).
                       e.g. if coeffs shape = (K, Nx, Ny, Nz, P), spatial_ndims = 3.
        """
        c = self.coeffs  # shape: (..., *spatial, P)
        if spatial_ndims <= 0:
            return torch.tensor(0.0, device=c.device, dtype=c.dtype)

        # indices of spatial dims
        spatial_axes = list(range(c.ndim - 1 - spatial_ndims, c.ndim - 1))
        spatial_sizes = [c.shape[d] for d in spatial_axes]
        if any(s < 3 for s in spatial_sizes):
            return torch.tensor(0.0, device=c.device, dtype=c.dtype)

        loss = 0.0
        count = 0

        for ax, size_d in zip(spatial_axes, spatial_sizes):
            idx_mid = slice(1, size_d - 1)
            idx_prev = slice(0, size_d - 2)
            idx_next = slice(2, size_d)

            slicer_mid = [slice(None)] * c.ndim
            slicer_prev = [slice(None)] * c.ndim
            slicer_next = [slice(None)] * c.ndim
            slicer_mid[ax] = idx_mid
            slicer_prev[ax] = idx_prev
            slicer_next[ax] = idx_next

            c_prev = c[tuple(slicer_prev)]
            c_mid = c[tuple(slicer_mid)]
            c_next = c[tuple(slicer_next)]
            d2 = c_prev - 2.0 * c_mid + c_next
            loss = loss + (d2 ** 2).mean()
            count += 1

        return lam * (loss / max(count, 1))


CUBIC_B_SPLINE_MATRIX = (1.0 / 6.0) * torch.tensor(
    [[1.,  4.,  1.,  0.],
     [-3., 0.,  3.,  0.],
     [3., -6.,  3.,  0.],
     [-1., 3., -3.,  1.]]
)

def cubic_bspline_weights(u: torch.Tensor) -> torch.Tensor:
    M = CUBIC_B_SPLINE_MATRIX.to(u.device, u.dtype)
    U = torch.stack(
        [torch.ones_like(u), u, u * u, u * u * u],
        dim=-1,
    )  # (..., 4)
    return U @ M  # (..., 4)


def spatial_cubic_bspline_interpolate(weights: torch.Tensor,
                                      coords: torch.Tensor) -> torch.Tensor:
    """
    weights: (Nx, Ny[, Nz[, Nt]]) scalar lattice (D = len(weights.shape)).
    coords:  (..., D) in lattice index space [0..N_d-1]
    returns: (...,)
    """
    spatial_shape = weights.shape
    D = len(spatial_shape)
    assert 2 <= D <= 4

    device = weights.device
    dtype = weights.dtype
    weights_flat = weights.reshape(-1)

    sizes = torch.tensor(spatial_shape, device=device, dtype=torch.long)
    strides = torch.empty(D, dtype=torch.long, device=device)
    strides[-1] = 1
    for d in range(D - 2, -1, -1):
        strides[d] = strides[d + 1] * sizes[d + 1]

    coords = coords.to(device=device, dtype=dtype)
    orig_shape = coords.shape[:-1]
    coords_flat = coords.reshape(-1, D)  # (P, D)
    P = coords_flat.shape[0]

    floor_vals = torch.floor(coords_flat)
    i0 = floor_vals.long() - 1
    u = coords_flat - floor_vals

    grids = torch.meshgrid(
        *[torch.arange(4, device=device) for _ in range(D)],
        indexing="ij"
    )
    offsets = torch.stack([g.reshape(-1) for g in grids], dim=-1)  # (K, D)
    K = offsets.shape[0]

    w_per_dim = []
    for d in range(D):
        w_d = cubic_bspline_weights(u[:, d])     # (P,4)
        w_dg = w_d[:, offsets[:, d]]             # (P,K)
        w_per_dim.append(w_dg)

    w_all = torch.ones(P, K, device=device, dtype=dtype)
    for w_dg in w_per_dim:
        w_all = w_all * w_dg

    i0_exp = i0.unsqueeze(1)        # (P,1,D)
    off_exp = offsets.unsqueeze(0)  # (1,K,D)
    j = i0_exp + off_exp
    j = torch.clamp(j, min=0, max=(sizes - 1))

    linear_idx = (j * strides.view(1, 1, D)).sum(dim=-1)  # (P,K)
    neighbor_vals = weights_flat[linear_idx]              # (P,K)

    vals = (neighbor_vals * w_all).sum(dim=-1)            # (P,)
    return vals.view(*orig_shape)

class SpatioTemporalLogitField(nn.Module):
    """
    Multi-resolution spatio-temporal logit field:

      f_k(y, τ) = sum_{res} [ f_static,res,k(y) + f_dynamic,res,k(y, τ) ]

    where each term is represented by cubic B-spline bases in space,
    and dynamic terms also by B-splines in time.

    Args:
      num_classes: K
      spatial_lattice_shapes: list of spatial lattice shapes per resolution,
          e.g. [(8,8,8), (16,16,16), (32,32,32)]
      knot_positions: time knot positions (interior), shared for all resolutions
      degree_time: temporal spline degree (3 = cubic)
    """

    def __init__(self,
                 num_classes: int,
                 spatial_lattice_shapes: List[Tuple[int, ...]],
                 knot_positions: List[float],
                 degree_time: int = 3,
                 init: str = "randn"):
        super().__init__()
        self.num_classes = num_classes
        self.spatial_lattice_shapes = spatial_lattice_shapes
        self.num_resolutions = len(spatial_lattice_shapes)

        # per-resolution modules / params
        self.dynamic = nn.ModuleList()
        self.static = nn.ParameterList()

        for shape in spatial_lattice_shapes:
            grid_shape = (num_classes, *shape)  # class dim + spatial dims

            # dynamic (time-dependent) part
            dyn = TimeVaryingBSplineWeights(
                grid_shape=grid_shape,
                knot_positions=knot_positions,
                degree=degree_time,
                init=init,
            )
            self.dynamic.append(dyn)

            # static (time-independent) part
            stat_param = nn.Parameter(torch.zeros(num_classes, *shape))
            if init == "randn":
                nn.init.normal_(stat_param, mean=0.0, std=0.01)
            self.static.append(stat_param)

    def forward(self, t: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
        """
        t: scalar or (T,) tensor of ages
        coords: (..., D) spatial coords in index space of the **image** grid
                (same coords for all resolutions / classes).

        Returns:
          if t scalar: (..., K)
          if t (T,):  (T, ..., K)
        """
        if not torch.is_tensor(t):
            t = torch.tensor(t, dtype=self.static[0].dtype, device=self.static[0].device)

        scalar_time = (t.ndim == 0)
        t_vec = t.view(-1)  # (T,)
        T = t_vec.shape[0]

        device = self.static[0].device
        coords = coords.to(device=device)

        out_per_t = []

        for ti in range(T):
            tau = t_vec[ti:ti+1]          # (1,)
            logits_t = None              # (..., K)

            # Sum over resolutions
            for res_idx, shape in enumerate(self.spatial_lattice_shapes):
                dyn_module = self.dynamic[res_idx]
                stat = self.static[res_idx]          # (K, *shape)

                # dynamic lattice at this time: (1, K, *shape)
                dyn_lattice = dyn_module(tau)[0]     # (K, *shape)

                # combined lattice weights per class
                lattice = stat + dyn_lattice         # (K, *shape)

                # spatial dimension D
                D = len(shape)

                # Evaluate for each class and stack
                class_logits = []
                for k in range(self.num_classes):
                    vals_k = spatial_cubic_bspline_interpolate(
                        lattice[k],  # (*shape)
                        coords       # (..., D)
                    )               # (...,)
                    class_logits.append(vals_k)

                logits_res = torch.stack(class_logits, dim=-1)  # (..., K)
                if logits_t is None:
                    logits_t = logits_res
                else:
                    logits_t = logits_t + logits_res

            out_per_t.append(logits_t)

        out = torch.stack(out_per_t, dim=0)  # (T, ..., K)
        if scalar_time:
            out = out[0]                      # (..., K)
        return out

    # ----- regularisation -----

    def temporal_smoothing_loss(self, lam_time: float) -> torch.Tensor:
        loss = 0.0
        for dyn, shape in zip(self.dynamic, self.spatial_lattice_shapes):
            spatial_ndims = len(shape)  # exclude class dim and time-basis dim
            loss = loss + dyn.temporal_smoothing_loss(lam_time)
        return loss

    def spatial_smoothing_loss(self, lam_space: float) -> torch.Tensor:
        """
        Spatial smoothing on BOTH static and dynamic lattices.
        """
        device = self.static[0].device
        dtype = self.static[0].dtype
        total = torch.tensor(0.0, device=device, dtype=dtype)

        # dynamic (uses helper inside dyn)
        for dyn, shape in zip(self.dynamic, self.spatial_lattice_shapes):
            spatial_ndims = len(shape)
            total = total + dyn.spatial_smoothing_loss(lam_space, spatial_ndims)

        # static: second-order finite differences over spatial dims
        for stat, shape in zip(self.static, self.spatial_lattice_shapes):
            c = stat  # (K, *shape)
            D = len(shape)
            if any(s < 3 for s in shape):
                continue

            for d, size_d in enumerate(shape):
                axis = 1 + d   # skip class dim
                idx_mid = slice(1, size_d - 1)
                idx_prev = slice(0, size_d - 2)
                idx_next = slice(2, size_d)

                slicer_mid = [slice(None)] * c.ndim
                slicer_prev = [slice(None)] * c.ndim
                slicer_next = [slice(None)] * c.ndim
                slicer_mid[axis] = idx_mid
                slicer_prev[axis] = idx_prev
                slicer_next[axis] = idx_next

                c_prev = c[tuple(slicer_prev)]
                c_mid = c[tuple(slicer_mid)]
                c_next = c[tuple(slicer_next)]
                d2 = c_prev - 2.0 * c_mid + c_next
                total = total + lam_space * (d2 ** 2).mean()

        return total

    def total_smoothing_loss(self, lam_time: float, lam_space: float) -> torch.Tensor:
        return self.temporal_smoothing_loss(lam_time) + self.spatial_smoothing_loss(lam_space)


import torch.nn.functional as F

# Example config
num_classes = 4
spatial_lattice_shapes = [(8, 8, 8), (16, 16, 16), (32, 32, 32)]
knot_positions = [0., 1., 2., 4., 8., 16.]

model = SpatioTemporalLogitField(
    num_classes=num_classes,
    spatial_lattice_shapes=spatial_lattice_shapes,
    knot_positions=knot_positions,
    degree_time=3,
    init="randn",
)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Suppose one subject:
# label_vol: (Z, Y, X) with integer labels 0..K-1
# age: scalar float
def training_step(label_vol: torch.Tensor, age: float,
                  lam_time: float = 1e-3,
                  lam_space: float = 1e-4):

    model.train()
    optimizer.zero_grad()

    Z, Y, X = label_vol.shape
    device = next(model.parameters()).device
    label_vol = label_vol.to(device)

    # Build voxel coordinate grid in index space
    zz, yy, xx = torch.meshgrid(
        torch.arange(Z, device=device),
        torch.arange(Y, device=device),
        torch.arange(X, device=device),
        indexing="ij",
    )
    coords = torch.stack([zz, yy, xx], dim=-1)  # (Z, Y, X, 3)

    # Forward: logits at this age
    logits = model(age, coords)          # (Z, Y, X, K)

    # Cross-entropy expects (N, K) logits and (N,) labels
    logits_flat = logits.view(-1, num_classes)      # (N_vox, K)
    labels_flat = label_vol.view(-1)                # (N_vox,)

    ce_loss = F.cross_entropy(logits_flat, labels_flat)

    smooth_loss = model.total_smoothing_loss(lam_time, lam_space)

    loss = ce_loss + smooth_loss
    loss.backward()
    optimizer.step()

    return loss.item(), ce_loss.item(), smooth_loss.item()
