import torch.utils.data as data
import numpy as np
import torch


class Segment1D(data.Dataset):
    """
    1D neural SDF dataset: the 'shape' is a closed interval [left, right].
    Boundary (manifold) points are the two endpoints.
    Outward normals: -1 at the left endpoint, +1 at the right endpoint.
    GT SDF: f(x) = |x - center| - half_len  (positive outside, negative inside).
    """
    def __init__(self, n_points, n_samples=128, res=256, sample_type='grid',
                 endpoints=(-0.5, 0.5), nonmnfld_range=None):
        self.left, self.right = float(endpoints[0]), float(endpoints[1])
        self.center = (self.left + self.right) / 2.0
        self.half_len = (self.right - self.left) / 2.0
        self.n_points = n_points        # non-manifold points sampled per batch item
        self.n_samples = n_samples
        self.grid_res = res

        # ---- manifold: 2 endpoints, shape [n_samples, 2, 1] ----
        pts = np.array([[self.left], [self.right]], dtype=np.float32)  # [2, 1]
        self.points = np.tile(pts[None], (n_samples, 1, 1))            # [n_samples, 2, 1]

        # outward normals in 1D
        normals = np.array([[-1.0], [1.0]], dtype=np.float32)          # [2, 1]
        self.mnfld_n = np.tile(normals[None], (n_samples, 1, 1))       # [n_samples, 2, 1]

        # ---- non-manifold grid ----
        if nonmnfld_range is not None:
            grid_range = float(nonmnfld_range)
        else:
            grid_range = max(1.2, max(abs(self.left), abs(self.right)) * 1.5)
        x = np.linspace(-grid_range, grid_range, res).astype(np.float32)
        self.grid_points = x[:, None]                                  # [res, 1]
        self.grid_range = grid_range

        # GT signed distance
        self.nonmnfld_dist = (np.abs(x - self.center) - self.half_len).astype(np.float32)

        # GT gradient direction (outward normal field)
        gt_n = np.sign(x - self.center).astype(np.float32)
        gt_n[gt_n == 0] = 1.0
        self.nonmnfld_n = gt_n[:, None]                                # [res, 1]

        # for compatibility with utils.compute_deriv_props (not used in 1D test)
        self.dist_img = self.nonmnfld_dist

        if sample_type != 'grid':
            raise ValueError(f"Unsupported sample_type for 1D: {sample_type}")
        self.nonmnfld_points = self.grid_points

        self.point_idxs = np.array([0, 1])
        self.nonmnfld_points_idxs = np.arange(res)
        self.sample_probs = np.ones(res, dtype=np.float32) / res

        self.generate_batch_indices()

    def generate_batch_indices(self):
        # manifold: always both endpoints
        self.mnfld_idx = np.tile(np.array([0, 1]), (self.n_samples, 1))
        # non-manifold: sample n_points from the grid
        self.nonmnfld_idx = np.stack([
            np.random.choice(self.nonmnfld_points_idxs, self.n_points)
            for _ in range(self.n_samples)
        ])

    def __getitem__(self, index):
        midx = self.mnfld_idx[index]
        nidx = self.nonmnfld_idx[index]
        return {
            'points':          self.points[index, midx, :],      # [2, 1]
            'mnfld_n':         self.mnfld_n[index, midx, :],     # [2, 1]
            'nonmnfld_dist':   self.nonmnfld_dist[nidx],         # [n_points]
            'nonmnfld_n':      self.nonmnfld_n[nidx],            # [n_points, 1]
            'nonmnfld_points': self.nonmnfld_points[nidx],       # [n_points, 1]
        }

    def __len__(self):
        return self.n_samples
