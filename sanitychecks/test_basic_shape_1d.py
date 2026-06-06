import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basic_shape_dataset1d
import torch
import numpy as np
import models.Net as Net
import sc_args
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

# get training parameters
args = sc_args.get_test_args()
gpu_idx, logdir = args.gpu_idx, args.logdir

device = torch.device("cuda:" + str(gpu_idx) if torch.cuda.is_available() else "cpu")

torch.manual_seed(args.seed)
np.random.seed(args.seed)

test_set = basic_shape_dataset1d.Segment1D(
    n_points=args.n_points, n_samples=1, res=args.grid_res,
    sample_type=args.nonmnfld_sample_type
)

SINR = Net.Network(latent_size=args.latent_size, in_dim=1,
                   decoder_hidden_dim=args.decoder_hidden_dim,
                   nl=args.nl, encoder_type='none', neuron_type=args.neuron_type,
                   decoder_n_hidden_layers=args.decoder_n_hidden_layers,
                   init_type=args.init_type)

model_dir = os.path.join(logdir, 'trained_models')
output_dir = os.path.join(logdir, 'vis_1')
os.makedirs(output_dir, exist_ok=True)

# 1D evaluation grid (dense), same range as training grid
grid_range = test_set.grid_range
x_eval = np.linspace(-grid_range, grid_range, 1000).astype(np.float32)
gt_sdf = np.abs(x_eval - test_set.center) - test_set.half_len

# boundary points tensor: [1, 2, 1]
bnd_pts = torch.tensor(test_set.points[0], dtype=torch.float32, device=device).unsqueeze(0)
# evaluation points: [1, 1000, 1]
x_tensor = torch.tensor(x_eval[:, None], dtype=torch.float32, device=device).unsqueeze(0)

for epoch in args.epoch_n:
    model_filename = os.path.join(model_dir, 'model_%d.pth' % epoch)
    SINR.load_state_dict(torch.load(model_filename, map_location=device))
    SINR.to(device)
    SINR.eval()

    print("Evaluating epoch {}".format(epoch))

    with torch.no_grad():
        output_pred = SINR(x_tensor, bnd_pts)
        f_pred = output_pred['nonmanifold_pnts_pred'].squeeze().cpu().numpy()  # [1000]

    def make_plot(xlim, suffix):
        plt.rcParams.update({'font.size': 22})
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(x_eval, gt_sdf, 'b--', linewidth=2, label='GT SDF')
        ax.plot(x_eval, f_pred, 'r-',  linewidth=2, label='Predicted SDF')
        ax.scatter([test_set.left, test_set.right], [0, 0],
                   color='k', marker='v', s=80, zorder=5, label='Boundary')
        ax.axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.set_xlim(*xlim)
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.set_title('1D Neural SDF')
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(output_dir, '{}_{}.png'.format(suffix, str(epoch).zfill(6)))
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print("Saved {}".format(out_path))

    make_plot((-grid_range, grid_range), 'sdf')
    make_plot((test_set.left, test_set.right), 'sdf_interior')

# ── Evolution plot: all checkpoints in trained_models/ on one figure ──────────
all_ckpts = sorted(
    [f for f in os.listdir(model_dir) if f.startswith('model_') and f.endswith('.pth')],
    key=lambda f: int(f.split('_')[1].split('.')[0])
)
all_epochs = [int(f.split('_')[1].split('.')[0]) for f in all_ckpts]

x_interior = np.linspace(test_set.left, test_set.right, 1000).astype(np.float32)
gt_interior = np.abs(x_interior - test_set.center) - test_set.half_len
x_int_t = torch.tensor(x_interior[:, None], dtype=torch.float32, device=device).unsqueeze(0)

colors = plt.cm.plasma(np.linspace(0.05, 0.95, len(all_epochs)))

plt.rcParams.update({'font.size': 14})
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_interior, gt_interior, 'k--', linewidth=2.5, label='GT SDF', zorder=10)

for i, (ckpt, ep, color) in enumerate(zip(all_ckpts, all_epochs, colors)):
    SINR.load_state_dict(torch.load(os.path.join(model_dir, ckpt), map_location=device))
    SINR.eval()
    with torch.no_grad():
        pred = SINR(x_int_t, bnd_pts)['nonmanifold_pnts_pred'].squeeze().cpu().numpy()

    if i == 0:
        c, lw, alpha, label, zorder = color,      3.5, 1.0, 'Initial',  5
    elif i == len(all_ckpts) - 1:
        c, lw, alpha, label, zorder = 'tab:green', 3.5, 1.0, 'Final',    5
    else:
        c, lw, alpha, label, zorder = color,      1.2, 0.7,  None,       2
    ax.plot(x_interior, pred, color=c, linewidth=lw, alpha=alpha, label=label, zorder=zorder)

ax.scatter([test_set.left, test_set.right], [0, 0],
           color='k', zorder=11, s=60, marker='v', label='Boundary')
ax.axhline(0, color='gray', linewidth=0.7, alpha=0.5)
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.set_title('Evolution of predicted SDF on [{}, {}]'.format(test_set.left, test_set.right))

sm = plt.cm.ScalarMappable(cmap='plasma', norm=plt.Normalize(vmin=min(all_epochs), vmax=max(all_epochs)))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)
cbar.set_label('Training step')
ax.legend()
fig.tight_layout()

evo_path = os.path.join(logdir, 'evolution_interior.png')
fig.savefig(evo_path, dpi=150)
plt.close(fig)
print("Saved {}".format(evo_path))

# ── Eikonal / manifold loss curves ────────────────────────────────────────────
os.system('python3 {} {}'.format(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), 'plot_eikonal.py'),
    logdir))
