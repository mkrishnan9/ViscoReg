"""
Phase-1 pretraining: fit the INR to a piecewise-linear function using plain MSE.

Keypoints are passed via --keypoints_x and --keypoints_y.
Default (W-shape):
  x : -0.5   -0.25    0.0   0.25    0.5
  u :  0.0   -0.25    0.0  -0.25    0.0

No SDF losses, no eikonal, no manifold constraint — pure regression.
Saves trained_models/model_final.pth when done.
"""
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
import torch.optim as optim
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import models.Net as Net


def get_args():
    p = argparse.ArgumentParser(description='Phase-1 pretrain: function fitting')
    p.add_argument('--logdir', type=str, required=True)
    p.add_argument('--gpu_idx', type=int, default=0)
    p.add_argument('--n_steps', type=int, default=50000,
                   help='number of gradient steps')
    p.add_argument('--batch_size', type=int, default=4096,
                   help='random samples per step')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--decoder_n_hidden_layers', type=int, default=1)
    p.add_argument('--decoder_hidden_dim', type=int, default=5)
    p.add_argument('--nl', type=str, default='sine')
    p.add_argument('--neuron_type', type=str, default='linear')
    p.add_argument('--init_type', type=str, default='siren')
    p.add_argument('--sphere_init_params', nargs='+', type=float,
                   default=[1.6, 0.1])
    p.add_argument('--keypoints_x', nargs='+', type=float,
                   default=[-0.5, -0.25, 0.0, 0.25, 0.5],
                   help='x-coordinates of target function keypoints (must be sorted)')
    p.add_argument('--keypoints_y', nargs='+', type=float,
                   default=[0.0, -0.25, 0.0, -0.25, 0.0],
                   help='y-values of target function keypoints')
    return p.parse_args()


args = get_args()

_KX = np.array(args.keypoints_x, dtype=np.float32)
_KY = np.array(args.keypoints_y, dtype=np.float32)
assert len(_KX) == len(_KY), "keypoints_x and keypoints_y must have the same length"

def target_fn(x_np):
    """Piecewise-linear interp; clips to boundary values outside the keypoint range."""
    return np.interp(x_np, _KX, _KY).astype(np.float32)

device = torch.device(f'cuda:{args.gpu_idx}' if torch.cuda.is_available() else 'cpu')

model_outdir = os.path.join(args.logdir, 'trained_models')
vis_dir = os.path.join(args.logdir, 'vis_pretrain')
os.makedirs(model_outdir, exist_ok=True)
os.makedirs(vis_dir, exist_ok=True)
os.system(f'cp {__file__} {args.logdir}')
os.system(f'cp ../models/Net.py {args.logdir}')

# dense eval grid for plotting
x_lo, x_hi = float(_KX[0]), float(_KX[-1])
x_plot = np.linspace(x_lo, x_hi, 1000).astype(np.float32)
y_target = target_fn(x_plot)
x_plot_t = torch.tensor(x_plot, dtype=torch.float32).reshape(1, -1, 1)

def save_plot(step):
    SINR.eval()
    with torch.no_grad():
        pred_np = SINR(x_plot_t.to(device), None)['nonmanifold_pnts_pred'].squeeze().cpu().numpy()
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x_plot, y_target, 'b--', linewidth=2, label='Target')
    ax.plot(x_plot, pred_np,  'r-',  linewidth=2, label='Network')
    ax.scatter(_KX, _KY, color='k', zorder=5, s=60, label='Keypoints')
    ax.set_xlabel('x')
    ax.set_ylabel('u(x)')
    ax.set_title(f'Pretrain step {step}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(vis_dir, f'step_{str(step).zfill(6)}.png'), dpi=120)
    plt.close(fig)

SINR = Net.Network(
    latent_size=0,
    in_dim=1,
    decoder_hidden_dim=args.decoder_hidden_dim,
    nl=args.nl,
    encoder_type='none',
    neuron_type=args.neuron_type,
    decoder_n_hidden_layers=args.decoder_n_hidden_layers,
    init_type=args.init_type,
    sphere_init_params=args.sphere_init_params,
)
SINR.to(device)

optimizer = optim.Adam(SINR.parameters(), lr=args.lr)

log_path = os.path.join(args.logdir, 'pretrain.log')
log_file = open(log_path, 'w')

rng = np.random.default_rng(42)

for step in range(args.n_steps):
    x_np = rng.uniform(x_lo, x_hi, size=(args.batch_size,)).astype(np.float32)
    y_np = target_fn(x_np)

    # network expects (bs=1, npoints, in_dim=1)
    x_t = torch.tensor(x_np, dtype=torch.float32, device=device).reshape(1, -1, 1)
    y_t = torch.tensor(y_np, dtype=torch.float32, device=device).reshape(1, -1)

    SINR.train()
    optimizer.zero_grad()
    pred = SINR(x_t, None)['nonmanifold_pnts_pred']  # (1, batch_size)
    loss = ((pred - y_t) ** 2).mean()
    loss.backward()
    optimizer.step()

    if step % 5000 == 0 or step == args.n_steps - 1:
        msg = f'Step {step:6d}/{args.n_steps}  MSE={loss.item():.8f}'
        print(msg)
        log_file.write(msg + '\n')
        log_file.flush()
        save_plot(step)

torch.save(SINR.state_dict(), os.path.join(model_outdir, 'model_final.pth'))
msg = f'Saved pretrained model → {model_outdir}/model_final.pth'
print(msg)
log_file.write(msg + '\n')
log_file.close()
