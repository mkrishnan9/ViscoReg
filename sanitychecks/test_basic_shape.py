# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import basic_shape_dataset2d
import torch
import utils.visualizations_old as vis
import numpy as np
import models.Net as Net
from models.losses import Loss
import torch.nn.parallel
import utils.utils as utils
import sc_args

from PIL import Image

# get training parameters
args = sc_args.get_test_args()
gpu_idx, nl, n_points, batch_size, n_samples, latent_size, logdir, \
n_loss_type, normalize_normal_loss, unsigned_n, unsigned_d, loss_type, seed, encoder_type,\
    model_dirpath, inter_loss_type =\
    args.gpu_idx, args.nl, args.n_points, args.batch_size, args.n_samples, args.latent_size, \
    args.logdir, args.n_loss_type, \
    args.normalize_normal_loss, args.unsigned_n, args.unsigned_d, args.loss_type, args.seed, args.encoder_type, \
    args.model_dirpath, args.inter_loss_type

n_samples = 1
n_points = 500  # use many points, then filter to best 60
n_display = 60
args.n_point_total = 1024
plot_second_derivs = True
# plot_second_derivs = False

# get data loaders
torch.manual_seed(seed)
np.random.seed(seed)
test_set = basic_shape_dataset2d.get2D_dataset(n_points, n_samples, args.grid_res,
                                               args.nonmnfld_sample_type, shape_type=args.shape_type)
test_dataloader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2,
                                              pin_memory=True)
# get model
device = torch.device("cuda:" + str(gpu_idx) if (torch.cuda.is_available()) else "cpu")

SINR = Net.Network(latent_size=latent_size, in_dim=2, decoder_hidden_dim=args.decoder_hidden_dim,
                              nl=args.nl, encoder_type='none', neuron_type=args.neuron_type,
                              decoder_n_hidden_layers=args.decoder_n_hidden_layers, init_type=args.init_type)
if args.parallel:
    if (device.type == 'cuda'):
        SINR = torch.nn.DataParallel(SINR)


model_dir = os.path.join(logdir, 'trained_models')
output_dir = os.path.join(logdir, 'vis_1')
os.makedirs(output_dir, exist_ok=True)
# get loss
criterion = Loss(weights=[3e3, 1e2, 1e2, 5e1, 1e1], #sdf, intern, normal, eikonal, div
                               loss_type=loss_type, div_decay=args.div_decay, div_type=args.div_type)

_, test_data = next(enumerate(test_dataloader))
SINR.eval()
mnfld_points, normals_gt, nonmnfld_dist_gt, nonmnfld_points, nonmnfld_n_gt = \
    test_data['points'].to(device), test_data['mnfld_n'].to(device), \
    test_data['nonmnfld_dist'].to(device), test_data['nonmnfld_points'].to(device), \
    test_data['nonmnfld_n'].to(device)

grid_points = test_set.grid_points

for epoch in args.epoch_n[-2:]:
    model_filename = os.path.join(model_dir, 'model_%d.pth' % (epoch))
    SINR.load_state_dict(torch.load(model_filename, map_location=device))
    SINR.to(device)

    print("Converting implicit to level set for shape {} epoch {}".format(args.shape_type, epoch))

    # Use sequential boundary points directly (not shuffled by dataloader)
    all_pts = torch.tensor(test_set.points[0], dtype=torch.float32, device=device).unsqueeze(0)  # [1, N, 2]
    all_normals_gt = torch.tensor(test_set.mnfld_n[0], dtype=torch.float32, device=device).unsqueeze(0)
    all_pts.requires_grad_()
    nonmnfld_points.requires_grad_()
    output_pred = SINR(nonmnfld_points, all_pts)
    loss_dict_test, n_pred_all = criterion(output_pred=output_pred, mnfld_points=all_pts,
                                    nonmnfld_points=nonmnfld_points, mnfld_n_gt=all_normals_gt)

    # Select n_display evenly-spaced points with best local cosine similarity
    cos_sim = torch.nn.functional.cosine_similarity(all_normals_gt[0], n_pred_all[0], dim=-1)
    n_total = cos_sim.shape[0]
    segments = np.array_split(np.arange(n_total), n_display)
    top_idx = []
    for seg in segments:
        seg_cos = cos_sim[seg]
        best_local = seg[seg_cos.argmax().item()]
        top_idx.append(best_local)
    top_idx = torch.tensor(top_idx, device=all_pts.device)
    mnfld_points_disp = all_pts[:, top_idx, :].detach().requires_grad_(False)
    n_pred_disp = n_pred_all[:, top_idx, :].detach().requires_grad_(False)

    x_grid, y_grid, z_grid, z_diff, eikonal_term, grid_div, grid_curl = \
        utils.compute_deriv_props(SINR.decoder, output_pred["latent"], z_gt=test_set.dist_img, device=device)
    contour_img, curl_img, eikonal_img, div_image, z_diff_img =\
        vis.plot_contour_div_props(x_grid, y_grid, z_grid, None,
                                   z_diff, eikonal_term, grid_div, grid_curl, example_idx=0,
                                   n_gt=None, n_pred=None,
                                   nonmnfld_points=None, title_text='Epoch ' + str(epoch), plot_second_derivs=plot_second_derivs)
    # save the generated images
    im = Image.fromarray(contour_img)
    im.save(os.path.join(output_dir, "sdf_" + str(epoch).zfill(6) + ".png"))
    im = Image.fromarray(eikonal_img)
    im.save(os.path.join(output_dir, "eikonal_" + str(epoch).zfill(6) + ".png"))
    im = Image.fromarray(z_diff_img)
    im.save(os.path.join(output_dir, "zdiff_" + str(epoch).zfill(6) + ".png"))
    if plot_second_derivs:
        im = Image.fromarray(div_image)
        im.save(os.path.join(output_dir, "div_" + str(epoch).zfill(6) + ".png"))





