"""
Metrics computation script for ScutSurf benchmark
Uses official ScutSurf evaluation metrics: Chamfer-L2, F-score, Normal Consistency, NFS
"""
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import surface_recon_args

# Import official ScutSurf vanilla metrics
from vanilla_metrics import eval_pointcloud

args = surface_recon_args.get_train_args()

# Import official ScutSurf neural metrics only if needed
if False:
    import torch
    import torch.nn.functional as F
    sys.path.append('/fs/nexus-scratch/mkrishn9/SCUTSurface-code/metrics/neural_metric')
    import utils as neural_utils
    from network import test_net256
    from tqdm import tqdm

# Setup paths
recon_mesh_path = args.results_path  # Path to reconstructed meshes
gt_path = os.path.join(args.dataset_path, "real_gt")  # Ground truth .xyz files


def compute_nfs(neural_model, pred_ply, gt_xyz, voxel_size=0.1, point_per_patch=8000):
    """
    Compute Neural Feature Similarity (NFS) using pretrained neural network.
    Based on official ScutSurf neural metric implementation.
    """
    try:
        # Load and prepare point clouds
        gt_xyzn = neural_utils.load_xyz(gt_xyz)
        pointcloud, normals = neural_utils.sampleGT(pred_ply, samplepointsnum=len(gt_xyzn))
        pred_xyzn = np.concatenate((pointcloud, normals), axis=1)

        # Normalize to unit scale
        scale = np.abs(gt_xyzn[:, :3]).max()
        gt_xyzn[:, :3] = gt_xyzn[:, :3] / scale
        pred_xyzn[:, :3] = pred_xyzn[:, :3] / scale

        # Voxelize for patch-based comparison
        test_pairs = neural_utils.voxelize_for_test_overlap(pred_xyzn, gt_xyzn, voxel_size)

        # Compute similarity for each patch
        similarities = []
        for batch in test_pairs:
            pred_pts = batch['pred_pts']
            gt_pts = batch['gt_pts']

            if (len(pred_pts) == 0) or (len(gt_pts) == 0):
                similarities.append(0.0)
            else:
                # Sample points from patch
                pred_choice = np.random.choice(len(pred_pts), point_per_patch, replace=True)
                pred_pts_sampled = pred_pts[pred_choice]
                gt_choice = np.random.choice(len(gt_pts), point_per_patch, replace=True)
                gt_pts_sampled = gt_pts[gt_choice]

                # Convert to tensors
                pred_tensor = torch.FloatTensor(pred_pts_sampled).view(1, -1, 6).cuda()
                gt_tensor = torch.FloatTensor(gt_pts_sampled).view(1, -1, 6).cuda()

                # Extract features
                with torch.no_grad():
                    feat_pred = neural_model(pred_tensor)
                    feat_gt = neural_model(gt_tensor)

                    # Compute cosine similarity
                    score = torch.abs(F.cosine_similarity(feat_pred.view(1, -1), feat_gt.view(1, -1))).sum()
                    similarities.append(score.detach().cpu().numpy())

        # Return averaged score
        neural_score = np.sum(np.array(similarities)) / len(test_pairs)
        return neural_score

    except Exception as e:
        print(f"Error computing NFS: {e}")
        return None

# Print metrics to file in logdir as well
os.makedirs(args.logdir, exist_ok=True)  # Ensure directory exists
out_path = os.path.join(args.logdir, 'metric_summary.txt')
import builtins as __builtin__
def print(*args, **kwargs):
    # Override print function to also print to file
    __builtin__.print(*args, **kwargs)
    with open(out_path, 'a') as fp:
        __builtin__.print(*args, file=fp, **kwargs)

print("=" * 80)
print("ScutSurf Benchmark Evaluation")
if False:
    print("Using Official ScutSurf Metrics (Vanilla + Neural)")
else:
    print("Using Official ScutSurf Metrics (Vanilla only)")
print("=" * 80)
print(f"Reconstruction meshes: {recon_mesh_path}")
print(f"Ground truth: {gt_path}")
print()

# Load pretrained neural network for NFS computation
neural_model = None
if False:
    print("Loading pretrained neural network for NFS computation...")
    model_path = '/fs/nexus-scratch/mkrishn9/SCUTSurface-code/metrics/neural_metric/save/T197_Scaled_net256_Adam/epoch-1000.pth'
    neural_model = test_net256(point_dim=6, gf_dim=256).cuda()
    checkpoint = torch.load(model_path)
    neural_model.load_state_dict(checkpoint['model_sd'])
    neural_model.eval()
    print("Neural network loaded successfully!")
    print()

# All ScutSurf shapes (scan files have _pcd suffix, GT files don't)
shapes = [
    'bottle_shampoo', 'bowl_chinese', 'cloth_duck',
    'coffe_bottle_metal', 'coffe_bottle_plastic', 'cup1',
    'flower_pot_2', 'flower_pot', 'gift_box',
    'lock_pengfen', 'marker', 'mouse_two',
    'rabbit', 'romoter', 'screwnew',
    'tap2', 'toy_cat', 'toy_duck',
    'wrench', 'xiaojiejie2'
]




# Store all results for aggregation
all_results = []
successful_evaluations = []

if False:
    print(f"{'Shape':<25} {'CD-L2':<10} {'F@0.5':<10} {'NormCons':<10} {'NFS':<10} {'Status':<15}")
    print("-" * 90)
else:
    print(f"{'Shape':<25} {'CD-L2':<10} {'F@0.5':<10} {'NormCons':<10} {'Status':<15}")
    print("-" * 80)

for shape in shapes:
    # Paths
    recon_file = os.path.join(recon_mesh_path, f'{shape}_pcd.ply')
    gt_file = os.path.join(gt_path, f'{shape}.xyz')

    # Check if files exist
    if not os.path.exists(recon_file):
        if False:
            print(f'{shape:<25} {"---":<10} {"---":<10} {"---":<10} {"---":<10} {"Missing recon":<15}')
        else:
            print(f'{shape:<25} {"---":<10} {"---":<10} {"---":<10} {"Missing recon":<15}')
        continue

    if not os.path.exists(gt_file):
        if False:
            print(f'{shape:<25} {"---":<10} {"---":<10} {"---":<10} {"---":<10} {"Missing GT":<15}')
        else:
            print(f'{shape:<25} {"---":<10} {"---":<10} {"---":<10} {"Missing GT":<15}')
        continue

    try:
        # Evaluate using official ScutSurf vanilla metrics
        # For real_obj, sample points = number of GT points (determined automatically)
        eval_dict = eval_pointcloud(recon_file, gt_file, samplepoint=200000, eval_type='real_obj')

        # Extract key vanilla metrics
        chamfer_l2 = eval_dict['chamfer-L2']
        fscore_05 = eval_dict['f-score-5']  # F-score at 0.5 threshold (for real objects)
        normal_cons = eval_dict['normals']

        # Compute Neural Feature Similarity (NFS) if requested
        nfs_score = None
        if False:
            nfs_score = compute_nfs(neural_model, recon_file, gt_file, voxel_size=0.1, point_per_patch=8000)

        # Store full results
        result_entry = {
            'shape': shape,
            'nfs': nfs_score,
            **eval_dict
        }
        all_results.append(result_entry)
        successful_evaluations.append(shape)

        # Print row
        if False:
            nfs_str = f'{nfs_score:<10.4f}' if nfs_score is not None else f'{"---":<10}'
            print(f'{shape:<25} {chamfer_l2:<10.5f} {fscore_05:<10.4f} {normal_cons:<10.4f} {nfs_str} {"OK":<15}')
        else:
            print(f'{shape:<25} {chamfer_l2:<10.5f} {fscore_05:<10.4f} {normal_cons:<10.4f} {"OK":<15}')

    except Exception as e:
        if False:
            print(f'{shape:<25} {"---":<10} {"---":<10} {"---":<10} {"---":<10} {f"Error: {str(e)[:10]}":<15}')
        else:
            print(f'{shape:<25} {"---":<10} {"---":<10} {"---":<10} {f"Error: {str(e)}":<15}')
        continue

print()
print("=" * 80)
print("SUMMARY STATISTICS")
print("=" * 80)

if len(all_results) > 0:
    print(f"Successfully evaluated: {len(all_results)}/{len(shapes)} shapes")
    print()

    # Define metrics to aggregate
    metric_names = {
        'chamfer-L2': 'Chamfer-L2 Distance',
        'f-score-5': 'F-Score @ 0.5',
        'normals': 'Normal Consistency',
        'nfs': 'Neural Feature Similarity (NFS)',
        'CD_Acc': 'CD Accuracy (pred→GT)',
        'CD_Comp': 'CD Completeness (GT→pred)',
        'N_Acc': 'Normal Accuracy',
        'N_Comp': 'Normal Completeness',
        'F_Acc_5': 'F-Score Precision @ 0.5',
        'F_Comp_5': 'F-Score Recall @ 0.5',
    }

    print(f"{'Metric':<35} {'Mean':<12} {'Median':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-" * 95)

    for metric_key, metric_name in metric_names.items():
        values = [r[metric_key] for r in all_results if metric_key in r]
        if len(values) > 0:
            values_np = np.array(values)
            mean_val = np.mean(values_np)
            median_val = np.median(values_np)
            std_val = np.std(values_np)
            min_val = np.min(values_np)
            max_val = np.max(values_np)

            print(f'{metric_name:<35} {mean_val:<12.5f} {median_val:<12.5f} {std_val:<12.5f} {min_val:<12.5f} {max_val:<12.5f}')

    print()
    print("=" * 80)
    print("KEY BENCHMARK METRICS (ScutSurf Official)")
    print("=" * 80)

    # Highlight the main metrics used in ScutSurf benchmark
    chamfer_values = np.array([r['chamfer-L2'] for r in all_results])
    fscore_values = np.array([r['f-score-5'] for r in all_results])
    normal_values = np.array([r['normals'] for r in all_results])
    nfs_values = np.array([r['nfs'] for r in all_results if r['nfs'] is not None])

    print(f"Chamfer-L2:          {np.mean(chamfer_values):.5f} ± {np.std(chamfer_values):.5f}")
    print(f"F-Score @ 0.5:       {np.mean(fscore_values):.4f} ± {np.std(fscore_values):.4f}")
    print(f"Normal Consistency:  {np.mean(normal_values):.4f} ± {np.std(normal_values):.4f}")
    if False and len(nfs_values) > 0:
        print(f"NFS:                 {np.mean(nfs_values):.4f} ± {np.std(nfs_values):.4f}")
    print()

    print("Per-shape results:")
    if False:
        print("-" * 90)
        for shape in successful_evaluations:
            result = next(r for r in all_results if r['shape'] == shape)
            nfs_str = f"{result['nfs']:.4f}" if result['nfs'] is not None else "---"
            print(f"{shape:<25} CD-L2: {result['chamfer-L2']:.5f}  F@0.5: {result['f-score-5']:.4f}  NC: {result['normals']:.4f}  NFS: {nfs_str}")
    else:
        print("-" * 80)
        for shape in successful_evaluations:
            result = next(r for r in all_results if r['shape'] == shape)
            print(f"{shape:<25} CD-L2: {result['chamfer-L2']:.5f}  F@0.5: {result['f-score-5']:.4f}  NC: {result['normals']:.4f}")

else:
    print("No shapes were successfully evaluated!")
    print("Please check:")
    print(f"  1. Reconstruction meshes in: {recon_mesh_path}")
    print(f"  2. Ground truth files in: {gt_path}")

print()
print("=" * 80)
print("Evaluation complete!")
print(f"Results saved to: {out_path}")
print("=" * 80)
