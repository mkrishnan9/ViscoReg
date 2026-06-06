# This file is partly based on DiGS: https://github.com/Chumbyte/DiGS

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.utils as utils

def eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

    return eikonal_term


# def dw_eikonal_loss(nonmnfld_grad, mnfld_grad, eikonal_type='abs'):
#     # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
#     # shape is (bs, num_points, dim=3) for both grads
#     # Eikonal
#     if nonmnfld_grad is not None and mnfld_grad is not None:
#         all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
#     elif nonmnfld_grad is not None:
#         all_grads = nonmnfld_grad
#     elif mnfld_grad is not None:
#         all_grads = mnfld_grad

#     if eikonal_type == 'abs':
#         eik_ls = all_grads.norm(2, dim=2)
#         eikonal_term = ((all_grads.norm(2, dim=2) - 1).abs()).mean()
#     else:
#         eikonal_term = ((all_grads.norm(2, dim=2) - 1).square()).mean()

#     return eikonal_term


def visc_eikonal_loss(nonmnfld_grad, mnfld_grad, laplace, eps, eikonal_type='abs'):
    # Compute the eikonal loss that penalises when ||grad(f)|| != 1 for points on and off the manifold
    # shape is (bs, num_points, dim=3) for both grads
    # Eikonal
    if nonmnfld_grad is not None and mnfld_grad is not None:
        all_grads = torch.cat([nonmnfld_grad, mnfld_grad], dim=-2)
    elif nonmnfld_grad is not None:
        all_grads = nonmnfld_grad
    elif mnfld_grad is not None:
        all_grads = mnfld_grad

    if eikonal_type == 'abs':
        eikonal_term = ((all_grads.norm(2, dim=2) - 1 -eps*laplace).abs()).mean()
    else:
        eikonal_term = ((all_grads.norm(2, dim=2) - 1 -eps*laplace).square()).mean()

    return eikonal_term

def latent_rg_loss(latent_reg, device):
    # compute the VAE latent representation regularization loss
    if latent_reg is not None:
        reg_loss = latent_reg.mean()
    else:
        reg_loss = torch.tensor([0.0], device=device)

    return reg_loss


def directional_div(points, grads):
    dot_grad = (grads * grads).sum(dim=-1, keepdim=True)
    hvp = torch.ones_like(dot_grad)
    hvp = 0.5 * torch.autograd.grad(dot_grad, points, hvp, retain_graph=True, create_graph=True)[0]
    div = (grads * hvp).sum(dim=-1) / (torch.sum(grads ** 2, dim=-1) + 1e-5)
    return div

def full_div(points, grads):
    dx = utils.gradient(points, grads[:, :, 0])
    if points.shape[-1] == 1:
        div = dx[:, :, 0]
    elif points.shape[-1] == 3:
        dy = utils.gradient(points, grads[:, :, 1])
        dz = utils.gradient(points, grads[:, :, 2])
        div = dx[:, :, 0] + dy[:, :, 1] + dz[:, :, 2]
    else:
        dy = utils.gradient(points, grads[:, :, 1])
        div = dx[:, :, 0] + dy[:, :, 1]
    div[div.isnan()] = 0
    return div

class Loss(nn.Module):
    def __init__(self, weights=[3e3, 1e2, 1e2, 5e1, 1e2], loss_type='siren', div_decay='none', div_type='dir_l1',adaptive_epsilon_params=None):
        super().__init__()
        self.weights = weights #sdf, intern, normal, eikonal, div
        self.loss_type = loss_type
        self.div_decay = div_decay
        self.div_type = div_type
        self.use_div = True if 'div' in self.loss_type else False

        self.viscosity_weight_idx = 4


        if self.div_decay == 'adaptive':
            if adaptive_epsilon_params is None:
                # Provide some sensible defaults if not given
                self.adaptive_epsilon_params = {
                    'initial_epsilon': 0.4,     # Starting value for the baseline decay
                    'gamma_decay_factor': 0.01, # Target reduction factor (e.g., aims for 1% of initial_epsilon by end)
                    'ema_beta': 0.99,           # Smoothing factor for Eikonal loss EMA
                    'sensitivity_S': 1.0,       # Scales the Eikonal loss contribution
                    'epsilon_min': 0.0          # Minimum value for epsilon
                }
            else:
                self.adaptive_epsilon_params = {
                    'initial_epsilon': adaptive_epsilon_params[0],     # Starting value for the baseline decay
                    'gamma_decay_factor': adaptive_epsilon_params[1], # Target reduction factor (e.g., aims for 1% of initial_epsilon by end)
                    'ema_beta': adaptive_epsilon_params[2],           # Smoothing factor for Eikonal loss EMA
                    'sensitivity_S': adaptive_epsilon_params[3],       # Scales the Eikonal loss contribution
                    'epsilon_min': adaptive_epsilon_params[4]          # Minimum value for epsilon
                }

            # print(self.adaptive_epsilon_params)


            self.current_ema_eikonal_residual = 0.0



    def forward(self, output_pred, mnfld_points, nonmnfld_points, mnfld_n_gt=None):
        dims = mnfld_points.shape[-1]
        device = mnfld_points.device

        #########################################
        # Compute required terms
        #########################################

        non_manifold_pred = output_pred["nonmanifold_pnts_pred"]
        manifold_pred = output_pred["manifold_pnts_pred"]
        latent_reg = output_pred["latent_reg"]
        latent = output_pred["latent"]

        div_loss = torch.tensor([0.0], device=mnfld_points.device)

        # compute gradients for div (divergence), curl and curv (curvature)
        if manifold_pred is not None:
            mnfld_grad = utils.gradient(mnfld_points, manifold_pred)
        else:
            mnfld_grad = None

        nonmnfld_grad = utils.gradient(nonmnfld_points, non_manifold_pred)

        # div_term
        if self.use_div and self.weights[4] > 0.0:

            if self.div_type == 'full_l2':
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.clamp(torch.square(nonmnfld_divergence), 0.1, 50)
            elif self.div_type == 'full_l1':
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.clamp(torch.abs(nonmnfld_divergence), 0.1, 50)
            elif self.div_type == 'dir_l2':
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.square(nonmnfld_divergence)
            elif self.div_type == 'dir_l1':
                nonmnfld_divergence = directional_div(nonmnfld_points, nonmnfld_grad)
                nonmnfld_divergence_term = torch.abs(nonmnfld_divergence)
            else:
                raise Warning("unsupported divergence type. only supports dir_l1, dir_l2, full_l1, full_l2")

            div_loss = nonmnfld_divergence_term.mean() #+ mnfld_divergence_term.mean()

        # eikonal term

        eikonal_term = eikonal_loss(nonmnfld_grad, mnfld_grad=mnfld_grad, eikonal_type='abs')

        # latent regulariation for multiple shape learning
        latent_reg_term = latent_rg_loss(latent_reg, device)

        # normal term
        if mnfld_n_gt is not None:
            if 'igr' in self.loss_type:
                normal_term = ((mnfld_grad - mnfld_n_gt).abs()).norm(2, dim=1).mean()
            else:
                normal_term = (1 - torch.abs(torch.nn.functional.cosine_similarity(mnfld_grad, mnfld_n_gt, dim=-1))).mean()
        else:
            normal_term = torch.tensor(0.0).to(device)

        # signed distance function term
        sdf_term = torch.abs(manifold_pred).mean()

        # inter term
        inter_term = torch.exp(-1e2 * torch.abs(non_manifold_pred)).mean()

        #########################################
        # Losses
        #########################################

        # losses used in the paper
        if self.loss_type == 'siren': # SIREN loss
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'siren_wo_n': # SIREN loss without normal constraint
            self.weights[2] = 0
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'igr': # IGR loss
            self.weights[1] = 0
            loss = self.weights[0]*sdf_term + self.weights[2]*normal_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'igr_wo_n': # IGR without normals loss
            self.weights[1] = 0
            self.weights[2] = 0
            loss = self.weights[0]*sdf_term + self.weights[3]*eikonal_term
        elif self.loss_type == 'siren_w_div': # SIREN loss with divergence term
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + \
                   self.weights[2]*normal_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_wo_n_w_div':  # SIREN loss without normals and with divergence constraint
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term + \
                   self.weights[4] * div_loss
        elif self.loss_type == 'siren_wo_n_w_visc':  # SIREN loss without normals and with divergence constraint
            if self.weights[4] > 0:
                nonmnfld_divergence = full_div(nonmnfld_points, nonmnfld_grad)
                visc_eikonal_term = visc_eikonal_loss(nonmnfld_grad, mnfld_grad=None, laplace=nonmnfld_divergence, eps=self.weights[4] , eikonal_type='abs')
                eikonal_term0 = visc_eikonal_term+ eikonal_loss(None, mnfld_grad=mnfld_grad, eikonal_type='abs')
            else:
                eikonal_term0 = eikonal_term
            loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*eikonal_term0
        # elif self.loss_type == 'siren_wo_n_w_dw':
        #     loss = self.weights[0]*sdf_term + self.weights[1] * inter_term + self.weights[3]*dw_eikonal_term
        else:
            raise Warning("unrecognized loss type")



        # If multiple surface reconstruction, then latent and latent_reg are defined so reg_term need to be used
        if latent is not None and latent_reg is not None:
            loss += self.weights[5] * latent_reg_term

        return {"loss": loss, 'sdf_term': sdf_term, 'inter_term': inter_term, 'latent_reg_term': latent_reg_term,
                'eikonal_term': eikonal_term.detach(), 'normals_loss': normal_term, 'div_loss': div_loss
                }, mnfld_grad

    def update_div_weight(self, current_iteration, n_iterations, params=None):
        # `params`` should be (start_weight, *optional middle, end_weight) where optional middle is of the form [percent, value]*
        # Thus (1e2, 0.5, 1e2 0.7 0.0, 0.0) means that the weight at [0, 0.5, 0.75, 1] of the training process, the weight should
        #   be [1e2,1e2,0.0,0.0]. Between these points, the weights change as per the div_decay parameter, e.g. linearly, quintic, step etc.
        #   Thus the weight stays at 1e2 from 0-0.5, decay from 1e2 to 0.0 from 0.5-0.75, and then stays at 0.0 from 0.75-1.

        if not hasattr(self, 'decay_params_list'):
            assert len(params) >= 2, params
            assert len(params[1:-1]) % 2 == 0
            self.decay_params_list = list(zip([params[0], *params[1:-1][1::2], params[-1]],[0, *params[1:-1][::2], 1]))

        curr = current_iteration / n_iterations
        we, e = min([tup for tup in self.decay_params_list if tup[1]>= curr], key=lambda tup: tup[1])
        w0, s = max([tup for tup in self.decay_params_list if tup[1]<= curr], key=lambda tup: tup[1])

        # Divergence term anealing functions
        if self.div_decay == 'linear': # linearly decrease weight from iter s to iter e
            if current_iteration < s*n_iterations:
                self.weights[4] = w0
            elif  current_iteration >= s*n_iterations and current_iteration < e*n_iterations:
                self.weights[4] = w0 + (we - w0) * (current_iteration/n_iterations - s) / (e - s)
            else:
                self.weights[4] = we
        elif self.div_decay == 'quintic': # linearly decrease weight from iter s to iter e
            if current_iteration < s*n_iterations:
                self.weights[4] = w0
            elif  current_iteration >= s*n_iterations and current_iteration < e*n_iterations:
                self.weights[4] = w0 + (we - w0) * (1 - (1 -(current_iteration/n_iterations - s) / (e - s))**5)
            else:
                self.weights[4] = we
        elif self.div_decay == 'step': # change weight at s
            if current_iteration < s*n_iterations:
                self.weights[4] = w0
            else:
                self.weights[4] = we
        elif self.div_decay == 'none':
            pass
        else:
            raise Warning("unsupported div decay value")

    def update_epsilon_schedule(self, current_iteration, n_iterations,
                                current_pure_eikonal_residual_batch=None,
                                params=None): # Existing params for fixed schedules

        if self.div_decay == 'adaptive':
            if current_pure_eikonal_residual_batch is None:
                raise ValueError("current_pure_eikonal_residual_batch is required for adaptive decay.")

            adapt_params = self.adaptive_epsilon_params

            # Update EMA of Eikonal residual
            self.current_ema_eikonal_residual = (adapt_params['ema_beta'] * self.current_ema_eikonal_residual +
                                                (1 - adapt_params['ema_beta']) * current_pure_eikonal_residual_batch.item()) # .item() to get scalar


            if current_iteration < 0.6*n_iterations:
                # Calculate baseline target decay
                progress = current_iteration / (0.6*n_iterations)
                # Using initial_epsilon * (gamma_factor)^(progress)
                eps_target_decay = adapt_params['initial_epsilon'] * (adapt_params['gamma_decay_factor'] ** progress)
                # Calculate adaptive epsilon
                adaptive_component = adapt_params['sensitivity_S'] * self.current_ema_eikonal_residual
                new_epsilon = max(adapt_params['epsilon_min'], eps_target_decay + adaptive_component)
            else:
                new_epsilon = 0.0


            # if current_iteration % 100 ==0:
            #     print(new_epsilon)
            #     print(current_pure_eikonal_residual_batch.item())

            self.weights[self.viscosity_weight_idx] = new_epsilon

        elif self.div_decay in ['linear', 'quintic', 'step']:
            if not hasattr(self, 'decay_params_list') or self._last_params_div_decay != params : # Re-init if params changed
                 assert params is not None and len(params) >= 2, params
                 assert len(params[1:-1]) % 2 == 0
                 self.decay_params_list = list(zip([params[0]] + list(params[1:-1][1::2]) + [params[-1]],
                                                   [0.0] + list(params[1:-1][::2]) + [1.0]))
                 self._last_params_div_decay = params


            curr_progress_ratio = current_iteration / n_iterations

            # Find relevant segments for interpolation
            # Segment start point (w_start, p_start)
            w_start, p_start = max([tup for tup in self.decay_params_list if tup[1] <= curr_progress_ratio],
                                   key=lambda tup: tup[1])
            # Segment end point (w_end, p_end) - find the *next* point after p_start
            # Filter points greater than p_start, then find the one with min percentage
            future_points = [tup for tup in self.decay_params_list if tup[1] > p_start]
            if not future_points: # We are at or past the last defined point
                w_end, p_end = self.decay_params_list[-1] # Use the last point
            else:
                w_end, p_end = min(future_points, key=lambda tup: tup[1])

            if self.div_decay == 'linear':
                if curr_progress_ratio <= p_start : # Before or at the start of a new segment
                    self.weights[self.viscosity_weight_idx] = w_start
                elif curr_progress_ratio >= p_end: # At or after the end of current segment
                    self.weights[self.viscosity_weight_idx] = w_end
                else: # Interpolate
                    # Ensure p_end > p_start to avoid division by zero if points are too close
                    segment_duration_ratio = p_end - p_start
                    if segment_duration_ratio > 1e-9: # Effectively non-zero duration
                        progress_in_segment = (curr_progress_ratio - p_start) / segment_duration_ratio
                        self.weights[self.viscosity_weight_idx] = w_start + (w_end - w_start) * progress_in_segment
                    else: # Segment has zero effective duration, jump to w_end if we are at p_end
                         self.weights[self.viscosity_weight_idx] = w_end if curr_progress_ratio >= p_end else w_start

            elif self.div_decay == 'quintic':
                if curr_progress_ratio <= p_start:
                    self.weights[self.viscosity_weight_idx] = w_start
                elif curr_progress_ratio >= p_end:
                     self.weights[self.viscosity_weight_idx] = w_end
                else:
                    segment_duration_ratio = p_end - p_start
                    if segment_duration_ratio > 1e-9:
                        progress_in_segment = (curr_progress_ratio - p_start) / segment_duration_ratio
                        interp_factor = 1 - (1 - progress_in_segment)**5
                        self.weights[self.viscosity_weight_idx] = w_start + (w_end - w_start) * interp_factor
                    else:
                         self.weights[self.viscosity_weight_idx] = w_end if curr_progress_ratio >= p_end else w_start

            elif self.div_decay == 'step':
                self.weights[self.viscosity_weight_idx] = w_start # Value changes when p_start is crossed

        elif self.div_decay == 'none':
            pass # Epsilon (self.weights[4]) remains constant as initially set
        else:
            raise Warning(f"unsupported epsilon decay type: {self.div_decay}")

        # For logging the current epsilon
        return self.weights[self.viscosity_weight_idx]