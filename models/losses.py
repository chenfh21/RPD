import math
import pdb
from typing import List, Optional

import torch
from torch import nn


# ----------------------------------------------CROSS ENTROPY-------------------------------------
# nn.functional.cross_entropy()


# def cross_entropy(
#     input: Tensor,
#     target: Tensor,
#     weight: Optional[Tensor] = None,
#     size_average: Optional[bool] = None,
#     ignore_index: int = -100,
#     reduce: Optional[bool] = None,
#     reduction: str = "mean",
#     label_smoothing: float = 0.0,
# ) -> Tensor:

class CrossEntropy(nn.Module):
    def __init__(self, weights: Optional[List] = None):
        super(CrossEntropy, self).__init__()

        if weights is not None:
            self.weights = torch.Tensor(weights)
        else:
            self.weights = None

    def forward(self, inputs: torch.Tensor, target: torch.Tensor, mode: str,
                mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Compute cross entropy loss.

        Args:
            inputs(torch.Tensor): Unnormalized input tensor (logits) of shape [B x C x H x W]
            target(torch.Tensor): Ground-truth target tensor of shape [B x H x W]
            mode(str): train, val, or test
            mask_keep(Optional[torch.Tensor], optional): Mask of pixel of shape [B x H x W] which should be kept during
                                                loss computation (1: =keep, 0 :=ignore). Default to None := keep all.

        Returns:
            torch.Tensor: loss value (scalar)
        """
        assert mode in ['train', 'val', 'test']

        if mask_keep is not None:
            target[mask_keep == False] = 0

        # get the number classes and device
        batch_size, num_classes, height, width = inputs.shape
        input_device = inputs.device

        # convert logits to softmax probabilities
        probs = nn.functional.softmax(inputs, dim=1)  # [N x n_classes x H x W]
        del inputs

        # apply one-hot encoding to ground truth annotations
        target_one_hot = to_one_hot(target, int(num_classes))  # [N x n_classes x H x W]
        target_one_hot = target_one_hot.bool()
        del target

        # prepare to ignore certain pixels which should not be considered during loss computation
        if mask_keep is None:
            # consider all pixels to compute the loss
            mask_keep = torch.ones((batch_size, 1, height, width), dtype=torch.bool,
                                   device=input_device)  # [N x 1 x H x W]
        else:
            # get the dimension correctly
            mask_keep = mask_keep.unsqueeze(1)  # [N x 1 x H x W]

        # set ignore pixels to false
        target_one_hot = target_one_hot * mask_keep

        # gather the predicted probabilities of each ground truth category
        probs_gathered = probs[target_one_hot]  # M = N * (H * W) entries

        # make sure that probs are numerically stable when passed to log function: log(0) -> inf
        probs_gathered = torch.clip(probs_gathered, 1e-12, 1.0)

        # compute loss
        losses = -torch.log(probs_gathered)  # M = N * (H * W) entries
        del probs_gathered

        assert losses.shape[0] == torch.sum(mask_keep)
        del mask_keep

        # create weight matrix
        if self.weights is not None:
            if input_device != self.weights.device:
                self.weights = self.weights.to(input_device)

            weight_matrix = (target_one_hot.permute(0, 2, 3, 1) * self.weights).permute(0, 3, 1, 2)
            weight_gathered = weight_matrix[target_one_hot]
            assert torch.all(weight_gathered > 0)

            # compute weighted loss for each prediction
            losses *= weight_gathered

        return torch.mean(losses)


# -------------------------------------Generalized-Jensen-Shannon Divergence -------------------------------------------
def gjs_div_loss(p1_logits: torch.Tensor, p2_logits: torch.Tensor, p3_logits: torch.Tensor) -> torch.Tensor:
    p1_probs = nn.functional.softmax(p1_logits, dim=1)  # [BxCxHxW]
    p2_probs = nn.functional.softmax(p2_logits, dim=1)
    p3_probs = nn.functional.softmax(p3_logits, dim=1)

    m_probs = (p1_probs + p2_probs + p3_probs) / 3.0  # [B x C x H x W]
    m_probs = torch.clamp(m_probs, 1e-7, 1.0).log()

    loss1 = nn.functional.kl_div(input=m_probs, target=p1_probs, reduction='none', log_target=False)  # [B x C x H x W]
    loss1 = torch.sum(loss1, dim=1)  # [B x H x W]

    loss2 = nn.functional.kl_div(input=m_probs, target=p2_probs, reduction='none', log_target=False)  # [B x C x H x W]
    loss2 = torch.sum(loss2, dim=1)  # [B x H x W]

    loss3 = nn.functional.kl_div(input=m_probs, target=p3_probs, reduction='none', log_target=False)  # [B x C x H x W]
    loss3 = torch.sum(loss3, dim=1)  # [B x H x W]

    loss = (loss1 + loss2 + loss3) / 3.0  # [B x H x W]
    loss = loss.mean()

    return loss


# -----------------------------------------Jensen-Shannon Divergence ---------------------------------------------------
def js_div_loss(p_logits: torch.Tensor, q_logits: torch.Tensor,
                mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute Jensen-Shannon divergence.

    Args:
        p(torch.Tensor): 1st distributions of shape [B x C x H x W]
        q(torch.Tensor): 2nd distributions of shape [B x C x H x W]
        mask_keep(Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during loss
                                                    during loss computation (1 := keep, 0 :=ignore). Defaults to None.

    Returns:
          torch.Tensor: loss value
    """
    p_probs = nn.functional.softmax(p_logits, dim=1)
    q_probs = nn.functional.softmax(q_logits, dim=1)
    m_probs = (p_probs + q_probs) * 0.5

    p_probs = torch.clamp(p_probs, 1e-12, 1)
    q_probs = torch.clamp(q_probs, 1e-12, 1)
    m_probs = torch.clamp(m_probs, 1e-12, 1)

    kl_p_m = p_probs * torch.log(p_probs / m_probs)  # [B, C, H, W]
    kl_p_m = torch.sum(kl_p_m, dim=1)  # [B, H, W]

    kl_q_m = q_probs * torch.log(q_probs / m_probs)  # [B, C, H, W]
    kl_q_m = torch.sum(kl_q_m, dim=1)  # [B, H, W]

    # compute Jensen_Shannon divergence
    js_p_q = (0.5 * kl_p_m) + (0.5 * kl_q_m)  # [B, H, W]

    if mask_keep is not None:
        # [M] where M is number of pixel which should be kept according to mask_keep (i.e., torch.sum(mask_keep)=M)
        js_p_q = js_p_q[mask_keep]

    loss = torch.mean(js_p_q)

    assert loss >= 0, f"Invalid loss for js divergence: {loss}"  # lower bound
    assert loss <= math.log(2), f"Invalid loss for js divergence: {loss}"  # upper bound

    return loss


# ------------------------------------------ Kullback–Leibler Divergence -----------------------------------------------
def kl_div_loss(x_logits_pred: torch.Tensor, x_logits_true: torch.Tensor,
                mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
    """ Compute KL-Divergence.

    There are difference ways to compute the Kullback-Leibler Divergence.
    We refer to https://machinelearningmastery.com/divergence-between-probability-distributions/ for more information.

    Args:
        x_logits_pred(torch.Tensor): Source distributions of shape [B x C x H x W]
        x_logits_true(torch.Tensor): Target distributions of shape [B x C x H x W]
        mask_keep(Optional[torch.Tensor], optional): Mask of pixels of shape [B x H x W] which should be kept during
                                                     loss computation (1 :=keep, 0 :=ignore). Defaults to None
    """
    x_pred = nn.functional.softmax(x_logits_pred, dim=1)
    x_true = nn.functional.softmax(x_logits_true, dim=1)

    x_pred = torch.clamp(x_pred, 1e-12, 1)
    x_true = torch.clamp(x_true, 1e-12, 1)

    loss = x_true * torch.log(x_true) / (x_pred)  # [B x C x H x W]
    loss = torch.sum(loss, dim=1)  # [B x H x W]

    if mask_keep is not None:
        loss = loss[mask_keep]

    loss = torch.mean(loss)

    assert loss >= 0, f"Invalid loss for kl divergence: {loss}"

    return loss


# ------------------------------------------UTILS-----------------------------------------------------------------------
def get_div_loss_weight():
    pass


def to_one_hot(tensor: torch.Tensor, n_classes: int) -> torch.Tensor:
    """ Convert tensor to its one hot encoded version.

    Props go to https://github.com/PRBonn/bonnetal/blob/master/train/common/onehot.py

    Args:
        tensor(torch.Tensor): ground truth tensor of shape [N x n_classes x H x W]
    """
    if len(tensor.size()) == 1:
        b = tensor.size(0)
        if tensor.is_cuda:
            one_hot = torch.zeros(b, n_classes, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(b, n_classes).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 2:
        n, b = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, b, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, b).scatter_(1, tensor.unsqueeze(1), 1)
    elif len(tensor.size()) == 3:
        n, h, w = tensor.size()
        if tensor.is_cuda:
            one_hot = torch.zeros(n, n_classes, h, w, device=torch.device('cuda')).scatter_(1, tensor.unsqueeze(1), 1)
        else:
            one_hot = torch.zeros(n, n_classes, h, w).scatter_(1, tensor.unsqueeze(1), 1)
    return one_hot


def get_criterion(cfg) -> nn.Module:
    loss_name = cfg['train']['loss']

    if loss_name == 'xentropy':
        weights = cfg['train']['class_weights']

    return CrossEntropy(weights)

# t = torch.tensor([[[[-0.9141, 1.7542, 0.0031, 0.3182],
#                     [-1.1969, 0.2962, 1.0228, 1.3369],
#                     [0.4772, -0.7530, 0.1231, 0.6173],
#                     [-0.4054, 0.0192, -0.6614, -0.5508]],
#
#                    [[-0.6260, -0.1832, -0.1700, 0.7247],
#                     [-0.2889, -0.1736, 2.2228, 0.6965],
#                     [-1.3260, 0.1234, 1.2445, -0.5093],
#                     [-0.2715, 0.1498, -1.3679, -1.6469]]],
#
#                   [[[-0.1051, 0.5952, -2.6944, -0.2620],
#                     [0.9074, -0.8733, 2.6921, 0.5763],
#                     [0.0703, 0.0956, 0.9393, 0.1100],
#                     [-0.9328, 1.1804, -2.7984, -0.7535]],
#
#                    [[0.3506, 0.6880, -0.2386, 0.2591],
#                     [-1.1779, -0.0572, 0.5429, -1.3103],
#                     [1.7901, 0.1453, 1.3634, 0.0388],
#                     [-1.4310, 0.5754, -0.5323, -1.3156]]]])
# b = t.size(0)
# print(b)
# one_hot = torch.zeros(b, 3).scatter_(1, t.unsqueeze(1), 1)
# print(one_hot)
