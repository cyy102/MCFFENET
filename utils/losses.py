import torch
import torch.nn as nn
import torch.nn.functional as F


def soft_dice_score(output, target, smooth=0.0, eps=1e-7, dims=None):
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    return (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)


class DiceLoss(nn.Module):
    """Multi-class Dice loss — identical to AFENet's DiceLoss.

    - Uses log_softmax().exp() for numerical stability.
    - Masks out classes with no ground-truth pixels (empty class).
    - Supports ignore_index.
    """

    def __init__(self, smooth=0.05, ignore_index=None, eps=1e-7):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, y_pred, y_true):
        # y_pred: (B, C, H, W) logits, y_true: (B, H, W) long
        bs = y_true.size(0)
        num_classes = y_pred.size(1)

        # numerical-stable softmax
        y_pred = y_pred.log_softmax(dim=1).exp()

        dims = (0, 2)

        y_true = y_true.view(bs, -1)
        y_pred = y_pred.view(bs, num_classes, -1)

        if self.ignore_index is not None:
            mask = y_true != self.ignore_index
            y_pred = y_pred * mask.unsqueeze(1)
            y_true = F.one_hot((y_true * mask).to(torch.long), num_classes)
            y_true = y_true.permute(0, 2, 1) * mask.unsqueeze(1)
        else:
            y_true = F.one_hot(y_true, num_classes)
            y_true = y_true.permute(0, 2, 1)

        scores = soft_dice_score(y_pred, y_true.type_as(y_pred),
                                 smooth=self.smooth, eps=self.eps, dims=dims)
        loss = 1.0 - scores

        # mask out empty classes (no GT pixels)
        mask = y_true.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return loss.mean()


class SoftBCEWithLogitsLoss(nn.Module):
    """Multi-class BCE loss with label smoothing and ignore_index — identical to AFENet's."""

    def __init__(self, weight=None, ignore_index=None, reduction='mean',
                 smooth_factor=None, pos_weight=None):
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.smooth_factor = smooth_factor
        self.register_buffer("weight", weight)
        self.register_buffer("pos_weight", pos_weight)

    def forward(self, input, target):
        C = input.size(1)

        if self.ignore_index is not None:
            not_ignored = (target != self.ignore_index).unsqueeze(1).float()
            target = target.masked_fill(target == self.ignore_index, 0)

        # one-hot encode: (B, H, W) -> (B, C, H, W)
        target = F.one_hot(target.long(), num_classes=C).permute(0, 3, 1, 2).float()

        if self.smooth_factor is not None:
            soft_targets = ((1 - target) * self.smooth_factor + target * (1 - self.smooth_factor)).type_as(input)
        else:
            soft_targets = target.type_as(input)

        loss = F.binary_cross_entropy_with_logits(
            input, soft_targets, self.weight, pos_weight=self.pos_weight, reduction="none"
        )

        if self.ignore_index is not None:
            loss = loss * not_ignored

        if self.reduction == "mean":
            if self.ignore_index is not None:
                return loss.sum() / not_ignored.sum().clamp_min(1)
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class JointLoss(nn.Module):
    """CE (SoftBCE) + Dice — identical to AFENet's JointLoss."""

    def __init__(self, first, second, first_weight=1.0, second_weight=1.0):
        super().__init__()
        self.first = first
        self.second = second
        self.first_weight = first_weight
        self.second_weight = second_weight

    def forward(self, *input):
        loss1 = self.first(*input)
        loss2 = self.second(*input)
        return self.first_weight * loss1 + self.second_weight * loss2
