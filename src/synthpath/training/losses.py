import torch.nn.functional as F
import torch
import torch.nn as nn
from monai.networks import one_hot
from typing import Literal, Sequence

def normalise_weights(w, loss_fn):
    
    '''
        In dice loss we take the mean over the classes (dim=1) as well as the batch and spatial dims. 
        Hence, we require this step so that we perform a weighted average. I.e. we want the weight to
        be the same in the case when a single class is given weight, regardless of how many classes 
        there are.
        E.g. if w is given as [0,0,1] we convert to [0,0,3] which is correct because when we 
        take the mean over dim=1 we divide by 3. Equivalently if w is [0,0,0,0,1] we convert to
        [0,0,0,0,5] so that when we divide by 5 we get the same weight for the final class.
        For ce loss we don't want to do this since we sum over dim=1 before taking the mean over
        batch and spatial dims (imagine the case where we get a perfect prediction).
    '''
    
    assert loss_fn in ["dice", "ce"]
    w = w / w.sum()
    if loss_fn == "dice":
        w = w * len(w)
    return w


class DiceLoss(nn.Module):
    
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: Literal["none", "mean", "sum"] = "none",
        batch: bool = False,
        weight: Sequence[float] | None = None,
        sensible_weight_norm: bool = False,
        keepdim: bool = True
    ):
        super().__init__()
        
        assert not (softmax and sigmoid), f"only one of softmax and sigmoid can be applied"
        assert reduction in ["none", "mean", "sum"], "reduction must be one of 'none', 'mean', or 'sum'"
        
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.reduction = reduction
        self.batch = batch
        self.weight = weight
        self.sensible_weight_norm = sensible_weight_norm
        self.keepdim = keepdim
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        
        # Activation
        if self.softmax:
            pred = F.softmax(pred, dim=1)
        if self.sigmoid:
            pred = F.sigmoid(pred) 
        
        # One hot
        if self.to_onehot_y:
            target = one_hot(target, num_classes=pred.shape[1], dim=1)
        assert pred.shape[1] == target.shape[1], f"pred channels ({pred.shape[1]}) and target channels ({target.shape[1]}) do not match. \
Consider setting to_onehot_y=True"
            
        # Include background
        if not self.include_background:
            assert pred.shape[1] > 1, f"include_background set to False, but single channel prediction"
            pred = pred[:, 1:]
            target = target[:, 1:]
            
        # Weights
        if self.weight is not None:
            weight = torch.as_tensor(self.weight).to(device=pred.device)
            assert weight.shape[0] == pred.shape[1], f"number of weights is {weight.shape[0]}, but number of channels is {pred.shape[1]}. \
Do not provide a weight for channel 0 if include_background is False"
            if self.sensible_weight_norm:
                weight = normalise_weights(weight, "dice")
                
        # Loss computation
        dims_list = [i for i in range(2, len(pred.shape))]
        dims = tuple([0] + dims_list) if self.batch else tuple(dims_list)
        numerator = torch.sum(pred * target, dim=dims, keepdim=self.keepdim)
        denominator = torch.sum(pred, dim=dims, keepdim=self.keepdim) + torch.sum(target, dim=dims, keepdim=self.keepdim)
        loss = 1.0 - (2.0 * numerator + 1e-5)/(denominator + 1e-5)
        if self.weight is not None:
            weight = weight.view(1, loss.shape[1], *([1]*len(loss.shape[2:])))
            loss = weight * loss
            
        # Reduction
        if self.reduction == "none":
            return loss
        dims = tuple([i for i in range(len(loss.shape))])
        if self.reduction == "mean":
            loss = torch.mean(loss, dim=dims,  keepdim=self.keepdim)
        if self.reduction == "sum":
            loss = torch.sum(loss, dim=dims, keepdim=self.keepdim)
            
        return loss
        

class CELoss(nn.Module):
    
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: Literal["none", "mean", "sum"] = "none",
        weight: Sequence[float] | None = None,
        sensible_weight_norm: bool = False,
        sum_channels: bool = False,
        keepdim: bool = True
    ):
        super().__init__()
        
        assert not (softmax and sigmoid), f"only one of softmax and sigmoid can be applied"
        assert softmax or sigmoid, f"softmax or sigmoid must be true since taking log as a sepate step is not stable"
        assert reduction in ["none", "mean", "sum"], "reduction must be one of 'none', 'mean', or 'sum'"
        if reduction != "none":
            assert sum_channels, "sum_channels should be set to True if reduction is not 'none'"
        
        self.include_background = include_background
        self.to_onehot_y = to_onehot_y
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.reduction = reduction
        self.weight = weight
        self.sensible_weight_norm = sensible_weight_norm
        self.sum_channels = sum_channels
        self.keepdim = keepdim
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        
        # Activation
        if self.softmax:
            pred = F.log_softmax(pred, dim=1)
        if self.sigmoid:
            pred = F.logsigmoid(pred)

        # One hot
        if self.to_onehot_y:
            target = one_hot(target, num_classes=pred.shape[1], dim=1)
        assert pred.shape[1] == target.shape[1], f"pred channels ({pred.shape[1]}) and target channels ({target.shape[1]}) do not match. \
Consider setting to_onehot_y=True"
            
        # Exclude background
        if not self.include_background:
            assert pred.shape[1] > 1, f"include_background set to False, but single channel prediction"
            pred = pred[:, 1:]
            target = target[:, 1:]
            
        # Weights
        if self.weight is not None:
            weight = torch.as_tensor(self.weight).to(device=pred.device)
            assert weight.shape[0] == pred.shape[1], f"number of weights is {weight.shape[0]}, but number of channels is {pred.shape[1]}. \
Do not provide a weight for channel 0 if include_background is False"
            if self.sensible_weight_norm:
                weight = normalise_weights(weight, "ce")
                
        # Loss computation
        if self.weight is not None:
            weight = weight.view(1, pred.shape[1], *([1]*len(pred.shape[2:])))
            target = weight * target
        loss = - (pred * target)
        
        # Sum channels
        if self.sum_channels:
            loss = torch.sum(loss, dim=1, keepdim=self.keepdim)
            
        # Reduction
        assert self.reduction in ["none", "mean", "sum"], "reduction must be one of 'none', 'mean', or 'sum'"
        if self.reduction == "none":
            return loss
        dims = tuple([i for i in range(len(loss.shape))])
        if self.reduction == "mean":
            loss = torch.mean(loss, dim=dims, keepdim=self.keepdim)
        if self.reduction == "sum":
            loss = torch.sum(loss, dim=dims, keepdim=self.keepdim)
            
        return loss


class DiceCELoss(nn.Module):
    
    def __init__(
        self,
        include_background: bool = True,
        to_onehot_y: bool = False,
        sigmoid: bool = False,
        softmax: bool = False,
        reduction: Literal["none", "mean", "sum"] = "none",
        batch: bool = False,
        weight: Sequence[float] | None = None,
        sensible_weight_norm: bool = False,
        sum_channels_ce: bool = False,
        keepdim: bool = True,
        dice_weight: float | None = None,
        return_separate: bool = False
    ):
        super().__init__()
       
        if not return_separate:
            assert (dice_weight is not None) and (0 <= dice_weight <= 1), "if not return_separate, dice_weight must be in range [0, 1]"
            assert reduction is not None, "if not return_separate, reduction cannont be 'none' because DiceLoss and CELoss have different shapes"
        
        self.dice_weight = dice_weight
        self.return_separate = return_separate

        self.dice_loss = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            reduction=reduction,
            batch=batch,
            weight=weight,
            sensible_weight_norm=sensible_weight_norm,
            keepdim=keepdim
        )
        
        self.ce_loss = CELoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            sigmoid=sigmoid,
            softmax=softmax,
            reduction=reduction,
            weight=weight,
            sensible_weight_norm=sensible_weight_norm,
            sum_channels=sum_channels_ce,
            keepdim=keepdim
        )
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor):
     
        dice_loss = self.dice_loss(pred, target)
        ce_loss = self.ce_loss(pred, target)
        
        if self.return_separate:
            return dice_loss, ce_loss
        else:
            return self.dice_weight * dice_loss + (1.0 - self.dice_weight) * ce_loss




















# class MonaiDiceLoss(DiceLoss):
#     def __init__(self, class_weights=None):
#         if class_weights is not None:
#             class_weights = normalise_weights(class_weights, "dice")
#         super().__init__(include_background=False, to_onehot_y=True, softmax=True, reduction="none", weight=class_weights)
#         self.class_weight = class_weights # self.class_weights is not set until forward call which is an issue when loading the state dict before a forward call
        
        
# class CrossEntropyLoss(nn.Module):
#     def __init__(self, class_weights):
#         super().__init__()
#         if class_weights is not None:
#             class_weights = normalise_weights(class_weights, "ce")
#         self.class_weights = class_weights
        
#     def __call__(self, inp, target):
#         device = inp.device
#         target = one_hot(target, num_classes=inp.shape[1], dim=1)[:, 1:]
#         inp = F.log_softmax(inp, dim=1)[:, 1:]
#         if self.class_weights is not None:
#             w = self.class_weights.view(1, inp.shape[1], *([1]*len(inp.shape[2:]))).to(device=device)
#             return - (inp * w * target)
#         else:
#             return - (inp * target)
    
    
# class DiceCELoss(nn.Module):
#     def __init__(self, class_weights):
#         super().__init__()
#         self.dice_loss = MonaiDiceLoss(class_weights=class_weights)
#         self.ce_loss = CrossEntropyLoss(class_weights=class_weights)
        
#     def __call__(self, inp, target):
#         return self.dice_loss(inp, target), self.ce_loss(inp, target)