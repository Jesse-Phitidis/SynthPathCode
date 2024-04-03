import torch
from synthpath.training.constants import ordered_region_list # not including background
from monai.networks import one_hot


def format_loss(loss: torch.Tensor, dice_weight: float | None, path_weight: float = 0.5) -> tuple[torch.tensor, dict]:
    
    if isinstance(loss[0], tuple): # Both losses from DiceCELoss returned since they have different shapes but we still want class means
        tasks = []
        for loss_task in loss:
            loss_dice, loss_ce = loss_task
            loss_dice = torch.mean(loss_dice, dim=(0,2,3,4))
            loss_ce = torch.mean(loss_ce, dim=(0,2,3,4))
            catted = torch.stack([loss_dice * dice_weight, loss_ce * (1 - dice_weight)], dim=0)
            losses = torch.mean(catted, dim=0)
            tasks.append(losses)
        losses = torch.cat([tasks[0] * (1 - path_weight), tasks[1] * path_weight], dim=0)
        #losses = torch.mean(losses, dim=0)
    else: # Not from DiceCELoss
        tasks = []
        for loss_task in loss:
            losses = torch.mean(loss_task, dim=(0,2,3,4))
            tasks.append(losses)
        losses = torch.cat([tasks[0] * (1 - path_weight), tasks[1] * path_weight], dim=0)
        #losses = torch.mean(losses, dim=0)
            
    
    if isinstance(loss[0], tuple): # Again, if we have used DiceCELoss
        tasks = []
        for loss_task in loss:
            loss_dice, loss_ce = loss_task
            loss_dice = torch.mean(loss_dice)
            loss_ce = torch.sum(loss_ce, dim=1).mean()
            loss = (loss_dice * dice_weight) + (loss_ce * (1 - dice_weight))
            tasks.append(loss)
        loss = torch.mean(torch.stack([tasks[0] * (1 - path_weight), tasks[1] * path_weight], dim=0))
    elif loss[0].shape[2] == 1: # From dice loss i.e. shape should be [1,19,1,1,1]
        tasks = []
        for loss_task in loss:
            loss = torch.mean(loss_task)
            tasks.append(loss)
        loss = torch.mean(torch.stack([tasks[0] * (1 - path_weight), tasks[1] * path_weight], dim=0))
    else: # From cross entropy i.e. shape should be [1,19,96,96,96]
        tasks = []
        for loss_task in loss:
            loss = torch.sum(loss_task, dim=1).mean()
            tasks.append(loss)
        loss = torch.mean(torch.stack([tasks[0] * (1 - path_weight), tasks[1] * path_weight], dim=0))
    
    loss_dict = {}
    
    for region, loss_val in zip(ordered_region_list, losses):
        loss_dict[f"all_losses/{region}"] = loss_val
        
    return loss, loss_dict






# def format_loss(loss: torch.Tensor, dice_weight: float | None) -> tuple[torch.tensor, dict]:
    
#     if isinstance(loss, tuple): # Both losses from DiceCELoss returned since they have different shapes but we still want class means
#         loss_dice, loss_ce = loss
#         loss_dice = torch.mean(loss_dice, dim=(0,2,3,4))
#         loss_ce = torch.mean(loss_ce, dim=(0,2,3,4))
#         catted = torch.stack([loss_dice * dice_weight, loss_ce * (1 - dice_weight)], dim=0)
#         losses = torch.mean(catted, dim=0)
#     else: # Not from DiceCELoss
#         losses = torch.mean(loss, dim=(0,2,3,4))
    
#     if isinstance(loss, tuple): # Again, if we have used DiceCELoss
#         loss_dice, loss_ce = loss
#         loss_dice = torch.mean(loss_dice)
#         loss_ce = torch.sum(loss_ce, dim=1).mean()
#         loss = (loss_dice * dice_weight) + (loss_ce * (1 - dice_weight))
#     elif loss.shape[2] == 1: # From dice loss i.e. shape should be [1,19,1,1,1]
#         loss = torch.mean(loss)
#     else: # From cross entropy i.e. shape should be [1,19,96,96,96]
#         loss = torch.sum(loss, dim=1).mean()
    
#     loss_dict = {}
    
#     for region, loss_val in zip(ordered_region_list, losses):
#         loss_dict[f"all_losses/{region}"] = loss_val
        
#     return loss, loss_dict


def format_val(dice_anat: torch.tensor, dice_path: torch.tensor, hd95_anat: torch.tensor, hd95_path: torch.tensor) -> dict:
    dice_anat = torch.nanmean(dice_anat, dim=0)
    dice_path = torch.nanmean(dice_path, dim=0)
    hd95_anat = torch.nanmean(hd95_anat, dim=0)
    hd95_path = torch.nanmean(hd95_path, dim=0)
    
    val_dict = {}
    
    for region, dice_val_anat, hd95_val_anat in zip(ordered_region_list, dice_anat, hd95_anat):
        val_dict[f"all_dice/{region}"] = dice_val_anat.item()
        val_dict[f"all_hd95/{region}"] = hd95_val_anat.item() * 2

    val_dict[f"all_dice/stroke"] = dice_path.item()
    val_dict[f"all_hd95/stroke"] = hd95_path.item() * 2
    val_dict[f"dice_stroke"] = dice_path.item()
    val_dict[f"hd95_stroke"] = hd95_path.item() * 2
        
    return val_dict


def format_val_dice_only(dice_anat: torch.tensor, dice_path: torch.tensor) -> dict:
    dice_anat = torch.nanmean(dice_anat, dim=0)
    dice_path = torch.nanmean(dice_path, dim=0)
    
    val_dict = {}
    
    for region, dice_val_anat in zip(ordered_region_list, dice_anat):
        val_dict[f"all_dice/{region}"] = dice_val_anat.item()
    
    val_dict[f"all_dice/stroke"] = dice_path.item()
    val_dict[f"dice_stroke"] = dice_path.item()

        
    return val_dict


def old_format_test(dice_anat: torch.tensor, dice_path: torch.tensor, hd95_anat: torch.tensor, hd95_path: torch.tensor) -> dict:
    
    dice_anat = torch.nanmean(dice_anat, dim=0)
    dice_path = torch.nanmean(dice_path, dim=0)
    hd95_anat = torch.nanmean(hd95_anat, dim=0)
    hd95_path = torch.nanmean(hd95_path, dim=0)
    
    test_dict = {}
    test_dict["dice"] = {}
    test_dict["hd95"] = {}
    
    for region, dice_val, hd95_val in zip(ordered_region_list, dice_anat, hd95_anat):
        test_dict["dice"].update({region: dice_val.item()})
        test_dict["hd95"].update({region: hd95_val.item() * 2})
    test_dict["dice"].update({"stroke": dice_path.item()})
    test_dict["hd95"].update({"stroke": hd95_path.item() * 2})
        
    return test_dict


def inverse_sigmoid(y: float) -> float:
    y = torch.as_tensor(y)
    return -torch.log((1 - y) / y)



def format_test(
    dice_anat: torch.tensor, 
    dice_path: torch.tensor, 
    hd95_anat: torch.tensor, 
    hd95_path: torch.tensor, 
    pre_path: torch.tensor, 
    rec_path: torch.tensor, 
    lf1_path: torch.tensor, 
    lpre_path: torch.tensor, 
    lrec_path: torch.tensor, 
    ap_path: torch.tensor
    ) -> dict:
    
    dice_anat = torch.nanmean(dice_anat, dim=0)
    dice_path = torch.nanmean(dice_path, dim=0)
    hd95_anat = torch.nanmean(hd95_anat, dim=0)
    hd95_path = torch.nanmean(hd95_path, dim=0)
    
    pre_path = torch.nanmean(pre_path, dim=0)
    rec_path = torch.nanmean(rec_path, dim=0)
    lf1_path = torch.nanmean(lf1_path, dim=0)
    lpre_path = torch.nanmean(lpre_path, dim=0)
    lrec_path = torch.nanmean(lrec_path, dim=0)
    ap_path = torch.nanmean(ap_path, dim=0)
    
    test_dict = {}
    test_dict["dice"] = {}
    test_dict["hd95"] = {}
    
    test_dict["pre"] = {}
    test_dict["rec"] = {}
    test_dict["lf1"] = {}
    test_dict["lpre"] = {}
    test_dict["lrec"] = {}
    test_dict["ap"] = {}
    
    for region, dice_val, hd95_val in zip(ordered_region_list, dice_anat, hd95_anat):
        test_dict["dice"].update({region: dice_val.item()})
        test_dict["hd95"].update({region: hd95_val.item() * 2})
    test_dict["dice"].update({"stroke": dice_path.item()})
    test_dict["hd95"].update({"stroke": hd95_path.item() * 2})
    
    test_dict["pre"].update({"stroke": pre_path.item()})
    test_dict["rec"].update({"stroke": rec_path.item()})
    test_dict["lf1"].update({"stroke": lf1_path.item()})
    test_dict["lpre"].update({"stroke": lpre_path.item()})
    test_dict["lrec"].update({"stroke": lrec_path.item()})
    test_dict["ap"].update({"stroke": ap_path.item()})
        
    return test_dict


def inverse_sigmoid(y: float) -> float:
    y = torch.as_tensor(y)
    return -torch.log((1 - y) / y)