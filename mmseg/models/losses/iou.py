import torch
def iou(preds, labels, mask=None, per_class=False,mode = 'iou'):
    """iou calculation.
    Args:
        pred (torch.Tensor): The model prediction, shape (N, num_class, ...)
        labels (torch.Tensor): The target of each prediction, shape (N, num_class, ...)
        mask (float, optional): The target of each prediction, shape (N,1, ...)
    Returns:
        float | tuple[float]: If the input ``topk`` is a single integer,
            the function will return a single float as accuracy. If
            ``topk`` is a tuple containing multiple integers, the
            function will return a tuple containing accuracies of
            each ``topk`` number.
    """
    num_class = preds.shape[1]
    # preds.shape:[n,c,h,w]   labels.shape:[n,c,h,w]  mask.shape:[n,1,h,w]
    if mode == 'iou':
        preds = preds.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1)   # preds.shape:[c,n x h x w]
        labels = labels.flatten(2, -1).permute(1, 0, 2).reshape(num_class, -1) # labelss.shape:[c,n x h x w]
    # mask is used to discriminate background and foreground 
    if mask is not None:

        # ignore background both in preds and labels
        preds = preds[:, mask.flatten()]
        labels = labels[:, mask.flatten()]
            # print('labels_filtered',preds.size(),labels.size(),mask.size())
    # else:
    #     if mask is not None:
    #         # 将 mask 转换为形状为 [11277] 的一维布尔张量
    #         mask_1d = mask.squeeze()  # 如果 mask 是布尔类型，可以直接用；如果不是，需要先转换为布尔类型
    
    #         preds_filtered = preds[mask_1d]
    #         labels_filtered = labels[mask_1d]
    true_pos = preds & labels
    false_pos = preds & ~labels
    false_neg = ~preds & labels
    tp = true_pos.long()
    fp = false_pos.long()
    fn = false_neg.long()

    # tp.sum() means all the tp in all the images,tp.sum(-1) means tp in each class
    if not per_class:
        return tp.sum().float() / (tp.sum() + fn.sum() + fp.sum()).float()
    else:
        # valids: only to choose the positive samples,tp.sum(-1).shape: [1,num_classes]
        valid = labels.int().sum(-1)>0
        return tp.sum(-1), fp.sum(-1), fn.sum(-1), valid
