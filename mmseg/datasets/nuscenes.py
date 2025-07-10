# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import os

import mmcv
import torch
import json
import numpy as np
from PIL import Image

from mmcv.utils import print_log
from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset
from tqdm import trange
def apply_mask(data, mask):
    """
    Apply mask to data.

    Args:
        data (torch.Tensor): Input tensor of shape [N, C].
        mask (np.ndarray): Mask array of shape [H, W].

    Returns:
        torch.Tensor: Masked data tensor.
    """
    n, c = data.shape
    h, w = mask.shape
    masked_data = torch.zeros_like(data)
    
    # Reshape data to match spatial dimensions
    reshaped_data = data.view(n, h, w)
    
    for i in range(c):
        masked_data[:, i] = reshaped_data[:, :, :].view(n, -1)[:, mask.flatten()]
    
    return masked_data
def calculate_distance_masks(shape, pixel_size=0.5, ego_point=None):
    """
    Create distance masks for different ranges.

    Args:
        shape (tuple): The shape of the BEV map.
        pixel_size (float): The size each pixel represents in meters.
        ego_point (tuple): Ego point location (y, x).

    Returns:
        dict: A dictionary containing binary masks for each range.
    """
    masks = {}
    h, w = shape
    
    # Calculate distances from the ego point
    y_ego, x_ego = ego_point if ego_point else (h - 1, w // 2)
    y, x = np.indices((h, w))
    dist = np.sqrt((y - y_ego) ** 2 + (x - x_ego) ** 2) * pixel_size

    # Define distance ranges and create corresponding masks
    masks['0-15m'] = (dist <= 15)
    masks['15-30m'] = (dist > 15) & (dist <= 30)
    masks['30-inf'] = (dist > 30)

    return masks
def covert_color(input):
    str1 = input[1:3]
    str2 = input[3:5]
    str3 = input[5:7]
    r = int('0x' + str1, 16)
    g = int('0x' + str2, 16)
    b = int('0x' + str3, 16)
    return (r, g, b)


def visualize_map_mask(map_mask):
    color_map = ['#a6cee3','#1f78b4','#b2df8a','#33a02c','#fb9a99',
                 '#e31a1c','#fdbf6f','#ff7f00','#cab2d6','#6a3d9a',
                 '#7e772e','#00ff00','#0000ff','#00ffff','#303030']
    ori_shape = map_mask.shape
    vis = np.zeros((ori_shape[1], ori_shape[2], 3),dtype=np.uint8)
    vis = vis.reshape(-1,3)
    map_mask = map_mask.reshape(ori_shape[0],-1)
    for layer_id in range(map_mask.shape[0]):
        keep = np.where(map_mask[layer_id,:])[0]
        for i in range(3):
            vis[keep, 2-i] = covert_color(color_map[layer_id])[i]
    return vis.reshape(ori_shape[1], ori_shape[2], 3)


@DATASETS.register_module()
class NuscenesDataset(CustomDataset):
    """NuScenes dataset.

    In segmentation map annotation for NuScenes dataset, 0 stands for background. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
    CLASSES = ('drivable_area', 'ped_crossing', 'walkway', 'carpark',
               'car', 'truck', 'bus', 'trailer', 'construction_vehicle',
               'pedestrian', 'motorcycle', 'bicycle', 'traffic_cone', 'barrier')

    PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
               [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
               [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
               [150, 5, 61], [120, 120, 70], [8, 255, 51]]

    def __init__(self, **kwargs):
        super(NuscenesDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=True,
            **kwargs)

    def results2img(self, results, imgfile_prefix, to_label_id):
        """Write the segmentation results to images.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            imgfile_prefix (str): The filename prefix of the png files.
                If the prefix is "somepath/xxx",
                the png files will be named "somepath/xxx.png".
            to_label_id (bool): whether convert output to label_id for
                submission

        Returns:
            list[str: str]: result txt files which contains corresponding
            semantic segmentation images.
        """
        mmcv.mkdir_or_exist(imgfile_prefix)
        result_files = []
        prog_bar = mmcv.ProgressBar(len(self))
        for idx in range(len(self)):
            result = results[idx]

            filename = self.img_infos[idx]['filename']
            basename = osp.splitext(osp.basename(filename))[0]

            png_filename = osp.join(imgfile_prefix, f'{basename}.png')

            result = result + 1

            output = Image.fromarray(result.astype(np.uint8))
            output.save(png_filename)
            result_files.append(png_filename)

            prog_bar.update()

        return result_files

    def prepare_test_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        return self.pipeline(results)
    
    # def format_results(self, results, imgfile_prefix=None, to_label_id=True):
    #     """Format the results into dir for visualization.

    #     Args:
    #         results (list): Testing results of the dataset.
    #         imgfile_prefix (str | None): The prefix of images files. It
    #             includes the file path and the prefix of filename, e.g.,
    #             "a/b/prefix". If not specified, a temp file will be created.
    #             Default: None.
    #         to_label_id (bool): whether convert output to label_id for
    #             submission. Default: False

    #     Returns:
    #         tuple: (result_files, tmp_dir), result_files is a list containing
    #            the image paths, tmp_dir is the temporal directory created
    #             for saving json/png files when img_prefix is not specified.
    #     """

    #     assert isinstance(results, list), 'results must be a list'

    #     imgfile_prefix = osp.join(imgfile_prefix, 'vis')
    #     if not osp.exists(imgfile_prefix):
    #         os.makedirs(imgfile_prefix)
    #     print_log('\n Start formatting the result')
    
    #     for id in trange(len(results)):
    #         # print( results[id])
    #         pred, gt, img_path = results[id]
    #         b,c,h,w = pred.shape
    #         assert pred.shape[0]==1 and gt.shape[0]==1
    #         pred = pred[0]
    #         gt = gt[0]
    #         gt[-1, ...] = np.invert(gt[-1, ...])
    #         pred = np.concatenate([pred, gt[-1,...][None,...]], axis=0)
    #         pred_vis = visualize_map_mask(pred)
    #         gt_vis = visualize_map_mask(gt)
    #         img = mmcv.imread(img_path, backend='cv2')
    #         img = mmcv.imresize(img,(int(float(img.shape[1])*h/float(img.shape[0])), h))
    #         vis = np.concatenate([img, pred_vis[::-1,...], gt_vis[::-1,]], axis=1)
    #         save_path = osp.join(imgfile_prefix, os.path.basename(img_path))
    #         mmcv.imwrite(vis, save_path)

    # def evaluate(self,
    #              results,
    #              metric='mIoU',
    #              logger=None,
    #              efficient_test=False,
    #              **kwargs):
    #     """Calculate the evaluate result according to the metric type.

    #         Args:
    #             results (list): Testing results of the dataset.
    #             metric (str | list[str]): Type of evalutate metric, mIoU is in consistent
    #                 with "Predicting Semantic Map Representations from Images with
    #                 Pyramid Occupancy Networks. CVPR2020", where per class fp,fn,tp are
    #                 calculated on the hold dataset first. mIOUv1 calculates the per
    #                 class iou in each image first and average the result between the
    #                 valid images (i.e. for class c, there is positive sample point in
    #                 this image). mIOUv2 calculates the per image iou first and average
    #                 the result between all images.
    #             logger (logging.Logger | None | str): Logger used for printing
    #                 related information during evaluation. Default: None.

    #         Returns:
    #             tuple: (result_files, tmp_dir), result_files is a list containing
    #                the image paths, tmp_dir is the temporal directory created
    #                 for saving json/png files when img_prefix is not specified.
    #         """
    #     if isinstance(metric, str):
    #         metric = [metric]
    #     allowed_metrics = ['mIoU', 'mIoUv1', 'mIoUv2']
    #     if not set(metric).issubset(set(allowed_metrics)):
    #         raise KeyError('metric {} is not supported'.format(metric))
    #     tp = torch.cat([res[0][None, ...] for res in results], dim=0) #N*C
    #     fp = torch.cat([res[1][None, ...] for res in results], dim=0) #N*C
    #     fn = torch.cat([res[2][None, ...] for res in results], dim=0) #N*C
    #     valids = torch.cat([res[3][None,...] for res in results],dim=0) #N*C
    #     for met in metric:
    #         if met=='mIoU':
    #             ious = tp.sum(0).float()/(tp.sum(0)+fp.sum(0)+fn.sum(0)).float()
    #             print_log('\nper class results (iou):', logger)
    #             for cid in range(len(self.CLASSES)):
    #                 print_log('%.04f:%s tp:%d fp:%d fn:%d' % (ious[cid], self.CLASSES[cid], tp.sum(0)[cid],fp.sum(0)[cid],fn.sum(0)[cid]), logger)
    #             print_log('%s: %.04f' % (met, ious.mean()), logger)
    #         elif met == 'mIoUv1':
    #             ious = tp.float() / (tp + fp + fn).float()
    #             print_log('\nper class results (iou):', logger)
    #             miou, valid_class = 0, 0
    #             for cid in range(len(self.CLASSES)):
    #                 iou_c = ious[:, cid][valids[:, cid]]
    #                 if iou_c.shape[0] > 0:
    #                     iou_c = iou_c.mean()
    #                     miou += iou_c
    #                     valid_class += 1
    #                 else:
    #                     iou_c = -1
    #                 print_log('%.04f:%s' % (iou_c, self.CLASSES[cid]), logger)
    #             print_log('%s: %.04f' % (met, miou / valid_class), logger)
    #         elif met == 'mIoUv2':
    #             ious = tp.sum(-1).float() / (tp.sum(-1) + fp.sum(-1) + fn.sum(-1)).float()
    #             print_log('\n%s: %.04f' % (met, ious.mean()), logger)
    #         else:
    #             assert False, 'nuknown metric type %s'%metric


    def calculate_distance_masks(shape, pixel_size=0.5, ego_point=None):
        """
        Create distance masks for different ranges.

        Args:
            shape (tuple): The shape of the BEV map.
            pixel_size (float): The size each pixel represents in meters.
            ego_point (tuple): Ego point location (y, x).

        Returns:
            dict: A dictionary containing binary masks for each range.
        """
        masks = {}
        h, w = shape
        
        # Calculate distances from the ego point
        y_ego, x_ego = ego_point if ego_point else (h - 1, w // 2)
        y, x = np.indices((h, w))
        dist = np.sqrt((y - y_ego) ** 2 + (x - x_ego) ** 2) * pixel_size

        # Define distance ranges and create corresponding masks
        masks['0-15m'] = (dist <= 15)
        masks['15-30m'] = (dist > 15) & (dist <= 30)
        masks['30-inf'] = (dist > 30)

        return masks
    def format_results(self, results, imgfile_prefix=None, to_label_id=True):
            """Format the results into dir for visualization.

            Args:
                results (list): Testing results of the dataset.
                imgfile_prefix (str | None): The prefix of images files. It
                    includes the file path and the prefix of filename, e.g.,
                    "a/b/prefix". If not specified, a temp file will be created.
                    Default: None.
                to_label_id (bool): whether convert output to label_id for
                    submission. Default: False

            Returns:
                tuple: (result_files, tmp_dir), result_files is a list containing
                the image paths, tmp_dir is the temporal directory created
                    for saving json/png files when img_prefix is not specified.
            """

            assert isinstance(results, list), 'results must be a list'

            imgfile_prefix = osp.join(imgfile_prefix, 'vis')
            if not osp.exists(imgfile_prefix):
                os.makedirs(imgfile_prefix)
            print_log('\n Start formatting the result')

            # Calculate distance masks
            masks = calculate_distance_masks((98, 100))

            result_files = []
            for id in trange(len(results)):
                pred, gt, img_path = results[id]
                b, c, h, w = pred.shape
                assert pred.shape[0] == 1 and gt.shape[0] == 1
                pred = pred[0]
                gt = gt[0]
                gt[-1, ...] = np.invert(gt[-1, ...])
                pred = np.concatenate([pred, gt[-1, ...][None, ...]], axis=0)

                # Process each distance range
                for range_name, mask in masks.items():
                    masked_pred = pred.copy()
                    masked_gt = gt.copy()

                    # Apply mask to predictions and ground truth
                    for cls_idx in range(masked_pred.shape[0]):
                        masked_pred[cls_idx, ~mask] = 0
                        masked_gt[cls_idx, ~mask] = 0

                    # Visualize the masked predictions and ground truth
                    pred_vis = visualize_map_mask(masked_pred)
                    gt_vis = visualize_map_mask(masked_gt)
                    img = mmcv.imread(img_path, backend='cv2')
                    img = mmcv.imresize(img, (int(float(img.shape[1]) * h / float(img.shape[0])), h))
                    vis = np.concatenate([img, pred_vis[::-1, ...], gt_vis[::-1, ...]], axis=1)

                    # Save the visualized results
                    save_path = osp.join(imgfile_prefix, f'{range_name}_{os.path.basename(img_path)}')
                    mmcv.imwrite(vis, save_path)
                    result_files.append(save_path)

            return result_files, imgfile_prefix

    def evaluate(self, results, metric='mIoU', logger=None, efficient_test=False, **kwargs):
        """Calculate the evaluate result according to the metric type.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Type of evaluation metric.
            logger (logging.Logger | None | str): Logger used for printing related information during evaluation. Default: None.

        Returns:
            dict: Evaluation results for each metric and distance range.
        """

        if isinstance(metric, str):
            metric = [metric]
        allowed_metrics = ['mIoU', 'mIoUv1', 'mIoUv2']
        if not set(metric).issubset(set(allowed_metrics)):
            raise KeyError(f'Metric {metric} is not supported')

        # Calculate distance masks based on specific parameters
        masks = calculate_distance_masks((98, 100), pixel_size=0.5, ego_point=(97, 50))

        eval_results = {range_name: {} for range_name in masks.keys()}
        global_tp, global_fp, global_fn = 0, 0, 0  # For global mIoU calculation
        class_iou_sums = torch.zeros(len(self.CLASSES))  # For mean IoU per class
        class_counts = torch.zeros(len(self.CLASSES))  # To track valid classes

        i = 0
        for met in metric:
            for range_name, mask in masks.items():
                tp = torch.cat([res[0][None, ...] for res in results[i]], dim=0)  # N*C
                fp = torch.cat([res[1][None, ...] for res in results[i]], dim=0)  # N*C
                fn = torch.cat([res[2][None, ...] for res in results[i]], dim=0)  # N*C
                valids = torch.cat([res[3][None, ...] for res in results[i]], dim=0)  # N*C
                i += 1

                if met == 'mIoU':
                    ious = tp.sum(0).float() / (tp.sum(0) + fp.sum(0) + fn.sum(0)).float()
                    print_log(f'\n{range_name}: per class results (iou):', logger)
                    for cid in range(len(self.CLASSES)):
                        print_log('%.04f:%s tp:%d fp:%d fn:%d' % (
                            ious[cid], self.CLASSES[cid], tp.sum(0)[cid], fp.sum(0)[cid], fn.sum(0)[cid]), logger)
                        class_iou_sums[cid] += ious[cid]  # Accumulate IoU for each class
                        class_counts[cid] += 1  # Count valid classes
                    print_log('%s: %.04f' % (met, ious.mean()), logger)
                    eval_results[range_name][met] = ious.mean().item()

                    # Accumulate global TP, FP, FN for global mIoU
                    global_tp += tp.sum(0)
                    global_fp += fp.sum(0)
                    global_fn += fn.sum(0)

                elif met == 'mIoUv1':
                    ious = tp.float() / (tp + fp + fn).float()
                    miou, valid_class = 0, 0
                    for cid in range(len(self.CLASSES)):
                        iou_c = ious[:, cid][valids[:, cid].bool()]
                        if iou_c.shape[0] > 0:
                            iou_c = iou_c.mean()
                            miou += iou_c
                            valid_class += 1
                        else:
                            iou_c = -1
                        print_log('%.04f:%s' % (iou_c, self.CLASSES[cid]), logger)
                    eval_results[range_name][met] = miou / valid_class if valid_class > 0 else 0

                    # Accumulate global TP, FP, FN for global mIoU
                    global_tp += tp.sum(0)
                    global_fp += fp.sum(0)
                    global_fn += fn.sum(0)

                elif met == 'mIoUv2':
                    ious = tp.sum(-1).float() / (tp.sum(-1) + fp.sum(-1) + fn.sum(-1)).float()
                    print_log(f'\n{range_name}: {met}: %.04f' % ious.mean(), logger)
                    eval_results[range_name][met] = ious.mean().item()

                    # Accumulate global TP, FP, FN for global mIoU
                    global_tp += tp.sum(-1).sum(0)
                    global_fp += fp.sum(-1).sum(0)
                    global_fn += fn.sum(-1).sum(0)

                else:
                    assert False, f'Unknown metric type {metric}'

        # Calculate global mIoU
        global_ious = global_tp.float() / (global_tp + global_fp + global_fn).float()
        eval_results['Global'] = {}
        eval_results['Global']['mIoU'] = global_ious.mean().item()

        # Calculate mean IoU per class
        mean_class_iou = class_iou_sums[class_counts > 0].mean().item()
        eval_results['Global']['meanClassIoU'] = mean_class_iou
        print_log('%s: %.04f' % (met, global_ious.mean()), logger)
        return eval_results