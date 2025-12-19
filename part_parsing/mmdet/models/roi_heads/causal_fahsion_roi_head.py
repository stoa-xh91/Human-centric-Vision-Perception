import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.core import (bbox2roi, multiclass_nms, bbox2result)
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class CausalFashionRoIHead(StandardRoIHead):

    def __init__(self, *arg, use_aug=False, **kwargs):
        super(CausalFashionRoIHead, self).__init__(*arg, **kwargs)
        self.use_aug = use_aug
        self.avg_pooler = nn.AdaptiveAvgPool2d((1, 1))

    def forward_train(self,
                      x,
                      img_metas,
                      proposal_list,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      **kwargs):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        # assign gts and sample proposals
        if self.with_bbox or self.with_mask:
            num_imgs = len(img_metas)
            if gt_bboxes_ignore is None:
                gt_bboxes_ignore = [None for _ in range(num_imgs)]
            sampling_results = []
            for i in range(num_imgs):
                assign_result = self.bbox_assigner.assign(
                    proposal_list[i], gt_bboxes[i], gt_bboxes_ignore[i],
                    gt_labels[i])
                sampling_result = self.bbox_sampler.sample(
                    assign_result,
                    proposal_list[i],
                    gt_bboxes[i],
                    gt_labels[i],
                    feats=[lvl_feat[i][None] for lvl_feat in x])
                sampling_results.append(sampling_result)

        losses = dict()
        # bbox head forward and loss
        if self.with_bbox:
            bbox_results = self._bbox_forward_train(x, sampling_results,
                                                    gt_bboxes, gt_labels,
                                                    img_metas)
            losses.update(bbox_results['loss_bbox'])

        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x, sampling_results,
                                                    bbox_results['bbox_feats'],
                                                    gt_masks, img_metas, **kwargs)
            losses.update(mask_results['loss_mask'])
            losses.update({'loss_factor': mask_results['loss_factor']})

        if len(sampling_results) != len(img_metas):
            sampling_results = sampling_results[:len(img_metas)]

        # return output
        results = dict(
            roi_losses=losses,
            sampling_results=sampling_results,
            bbox_results=bbox_results)
        return results

    def _mask_forward_train(self, x, sampling_results, bbox_feats, gt_masks,
                            img_metas, **kwargs):
        """Run forward function and calculate loss for mask head in
        training."""

        if self.use_aug:
            aug_x = kwargs['aug_x']
            combine_x = []
            for single_x, aug_single_x in zip(x, aug_x):
                combine_x.append(torch.cat([single_x, aug_single_x], dim=0))
            aug_sampling_results = sampling_results
            sampling_results.extend(aug_sampling_results)
        else:
            combine_x = x

        rois = []
        for res in sampling_results:
            rois.append(res.pos_bboxes)
        pos_rois = bbox2roi(rois)
        mask_results = self._mask_forward(combine_x, pos_rois)
        num_img = len(sampling_results) // 2
        sampling_results = sampling_results[:num_img]
        mask_targets = self.mask_head.get_targets(sampling_results, gt_masks,
                                                  self.train_cfg)
        pos_labels = torch.cat([res.pos_gt_labels for res in sampling_results])
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        mask_targets, pos_labels)
        if self.use_aug:
            loss_factor = self.loss_factor(mask_results['mask_feats'])

        mask_results.update(
            loss_mask=loss_mask,
            mask_targets=mask_targets,
            loss_factor=loss_factor)
        return mask_results

    def loss_factor(self, mask_feats):

        # factor loss
        if self.use_aug:
            num_pos = len(mask_feats) // 2
            cnt_feats = mask_feats[:num_pos, ...]
            cnt_aug_feats = mask_feats[num_pos:, ...]

            cnt_feats = self.avg_pooler(cnt_feats).squeeze()
            cnt_aug_feats = self.avg_pooler(cnt_aug_feats).squeeze()

            cnt_feats = F.normalize(cnt_feats, p=2, dim=-1)
            cnt_aug_feats = F.normalize(cnt_aug_feats, p=2, dim=-1)

            factor_loss = (cnt_feats * cnt_aug_feats).sum(dim=-1).add_(-1).pow_(2).mean() * 2

        return factor_loss

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        """Mask head forward function used in both training and testing."""
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor(
                x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        if self.use_aug and self.training:
            num_pos = len(mask_feats) // 2
        else:
            num_pos = len(mask_feats)
        mask_pred = self.mask_head(mask_feats[:num_pos])
        mask_results = dict(
            mask_pred=mask_pred,
            mask_feats=mask_feats)
        return mask_results
    
    def simple_test(self,
                    x,
                    proposal_list,
                    img_metas,
                    proposals=None,
                    rescale=False):
        """Test without augmentation.

        Args:
            x (tuple[Tensor]): Features from upstream network. Each
                has shape (batch_size, c, h, w).
            proposal_list (list(Tensor)): Proposals from rpn head.
                Each has shape (num_proposals, 5), last dimension
                5 represent (x1, y1, x2, y2, score).
            img_metas (list[dict]): Meta information of images.
            rescale (bool): Whether to rescale the results to
                the original image. Default: True.

        Returns:
            list[list[np.ndarray]] or list[tuple]: When no mask branch,
            it is bbox results of each image and classes with type
            `list[list[np.ndarray]]`. The outer list
            corresponds to each image. The inner list
            corresponds to each class. When the model has mask branch,
            it contains bbox results and mask results.
            The outer list corresponds to each image, and first element
            of tuple is bbox results, second element is mask results.
        """
        assert self.with_bbox, 'Bbox head must be implemented.'

        det_bboxes, det_labels, det_scores = self.simple_test_bboxes(
            x, img_metas, proposal_list, self.test_cfg, rescale=rescale)

        bbox_results = [
            bbox2result(det_bboxes[i], det_labels[i],
                        self.bbox_head.num_classes)
            for i in range(len(det_bboxes))
        ]

        if not self.with_mask:
            segm_results = None
        else:
            segm_results = self.simple_test_mask(
                x, img_metas, det_bboxes, det_labels, rescale=rescale)
        
        results = dict(
            det_results=bbox_results[0],
            segm_results=segm_results[0],
            garments_bboxes=det_bboxes[0],
            garments_labels=det_labels[0],
            garments_scores=det_scores[0])
        return results

    def simple_test_bboxes(self,
                           x,
                           img_metas,
                           proposals,
                           rcnn_test_cfg,
                           rescale=False):
        """Test only det bboxes without augmentation.

        Args:
            x (tuple[Tensor]): Feature maps of all scale level.
            img_metas (list[dict]): Image meta info.
            proposals (List[Tensor]): Region proposals.
            rcnn_test_cfg (obj:`ConfigDict`): `test_cfg` of R-CNN.
            rescale (bool): If True, return boxes in original image space.
                Default: False.

        Returns:
            tuple[list[Tensor], list[Tensor]]: The first list contains
                the boxes of the corresponding image in a batch, each
                tensor has the shape (num_boxes, 5) and last dimension
                5 represent (tl_x, tl_y, br_x, br_y, score). Each Tensor
                in the second list is the labels with shape (num_boxes, ).
                The length of both lists should be equal to batch_size.
        """

        rois = bbox2roi(proposals)

        if rois.shape[0] == 0:
            batch_size = len(proposals)
            det_bbox = rois.new_zeros(0, 5)
            det_label = rois.new_zeros((0, ), dtype=torch.long)
            if rcnn_test_cfg is None:
                det_bbox = det_bbox[:, :4]
                det_label = rois.new_zeros(
                    (0, self.bbox_head.fc_cls.out_features))
            # There is no proposal in the whole batch
            return [det_bbox] * batch_size, [det_label] * batch_size

        bbox_results = self._bbox_forward(x, rois)
        img_shapes = tuple(meta['img_shape'] for meta in img_metas)
        scale_factors = tuple(meta['scale_factor'] for meta in img_metas)

        # split batch bbox prediction back to each image
        cls_score = bbox_results['cls_score']
        bbox_pred = bbox_results['bbox_pred']
        num_proposals_per_img = tuple(len(p) for p in proposals)
        rois = rois.split(num_proposals_per_img, 0)
        cls_score = cls_score.split(num_proposals_per_img, 0)

        # some detector with_reg is False, bbox_pred will be None
        if bbox_pred is not None:
            # TODO move this to a sabl_roi_head
            # the bbox prediction of some detectors like SABL is not Tensor
            if isinstance(bbox_pred, torch.Tensor):
                bbox_pred = bbox_pred.split(num_proposals_per_img, 0)
            else:
                bbox_pred = self.bbox_head.bbox_pred_split(
                    bbox_pred, num_proposals_per_img)
        else:
            bbox_pred = (None, ) * len(proposals)

        # apply bbox post-processing to each image individually
        det_bboxes = []
        det_labels = []
        det_scores = []
        for i in range(len(proposals)):
            if rois[i].shape[0] == 0:
                # There is no proposal in the single image
                det_bbox = rois[i].new_zeros(0, 5)
                det_label = rois[i].new_zeros((0, ), dtype=torch.long)
                if rcnn_test_cfg is None:
                    det_bbox = det_bbox[:, :4]
                    det_label = rois[i].new_zeros(
                        (0, self.bbox_head.fc_cls.out_features))

            else:
                bboxes, scores = self.bbox_head.get_bboxes(
                    rois[i],
                    cls_score[i],
                    bbox_pred[i],
                    img_shapes[i],
                    scale_factors[i],
                    rescale=rescale)
                det_bbox, det_label, inds = multiclass_nms(bboxes, scores,
                    rcnn_test_cfg.score_thr, rcnn_test_cfg.nms, rcnn_test_cfg.max_per_img, return_inds=True)
                det_score = cls_score[i][inds // rcnn_test_cfg.num_classes]
                det_score = det_score.softmax(-1)

            det_bboxes.append(det_bbox)
            det_labels.append(det_label)
            det_scores.append(det_score)
        return det_bboxes, det_labels, det_scores
