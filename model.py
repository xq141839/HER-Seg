from SAM.LGViT import LGViT
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from SAM.image_encoder import ImageEncoderViT
from SAM.ecm_decoder import MaskDecoder
from SAM.prompt_encoder import PromptEncoder
from SAM.transformer import TwoWayTransformer
from SAM.common import LayerNorm2d
from typing import List, Tuple, Type, Optional
from transforms import ResizeLongestSide
import os
from tqdm import tqdm
from functools import partial
from sam2.modeling.sam2_base import SAM2Base
from sam2.modeling.position_encoding import PositionEmbeddingSine
from sam2.utils.transforms import SAM2Transforms

def PointGenerator(mask, visual=False):
    np.random.seed(42)
    point_coord = []
    point_class = []
    box_coord = []

    if visual!=True:

        mask = mask.cpu().detach().numpy()
    else:
        mask[mask < 255] = 0

    mask_shape = mask.shape
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8)*255)
    for i in range(1, num_labels):
        row, col = int(centroids[i][0]), int(centroids[i][1])
        box_coord.append([stats[i][0], stats[i][1], stats[i][0]+stats[i][2], stats[i][1]+stats[i][3]])
        point_coord.append([row, col])
        point_class.append(1)

    return point_coord, point_class, mask_shape, box_coord

class MHRMedSeg(nn.Module):
    def __init__(self, dim=96, img_size=1024):
        super(MHRMedSeg, self).__init__()

        self.img_size = img_size
        self.pt = ResizeLongestSide(img_size)

        self.image_encoder = LGViT(embed_dim=dim, img_size=img_size)

        self.prompt_encoder = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(img_size // 16, img_size // 16), # 1024 // 16
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            )
        
        self.mask_decoder = MaskDecoder(
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=256,
                num_heads=8,
            ),
            transformer_dim=256,
            )
        
        self.neck = nn.Sequential(
            nn.Conv2d(
                dim,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )
  
    def forward(self, x, mask=None, domain_seq=None, img_id=None):

        b = x.shape[0]

        image_embeddings = self.image_encoder(x)

        B, _, C = image_embeddings.size()
        image_embeddings = image_embeddings.view(B, self.img_size // 16, self.img_size // 16, C)
        image_embeddings = image_embeddings.permute(0, 3, 1, 2)
        image_embeddings = self.neck(image_embeddings)
        
        outputs_mask = []

        for idx in range(b): # for each batch 

            # point_coord, point_class, inst_shape, box_coord = PointGenerator(mask[idx][0])
            # point_coord = torch.tensor(point_coord)
            # point_class = torch.tensor(point_class)
            # point_coord_trans = self.pt.apply_coords_torch(point_coord, inst_shape)

            # point = (point_coord_trans.unsqueeze(0).cuda(), point_class.unsqueeze(0).cuda())
            # bboxes = torch.tensor(box_coord).cuda()

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points=None,
                boxes=None,
                masks=None,
            )
   
            low_res_masks = self.mask_decoder(
                image_embeddings=image_embeddings[idx].unsqueeze(0),
                image_pe=self.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )

            masks = F.interpolate(low_res_masks, (self.img_size, self.img_size), mode="bilinear", align_corners=False)

            outputs_mask.append(masks.squeeze(0))


        return torch.stack(outputs_mask, dim=0)
    

class PFD(nn.Module):
    def __init__(self, dim=192, img_size=1024):
        super(PFD, self).__init__()

        self.img_size = img_size

        self.image_encoder = PFD(embed_dim=dim, img_size=img_size)        
        
        self.neck = nn.Sequential(
            nn.Conv2d(
                dim,
                256,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm2d(256),
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm2d(256),
        )


  
    def forward(self, x, mask=None, domain_seq=None, img_id=None):

        b = x.shape[0]

        image_embeddings = self.image_encoder(x)

        B, _, C = image_embeddings.size()
        image_embeddings = image_embeddings.view(B, self.img_size // 16, self.img_size // 16, C)
        image_embeddings = image_embeddings.permute(0, 3, 1, 2)
        image_embeddings = self.neck(image_embeddings)

        return image_embeddings


class PFDTeacher(nn.Module):
    def __init__(self, dim=192, img_size=224):
        super(PFDTeacher, self).__init__()

        self.image_encoder = ImageEncoderViT(
            depth=32,
            embed_dim=1280,
            img_size=1024,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=16,
            patch_size=16,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=[7, 15, 23, 31],
            window_size=14,
            out_chans=256
        )

        # self.image_encoder = ImageEncoderViT(
        #     depth=12,
        #     embed_dim=768,
        #     img_size=1024,
        #     mlp_ratio=4,
        #     norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
        #     num_heads=12,
        #     patch_size=16,
        #     qkv_bias=True,
        #     use_rel_pos=True,
        #     global_attn_indexes=[2, 5, 8, 11],
        #     window_size=14,
        #     out_chans=256
        # )

  
    def forward(self, x, mask=None, domain_seq=None, img_id=None):
        
        return self.image_encoder(x) 
    

class MHRMedSegSAM2(nn.Module):
    def __init__(self, sam_model = SAM2Base, dim=24, img_size=1024):
        super(MHRMedSegSAM2, self).__init__()

        self.device = "cuda"
        self._transforms = SAM2Transforms(
            resolution=img_size,
            mask_threshold=0.0,
            max_hole_area=0.0,
            max_sprinkle_area=0.0,
        )
        self.model = sam_model
        self._features = None
        self._bb_feat_sizes = [
            (256, 256),
            (128, 128),
            (64, 64),
        ]

        self.image_encoder = LGViT(embed_dim=dim, img_size=img_size)
        self.neck = FpnNeck(position_encoding=PositionEmbeddingSine(num_pos_feats=256, normalize=True, temperature=10000),
                            d_model=256, backbone_channel_list=[192, 96, 48, 24], fpn_top_down_levels=[2,3], fpn_interp_model="nearest")

    def _prep_prompts(
        self, point_coords, point_labels, box, mask_logits, normalize_coords, img_idx=-1
    ):

        unnorm_coords, labels, unnorm_box, mask_input = None, None, None, None
        if point_coords is not None:
            assert (
                point_labels is not None
            ), "point_labels must be supplied if point_coords is supplied."
            point_coords = torch.as_tensor(
                point_coords, dtype=torch.float, device=self.device
            )
            unnorm_coords = self._transforms.transform_coords(
                point_coords, normalize=normalize_coords, orig_hw=(1024, 1024) # fixed
            )
            labels = torch.as_tensor(point_labels, dtype=torch.int, device=self.device)
            if len(unnorm_coords.shape) == 2:
                unnorm_coords, labels = unnorm_coords[None, ...], labels[None, ...]
        if box is not None:
            box = torch.as_tensor(box, dtype=torch.float, device=self.device)
            unnorm_box = self._transforms.transform_boxes(
                box, normalize=normalize_coords, orig_hw=(1024, 1024)
            )  # Bx2x2
        if mask_logits is not None:
            mask_input = torch.as_tensor(
                mask_logits, dtype=torch.float, device=self.device
            )
            if len(mask_input.shape) == 3:
                mask_input = mask_input[None, :, :, :]
        return mask_input, unnorm_coords, labels, unnorm_box

    def _predict(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        boxes: Optional[torch.Tensor] = None,
        mask_input: Optional[torch.Tensor] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        img_idx: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Predict masks for the given input prompts, using the currently set image.
        Input prompts are batched torch tensors and are expected to already be
        transformed to the input frame using SAM2Transforms.

        Arguments:
          point_coords (torch.Tensor or None): A BxNx2 array of point prompts to the
            model. Each point is in (X,Y) in pixels.
          point_labels (torch.Tensor or None): A BxN array of labels for the
            point prompts. 1 indicates a foreground point and 0 indicates a
            background point.
          boxes (np.ndarray or None): A Bx4 array given a box prompt to the
            model, in XYXY format.
          mask_input (np.ndarray): A low resolution mask input to the model, typically
            coming from a previous prediction iteration. Has form Bx1xHxW, where
            for SAM, H=W=256. Masks returned by a previous iteration of the
            predict method do not need further transformation.
          multimask_output (bool): If true, the model will return three masks.
            For ambiguous input prompts (such as a single click), this will often
            produce better masks than a single prediction. If only a single
            mask is needed, the model's predicted quality score can be used
            to select the best mask. For non-ambiguous prompts, such as multiple
            input prompts, multimask_output=False can give better results.
          return_logits (bool): If true, returns un-thresholded masks logits
            instead of a binary mask.

        Returns:
          (torch.Tensor): The output masks in BxCxHxW format, where C is the
            number of masks, and (H, W) is the original image size.
          (torch.Tensor): An array of shape BxC containing the model's
            predictions for the quality of each mask.
          (torch.Tensor): An array of shape BxCxHxW, where C is the number
            of masks and H=W=256. These low res logits can be passed to
            a subsequent iteration as mask input.
        """

        if point_coords is not None:
            concat_points = (point_coords, point_labels)
        else:
            concat_points = None

        # Embed prompts
        if boxes is not None:
            box_coords = boxes.reshape(-1, 2, 2)
            box_labels = torch.tensor([[2, 3]], dtype=torch.int, device=boxes.device)
            box_labels = box_labels.repeat(boxes.size(0), 1)
            # we merge "boxes" and "points" into a single "concat_points" input (where
            # boxes are added at the beginning) to sam_prompt_encoder
            if concat_points is not None:
                concat_coords = torch.cat([box_coords, concat_points[0]], dim=1)
                concat_labels = torch.cat([box_labels, concat_points[1]], dim=1)
                concat_points = (concat_coords, concat_labels)
            else:
                concat_points = (box_coords, box_labels)

        sparse_embeddings, dense_embeddings = self.model.sam_prompt_encoder(
            points=concat_points,
            boxes=None,
            masks=mask_input,
        )

        # Predict masks
        batched_mode = (
            concat_points is not None and concat_points[0].shape[0] > 1
        )  # multi object prediction
        high_res_features = [
            feat_level[img_idx].unsqueeze(0)
            for feat_level in self._features["high_res_feats"]
        ]
        low_res_masks, iou_predictions, _, _ = self.model.sam_mask_decoder(
            image_embeddings=self._features["image_embed"][img_idx].unsqueeze(0),
            image_pe=self.model.sam_prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=multimask_output,
            repeat_image=batched_mode,
            high_res_features=high_res_features,
        )

        masks = F.interpolate(low_res_masks, (1024, 1024), mode="bilinear", align_corners=False)
        return masks



  
    def forward(self, x, gt=None, domain_seq=None, img_id=None):

        batch_size = x.shape[0]

        image_embeddings = self.image_encoder(x)
        features, pos = self.neck(image_embeddings)
        features, pos = features[: -1], pos[: -1]
        src = features[-1]
        backbone_out = {
            "vision_features": src,
            "vision_pos_enc": pos,
            "backbone_fpn": features,
        }
        backbone_out = self.model.forward_image(backbone_out)

        _, vision_feats, _, _ = self.model._prepare_backbone_features(backbone_out)

        vision_feats[-1] = vision_feats[-1] + self.model.no_mem_embed
        

        feats = [
            feat.permute(1, 2, 0).view(batch_size, -1, *feat_size)
            for feat, feat_size in zip(vision_feats[::-1], self._bb_feat_sizes[::-1])
        ][::-1]

        self._features = {"image_embed": feats[-1], "high_res_feats": feats[:-1]} # image encoder end

        num_images = len(self._features["image_embed"])
        all_masks = []
        all_ious = []
        all_low_res_masks = []
        outputs_mask = []
        normalize_coords = True
        return_logits = False
        multimask_output = False
        for img_idx in range(num_images):


            point_coord, point_class, inst_shape, box_coord = PointGenerator(gt[img_idx][0])
            point_coord = torch.tensor(point_coord)
            point_class = torch.tensor(point_class)

            bboxes = torch.tensor(box_coord).cuda()
            # Transform input prompts
            point_coords = None
            point_labels = None
            box = bboxes
            mask_input = None
            mask_input, unnorm_coords, labels, unnorm_box = self._prep_prompts(
                point_coords,
                point_labels,
                box,
                mask_input,
                normalize_coords,
                img_idx=img_idx,
            )

            masks = self._predict(
                unnorm_coords,
                labels,
                unnorm_box,
                mask_input,
                multimask_output,
                return_logits=return_logits,
                img_idx=img_idx,
            )

            outputs_mask.append(masks.squeeze(0))

        return torch.stack(outputs_mask, dim=0)


class FpnNeck(nn.Module):
    """
    A modified variant of Feature Pyramid Network (FPN) neck
    (we remove output conv and also do bicubic interpolation similar to ViT
    pos embed interpolation)
    """

    def __init__(
        self,
        position_encoding: nn.Module,
        d_model: int,
        backbone_channel_list: List[int],
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        fpn_interp_model: str = "bilinear",
        fuse_type: str = "sum",
        fpn_top_down_levels: Optional[List[int]] = None,
    ):
        """Initialize the neck
        :param trunk: the backbone
        :param position_encoding: the positional encoding to use
        :param d_model: the dimension of the model
        :param neck_norm: the normalization to use
        """
        super().__init__()
        self.position_encoding = position_encoding
        self.convs = nn.ModuleList()
        self.backbone_channel_list = backbone_channel_list
        for dim in backbone_channel_list:
            current = nn.Sequential()
            current.add_module(
                "conv",
                nn.Conv2d(
                    in_channels=dim,
                    out_channels=d_model,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                ),
            )

            self.convs.append(current)
        self.fpn_interp_model = fpn_interp_model
        assert fuse_type in ["sum", "avg"]
        self.fuse_type = fuse_type

        # levels to have top-down features in its outputs
        # e.g. if fpn_top_down_levels is [2, 3], then only outputs of level 2 and 3
        # have top-down propagation, while outputs of level 0 and level 1 have only
        # lateral features from the same backbone level.
        if fpn_top_down_levels is None:
            # default is to have top-down features on all levels
            fpn_top_down_levels = range(len(self.convs))
        self.fpn_top_down_levels = list(fpn_top_down_levels)

    def forward(self, xs: List[torch.Tensor]):

        out = [None] * len(self.convs)
        pos = [None] * len(self.convs)
        assert len(xs) == len(self.convs)
        # fpn forward pass
        # see https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/backbone/fpn.py
        prev_features = None
        # forward in top-down order (from low to high resolution)
        n = len(self.convs) - 1
        for i in range(n, -1, -1):
            x = xs[i]
            lateral_features = self.convs[n - i](x)
            if i in self.fpn_top_down_levels and prev_features is not None:
                top_down_features = F.interpolate(
                    prev_features.to(dtype=torch.float32),
                    scale_factor=2.0,
                    mode=self.fpn_interp_model,
                    align_corners=(
                        None if self.fpn_interp_model == "nearest" else False
                    ),
                    antialias=False,
                )
                prev_features = lateral_features + top_down_features
                if self.fuse_type == "avg":
                    prev_features /= 2
            else:
                prev_features = lateral_features
            x_out = prev_features
            out[i] = x_out
            pos[i] = self.position_encoding(x_out).to(x_out.dtype)

        return out, pos


        
    

