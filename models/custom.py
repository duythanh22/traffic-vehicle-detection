import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection import FasterRCNN
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights


class VehicleDetectionModel(torch.nn.Module):
    def __init__(self, num_classes, pretrained=True):
        super(VehicleDetectionModel, self).__init__()

        # Use MobileNetV3 as backbone
        weights = MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_large(weights=weights)

        # Remove the last block and classifier
        backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))

        # Get the output channels of backbone
        backbone_out = 960  # MobileNetV3-Large's last conv layer output channels

        # Modify anchor sizes and aspect ratios for vehicle detection
        anchor_generator = AnchorGenerator(
            sizes=((32, 64, 128, 256),) * 6,  # Modified for various vehicle sizes
            aspect_ratios=((0.5, 1.0, 2.0),) * 6  # Modified for vehicle shapes
        )

        # ROI pooler
        roi_pooler = torchvision.ops.MultiScaleRoIAlign(
            featmap_names=['0'],  # MobileNet has single feature map
            output_size=7,
            sampling_ratio=2
        )

        # Create FasterRCNN with MobileNet backbone
        self.model = FasterRCNN(
            backbone=backbone,
            num_classes=num_classes,
            rpn_anchor_generator=anchor_generator,
            box_roi_pool=roi_pooler,
            min_size=320,  # MobileNet optimal input size
            max_size=640,
            rpn_pre_nms_top_n_train=2000,  # Increased RPN proposals
            rpn_pre_nms_top_n_test=1000,
            rpn_post_nms_top_n_train=1000,
            rpn_post_nms_top_n_test=500,
            rpn_nms_thresh=0.7,  # Adjusted NMS threshold
            box_score_thresh=0.05,  # Lower threshold to detect more vehicles
            box_nms_thresh=0.5,
            box_detections_per_img=100,  # Increased max detections per image
            rpn_fg_iou_thresh=0.7,  # Adjusted IoU thresholds for better vehicle detection
            rpn_bg_iou_thresh=0.3,
            box_fg_iou_thresh=0.6,
            box_bg_iou_thresh=0.4
        )

        # Replace the box predictor with a new one
        in_features = backbone_out
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Add custom neck (feature pyramid) for better multi-scale detection
        self.fpn = torch.nn.Sequential(
            torch.nn.Conv2d(backbone_out, 256, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, images, targets=None):
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")

        return self.model(images, targets)


def create_model(num_classes, pretrained=True):
    """
    Create and configure the vehicle detection model with MobileNet backbone

    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Whether to use pretrained backbone

    Returns:
        model: Configured vehicle detection model
    """
    model = VehicleDetectionModel(num_classes=num_classes, pretrained=pretrained)

    # Initialize weights for better convergence
    def weights_init(m):
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.GroupNorm)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)

    # Apply weight initialization only to newly added layers
    model.fpn.apply(weights_init)
    model.model.roi_heads.box_predictor.apply(weights_init)

    return model