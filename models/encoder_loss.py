import torch.nn as nn
import torch
from models.losses import DiceLoss, WeightedPixelWiseNLLoss, FocalLoss
from torch import Tensor
from typing import Tuple, Dict


class ADLoss(nn.Module):
    def __init__(self,
                 loss_weights: Tuple,
                 tl_weights: Tuple,
                 seg_loss: str,
                 device: str = 'cuda'):
        """
        @param loss_weights: Loss weights associated to segmentation, traffic light, pedestrian and vehicle
                             affordances loss. In that order.
        @param tl_weights:  Weights associated to traffic light classification. Expected: (green_status_weight,
                            red_status_weight).
        @param seg_loss: Type of segmentation loss.
        @param device: Device used for loss calculation. Expected: cpu or cuda.


        """
        super(ADLoss, self).__init__()

        self._device = device
        if seg_loss == 'dice':
            self._seg_loss = DiceLoss()
        elif seg_loss == 'wnll':
            self._seg_loss = WeightedPixelWiseNLLoss(weights={
                0: 0.3,
                1: 0.1,
                2: 0.15,
                3: 0.1,
                4: 0.1,
                5: 0.05,
                6: 0.5
            })
        else:
            self._seg_loss = FocalLoss()

        self._loss_weights = loss_weights
        self._tl_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(tl_weights, device=self._device))
        self._va_loss = nn.MSELoss()

    def __call__(self, prediction: Dict[str, Tensor], target: Dict[str, Tensor]) -> Dict[str, torch.FloatTensor]:
        """
        Calculates weighted loss.
        @param prediction: Dictionary with predictions. The following format is expected:
                            {
                                'segmentation': torch.Tensor(...),
                                'traffic_light_status': torch.Tensor(...),
                                'vehicle_affordances': torch.Tensor(...)
                            }
        @param target: Dictionary with expected values or ground truth. The following format is expected:
                            {
                                'segmentation': torch.Tensor(...),
                                'traffic_light_status': torch.Tensor(...),
                                'vehicle_affordances': torch.Tensor(...),
                            }
        @return: <class Dict> {
            'loss': torch.FloatTensor,
            'seg_loss': torch.FloatTensor,
            'tl_loss': torch.FloatTensor,
            'va_loss': torch.FloatTensor
        }
        """
        l1 = self._seg_loss(prediction['segmentation'], target['segmentation'])
        l2 = self._tl_loss(prediction['traffic_light_status'], target['traffic_light_status'])
        l3 = self._va_loss(prediction['vehicle_affordances'], target['vehicle_affordances'])
        loss = l1 * self._loss_weights[0] + l2 * self._loss_weights[1] + l3 * self._loss_weights[2]
        return loss
