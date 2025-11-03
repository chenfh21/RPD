import os
import pdb
from typing import Any, Dict, List, Optional, Tuple

import yaml
import pytorch_lightning as pl
import torch
from torch import nn, optim
import torchmetrics

from pdc_modules.losses import get_div_loss_weight, js_div_loss, gjs_div_loss, kl_div_loss


class SegmentationNetwork(pl.LightningModule):
    def __init__(self,
                 network: nn.Module,
                 criterion: nn.Module,
                 learning_rate: float,
                 weight_decay: float,
                 train_step_settings: Optional[List[str]] = None,
                 val_step_settings: Optional[List[str]] = None,
                 test_step_settings: Optional[List[str]] = None,
                 ckpt_path=None):  # 加载骨干网络的预训练权重的检查点文件的可选路径
        super(SegmentationNetwork, self).__init__()

        self.network = network
        self.criterion = criterion
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        # evaluation metrics for all classes
        self.metric_train_iou = torchmetrics.JaccardIndex(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_val_iou = torchmetrics.JaccardIndex(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_test_iou = torchmetrics.JaccardIndex(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)

        # Precision
        self.metric_train_precision = torchmetrics.Precision(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_val_precision = torchmetrics.Precision(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_test_precision = torchmetrics.Precision(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        # F1
        self.metric_train_f1 = torchmetrics.F1Score(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_val_f1 = torchmetrics.F1Score(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_test_f1 = torchmetrics.F1Score(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        # Accuracy
        self.metric_train_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_val_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_test_acc = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        # micro Accuracy
        self.metric_train_acc_micro = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.network.num_classes)
        self.metric_val_acc_micro = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.network.num_classes)
        self.metric_test_acc_micro = torchmetrics.Accuracy(
            task='multiclass',
            num_classes=self.network.num_classes)
        # Recall
        self.metric_train_recall = torchmetrics.Recall(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_val_recall = torchmetrics.Recall(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)
        self.metric_test_recall = torchmetrics.Recall(
            task='multiclass',
            num_classes=self.network.num_classes,
            average=None)

        if train_step_settings is not None:
            self.train_step_settings = train_step_settings
        else:
            self.train_step_settings = []

        if val_step_settings is not None:
            self.val_step_settings = val_step_settings
        else:
            self.val_step_settings = []

        if test_step_settings is not None:
            self.test_step_settings = test_step_settings
        else:
            self.test_step_settings = []

        if ckpt_path is not None:
            print("Load pretrained weights of backbone.")
            ckpt_dict = torch.load(ckpt_path)
            self.load_state_dict(ckpt_dict['state_dict'], strict=False)

        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []

    def compute_loss(self, logits: torch.Tensor, y: torch.Tensor, mode: str,
                     mask_keep: Optional[torch.Tensor] = None) -> torch.Tensor:
        """ Compute cross entropy loss based on logits and ground-truths.

        Args:
            logits (torch.Tensor): logits [B x num_classes x H x W]
            y (torch.Tensor): ground-truth [B x H x W]
            mode (str): train, val, test
            mask_keep (Optional[torch.Tensor]， optional): 1:= consider annotation, 0 := do not consider annotation
                                                          [B x H x W]. Defaults to None.

        Returns:
            torch.Tensor: loss
        """
        return self.criterion(logits, y, mode=mode, mask_keep=mask_keep)

    def forward(self, img_batch: torch.Tensor) -> torch.Tensor:
        """ Forward pass of backbone network.

        Args:
            img_batch (torch.Tensor): input image(s) [B x C x H x W]

        Returns:
            torch.Tensor, torch.Tensor: predictions of segmentation head [B x num_classes x H x W]
        """
        output = self.network.forward(img_batch)

        return output

    def training_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
        if not self.train_step_settings:
            raise ValueError('You need to specify the settings for the training step.')

        processed_steps = []
        if 'regular' in self.train_step_settings:
            logits_regular = self.forward(batch['input_image'])

            mask_keep_regular = batch['anno'] != 255
            loss_regular = self.compute_loss(logits_regular, batch['anno'], mode='train',
                                             mask_keep=mask_keep_regular)

            processed_steps.append('regular')
            # update training metrics
            pred = torch.argmax(logits_regular.detach(), dim=1)
            self.metric_train_iou.update(pred, batch['anno'])
            self.metric_train_precision.update(pred, batch['anno'])
            self.metric_train_f1.update(pred, batch['anno'])
            self.metric_train_acc.update(pred, batch['anno'])
            self.metric_train_acc_micro.update(pred, batch['anno'])
            self.metric_train_recall.update(pred, batch['anno'])

        if (self.trainer.current_epoch == 0) and (batch_idx == 0):
            print("Processed steps during training：", processed_steps)

        # accumulate all losses
        my_local_loss_values = {var_name: var_value for var_name, var_value in locals().items() if
                                var_name.startswith('loss')}
        loss = torch.sum(torch.stack(list(my_local_loss_values.values())))

        out_dict = {'loss': loss, 'logits': logits_regular, 'anno': batch['anno']}
        self.training_step_outputs.append(out_dict)
        out_dict.update(my_local_loss_values)
        del my_local_loss_values

        return out_dict

    def on_train_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch

        # compute loss(es) over all batches and log
        dict_keys = self.training_step_outputs[0].keys()
        for key in dict_keys:
            if key.startswith('loss'):
                loss_accumulated = torch.stack([x[key] for x in self.training_step_outputs])
                loss_avg = loss_accumulated.mean().detach()
                self.logger.experiment.add_scalars(f'{key}', {'train': loss_avg}, epoch)

        # the following logging is required *ModelCheckpoint* to work as expected
        losses = torch.stack([x['loss'] for x in self.training_step_outputs])
        train_loss_avg = losses.mean().detach()
        self.log("train_loss", train_loss_avg, on_epoch=True, sync_dist=True)

        # compute final metrics over all batches
        iou_per_class = self.metric_train_iou.compute().detach()
        precision_per_class = self.metric_train_precision.compute().detach()
        f1_per_class = self.metric_train_f1.compute().detach()
        acc_per_class = self.metric_train_acc.compute().detach()
        acc_micro = self.metric_train_acc_micro.compute().detach()
        recall_per_class = self.metric_train_recall.compute().detach()

        mIoU = iou_per_class.mean()
        mPrecision = precision_per_class.mean()
        mF1 = f1_per_class.mean()
        mAcc = acc_per_class.mean()
        mRecall = recall_per_class.mean()

        self.metric_train_iou.reset()
        self.metric_train_precision.reset()
        self.metric_train_f1.reset()
        self.metric_train_acc.reset()
        self.metric_train_acc_micro.reset()
        self.metric_train_recall.reset()

        for class_index, iou_class in enumerate(iou_per_class):
            self.logger.experiment.add_scalars(f'iou_class_{class_index}', {'train': iou_class}, epoch)
            self.logger.experiment.add_scalars(f'precision_class_{class_index}',
                                               {'train': precision_per_class[class_index]}, epoch)
            self.logger.experiment.add_scalars(f'f1_class_{class_index}', {'train': f1_per_class[class_index]}, epoch)
            self.logger.experiment.add_scalars(f'acc_class_{class_index}', {'train': acc_per_class[class_index]}, epoch)
            self.logger.experiment.add_scalars(f'recall_class_{class_index}', {'train': recall_per_class[class_index]},
                                               epoch)

        self.logger.experiment.add_scalars('mIoU', {'train': mIoU}, epoch)
        self.logger.experiment.add_scalars('mPrecision', {'train': mPrecision}, epoch)
        self.logger.experiment.add_scalars('mF1', {'train': mF1}, epoch)
        self.logger.experiment.add_scalars('mAcc', {'train': mAcc}, epoch)
        self.logger.experiment.add_scalars('OverallAcc', {'train': acc_micro}, epoch)
        self.logger.experiment.add_scalars('mRecall', {'train': mRecall}, epoch)

        self.log('train_mIoU', mIoU, on_epoch=True, sync_dist=True)
        self.log('train_mPrecision', mPrecision, on_epoch=True, sync_dist=True)
        self.log('train_mF1', mF1, on_epoch=True, sync_dist=True)
        self.log('train_mAcc', mAcc, on_epoch=True, sync_dist=True)
        self.log('train_OverallAcc', acc_micro, on_epoch=True, sync_dist=True)
        self.log('train_mRecall', mRecall, on_epoch=True, sync_dist=True)

        self.training_step_outputs.clear()

    def validation_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
        if not self.val_step_settings:
            raise ValueError('You need to specify the settings for the validation step.')
        assert len(self.val_step_settings) == 1

        # predictions
        if 'regular' in self.val_step_settings:
            logits = self.forward(batch['input_image'])

        # objective
        loss = self.compute_loss(logits, batch['anno'], mode='val')
        # self.validation_step_outputs.append({'loss': loss})

        # update validation metrics
        pred = torch.argmax(logits, dim=1)
        self.metric_val_iou.update(pred, batch['anno'])  # IoU
        self.metric_val_precision.update(pred, batch['anno'])  # Precision
        self.metric_val_f1.update(pred, batch['anno'])  # F1
        self.metric_val_acc.update(pred, batch['anno'])  # Accuracy
        self.metric_val_acc_micro.update(pred, batch['anno'])  # micro Accuracy
        self.metric_val_recall.update(pred, batch['anno'])  # Sensitivity/Recall

        validation_out = {'loss': loss, 'logits': logits, 'anno': batch['anno']}
        self.validation_step_outputs.append(validation_out)

        return validation_out

    def on_validation_epoch_end(self) -> None:
        # compute loss over all batches
        losses = torch.stack([x['loss'] for x in self.validation_step_outputs])
        val_loss_avg = losses.mean()

        # logging
        epoch = self.trainer.current_epoch
        self.logger.experiment.add_scalars('loss', {'val': val_loss_avg}, epoch)
        self.log('val_loss', val_loss_avg, on_epoch=True, sync_dist=True)

        # compute final metrics over all batches
        iou_per_class = self.metric_val_iou.compute()
        precision_per_class = self.metric_val_precision.compute()
        f1_per_class = self.metric_val_f1.compute()
        acc_per_class = self.metric_val_acc.compute()
        acc_micro = self.metric_val_acc_micro.compute()
        recall_per_class = self.metric_val_recall.compute()

        mIoU = iou_per_class.mean()
        mPrecision = precision_per_class.mean()
        mF1 = f1_per_class.mean()
        mAcc = acc_per_class.mean()
        mRecall = recall_per_class.mean()

        self.metric_val_iou.reset()
        self.metric_val_precision.reset()
        self.metric_val_f1.reset()
        self.metric_val_acc.reset()
        self.metric_val_acc_micro.reset()
        self.metric_val_recall.reset()

        for class_index, iou_class in enumerate(iou_per_class):
            self.logger.experiment.add_scalars(f'iou_class_{class_index}', {'val': iou_class}, epoch)
            self.logger.experiment.add_scalars(f'precision_class_{class_index}',
                                               {'val': precision_per_class[class_index]}, epoch)
            self.logger.experiment.add_scalars(f'f1_class_{class_index}', {'val': f1_per_class[class_index]}, epoch)
            self.logger.experiment.add_scalars(f'acc_class_{class_index}', {'val': acc_per_class[class_index]}, epoch)
            self.logger.experiment.add_scalars(f'recall_class_{class_index}', {'val': recall_per_class[class_index]},
                                               epoch)

        self.logger.experiment.add_scalars('mIoU', {'val': mIoU}, epoch)
        self.logger.experiment.add_scalars('mPrecision', {'val': mPrecision}, epoch)
        self.logger.experiment.add_scalars('mF1', {'val': mF1}, epoch)
        self.logger.experiment.add_scalars('mAcc', {'val': mAcc}, epoch)
        self.logger.experiment.add_scalars('OverallAcc', {'val': acc_micro}, epoch)
        self.logger.experiment.add_scalars('mRecall', {'val': mRecall}, epoch)

        self.log('val_mIoU', mIoU, on_epoch=True, sync_dist=True)
        self.log('val_mPrecision', mPrecision, on_epoch=True, sync_dist=True)
        self.log('val_mF1', mF1, on_epoch=True, sync_dist=True)
        self.log('val_mAcc', mAcc, on_epoch=True, sync_dist=True)
        self.log('val_OverallAcc', acc_micro, on_epoch=True, sync_dist=True)
        self.log('val_mRecall', mRecall, on_epoch=True, sync_dist=True)

        path_to_classwise_dir = os.path.join(self.trainer.log_dir, 'val', 'evaluation', f'epoch-{epoch:06d}')
        save_metric(iou_per_class, path_to_classwise_dir, 'IoU')
        save_metric(precision_per_class, path_to_classwise_dir, 'Precision')
        save_metric(f1_per_class, path_to_classwise_dir, 'F1')
        save_metric(acc_per_class, path_to_classwise_dir, 'Accuracy')
        save_metric(recall_per_class, path_to_classwise_dir, 'Recall')

        self.validation_step_outputs.clear()

    def test_step(self, batch: dict, batch_idx: int) -> Dict[str, Any]:
        if not self.test_step_settings:
            raise ValueError('You need to specify the settings for the test step.')
        assert len(self.test_step_settings) == 1

        # predictions
        if 'regular' in self.test_step_settings:
            logits = self.forward(batch['input_image'])

        pred_cls = torch.argmax(logits, dim=1)

        # regular evaluation for all classes
        self.metric_test_iou.update(pred_cls, batch['anno'])
        self.metric_test_precision.update(pred_cls, batch['anno'])
        self.metric_test_f1.update(pred_cls, batch['anno'])
        self.metric_test_acc.update(pred_cls, batch['anno'])
        self.metric_test_acc_micro.update(pred_cls, batch['anno'])
        self.metric_test_recall.update(pred_cls, batch['anno'])

        test_out = {'logits': logits, 'anno': batch['anno']}
        self.test_step_outputs.append(test_out)

        return test_out

    def on_test_epoch_end(self) -> None:
        # compute final metrics over all batches
        iou_per_classes = self.metric_test_iou.compute()
        precision_per_classes = self.metric_test_precision.compute()
        f1_per_classes = self.metric_test_f1.compute()
        acc_per_classes = self.metric_test_acc.compute()
        recall_per_classes = self.metric_test_recall.compute()

        mIoU = round(float(iou_per_classes.mean().cpu()), 3)
        mPrecision = round(float(precision_per_classes.mean().cpu()), 3)
        mF1 = round(float(f1_per_classes.mean().cpu()), 3)
        mAcc = round(float(acc_per_classes.mean().cpu()), 3)
        mRecall = round(float(recall_per_classes.mean().cpu()), 3)

        print(f'Test mIoU: {mIoU}')
        print(f'Test mPrecision: {mPrecision}')
        print(f'Test mF1: {mF1}')
        print(f'Test mAcc: {mAcc}')
        print(f'Test mRecall: {mRecall}')

        self.metric_test_iou.reset()
        self.metric_test_precision.reset()
        self.metric_test_f1.reset()
        self.metric_test_acc.reset()
        self.metric_test_recall.reset()

        # logging
        epoch = self.trainer.current_epoch

        path_to_classwise_dir = os.path.join(self.trainer.log_dir, 'evaluation', f'epoch-{epoch:06d}')

        save_metric(iou_per_classes, path_to_classwise_dir, 'IoU')
        save_metric(precision_per_classes, path_to_classwise_dir, 'Precision')
        save_metric(f1_per_classes, path_to_classwise_dir, 'F1')
        save_metric(acc_per_classes, path_to_classwise_dir, 'Accuracy')
        save_metric(recall_per_classes, path_to_classwise_dir, 'Recall')

        self.test_step_outputs.clear()

    def lr_scaling(self, current_epoch: int) -> float:
        warm_up_epochs = 16
        if current_epoch <= warm_up_epochs:
            lr_scale = current_epoch / warm_up_epochs
        else:
            lr_scale = pow(
                (1 - ((current_epoch - (warm_up_epochs + 1)) / (self.trainer.max_epochs - (warm_up_epochs + 1)))), 3.0)

        return lr_scale

    def configure_optimizers(self) -> Tuple[List[optim.Optimizer], List[optim.lr_scheduler.LambdaLR]]:
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)

        # lambdal = lambda epoch: pow((1 - 1(epoch / self.trainer.max_epochs)), 3.0) # 1.25
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.lr_scaling)

        return [optimizer], [scheduler]


def save_metric(metrics: torch.Tensor, path_to_dir: str, name: str) -> None:
    if not os.path.exists(path_to_dir):
        os.makedirs(path_to_dir, exist_ok=True)

    info = {}
    for cls_index, metric in enumerate(metrics):
        info[f'class_{cls_index}'] = round(float(metric), 5)

    info[f'm{name}'] = round(float(metrics.mean()), 5)

    fpath = os.path.join(path_to_dir, f"{name}.yaml")  # 这里会自动创建该文件
    with open(fpath, 'w') as ostream:
        yaml.dump(info, ostream)
