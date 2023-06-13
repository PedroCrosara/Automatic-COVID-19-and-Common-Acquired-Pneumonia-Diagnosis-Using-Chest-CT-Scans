import segmentation_models_pytorch as smp
import pytorch_lightning as pl
import torch

class LungModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # preprocessing parameteres for image
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # for image segmentation dice loss could be the best first choice
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)
        
        self.save_hyperparameters()

    def forward(self, image):
        # normalize image here
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        
        image = batch["image"]

        # Shape of the image should be (batch_size, num_channels, height, width)
        # if you work with grayscale images, expand channels dim to have [batch_size, 1, height, width]
        assert image.ndim == 4

        # Check that image dimensions are divisible by 32, 
        # encoder and decoder connected by `skip connections` and usually encoder have 5 stages of 
        # downsampling by factor 2 (2 ^ 5 = 32); e.g. if we have image with shape 65x65 we will have 
        # following shapes of features in encoder and decoder: 84, 42, 21, 10, 5 -> 5, 10, 20, 40, 80
        # and we will get an error trying to concat these features
        h, w = image.shape[2:]
        assert h % 32 == 0 and w % 32 == 0

        mask = batch["mask"]

        # Shape of the mask should be [batch_size, num_classes, height, width]
        # for binary segmentation num_classes = 1
        assert mask.ndim == 4

        # Check that mask values in between 0 and 1, NOT 0 and 255 for binary segmentation
        assert mask.max() <= 1.0 and mask.min() >= 0

        logits_mask = self.forward(image)
        
        # Predicted mask contains logits, and loss_fn param `from_logits` is set to True
        loss = self.loss_fn(logits_mask, mask)

        # Lets compute metrics for some threshold
        # first convert mask values to probabilities, then 
        # apply thresholding
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()

        # We will compute IoU metric by two ways
        #   1. dataset-wise
        #   2. image-wise
        # but for now we just compute true positive, false positive, false negative and
        # true negative 'pixels' for each image and class
        # these values will be aggregated in the end of an epoch
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        # aggregate step metics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # per image IoU means that we first calculate IoU score for each image 
        # and then compute mean over these scores
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        
        # dataset IoU means that we aggregate intersection and union over whole dataset
        # and then compute IoU score. The difference between dataset_iou and per_image_iou scores
        # in this particular case will not be much, however for dataset 
        # with "empty" images (images without target class) a large gap could be observed. 
        # Empty images influence a lot on per_image_iou and much less on dataset_iou.
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        
        accuracy = smp.metrics.accuracy(tp, fp, fn, tn, reduction="macro")
        f1_score = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_accuracy": accuracy,
            f"{stage}_f1_score": f1_score,
            
        }
        
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")            

    def training_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")  

    def test_epoch_end(self, outputs):
        return self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.0001)


import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
import os
import cv2
import numpy as np

class CustomAuxLungDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]
        
        return image

import logging


def get_lungs(images, resize=(224,224), return_masks=False, weight=1):
    '''
    input format should be (16,512,512,3)
    '''
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    
    #ckpt_path = './weights/' + os.listdir('./weights')[-1]
    ckpt_path = '/media/riccelli/Disco 1/datasets_covid/weights_vh/lung_segmentation/lung-paper-se_resnext101_32x4d-UnetPlusPlus-2-4304-250.ckpt'
    
    model = LungModel.load_from_checkpoint(ckpt_path, 
                                       ) 
    aux_transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])
    
    aux_dataset = CustomAuxLungDataset(images, aux_transforms)
    aux_dataloader = DataLoader(aux_dataset, batch_size=16, shuffle=False, num_workers=20)
    
    trainer = pl.Trainer(accelerator='gpu', devices=1, enable_progress_bar = False)
    predictions = trainer.predict(model, aux_dataloader)
    #predictions = torch.stack(predictions)
    predictions = torch.cat(predictions)
    pr_masks = (predictions.sigmoid() > 0.5).int().squeeze(0)
       
    results = []
    results_masks = []
    for image, pr_mask in zip(images, pr_masks):
        msk = pr_mask.numpy().astype(np.uint8).squeeze()
        #print(msk.shape)
        aux = np.array([msk, msk, msk]).transpose(1,2,0)
        aux = cv2.resize(aux, (512,512))
        results_masks.append(aux*255)
        seg = image*aux
        #positions = np.nonzero(seg)

        # top = positions[0].min()
        # bottom = positions[0].max()
        # left = positions[1].min()
        # right = positions[1].max()
        #bigger_side = max(bottom - top, right - left)
        #seg = seg[top:bottom, left:right, :]
        #print(seg.shape)
        #seg = seg[top:top+bigger_side, left:left+bigger_side, :]
        if resize:
            seg = cv2.resize(seg, (224,224))
        #plt.imshow(seg)
        #plt.show()
        results.append(seg)
    
    if return_masks:
        return np.array(results), np.array(results_masks)
    else:
        return np.array(results)

def get_lungs_for_lesion(images, resize=(224,224)):
    '''
    input format should be (16,512,512,3)
    '''
    #ckpt_path = './weights/' + os.listdir('./weights')[-1]
    ckpt_path = './weights/resnet50-Unet-4-valid_accuracy=1.00-valid_dataset_iou=0.97-valid_f1_score=0.99-250.ckpt'
    model = LungModel.load_from_checkpoint(ckpt_path, 
                                       ) 
    aux_transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])
    
    aux_dataset = CustomAuxLungDataset(images, aux_transforms)
    aux_dataloader = DataLoader(aux_dataset, batch_size=16, shuffle=False)
    
    trainer = pl.Trainer()
    predictions = trainer.predict(model, aux_dataloader)
    print(len(predictions))
    print(predictions[0].shape)
    predictions = torch.stack(predictions)
    pr_masks = (predictions.sigmoid() > 0.5).int().squeeze(0)
       
    results = []
    for image, pr_mask in zip(images, pr_masks):
        msk = pr_mask.numpy().astype(np.uint8).squeeze()
        aux = np.array([msk, msk, msk]).transpose(1,2,0)
        aux = cv2.resize(aux, (512,512))
        seg = image*aux
        positions = np.nonzero(seg)

        top = positions[0].min()
        bottom = positions[0].max()
        left = positions[1].min()
        right = positions[1].max()
        #bigger_side = max(bottom - top, right - left)
        #seg = seg[top:bottom, left:right, :]
        #print(seg.shape)
        #seg = seg[top:top+bigger_side, left:left+bigger_side, :]
        if resize:
            seg = cv2.resize(seg, (224,224))
        #plt.imshow(seg)
        #plt.show()
        results.append(seg)
    
    return np.array(results)
