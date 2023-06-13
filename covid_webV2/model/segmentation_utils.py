from torch.utils.data import Dataset, DataLoader
import logging
from model.segmentation_model import SegmentationModel
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl
import numpy as np
import cv2
import streamlit as st
from config import get_lung_segmentation_ckpt, get_lesion_segmentation_ckpt

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
    
def get_lungs(images, resize=(224,224), return_masks=False, weight=1):
    '''
    input format should be (16,512,512,3)
    '''
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
    
    ckpt_path = get_lung_segmentation_ckpt()
    model = SegmentationModel.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu")
                                       ) 
    aux_transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])
    
    aux_dataset = CustomAuxLungDataset(images, aux_transforms)
    aux_dataloader = DataLoader(aux_dataset, batch_size=1, shuffle=False, num_workers=8)
    
    trainer = pl.Trainer(accelerator='cpu', enable_progress_bar = False)
    predictions = trainer.predict(model, aux_dataloader)
    predictions = torch.cat(predictions)
    pr_masks = (predictions.sigmoid() > 0.5).int().squeeze(0)
       
    results = []
    results_masks = []
    slice_lung_area = []
    right_lung_area_array = []
    left_lung_area_array = []
    total_left_lung_area = 0
    total_right_lung_area = 0

    for image, pr_mask in zip(images, pr_masks):
        msk = pr_mask.numpy().astype(np.uint8).squeeze()
        aux = np.array([msk, msk, msk]).transpose(1,2,0)
        aux = cv2.resize(aux, (512,512))
        results_masks.append(aux*255)
        seg = image*aux
        if resize:
            seg = cv2.resize(seg, (224,224))
        results.append(seg)
        slice_lung_area.append(np.count_nonzero(aux))

        contours_lung, hierarchy = cv2.findContours(aux[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        left_lung = np.zeros((aux.shape[0],aux.shape[1],3),dtype = np.uint8)
        right_lung = np.zeros((aux.shape[0],aux.shape[1],3),dtype = np.uint8)
        left_lung_area = 0
        right_lung_area = 0

        if len(contours_lung) > 0:
            for cnt in contours_lung:
                leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
                rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])

                left_dist = abs(leftmost[0] - 0)
                right_dist = abs(aux.shape[1] - rightmost[0])

                if left_dist < right_dist:
                    left_lung = cv2.drawContours(left_lung, [cnt], 0, (255,0,0), -1)
                    left_lung_area += cv2.contourArea(cnt)
                    total_left_lung_area += left_lung_area
                else:
                    right_lung = cv2.drawContours(right_lung, [cnt], 0, (0,255,0), -1)
                    right_lung_area += cv2.contourArea(cnt)
                    total_right_lung_area += right_lung_area

        left_lung_area_array.append(left_lung_area)
        right_lung_area_array.append(right_lung_area)

    st.session_state.left_lung_area_array = left_lung_area_array
    st.session_state.right_lung_area_array = right_lung_area_array

    st.session_state.total_left_lung_area = total_left_lung_area
    st.session_state.total_right_lung_area = total_right_lung_area
    
    if 'slice_lung_area' not in st.session_state:
        st.session_state.slice_lung_area = slice_lung_area
    else:
        st.session_state.slice_lung_area = slice_lung_area
    
    if return_masks:
        return np.array(results), np.array(results_masks)
    else:
        return np.array(results)
    
def get_lesions(images, resize=(224,224), return_masks=False, weight=1, thresh=0.5):

    '''
    input format should be (16,512,512,3)
    '''
    logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)        

    ckpt_path = get_lesion_segmentation_ckpt()
    
    model = SegmentationModel.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu")) 
                   
    aux_transforms = A.Compose([
        A.Resize(224, 224), 
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        ToTensorV2()
    ])
    aux_dataset = CustomAuxLungDataset(images, aux_transforms)
    aux_dataloader = DataLoader(aux_dataset, batch_size=1, shuffle=False, num_workers=8)
    trainer = pl.Trainer(accelerator='cpu', enable_progress_bar = False)
    predictions = trainer.predict(model, aux_dataloader)
    predictions = torch.cat(predictions)
    pr_masks = (predictions.sigmoid() > thresh).int().squeeze(0)
    results = []
    results_masks = []
    slice_lesion_area = []
    biggest_lesion = 0
    biggest_lesion_index = None

    right_lesion_area_array = []
    left_lesion_area_array = []
    total_left_lesion_area = 0
    total_right_lesion_area = 0

    i = 0
    for image, pr_mask in zip(images, pr_masks):
        msk = pr_mask.numpy().astype(np.uint8).squeeze()
        aux = np.array([msk, msk, msk]).transpose(1,2,0)
        aux = cv2.resize(aux, (512,512))
        results_masks.append(aux*255)
        seg = image*aux
        if resize:
            seg = cv2.resize(seg, (224,224))
        results.append(seg)
        if np.count_nonzero(aux) > biggest_lesion:
            biggest_lesion = np.count_nonzero(aux)
            biggest_lesion_index = i
        slice_lesion_area.append(np.count_nonzero(aux))
        i += 1

        contours_lesion, hierarchy = cv2.findContours(aux[:,:,0], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        left_lesion = np.zeros((aux.shape[0],aux.shape[1],3),dtype = np.uint8)
        right_lesion = np.zeros((aux.shape[0],aux.shape[1],3),dtype = np.uint8)
        left_lesion_area = 0
        right_lesion_area = 0

        if len(contours_lesion) > 0:
            for cnt in contours_lesion:
                leftmost = tuple(cnt[cnt[:,:,0].argmin()][0])
                rightmost = tuple(cnt[cnt[:,:,0].argmax()][0])

                left_dist = abs(leftmost[0] - 0)
                right_dist = abs(aux.shape[1] - rightmost[0])

                if left_dist < right_dist:
                    left_lesion = cv2.drawContours(left_lesion, [cnt], 0, (255,0,0), -1)
                    left_lesion_area += cv2.contourArea(cnt)
                    total_left_lesion_area += left_lesion_area
                else:
                    right_lesion = cv2.drawContours(right_lesion, [cnt], 0, (0,255,0), -1)
                    right_lesion_area += cv2.contourArea(cnt)
                    total_right_lesion_area += right_lesion_area

        left_lesion_area_array.append(left_lesion_area)
        right_lesion_area_array.append(right_lesion_area)
    
    st.session_state.left_lesion_area_array = left_lesion_area_array
    st.session_state.right_lesion_area_array = right_lesion_area_array

    st.session_state.total_left_lesion_area = total_left_lesion_area
    st.session_state.total_right_lesion_area = total_right_lesion_area
    st.write(st.session_state.total_left_lesion_area)

    st.session_state.biggest_lesion_index = biggest_lesion_index

    if 'slice_lesion_area' not in st.session_state:
        st.session_state.slice_lesion_area = slice_lesion_area
    else:
        st.session_state.slice_lesion_area = slice_lesion_area
    
    if return_masks:
        return np.array(results), np.array(results_masks)
    else:
        return np.array(results)