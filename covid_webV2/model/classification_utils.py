from model.classification_model import ClassificationModel
import streamlit as st
from PIL import Image
import numpy as np
from torchvision import transforms as T
import torch
from config import get_classification_ckpt

def run_classification():
    ckpt_path = get_classification_ckpt()
    loaded_model = ClassificationModel.load_from_checkpoint(ckpt_path, map_location=torch.device("cpu"))   
    image = st.session_state.segmented_lungs[st.session_state.biggest_lesion_index]
    image = Image.fromarray(np.uint8(image)).convert('RGB')
    #img = transforms.functional.adjust_gamma(img, .8)
    image = T.Resize(size=(224,224))(image)
    image = T.ToTensor()(image)
    image = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(image)
    image = torch.unsqueeze(image,0)
    loaded_model.eval()
    y_pred = loaded_model(image)
    y_pred = y_pred.argmax(dim=-1) 
    return y_pred
