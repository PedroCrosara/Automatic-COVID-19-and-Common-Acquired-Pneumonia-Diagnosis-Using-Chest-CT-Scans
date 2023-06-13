import zipfile
import pydicom
import streamlit as st
import numpy as np
from scipy import ndimage
import cv2
import os

# Function to upload a zip directory
def upload_zip_dir():
    uploaded_zip = st.file_uploader("Choose a zip directory to upload", type="zip")
    if uploaded_zip is not None:
        # Extract the contents of the zip directory
        with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
            zip_ref.extractall('./exam')
        os.chdir('./exam')
        # Get all DICOM files in the extracted directory
        dicom_files = [f for f in os.listdir('./') if f.endswith('.dcm')]
        # Show the number of DICOM files uploaded
        st.write("Number of DICOM files uploaded: ", len(dicom_files))
        # Add a slider to slide through the DICOM files
        #
        processed_dicom = preprocess_dicom(dicom_files)
        st.session_state.processed = True
        return processed_dicom

        # Plot the selected DICOM file
        #plot_dicom(dicom_files[file_idx])

def preprocess_dicom(dicom_file):
    exam_dict = {}
    for file in dicom_file:        
        ds = pydicom.dcmread(file)
    
        volume = ds.pixel_array
        
        slope = 1
        intercept = -1024
        
        volume = (volume*slope+intercept) 
        img_min = -1250
        img_max = 250
        volume[volume<img_min] = img_min 
        volume[volume>img_max] = img_max
        volume = (volume - img_min) / (img_max - img_min)*255.0 
        volume = cv2.cvtColor(volume.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    
        exam_dict[ds.SliceLocation] = volume # might sort dict if necessary
        myKeys = list(exam_dict.keys())
        myKeys.sort(reverse=True)
        exam_dict = {i: exam_dict[i] for i in myKeys}
        
    return resize_volume_covidctmd(np.array(list(exam_dict.values())))

def resize_volume_covidctmd(img):
    """Resize across z-axis"""
    # Set the desired depth
    desired_depth = img.shape[0]
    desired_width = 512
    desired_height = 512
    # Get current depth
    current_depth = img.shape[0]
    # Compute depth factor
    depth = current_depth / desired_depth
    depth_factor = 1 / depth
    # Resize across z-axis
    img = ndimage.zoom(img, (depth_factor, 1, 1, 1), order=1)
    img = np.flip(img, 2)
    return img