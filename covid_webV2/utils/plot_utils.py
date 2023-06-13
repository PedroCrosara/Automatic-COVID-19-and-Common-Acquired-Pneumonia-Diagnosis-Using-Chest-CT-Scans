from PIL import Image
import numpy as np
import streamlit as st
from matplotlib import pyplot as plt
import cv2
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def plot_dicom(dicom_file):
    # Read the DICOM file using pydicom
    #ds = pydicom.dcmread(dicom_file)
    # Get the pixel data from the DICOM file
    #pixel_data = ds.pixel_array
    image = Image.fromarray(np.uint8(dicom_file))
    #st.pyplot(fig)
    st.image(image,width=400)

def plot_3d(image_lung, image_lesion):
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')    # Fancy indexing: `verts[faces]` to generate a collection of    
    # triangles

    # Position the scan upright, 
    # so the head of the patient would be at the top facing the   
    # camera
    p_lung = np.squeeze(image_lung[:,:,:,0])
    p_lung = p_lung.transpose(2,1,0)
    p_lung = np.flip(p_lung, axis=2)
    p_lung = cv2.resize(p_lung, (128,128))
    verts, faces, _, _ = measure.marching_cubes(p_lung)
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, .45, .75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    p_lesion = np.squeeze(image_lesion[:,:,:,0])
    p_lesion = p_lesion.transpose(2,1,0)
    p_lesion = np.flip(p_lesion, axis=2)
    p_lesion = cv2.resize(p_lesion, (128,128))
    verts, faces, _, _ = measure.marching_cubes(p_lesion)
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [1., 0., 0.]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p_lung.shape[0])
    ax.set_ylim(0, p_lung.shape[1])
    ax.set_zlim(0, p_lung.shape[2])    

    plot = st.empty()
    plot.pyplot()

    # add a close button below the plot
    if st.button("Close Plot"):
        plot.empty()

def create_slider(processed_dicom):

    fig, axs = plt.subplots(1, 4, figsize=(9, 6))
    fig.set_facecolor('none')

    file_idx = st.slider("Select a DICOM file", 0, len(processed_dicom) - 1)
    if 'file_idx' not in st.session_state:
        st.session_state.file_idx = file_idx
    else:
        st.session_state.file_idx = file_idx
    #plot_dicom(processed_dicom[file_idx])
    axs[0].imshow(Image.fromarray(np.uint8(processed_dicom[file_idx])))
    axs[0].axis('off')

    if 'drawn_dicom' in st.session_state:
        axs[1].imshow(Image.fromarray(st.session_state.drawn_dicom[file_idx]))
        axs[1].axis('off')
    else:
        axs[1].imshow(Image.fromarray(np.zeros((512,512,3), np.uint8)))
        axs[1].axis('off')
        axs[2].imshow(Image.fromarray(np.zeros((512,512,3), np.uint8)))
        axs[2].axis('off')
        axs[3].imshow(Image.fromarray(np.zeros((512,512,3), np.uint8)))
        axs[3].axis('off')

    if 'segmented_lungs' in st.session_state:
        #plot_dicom(st.session_state.segmented_lungs[file_idx])
        axs[2].imshow(Image.fromarray(np.uint8(st.session_state.segmented_lungs[file_idx])))
        axs[2].axis('off')
    else:
        axs[2].imshow(Image.fromarray(np.zeros((512,512,3), np.uint8)))
        axs[2].axis('off')
        axs[3].imshow(Image.fromarray(np.zeros((512,512,3), np.uint8)))
        axs[3].axis('off')

    if 'segmented_lesions' in st.session_state:
        #plot_dicom(st.session_state.segmented_lesions[file_idx])
        axs[3].imshow(Image.fromarray(np.uint8(st.session_state.segmented_lesions[file_idx])))
        axs[3].axis('off')
    else:
        axs[3].imshow(Image.fromarray(np.zeros((512,512,3), np.uint8)))
        axs[3].axis('off')

    # Remove spacing between subplots
    plt.subplots_adjust(wspace=0, hspace=0)

    # Display the subplot using Streamlit
    st.pyplot(fig)

    if 'slice_lesion_area' in st.session_state:
        slice_lung_area = st.session_state.slice_lung_area
        slice_lesion_area = st.session_state.slice_lesion_area

        total_left_lung_area = st.session_state.total_left_lung_area
        total_right_lung_area = st.session_state.total_right_lung_area
        total_left_lesion_area = st.session_state.total_left_lesion_area
        total_right_lesion_area = st.session_state.total_right_lesion_area

        left_lung_area_array = st.session_state.left_lung_area_array
        right_lung_area_array = st.session_state.right_lung_area_array

        left_lesion_area_array = st.session_state.left_lesion_area_array
        right_lesion_area_array = st.session_state.right_lesion_area_array

        if left_lung_area_array[file_idx] > 0:
            #st.write(f'Slice percentage area: {round(100*slice_lesion_area[file_idx]/slice_lung_area[file_idx],2)}%')
            st.write(f'Slice left lesion/lung percentage: {round(100*left_lesion_area_array[file_idx]/left_lung_area_array[file_idx],2)}%')
            #st.write(f'Slice right lesion/lung percentage: {round(100*right_lesion_area_array[file_idx]/right_lung_area_array[file_idx],2)}%')
        else:
            st.write('Left lung slice percentage area is zero')

        if right_lung_area_array[file_idx] > 0:
            #st.write(f'Slice percentage area: {round(100*slice_lesion_area[file_idx]/slice_lung_area[file_idx],2)}%')
            #st.write(f'Slice left lesion/lung percentage: {round(100*left_lesion_area_array[file_idx]/left_lung_area_array[file_idx],2)}%')
            st.write(f'Slice right lesion/lung percentage: {round(100*right_lesion_area_array[file_idx]/right_lung_area_array[file_idx],2)}%')
        else:
            st.write('Right lung slice percentage area is zero')
        
        if np.sum(slice_lesion_area) > 0:
            # st.write(f'Full CT percentage area: {round(100*np.sum(slice_lesion_area)/np.sum(slice_lung_area),2)}%')
            st.write(f'Total left lesion/lung percentage: {round(100*total_left_lesion_area/total_left_lung_area,2)}%')
            st.write(f'Total right lesion/lung percentage: {round(100*total_right_lesion_area/total_right_lung_area,2)}%')
        else:
            st.write('Full CT lung percentage area is zero')

def draw_dicom_slices(processed_dicom, mask_lungs, mask_lesions=None):

    drawn_dicom = []

    if mask_lesions is None:
        for img, lung in zip(processed_dicom, mask_lungs):
            lung = lung[:,:,0].astype(np.uint8)
            contours, hierarchy = cv2.findContours(lung, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            painted_img = cv2.drawContours(img, contours, -1, (0,255,0), 2)
            drawn_dicom.append(painted_img)
        return np.array(drawn_dicom)
        
    else:
        for img, lung, lesion in zip(processed_dicom, mask_lungs, mask_lesions):
            lung = lung[:,:,0].astype(np.uint8)
            contours, hierarchy = cv2.findContours(lung, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            painted_img = cv2.drawContours(img, contours, -1, (0,255,0), 2)          
            lesion = lesion[:,:,0].astype(np.uint8)
            contours, hierarchy = cv2.findContours(lesion, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            painted_img = cv2.drawContours(painted_img, contours, -1, (255,0,0), 2)
            drawn_dicom.append(painted_img)
        return np.array(drawn_dicom)