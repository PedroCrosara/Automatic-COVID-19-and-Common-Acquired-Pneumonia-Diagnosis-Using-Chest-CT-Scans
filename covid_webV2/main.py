import streamlit as st
from model.segmentation_utils import get_lungs, get_lesions
from utils.plot_utils import plot_3d, draw_dicom_slices
from model.classification_utils import run_classification
from utils.upload_utils import upload_zip_dir
from utils.plot_utils import create_slider

st.set_option('deprecation.showPyplotGlobalUse', False)

def b1_callback():
    plot_3d(st.session_state.mask_lungs, st.session_state.segmented_lesions)

def b2_callback(): # lung segmentation
    segmented_lungs, mask_lungs = get_lungs(st.session_state.processed_dicom, resize=None, return_masks=True)

    if 'segmented_lungs' not in st.session_state:
            st.session_state.segmented_lungs = segmented_lungs
            st.session_state.mask_lungs = mask_lungs
    else:
        st.session_state.segmented_lungs = segmented_lungs 
        st.session_state.mask_lungs = mask_lungs

    drawn_dicom = draw_dicom_slices(st.session_state.processed_dicom.copy(), mask_lungs)

    st.session_state.drawn_dicom = drawn_dicom


def b3_callback(): # lesion segmentation
    if 'segmented_lungs' not in st.session_state:
            st.session_state.segmented_lungs, st.session_state.mask_lungs = get_lungs(st.session_state.processed_dicom, resize=None, return_masks=True)
            segmented_lungs = st.session_state.segmented_lungs
            mask_lungs = st.session_state.mask_lungs
    else:
        segmented_lungs = st.session_state.segmented_lungs
        mask_lungs = st.session_state.mask_lungs

    segmented_lesions, mask_lesions = get_lesions(segmented_lungs, resize=None, return_masks=True)

    if 'segmented_lesions' not in st.session_state:
            st.session_state.segmented_lesions = segmented_lesions
            st.session_state.mask_lesions = mask_lesions
    else:
        st.session_state.mask_lesions = mask_lesions

    drawn_dicom = draw_dicom_slices(st.session_state.processed_dicom.copy(), mask_lungs, mask_lesions)

    st.session_state.drawn_dicom = drawn_dicom

def b4_callback(): # ct classification
    if st.session_state.biggest_lesion_index is not None:
        y_pred = run_classification()

        if y_pred == 0:
            st.session_state.disease_class = 'Chance of Common Acquired Pneumonia'
        if y_pred == 1:
            st.session_state.disease_class = 'Chance of COVID-19'
    else:
        st.session_state.disease_class = 'Chance of no CAP nor COVID-19'

    st.write(st.session_state.disease_class)

def web_page():
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.button("3D view", key="exe", on_click=b1_callback)
    with c2:
        st.button("Lung Segmentation", key="boa", on_click=b2_callback)
    with c3:
        st.button("Lesion Segmentation", key="rum", on_click=b3_callback)
    with c4:
        st.button("Classification", key="pes", on_click=b4_callback)

# Main function
def main():
    st.set_page_config(page_title="Upload and Plot DICOM Zip Directory", page_icon=":hospital:", layout="wide")
    st.title("Upload and Plot DICOM Zip Directory")
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    if not st.session_state.processed:
        if 'processed_dicom' not in st.session_state:
            st.session_state.processed_dicom = upload_zip_dir()
        else:
            st.session_state.processed_dicom = upload_zip_dir()    
    if st.session_state.processed:
        web_page()
        create_slider(st.session_state.processed_dicom)

if __name__ == '__main__':
    main()