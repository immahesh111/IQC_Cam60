import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import h5py
import os

def merge_h5_files(file_parts, merged_filename="merged_model.h5"):
    """Merges multiple h5 files into a single h5 file with better error checks."""
    try:
        with h5py.File(merged_filename, 'w') as dest_file:
            first = True
            for part_filename in file_parts:
                try:
                    # Add a check to ensure the file is valid
                    if not h5py.is_hdf5(part_filename):
                        st.error(f"Error: {part_filename} is not a valid HDF5 file.")
                        return None

                    with h5py.File(part_filename, 'r') as src_file:
                        for key in src_file.keys():
                            if first:  # Copy all groups from the first file
                                src_file.copy(key, dest_file)
                            else:  # Subsequent files: Only copy if the group doesn't exist
                                if key not in dest_file:
                                    src_file.copy(key, dest_file)
                    first = False
                except FileNotFoundError:
                    st.error(f"Error: File not found: {part_filename}")
                    return None
                except OSError as e:
                    st.error(f"OSError: Unable to open {part_filename}. It may be corrupted. Details: {e}")
                    return None
                except Exception as e:
                    st.error(f"An error occurred while processing {part_filename}: {e}")
                    return None
        st.success(f"Successfully merged files into {merged_filename}")
        return merged_filename
    except Exception as e:
        st.error(f"Error during merging process: {e}")
        return None

@st.cache_resource
def load_merged_model(model_parts):
    merged_model_path = "merged_model.h5"

    # Check if merged_model.h5 exists, if not merge
    if not os.path.exists(merged_model_path):
        merged_model_path = merge_h5_files(model_parts, merged_model_path)
        if not merged_model_path:
            st.error("Model merging failed. Please check the file parts.")
            return None
    try:
        return tf.keras.models.load_model(merged_model_path)
    except Exception as e:
        st.error(f"Error loading the merged model: {e}")
        return None

def model_prediction(image, model):
    if model is None:
        st.error("Model not loaded. Cannot make predictions.")
        return None
    img = cv2.resize(image, (224,224))
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

st.sidebar.title('Mobile Camera Inspection')
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "About", "Mobile Inspection", "Live Inspection"])

model_parts = [
    "model_part_0.h5",
    "model_part_1.h5",
    "model_part_2.h5",
    "model_part_3.h5",
    "model_part_4.h5",
    "model_part_5.h5",
    "model_part_6.h5",
    "model_part_7.h5",
    "model_part_8.h5",
    "model_part_9.h5",
]

# Check if files are present and valid BEFORE attempting anything
all_files_valid = True
for part in model_parts:
    if not os.path.exists(part):
        st.error(f"Error: {part} not found.  Ensure all model parts are in the same directory.")
        all_files_valid = False
    elif not h5py.is_hdf5(part):
        st.error(f"Error: {part} is not a valid HDF5 file.")
        all_files_valid = False

if not all_files_valid:
    st.stop() # Terminate the app if files are missing or invalid

model = load_merged_model(model_parts)

if app_mode == "Home":
    st.header('MOBILE CAMERA INSPECTION SYSTEM')
    image_path = 'home_page.png'
    st.image(image_path, use_column_width=True)
    st.markdown('''Welcome to the Mobile Screen Inspection System...''')

elif app_mode == "About":
    st.header("About")
    st.markdown('''### About Dataset...''')

elif app_mode == "Mobile Inspection":
    st.header('Mobile Inspection')
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, width=400, use_column_width=True)

        if st.button("Predict"):
            result_index = model_prediction(np.array(image), model)
            if result_index is not None: # Only proceed if prediction worked
                class_names = ['Good', 'NG_Crack','NG_Dent','NG_Dust','NG_Fingerprint','NG_Scratch']
                prediction = class_names[result_index]
                st.write("Prediction: " + prediction)
                # (rest of the Mobile Inspection code...)
            else:
                st.error("Prediction failed. Check model loading and input image.")

elif app_mode == "Live Inspection":
    st.header('Live Mobile Screen Inspection')
    run = st.checkbox('Run')
    FRAME_WINDOW = st.image([])
    cap = cv2.VideoCapture(1)

    while run:
        ret, frame = cap.read()

        if ret:
            result_index = model_prediction(frame, model)
            if result_index is not None:
                class_names = ['Good', 'NG_Crack','NG_Dent','NG_Dust','NG_Fingerprint','NG_Scratch']
                prediction = class_names[result_index]
                # (rest of the Live Inspection code...)
            else:
                st.error("Prediction failed. Check model loading and camera input.")
        else:
            st.warning("Failed to capture frame. Check camera connection.")

    cap.release()
