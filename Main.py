import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# TensorFlow model prediction function
def model_prediction(image):
    model = tf.keras.models.load_model('pruned_model.keras')
    img = cv2.resize(image, (224,224))  # Resize image to match model input
    input_arr = tf.keras.preprocessing.image.img_to_array(img)
    input_arr = np.array([input_arr])  # Convert single image to a batch.
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar configuration for Streamlit app
st.sidebar.title('Mobile Camera Inspection')
app_mode = st.sidebar.selectbox("Choose the app mode", ["Home", "About", "Mobile Inspection", "Live Inspection"])

# Home page
if app_mode == "Home":
    st.header('MOBILE CAMERA INSPECTION SYSTEM')
    image_path = 'home_page.png'
    st.image(image_path, use_column_width=True)
    
    st.markdown('''
                Welcome to the Mobile Screen Inspection System.
                This system is designed to help you inspect your mobile screen whether it is OK or NG.
                Please select the "Mobile Inspection" mode to upload an image of the mobile screen.
                ''')

# About page
elif app_mode == "About":
    st.header("About")
    st.markdown('''
                ### About Dataset
                Go to Mobile Inspection mode to upload an image of the mobile screen. 
                The system will predict if there is a fault in the mobile screen.
                ''')

# Mobile Inspection page
elif app_mode == "Mobile Inspection":
    st.header('Mobile Inspection')
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image is not None:
        image = Image.open(test_image)
        st.image(image, width=400, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            # Get prediction result
            result_index = model_prediction(np.array(image))
            
            # Reading Labels
            class_names = ['Good', 'NG_Crack','NG_Dent','NG_Dust','NG_Fingerprint','NG_Scratch']
            prediction = class_names[result_index]
            
            # Display the prediction
            st.write("Prediction: " + prediction)
            
            # Change background color based on prediction
            if prediction == 'Good':
                st.markdown(
                    """
                    <style>
                    .reportview-container {
                        background-color: #DFFFD6; /* Light Green */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.success("Mobile Screen is: " + prediction)
            elif prediction == 'NG_Crack' or prediction == 'NG_Dent' or prediction == 'NG_Dust' or prediction == 'NG_Fingerprint' or prediction == 'NG_Scratch':
                st.markdown(
                    """
                    <style>
                    .reportview-container {
                        background-color: #FFCCCB; /* Light Red */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )
                st.error("Mobile Screen is: " + prediction)


# Assuming model_prediction is defined elsewhere
if app_mode == "Live Inspection":
    st.header('Live Mobile Screen Inspection')

    run = st.checkbox('Run')

    FRAME_WINDOW = st.image([])  # Placeholder for displaying frames

    # Assign a key to the camera input to manage its session state
    camera_input = st.camera_input("Take a picture", key='camera_input')

    if run:
        if camera_input is not None:
            # Convert image data to OpenCV format
            file_bytes = np.asarray(bytearray(camera_input.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

            # Perform prediction
            result_index = model_prediction(frame)

            class_names = ['Good', 'NG_Crack','NG_Dent','NG_Dust','NG_Fingerprint','NG_Scratch']
            prediction = class_names[result_index]

            # Determine text color based on prediction
            color = (0, 255, 0) if prediction == 'Good' else (0, 0, 255)  # Green for 'Good', Red for others

            # Annotate the frame with prediction
            cv2.putText(frame, f'Prediction: {prediction}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # Convert frame to RGB for display in Streamlit
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            FRAME_WINDOW.image(frame_rgb, channels='RGB')

            # Clear the camera input from session state to reset it
            if 'camera_input' in st.session_state:
                del st.session_state.camera_input

            # Clear all cached data and resources to ensure fresh predictions
            st.cache_data.clear()
            st.cache_resource.clear()
        else:
            st.warning("Please enable your camera and take a picture.")



