import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import os
from PIL import Image

# Load and preprocess the image
def model_predict(image_path):
    model = tf.keras.models.load_model(r"C:\Users\Dell\Desktop\New Folder\CNN_plant_disease_model.keras")
    img = cv2.imread(image_path)
    H, W, C = 224, 224, 3
    img = cv2.resize(img, (H, W))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img).astype("float32") / 255.0
    img = img.reshape(1, H, W, C)
    prediction = np.argmax(model.predict(img), axis=-1)[0]
    return prediction

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION"])

# Display main image
img = Image.open(r"C:\Users\Dell\Desktop\New Folder\farm.jpeg")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h1 style='text-align: center;'>Plant Disease Detection System for Sustainable Agriculture</h1>", unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")

    if test_image is not None:
        save_path = os.path.join(os.getcwd(), test_image.name)
        with open(save_path, "wb") as f:
            f.write(test_image.getbuffer())

    if st.button("Show Image"):
        st.image(test_image, use_column_width=True)

    # Define class labels **outside** the button to avoid scope issues
    class_name = [
        'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
        'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
        'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
        'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
        'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
        'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
        'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
        'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
        'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
        'Tomato___healthy'
    ]

    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_predict(save_path)
        predicted_disease = class_name[result_index]
        st.success(f"Model is Predicting it's a {predicted_disease}")

        # Store result in session state for advice retrieval
        st.session_state['predicted_disease'] = predicted_disease

# Disease advice dictionary
disease_advice = {
    'Apple___Apple_scab': "Apply fungicide sprays (e.g., Captan, Mancozeb) and remove infected leaves.",
    'Apple___Black_rot': "Prune infected branches, remove mummified fruit, and apply copper-based fungicides.",
    'Apple___Cedar_apple_rust': "Use resistant apple varieties and apply fungicide sprays at bud break.",
    'Apple___healthy': "No treatment needed. Maintain proper orchard hygiene.",
    'Blueberry___healthy': "Your plant is healthy! Maintain soil moisture and monitor for pests.",
    'Cherry_(including_sour)___Powdery_mildew': "Apply sulfur-based or neem oil sprays and prune overcrowded branches.",
    'Cherry_(including_sour)___healthy': "No treatment required. Ensure proper watering and pruning.",
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Use resistant varieties, rotate crops, and apply fungicides like azoxystrobin.",
    'Corn_(maize)___Common_rust_': "Plant resistant hybrids and apply fungicides if necessary.",
    'Corn_(maize)___Northern_Leaf_Blight': "Remove infected debris, use fungicides like propiconazole, and practice crop rotation.",
    'Corn_(maize)___healthy': "Your corn is healthy! Maintain proper irrigation and nutrient balance.",
    'Grape___Black_rot': "Remove infected leaves and fruits, ensure good air circulation, and apply fungicides like myclobutanil.",
    'Grape___Esca_(Black_Measles)': "Prune infected vines, avoid overwatering, and apply fungicides like flutriafol.",
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Use copper fungicides and ensure proper vineyard spacing for airflow.",
    'Grape___healthy': "No issues detected. Keep monitoring for signs of disease.",
    'Orange___Haunglongbing_(Citrus_greening)': "Control psyllid insects with insecticides and remove infected trees if necessary.",
    'Peach___Bacterial_spot': "Apply copper sprays before bud break and avoid overhead irrigation.",
    'Peach___healthy': "No issues detected. Keep monitoring for pests and diseases.",
    'Pepper,_bell___Bacterial_spot': "Apply copper fungicides and practice crop rotation.",
    'Pepper,_bell___healthy': "No treatment needed. Maintain optimal soil moisture and nutrients.",
    'Potato___Early_blight': "Use fungicides like chlorothalonil and remove infected leaves.",
    'Potato___Late_blight': "Apply fungicides like metalaxyl and avoid excessive moisture.",
    'Potato___healthy': "Your potato plants are healthy! Keep monitoring for any signs of disease.",
    'Raspberry___healthy': "No issues detected. Ensure proper pruning for airflow.",
    'Soybean___healthy': "No disease detected. Maintain proper soil health and irrigation.",
    'Squash___Powdery_mildew': "Use sulfur-based fungicides and avoid overhead watering.",
    'Strawberry___Leaf_scorch': "Remove infected leaves and apply fungicides like Captan.",
    'Strawberry___healthy': "Your strawberry plants are healthy! Keep monitoring for pests.",
    'Tomato___Bacterial_spot': "Apply copper sprays and avoid overhead watering.",
    'Tomato___Early_blight': "Rotate crops, remove infected leaves, and use fungicides like chlorothalonil.",
    'Tomato___Late_blight': "Apply copper-based fungicides and remove affected leaves.",
    'Tomato___Leaf_Mold': "Improve air circulation, remove affected leaves, and apply fungicides.",
    'Tomato___Septoria_leaf_spot': "Use fungicides like chlorothalonil and practice crop rotation.",
    'Tomato___Spider_mites Two-spotted_spider_mite': "Use neem oil or insecticidal soap to control mites.",
    'Tomato___Target_Spot': "Apply fungicides and remove infected leaves.",
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Use virus-resistant seeds and control whiteflies.",
    'Tomato___Tomato_mosaic_virus': "Remove infected plants and disinfect gardening tools.",
    'Tomato___healthy': "Your tomato plant is healthy! Maintain good watering and fertilization practices."
}

# Advice button
if st.button("Get Advice"):
    predicted_disease = st.session_state.get('predicted_disease', None)
    if predicted_disease:
        advice = disease_advice.get(predicted_disease, "No specific advice available.")
        st.info(f"Advice: {advice}")
    else:
        st.warning("Please predict the disease first before requesting advice.")


