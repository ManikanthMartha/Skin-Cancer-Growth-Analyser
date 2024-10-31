# import streamlit as st
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image
# from PIL import Image

# # Load the model once, cached
# @st.cache
# def load_classification_model():
#     return load_model('densenet_classify.h5')

# model = load_classification_model()

# # Dictionary to map predicted index to lesion type
# lesion_type_dict = {
#     0: 'Melanocytic nevi',
#     1: 'Melanoma',
#     2: 'Benign keratosis-like lesions',
#     3: 'Basal cell carcinoma',
#     4: 'Actinic keratoses',
#     5: 'Vascular lesions',
#     6: 'Dermatofibroma'
# }

# def predict_class(uploaded_image):
#     # Convert the uploaded image to a numpy array and preprocess it
#     img = uploaded_image.resize((128, 128))  # Resize to model input size
#     img_array = image.img_to_array(img)
#     img_array = np.expand_dims(img_array, axis=0)
#     img_array /= 255.0  # Scale image to [0, 1] range

#     # Make predictions
#     predictions = model.predict(img_array)[0]
    
#     # Map predictions to lesion types with confidence percentages
#     lesion_confidences = {lesion_type_dict[i]: float(pred * 100) for i, pred in enumerate(predictions)}
    
#     return lesion_confidences

# # Streamlit app interface
# st.title("Skin Lesion Classification")

# model = load_classification_model()
# # Upload an image
# uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

# # Run the classification when the "Classify" button is clicked
# if uploaded_file:
#     img = Image.open(uploaded_file)
#     st.image(img, caption="Uploaded Image", use_column_width=True)
    
#     if st.button("Classify"):
#         # Get predictions directly from the uploaded image
#         lesion_confidences = predict_class(img)
        
#         st.subheader("Classification Results:")
#         sorted_confidences = sorted(lesion_confidences.items(), key=lambda x: x[1], reverse=True)
#         for lesion, confidence in sorted_confidences:
#             st.write(f"{lesion}: {confidence:.2f}%")





import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image

# Define custom loss functions and metrics
def dice_coef(y_true, y_pred):
    smooth = 1.
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)

def bce_dice_loss(y_true, y_pred):
    bce = tf.keras.losses.BinaryCrossentropy()
    return bce(y_true, y_pred) + dice_loss(y_true, y_pred)

def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal = alpha * K.pow(1.0 - p_t, gamma) * bce
    return K.mean(focal)

def focal_dice_loss(y_true, y_pred):
    return focal_loss(y_true, y_pred) + dice_loss(y_true, y_pred)

# Preprocess uploaded image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

# Predict mask
def predict_mask(model, img_array):
    predictions = model.predict(img_array)
    return predictions

# Display results with mask overlay
def display_results(original_img, predicted_mask):
    original_img = original_img.resize((224, 224))
    original_img = np.array(original_img) / 255.0
    predicted_mask = predicted_mask[0, :, :, 0]
    
    # Create plot
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    
    # Original Image
    ax[0].imshow(original_img)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    
    # Predicted Mask
    ax[1].imshow(predicted_mask, cmap='gray')
    ax[1].set_title('Predicted Mask')
    ax[1].axis('off')
    
    # Overlay Mask on Original Image
    ax[2].imshow(original_img)
    ax[2].imshow(predicted_mask, cmap='jet', alpha=0.5)  # Overlay with transparency
    ax[2].set_title('Overlay of Mask on Image')
    ax[2].axis('off')
    
    st.pyplot(fig)

# Load model with custom objects
@st.cache
def load_segmentation_model():
    custom_objects = {
        'bce_dice_loss': bce_dice_loss,
        'focal_dice_loss': focal_dice_loss,
        'binary_io_u': tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5),
        'dice_coef': dice_coef,
    }
    model = load_model('./unetplusplus_finaleee.h5', custom_objects=custom_objects)
    return model

# Streamlit app
st.title("Skin Lesion Segmentation with U-Net++")

# Upload an image
uploaded_file = st.file_uploader("Upload an Image", type=['jpg', 'jpeg', 'png'])

# Predict and display results if an image is uploaded
if uploaded_file:
    # Load and preprocess the uploaded image
    img = Image.open(uploaded_file)
    img_array = preprocess_image(img)
    
    # Load model and predict mask
    model = load_segmentation_model()
    predicted_mask = predict_mask(model, img_array)
    
    # Display the results
    display_results(img, predicted_mask)


