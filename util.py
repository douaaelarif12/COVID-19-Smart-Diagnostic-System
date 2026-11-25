import base64
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2

from inference import check_image_contrast, predict_image, prepare_image

def set_background(image_file):
    """
    Définit une image de fond pour l'application Streamlit.
    Ajoute un voile blanc transparent (overlay) pour que le texte reste lisible.
    """
    try:
        with open(image_file, "rb") as f:
            img_data = f.read()
        b64_encoded = base64.b64encode(img_data).decode()
        
        # Le 'rgba(255, 255, 255, 0.85)' crée un voile blanc à 85% d'opacité
        # Vous pouvez ajuster 0.85 selon vos goûts (entre 0.0 et 1.0)
        style = f"""
            <style>
            .stApp {{
                background-image: linear-gradient(rgba(255, 255, 255, 0.85), rgba(255, 255, 255, 0.85)), url(data:image/png;base64,{b64_encoded});
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
            </style>
        """
        st.markdown(style, unsafe_allow_html=True)
    except Exception as e:
        print(f"Erreur fond d'écran: {e}")

def classify(image, model, class_names):
    """
    Prétraitement et classification spécifique pour le modèle COVID :
    - Redimensionnement 224x224
    - Normalisation / 255.0
    """
    result = predict_image(image, model, class_names)
    return result.class_name, result.score

def is_chest_xray(image):
    """Alias historique : ne vérifie que le contraste, pas le contenu thoracique."""
    is_valid, _, reason = check_image_contrast(image)
    return is_valid, 100 if is_valid else 30, reason

def get_doctor_animation_data(emotion):
    """Retourne l'URL de l'animation et les icônes selon l'émotion."""
    if emotion == "happy":
        return {
            "url": "https://assets9.lottiefiles.com/packages/lf20_jcsfwbvi.json", 
            "fallback": "https://assets5.lottiefiles.com/packages/lf20_x62chJ.json",
            "icon": "✅",
            "status": "Classe majoritaire : Normal"
        }
    else:
        return {
            "url": "https://assets2.lottiefiles.com/packages/lf20_qpwbqhz9.json", 
            "fallback": "https://assets4.lottiefiles.com/packages/lf20_5tl1xxnz.json",
            "icon": "⚠️",
            "status": "Classe majoritaire : anomalie"
        }

def get_empathetic_message(class_name, confidence):
    """Génère un message adapté au résultat COVID vs Normal."""
    # On vérifie si "NORMAL" n'est PAS dans le nom (donc c'est malade)
    is_sick = "NORMAL" not in class_name.upper()
    
    if is_sick:
        return {
            "greeting": "Résultat du modèle",
            "main_message": f"Le modèle classe cette image dans la catégorie **{class_name}** avec un score de {confidence*100:.1f}%.",
            "reassurance": "Ce résultat expérimental n'est pas un diagnostic médical.",
            "urgency": "Faites interpréter l'imagerie et vos symptômes par un professionnel de santé.",
            "prevention": "",
            "closing": "Demandez un avis médical pour toute interprétation clinique."
        }
    else:
        return {
            "greeting": "Résultat du modèle",
            "main_message": f"Le modèle classe cette image dans la catégorie **Normal** avec un score de {confidence*100:.1f}%.",
            "reassurance": "Un classement 'Normal' n'exclut pas une maladie et ne remplace pas un avis médical.",
            "urgency": "",
            "prevention": "Demandez un avis médical si vous avez des symptômes ou une inquiétude.",
            "closing": "Ce résultat reste expérimental."
        }

def simple_attention_map(model, image, class_idx):
    """
    Génère une Heatmap (Grad-CAM simplifié) pour visualiser les zones.
    """
    try:
        # Prétraitement identique à classify
        display_image = ImageOps.fit(
            ImageOps.exif_transpose(image).convert("RGB"),
            (224, 224),
            Image.Resampling.LANCZOS,
        )
        img_array = prepare_image(image)

        # Trouver la dernière couche de convolution
        last_conv_layer_name = None
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                last_conv_layer_name = layer.name
                break
        
        if not last_conv_layer_name:
            return None

        # Création du modèle pour Grad-CAM
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            # Sécurité index
            if class_idx >= predictions.shape[1]:
                class_idx = 0
            loss = predictions[:, class_idx]

        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]

        gate_f = tf.reduce_mean(grads, axis=(0, 1))
        cam = tf.reduce_sum(tf.multiply(gate_f, output), axis=-1)

        cam = cam.numpy()
        # Redimensionnement à la taille de l'image (224x224)
        cam = cv2.resize(cam, (224, 224))
        cam = np.maximum(cam, 0) # ReLU
        
        # Normalisation
        if cam.max() != cam.min():
            heatmap = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            heatmap = cam

        # Superposition
        img_rgb = np.array(display_image)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        superimposed_img = heatmap_colored * 0.4 + img_rgb * 0.6
        return np.uint8(superimposed_img)

    except Exception as e:
        print(f"Erreur Heatmap: {e}")
        return None
