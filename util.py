import base64
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
import tensorflow as tf
import cv2

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
    # 1. Redimensionner l'image en 224x224 (LANCZOS pour la qualité)
    image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)

    # 2. Convertir en tableau numpy
    img_array = np.asarray(image)

    # 3. Normaliser (diviser par 255.0)
    # Important : convertir en float32 avant la division
    normalized_image_array = (img_array.astype(np.float32) / 255.0)

    # 4. Ajouter la dimension du batch (1, 224, 224, 3)
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array

    # 5. Prédiction
    prediction = model.predict(data)
    
    # 6. Récupérer la classe
    index = np.argmax(prediction)
    
    # Sécurité pour éviter "List index out of range"
    if index < len(class_names):
        class_name = class_names[index]
        confidence_score = prediction[0][index]
    else:
        class_name = "Inconnu"
        confidence_score = 0.0

    return class_name, confidence_score

def is_chest_xray(image):
    """Contrôle élémentaire de contraste, sans validation du contenu médical."""
    # Conversion en niveaux de gris
    img_gray = ImageOps.grayscale(image)
    img_array = np.array(img_gray)
    
    # Analyse statistique simple
    # mean_val = np.mean(img_array)
    std_val = np.std(img_array)
    
    # Si l'écart type est très faible, l'image est probablement unie (toute noire ou toute blanche)
    if std_val < 10: 
        return False, 30, "L'image est trop uniforme pour une analyse fiable."
    
    return True, 100, (
        "Le contraste est suffisant pour poursuivre le traitement. "
        "Ce contrôle ne confirme pas qu'il s'agit d'une radiographie thoracique valide."
    )

def get_doctor_animation_data(emotion):
    """Retourne l'URL de l'animation et les icônes selon l'émotion."""
    if emotion == "happy":
        return {
            "url": "https://assets9.lottiefiles.com/packages/lf20_jcsfwbvi.json", 
            "fallback": "https://assets5.lottiefiles.com/packages/lf20_x62chJ.json",
            "icon": "✅",
            "status": "Rassurant"
        }
    else:
        return {
            "url": "https://assets2.lottiefiles.com/packages/lf20_qpwbqhz9.json", 
            "fallback": "https://assets4.lottiefiles.com/packages/lf20_5tl1xxnz.json",
            "icon": "⚠️",
            "status": "Attention Requise"
        }

def get_empathetic_message(class_name, confidence):
    """Génère un message adapté au résultat COVID vs Normal."""
    # On vérifie si "NORMAL" n'est PAS dans le nom (donc c'est malade)
    is_sick = "NORMAL" not in class_name.upper()
    
    if is_sick:
        return {
            "greeting": "Bonjour, voici l'analyse de votre imagerie.",
            "main_message": f"Le modèle classe cette image dans la catégorie **{class_name}** avec un score de {confidence*100:.1f}%.",
            "reassurance": "Ce résultat expérimental n'est pas un diagnostic médical.",
            "urgency": "Faites interpréter l'imagerie et vos symptômes par un professionnel de santé.",
            "prevention": "",
            "closing": "Prenez soin de vous et protégez vos proches."
        }
    else:
        return {
            "greeting": "Bonjour, j'ai de bonnes nouvelles.",
            "main_message": f"Le modèle classe cette image dans la catégorie **Normal** avec un score de {confidence*100:.1f}%.",
            "reassurance": "Un classement 'Normal' n'exclut pas une maladie et ne remplace pas un avis médical.",
            "urgency": "",
            "prevention": "Continuez à appliquer les gestes barrières pour rester en bonne santé.",
            "closing": "Votre santé est notre priorité."
        }

def simple_attention_map(model, image, class_idx):
    """
    Génère une Heatmap (Grad-CAM simplifié) pour visualiser les zones.
    """
    try:
        # Prétraitement identique à classify
        image = ImageOps.fit(image, (224, 224), Image.Resampling.LANCZOS)
        img_array = np.asarray(image).astype(np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

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
        img_rgb = np.array(image)
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
        
        superimposed_img = heatmap_colored * 0.4 + img_rgb * 0.6
        return np.uint8(superimposed_img)

    except Exception as e:
        print(f"Erreur Heatmap: {e}")
        return None
