import streamlit as st
from keras.models import load_model
from PIL import Image
import numpy as np
import requests
from streamlit_lottie import st_lottie
from gtts import gTTS
import os
import tempfile
import time
from pathlib import Path

# Import des fonctions utilitaires
from util import (
    classify, 
    set_background, 
    simple_attention_map,
    get_empathetic_message,
    get_doctor_animation_data,
    is_chest_xray
)

# Configuration de la page
st.set_page_config(
    page_title="Diagnostic COVID-19 - IA Médicale",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Styles CSS personnalisés
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        color: #e53935;
        font-size: 3em;
        font-weight: bold;
        margin-bottom: 10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .subtitle {
        text-align: center;
        color: #424242;
        font-size: 1.2em;
        margin-bottom: 30px;
    }
    .doctor-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 15px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        margin: 20px 0;
        animation: fadeIn 0.5s;
    }
    .doctor-card-happy {
        background: linear-gradient(135deg, #4caf50 0%, #81c784 100%);
    }
    .doctor-card-concerned {
        background: linear-gradient(135deg, #f44336 0%, #ef5350 100%);
    }
    .alert-box {
        background-color: #fff3cd;
        border: 2px solid #ffc107;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 2px solid #28a745;
        padding: 20px;
        border-radius: 10px;
        margin: 20px 0;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

# Fonction pour charger les animations Lottie
@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Définir le fond
try:
    if not Path('./bgs/bg5.png').is_file():
        raise FileNotFoundError("Fond optionnel absent.")
    set_background('./bgs/bg5.png')
except:
    pass

# Titre principal
st.markdown('<h1 class="main-title">🦠 Système de Détection COVID-19</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Analyse d\'imagerie thoracique par Intelligence Artificielle</p>', unsafe_allow_html=True)

# Sidebar (Minimaliste)
with st.sidebar:
    st.markdown("### 📊 Statistiques")
    if 'total_analyses' not in st.session_state:
        st.session_state.total_analyses = 0
    st.metric("Analyses effectuées", st.session_state.total_analyses)
    

   

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model_and_labels():
    model_path = Path('./model/covid_detector.h5')
    labels_path = Path('./model/class_names.txt')
    if not model_path.is_file():
        raise FileNotFoundError(
            "Modèle absent : ajoutez model/covid_detector.h5 (voir model/README.md)."
        )
    if not labels_path.is_file():
        raise FileNotFoundError(
            "Étiquettes absentes : créez model/class_names.txt dans l'ordre exact des sorties."
        )
    model = load_model(model_path, compile=False)

    class_names = [line.strip() for line in labels_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    num_classes = int(model.output_shape[-1])
    if len(class_names) != num_classes:
        raise ValueError(
            f"Le modèle produit {num_classes} sorties, mais class_names.txt contient {len(class_names)} classes."
        )

    return model, class_names

try:
    with st.spinner("🔄 Chargement du modèle de détection COVID-19..."):
        model, class_names = load_model_and_labels()
    st.success("✅ Modèle COVID chargé avec succès !")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement: {str(e)}")
    st.stop()

# Upload
st.markdown("## 📤 Téléchargement de l'image (Radio/Scanner)")
st.markdown("**Veuillez uploader une radiographie ou un scanner thoracique**")

col1, col2 = st.columns([2, 1])
with col1:
    file = st.file_uploader(
        'Radiographie thoracique',
        type=['jpeg', 'jpg', 'png'],
        label_visibility="collapsed",
    )
with col2:
    # J'ai supprimé l'animation d'attente ici aussi si vous vouliez que ce soit totalement clean
    # Mais je garde le "waiting" à côté du bouton upload car c'est discret.
    lottie_waiting = load_lottie_url("https://assets4.lottiefiles.com/packages/lf20_5tl1xxnz.json")
    if lottie_waiting and file is None:
        st_lottie(lottie_waiting, height=200, key="waiting_animation")

# Traitement
if file is not None:
    image = Image.open(file).convert('RGB')
    st.session_state.total_analyses += 1
    
    st.markdown("---")
    
    # ÉTAPE 1: VALIDATION
    st.markdown("## 🔍 Étape 1: Validation de l'image")
    col_img, col_validation = st.columns([1, 1])
    
    with col_img:
        st.markdown("### 📷 Image reçue")
        st.image(image, use_column_width=True, caption="Image analysée")
    
    with col_validation:
        st.markdown("### ✅ Vérification")
        with st.spinner("🔍 Vérification en cours..."):
            is_valid, validation_score, validation_reason = is_chest_xray(image)
        
        if is_valid:
            st.success("✅ **Contraste suffisant**")
            st.info(validation_reason)
        else:
            st.error("❌ **Attention**")
            st.warning("L'image ne semble pas être une radiographie standard. L'analyse continue malgré tout.")
    
    # ÉTAPE 2: CLASSIFICATION
    st.markdown("---")
    st.markdown("## 🤖 Étape 2: Analyse Détection COVID-19")
    
    with st.spinner("🧠 Analyse des tissus pulmonaires en cours..."):
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)
        
        class_name, conf_score = classify(image, model, class_names)
        
        try:
            class_idx = class_names.index(class_name)
        except:
            class_idx = 0
    
    st.success("✅ Analyse terminée !")
    
    # --- LOGIQUE D'ÉMOTION DU MÉDECIN ---
    is_sick = "NORMAL" not in class_name.upper()
    emotion = "concerned" if is_sick else "happy"
    
    doctor_data = get_doctor_animation_data(emotion)
    empathetic_msg = get_empathetic_message(class_name, conf_score)
    
    # RÉSULTATS
    st.markdown("---")
    st.markdown("## 🎯 Résultat expérimental de classification")
    
    col_anim1, col_anim2, col_anim3 = st.columns([1, 2, 1])
    with col_anim2:
        lottie_result = load_lottie_url(doctor_data["url"])
        if not lottie_result:
            lottie_result = load_lottie_url(doctor_data["fallback"])
        
        if lottie_result:
            st_lottie(lottie_result, height=250, key=f"result_animation_{emotion}")
        else:
            st.markdown(f"<h1 style='text-align: center; font-size: 5em;'>{doctor_data['icon']}</h1>", unsafe_allow_html=True)
    
    card_class = "doctor-card-happy" if emotion == "happy" else "doctor-card-concerned"
    st.markdown(f"""
    <div class='doctor-card {card_class}'>
        <h2 style='text-align: center; margin: 0;'>{doctor_data['icon']} {doctor_data['status']}</h2>
        <h3 style='text-align: center; margin: 10px 0;'>Résultat: {class_name}</h3>
        <h3 style='text-align: center; margin: 10px 0;'>Probabilité: {conf_score*100:.2f}%</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 📊 Probabilités calculées")
    confidence_percentage = int(conf_score * 100)
    
    bar_color = "red" if emotion == "concerned" else "green"
    st.markdown(f"<style>.stProgress > div > div > div > div {{ background-color: {bar_color}; }}</style>", unsafe_allow_html=True)
    st.progress(float(conf_score))
    
    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        st.metric("Confiance du modèle", f"{confidence_percentage}%")
    with col_conf2:
        if conf_score > 0.90:
            st.metric("Certitude", "Très élevée", delta="+++")
        elif conf_score > 0.75:
            st.metric("Certitude", "Élevée", delta="++")
        else:
            st.metric("Certitude", "Moyenne", delta="⚠️")
    
    # MESSAGE MÉDECIN
    st.markdown("---")
    st.markdown("## 💬 Interprétation")
    
    st.markdown(f"### {empathetic_msg['greeting']}")
    st.markdown(empathetic_msg['main_message'])
    
    if is_sick:
        st.markdown(f'<div class="alert-box">{empathetic_msg["reassurance"]}</div>', unsafe_allow_html=True)
        st.error(empathetic_msg['urgency'])
    else:
        st.markdown(f'<div class="success-box">{empathetic_msg["reassurance"]}</div>', unsafe_allow_html=True)
        st.info(empathetic_msg['prevention'])
    
    # AUDIO
    st.markdown("---")
    st.markdown("## 🔊 Rapport Audio")
    
    col_audio1, col_audio2 = st.columns([3, 2])
    with col_audio1:
        if st.button("🎙️ Écouter le rapport", use_container_width=True):
            with st.spinner("🎙️ Génération de la voix..."):
                try:
                    if is_sick:
                        text_to_speak = f"Le modèle expérimental classe l'image dans la catégorie {class_name}, avec un score de {int(conf_score*100)} pourcents. Ce résultat n'est pas un diagnostic."
                    else:
                        text_to_speak = f"Le modèle expérimental classe l'image dans la catégorie normale, avec un score de {int(conf_score*100)} pourcents. Ce résultat n'exclut pas une maladie."

                    tts = gTTS(text=text_to_speak, lang='fr', slow=False)
                    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
                    temp_filename = temp_file.name
                    temp_file.close()
                    tts.save(temp_filename)
                    with open(temp_filename, 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    with col_audio2:
                        lottie_talking = load_lottie_url("https://assets1.lottiefiles.com/packages/lf20_kyu7xb1v.json")
                        if lottie_talking:
                            st_lottie(lottie_talking, height=150, key="talking_animation")
                    st.success("✅ Audio généré !")
                    st.audio(audio_bytes, format='audio/mp3')
                    try:
                        time.sleep(0.5)
                        os.unlink(temp_filename)
                    except:
                        pass
                except Exception as e:
                    st.error(f"❌ Erreur audio: {str(e)}")
    
    # ATTENTION MAP
    st.markdown("---")
    st.markdown("## 🔥 Zones affectées (Heatmap)")
    
    with st.spinner("🔥 Localisation des zones suspectes..."):
        attention_map = simple_attention_map(model, image, class_idx)
    
    if attention_map is not None:
        col_heat1, col_heat2 = st.columns(2)
        with col_heat1:
            st.markdown("#### 📷 Image originale")
            st.image(image, use_column_width=True)
        with col_heat2:
            st.markdown("#### 🔥 Zones identifiées par l'IA")
            st.image(attention_map, use_column_width=True)
    else:
        st.warning("⚠️ Impossible de générer la carte d'attention")

    # Nouvelle analyse
    st.markdown("---")
    if st.button("🔄 Analyser une autre image", use_container_width=True):
        st.rerun()

# J'ai supprimé tout le bloc 'else' qui était ici (le message d'accueil et l'avion)
