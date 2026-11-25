import streamlit as st
from keras.models import load_model
import requests
from streamlit_lottie import st_lottie
from gtts import gTTS
import os
import tempfile
import time
import hashlib
from pathlib import Path

from inference import (
    InvalidImageError,
    check_image_contrast,
    decode_image,
    predict_image,
)
from model_config import load_and_validate_model_configuration

# Import des fonctions utilitaires
from util import (
    set_background,
    simple_attention_map,
    get_empathetic_message,
    get_doctor_animation_data,
)

# Configuration de la page
st.set_page_config(
    page_title="Classification thoracique expérimentale",
    page_icon="🦠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Fonction pour charger les animations Lottie
@st.cache_data
def load_lottie_url(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except requests.RequestException:
        return None

# Définir le fond
try:
    if not Path('./bgs/bg5.png').is_file():
        raise FileNotFoundError("Fond optionnel absent.")
    set_background('./bgs/bg5.png')
except (FileNotFoundError, OSError):
    pass

# Titre principal
st.title("🫁 Classification thoracique expérimentale", text_alignment="center")
st.caption(
    "Démonstrateur académique de classification d’images par Deep Learning",
    text_alignment="center",
)
st.warning(
    "**Usage pédagogique uniquement.** Cette application n’est pas un dispositif "
    "médical, ne fournit pas de diagnostic et ne doit pas guider une décision de "
    "soin. Faites interpréter toute radiographie par un professionnel qualifié.",
    icon=":material/warning:",
)

st.session_state.setdefault("total_analyses", 0)
st.session_state.setdefault("last_counted_file", None)

# Sidebar (Minimaliste)
with st.sidebar:
    st.markdown("### À propos")
    st.caption("CNN Keras · entrée RGB 224 × 224 · trois classes")
    st.markdown("### 📊 Session")
    st.metric("Analyses effectuées", st.session_state.total_analyses)
    

   

# --- CHARGEMENT DU MODÈLE ---
@st.cache_resource
def load_model_and_labels():
    model_path = Path('./model/covid_detector.h5')
    labels_path = Path('./model/class_names.txt')
    metadata_path = Path('./model/model_metadata.json')
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

    model_configuration = load_and_validate_model_configuration(
        model, model_path, class_names, metadata_path
    )

    return model, class_names, model_configuration

try:
    with st.spinner("🔄 Chargement du modèle de détection COVID-19..."):
        model, class_names, model_configuration = load_model_and_labels()
    st.success("✅ Modèle chargé et intégrité technique vérifiée.")
    if not model_configuration["configuration_verified"]:
        st.warning(
            "⚠️ Configuration scientifique non certifiée : l'ordre des classes et le "
            "prétraitement proviennent du code d'inférence d'origine, mais les artefacts "
            "d'entraînement nécessaires à leur confirmation sont absents. Les résultats "
            "restent expérimentaux."
        )
    with st.expander("Configuration technique du modèle"):
        st.write("Classes déclarées :", class_names)
        st.code(model_configuration["model_sha256"], language=None)
        st.caption("Empreinte SHA-256 du fichier local chargé")
except Exception as e:
    st.error(f"❌ Erreur lors du chargement: {str(e)}")
    st.stop()

# Upload
st.markdown("## 1. Importer une image")
st.caption("Formats acceptés : JPEG ou PNG · 10 Mio maximum · 64 pixels minimum par côté")

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
    contents = file.getvalue()
    try:
        image = decode_image(contents)
    except InvalidImageError as exc:
        st.error(f"Image refusée : {exc}")
        st.stop()

    file_fingerprint = hashlib.sha256(contents).hexdigest()
    if st.session_state.get("last_counted_file") != file_fingerprint:
        st.session_state.total_analyses += 1
        st.session_state.last_counted_file = file_fingerprint
    
    # ÉTAPE 1: VALIDATION
    st.markdown("## 2. Contrôle technique")
    col_img, col_validation = st.columns([1, 1])
    
    with col_img:
        st.markdown("### 📷 Image reçue")
        st.image(image, width="stretch", caption="Image analysée")
    
    with col_validation:
        st.markdown("### Contrôle de contraste")
        with st.spinner("🔍 Vérification en cours..."):
            is_valid, _, validation_reason = check_image_contrast(image)
        
        if is_valid:
            st.success("✅ **Contraste suffisant**")
            st.info(validation_reason)
        else:
            st.error("❌ **Attention**")
            st.warning("L’image ne présente pas un contraste suffisant pour poursuivre.")
            st.stop()

    confirmed_xray = st.checkbox(
        "Je confirme qu’il s’agit d’une radiographie thoracique dé-identifiée "
        "et destinée uniquement à une démonstration pédagogique."
    )
    if not confirmed_xray:
        st.info("Confirmez la nature de l’image avant de lancer la classification.")
        st.stop()
    
    # ÉTAPE 2: CLASSIFICATION
    st.markdown("## 3. Classification expérimentale")
    
    with st.spinner("🧠 Analyse des tissus pulmonaires en cours..."):
        result = predict_image(image, model, class_names)
        class_name = result.class_name
        conf_score = result.score
        class_idx = result.class_index
    
    st.success("✅ Analyse terminée !")
    
    # --- LOGIQUE D'ÉMOTION DU MÉDECIN ---
    is_sick = "NORMAL" not in class_name.upper()
    emotion = "concerned" if is_sick else "happy"
    
    doctor_data = get_doctor_animation_data(emotion)
    empathetic_msg = get_empathetic_message(class_name, conf_score)
    
    # RÉSULTATS
    st.markdown("## Résultat expérimental")
    
    col_anim1, col_anim2, col_anim3 = st.columns([1, 2, 1])
    with col_anim2:
        lottie_result = load_lottie_url(doctor_data["url"])
        if not lottie_result:
            lottie_result = load_lottie_url(doctor_data["fallback"])
        
        if lottie_result:
            st_lottie(lottie_result, height=250, key=f"result_animation_{emotion}")
        else:
            st.title(doctor_data["icon"], text_alignment="center")

    with st.container(border=True):
        st.subheader(f"{doctor_data['icon']} {doctor_data['status']}")
        result_col, score_col = st.columns(2)
        result_col.metric("Classe majoritaire", class_name)
        score_col.metric("Score du modèle", f"{conf_score * 100:.2f}%")
    
    st.markdown("### Scores par classe")
    confidence_percentage = int(conf_score * 100)
    
    for score_class, score_value in result.scores.items():
        st.caption(f"{score_class} — {score_value * 100:.2f}%")
        st.progress(float(score_value))

    col_conf1, col_conf2 = st.columns(2)
    with col_conf1:
        st.metric("Score maximal", f"{confidence_percentage}%")
    with col_conf2:
        st.metric("Classe majoritaire", class_name)
    st.caption(
        "Ces scores sont les sorties Softmax du réseau. Ils ne représentent pas "
        "une probabilité clinique et leur calibration n’a pas été évaluée."
    )
    
    # MESSAGE MÉDECIN
    st.markdown("## Interprétation")
    
    st.markdown(f"### {empathetic_msg['greeting']}")
    st.markdown(empathetic_msg['main_message'])
    
    if is_sick:
        st.warning(empathetic_msg["reassurance"], icon=":material/warning:")
        st.error(empathetic_msg['urgency'], icon=":material/medical_services:")
    else:
        st.info(empathetic_msg["reassurance"], icon=":material/info:")
        st.caption(empathetic_msg['prevention'])
    
    # AUDIO
    st.markdown("## Rapport audio")
    
    col_audio1, col_audio2 = st.columns([3, 2])
    with col_audio1:
        if st.button(
            "Écouter le rapport",
            icon=":material/volume_up:",
            width="stretch",
        ):
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
                    except OSError:
                        pass
                except Exception as e:
                    st.error(f"❌ Erreur audio: {str(e)}")
    
    # ATTENTION MAP
    st.markdown("## 4. Carte d’attention Grad-CAM")
    st.caption(
        "Visualisation indicative des régions qui influencent la sortie du réseau ; "
        "elle ne localise pas une lésion et ne constitue pas une explication clinique."
    )
    
    with st.spinner("Génération de la carte d’attention..."):
        attention_map = simple_attention_map(model, image, class_idx)
    
    if attention_map is not None:
        col_heat1, col_heat2 = st.columns(2)
        with col_heat1:
            st.markdown("#### 📷 Image originale")
            st.image(image, width="stretch")
        with col_heat2:
            st.markdown("#### Carte d’attention superposée")
            st.image(attention_map, width="stretch")
    else:
        st.warning("⚠️ Impossible de générer la carte d'attention")

    # Nouvelle analyse
    if st.button(
        "Analyser une autre image",
        icon=":material/refresh:",
        width="stretch",
    ):
        st.rerun()
