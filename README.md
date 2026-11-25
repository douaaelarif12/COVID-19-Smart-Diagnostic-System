# COVID-19 X-ray Deep Learning Demo

Prototype académique de classification d'images thoraciques avec TensorFlow/Keras. Le projet fournit une interface Streamlit, une API FastAPI et une visualisation Grad-CAM.

> [!WARNING]
> Ce logiciel est une démonstration pédagogique. Il n'est pas validé comme dispositif médical, ne fournit pas de diagnostic et ne doit pas guider une décision de soin. Toute interprétation clinique appartient à un professionnel de santé qualifié.

## Fonctionnalités

- prétraitement RGB en `224 × 224` et normalisation dans `[0, 1]` ;
- classification multiclasse selon les étiquettes du modèle entraîné ;
- score produit par la sortie du réseau ;
- carte Grad-CAM calculée depuis la dernière couche convolutionnelle compatible ;
- interface Streamlit pour analyser une image ;
- API FastAPI avec documentation OpenAPI sur `/docs` ;
- rapport audio optionnel avec gTTS.

## Limites actuelles

- Le fichier entraîné `covid_detector.h5` n'est pas publié dans le dépôt ; il doit être fourni localement.
- Les données d'entraînement, le protocole de séparation et les métriques ne sont pas inclus ; aucune performance clinique n'est donc revendiquée.
- L'ordre des classes et le prétraitement sont cohérents avec le code d'inférence d'origine, mais ils ne sont pas confirmés par un artefact d'entraînement disponible.
- Le contrôle de qualité vérifie seulement des propriétés élémentaires de l'image. Il ne prouve pas qu'une image est une radiographie thoracique valide.
- Le score du réseau ne doit pas être interprété comme une probabilité clinique calibrée.

## Structure

```text
.
├── app.py                         # Interface Streamlit
├── api.py                         # API FastAPI
├── util.py                        # Prétraitement, messages et Grad-CAM
├── model/
│   ├── README.md
│   ├── class_names.example.txt
│   ├── class_names.txt
│   └── model_metadata.json        # Empreinte, contrat d'entrée/sortie et provenance
├── model_config.py                # Validation bloquante de la configuration
├── bgs/                           # Fond optionnel
├── Dockerfile
├── docker-compose.yml
├── requirements-docker.txt
├── requirements.txt
└── run.bat
```

## Préparation du modèle

1. Placez le modèle dans `model/covid_detector.h5`.
2. Vérifiez `model/class_names.txt`, qui doit contenir une classe par ligne dans l'ordre exact des sorties du modèle.

Exemple :

```text
Normal
Pneumonie Virale
COVID-19
```

Le modèle inspecté produit trois sorties Softmax. Le code source du livrable d'origine associe ces sorties, dans l'ordre, à `Normal`, `Pneumonie Virale` et `COVID-19`. Le fichier `labels.txt` fourni ne contient que deux lignes et appartient à un autre modèle binaire ; il n'est donc pas compatible avec ce modèle.

L'application compare au démarrage le fichier local à `model/model_metadata.json` : empreinte SHA-256, forme d'entrée, nombre de sorties, activation et classes déclarées. Cela garantit la cohérence technique du fichier utilisé. En l'absence du notebook ou du script d'entraînement, l'ordre des classes et le prétraitement ne peuvent toutefois pas être certifiés scientifiquement ; cet état est affiché dans Streamlit et exposé par `/health` et `/predict`.

## Installation

TensorFlow 2.12 nécessite une version de Python compatible ; Python 3.10 est recommandé pour reproduire cet environnement.

```bash
python -m venv .venv
```

Sous Windows :

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Lancement

Interface Streamlit :

```bash
streamlit run app.py
```

API FastAPI :

```bash
uvicorn api:app --reload
```

- Interface : <http://localhost:8501>
- API : <http://localhost:8000>
- Swagger : <http://localhost:8000/docs>

## Lancement avec Docker

Docker évite l'incompatibilité entre TensorFlow 2.12 et Python 3.14 sur la machine hôte. Le modèle reste local et est monté en lecture seule dans les conteneurs ; il n'est pas intégré à l'image Docker.

Vérifiez d'abord la présence de :

```text
model/covid_detector.h5
model/class_names.txt
model/model_metadata.json
```

Construisez et démarrez Streamlit et FastAPI :

```bash
docker compose up --build -d
```

Vérifiez leur état :

```bash
docker compose ps
```

Consultez les journaux :

```bash
docker compose logs -f
```

Arrêtez les services :

```bash
docker compose down
```

## Vérifications recommandées

Avant toute démonstration, documentez la provenance des données, la répartition entraînement/validation/test, les métriques par classe, la matrice de confusion, les limites et les biais. Testez également des images hors distribution et des entrées invalides.

