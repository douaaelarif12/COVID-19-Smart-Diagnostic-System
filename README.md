# COVID-19 X-ray Deep Learning Demo

[![Tests](https://github.com/douaaelarif12/COVID-19-Smart-Diagnostic-System/actions/workflows/ci.yml/badge.svg)](https://github.com/douaaelarif12/COVID-19-Smart-Diagnostic-System/actions/workflows/ci.yml)

Prototype académique de classification d'images thoraciques avec TensorFlow/Keras. Le projet fournit une interface Streamlit, une API FastAPI et une visualisation Grad-CAM.

> [!WARNING]
> Ce logiciel est une démonstration pédagogique. Il n'est pas validé comme dispositif médical, ne fournit pas de diagnostic et ne doit pas guider une décision de soin. Toute interprétation clinique appartient à un professionnel de santé qualifié.

## Fonctionnalités

- prétraitement RGB en `224 × 224` et normalisation dans `[0, 1]` ;
- validation des fichiers JPEG/PNG, de leur taille et de leur résolution ;
- classification multiclasse selon les étiquettes du modèle entraîné ;
- scores Softmax pour les trois classes, sans les présenter comme des probabilités cliniques ;
- carte Grad-CAM calculée depuis la dernière couche convolutionnelle compatible ;
- interface Streamlit pour analyser une image ;
- API FastAPI avec documentation OpenAPI sur `/docs` ;
- rapport audio optionnel avec gTTS.
- outil d'évaluation reproductible avec matrice de confusion, accuracy, précision, rappel et F1-score ;
- tests unitaires et intégration continue GitHub Actions.

## Limites actuelles

- Le fichier entraîné `covid_detector.h5` n'est pas publié dans le dépôt ; il doit être fourni localement.
- Les données d'entraînement, le protocole de séparation et les métriques ne sont pas inclus ; aucune performance clinique n'est donc revendiquée.
- L'ordre des classes et le prétraitement sont cohérents avec le code d'inférence d'origine, mais ils ne sont pas confirmés par un artefact d'entraînement disponible.
- Le contrôle de qualité vérifie seulement des propriétés élémentaires de l'image. Il ne prouve pas qu'une image est une radiographie thoracique valide.
- Le score du réseau ne doit pas être interprété comme une probabilité clinique calibrée.
- L'outil ne détecte pas automatiquement si l'image contient réellement un thorax ; le contrôle actuel mesure seulement le contraste.
- Une image hors domaine peut recevoir un score Softmax élevé. Streamlit exige donc une confirmation humaine et l'API indique toujours `chest_xray_verified: false`.

## Structure

```text
.
├── app.py                         # Interface Streamlit
├── api.py                         # API FastAPI
├── inference.py                   # Validation, prétraitement et prédiction partagés
├── util.py                        # Messages et Grad-CAM
├── evaluate.py                    # Évaluation sur un jeu de test indépendant
├── model/
│   ├── README.md
│   ├── class_names.example.txt
│   ├── class_names.txt
│   └── model_metadata.json        # Empreinte, contrat d'entrée/sortie et provenance
├── model_config.py                # Validation bloquante de la configuration
├── tests/                         # Tests unitaires sans modèle privé
├── SECURITY.md                    # Modèle de menace et risque ML résiduel
├── .github/workflows/ci.yml       # Intégration continue
├── .streamlit/config.toml         # Thème et limite d'upload
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

Le projet utilise Python 3.11 avec TensorFlow/Keras 2.12 afin de préserver la
compatibilité du modèle HDF5 tout en utilisant des composants web maintenus.

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

Sous Windows, `run.bat` propose le lancement local ou Docker et ouvre la
documentation avec Google Chrome lorsque celui-ci est installé.

### Exemple API

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@radiographie.png"
```

La réponse contient la classe majoritaire, son index, le score de chaque classe,
le statut de vérification de la configuration et l'avertissement médical. Les
anciens champs `confidence_score` et `confidence_percentage` restent présents
pour préserver la compatibilité, mais ils représentent un score du modèle.

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

Les processus des conteneurs s'exécutent avec un utilisateur non privilégié. Le
modèle local est monté en lecture seule et n'est jamais copié dans l'image.

## Évaluer le modèle

Utilisez exclusivement un jeu de **test indépendant**, jamais des images ayant
servi à l'entraînement. Son organisation doit reprendre exactement les classes :

```text
test_dataset/
├── Normal/
├── Pneumonie Virale/
└── COVID-19/
```

Lancez ensuite :

```bash
python evaluate.py --dataset chemin/vers/test_dataset
```

Dans Docker :

```bash
docker compose run --rm -v "chemin/vers/test_dataset:/data/test:ro" api \
  python evaluate.py --dataset /data/test
```

Les fichiers `reports/evaluation/metrics.json` et `predictions.csv` sont générés
localement et ignorés par Git. Sans dataset de test, aucune métrique n'est
annoncée dans ce dépôt.

## Tests

Les tests légers ne nécessitent pas le modèle privé :

```bash
python -m unittest discover -s tests -v
```

Les vérifications d'intégration nécessitent le modèle local :

```bash
docker compose up --build -d
docker compose ps
```

## Confidentialité

N'ajoutez jamais au dépôt des radiographies contenant des données personnelles.
Les dossiers `data/`, `datasets/` et `reports/` sont ignorés par Git. L'image
envoyée à Streamlit ou FastAPI est traitée en mémoire ; seule la génération audio
optionnelle utilise le service externe gTTS, avec un texte de résultat et non
l'image elle-même.

## Vérifications recommandées

Avant toute démonstration, documentez la provenance des données, la répartition entraînement/validation/test, les métriques par classe, la matrice de confusion, les limites et les biais. Testez également des images hors distribution et des entrées invalides.

