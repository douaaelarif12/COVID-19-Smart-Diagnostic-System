# Modèle entraîné

Le modèle n'est pas inclus dans ce dépôt. Placez le fichier entraîné à cet emplacement :

```text
model/covid_detector.h5
```

Les fichiers versionnés `class_names.txt` et `model_metadata.json` décrivent uniquement le modèle dont l'empreinte SHA-256 est :

```text
61a59482d674f9c1e78f487dd55f805e87bbfa6511bd0ee0d8cfeb9aa64ecc31
```

Au démarrage, l'application vérifie cette empreinte, la forme d'entrée, le nombre de sorties, l'activation et l'ordre déclaré des classes. Une divergence bloque le chargement au lieu de produire silencieusement des résultats incorrects.

## Statut de vérification

- **Vérifié techniquement :** entrée `(224, 224, 3)`, trois sorties Softmax et intégrité du fichier HDF5.
- **Non confirmé par l'entraînement :** ordre `Normal`, `Pneumonie Virale`, `COVID-19` et normalisation `/255.0`.
- **Source disponible :** ces deux éléments proviennent du code d'inférence du livrable d'origine.
- **Source manquante :** aucun notebook, script ou générateur de données d'entraînement n'a été retrouvé pour les confirmer.

Le fichier `labels.txt` remis séparément contient seulement `PNEUMONIA` et `NORMAL`. Il appartient à un autre modèle binaire et ne doit pas remplacer `class_names.txt`.

Pour utiliser un autre modèle, ne modifiez pas uniquement les classes : générez aussi des métadonnées correspondant au nouveau fichier et confirmez l'ordre depuis les artefacts d'entraînement.

