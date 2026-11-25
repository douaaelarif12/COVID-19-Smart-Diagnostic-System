# Politique de sécurité

## Signaler un problème

N'ouvrez pas publiquement un ticket contenant des données médicales, une image de
patient, un secret ou les détails permettant d'exploiter une vulnérabilité.
Contactez d'abord la mainteneuse du dépôt par un canal privé GitHub.

## Modèles de confiance

L'application ne doit charger que le fichier `covid_detector.h5` connu. Son
empreinte SHA-256 est vérifiée au démarrage contre `model/model_metadata.json`.
Ne remplacez jamais l'empreinte pour exécuter un fichier HDF5 reçu d'une source
inconnue : les formats de modèles sérialisés doivent être considérés comme du
contenu potentiellement dangereux.

## Risque résiduel de la pile ML

Le modèle historique dépend de TensorFlow 2.12 et Keras 2.12. Des avis de
sécurité existent sur cette génération de Keras et sur sa version de Protobuf.
Streamlit 1.60 impose également Pillow `<13`. Les composants web et d'upload ont
été modernisés, mais une migration complète vers Keras 3 exige une conversion
contrôlée du modèle et une comparaison des sorties sur un jeu de non-régression.

Mesures compensatoires actuelles :

- empreinte SHA-256 obligatoire avant le chargement ;
- modèle monté en lecture seule dans Docker ;
- conteneur exécuté par un utilisateur non privilégié ;
- formats, taille et résolution des images contrôlés ;
- aucune donnée médicale ni aucun modèle inclus dans le dépôt public.

## Données sensibles

Utilisez uniquement des images dé-identifiées et autorisées. Les dossiers
`data/`, `datasets/` et `reports/` sont ignorés par Git, mais cette protection ne
remplace pas une procédure de gouvernance des données médicales.
