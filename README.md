## Installation de l'environnement virtuel

Créer l'environnement à partir du fichier yaml
```bash
conda env create -f environment.yml
```

Activer l'environnement
```bash
conda activate projet_4
```

Quitter l'environnement
```bash
conda deactivate projet_4
```

Supprimer l'environnement
```bash
conda env remove --name projet_4
```

## Téléchargement du jeu de données

Récupérer les jeux de données <a href = https://s3-eu-west-1.amazonaws.com/static.oc-static.com/prod/courses/files/Parcours_data_scientist/Projet+-+Impl%C3%A9menter+un+mod%C3%A8le+de+scoring/Projet+Mise+en+prod+-+home-credit-default-risk.zip>à cette adresse</a>

Dézipper et placer les fichiers csv dans le dossier "data/"

