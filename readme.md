# Solar Research: Modélisation et Prédiction de l'Irradiance Urbaine

Ce projet est une plateforme d'expérimentation locale (Data Science) dédiée au calcul de l'irradiance solaire sur des façades urbaines. Il s'agit d'une refactorisation complète d'une pipeline de production vers un environnement synchrone optimisé pour l'analyse sous **Jupyter Notebook**.

## 🔬 Concepts Clés
- **Modèle de Perez :** Calcul de l'irradiance globale inclinée (GTI) en tenant compte de l'anisotropie du ciel.
- **Vision Fisheye :** Utilisation de l'IA **MiDaS** pour segmenter le ciel et extraire un profil d'horizon à 360°.
- **Optimisation $k_d$ :** Calibration de la fraction diffuse via un solveur *Grid Search* sur mesures réelles.

## 📁 Structure du Projet
Le code est divisé en 6 fichiers sources dans `src/` :
- `names.py` : Dataclasses et structures de données (contrats).
- `values.py` : Constantes physiques et matrices de Perez.
- `utils.py` : Fonctions mathématiques pures.
- `vision.py` : Traitement d'images et extraction d'horizon.
- `solver.py` : Algorithme d'optimisation du paramètre $k_d$.
- `SolarModel.py` : Orchestrateur principal (Façade du projet).

## ⚙️ Installation et Environnement
Le projet utilise **Poetry** pour la gestion des dépendances et un environnement virtuel local (`.venv`).

### 1. Initialisation de l'environnement
```bash
# S'assurer que le .venv est créé à la racine du projet
poetry config virtualenvs.in-project true

# Installation des dépendances standard via Poetry
poetry install
```

### 2. Installation de l'astronomie (sg2)
Le package `sg2` doit être installé manuellement **à l'intérieur** du `.venv` pour contourner les limitations du serveur privé :
```powershell
# Sur Windows (PowerShell)
./.venv/Scripts/python.exe -m pip install sg2 -f https://pip.oie-lab.net/python/
```

## 🧪 Validation et Tests (Notebooks)
Le projet inclut deux notebooks de test interactifs permettant de vérifier la conformité des sorties des différentes fonctions :

1. **`01_test_vision.ipynb`** : Dédié à la partie perception. Il permet de visualiser le masque MiDaS, la projection équirectangulaire et la validité du profil d'horizon extrait de l'image.
2. **`02_test_model.ipynb`** : Dédié à la partie physique. Il valide l'intégration des mesures, le bon fonctionnement du solveur pour trouver le $k_d$ optimal et la cohérence de la projection finale sur les pyranomètres virtuels.

### Lancement
Activez l'environnement via `poetry shell` puis lancez :
```bash
jupyter lab
```

## 🛠 Maintenance
- **Ajouter un paquet :** `poetry add <package>`
- **Note importante :** N'utilisez pas `poetry install --sync` car cela supprimerait `sg2` (installé manuellement via pip).