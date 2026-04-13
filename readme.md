
# Solar Research: Modélisation et Prédiction de l'Irradiance Urbaine

Ce projet est une plateforme de Recherche et Développement (Data Science) dédiée au calcul de l'irradiance solaire en milieu urbain dense. 

Refactorisé à partir d'un pipeline de production asynchrone, cet environnement s'exécute localement via des **Jupyter Notebooks**. Il permet aux chercheurs et ingénieurs de visualiser, tester et optimiser les algorithmes de vision par ordinateur et de physique atmosphérique (Modèle de Perez) en temps réel.

## ✨ Fonctionnalités
- **Extraction d'Horizon par IA :** Utilisation du réseau de neurones **MiDaS** (estimation de profondeur) pour détourer les bâtiments sur des images fisheye et générer des profils d'horizon à 360°.
- **Horizons Multiples (Toit vs Façade) :** Découplage géométrique permettant d'utiliser l'horizon réel du toit pour calibrer l'atmosphère, et un horizon virtuel altéré pour projeter l'énergie sur les façades cibles.
- **Pipeline 1D (Perez Standard) :** Calibration de la fraction diffuse ($k_d$) par méthode de *Grid Search* en utilisant les matrices empiriques de Perez (1990).
- **Pipeline 3D (Machine Learning) :** Solveur avancé optimisant simultanément la fraction diffuse ($k_d$), la brillance circumsolaire ($f_1$) et la brillance d'horizon ($f_2$). Cette approche s'affranchit des tables empiriques pour s'adapter aux micro-climats urbains atypiques.

## 📁 Structure du Projet

Le code est modulaire et respecte une séparation stricte des responsabilités :

```text
solar_research/
├── data/                    # Données d'entrée (images fisheye, JSON de mesures, cache)
├── notebooks/               # Environnements de test interactifs
│   ├── 01_test_vision.ipynb # Visualisation MiDaS et extraction d'horizon
│   └── 02_test_model_3d.ipynb # Comparaison des solveurs 1D vs 3D et projection
│
├── src/                     # Code métier refactorisé
│   ├── names.py             # Contrats de données (@dataclass : RealPyrano, etc.)
│   ├── values.py            # Constantes physiques et tables empiriques de Perez
│   ├── utils.py             # Fonctions mathématiques, astronomie (sg2) et intégration Riso
│   ├── vision.py            # Pipeline optique (Fisheye -> Equirectangulaire -> Horizon)
│   ├── solver.py            # Algorithmes d'optimisation (Grid Search 1D et L-BFGS-B 3D)
│   └── SolarModel.py        # Orchestrateur (API publique pour les notebooks)
│
└── pyproject.toml           # Configuration Poetry (package-mode = false)
```

## ⚙️ Installation de l'Environnement

Le projet utilise **Poetry** pour garantir la reproductibilité des dépendances scientifiques. 
*Prérequis : Python 3.11 ou 3.12 (Python 3.13 n'est pas supporté par certaines librairies C++).*

### 1. Installation des dépendances standards
```powershell
# S'assurer que Poetry crée le .venv à l'intérieur du projet
poetry config virtualenvs.in-project true

# Installer les dépendances (Numpy, Pandas, Torch, OpenCV, SciPy, etc.)
poetry install
```

### 2. Installation de l'astronomie (sg2)
Le module de calcul astronomique `sg2` (OIE Lab) est hébergé sur un serveur de recherche privé. Il doit être installé manuellement à l'intérieur du `.venv` en utilisant le binaire local :

**Sur Windows (PowerShell) :**
```powershell
./.venv/Scripts/python.exe -m pip install sg2 -f https://pip.oie-lab.net/python/
```

## 🚀 Utilisation (Jupyter Lab)

1. Activez l'environnement :
   ```powershell
   # Activation sous Windows
   .\.venv\Scripts\activate
   ```
2. Lancez l'interface Jupyter :
   ```powershell
   jupyter lab
   ```
3. **Ordre d'exécution recommandé :**
   * Ouvrez `01_test_vision.ipynb` pour vérifier que le modèle MiDaS est bien téléchargé (en cache) et que le masque de la ville est correctement généré.
   * Ouvrez `02_test_model_3d.ipynb` pour lancer la simulation comparative des solveurs 1D et 3D sur une journée complète, avec affichage des graphiques d'irradiance sur les façades virtuelles.

## 📊 Outputs et Interprétation
Le modèle génère des DataFrames d'irradiance temporelle et exporte les résultats dans le dossier `data/`.
* Un gain significatif de précision ($>20\%$) en **Pipeline 3D** indique des conditions atmosphériques locales (ex: forte pollution urbaine) justifiant la libération des variables $f_1$ et $f_2$.
