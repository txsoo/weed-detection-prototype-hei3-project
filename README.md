# Pomone - Détection de Mauvaises Herbes en Temps Réel

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![YOLOv11](https://img.shields.io/badge/YOLO-v11-green.svg)](https://github.com/ultralytics/ultralytics)

**Pomone** est un système de détection de mauvaises herbes en temps réel utilisant YOLO et une interface HUD (Heads-Up Display) transparente. Le système capture l'écran en continu, effectue des inférences de détection d'objets et affiche les résultats sous forme d'overlay transparent par-dessus n'importe quelle application.

## 🌿 À Propos

Ce projet implémente un système de détection capable d'identifier **25 types de mauvaises herbes** différentes en temps réel. L'overlay transparent permet de visualiser les détections sans interrompre le flux de travail, idéal pour l'analyse d'images agricoles, la recherche ou l'éducation.

### Classes Détectées

Le modèle peut identifier les 25 espèces de mauvaises herbes suivantes :

- Alligatorweed
- Asiatic Smartweed
- Bidens pilosa
- Black nightshade
- Ceylon spinach
- Chinese knotweed
- Common Dayflower
- Indian aster
- Mock strawberry
- Shepherd's Purse
- Viola
- Abutilon theophrasti
- Barnyard grass
- Billygoat weed
- Cocklebur
- Crabgrass
- Field thistle
- Goosefoots
- Green foxtail
- Horseweed
- Pigweed
- Plantain
- Purslane
- Sedge
- White smart weed

## 📊 Dataset

Ce projet utilise le dataset **Weed25** disponible sur Roboflow Universe :

- **Nom du dataset** : Weed25 Labeling
- **Workspace** : weed25-dataset
- **Version** : 4
- **Licence** : CC BY 4.0
- **URL** : [https://universe.roboflow.com/weed25-dataset/weed25-labeling/dataset/4](https://universe.roboflow.com/weed25-dataset/weed25-labeling/dataset/4)

Le dataset contient des images annotées de 25 espèces différentes de mauvaises herbes, permettant l'entraînement de modèles de détection d'objets pour l'agriculture de précision.

## ✨ Fonctionnalités

- 🎯 **Détection en temps réel** : Inférence continue sur la capture d'écran
- 🪟 **Overlay transparent** : Affichage HUD non-intrusif par-dessus n'importe quelle application
- 🖥️ **Support multi-moniteurs** : Sélection du moniteur à capturer
- ⚡ **Accélération GPU** : Support CUDA pour des performances optimales
- 📊 **Affichage FPS** : Statistiques de performance en temps réel
- 🎨 **Couleurs par classe** : Chaque espèce a une couleur distinctive
- ⚙️ **Hautement configurable** : Seuils de confiance, taille d'inférence, intervalle de frames, etc.

## 🚀 Installation

### Prérequis

- Python 3.8 ou supérieur
- GPU NVIDIA avec support CUDA (recommandé pour de meilleures performances)
- Windows, Linux ou macOS

### Installation des Dépendances

1. Clonez le dépôt :
```bash
git clone https://github.com/votre-username/pomone.git
cd pomone
```

2. Installez les dépendances :
```bash
pip install -r requirements.txt
```

Les principales dépendances incluent :
- `ultralytics` : Framework YOLO
- `opencv-python` : Traitement d'images
- `PyQt5` : Interface graphique overlay
- `mss` : Capture d'écran rapide
- `torch` : Backend PyTorch avec support CUDA

## 📖 Utilisation

### Démarrage Rapide

```bash
python overlay.py --weights runs/detect/train/weights/best.pt --data Architecture/data.yaml --show-fps
```

### Lister les Moniteurs Disponibles

```bash
python overlay.py --list-monitors
```

### Options de Ligne de Commande

| Option | Type | Défaut | Description |
|--------|------|--------|-------------|
| `--weights` | Path | `runs/detect/train/weights/best.pt` | Chemin vers le fichier de poids YOLO |
| `--data` | Path | `None` | Fichier YAML du dataset pour les noms de classes |
| `--monitor` | int | `1` | Index du moniteur à capturer |
| `--list-monitors` | flag | - | Liste les moniteurs disponibles et quitte |
| `--conf` | float | `0.25` | Seuil de confiance pour les détections |
| `--frame-interval` | float | `0.05` | Délai minimum entre les inférences (secondes) |
| `--device` | str | `None` | Identifiant du device Torch (ex: '0' ou 'cpu') |
| `--imgsz` | int | `None` | Taille d'image pour l'inférence |
| `--show-fps` | flag | - | Affiche le compteur d'objets et les FPS |
| `--thickness` | int | `3` | Épaisseur des boîtes englobantes (pixels) |
| `--font-size` | int | `14` | Taille de la police de l'overlay (points) |

### Exemples d'Utilisation

**Détection avec affichage FPS et seuil de confiance élevé :**
```bash
python overlay.py --weights yolo11n.pt --data Architecture/data.yaml --conf 0.5 --show-fps
```

**Utilisation du CPU uniquement :**
```bash
python overlay.py --weights yolo11n.pt --device cpu
```

**Capture du deuxième moniteur avec inférence plus lente :**
```bash
python overlay.py --monitor 2 --frame-interval 0.1
```

**Personnalisation de l'apparence :**
```bash
python overlay.py --thickness 5 --font-size 16 --show-fps
```

## 🏗️ Architecture

Le projet est structuré autour de trois composants principaux :

### 1. `DetectionWorker` (Thread)
- Capture l'écran en continu avec `mss`
- Effectue les inférences YOLO sur chaque frame
- Transmet les détections au composant d'affichage

### 2. `DetectionOverlay` (QWidget)
- Fenêtre PyQt5 transparente et toujours au premier plan
- Reçoit les détections et les affiche
- Gère le rendu des boîtes englobantes et des labels

### 3. Système de Détection
- Chargement du modèle YOLO
- Résolution des noms de classes depuis le fichier data.yaml
- Post-traitement des résultats d'inférence

## 🎓 Entraînement du Modèle

Pour entraîner votre propre modèle sur le dataset Weed25 :

```python
from ultralytics import YOLO

# Charger un modèle pré-entraîné
model = YOLO('yolo11n.pt')

# Entraîner le modèle
results = model.train(
    data='Architecture/data.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    device=0  # GPU
)
```

Les poids entraînés seront sauvegardés dans `runs/detect/train/weights/best.pt`.

## 🛠️ Configuration

Le fichier `Architecture/data.yaml` contient la configuration du dataset :

```yaml
train: ../train/images
val: ../valid/images
test: ../test/images

nc: 25
names: ['Alligatorweed', 'Asiatic_Smartweed', ...]

roboflow:
  workspace: weed25-dataset
  project: weed25-labeling
  version: 4
  license: CC BY 4.0
  url: https://universe.roboflow.com/weed25-dataset/weed25-labeling/dataset/4
```

## 🔧 Dépannage

### L'overlay ne s'affiche pas
- Vérifiez que le fichier de poids existe
- Assurez-vous que PyQt5 est correctement installé
- Essayez de changer le moniteur avec `--monitor`

### Performances faibles
- Utilisez un GPU avec `--device 0`
- Augmentez `--frame-interval` pour réduire la fréquence d'inférence
- Utilisez un modèle plus petit (yolo11n.pt au lieu de yolo11s.pt)
- Réduisez `--imgsz` pour des inférences plus rapides

### Erreurs CUDA
- Vérifiez que PyTorch est installé avec support CUDA
- Utilisez `--device cpu` pour forcer l'utilisation du CPU

## 📝 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

Le dataset utilisé (Weed25) est sous licence **CC BY 4.0**.

## 🙏 Remerciements

- **Roboflow** pour l'hébergement du dataset Weed25
- **Ultralytics** pour le framework YOLO
- La communauté open-source pour les outils et bibliothèques utilisés

## 📧 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une issue sur GitHub.

---

**Note** : Ce projet est destiné à des fins éducatives et de recherche. Pour une utilisation en production dans des applications agricoles, des tests et validations supplémentaires sont recommandés.
