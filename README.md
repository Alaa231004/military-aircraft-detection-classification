# Détection et classification d'aéronefs militaires

Pipeline de Computer Vision pour **détecter** et **identifier** automatiquement des aéronefs militaires dans des images aériennes, parmi **101 classes**.

**Auteurs :** Alaa CHACHA · Hajar GUEMILI

## Approche

| Tâche | Modèle | Rôle |
|---|---|---|
| Détection | **YOLOv8s** (transfer learning COCO) | Localiser les aéronefs (bounding boxes) |
| Classification | **Random Forest** (HOG + histogramme couleur + LBP) | Approche classique (baseline) |
| Classification | **CNN ResNet50** (fine-tuning ImageNet) | Deep learning |
| Classification | **EfficientNetV2-S** | Deep learning |

## Dataset

[Military Aircraft Detection Dataset (Kaggle)](https://www.kaggle.com/datasets/a2015003713/militaryaircraftdetectiondataset) :
23 126 images · 41 441 annotations · 101 classes · ~1,79 avion/image.

- `dataset/` : images brutes + annotations → détection (YOLOv8)
- `crop/` : aéronefs découpés par classe → classification
- `labels_with_split.csv` : `filename, width, height, class, xmin, ymin, xmax, ymax, split`

Répartition : train 17 326 · val 4 256 · test 1 544 images. Le dataset est **fortement déséquilibré** (F16 : 2 061 annotations contre 11 pour MQ20).

## Preprocessing (détection)

1. Suppression des bbox invalides (inversées, hors image, < 10 px) : 59 annotations retirées
2. Encodage des 101 classes (0 à 100) → `classes.txt`
3. Conversion Pascal VOC → format YOLO (coordonnées normalisées `cx, cy, w, h`)
4. Structure `train/val/test` (`images/` + `labels/`)
5. Fichier de configuration `aircraft.yaml`

## Résultats

### Détection – YOLOv8s (50 epochs, Tesla T4, 640×640)

| mAP@0.5 | Précision | Rappel | Vitesse |
|---|---|---|---|
| 0,743 | 0,821 | 0,698 | 8,6 ms/image |

### Classification (sur les crops, split test)

| Modèle | Top-1 | Top-5 | Vitesse |
|---|---|---|---|
| Random Forest (HOG+couleur+LBP) | 16,29 % | 36,15 % | 6 837 FPS |
| EfficientNetV2-S | 50,96 % | 79,70 % | 172,2 FPS |
| **CNN ResNet50** | **70,49 %** | **89,27 %** | 122,6 FPS |

Le CNN ResNet50 offre le meilleur compromis ; l'approche classique (Random Forest) est très rapide mais peu discriminante sur 101 classes.

### Limites
- Déséquilibre des classes → faible performance sur les classes rares (piste : augmentation de données ciblée)
- Modèles entraînés sur Kaggle (GPU Tesla T4)

## Structure du dépôt

```
├── notebooks/
│   ├── 01_yolov8_detection.ipynb                    # Détection YOLOv8 (preprocessing → entraînement → évaluation)
│   └── 02_detection_et_classification_complet.ipynb # Détection + Random Forest, ResNet50, EfficientNetV2
├── docs/
│   └── presentation_projet.pdf                      # Présentation du projet
├── requirements.txt
└── README.md
```

## Reproduire

Les notebooks sont écrits pour **Kaggle** (chemins `/kaggle/input/...` et `/kaggle/working`).

1. Créer un notebook Kaggle avec GPU et ajouter le dataset ci-dessus
2. Importer un des notebooks et exécuter les cellules dans l'ordre

Pour une exécution locale, adapter `BASE_PATH` et `WORK_PATH` puis `pip install -r requirements.txt`.

> Les poids (`best.pt`, `best_cnn.pth`, …) ne sont pas versionnés (trop lourds) ; ils sont régénérés par l'entraînement.
