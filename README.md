# 🤖 Projet G02 — Fine-tuning BERT | P02 : Régularisation et Généralisation

> **Cours** : Optimisation d'hyperparamètres et analyse du loss landscape  
> **Groupe** : G02  
> **Date limite** : 13 mars 2026  
> **Contact enseignant** : mbialaura12@gmail.com  

---

## 📋 Spécifications du sujet

| Paramètre | Valeur |
|---|---|
| **Dataset** | D01 — IMDb reviews (50k critiques, 2 classes pos/nég) |
| **Modèle** | M02 — BERT-base-uncased (110M paramètres) |
| **Problématique** | P02 — Régularisation et Généralisation |
| **Méthode d'optimisation** | Optuna (Grid Search bayésien) |
| **Métrique principale** | F1-score (macro) |

---

## ❓ Question de recherche (P02)

> **Comment le *weight decay* et le *dropout* affectent-ils la généralisation de BERT-base sur la classification de sentiments IMDb ?**

### Protocole expérimental (conforme §4.2 du sujet)

| Hyperparamètre | Valeurs testées |
|---|---|
| Weight Decay | {1e⁻⁵, 1e⁻⁴, 1e⁻³, 1e⁻²} |
| Dropout | {0.0, 0.1, 0.3} |
| **Total** | **4 × 3 = 12 combinaisons** |

Hyperparamètres **fixés** entre les trials (conditions équitables) :
- Learning rate : 2e-5
- Batch size : 8  
- Warmup steps : 100
- Optimiseur : AdamW

---

## 🗂️ Structure du projet

```
projet_G02/
├── main.py                  # ← Point d'entrée principal
├── requirements.txt         # Dépendances Python
├── README.md                # Ce fichier
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py       # Chargement IMDb + sous-échantillonnage CPU
│   ├── model_setup.py       # Initialisation BERT-base avec dropout configurable
│   ├── optimization.py      # Étude Optuna Grid Search P02
│   └── visualization.py     # Toutes les figures du projet
│
├── notebooks/
│   └── analysis.ipynb       # Analyse interactive des résultats
│
├── results/                 # Généré automatiquement
│   ├── optuna_results.csv   # Tableau des 12 combinaisons
│   └── optuna_details.json  # Historique détaillé par époque
│
└── figures/                 # Généré automatiquement
    ├── heatmap_performance.png
    ├── convergence_curves.png
    ├── overfitting_gap.png
    ├── loss_landscape.png
    ├── confusion_matrix.png
    └── results_table.png
```

---

## ⚙️ Installation

```bash
# 1. Cloner le dépôt
git clone <url_du_repo>
cd projet_G02

# 2. Installer les dépendances
pip install -r requirements.txt

# 3. (Optionnel) Vérifier PyTorch
python -c "import torch; print(torch.__version__)"
```

---

## 🚀 Exécution

### Pipeline complet (recommandé)

```bash
python main.py
```

Cela lance séquentiellement :
1. Chargement du dataset IMDb (sous-ensemble CPU)
2. Grid Search Optuna sur 12 combinaisons (weight_decay × dropout)
3. Ré-entraînement du meilleur et du pire modèle
4. Génération des figures (loss landscape, heatmap, courbes...)
5. Rapport de synthèse terminal

### Options CLI

```bash
# Sauter Optuna si déjà exécuté (résultats dans results/)
python main.py --skip-optuna

# Nombre de trials (12 = grid complet P02)
python main.py --n-trials 12

# Figures uniquement (à partir de résultats existants)
python main.py --figures-only
```

### Modules individuels

```bash
# Test du chargement des données
python -m src.data_loader

# Test du chargement du modèle
python -m src.model_setup

# Optimisation seule
python -m src.optimization
```

---

## 🧠 Adaptation aux contraintes CPU (§1.2 du sujet)

| Contrainte | Solution implémentée |
|---|---|
| Pas de GPU | float32, `torch.set_num_threads(4)` |
| RAM limitée | 800 exemples train, batch_size=8, max_length=128 |
| Temps contraint | max_steps=300, early stopping Optuna |
| Connexion limitée | Modèle téléchargeable une seule fois via HuggingFace cache |

---

## 📊 Résultats attendus

Les figures générées permettent d'analyser :

1. **`heatmap_performance.png`** — Quelle combinaison (wd, dropout) maximise le F1 ?
2. **`convergence_curves.png`** — Le modèle overfite-t-il au fil des époques ?
3. **`overfitting_gap.png`** — Comment wd et dropout réduisent-ils le gap train/val ?
4. **`loss_landscape.png`** — Le meilleur modèle converge-t-il vers un minimum plat ?
5. **`confusion_matrix.png`** — Performance finale sur le jeu de test
6. **`results_table.png`** — Tableau comparatif des 12 combinaisons

---

## 📐 Métriques clés

- **F1-score (macro)** — Métrique principale (équilibrée sur les 2 classes)
- **Overfitting Gap** = `accuracy_train − accuracy_val` (↓ = meilleure généralisation)
- **Sharpness** = `(1/N) Σ |L(θ + ε·dᵢ) − L(θ)|` (§6.3 du sujet, ↓ = minimum plat)

---

## 📦 Dépendances principales

```
torch>=2.0.0          # Deep learning
transformers>=4.35.0  # BERT-base, AutoTokenizer
datasets>=2.14.0      # IMDb (HuggingFace Hub)
optuna>=3.3.0         # Optimisation bayésienne
scikit-learn>=1.3.0   # Métriques (F1, accuracy, confusion matrix)
matplotlib>=3.7.0     # Visualisations
seaborn>=0.12.0       # Heatmaps
```

---

## 👥 Groupe G02

Dépôt soumis par mail à mbialaura12@gmail.com avant le **13 mars 2026**.  
Format : **Rapport PDF (8-10 pages) + lien GitHub**.
