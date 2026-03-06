"""
==============================================================================
 G02 — IMDb + BERT-base | Problématique P02 : Régularisation et Généralisation
==============================================================================
Module : visualization.py
Rôle   : Toutes les visualisations du projet :
          1. Heatmap des performances (weight_decay × dropout)
          2. Courbes de convergence (train vs val)
          3. Analyse de l'écart train/val (overfitting gap)
          4. Loss landscape 1D (méthode simplifiée CPU)
          5. Métriques de platitude (Sharpness)

Auteur : AZONFACK DOLVIANE MYRIAM, FONKOUA GANKE VOLTAIRE, OGNIMBA SADRI (Groupe Projet G02)
Date   : Mars 2026
"""

import os
import json
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")   # backend non-interactif (pas d'affichage X11)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# ── Palette cohérente pour tout le projet ─────────────────────
PALETTE = {
    "train":  "#2196F3",   # bleu
    "val":    "#F44336",   # rouge
    "gap":    "#FF9800",   # orange
    "flat":   "#4CAF50",   # vert (minima plats)
    "sharp":  "#9C27B0",   # violet (minima pointus)
}


# ──────────────────────────────────────────────────────────────
#  1. Heatmap des performances
# ──────────────────────────────────────────────────────────────
def plot_performance_heatmap(results_csv: str = "results/optuna_results.csv") -> None:
    """
    Génère une heatmap 2D (weight_decay × dropout → val_F1).

    Permet de visualiser d'un coup d'œil quelle combinaison
    de régularisation maximise la généralisation.
    """
    df = pd.read_csv(results_csv)

    pivot = df.pivot_table(
        index   = "dropout",
        columns = "weight_decay",
        values  = "val_f1",
        aggfunc = "max",
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    sns.heatmap(
        pivot,
        annot      = True,
        fmt        = ".4f",
        cmap       = "YlOrRd",
        linewidths = 0.5,
        ax         = ax,
        cbar_kws   = {"label": "F1-score (validation)"},
    )

    ax.set_title(
        "G02 — P02 : Heatmap des performances\n"
        "(BERT-base, IMDb) — Grid Search Optuna",
        fontsize=14, fontweight="bold", pad=14,
    )
    ax.set_xlabel("Weight Decay",  fontsize=12)
    ax.set_ylabel("Dropout Rate",  fontsize=12)

    # Annotations des colonnes en notation scientifique
    ax.set_xticklabels(
        [f"{float(x.get_text()):.0e}" for x in ax.get_xticklabels()],
        rotation=0,
    )

    plt.tight_layout()
    path = f"{FIGURES_DIR}/heatmap_performance.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap sauvegardée → {path}")


# ──────────────────────────────────────────────────────────────
#  2. Courbes de convergence
# ──────────────────────────────────────────────────────────────
def plot_convergence_curves(details_json: str = "results/optuna_details.json") -> None:
    """
    Trace les courbes loss / accuracy train vs validation pour
    les 4 trials représentatifs (meilleur, pire, 2 intermédiaires).

    Ces courbes révèlent si le modèle overfite (train >> val)
    ou sous-apprend (train ≈ val mais bas).
    """
    with open(details_json) as f:
        details = json.load(f)

    # Trie par val_f1 et sélectionne 4 trials représentatifs
    sorted_trials = sorted(details, key=lambda t: t["val_f1"], reverse=True)
    selected = [
        sorted_trials[0],                            # meilleur
        sorted_trials[len(sorted_trials)//3],        # 1er tiers
        sorted_trials[2*len(sorted_trials)//3],      # 2e tiers
        sorted_trials[-1],                           # pire
    ]
    labels_map = ["🏆 Meilleur", "2e quartile", "3e quartile", "⚠️ Pire"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(
        "G02 — P02 : Courbes de convergence par trial\n"
        "(BERT-base, IMDb, Grid Search Optuna)",
        fontsize=14, fontweight="bold",
    )

    for ax, trial, label in zip(axes.flat, selected, labels_map):
        h   = trial["history"]
        wd  = trial["params"]["weight_decay"]
        dp  = trial["params"]["dropout"]
        eps = h["epoch"]

        ax.plot(eps, h["train_acc"], color=PALETTE["train"],
                marker="o", lw=2, label="Accuracy train")
        ax.plot(eps, h["val_acc"],   color=PALETTE["val"],
                marker="s", lw=2, linestyle="--", label="Accuracy val")
        ax.fill_between(
            eps, h["train_acc"], h["val_acc"],
            alpha=0.15, color=PALETTE["gap"], label="Gap (overfitting)"
        )

        ax.set_title(
            f"{label}\nwd={wd:.0e} | dropout={dp} | val_F1={trial['val_f1']:.4f}",
            fontsize=11,
        )
        ax.set_xlabel("Époque")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0.4, 1.05)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    plt.tight_layout()
    path = f"{FIGURES_DIR}/convergence_curves.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📈 Courbes de convergence → {path}")


# ──────────────────────────────────────────────────────────────
#  3. Analyse de l'écart train/val (overfitting gap)
# ──────────────────────────────────────────────────────────────
def plot_overfitting_gap(details_json: str = "results/optuna_details.json") -> None:
    """
    Visualise comment weight_decay et dropout réduisent
    l'écart entre accuracy d'entraînement et de validation.

    Un gap faible signifie que le modèle généralise bien
    (objectif central de la problématique P02).
    """
    with open(details_json) as f:
        details = json.load(f)

    rows = []
    for t in details:
        h = t["history"]
        # Gap de la dernière époque
        final_gap = h["gap"][-1] if h["gap"] else None
        rows.append({
            "weight_decay": t["params"]["weight_decay"],
            "dropout":      t["params"]["dropout"],
            "final_gap":    final_gap,
            "val_f1":       t["val_f1"],
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "G02 — P02 : Analyse de l'overfitting gap (acc_train − acc_val)\n"
        "Un gap faible indique une meilleure généralisation",
        fontsize=13, fontweight="bold",
    )

    # ── Sous-figure 1 : Gap vs weight_decay (couleur = dropout) ──
    ax = axes[0]
    for dp in sorted(df["dropout"].unique()):
        sub = df[df["dropout"] == dp].sort_values("weight_decay")
        ax.plot(
            sub["weight_decay"], sub["final_gap"],
            marker="o", lw=2, label=f"dropout={dp}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Weight Decay (log scale)", fontsize=12)
    ax.set_ylabel("Gap final (train_acc − val_acc)", fontsize=12)
    ax.set_title("Influence du weight decay sur l'overfitting")
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Sous-figure 2 : scatter gap vs val_F1 ────────────────────
    ax = axes[1]
    sc = ax.scatter(
        df["final_gap"], df["val_f1"],
        c     = np.log10(df["weight_decay"].astype(float)),
        cmap  = "plasma",
        s     = 120,
        alpha = 0.85,
        edgecolors = "k",
        linewidths = 0.5,
    )
    fig.colorbar(sc, ax=ax, label="log₁₀(weight_decay)")
    ax.set_xlabel("Gap final (overfitting)", fontsize=12)
    ax.set_ylabel("F1-score (validation)",   fontsize=12)
    ax.set_title("Relation gap ↔ généralisation")
    ax.grid(True, alpha=0.3)

    # Annotation du meilleur point
    best_row = df.loc[df["val_f1"].idxmax()]
    ax.annotate(
        f"Meilleur\nwd={best_row['weight_decay']:.0e}\ndp={best_row['dropout']}",
        xy     = (best_row["final_gap"], best_row["val_f1"]),
        xytext = (best_row["final_gap"] + 0.01, best_row["val_f1"] - 0.02),
        arrowprops = dict(arrowstyle="->", color="black"),
        fontsize = 9,
    )

    plt.tight_layout()
    path = f"{FIGURES_DIR}/overfitting_gap.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📉 Analyse overfitting gap → {path}")


# ──────────────────────────────────────────────────────────────
#  4. Loss Landscape 1D (méthode §6.1 du sujet)
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def _evaluate_on_subset(model, dataset, device, n_samples: int = 50) -> float:
    """Évalue la loss sur un mini-sous-ensemble sans gradient."""
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    total_loss, steps = 0.0, 0
    for batch in loader:
        if steps * 16 >= n_samples:
            break
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        total_loss += out.loss.item()
        steps += 1
    return total_loss / max(steps, 1)


def compute_loss_landscape_1d(
    model,
    dataset,
    device,
    n_points: int = 10,
    epsilon:  float = 0.05,
) -> tuple:
    """
    Calcule le loss landscape 1D par perturbation aléatoire normalisée.
    Implémentation directe de la §6.1 du sujet (version CPU légère).

    Algorithme :
      1. Sauvegarder les paramètres θ₀
      2. Choisir une direction aléatoire d (normalisée)
      3. Pour α ∈ [-ε, +ε] : évaluer L(θ₀ + α·d)
      4. Restaurer θ₀

    Un minimum PLAT  → la loss varie peu autour de θ₀
    Un minimum POINTU → la loss monte fortement autour de θ₀

    Retourne
    --------
    (alphas: ndarray, losses: list)
    """
    model.eval()
    original_params = [p.clone().detach() for p in model.parameters()]

    # Direction aléatoire normalisée (même convention que §6.1)
    direction = [torch.randn_like(p) for p in model.parameters()]
    total_norm = sum(torch.norm(d).item() for d in direction)
    direction  = [d / total_norm for d in direction]

    alphas = np.linspace(-epsilon, epsilon, n_points)
    losses = []

    for alpha in alphas:
        # Perturbation : θ = θ₀ + α·d
        for p, p0, d in zip(model.parameters(), original_params, direction):
            p.data = p0 + alpha * d

        loss_val = _evaluate_on_subset(model, dataset, device, n_samples=50)
        losses.append(loss_val)

    # Restauration des paramètres originaux
    for p, p0 in zip(model.parameters(), original_params):
        p.data = p0.clone()

    return alphas, losses


def compute_sharpness(alphas: np.ndarray, losses: list) -> float:
    """
    Calcule la métrique de Sharpness (§6.3 du sujet) :

        Sharpness = (1/N) Σ |L(θ + ε·dᵢ) − L(θ)|

    Un Sharpness élevé → minimum pointu → mauvaise généralisation
    Un Sharpness faible → minimum plat  → bonne généralisation

    Paramètres
    ----------
    alphas : vecteur des perturbations
    losses : liste des losses correspondantes

    Retourne
    --------
    float : valeur de la sharpness
    """
    # Trouver la loss au centre (α ≈ 0)
    center_idx = len(alphas) // 2
    center_loss = losses[center_idx]

    # Formule §6.3
    sharpness = np.mean([abs(l - center_loss) for l in losses])
    return sharpness


def plot_loss_landscape(
    models_dict: dict,
    val_dataset,
    device,
    n_points: int = 10,
    epsilon:  float = 0.05,
) -> None:
    """
    Compare le loss landscape de plusieurs configurations
    (différents couples weight_decay / dropout).

    Paramètres
    ----------
    models_dict : dict{label: model}  (ex. {'wd=1e-5 dp=0.0': model_A, ...})
    val_dataset : IMDbDataset (validation)
    device      : torch.device
    n_points    : nombre de points sur l'axe alpha
    epsilon     : amplitude de la perturbation
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "G02 — P02 : Loss Landscape 1D\n"
        "Comparaison minima plats vs pointus selon la régularisation",
        fontsize=13, fontweight="bold",
    )

    colors     = plt.cm.tab10(np.linspace(0, 0.8, len(models_dict)))
    sharpness_data = {}

    ax_landscape = axes[0]
    ax_sharpness = axes[1]

    for (label, model), color in zip(models_dict.items(), colors):
        print(f"   🏔️  Loss landscape pour : {label}")
        alphas, losses = compute_loss_landscape_1d(
            model, val_dataset, device, n_points=n_points, epsilon=epsilon
        )
        sharp = compute_sharpness(alphas, losses)
        sharpness_data[label] = sharp

        ax_landscape.plot(
            alphas, losses,
            label  = f"{label}\n(Sharp={sharp:.4f})",
            color  = color,
            lw     = 2.2,
            marker = "o",
            ms     = 5,
        )

    # ── Sous-figure 1 : Loss landscape ───────────────────────
    ax_landscape.axvline(0, color="gray", lw=0.8, linestyle=":")
    ax_landscape.set_xlabel("Direction de perturbation α", fontsize=12)
    ax_landscape.set_ylabel("Loss",                        fontsize=12)
    ax_landscape.set_title("Loss landscape 1D")
    ax_landscape.legend(fontsize=9)
    ax_landscape.grid(True, alpha=0.3)

    # ── Sous-figure 2 : Barplot Sharpness ────────────────────
    labels_bar = list(sharpness_data.keys())
    values_bar = list(sharpness_data.values())
    bar_colors = [PALETTE["flat"] if v == min(values_bar) else PALETTE["sharp"]
                  for v in values_bar]

    bars = ax_sharpness.barh(labels_bar, values_bar, color=bar_colors, edgecolor="k", lw=0.5)
    ax_sharpness.set_xlabel("Sharpness (↓ meilleur)", fontsize=12)
    ax_sharpness.set_title("Métrique de platitude des minima\n(§6.3 du sujet)")
    ax_sharpness.grid(True, alpha=0.3, axis="x")

    # Annotation valeurs
    for bar, val in zip(bars, values_bar):
        ax_sharpness.text(
            val + 0.0005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9,
        )

    plt.tight_layout()
    path = f"{FIGURES_DIR}/loss_landscape.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🏔️  Loss landscape sauvegardé → {path}")

    return sharpness_data


# ──────────────────────────────────────────────────────────────
#  5. Matrice de confusion du meilleur modèle
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def plot_confusion_matrix(model, test_dataset, device) -> None:
    """
    Affiche la matrice de confusion sur le jeu de test,
    avec les vraies étiquettes (Négatif / Positif).
    """
    model.eval()
    loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    all_preds, all_labels = [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(out.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy())

    cm   = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Négatif", "Positif"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        "G02 — Matrice de confusion (test)\nMeilleur modèle BERT-base",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = f"{FIGURES_DIR}/confusion_matrix.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  🔲 Matrice de confusion → {path}")


# ──────────────────────────────────────────────────────────────
#  6. Tableau de synthèse des résultats
# ──────────────────────────────────────────────────────────────
def plot_results_table(results_csv: str = "results/optuna_results.csv") -> None:
    """
    Génère une figure-tableau des 12 combinaisons triées par val_F1.
    Utile pour le rapport (section Résultats - Optimisation).
    """
    df = pd.read_csv(results_csv).sort_values("val_f1", ascending=False)
    df["weight_decay"] = df["weight_decay"].apply(lambda x: f"{float(x):.0e}")
    df.insert(0, "Rang", range(1, len(df) + 1))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    table = ax.table(
        cellText  = df.values,
        colLabels = df.columns,
        loc       = "center",
        cellLoc   = "center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    # Coloration de la meilleure ligne
    for j in range(len(df.columns)):
        table[1, j].set_facecolor("#C8E6C9")   # vert clair = meilleur

    ax.set_title(
        "G02 — P02 : Tableau de synthèse des 12 combinaisons Optuna\n"
        "(Trié par F1-score validation décroissant)",
        fontsize=12, fontweight="bold", pad=20,
    )

    plt.tight_layout()
    path = f"{FIGURES_DIR}/results_table.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  📋 Tableau de synthèse → {path}")


# ──────────────────────────────────────────────────────────────
#  Génération complète (appelée depuis main.py)
# ──────────────────────────────────────────────────────────────
def generate_all_figures(
    models_best_worst: dict = None,
    val_dataset=None,
    test_dataset=None,
    device=None,
) -> None:
    """
    Orchestre la génération de toutes les figures du projet G02.

    Si les résultats Optuna existent déjà (CSV + JSON),
    les figures 1 à 3 et 6 sont générées automatiquement.
    Les figures 4 et 5 nécessitent les modèles en mémoire.
    """
    print("\n🎨 Génération des figures...")

    if os.path.exists("results/optuna_results.csv"):
        plot_performance_heatmap()
        plot_results_table()
    if os.path.exists("results/optuna_details.json"):
        plot_convergence_curves()
        plot_overfitting_gap()

    if models_best_worst and val_dataset and device:
        sharpness = plot_loss_landscape(
            models_best_worst, val_dataset, device
        )

    if models_best_worst and test_dataset and device:
        best_model = list(models_best_worst.values())[0]
        plot_confusion_matrix(best_model, test_dataset, device)

    print(f"\n✅ Toutes les figures dans '{FIGURES_DIR}/'")
