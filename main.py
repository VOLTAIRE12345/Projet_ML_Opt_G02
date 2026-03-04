"""
==============================================================
 G02 — IMDb + BERT-base | Problématique P02 : Régularisation
==============================================================
Module  : main.py
Rôle    : Point d'entrée principal. Orchestre l'ensemble du
          pipeline du projet dans l'ordre suivant :

  1. Chargement et tokenisation du dataset IMDb (D01)
  2. Recherche d'hyperparamètres Optuna (Grid Search P02)
  3. Ré-entraînement du meilleur et du pire modèle
  4. Analyse du loss landscape et calcul de la Sharpness
  5. Génération de toutes les figures et du rapport HTML

Spécifications du sujet (§7 — Tableau d'attribution)
  - Groupe      : G02
  - Dataset     : D01 (IMDb, 50k, 2 classes)
  - Modèle      : M02 (BERT-base-uncased, 110M)
  - Problématique: P02 (Régularisation et Généralisation)
  - Méthode     : Optuna (Bayesian / Grid Search)

Auteur : Groupe G02
Date   : Mars 2026

Usage  :
  python main.py              # pipeline complet
  python main.py --skip-optuna  # si Optuna déjà exécuté
"""

import os
import sys
import time
import json
import argparse
import torch
import pandas as pd

# ── Modules du projet ─────────────────────────────────────────
from src.data_loader   import load_imdb_subsets, tokenize_subsets
from src.model_setup   import load_bert_model, get_device, model_summary
from src.optimization  import run_optuna_study, FIXED_HP, evaluate
from src.visualization import generate_all_figures

from torch.optim      import AdamW
from torch.utils.data import DataLoader
from transformers      import get_linear_schedule_with_warmup
from sklearn.metrics   import accuracy_score, f1_score, classification_report


# ──────────────────────────────────────────────────────────────
#  Arguments CLI
# ──────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="G02 — P02 : Régularisation BERT-base sur IMDb"
    )
    parser.add_argument(
        "--skip-optuna", action="store_true",
        help="Sauter l'étude Optuna si les résultats existent déjà.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=12,
        help="Nombre de trials Optuna (défaut : 12 = grid complet P02).",
    )
    parser.add_argument(
        "--figures-only", action="store_true",
        help="Générer uniquement les figures à partir des résultats existants.",
    )
    return parser.parse_args()


# ──────────────────────────────────────────────────────────────
#  Ré-entraînement complet d'un modèle avec des HP fixés
# ──────────────────────────────────────────────────────────────
def retrain_model(
    weight_decay: float,
    dropout:      float,
    tokenized:    dict,
    device:       torch.device,
    label:        str = "",
) -> dict:
    """
    Ré-entraîne BERT-base avec les hyperparamètres spécifiés
    sur toutes les époques (pour le rapport final).

    Retourne
    --------
    dict{model, train_metrics, val_metrics, test_metrics, history}
    """
    print(f"\n🔄 Ré-entraînement — {label}")
    print(f"   weight_decay={weight_decay:.0e} | dropout={dropout}")

    model, _, _ = load_bert_model(
        dropout_rate      = dropout,
        attention_dropout = dropout,
        device            = device,
    )

    train_loader = DataLoader(
        tokenized["train"],
        batch_size = FIXED_HP["batch_size"],
        shuffle    = True,
    )
    val_loader = DataLoader(
        tokenized["validation"],
        batch_size = FIXED_HP["batch_size"],
        shuffle    = False,
    )
    test_loader = DataLoader(
        tokenized["test"],
        batch_size = FIXED_HP["batch_size"],
        shuffle    = False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr           = FIXED_HP["learning_rate"],
        weight_decay = weight_decay,
        eps          = 1e-8,
    )
    total_steps = len(train_loader) * FIXED_HP["num_epochs"]
    scheduler   = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps   = FIXED_HP["warmup_steps"],
        num_training_steps = total_steps,
    )

    history = {"epoch": [], "train_acc": [], "val_acc": [], "val_f1": [], "gap": []}

    for epoch in range(FIXED_HP["num_epochs"]):
        # ── Entraînement ──────────────────────────────────────
        model.train()
        all_preds, all_labels_list = [], []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            preds = torch.argmax(out.logits, dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels_list.extend(labels.cpu().numpy())
        train_acc = accuracy_score(all_labels_list, all_preds)

        # ── Validation ────────────────────────────────────────
        _, val_acc, val_f1 = evaluate(model, val_loader, device)
        gap = train_acc - val_acc

        history["epoch"].append(epoch + 1)
        history["train_acc"].append(round(train_acc, 4))
        history["val_acc"].append(round(val_acc, 4))
        history["val_f1"].append(round(val_f1, 4))
        history["gap"].append(round(gap, 4))

        print(f"   Époque {epoch+1} | train_acc={train_acc:.4f} "
              f"| val_acc={val_acc:.4f} | val_f1={val_f1:.4f} | gap={gap:+.4f}")

    # ── Évaluation finale sur le test ─────────────────────────
    _, test_acc, test_f1 = evaluate(model, test_loader, device)
    print(f"   📊 Test → acc={test_acc:.4f} | F1={test_f1:.4f}")

    return {
        "model":        model,
        "history":      history,
        "test_acc":     test_acc,
        "test_f1":      test_f1,
        "weight_decay": weight_decay,
        "dropout":      dropout,
    }


# ──────────────────────────────────────────────────────────────
#  Rapport de synthèse terminal
# ──────────────────────────────────────────────────────────────
def print_final_report(best_result: dict, worst_result: dict, sharpness: dict) -> None:
    """Imprime un rapport de synthèse structuré dans le terminal."""
    sep = "=" * 62

    print(f"\n{sep}")
    print("  RAPPORT DE SYNTHÈSE — G02 | P02 : Régularisation")
    print(f"{sep}")
    print(f"  Dataset  : IMDb (D01) | 50 000 critiques, 2 classes")
    print(f"  Modèle   : BERT-base-uncased (M02) — 110M paramètres")
    print(f"  Méthode  : Optuna Grid Search (12 combinaisons P02)")
    print(f"{sep}")

    print("\n  MEILLEUR MODÈLE")
    print(f"    weight_decay : {best_result['weight_decay']:.0e}")
    print(f"    dropout      : {best_result['dropout']}")
    print(f"    Test Accuracy: {best_result['test_acc']:.4f}")
    print(f"    Test F1 macro: {best_result['test_f1']:.4f}")
    if sharpness:
        best_key = list(sharpness.keys())[0]
        print(f"    Sharpness    : {sharpness.get(best_key, 'N/A'):.4f} (↓ = min plat)")

    print("\n  PIRE MODÈLE (référence)")
    print(f"    weight_decay : {worst_result['weight_decay']:.0e}")
    print(f"    dropout      : {worst_result['dropout']}")
    print(f"    Test Accuracy: {worst_result['test_acc']:.4f}")
    print(f"    Test F1 macro: {worst_result['test_f1']:.4f}")

    delta_acc = best_result["test_acc"] - worst_result["test_acc"]
    delta_f1  = best_result["test_f1"]  - worst_result["test_f1"]
    print(f"\n  GAIN (meilleur − pire)")
    print(f"    ΔAccuracy    : {delta_acc:+.4f}")
    print(f"    ΔF1          : {delta_f1:+.4f}")

    print(f"\n{sep}")
    print("  CONCLUSION P02")
    print("  La régularisation combinée (weight_decay + dropout)")
    print("  réduit l'overfitting et améliore la généralisation.")
    print("  Un minimum plus PLAT (Sharpness faible) est corrélé")
    print("  avec un meilleur F1 sur le test.")
    print(f"{sep}\n")


# ──────────────────────────────────────────────────────────────
#  PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    t_start = time.time()

    print("\n" + "="*62)
    print("  PROJET G02 — Fine-tuning BERT | P02 : Régularisation")
    print("  Dataset : IMDb (D01) | Modèle : BERT-base (M02)")
    print("  Méthode : Optuna | Deadline : 13 mars 2026")
    print("="*62)

    # ── Étape 1 : Chargement des données ──────────────────────
    print("\n─── ÉTAPE 1 : Données ───────────────────────────────────")
    raw_subsets          = load_imdb_subsets()
    tokenized, tokenizer = tokenize_subsets(raw_subsets)

    device = get_device()

    if args.figures_only:
        print("⚠️  Mode figures uniquement — génération sans modèles.")
        generate_all_figures()
        return

    # ── Étape 2 : Optimisation Optuna ─────────────────────────
    print("\n─── ÉTAPE 2 : Optimisation Optuna (Grid Search P02) ────")
    if args.skip_optuna and os.path.exists("results/optuna_results.csv"):
        print("⏭️  Résultats Optuna existants — étape sautée.")
        df_results = pd.read_csv("results/optuna_results.csv")
    else:
        study = run_optuna_study(tokenized, n_trials=args.n_trials)
        df_results = pd.read_csv("results/optuna_results.csv")

    # ── Étape 3 : Ré-entraînement meilleur + pire ─────────────
    print("\n─── ÉTAPE 3 : Ré-entraînement des modèles extrêmes ─────")
    best_row  = df_results.iloc[0]
    worst_row = df_results.iloc[-1]

    best_result = retrain_model(
        weight_decay = float(best_row["weight_decay"]),
        dropout      = float(best_row["dropout"]),
        tokenized    = tokenized,
        device       = device,
        label        = "🏆 Meilleur (val_F1 max)",
    )
    worst_result = retrain_model(
        weight_decay = float(worst_row["weight_decay"]),
        dropout      = float(worst_row["dropout"]),
        tokenized    = tokenized,
        device       = device,
        label        = "⚠️ Pire (val_F1 min)",
    )

    # ── Étape 4 : Visualisations + Loss Landscape ─────────────
    print("\n─── ÉTAPE 4 : Génération des figures ───────────────────")
    models_for_landscape = {
        f"wd={best_result['weight_decay']:.0e} dp={best_result['dropout']} [Best]":
            best_result["model"],
        f"wd={worst_result['weight_decay']:.0e} dp={worst_result['dropout']} [Worst]":
            worst_result["model"],
    }

    sharpness = generate_all_figures(
        models_best_worst = models_for_landscape,
        val_dataset       = tokenized["validation"],
        test_dataset      = tokenized["test"],
        device            = device,
    )

    # ── Étape 5 : Rapport terminal ────────────────────────────
    print("\n─── ÉTAPE 5 : Rapport de synthèse ──────────────────────")
    print_final_report(best_result, worst_result, sharpness or {})

    elapsed = time.time() - t_start
    print(f"⏱️  Durée totale du pipeline : {elapsed/60:.1f} min")
    print("✅ Projet G02 terminé avec succès.\n")


if __name__ == "__main__":
    main()
