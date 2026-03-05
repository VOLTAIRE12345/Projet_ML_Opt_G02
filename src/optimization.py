"""
==============================================================
 G02 — IMDb + BERT-base | Problématique P02 : Régularisation
==============================================================
Module : optimization.py
Rôle   : Recherche d'hyperparamètres par Optuna (Bayésien TPE)
          avec protocole strict P02 :
            • Grid search : weight_decay ∈ {1e-5, 1e-4, 1e-3, 1e-2}
            •              dropout       ∈ {0.0, 0.1, 0.3}
            • Mesure de l'écart train/validation (overfitting gap)
            • Early stopping si la valeur cible n'est pas atteinte

Auteur : Groupe G02
Date   : Mars 2026
"""

import os
import json
import time
import optuna
import torch
import numpy as np
import pandas as pd
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from src.data_loader  import load_imdb_subsets, tokenize_subsets
from src.model_setup  import load_bert_model, get_device

# ── Silence des logs trop verbeux d'Optuna ────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ──────────────────────────────────────────────────────────────
#  Hyperparamètres FIXES (toujours identiques entre les essais)
#  Conformément au protocole P02 : "fixer tous les autres HP"
# ──────────────────────────────────────────────────────────────
FIXED_HP = {
    "learning_rate": 2e-5,     # LR standard pour le fine-tuning BERT
    "batch_size":    8,         # petit batch → adapté CPU
    "num_epochs":    3,         # 3 époques suffisantes pour détecter la généralisation
    "warmup_steps":  10,       # préchauffage du scheduler linéaire
    "max_steps":     9999,       # plafond de sécurité CPU
}

# ── Espaces de recherche P02 (grid exhaustif) ─────────────────
WEIGHT_DECAY_GRID = [1e-5, 1e-4, 1e-3, 1e-2]   # 4 valeurs
DROPOUT_GRID      = [0.0,  0.1,  0.3]            # 3 valeurs
# → 4 × 3 = 12 combinaisons à évaluer

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ──────────────────────────────────────────────────────────────
#  Entraînement d'une époque
# ──────────────────────────────────────────────────────────────
def train_one_epoch(model, loader, optimizer, scheduler, device) -> tuple:
    """
    Effectue une époque d'entraînement complète.

    Retourne
    --------
    (loss_moyenne, accuracy_train)
    """
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in tqdm(loader, desc="  Train", leave=False, ncols=80):
        # Transfert sur le device
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        # Forward pass (BERT calcule la loss en interne si labels fournis)
        optimizer.zero_grad()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        logits = outputs.logits

        # Backward pass + mise à jour
        loss.backward()
        # Gradient clipping → stabilité lors du fine-tuning de BERT
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        preds = torch.argmax(logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


# ──────────────────────────────────────────────────────────────
#  Évaluation
# ──────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, device) -> tuple:
    """
    Évalue le modèle sur un DataLoader.

    Retourne
    --------
    (loss_moyenne, accuracy, f1_macro)
    """
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        input_ids      = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels         = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        total_loss += outputs.loss.item()
        preds = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc      = accuracy_score(all_labels, all_preds)
    f1       = f1_score(all_labels, all_preds, average="macro")
    return avg_loss, acc, f1


# ──────────────────────────────────────────────────────────────
#  Fonction objectif Optuna (protocole P02)
# ──────────────────────────────────────────────────────────────
def make_objective(tokenized_datasets, device):
    """
    Fabrique la fonction objectif Optuna.

    Le protocole P02 demande :
      1. Grid search sur weight_decay et dropout
      2. Mesure de l'écart train/val (overfitting gap)
      3. Minimisation de l'écart → favorise la généralisation

    Valeur optimisée : F1-score de validation (métrique principale)

    Paramètres
    ----------
    tokenized_datasets : dict{split → IMDbDataset}
    device             : torch.device

    Retourne
    --------
    objective(trial) → float  (val_f1)
    """
    # DataLoaders créés une seule fois (hors de la boucle Optuna)
    train_loader = DataLoader(
        tokenized_datasets["train"],
        batch_size=FIXED_HP["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        tokenized_datasets["validation"],
        batch_size=FIXED_HP["batch_size"],
        shuffle=False,
    )

    def objective(trial: optuna.Trial) -> float:
        # ── Suggestion des hyperparamètres P02 ────────────────
        # Grid exhaustif : Optuna parcourt toutes les combinaisons
        weight_decay = trial.suggest_categorical("weight_decay", WEIGHT_DECAY_GRID)
        dropout      = trial.suggest_categorical("dropout",      DROPOUT_GRID)

        trial.set_user_attr("weight_decay", weight_decay)
        trial.set_user_attr("dropout",      dropout)

        print(f"\n🔬 Trial #{trial.number:02d} | "
              f"weight_decay={weight_decay:.0e} | dropout={dropout}")

        # ── Chargement du modèle (fresh pour chaque trial) ────
        model, _, _ = load_bert_model(
            dropout_rate      = dropout,
            attention_dropout = dropout,  # même taux pour les deux
            device            = device,
        )

        # ── Optimiseur : AdamW avec weight decay du trial ─────
        optimizer = AdamW(
            model.parameters(),
            lr           = FIXED_HP["learning_rate"],
            weight_decay = weight_decay,
            eps          = 1e-8,           # stabilité numérique
        )

        # ── Scheduler linéaire avec warmup ────────────────────
        total_steps  = min(FIXED_HP["max_steps"],
                           len(train_loader) * FIXED_HP["num_epochs"])
        scheduler    = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps   = FIXED_HP["warmup_steps"],
            num_training_steps = total_steps,
        )

        # ── Boucle d'entraînement ─────────────────────────────
        history = {"epoch": [], "train_loss": [], "train_acc": [],
                   "val_loss": [], "val_acc": [], "val_f1": [], "gap": []}

        best_val_f1   = 0.0
        steps_done    = 0
        t0            = time.time()

        for epoch in range(FIXED_HP["num_epochs"]):
            if steps_done >= FIXED_HP["max_steps"]:
                break

            train_loss, train_acc = train_one_epoch(
                model, train_loader, optimizer, scheduler, device
            )
            val_loss, val_acc, val_f1 = evaluate(model, val_loader, device)

            # Écart train/val → indicateur d'overfitting
            gap = train_acc - val_acc

            history["epoch"].append(epoch + 1)
            history["train_loss"].append(round(train_loss, 4))
            history["train_acc"].append(round(train_acc,   4))
            history["val_loss"].append(round(val_loss,     4))
            history["val_acc"].append(round(val_acc,       4))
            history["val_f1"].append(round(val_f1,         4))
            history["gap"].append(round(gap,               4))

            print(f"   Époque {epoch+1}/{FIXED_HP['num_epochs']} | "
                  f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_acc={val_acc:.4f} val_f1={val_f1:.4f} | gap={gap:+.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1

            # ── Pruning Optuna : coupe les trials peu prometteurs ──
            trial.report(val_f1, step=epoch)
            if trial.should_prune():
                print("   ✂️  Trial élagué par Optuna.")
                raise optuna.TrialPruned()

            steps_done += len(train_loader)

        elapsed = time.time() - t0

        # Sauvegarde de l'historique du trial
        trial.set_user_attr("history",   history)
        trial.set_user_attr("elapsed_s", round(elapsed, 1))
        trial.set_user_attr("best_val_f1", best_val_f1)

        print(f"   ✅ best_val_f1={best_val_f1:.4f}  "
              f"(durée: {elapsed:.0f}s)")

        return best_val_f1

    return objective


# ──────────────────────────────────────────────────────────────
#  Lancement de l'étude Optuna
# ──────────────────────────────────────────────────────────────
def run_optuna_study(tokenized_datasets: dict, n_trials: int = 12) -> optuna.Study:
    """
    Lance l'étude Optuna Grid Search sur les 12 combinaisons P02.

    Protocole P02 :
      - 4 valeurs de weight_decay × 3 valeurs de dropout = 12 trials
      - Objectif : maximiser le F1-score de validation
      - Pruner : MedianPruner pour couper les trials non compétitifs
      - Sampler : GridSampler → exploration exhaustive garantie

    Paramètres
    ----------
    tokenized_datasets : dict{split → IMDbDataset}
    n_trials           : nombre d'essais (≥ 12 pour le grid complet)

    Retourne
    --------
    study : optuna.Study (contient tous les résultats)
    """
    device = get_device()

    # ── Grid exhaustif : toutes les combinaisons P02 ──────────
    search_space = {
        "weight_decay": WEIGHT_DECAY_GRID,
        "dropout":      DROPOUT_GRID,
    }
    sampler = optuna.samplers.GridSampler(search_space)
    pruner  = optuna.pruners.MedianPruner(n_startup_trials=4, n_warmup_steps=1)

    study = optuna.create_study(
        study_name = "G02_P02_regularisation",
        direction  = "maximize",    # maximiser le F1-score
        sampler    = sampler,
        pruner     = pruner,
    )

    objective = make_objective(tokenized_datasets, device)

    print("\n" + "="*60)
    print("  ÉTUDE OPTUNA — G02 | P02 : Régularisation et Généralisation")
    print(f"  Dataset : IMDb (D01) | Modèle : BERT-base (M02)")
    print(f"  Grid : {len(WEIGHT_DECAY_GRID)} weight_decay × "
          f"{len(DROPOUT_GRID)} dropout = 12 combinaisons")
    print("="*60)

    study.optimize(
        objective,
        n_trials  = n_trials,
        show_progress_bar = False,
    )

    # ── Sauvegarde des résultats ───────────────────────────────
    _save_results(study)

    # ── Rapport terminal ───────────────────────────────────────
    best = study.best_trial
    print("\n" + "="*60)
    print("  MEILLEUR TRIAL TROUVÉ")
    print(f"  weight_decay = {best.params['weight_decay']:.0e}")
    print(f"  dropout      = {best.params['dropout']}")
    print(f"  val_F1       = {best.value:.4f}")
    print("="*60)

    return study


# ──────────────────────────────────────────────────────────────
#  Sauvegarde des résultats
# ──────────────────────────────────────────────────────────────
def _save_results(study: optuna.Study) -> None:
    """Sérialise tous les trials Optuna en CSV et JSON."""
    rows = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        rows.append({
            "trial":        t.number,
            "weight_decay": t.params.get("weight_decay"),
            "dropout":      t.params.get("dropout"),
            "val_f1":       t.value,
            "best_val_f1":  t.user_attrs.get("best_val_f1"),
            "elapsed_s":    t.user_attrs.get("elapsed_s"),
        })

    df = pd.DataFrame(rows).sort_values("val_f1", ascending=False)
    df.to_csv(f"{RESULTS_DIR}/optuna_results.csv", index=False)

    # JSON détaillé (avec historique par époque)
    details = []
    for t in study.trials:
        if t.state != optuna.trial.TrialState.COMPLETE:
            continue
        details.append({
            "trial":        t.number,
            "params":       t.params,
            "val_f1":       t.value,
            "history":      t.user_attrs.get("history"),
            "elapsed_s":    t.user_attrs.get("elapsed_s"),
        })
    with open(f"{RESULTS_DIR}/optuna_details.json", "w") as f:
        json.dump(details, f, indent=2)

    print(f"\n💾 Résultats sauvegardés dans '{RESULTS_DIR}/'")
    print(df.to_string(index=False))


# ──────────────────────────────────────────────────────────────
#  Point d'entrée
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    raw_subsets        = load_imdb_subsets()
    tokenized, tokenizer = tokenize_subsets(raw_subsets)
    study              = run_optuna_study(tokenized, n_trials=12)
