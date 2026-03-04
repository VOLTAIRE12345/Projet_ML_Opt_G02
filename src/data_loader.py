"""
==============================================================
 G02 — IMDb + BERT-base | Problématique P02 : Régularisation
==============================================================
Module : data_loader.py
Rôle   : Chargement, nettoyage et sous-échantillonnage du
          dataset IMDb (D01) pour un entraînement CPU-friendly.

Auteur : Groupe G02
Date   : Mars 2026
"""

import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset


# ──────────────────────────────────────────────────────────────
#  Constantes globales du projet G02
# ──────────────────────────────────────────────────────────────
DATASET_NAME      = "imdb"               # D01 : IMDb reviews (50k, 2 classes)
MODEL_NAME        = "bert-base-uncased"  # M02 : BERT-base (110M paramètres)
NUM_LABELS        = 2                    # 0 = négatif, 1 = positif
MAX_SEQ_LENGTH    = 128                  # troncature → raisonnable pour CPU
RANDOM_SEED       = 42                   # reproductibilité garantie

# Taille des sous-ensembles (adaptation CPU, cf. §2.2 du sujet)
N_TRAIN_PER_CLASS = 400   # → 800 exemples d'entraînement
N_VAL_PER_CLASS   = 100   # → 200 exemples de validation
N_TEST_PER_CLASS  = 100   # → 200 exemples de test


# ──────────────────────────────────────────────────────────────
#  Dataset PyTorch personnalisé
# ──────────────────────────────────────────────────────────────
class IMDbDataset(Dataset):
    """
    Wrapper PyTorch Dataset pour les encodages BERT tokenisés.

    Attributs
    ---------
    encodings : dict{str → list}  (input_ids, attention_mask, token_type_ids)
    labels    : list[int]          (0 ou 1)
    """

    def __init__(self, encodings: dict, labels: list):
        self.encodings = encodings
        self.labels    = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        """Retourne un dictionnaire de tenseurs pour l'indice idx."""
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ──────────────────────────────────────────────────────────────
#  Fonctions utilitaires
# ──────────────────────────────────────────────────────────────
def create_balanced_subset(
    examples: list,
    labels: list,
    n_per_class: int,
    seed: int = RANDOM_SEED
) -> tuple:
    """
    Tire un sous-ensemble ÉQUILIBRÉ par classe.

    La balance est cruciale pour éviter que le modèle soit biaisé
    vers la classe majoritaire (biais de distribution).

    Paramètres
    ----------
    examples     : liste de textes bruts
    labels       : liste d'étiquettes entières
    n_per_class  : nombre d'exemples max par classe
    seed         : graine NumPy pour la reproductibilité

    Retourne
    --------
    (texts_sélectionnés : list, labels_sélectionnés : list)
    """
    rng = np.random.default_rng(seed)
    selected_texts, selected_labels = [], []

    for label in sorted(set(labels)):
        indices = [i for i, l in enumerate(labels) if l == label]
        n       = min(n_per_class, len(indices))
        chosen  = rng.choice(indices, size=n, replace=False)
        selected_texts.extend([examples[i] for i in chosen])
        selected_labels.extend([labels[i]  for i in chosen])

    return selected_texts, selected_labels


# ──────────────────────────────────────────────────────────────
#  Chargement principal
# ──────────────────────────────────────────────────────────────
def load_imdb_subsets() -> dict:
    """
    Charge le dataset IMDb depuis HuggingFace Hub et renvoie
    trois sous-ensembles équilibrés adaptés à l'entraînement CPU.

    Note : IMDb ne dispose pas de split 'validation' officiel.
    On échantillonne donc depuis le split 'train' (25 000 ex.)
    avec des graines différentes pour éviter les recouvrements.

    Retourne
    --------
    dict{
        'train'      : (list[str], list[int]),
        'validation' : (list[str], list[int]),
        'test'       : (list[str], list[int]),
    }
    """
    print("📦 Chargement du dataset IMDb (D01) depuis HuggingFace...")
    raw = load_dataset(DATASET_NAME)

    # ── Splits bruts ──────────────────────────────────────────
    train_texts_raw  = raw["train"]["text"]
    train_labels_raw = raw["train"]["label"]
    test_texts_raw   = raw["test"]["text"]
    test_labels_raw  = raw["test"]["label"]

    # ── Sous-ensembles équilibrés (graines distinctes) ────────
    train_texts, train_labels = create_balanced_subset(
        train_texts_raw, train_labels_raw, N_TRAIN_PER_CLASS, seed=RANDOM_SEED
    )
    val_texts, val_labels = create_balanced_subset(
        train_texts_raw, train_labels_raw, N_VAL_PER_CLASS,   seed=RANDOM_SEED + 1
    )
    test_texts, test_labels = create_balanced_subset(
        test_texts_raw, test_labels_raw,   N_TEST_PER_CLASS,  seed=RANDOM_SEED + 2
    )

    print(f"  ✅ Train      : {len(train_texts):>4} exemples  "
          f"({sum(l==0 for l in train_labels)} nég / {sum(l==1 for l in train_labels)} pos)")
    print(f"  ✅ Validation : {len(val_texts):>4} exemples  "
          f"({sum(l==0 for l in val_labels)} nég / {sum(l==1 for l in val_labels)} pos)")
    print(f"  ✅ Test       : {len(test_texts):>4} exemples  "
          f"({sum(l==0 for l in test_labels)} nég / {sum(l==1 for l in test_labels)} pos)")

    return {
        "train":      (train_texts,  train_labels),
        "validation": (val_texts,    val_labels),
        "test":       (test_texts,   test_labels),
    }


def tokenize_subsets(
    subsets: dict,
    tokenizer=None,
    max_length: int = MAX_SEQ_LENGTH
) -> tuple:
    """
    Tokenise tous les splits avec le tokenizer BERT-base uncased.

    Le padding est appliqué à la longueur du batch (padding=True)
    et la troncature à max_length pour maîtriser la mémoire CPU.

    Paramètres
    ----------
    subsets    : dict retourné par load_imdb_subsets()
    tokenizer  : AutoTokenizer préchargé (ou None → chargement auto)
    max_length : longueur max de séquence en tokens

    Retourne
    --------
    (dict{split → IMDbDataset}, tokenizer)
    """
    if tokenizer is None:
        print(f"\n🔤 Chargement du tokenizer ({MODEL_NAME})...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    datasets_out = {}
    for split_name, (texts, labels) in subsets.items():
        encodings = tokenizer(
            texts,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors=None,   # listes Python → conversion en tenseur dans __getitem__
        )
        datasets_out[split_name] = IMDbDataset(encodings, labels)
        print(f"  🔤 {split_name:<12}: {len(labels)} exemples tokenisés "
              f"(max_length={max_length})")

    return datasets_out, tokenizer


# ──────────────────────────────────────────────────────────────
#  Test rapide (python -m src.data_loader)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    subsets           = load_imdb_subsets()
    tokenized, tok    = tokenize_subsets(subsets)
    sample            = tokenized["train"][0]
    print(f"\n📌 Exemple — input_ids shape : {sample['input_ids'].shape}")
    print(f"              label           : {sample['labels'].item()}")
