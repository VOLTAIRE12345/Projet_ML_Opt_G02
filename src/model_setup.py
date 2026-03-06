"""
==============================================================
 G02 — IMDb + BERT-base | Problématique P02 : Régularisation
==============================================================
Module : model_setup.py
Rôle   : Initialisation de BERT-base avec gestion adaptative
          du matériel (CPU / GPU) et configuration du dropout.

Auteur : Groupe G02
Date   : Mars 2026
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BertConfig,
)

from src.data_loader import MODEL_NAME, NUM_LABELS


# ──────────────────────────────────────────────────────────────
#  Détection automatique du device
# ──────────────────────────────────────────────────────────────
def get_device() -> torch.device:
    """
    Retourne le meilleur device disponible.
    BERT-base (110M params) nécessite idéalement un GPU ;
    l'adaptation CPU est assurée par les sous-ensembles réduits.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚙️  Device sélectionné : {device}")
    if device.type == "cpu":
        # Utilise tous les cœurs disponibles pour accélérer les opérations matricielles
        torch.set_num_threads(4)
        print(f"   → Threads CPU configurés : {torch.get_num_threads()}")
    return device


# ──────────────────────────────────────────────────────────────
#  Chargement du modèle BERT-base
# ──────────────────────────────────────────────────────────────
def load_bert_model(
    dropout_rate: float = 0.1,
    attention_dropout: float = 0.1,
    device: torch.device = None,
    num_labels: int = NUM_LABELS,
) -> tuple:
    """
    Charge BERT-base-uncased (M02) configuré pour la classification
    binaire de sentiments (IMDb).

    BERT possède deux points de dropout paramétrables :
      - hidden_dropout_prob     : dropout sur les couches cachées (MLP)
      - attention_probs_dropout : dropout sur les scores d'attention

    Ces deux valeurs sont les hyperparamètres centraux de P02.

    Paramètres
    ----------
    dropout_rate       : taux de dropout sur les couches cachées [0.0 – 0.5]
    attention_dropout  : taux de dropout sur l'attention [0.0 – 0.5]
    device             : torch.device (auto-détecté si None)
    num_labels         : nombre de classes (2 pour IMDb)

    Retourne
    --------
    (model, tokenizer, device)
    """
    if device is None:
        device = get_device()

    print(f"\n Chargement de BERT-base-uncased (M02)")
    print(f"   dropout_rate      = {dropout_rate}")
    print(f"   attention_dropout = {attention_dropout}")

    # ── Configuration personnalisée ────────────────────────────
    # On surcharge la config par défaut de BERT pour injecter
    # les hyperparamètres de régularisation d'Optuna.
    config = BertConfig.from_pretrained(
        MODEL_NAME,
        num_labels                 = num_labels,
        hidden_dropout_prob        = dropout_rate,       # dropout MLP
        attention_probs_dropout_prob = attention_dropout, # dropout attention
    )

    # ── Chargement des poids pré-entraînés ────────────────────
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        config     = config,
        # float32 sur CPU ; float16 sur GPU pour économiser la mémoire
        torch_dtype = torch.float32,
        ignore_mismatched_sizes = True,   # tête de classification réinitialisée
    )

    model = model.to(device)

    # ── Tokenizer (partagé avec le modèle) ────────────────────
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Paramètres entraînables : {n_params:,}")

    return model, tokenizer, device


# ──────────────────────────────────────────────────────────────
#  Réinitialisation légère (tête de classification)
# ──────────────────────────────────────────────────────────────
def reset_classifier_head(model) -> None:
    """
    Réinitialise uniquement la couche de classification finale.
    Utile entre deux essais Optuna pour repartir de poids neutres
    sans recharger entièrement BERT depuis le disque.
    """
    if hasattr(model, "classifier"):
        for layer in model.classifier.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        print("Tête de classification réinitialisée.")


# ──────────────────────────────────────────────────────────────
#  Résumé de l'architecture
# ──────────────────────────────────────────────────────────────
def model_summary(model) -> None:
    """Affiche un résumé synthétique du modèle chargé."""
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n Résumé du modèle BERT-base")
    print(f"   Paramètres totaux      : {total:>12,}")
    print(f"   Paramètres entraînables: {trainable:>12,}")
    print(f"   Couches Transformer    : {model.config.num_hidden_layers}")
    print(f"   Têtes d'attention      : {model.config.num_attention_heads}")
    print(f"   Taille cachée          : {model.config.hidden_size}")
    print(f"   dropout (hidden)       : {model.config.hidden_dropout_prob}")
    print(f"   dropout (attention)    : {model.config.attention_probs_dropout_prob}")


# ──────────────────────────────────────────────────────────────
#  Test rapide
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    model, tok, dev = load_bert_model(dropout_rate=0.1, attention_dropout=0.1)
    model_summary(model)
