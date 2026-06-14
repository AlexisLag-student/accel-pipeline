"""
================================================================================
annotation_utils_lite.py — Manual Annotation Utilities (scikit-learn-free)
================================================================================
FAIR4RS Compliance (Chue Hong et al., 2021 — https://doi.org/10.15497/RDA00068)
  Findable     : Unique identifier via CITATION.cff / GitHub release tag v1.0.0
  Accessible   : MIT Licence — open access, no authentication required
  Interoperable: Drop-in replacement for annotation_utils.py without scikit-learn
  Reusable     : Self-contained, no external ML dependency required

Metadata
--------
Title       : Accelerometry Analysis Pipeline for Physical Activity Habit Modelling
Authors     : EuroMov DHM Lab, Université de Montpellier
Version     : 1.0.0
Date        : 2026-01-01
Licence     : MIT (see LICENSE file)
Repository  : https://github.com/euromov-dhm-lab/accelerometry-pipeline
Language    : Python 3.12+
Cite as     : See CITATION.cff

Description
-----------
    Lightweight annotation utilities module — identical API to annotation_utils.py
    but implemented without scikit-learn. Loaded automatically by app_euromov.py
    when scikit-learn is not available in the environment.

    Metrics implemented from scratch (numpy only):
        - Accuracy
        - Precision, Recall, F1-score per class
        - Weighted F1-score
        - Cohen's Kappa (Cohen, 1960 [6])
        - Confusion matrix

Scientific References
---------------------
    [6] Cohen, J. (1960). A coefficient of agreement for nominal scales.
        Educational and Psychological Measurement, 20(1), 37-46.
        https://doi.org/10.1177/001316446002000104

    [7] Landis, J.R., & Koch, G.G. (1977). The measurement of observer
        agreement for categorical data. Biometrics, 159-174.
        https://doi.org/10.2307/2529310
================================================================================
"""

import numpy as np
import pandas as pd


def ajouter_annotation(annotations_list, heure_debut, heure_fin, activite):
    """Ajoute une annotation manuelle à la liste."""
    annotations_list.append({
        'Heure_Debut': heure_debut,
        'Heure_Fin': heure_fin,
        'Activite_Annotee': activite
    })
    return annotations_list


def creer_df_annotations(annotations_list, df_res, epoch_seconds=10):
    """Crée un DataFrame des annotations transformées à la granularité des époques."""
    if not annotations_list:
        return pd.DataFrame()
    
    df_annot = df_res[['Heure']].copy()
    df_annot['Activite_Annotee'] = 'Non annoté'
    
    for annot in annotations_list:
        heure_debut = pd.to_datetime(annot['Heure_Debut'], format='%H:%M').time()
        heure_fin = pd.to_datetime(annot['Heure_Fin'], format='%H:%M').time()
        activite = annot['Activite_Annotee']
        
        mask = (df_annot['Heure'].dt.time >= heure_debut) & (df_annot['Heure'].dt.time < heure_fin)
        df_annot.loc[mask, 'Activite_Annotee'] = activite
    
    return df_annot


def calculer_metriques_correspondance(df_res, df_annotations):
    """Calcule les métriques de correspondance SANS scikit-learn."""
    if df_annotations.empty or (df_annotations['Activite_Annotee'] == 'Non annoté').all():
        return None
    
    mask = df_annotations['Activite_Annotee'] != 'Non annoté'
    y_true = df_res.loc[mask, 'Activite_Lissee'].values
    y_pred = df_annotations.loc[mask, 'Activite_Annotee'].values
    
    if len(y_true) == 0:
        return None
    
    # Accuracy
    accuracy = np.mean(y_true == y_pred)
    
    # Confusion matrix manuelle
    classes = sorted(set(y_true) | set(y_pred))
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    
    class_to_idx = {c: i for i, c in enumerate(classes)}
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true], class_to_idx[pred]] += 1
    
    # Precision, Recall, F1 par classe
    precision = {}
    recall = {}
    f1 = {}
    support = {}
    
    for i, classe in enumerate(classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support[classe] = int(cm[i, :].sum())
        
        precision[classe] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall[classe] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        if precision[classe] + recall[classe] > 0:
            f1[classe] = 2 * (precision[classe] * recall[classe]) / (precision[classe] + recall[classe])
        else:
            f1[classe] = 0.0
    
    # F1-Score pondéré
    f1_weighted = sum(f1[c] * support[c] for c in classes) / sum(support.values())
    
    # Kappa de Cohen
    po = accuracy
    pe = sum((cm[i, :].sum() / len(y_true)) * (cm[:, i].sum() / len(y_true)) for i in range(len(classes)))
    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0
    
    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'kappa': kappa,
        'classes': classes,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm,
        'y_true': y_true,
        'y_pred': y_pred
    }


def generer_df_confusion_matrix(metriques):
    """Génère un DataFrame pour afficher la matrice de confusion."""
    if not metriques:
        return None
    
    classes = metriques['classes']
    cm = metriques['confusion_matrix']
    
    df_cm = pd.DataFrame(
        cm,
        index=[f'Annoté: {c}' for c in classes],
        columns=[f'Détecté: {c}' for c in classes]
    )
    
    return df_cm


def generer_rapport_correspondance(metriques):
    """Génère un rapport textuel des métriques de correspondance."""
    if not metriques:
        return "Aucune donnée annotée à comparer."
    
    rapport = f"""
### 📊 Statistiques de Correspondance

**Score Global:**
- **Exactitude (Accuracy)**: {metriques['accuracy']*100:.1f}%
- **F1-Score (pondéré)**: {metriques['f1_weighted']:.3f}
- **Kappa de Cohen**: {metriques['kappa']:.3f}

**Performances par Activité:**
"""
    
    for classe in metriques['classes']:
        precision = metriques['precision'][classe]
        recall = metriques['recall'][classe]
        f1 = metriques['f1'][classe]
        support = metriques['support'][classe]
        
        rapport += f"""
- **{classe}**
  - Precision: {precision:.3f} | Recall: {recall:.3f} | F1: {f1:.3f} | Epochs: {support}
"""
    
    return rapport