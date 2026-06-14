"""
================================================================================
figure_export.py — Publication Figure Generation — EuroMov DHM Lab
================================================================================
FAIR4RS Compliance (Chue Hong et al., 2021 — https://doi.org/10.15497/RDA00068)
  Findable     : Unique identifier via CITATION.cff / GitHub release tag v1.0.0
  Accessible   : MIT Licence — open access, no authentication required
  Interoperable: Outputs standard PNG files (300 DPI) usable in any document
  Reusable     : Standalone module, importable without Streamlit dependency

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
    Generates publication-ready figures from accelerometry data processed
    by the pipeline. Figures illustrate the signal at each processing step
    to support thesis writing and scientific reporting.

    Figures produced:
        fig1_vm_timeseries.png   — Raw VM signal over the full day
        fig2_vm_thresholds.png   — VM with classification thresholds overlaid
        fig3_smoothing.png       — Before/after 60 s modal smoothing comparison
        fig4_nonwear.png         — Non-wear periods on VM signal (Choi et al., 2011)
        fig5_sessions_*.png      — Session extraction for selected activities
        fig6_multiday_*.png      — Multi-day session recurrence timeline

    All figures saved to figures/ as PNG, 300 DPI, white background,
    publication style (matplotlib + seaborn).

Dependencies
------------
    matplotlib>=3.7.0, seaborn>=0.13.0, pandas>=2.0.0, numpy>=1.26.0
    (see requirements.txt for pinned versions)
================================================================================
"""

import os

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required for Streamlit
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

# ==============================================================================
# STYLE GLOBAL — publication ready
# ==============================================================================

FIGURES_DIR = "figures"

PALETTE = {
    'Non-Wear'          : '#b3b3b3',
    'Sedentary'         : '#66c2a5',
    'Light'             : '#fc8d62',
    'Cycling'           : '#8da0cb',
    'Moderate (Walking)': '#e78ac3',
    'Vigorous (Running)': '#a6d854',
}

# Intensity thresholds in counts/min (Sasaki et al., 2011 ; Aguilar-Farías et al., 2014)
SEUILS_CPM = {
    'Sedentary / Light boundary'    : 200,
    'Light / Moderate boundary'     : 2690,
    'Moderate / Vigorous boundary'  : 6167,
}

SEUIL_COLORS = {
    'Sedentary / Light boundary'   : '#66c2a5',
    'Light / Moderate boundary'    : '#fc8d62',
    'Moderate / Vigorous boundary' : '#e78ac3',
}


def _setup_style():
    """Applique le style matplotlib pour figures de publication."""
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update({
        'font.family'     : 'serif',
        'font.size'       : 11,
        'axes.titlesize'  : 12,
        'axes.labelsize'  : 11,
        'xtick.labelsize' : 9,
        'ytick.labelsize' : 9,
        'legend.fontsize' : 9,
        'figure.dpi'      : 150,
        'savefig.dpi'     : 300,
        'savefig.bbox'    : 'tight',
        'savefig.facecolor': 'white',
    })


def _save(fig, filename):
    """Sauvegarde une figure dans le dossier figures/ et la ferme."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    return path


def _heure_labels(df, n_ticks=8):
    """Retourne des indices et labels d'heures régulièrement espacés."""
    idx   = np.linspace(0, len(df) - 1, n_ticks, dtype=int)
    labels = [df['Heure'].iloc[i].strftime('%H:%M') for i in idx]
    return idx, labels


# ==============================================================================
# FIG 1 — VM time series (raw signal)
# ==============================================================================

def fig1_vm_timeseries(df, epoch_seconds=10, day_label=""):
    """
    Signal VM counts sur la journée complète.

    Montre le signal brut tel qu'il sort de la pipeline agcounts,
    avant toute classification. Illustre la variabilité temporelle
    de l'activité physique au cours de la journée.

    Paramètres
    ----------
    df : pd.DataFrame
        Données traitées avec colonnes 'Heure' et 'VM_Counts'.
    epoch_seconds : int
        Durée d'une époque en secondes (pour l'axe y en cpm).
    day_label : str
        Label du jour affiché dans le titre.

    Retourne
    --------
    str : chemin du fichier PNG sauvegardé.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    # Convert counts/epoch to counts/min for interpretability
    facteur_cpm = 60.0 / epoch_seconds
    vm_cpm = df['VM_Counts'] * facteur_cpm

    ax.plot(range(len(df)), vm_cpm, color='#2c7bb6', linewidth=0.8, alpha=0.85)
    ax.fill_between(range(len(df)), vm_cpm, alpha=0.15, color='#2c7bb6')

    idx, labels = _heure_labels(df)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Vector Magnitude (counts·min⁻¹)")
    title = "VM Accelerometry Signal — Daily Time Series"
    if day_label:
        title += f" ({day_label})"
    ax.set_title(title)
    ax.set_xlim(0, len(df) - 1)
    ax.set_ylim(bottom=0)

    sns.despine(ax=ax)
    return _save(fig, "fig1_vm_timeseries.png")


# ==============================================================================
# FIG 2 — VM with classification thresholds
# ==============================================================================

def fig2_vm_thresholds(df, epoch_seconds=10, day_label=""):
    """
    Signal VM avec seuils de classification superposés.

    Illustre l'origine physique des seuils en counts/min issus de la
    littérature (Sasaki et al., 2011 ; Aguilar-Farías et al., 2014).
    Chaque zone colorée correspond à une classe d'intensité.

    Paramètres
    ----------
    df : pd.DataFrame
        Données traitées avec colonnes 'Heure' et 'VM_Counts'.
    epoch_seconds : int
        Durée d'une époque en secondes.
    day_label : str
        Label du jour affiché dans le titre.

    Retourne
    --------
    str : chemin du fichier PNG sauvegardé.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 5))

    facteur_cpm = 60.0 / epoch_seconds
    vm_cpm = df['VM_Counts'] * facteur_cpm
    x      = range(len(df))
    y_max  = max(vm_cpm.max() * 1.1, SEUILS_CPM['Moderate / Vigorous boundary'] * 1.2)

    # Coloured intensity zones
    zones = [
        (0,    200,   '#66c2a5', 'Sedentary  (< 200 cpm)'),
        (200,  2690,  '#fc8d62', 'Light  (200–2 690 cpm)'),
        (2690, 6167,  '#e78ac3', 'Moderate  (2 690–6 167 cpm)'),
        (6167, y_max, '#a6d854', 'Vigorous  (≥ 6 167 cpm)'),
    ]
    for y0, y1, color, label in zones:
        ax.axhspan(y0, y1, alpha=0.12, color=color, label=label)

    # Threshold lines
    for label, cpm in SEUILS_CPM.items():
        ax.axhline(cpm, color=SEUIL_COLORS[label], linewidth=1.2,
                   linestyle='--', alpha=0.8)
        ax.text(len(df) * 0.01, cpm + y_max * 0.01, f'{cpm} cpm',
                fontsize=8, color=SEUIL_COLORS[label])

    ax.plot(x, vm_cpm, color='#2c7bb6', linewidth=0.8, alpha=0.9, zorder=5)

    idx, labels = _heure_labels(df)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Vector Magnitude (counts·min⁻¹)")
    title = "VM Signal with Intensity Classification Thresholds"
    if day_label:
        title += f" ({day_label})"
    ax.set_title(title)
    ax.set_xlim(0, len(df) - 1)
    ax.set_ylim(0, y_max)

    ax.legend(
        loc='upper left', bbox_to_anchor=(1.01, 1),
        borderaxespad=0, framealpha=0.9, fontsize=8
    )
    sns.despine(ax=ax)
    return _save(fig, "fig2_vm_thresholds.png")


# ==============================================================================
# FIG 3 — Smoothing comparison
# ==============================================================================

def fig3_smoothing(df, epoch_seconds=10, day_label=""):
    """
    Comparaison avant/après lissage modal glissant.

    Montre l'effet du lissage sur 60 secondes sur la classification
    d'activité. Le panneau du haut montre la classification brute
    (Activite), celui du bas la classification lissée (Activite_Lissee).
    Illustre la réduction des fluctuations transitoires.

    Paramètres
    ----------
    df : pd.DataFrame
        Données avec colonnes 'Activite', 'Activite_Lissee', 'Heure'.
    epoch_seconds : int
        Durée d'une époque en secondes.
    day_label : str
        Label du jour.

    Retourne
    --------
    str : chemin du fichier PNG sauvegardé.
    """
    if 'Activite' not in df.columns:
        return None

    _setup_style()

    # Map English labels back for plotting consistency
    label_map = {
        'Non-Wear'          : 0,
        'Sedentary'         : 1,
        'Light'             : 2,
        'Cycling'           : 3,
        'Moderate (Walking)': 4,
        'Vigorous (Running)': 5,
    }
    colors_list = list(PALETTE.values())

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.subplots_adjust(hspace=0.08)

    for ax, col, title in zip(
        axes,
        ['Activite', 'Activite_Lissee'],
        ['Raw classification (before smoothing)', 'Smoothed classification (60 s modal window)']
    ):
        series = df[col].map(label_map).fillna(1).astype(int)
        colors_epoch = [colors_list[min(v, len(colors_list)-1)] for v in series]

        for i, (val, color) in enumerate(zip(series, colors_epoch)):
            ax.bar(i, 1, width=1.0, color=color, linewidth=0, align='edge')

        ax.set_yticks([0.5])
        ax.set_yticklabels([title], fontsize=9)
        ax.set_ylim(0, 1)
        ax.tick_params(left=False)
        sns.despine(ax=ax, left=True)

    idx, labels = _heure_labels(df)
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(labels)
    axes[1].set_xlabel("Time of day")

    # Legend — placed to the right of the figure, outside the panels
    patches = [mpatches.Patch(color=PALETTE[k], label=k) for k in PALETTE]
    fig.legend(
        handles=patches,
        loc='center left',
        bbox_to_anchor=(1.01, 0.5),
        framealpha=0.9,
        fontsize=8
    )

    title = "Effect of Modal Smoothing on Activity Classification"
    if day_label:
        title += f" ({day_label})"
    fig.suptitle(title, fontsize=12)

    return _save(fig, "fig3_smoothing.png")


# ==============================================================================
# FIG 4 — Non-wear detection
# ==============================================================================

def fig4_nonwear(df, epoch_seconds=10, day_label=""):
    """
    Signal VM avec périodes de non-port identifiées.

    Illustre l'algorithme de Choi et al. (2011) : les périodes où le VM
    reste sous le seuil de quasi-immobilité pendant 90 minutes consécutives
    sont surlignées en gris. Le signal VM est tracé en bleu.

    Paramètres
    ----------
    df : pd.DataFrame
        Données avec colonnes 'VM_Counts', 'Activite_Lissee', 'Heure'.
    epoch_seconds : int
        Durée d'une époque en secondes.
    day_label : str
        Label du jour.

    Retourne
    --------
    str : chemin du fichier PNG sauvegardé.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    facteur_cpm = 60.0 / epoch_seconds
    vm_cpm = df['VM_Counts'] * facteur_cpm
    x      = np.arange(len(df))

    # Shade non-wear periods
    non_wear_mask = df['Activite_Lissee'] == 'Non-Wear'
    in_nw = False
    start_nw = 0
    for i, nw in enumerate(non_wear_mask):
        if nw and not in_nw:
            start_nw = i
            in_nw = True
        elif not nw and in_nw:
            ax.axvspan(start_nw, i, color='#b3b3b3', alpha=0.4,
                       label='Non-wear period' if start_nw == 0 else '')
            in_nw = False
    if in_nw:
        ax.axvspan(start_nw, len(df), color='#b3b3b3', alpha=0.4)

    ax.plot(x, vm_cpm, color='#2c7bb6', linewidth=0.8, alpha=0.9, zorder=5)

    idx, labels = _heure_labels(df)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)

    ax.set_xlabel("Time of day")
    ax.set_ylabel("Vector Magnitude (counts·min⁻¹)")
    title = "Non-Wear Period Detection (Choi et al., 2011)"
    if day_label:
        title += f" ({day_label})"
    ax.set_title(title)
    ax.set_xlim(0, len(df) - 1)
    ax.set_ylim(bottom=0)

    nw_patch  = mpatches.Patch(color='#b3b3b3', alpha=0.5, label='Non-wear period (≥ 90 min below threshold)')
    sig_patch = mpatches.Patch(color='#2c7bb6', label='VM signal')
    ax.legend(
        handles=[sig_patch, nw_patch],
        loc='upper left', bbox_to_anchor=(1.01, 1),
        borderaxespad=0, framealpha=0.9, fontsize=8
    )
    sns.despine(ax=ax)
    return _save(fig, "fig4_nonwear.png")


# ==============================================================================
# FIG 5 — Session extraction
# ==============================================================================

def fig5_sessions(df, epoch_seconds=10, duree_min=1.0, day_label="",
                  activites_highlight=None):
    """
    Extraction de toutes les sessions continues par activité sur la journée.

    Affiche le signal VM avec les classes d'activité sélectionnées surlignées.
    Les périodes de non-port sont grisées comme dans fig4 (Choi et al., 2011).
    Si aucune activité n'est sélectionnée, toutes sont affichées.

    Paramètres
    ----------
    df : pd.DataFrame
        Données avec colonnes 'VM_Counts', 'Activite_Lissee', 'Heure'.
    epoch_seconds : int
        Durée d'une époque en secondes.
    duree_min : float
        Durée minimale d'une session en minutes pour être surlignée.
    day_label : str
        Label du jour affiché dans le titre.
    activites_highlight : list[str] ou None
        Liste des activités à surligner. Si None ou liste vide,
        toutes les activités (hors Non-Wear) sont affichées.

    Retourne
    --------
    str : chemin du fichier PNG sauvegardé.
    """
    _setup_style()
    fig, ax = plt.subplots(figsize=(12, 4))

    facteur_cpm  = 60.0 / epoch_seconds
    vm_cpm       = df['VM_Counts'] * facteur_cpm
    seuil_epochs = duree_min * 60 / epoch_seconds

    # Default: all active classes
    if not activites_highlight:
        activites_highlight = [a for a in PALETTE if a != 'Non-Wear']

    legend_handles = []

    # 1. Shade Non-Wear periods in grey
    non_wear_mask = df['Activite_Lissee'] == 'Non-Wear'
    in_nw = False
    start_nw = 0
    first_nw = True
    for i, nw in enumerate(non_wear_mask):
        if nw and not in_nw:
            start_nw = i
            in_nw = True
        elif not nw and in_nw:
            ax.axvspan(start_nw, i, color='#b3b3b3', alpha=0.35)
            first_nw = False
            in_nw = False
    if in_nw:
        ax.axvspan(start_nw, len(df), color='#b3b3b3', alpha=0.35)

    if not first_nw or in_nw:
        legend_handles.append(
            mpatches.Patch(color='#b3b3b3', alpha=0.5, label='Non-wear period')
        )

    # 2. Shade session blocks for selected activities only
    changes = df['Activite_Lissee'].ne(df['Activite_Lissee'].shift()).cumsum()
    added_to_legend = set()

    for _, grp in df.groupby(changes):
        act = grp['Activite_Lissee'].iloc[0]
        if act == 'Non-Wear' or act not in activites_highlight:
            continue
        if len(grp) < seuil_epochs:
            continue

        i0    = grp.index[0]  - df.index[0]
        i1    = grp.index[-1] - df.index[0]
        color = PALETTE.get(act, '#aaaaaa')
        ax.axvspan(i0, i1 + 1, color=color, alpha=0.30)

        if act not in added_to_legend:
            legend_handles.append(
                mpatches.Patch(color=color, alpha=0.6, label=act)
            )
            added_to_legend.add(act)

    # 3. VM signal on top
    ax.plot(range(len(df)), vm_cpm, color='#2c7bb6',
            linewidth=0.8, alpha=0.9, zorder=5)
    legend_handles.insert(0, mpatches.Patch(color='#2c7bb6', label='VM signal'))

    idx, labels = _heure_labels(df)
    ax.set_xticks(idx)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Time of day")
    ax.set_ylabel("Vector Magnitude (counts·min⁻¹)")

    highlighted = ', '.join(activites_highlight) if activites_highlight else 'All'
    title = f"Session Extraction — {highlighted}"
    if day_label:
        title += f" ({day_label})"
    ax.set_title(title)
    ax.set_xlim(0, len(df) - 1)
    ax.set_ylim(bottom=0)

    ax.legend(
        handles=legend_handles,
        loc='upper left', bbox_to_anchor=(1.01, 1),
        borderaxespad=0, framealpha=0.9, fontsize=8
    )
    sns.despine(ax=ax)

    # Filename reflects selection
    suffix = '_'.join(a.replace(' ', '_').replace('(', '').replace(')', '')
                      for a in activites_highlight) if activites_highlight else 'all'
    return _save(fig, f"fig5_sessions_{suffix}.png")


# ==============================================================================
# FIG 6 — Multi-day session recurrence
# ==============================================================================

def fig6_multiday(agd_days, activites=None, epoch_seconds=10, duree_min=1.0):
    """
    Récurrence des sessions sur plusieurs jours — une ou plusieurs activités.

    Produit un graphique à double colonne par activité sélectionnée :
    panneau gauche = timeline des sessions (barres horizontales par jour),
    panneau droit = durée totale quotidienne. Chaque activité est dans une
    ligne de sous-graphiques séparée. Illustre le concept de récurrence
    et d'habituation sur la période d'enregistrement.

    Paramètres
    ----------
    agd_days : dict[str -> pd.DataFrame]
        Dictionnaire jour -> DataFrame traité.
    activites : list[str] ou None
        Liste des activités à représenter. Si None, utilise
        ['Moderate (Walking)', 'Vigorous (Running)', 'Cycling'].
    epoch_seconds : int
        Durée d'une époque en secondes.
    duree_min : float
        Durée minimale d'une session en minutes.

    Retourne
    --------
    str : chemin du fichier PNG sauvegardé.
    """
    if not activites:
        activites = ['Moderate (Walking)', 'Vigorous (Running)', 'Cycling']

    _setup_style()
    jours  = sorted(agd_days.keys())
    n_days = len(jours)
    n_acts = len(activites)

    fig, axes = plt.subplots(
        n_acts, 2,
        figsize=(14, max(3, n_days * 0.6) * n_acts),
        gridspec_kw={'width_ratios': [3, 1]},
        squeeze=False
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.05)

    for row, activite in enumerate(activites):
        color        = PALETTE.get(activite, '#8da0cb')
        ax_left      = axes[row, 0]
        ax_right     = axes[row, 1]
        seuil_epochs = duree_min * 60 / epoch_seconds

        # Collect sessions per day for this activity
        daily_data = []
        for jour in jours:
            df       = agd_days[jour]
            changes  = df['Activite_Lissee'].ne(df['Activite_Lissee'].shift()).cumsum()
            sessions = []
            for _, grp in df.groupby(changes):
                if grp['Activite_Lissee'].iloc[0] == activite and len(grp) >= seuil_epochs:
                    t_start = grp['Heure'].iloc[0]
                    t_end   = grp['Heure'].iloc[-1]
                    dur_min = len(grp) * epoch_seconds / 60
                    sessions.append((t_start, t_end, dur_min))
            total_min = sum(s[2] for s in sessions)
            daily_data.append({'jour': jour, 'sessions': sessions, 'total_min': total_min})

        # Left panel: session timeline
        for row_idx, d in enumerate(daily_data):
            for t_start, t_end, dur in d['sessions']:
                h_start = t_start.hour + t_start.minute / 60
                h_end   = t_end.hour   + t_end.minute   / 60
                ax_left.barh(row_idx, h_end - h_start, left=h_start,
                             height=0.6, color=color, alpha=0.75, edgecolor='white')

        ax_left.set_yticks(range(n_days))
        ax_left.set_yticklabels(
            [pd.Timestamp(j).strftime('%d %b') for j in jours], fontsize=9
        )
        ax_left.set_xlim(0, 24)
        ax_left.set_xticks(range(0, 25, 4))
        ax_left.set_xticklabels(
            [f'{h:02d}:00' for h in range(0, 25, 4)], fontsize=8
        )
        ax_left.set_title(f"{activite} — Daily Session Timeline", fontsize=10)
        ax_left.invert_yaxis()
        if row == n_acts - 1:
            ax_left.set_xlabel("Time of day (h)")
        sns.despine(ax=ax_left)

        # Right panel: total duration bar
        totals = [d['total_min'] for d in daily_data]
        ax_right.barh(range(n_days), totals, height=0.6, color=color, alpha=0.75)
        ax_right.set_yticks(range(n_days))
        ax_right.set_yticklabels([])
        ax_right.set_title("Total / day", fontsize=10)
        ax_right.invert_yaxis()
        if row == n_acts - 1:
            ax_right.set_xlabel("Duration (min)")
        sns.despine(ax=ax_right)

    suffix = '_'.join(a.replace(' ', '_').replace('(', '').replace(')', '')
                      for a in activites)
    fig.suptitle(
        f"Session Recurrence — {', '.join(activites)} (≥ {duree_min} min)",
        fontsize=12, y=1.01
    )
    fig.tight_layout()
    return _save(fig, f"fig6_multiday_{suffix}.png")


# ==============================================================================
# MASTER FUNCTION — generates all figures for one day
# ==============================================================================

def generer_toutes_figures(df, epoch_seconds=10, day_label="",
                           agd_days=None, activites_sessions=None):
    """
    Génère l'ensemble des figures pour un jour donné.

    Paramètres
    ----------
    df : pd.DataFrame
        Données traitées pour le jour sélectionné.
    epoch_seconds : int
        Durée d'une époque en secondes.
    day_label : str
        Label du jour affiché dans les titres.
    agd_days : dict, optionnel
        Dictionnaire tous jours pour fig6. Si None, fig6 est ignorée.
    activites_sessions : list[str], optionnel
        Activités à surligner dans fig5 et fig6.
        Si None ou liste vide, toutes les activités sont affichées.

    Retourne
    --------
    list[str] : liste des chemins des fichiers PNG sauvegardés.
    """
    paths = []

    paths.append(fig1_vm_timeseries(df, epoch_seconds, day_label))
    paths.append(fig2_vm_thresholds(df, epoch_seconds, day_label))

    p3 = fig3_smoothing(df, epoch_seconds, day_label)
    if p3:
        paths.append(p3)

    paths.append(fig4_nonwear(df, epoch_seconds, day_label))
    paths.append(fig5_sessions(df, epoch_seconds=epoch_seconds,
                               duree_min=1.0, day_label=day_label,
                               activites_highlight=activites_sessions))

    if agd_days and len(agd_days) > 1:
        paths.append(fig6_multiday(agd_days, activites=activites_sessions,
                                   epoch_seconds=epoch_seconds))

    return [p for p in paths if p is not None]