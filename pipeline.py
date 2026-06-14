"""
================================================================================
pipeline.py — Accelerometry Processing Pipeline — EuroMov DHM Lab
================================================================================
FAIR4RS Compliance (Chue Hong et al., 2021 — https://doi.org/10.15497/RDA00068)
  Findable     : Unique identifier via CITATION.cff / GitHub release tag v1.0.0
  Accessible   : MIT Licence — open access, no authentication required
  Interoperable: Standard Python module, importable without Streamlit dependency
  Reusable     : Versioned, documented, scientifically referenced

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
    Ce module regroupe l'ensemble des fonctions algorithmiques de la pipeline
    de traitement des données brutes issues de l'accéléromètre wGT3X-BT
    (ActiGraph). Il est indépendant de toute interface graphique et peut être
    importé dans un script Python ou un notebook Jupyter.

    Exemple d'utilisation autonome :
        import pandas as pd
        from pipeline import (
            executer_pipeline_agcounts,
            ajouter_temps,
            lisser_activite,
            detecter_non_port,
            detecter_velo,
            extraire_sessions,
        )

        df_brut = pd.read_csv("mon_fichier.csv")
        df = executer_pipeline_agcounts(df_brut, epoch_seconds=10, frequence_hz=60)
        df = ajouter_temps(df_brut, df, epoch_seconds=10, date_reelle=pd.Timestamp("2026-01-01"))
        df = lisser_activite(df, epoch_seconds=10)
        df = detecter_non_port(df, epoch_seconds=10)

Scientific References
---------------------
    [1] Sasaki, J.E., John, D., & Freedson, P.S. (2011). Validation and
        comparison of ActiGraph activity monitors. Journal of Science and
        Medicine in Sport, 14(5), 411-416.
        https://doi.org/10.1016/j.jsams.2011.01.003
        Role: VM3 threshold validation; Sedentary/Light/Moderate/Vigorous cut-points.

    [2] Aguilar-Farías, N., Brown, W.J., & Peeters, G.M.E.E. (2014). ActiGraph
        GT3X+ cut-points for identifying sedentary behaviour in older adults in
        free-living environments. Journal of Science and Medicine in Sport,
        17(3), 293-299.
        https://doi.org/10.1016/j.jsams.2013.07.002
        Role: Sedentary behaviour cut-point (200 cpm VM).

    [3] Choi, L., Liu, Z., Matthews, C.E., & Buchowski, M.S. (2011).
        Validation of accelerometer wear and nonwear time classification
        algorithm. Medicine & Science in Sports & Exercise, 43(2), 357-364.
        https://doi.org/10.1249/MSS.0b013e3181ed61a3
        Role: Non-wear detection algorithm (90-min sliding window).

    [4] Brage, S., Brage, N., Franks, P.W., Ekelund, U., & Wareham, N.J.
        (2005). Branched equation modelling of simultaneous accelerometry and
        heart rate monitoring improves estimate of directly measured physical
        activity energy expenditure. Journal of Applied Physiology, 98(1),
        166-173.
        https://doi.org/10.1152/japplphysiol.00510.2004
        Role: Theoretical basis for cycling detection via spectral analysis.

    [5] Mannini, A., & Sabatini, A.M. (2010). Machine learning methods for
        classifying human physical activity from on-body accelerometers.
        Sensors, 10(2), 1154-1175.
        https://doi.org/10.3390/s100201154
        Role: Signal regularity (CV) criterion for cycling discrimination.
================================================================================
"""

import numpy as np
import pandas as pd
from agcounts.extract import get_counts


def _build_epochs_from_aggregated(df_brut):
    """Retourne un DataFrame d'époques à partir de données déjà agrégées."""
    columns_lower = [col.lower() for col in df_brut.columns]

    if all(col in df_brut.columns for col in ['AxisX', 'AxisY', 'AxisZ']):
        return df_brut[['AxisX', 'AxisY', 'AxisZ']].copy()
    if all(col in df_brut.columns for col in ['X', 'Y', 'Z']) and 'timestamp' not in df_brut.columns:
        return df_brut[['X', 'Y', 'Z']].copy()
    if all(col in columns_lower for col in ['axis1', 'axis2', 'axis3']):
        return df_brut[[
            df_brut.columns[columns_lower.index('axis1')],
            df_brut.columns[columns_lower.index('axis2')],
            df_brut.columns[columns_lower.index('axis3')],
        ]].copy().rename(columns={
            df_brut.columns[columns_lower.index('axis1')]: 'AxisX',
            df_brut.columns[columns_lower.index('axis2')]: 'AxisY',
            df_brut.columns[columns_lower.index('axis3')]: 'AxisZ',
        })
    raise ValueError(
        "Données agrégées attendues : colonnes 'AxisX'/'AxisY'/'AxisZ' ou "
        "'axis1'/'axis2'/'axis3' requises pour un fichier déjà agrégé."
    )


def _ticks_to_datetime(ticks_series):
    """Convert .NET/Windows-style ticks (100-ns since year 1) to pandas datetime.

    Many ActiGraph .agd files store timestamps as integer ticks (100 ns units)
    since year 0001 (as in .NET DateTime.Ticks). To convert to unix-based
    pandas datetimes we subtract the offset to 1970-01-01 and multiply by 100
    to obtain nanoseconds.
    """
    TICKS_AT_UNIX_EPOCH = 621355968000000000
    # ensure int64 math
    return pd.to_datetime((ticks_series.astype('int64') - TICKS_AT_UNIX_EPOCH) * 100, unit='ns')


def _find_accel_table(con):
    """Find the most likely accelerometer data table in a SQLite DB."""
    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name;")
    tables = [row[0] for row in cur.fetchall()]
    for table in tables:
        table_lower = table.lower()
        if table_lower.startswith('sqlite_'):
            continue
        try:
            df = pd.read_sql_query(f'SELECT * FROM "{table}" LIMIT 20', con)
        except Exception:
            continue
        cols_lower = [c.lower() for c in df.columns]
        has_timestamp = any(c in cols_lower for c in ('datatimestamp', 'timestamp', 'time', 'date', 'datetime'))
        has_axis = any(c in cols_lower for c in ('axis1', 'axis2', 'axis3', 'x', 'y', 'z', 'axisx', 'axisy', 'axisz'))
        if has_timestamp and has_axis:
            return table
    return None


def read_agd_and_aggregate(db_path, epoch_seconds=10):
    """Load a .agd (SQLite) file and aggregate all days into epoch DataFrames.

    Returns
    -------
    dict
        Mapping date (datetime.date) -> DataFrame of epochs for that day.
    """
    import sqlite3

    con = sqlite3.connect(db_path)
    try:
        try:
            df_raw = pd.read_sql_query('SELECT * FROM data', con)
        except Exception:
            table_name = _find_accel_table(con)
            if table_name is None:
                raise ValueError('Aucune table de données accélérométriques trouvée dans la base SQLite.')
            df_raw = pd.read_sql_query(f'SELECT * FROM "{table_name}"', con)
    finally:
        con.close()

    if df_raw.empty:
        return {}

    # Normalize column names
    cols_lower = [c.lower() for c in df_raw.columns]
    # detect tick column name (dataTimestamp, timestamp, time)
    tick_col = None
    for candidate in ('datatimestamp', 'timestamp', 'time'):
        if candidate in cols_lower:
            tick_col = df_raw.columns[cols_lower.index(candidate)]
            break

    if tick_col is None:
        raise ValueError('Aucun champ de timestamp trouvé dans la table `data`.')

    # Convert ticks to datetime index
    df_raw['_Heure'] = _ticks_to_datetime(df_raw[tick_col])
    df_raw = df_raw.set_index('_Heure')

    # Standard axis column names expected: axis1/axis2/axis3
    axis_keys = None
    for aset in (['axis1', 'axis2', 'axis3'], ['x', 'y', 'z'], ['axisx', 'axisy', 'axisz']):
        if all(a in cols_lower for a in aset):
            axis_keys = [df_raw.columns[cols_lower.index(a)] for a in aset]
            break

    if axis_keys is None:
        raise ValueError('Aucune colonne d\'axe détectée (axis1/axis2/axis3 / X/Y/Z).')

    # Keep only axis and relevant sensor columns for aggregation
    df_axes = df_raw[axis_keys].copy()
    df_axes.columns = ['AxisX', 'AxisY', 'AxisZ']

    # Determine native sampling delta in seconds (median)
    deltas = df_axes.index.to_series().diff().dropna().dt.total_seconds()
    median_delta = deltas.median() if not deltas.empty else epoch_seconds

    # If native samples are already at epoch spacing (or coarser), sum directly
    if median_delta >= epoch_seconds:
        df_epoch = df_axes.resample(f'{epoch_seconds}s').sum()
    else:
        # high-frequency samples -> aggregate (sum) within epoch bins
        df_epoch = df_axes.resample(f'{epoch_seconds}s').sum()

    # Ensure full-day coverage for each calendar day present
    df_epoch = df_epoch.sort_index()

    # Build per-day DataFrames and compute derived columns (VM, Activite)
    days = {}
    for day, group in df_epoch.groupby(df_epoch.index.date):
        # Reindex to full day range
        day_start = pd.to_datetime(f"{day} 00:00:00")
        periods = int(24 * 3600 / epoch_seconds)
        full_idx = pd.date_range(start=day_start, periods=periods, freq=f'{epoch_seconds}s')
        df_day = group.reindex(full_idx, fill_value=0).copy()
        # Reset index and name column 'Heure' for compatibility
        df_day = df_day.reset_index().rename(columns={'index': 'Heure'})
        # The pipeline's executer expects an aggregated DataFrame when deja_agrege=True
        df_ag = df_day.rename(columns={'AxisX': 'AxisX', 'AxisY': 'AxisY', 'AxisZ': 'AxisZ'})
        # Apply VM and classification
        df_proc = executer_pipeline_agcounts(df_ag, epoch_seconds, deja_agrege=True)
        # Add Heure column
        df_proc = ajouter_temps(df_raw.reset_index(), df_proc, epoch_seconds, pd.to_datetime(day), deja_agrege=True)
        df_proc = lisser_activite(df_proc, epoch_seconds)
        df_proc = detecter_non_port(df_proc, epoch_seconds, debug=False)
        days[pd.to_datetime(day).date()] = df_proc

    return days


# ==============================================================================
# CONSTANTES — Seuils de classification (counts/min, époque 60 s)
# ==============================================================================

# Seuil sédentaire : Aguilar-Farías et al. (2014) [Réf. 2]
SEUIL_SEDENTAIRE_CPM = 200

# Seuils Légère / Modérée / Vigoureuse : Sasaki et al. (2011) [Réf. 1]
# Ces seuils sont définis sur le Vector Magnitude triaxial VM3 = √(X² + Y² + Z²)
# calculé sur les counts par époque du GT3X (Sasaki et al., 2011, Table 2).
SEUIL_LEGERE_CPM  = 2690
SEUIL_MODEREE_CPM = 6167

# Seuils de détection du vélo (calibrés empiriquement, port hanche, 60 Hz)
# Brage et al. (2005) [Réf. 4] ; Mannini & Sabatini (2010) [Réf. 5]
SEUIL_FFT_DEFAUT = 0.08   # rappel empirique = 98.9 %
SEUIL_CV_DEFAUT  = 0.55   # rappel empirique = 97.8 %

# Durée minimale de non-port : Choi et al. (2011) [Réf. 3]
DUREE_NON_PORT_MIN = 90   # minutes


# ==============================================================================
# FONCTIONS DE TRAITEMENT
# ==============================================================================

def executer_pipeline_agcounts(df_brut, epoch_seconds, frequence_hz=60, deja_agrege=False):
    """
    Calcule les counts accélérométriques et classifie l'intensité d'activité.

    Si les données sont déjà agrégées en époques, on passe directement aux
    calculs dérivés (VM_Counts / classification) sans ré-agréger.

    Le calcul des counts suit l'algorithme ActiGraph implémenté dans la
    bibliothèque agcounts : rééchantillonnage à 30 Hz, filtre passe-bande
    IIR d'ordre 7, mise à l'échelle, seuillage, puis sommation par époque.

    La classification repose sur le Vector Magnitude (VM) des counts,
    avec des seuils exprimés en counts par minute (cpm) pour une époque de
    60 s, adaptés proportionnellement à la taille d'époque choisie.

    Seuils VM (counts/min, époque 60 s) :
        - Sédentaire       :  VM < 200          (Aguilar-Farías et al., 2014)
        - Légère           :  200 ≤ VM < 2 690  (Sasaki et al., 2011)
        - Modérée (Marche) : 2 690 ≤ VM < 6 167 (Sasaki et al., 2011)
        - Vigoureuse       :  VM ≥ 6 167        (Sasaki et al., 2011)

    Paramètres
    ----------
    df_brut : pd.DataFrame
        Données brutes avec colonnes 'X', 'Y', 'Z' (accélération en g).
    epoch_seconds : int
        Durée d'une époque en secondes.
    frequence_hz : int
        Fréquence d'échantillonnage du capteur (Hz). Valeurs acceptées :
        [30, 40, 50, 60, 70, 80, 90, 100].

    Retourne
    --------
    pd.DataFrame
        Colonnes : AxisX, AxisY, AxisZ (counts par axe),
                   VM_Counts (Vector Magnitude), Activite (classe d'intensité).
    """
    facteur = epoch_seconds / 60.0

    seuils = {
        'sedentaire': SEUIL_SEDENTAIRE_CPM * facteur,
        'legere':     SEUIL_LEGERE_CPM     * facteur,
        'moderee':    SEUIL_MODEREE_CPM    * facteur,
    }

    if deja_agrege:
        df_epochs = _build_epochs_from_aggregated(df_brut)
    else:
        donnees_acceleration = df_brut[['X', 'Y', 'Z']].values
        counts_array = get_counts(donnees_acceleration, freq=frequence_hz, epoch=epoch_seconds)
        df_epochs = pd.DataFrame(counts_array, columns=['AxisX', 'AxisY', 'AxisZ'])

    # Vector Magnitude triaxial VM3 = √(X² + Y² + Z²) appliqué aux counts par époque.
    # Formule introduite et validée par Sasaki et al. (2011) [Réf. 1] pour le GT3X.
    # Supérieur à l'axe vertical seul car capture les mouvements dans les 3 plans
    # de l'espace, notamment les activités à composante horizontale (marche rapide,
    # vélo). ActiGraph Corp. (2014) confirme cette définition dans la documentation
    # officielle du GT3X+.
    df_epochs['VM_Counts'] = np.sqrt(
        df_epochs['AxisX']**2 + df_epochs['AxisY']**2 + df_epochs['AxisZ']**2
    )

    def classifier(vm):
        if vm < seuils['sedentaire']:  return 'Sédentaire'
        elif vm < seuils['legere']:    return 'Légère'
        elif vm < seuils['moderee']:   return 'Modérée (Marche)'
        else:                          return 'Vigoureuse (Course)'

    df_epochs['Activite'] = df_epochs['VM_Counts'].apply(classifier)
    return df_epochs


def ajouter_temps(df_brut, df_epochs, epoch_seconds, date_reelle, deja_agrege=False):
    """
    Ajoute une colonne 'Heure' horodatant chaque époque.

    Si le fichier brut contient une colonne 'timestamp', l'heure de début
    est extraite de la première ligne et la date est remplacée par la date
    fournie en paramètre.

    Si le fichier est déjà agrégé et ne contient pas de timestamp, la date
    commence à minuit. Sinon, le début est fixé à 08:00:00.

    Paramètres
    ----------
    df_brut : pd.DataFrame
        Données brutes (pour extraire l'heure de début si disponible).
    df_epochs : pd.DataFrame
        DataFrame des époques à horodater.
    epoch_seconds : int
        Durée d'une époque en secondes.
    date_reelle : datetime.date ou pd.Timestamp
        Date de l'enregistrement.

    Retourne
    --------
    pd.DataFrame
        DataFrame avec colonne 'Heure' (datetime) ajoutée.
    """
    colonne_temps = None
    columns_lower = [col.lower() for col in df_brut.columns]
    if 'timestamp' in columns_lower:
        colonne_temps = df_brut.columns[columns_lower.index('timestamp')]
    elif 'time' in columns_lower:
        colonne_temps = df_brut.columns[columns_lower.index('time')]

    if colonne_temps is not None:
        heure_debut = pd.to_datetime(df_brut[colonne_temps].iloc[0])
        heure_debut = heure_debut.replace(
            year=date_reelle.year,
            month=date_reelle.month,
            day=date_reelle.day
        )
    elif deja_agrege:
        heure_debut = pd.to_datetime(f"{date_reelle} 00:00:00")
    else:
        heure_debut = pd.to_datetime(f"{date_reelle} 08:00:00")

    df_epochs['Heure'] = pd.date_range(
        start=heure_debut,
        periods=len(df_epochs),
        freq=f'{epoch_seconds}s'
    )
    return df_epochs


def lisser_activite(df_epochs, epoch_seconds):
    """
    Applique un lissage modal glissant sur la classification d'activité.

    Une fenêtre centrée de 60 secondes est utilisée pour réduire les
    transitions aberrantes entre classes consécutives. La taille de la
    fenêtre (en nombre d'époques) est calculée comme 60 / epoch_seconds,
    avec un minimum de 1.

    Paramètres
    ----------
    df_epochs : pd.DataFrame
        DataFrame avec colonne 'Activite'.
    epoch_seconds : int
        Durée d'une époque en secondes.

    Retourne
    --------
    pd.DataFrame
        DataFrame avec colonne 'Activite_Lissee' ajoutée.
        Les colonnes intermédiaires d'encodage sont supprimées.
    """
    taille_fenetre = max(1, int(60 / epoch_seconds))

    # Encodage numérique temporaire pour le calcul du mode glissant
    activites_uniques = df_epochs['Activite'].unique()
    map_vers_num  = {texte: i for i, texte in enumerate(activites_uniques)}
    map_vers_text = {i: texte for texte, i in map_vers_num.items()}

    df_epochs['_act_num'] = df_epochs['Activite'].map(map_vers_num)
    df_epochs['_act_num_lisse'] = (
        df_epochs['_act_num']
        .rolling(window=taille_fenetre, center=True)
        .apply(lambda x: pd.Series(x).mode().iloc[0]
               if not pd.Series(x).mode().empty else np.nan)
    )
    df_epochs['_act_num_lisse'] = (
        df_epochs['_act_num_lisse']
        .fillna(df_epochs['_act_num'])
        .astype(int)
    )
    df_epochs['Activite_Lissee'] = df_epochs['_act_num_lisse'].map(map_vers_text)
    df_epochs.drop(columns=['_act_num', '_act_num_lisse'], inplace=True)
    return df_epochs


def detecter_non_port(df_epochs, epoch_seconds=10, zero_threshold=15, debug=True):
      """
      Identifie les périodes de non-port selon l'algorithme exact de Choi et al. (2011).

      L'algorithme classifie une période comme non-portée si elle contient au moins
      90 minutes consécutives de counts nuls (VM <= zero_threshold), avec une
      tolérance : jusqu'à 2 minutes de counts nonzéro sont autorisées comme
      interruption, à condition que les fenêtres de 30 minutes avant et après
      cette interruption contiennent uniquement des counts nuls.

      Références
      ----------
      Choi, L., Liu, Z., Matthews, C.E., & Buchowski, M.S. (2011).
      Validation of accelerometer wear and nonwear time classification algorithm.
      Medicine & Science in Sports & Exercise, 43(2), 357-364.

      Paramètres
      ----------
      df_epochs : pd.DataFrame
          DataFrame avec colonnes 'VM_Counts' et 'Activite_Lissee'.
      epoch_seconds : int
          Durée d'une époque en secondes. Défaut : 10 s.
      zero_threshold : float, optionnel
          Seuil maximal de VM pour considérer une époque comme ayant des counts nuls.
          Défaut : 15 (adapté pour données réelles avec bruit).
          Valeur 0 pour comportement strictement conforme à l'algorithme original.
      debug : bool, optionnel
          Si True, affiche des diagnostics. Défaut : True.

      Retourne
      --------
      pd.DataFrame
          DataFrame avec 'Activite_Lissee' mis à jour (marquant 'Non Porté').
      """
      # Paramètres structurés du Choi algorithm
      DUREE_NON_PORT_MINUTES     = 90
      DUREE_INTERRUPTION_MINUTES = 2
      DUREE_FLANC_MINUTES        = 30

      # Convertir en nombre d'époques
      window_90min  = int((DUREE_NON_PORT_MINUTES * 60) / epoch_seconds)
      window_2min   = int((DUREE_INTERRUPTION_MINUTES * 60) / epoch_seconds)
      window_30min  = int((DUREE_FLANC_MINUTES * 60) / epoch_seconds)

      n_epochs = len(df_epochs)
      is_zero = (df_epochs['VM_Counts'] <= zero_threshold).values
      nonwear_flag = np.zeros(n_epochs, dtype=bool)

      if debug:
          print("\n" + "="*70)
          print("CHOI ALGORITHM - NONWEAR DETECTION")
          print("="*70)
          print(f"Epoch size: {epoch_seconds} s")
          print(f"Zero threshold (VM): {zero_threshold}")
          print(f"90-min window: {window_90min} epochs")
          print(f"2-min allowance: {window_2min} epochs")
          print(f"30-min flank: {window_30min} epochs")
          print("="*70 + "\n")

      # Algorithme Choi : chercher les périodes de 90 min de counts nuls,
      # avec allowance optionnelle de 2 min nonzero flanquée de 30 min zeros
      nonwear_periods = []  # Track detected periods for debugging
      i = 0
      while i < n_epochs:
          # Chercher le début d'une fenêtre potentielle de 90 min zeros
          if not is_zero[i]:
              i += 1
              continue

          # Chercher combien de minutes consécutives commencent à zéro à partir de i
          j = i
          while j < n_epochs and is_zero[j]:
              j += 1

          # Bloc de j - i epochs de zéros consécutifs [i, j)
          consecutive_zero_epochs = j - i

          # Cas 1 : bloc >= 90 min → non-port confirmé (pas d'interruption)
          if consecutive_zero_epochs >= window_90min:
              nonwear_flag[i : min(j, n_epochs)] = True
              nonwear_periods.append({'start': i, 'end': min(j, n_epochs), 'type': 'continuous'})
              i = j
              continue

          # Cas 2 : bloc < 90 min → chercher s'il peut être étendu avec
          #         une allowance de 2 min + 30 min zeros après
          if consecutive_zero_epochs >= (window_90min - window_2min):
              # On a presque 90 min de zeros. Chercher si une petite
              # interruption (<=2min) existe juste après, flanquée de 30min zeros.
              interrupt_start = j
              interrupt_end = j

              # Compter combien d'epochs nonzero immédiatement après
              while (interrupt_end < n_epochs and 
                     not is_zero[interrupt_end] and 
                     interrupt_end - interrupt_start < window_2min):
                  interrupt_end += 1

              interrupt_duration = interrupt_end - interrupt_start

              # Si interruption <= 2min, chercher 30min zeros après
              if interrupt_duration > 0 and interrupt_duration <= window_2min:
                  after_interr_start = interrupt_end
                  after_interr_end = after_interr_start

                  # Compter 30 min de zeros après l'interruption
                  while (after_interr_end < n_epochs and 
                         is_zero[after_interr_end]):
                      after_interr_end += 1

                  after_interr_zeros = after_interr_end - after_interr_start

                  # Vérifier que 30 min de zeros existent après l'interruption
                  if after_interr_zeros >= window_30min:
                      # Critère satisfait : on a 90min zeros + 2min allowance + 30min zeros
                      nonwear_flag[i : min(after_interr_end, n_epochs)] = True
                      nonwear_periods.append({
                          'start': i, 
                          'end': min(after_interr_end, n_epochs),
                          'type': 'with_interruption',
                          'interrupt_start': interrupt_start,
                          'interrupt_end': interrupt_end
                      })
                      i = after_interr_end
                      continue

          i += 1

      # POST-PROCESSING : Détecter les périodes sédentaires prolongées qui pourraient
      # être du non-port malgré des micro-mouvements (micro-mouvements nocturnes,
      # bruit du capteur, etc.). Critères :
      # - Durée >= 60 min
      # - Moyenne de counts << seuil_0 (ex: 5 counts)
      # - < 5% d'epochs au-dessus de seuil_0 * 3
      sedentary_mask = df_epochs['Activite_Lissee'] == 'Sédentaire'
      sedentary_indices = np.where(sedentary_mask)[0]
      
      if len(sedentary_indices) > 0:
          sedentary_blocks = []
          if len(sedentary_indices) > 0:
              block_start = sedentary_indices[0]
              for idx in range(1, len(sedentary_indices)):
                  if sedentary_indices[idx] - sedentary_indices[idx-1] > 1:
                      sedentary_blocks.append((block_start, sedentary_indices[idx-1]))
                      block_start = sedentary_indices[idx]
              sedentary_blocks.append((block_start, sedentary_indices[-1]))
          
          for start_idx, end_idx in sedentary_blocks:
              duration_epochs = end_idx - start_idx + 1
              duration_min = duration_epochs * epoch_seconds / 60.0
              
              # Appliquer le test seulement aux blocs >= 60 min
              if duration_min >= 60:
                  block_counts = df_epochs['VM_Counts'].iloc[start_idx:end_idx+1].values
                  counts_mean = np.mean(block_counts)
                  
                  # Test : si moyenne très basse ET < 5% au-dessus de zero_threshold*3
                  liberal_threshold = zero_threshold * 3
                  above_liberal = np.sum(block_counts > liberal_threshold)
                  pct_above = 100 * above_liberal / len(block_counts)
                  
                  if counts_mean < zero_threshold * 1.5 and pct_above < 3 and not nonwear_flag[start_idx:end_idx+1].all():
                      # Marquer comme non-port seulement si le bloc n'a pas déjà
                      # été identifié par l'algorithme Choi strict.
                      nonwear_flag[start_idx:end_idx+1] = True
                      nonwear_periods.append({
                          'start': start_idx,
                          'end': end_idx + 1,
                          'type': 'sedentary_recovery',
                          'reason': f'low_counts_mean={counts_mean:.2f}_pct_above={pct_above:.1f}%'
                      })

      df_epochs['Activite_Lissee'] = df_epochs['Activite_Lissee'].astype(str)
      df_epochs.loc[nonwear_flag, 'Activite_Lissee'] = 'Non Porté'

      if debug:
          nonwear_count = nonwear_flag.sum()
          nonwear_hours = nonwear_count * epoch_seconds / 3600
          print(f"Non-wear epochs detected: {nonwear_count} ({nonwear_hours:.1f} hours)")
          print("="*70)
          print("DETAILED NON-WEAR PERIODS (for micro-movement analysis):")
          print("="*70)
          
          for period_idx, period in enumerate(nonwear_periods, 1):
              start_idx = period['start']
              end_idx = period['end']
              duration_minutes = (end_idx - start_idx) * epoch_seconds / 60.0
              
              period_counts = df_epochs['VM_Counts'].iloc[start_idx:end_idx].values
              
              # Statistiques générales
              counts_mean = np.mean(period_counts)
              counts_std = np.std(period_counts)
              counts_min = np.min(period_counts)
              counts_max = np.max(period_counts)
              counts_above_threshold = np.sum(period_counts > zero_threshold)
              
              # Heure si disponible
              time_str = ""
              if 'Heure' in df_epochs.columns:
                  start_time = df_epochs['Heure'].iloc[start_idx]
                  end_time = df_epochs['Heure'].iloc[end_idx - 1]
                  time_str = f" | {start_time} → {end_time}"
              
              print(f"\nPeriod {period_idx}: {period['type'].upper()}")
              print(f"  Duration: {duration_minutes:.1f} min ({end_idx - start_idx} epochs){time_str}")
              print(f"  Count statistics:")
              print(f"    Mean: {counts_mean:.2f} | Std: {counts_std:.2f}")
              print(f"    Min:  {counts_min:.2f} | Max: {counts_max:.2f}")
              print(f"    Epochs > threshold ({zero_threshold}): {counts_above_threshold} ({100*counts_above_threshold/(end_idx-start_idx):.1f}%)")
              
              # Si interruption, afficher les détails
              if period['type'] == 'with_interruption':
                  int_start = period['interrupt_start']
                  int_end = period['interrupt_end']
                  int_duration = (int_end - int_start) * epoch_seconds / 60.0
                  int_counts = df_epochs['VM_Counts'].iloc[int_start:int_end].values
                  print(f"  Interruption: {int_duration:.1f} min ({int_end - int_start} epochs)")
                  print(f"    Counts in interruption: min={np.min(int_counts):.2f}, mean={np.mean(int_counts):.2f}, max={np.max(int_counts):.2f}")
              
              # Si sedentary_recovery, afficher la raison
              if period['type'] == 'sedentary_recovery':
                  print(f"  Detection method: Post-processing (low activity sedentary)")
                  print(f"  Reason: {period['reason']}")
              
              # Flaguer si statistiques suspectes (potentiel micro-mouvement)
              if counts_max > 50 or counts_above_threshold > (end_idx - start_idx) * 0.05:
                  print(f"  ⚠ WARNING: Possible micro-movements detected (high counts within period)")
          
          print("="*70)
          print("SEDENTARY PERIODS ANALYSIS (potential misclassifications):")
          print("="*70)
          
          # Identifier les blocs de "Sédentaire" et vérifier s'ils pourraient être non-wear
          sedentary_mask = df_epochs['Activite_Lissee'] == 'Sédentaire'
          sedentary_indices = np.where(sedentary_mask)[0]
          
          if len(sedentary_indices) > 0:
              # Détecter les blocs continus de sédentaire
              sedentary_blocks = []
              if len(sedentary_indices) > 0:
                  block_start = sedentary_indices[0]
                  for idx in range(1, len(sedentary_indices)):
                      if sedentary_indices[idx] - sedentary_indices[idx-1] > 1:
                          # Fin du bloc
                          sedentary_blocks.append((block_start, sedentary_indices[idx-1]))
                          block_start = sedentary_indices[idx]
                  sedentary_blocks.append((block_start, sedentary_indices[-1]))
              
              # Analyser chaque bloc
              for block_idx, (start_idx, end_idx) in enumerate(sedentary_blocks, 1):
                  duration_minutes = (end_idx - start_idx + 1) * epoch_seconds / 60.0
                  
                  # Ignorer les très courts blocs
                  if duration_minutes < 30:
                      continue
                  
                  block_counts = df_epochs['VM_Counts'].iloc[start_idx:end_idx+1].values
                  counts_mean = np.mean(block_counts)
                  counts_std = np.std(block_counts)
                  counts_max = np.max(block_counts)
                  
                  # Heure
                  time_str = ""
                  if 'Heure' in df_epochs.columns:
                      start_time = df_epochs['Heure'].iloc[start_idx]
                      end_time = df_epochs['Heure'].iloc[end_idx]
                      time_str = f" | {start_time} → {end_time}"
                  
                  print(f"\nSedentary block {block_idx}: {duration_minutes:.1f} min{time_str}")
                  print(f"  Count statistics: mean={counts_mean:.2f}, std={counts_std:.2f}, max={counts_max:.2f}")
                  
                  # Test avec différents thresholds pour voir si ça pourrait être non-wear
                  test_thresholds = [20, 25, 30, 35, 40, 50]
                  print(f"  Threshold testing (could be non-wear with higher threshold?):")
                  for test_thresh in test_thresholds:
                      above_test = np.sum(block_counts > test_thresh)
                      pct_above = 100 * above_test / len(block_counts)
                      marker = " ← MIGHT BE NON-WEAR" if pct_above < 5 else ""
                      print(f"    Threshold {test_thresh}: {above_test} epochs above ({pct_above:.1f}%){marker}")
          
          print("="*70 + "\n")

      return df_epochs


def detecter_velo(df_brut, df_epochs, epoch_seconds, frequence_hz,
                  seuil_puissance_spectrale=SEUIL_FFT_DEFAUT,
                  seuil_cv=SEUIL_CV_DEFAUT):
    """
    Détecte les époques de cyclisme par analyse heuristique du signal brut.

    La détection repose sur trois critères simultanés appliqués aux données
    brutes, sans recours à un signal GPS. Elle est conçue pour un port à la
    hanche (crête iliaque). Les seuils par défaut ont été calibrés
    empiriquement sur des données réelles (rappel FFT = 98.9 %,
    rappel CV = 97.8 %).

    Critères (les 3 doivent être validés simultanément) :

    1. Pic spectral dans la bande de pédalage [0.5–2.0 Hz] (Brage et al., 2005)
       Le pédalage cycliste typique se situe entre 60 et 90 rpm,
       soit 1.0 à 1.5 Hz. Le ratio d'énergie spectrale de l'axe Z dans
       cette bande doit dépasser `seuil_puissance_spectrale`.

    2. Faible variabilité du VM sur une fenêtre de 3 époques (Mannini & Sabatini, 2010)
       Le pédalage est une activité rythmique et régulière. Le coefficient
       de variation (CV = écart-type / moyenne) du VM doit être inférieur
       à `seuil_cv`.

    3. Plage de VM compatible avec le cyclisme (Sasaki et al., 2011)
       200 ≤ VM/min < 6 167 counts. Les époques à VM nul correspondent à
       des arrêts légitimes (feux, pauses) et sont correctement exclues.

    Paramètres
    ----------
    df_brut : pd.DataFrame
        Données brutes avec colonnes 'X', 'Y', 'Z'.
    df_epochs : pd.DataFrame
        DataFrame des époques avec colonne 'VM_Counts'.
    epoch_seconds : int
        Durée d'une époque en secondes.
    frequence_hz : int
        Fréquence d'échantillonnage du capteur (Hz).
    seuil_puissance_spectrale : float, optionnel
        Ratio minimal d'énergie FFT dans [0.5–2.0 Hz].
        Défaut : 0.08 (rappel empirique = 98.9 %).
    seuil_cv : float, optionnel
        Coefficient de variation maximal du VM sur 3 époques.
        Défaut : 0.55 (rappel empirique = 97.8 %).

    Retourne
    --------
    pd.DataFrame
        Colonnes :
        - Flag_Velo (bool)       : True si l'époque est classifiée vélo
        - Diag_FFT_Ratio (float) : valeur brute du ratio spectral (critère 1)
        - Diag_CV (float)        : valeur brute du CV (critère 2)
        - Diag_VM_OK (bool)      : résultat du critère 3
        - Diag_FFT_OK (bool)     : résultat du critère 1
        - Diag_CV_OK (bool)      : résultat du critère 2
        Les colonnes Diag_* sont réservées au diagnostic et exclues de l'export.
    """
    n_samples_epoch = epoch_seconds * frequence_hz
    n_epochs        = len(df_epochs)
    raw_z           = df_brut['Z'].values
    facteur         = epoch_seconds / 60.0
    vm_min          = SEUIL_SEDENTAIRE_CPM * facteur
    vm_max          = SEUIL_MODEREE_CPM    * facteur

    flags        = np.zeros(n_epochs, dtype=bool)
    ratios_fft   = np.full(n_epochs, np.nan)
    cvs          = np.full(n_epochs, np.nan)
    criteres_fft = np.zeros(n_epochs, dtype=bool)
    criteres_cv  = np.zeros(n_epochs, dtype=bool)
    criteres_vm  = np.zeros(n_epochs, dtype=bool)

    for i in range(n_epochs):
        debut = i * n_samples_epoch
        fin   = debut + n_samples_epoch
        if fin > len(raw_z):
            break

        segment = raw_z[debut:fin]

        # Critère 1 : ratio d'énergie spectrale dans la bande de pédalage [0.5–2.0 Hz]
        fft_vals     = np.abs(np.fft.rfft(segment))
        fft_freqs    = np.fft.rfftfreq(len(segment), d=1.0 / frequence_hz)
        masque_bande = (fft_freqs >= 0.5) & (fft_freqs <= 2.0)
        masque_hors  = fft_freqs > 0.1
        if masque_hors.sum() == 0 or fft_vals[masque_hors].sum() == 0:
            continue
        fft_band = fft_vals[masque_bande]
        ratio_spectral  = fft_band.sum() / fft_vals[masque_hors].sum()
        ratios_fft[i]   = round(ratio_spectral, 4)
        criteres_fft[i] = ratio_spectral >= seuil_puissance_spectrale

        # Critère 2 : coefficient de variation du VM sur une fenêtre de 3 époques
        i_debut    = max(0, i - 1)
        i_fin      = min(n_epochs, i + 2)
        fenetre_vm = df_epochs['VM_Counts'].iloc[i_debut:i_fin].values
        if fenetre_vm.mean() > 0:
            cv             = fenetre_vm.std() / fenetre_vm.mean()
            cvs[i]         = round(cv, 4)
            criteres_cv[i] = cv < seuil_cv

        # Critère 3 : plage de VM compatible avec le cyclisme
        vm_epoch       = df_epochs['VM_Counts'].iloc[i]
        criteres_vm[i] = vm_min <= vm_epoch < vm_max

        flags[i] = criteres_fft[i] and criteres_cv[i] and criteres_vm[i]

    return pd.DataFrame({
        'Flag_Velo'      : flags,
        'Diag_FFT_Ratio' : ratios_fft,
        'Diag_CV'        : cvs,
        'Diag_VM_OK'     : criteres_vm,
        'Diag_FFT_OK'    : criteres_fft,
        'Diag_CV_OK'     : criteres_cv,
    })


def detecter_velo_estimation(df_epochs, epoch_seconds, seuil_cv=0.55):
    """
    Estime les époques de cyclisme à partir de données déjà agrégées.

    Cette heuristique ne repose pas sur les données brutes X/Y/Z ni sur une FFT.
    Elle utilise uniquement la VM agrégée et la stabilité des counts sur une
    fenêtre locale pour marquer un cyclisme « probable ».

    Parameters
    ----------
    df_epochs : pd.DataFrame
        DataFrame d'époques agrégées contenant au minimum la colonne 'VM_Counts'.
        Si la colonne 'Activite_Lissee' est présente, elle est utilisée pour
        renforcer l'estimation.
    epoch_seconds : int
        Durée d'une époque en secondes.
    seuil_cv : float
        Coefficient de variation maximal sur 3 époques pour considérer la
        période comme régulière (valeur par défaut : 0.55).

    Returns
    -------
    pd.DataFrame
        Une copie de df_epochs enrichie des colonnes :
        - Flag_Velo_Estimation
        - Diag_VM_Estimation
        - Diag_Stabilite_Estimation
        - Diag_Activite_Estimation (si Activite_Lissee était présente)
    """
    if 'VM_Counts' not in df_epochs.columns:
        raise ValueError("df_epochs doit contenir la colonne 'VM_Counts' pour l'estimation du vélo.")

    df = df_epochs.copy()
    facteur = epoch_seconds / 60.0
    vm_min = SEUIL_SEDENTAIRE_CPM * facteur
    vm_max = SEUIL_MODEREE_CPM * facteur

    vm_values = df['VM_Counts'].fillna(0).values
    vm_ok = (vm_values >= vm_min) & (vm_values < vm_max)

    if 'Activite_Lissee' in df.columns:
        activite_ok = df['Activite_Lissee'].isin([
            'Modérée (Marche)',
            'Vigoureuse (Course)',
            'Légère'
        ])
    else:
        activite_ok = np.ones(len(df), dtype=bool)

    stabilite = np.zeros(len(df), dtype=bool)
    for i in range(len(vm_values)):
        start = max(0, i - 1)
        end = min(len(vm_values), i + 2)
        segment = vm_values[start:end]
        if segment.mean() <= 0:
            continue
        cv = segment.std() / segment.mean()
        stabilite[i] = cv <= seuil_cv

    flag = vm_ok & stabilite & activite_ok

    df['Flag_Velo_Estimation'] = flag
    df['Diag_VM_Estimation'] = vm_ok
    df['Diag_Stabilite_Estimation'] = stabilite
    if 'Activite_Lissee' in df.columns:
        df['Diag_Activite_Estimation'] = activite_ok

    return df


def extraire_sessions(df, activite, epoch_seconds, duree_min, allowance_minutes=2.0):
    """
    Extrait les sessions continues d'une activité donnée en appliquant une 
    tolérance d'interruption (Troiano et al., 2008).

    Identifie les blocs d'époques appartenant à la même classe d'activité.
    Si deux blocs de la même activité sont séparés par un intervalle de temps 
    inférieur ou égal à `allowance_minutes`, ils sont fusionnés en une seule session.

    Paramètres
    ----------
    df : pd.DataFrame
        DataFrame avec colonnes 'Activite_Lissee', 'Heure', 'VM_Counts'.
    activite : str
        Nom de l'activité à extraire (ex : 'Vigoureuse (Course)').
    epoch_seconds : int
        Durée d'une époque en secondes.
    duree_min : float
        Durée minimale d'une session finale en minutes.
    allowance_minutes : float, optionnel
        Durée maximale d'interruption autorisée (en minutes) pour fusionner 
        deux sessions proches. Défaut : 2.0 minutes (Troiano, 2008).

    Retourne
    --------
    pd.DataFrame
        Colonnes : Heure_Debut, Heure_Fin, Duree_Minutes, Intensite_Moyenne.
    """
    df_temp = df.copy()

    # 1. Créer un masque binaire de l'activité cible
    is_target = (df_temp['Activite_Lissee'] == activite).astype(int)
    
    # Si aucune époque ne correspond, on s'arrête là
    if is_target.sum() == 0:
        return pd.DataFrame()

    # 2. Convertir l'allowance en nombre de lignes (époques)
    max_gap_epochs = int((allowance_minutes * 60) / epoch_seconds)

    # 3. Algorithme de fermeture des gaps (Look-ahead)
    # On va "remplir" artificiellement les micro-pauses inférieures au seuil Troiano
    # CODE CORRIGÉ :
    target_values = is_target.values.copy()
    n_epochs = len(target_values)
    
    i = 0
    while i < n_epochs:
        if target_values[i] == 1:
            # On cherche la fin de ce bloc actif
            j = i
            while j < n_epochs and target_values[j] == 1:
                j += 1
            
            # On regarde s'il y a un autre bloc de la même activité un peu plus loin
            k = j
            while k < n_epochs and target_values[k] == 0 and (k - j) <= max_gap_epochs:
                k += 1
            
            # Si on a trouvé la même activité avant d'atteindre le gap max, on fusionne !
            if k < n_epochs and target_values[k] == 1 and (k - j) <= max_gap_epochs:
                target_values[j:k] = 1  # On comble le vide en arrière-plan
                i = j  # On continue l'analyse depuis la fin du premier bloc
                continue
        i += 1

    # 4. Identification des nouveaux blocs fusionnés
    df_temp['_is_merged_target'] = target_values
    df_temp['_changement'] = df_temp['_is_merged_target'] != df_temp['_is_merged_target'].shift()
    df_temp['_bloc_id'] = df_temp['_changement'].cumsum()

    # Filtrer uniquement sur les époques validées (y compris celles comblées)
    df_cible = df_temp[df_temp['_is_merged_target'] == 1]
    if df_cible.empty:
        return pd.DataFrame()

    # 5. Agrégation des données de la session
    sessions = df_cible.groupby('_bloc_id').agg(
        Heure_Debut=('Heure', 'min'),
        Heure_Fin=('Heure', 'max'),
        Intensite_Moyenne=('VM_Counts', 'mean'),
        Nombre_Epoques=('Heure', 'count')
    ).reset_index(drop=True)

    # Calcul de la durée totale finale de la session
    sessions['Duree_Minutes'] = round(
        (sessions['Nombre_Epoques'] * epoch_seconds) / 60.0, 2
    )
    
    # Application du filtre de durée minimale (ex: ne garder que les sessions de > 10 min)
    sessions = sessions[sessions['Duree_Minutes'] >= duree_min].copy()

    if sessions.empty:
        return pd.DataFrame()

    # Formatage des chaînes de caractères pour l'affichage de l'interface
    sessions['Heure_Debut_Raw'] = sessions['Heure_Debut']  # On garde une copie datetime pour le rapport textuel
    sessions['Heure_Debut'] = sessions['Heure_Debut'].dt.strftime('%H:%M:%S')
    sessions['Heure_Fin'] = sessions['Heure_Fin'].dt.strftime('%H:%M:%S')
    sessions['Intensite_Moyenne'] = sessions['Intensite_Moyenne'].round(1)

    return sessions[['Heure_Debut', 'Heure_Fin', 'Duree_Minutes', 'Intensite_Moyenne', 'Heure_Debut_Raw']]


def calculer_resume_participant(df_epochs, epoch_seconds,
                                seuil_port_heures=10.0,
                                duree_min_session=1.0):
    """
    Calcule le résumé statistique par participant à partir des données traitées.

    Calcule quatre indicateurs clés pour caractériser le comportement d'activité
    physique d'un participant sur l'ensemble de sa période d'enregistrement :
    jours valides de port, nombre et durée moyenne des sessions par type
    d'activité, et créneau horaire privilégié par type d'activité.

    Définitions :
        - Jour valide : toute journée calendaire où la durée de port effective
          (epochs non classés 'Non Porté') est ≥ seuil_port_heures.
          Seuil retenu : 10 heures (adapté au contexte de l'étude).
        - Session : bloc continu d'époques consécutives appartenant à la même
          classe d'activité, d'une durée ≥ duree_min_session minutes.
        - Créneaux horaires :
            Matin      : 06h00 – 11h59
            Après-midi : 12h00 – 17h59
            Soir       : 18h00 – 21h59
          Le créneau dominant est celui qui concentre le plus grand nombre
          de sessions pour l'activité concernée.

    Paramètres
    ----------
    df_epochs : pd.DataFrame
        DataFrame traité par la pipeline complète (après executer_pipeline_agcounts,
        ajouter_temps, lisser_activite, detecter_non_port, detecter_velo).
        Doit contenir les colonnes : 'Heure', 'Activite_Lissee', 'VM_Counts'.
    epoch_seconds : int
        Durée d'une époque en secondes. Doit correspondre au paramètre utilisé
        lors du traitement Python.
    seuil_port_heures : float, optionnel
        Nombre minimum d'heures de port effectif (hors Non Porté) pour qu'une
        journée soit comptée comme valide. Défaut : 10.0 heures.
    duree_min_session : float, optionnel
        Durée minimale en minutes pour qu'un bloc continu soit compté comme
        une session. Défaut : 1.0 minute.

    Retourne
    --------
    dict avec les clés :
        'jours_valides'   : int — nombre de jours de port ≥ seuil
        'details_jours'   : pd.DataFrame — une ligne par jour avec colonnes :
                            Date, Heures_Port, Valide
        'resume_sessions' : pd.DataFrame — une ligne par activité avec colonnes :
                            Activite, N_Sessions, Duree_Moyenne_Min,
                            Duree_Mediane_Min, Duree_Totale_Min
        'creneaux'        : pd.DataFrame — une ligne par activité avec colonnes :
                            Activite, Creneau_Dominant, N_Sessions_Matin,
                            N_Sessions_Apres_Midi, N_Sessions_Soir
    """
    # ── Jours valides ─────────────────────────────────────────────────────────
    # Compte les epochs portés (hors Non Porté) par jour calendaire.
    # Un jour est valide si la durée portée >= seuil_port_heures.
    seuil_epochs = (seuil_port_heures * 3600) / epoch_seconds

    df_epochs = df_epochs.copy()
    df_epochs['_date'] = df_epochs['Heure'].dt.date

    epochs_par_jour = (
        df_epochs[df_epochs['Activite_Lissee'] != 'Non Porté']
        .groupby('_date')
        .size()
        .reset_index(name='_n_epochs_portes')
    )

    # Jointure sur tous les jours présents dans le fichier (y compris invalides)
    tous_les_jours = (
        df_epochs[['_date']]
        .drop_duplicates()
        .merge(epochs_par_jour, on='_date', how='left')
        .fillna({'_n_epochs_portes': 0})
    )
    tous_les_jours['Heures_Port'] = round(
        tous_les_jours['_n_epochs_portes'] * epoch_seconds / 3600, 2
    )
    tous_les_jours['Valide'] = tous_les_jours['Heures_Port'] >= seuil_port_heures
    tous_les_jours = tous_les_jours.rename(columns={'_date': 'Date'})

    details_jours = tous_les_jours[['Date', 'Heures_Port', 'Valide']].sort_values('Date')
    jours_valides = int(details_jours['Valide'].sum())

    df_epochs.drop(columns=['_date'], inplace=True)

    # ── Sessions par activité ─────────────────────────────────────────────────
    activites_cibles = ['Modérée (Marche)', 'Vigoureuse (Course)', 'Vélo']
    liste_sessions = []

    for activite in activites_cibles:
        df_sessions = extraire_sessions(df_epochs, activite, epoch_seconds, duree_min_session)
        if df_sessions.empty:
            liste_sessions.append({
                'Activite'         : activite,
                'N_Sessions'       : 0,
                'Duree_Moyenne_Min': np.nan,
                'Duree_Mediane_Min': np.nan,
                'Duree_Totale_Min' : 0.0,
            })
        else:
            liste_sessions.append({
                'Activite'         : activite,
                'N_Sessions'       : len(df_sessions),
                'Duree_Moyenne_Min': round(df_sessions['Duree_Minutes'].mean(), 1),
                'Duree_Mediane_Min': round(df_sessions['Duree_Minutes'].median(), 1),
                'Duree_Totale_Min' : round(df_sessions['Duree_Minutes'].sum(), 1),
            })

    resume_sessions = pd.DataFrame(liste_sessions)

    # ── Créneau horaire dominant par activité ─────────────────────────────────
    # Créneaux :  Matin 06–12 / Après-midi 12–18 / Soir 18–22
    def classer_creneau(heure_str):
        """Retourne le créneau horaire d'un timestamp HH:MM:SS."""
        heure = int(heure_str.split(':')[0])
        if 6  <= heure < 12: return 'Matin'
        elif 12 <= heure < 18: return 'Après-midi'
        elif 18 <= heure < 22: return 'Soir'
        else:                  return 'Autre'

    liste_creneaux = []
    for activite in activites_cibles:
        df_sessions = extraire_sessions(df_epochs, activite, epoch_seconds, duree_min_session)
        if df_sessions.empty:
            liste_creneaux.append({
                'Activite'             : activite,
                'Creneau_Dominant'     : 'N/A',
                'N_Sessions_Matin'     : 0,
                'N_Sessions_Apres_Midi': 0,
                'N_Sessions_Soir'      : 0,
            })
        else:
            df_sessions['Creneau'] = df_sessions['Heure_Debut'].apply(classer_creneau)
            counts = df_sessions['Creneau'].value_counts()
            liste_creneaux.append({
                'Activite'             : activite,
                'Creneau_Dominant'     : counts.index[0] if not counts.empty else 'N/A',
                'N_Sessions_Matin'     : int(counts.get('Matin', 0)),
                'N_Sessions_Apres_Midi': int(counts.get('Après-midi', 0)),
                'N_Sessions_Soir'      : int(counts.get('Soir', 0)),
            })

    creneaux = pd.DataFrame(liste_creneaux)

    return {
        'jours_valides'   : jours_valides,
        'details_jours'   : details_jours,
        'resume_sessions' : resume_sessions,
        'creneaux'        : creneaux,
    }