"""
================================================================================
app_euromov.py — Streamlit Application Entry Point — EuroMov DHM Lab
================================================================================
FAIR4RS Compliance (Chue Hong et al., 2021 — https://doi.org/10.15497/RDA00068)
  Findable     : Unique identifier via CITATION.cff / GitHub release tag v1.0.0
  Accessible   : MIT Licence — open access, no authentication required
  Interoperable: Standard Streamlit app; data I/O via CSV and .agd (SQLite)
  Reusable     : Versioned, documented; UI layer separated from pipeline logic

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
    Streamlit application entry point. This file exclusively handles the
    user interface (sidebar, tabs, charts, export).
    All algorithmic logic is delegated to the pipeline.py module.

Scientific References (full details in pipeline.py and CITATION.cff)
---------------------------------------------------------------------
    [1] Sasaki et al. (2011). J Sci Med Sport, 14(5), 411-416.
        https://doi.org/10.1016/j.jsams.2011.01.003
    [2] Aguilar-Farías et al. (2014). J Sci Med Sport, 17(3), 293-299.
        https://doi.org/10.1016/j.jsams.2013.07.002
    [3] Choi et al. (2011). Med Sci Sports Exerc, 43(2), 357-364.
        https://doi.org/10.1249/MSS.0b013e3181ed61a3
    [4] Brage et al. (2005). J Appl Physiol, 98(1), 166-173.
        https://doi.org/10.1152/japplphysiol.00510.2004
    [5] Mannini & Sabatini (2010). Sensors, 10(2), 1154-1175.
        https://doi.org/10.3390/s100201154
    [6] Cohen, J. (1960). A coefficient of agreement for nominal scales.
        Educational and Psychological Measurement, 20(1), 37-46.
        https://doi.org/10.1177/001316446002000104
    [7] Landis, J.R., & Koch, G.G. (1977). The measurement of observer
        agreement for categorical data. Biometrics, 159-174.
        https://doi.org/10.2307/2529310

Usage
-----
    streamlit run app_euromov.py
================================================================================
"""

import gc
import os
import hashlib
import tempfile
import time
import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import mannwhitneyu

from pipeline import (
    ajouter_temps,
    calculer_resume_participant,
    detecter_non_port,
    detecter_velo,
    executer_pipeline_agcounts,
    extraire_sessions,
    lisser_activite,
    read_agd_and_aggregate,
)

try:
    from annotation_utils import (
        ajouter_annotation,
        creer_df_annotations,
        calculer_metriques_correspondance,
        generer_df_confusion_matrix,
        generer_rapport_correspondance,
    )
except ImportError:
    # Fallback if scikit-learn is not installed
    from annotation_utils_lite import (
        ajouter_annotation,
        creer_df_annotations,
        calculer_metriques_correspondance,
        generer_df_confusion_matrix,
        generer_rapport_correspondance,
    )

from figure_export import generer_toutes_figures, FIGURES_DIR, PALETTE as FIG_PALETTE

# ==============================================================================
# PAGE CONFIGURATION
# ==============================================================================

st.set_page_config(page_title="EuroMov Accelerometry", layout="wide")
st.title("🏃‍♂️ Accelerometry Analysis — EuroMov DHM Lab Pipeline")

# ==============================================================================
# SIDEBAR — Settings
# ==============================================================================

with st.sidebar:
    st.header("⚙️ Settings")

    fichier_upload = st.file_uploader("Upload raw CSV or .agd file", type=["csv", "agd"]) 
    epoch_choisi = st.slider("Epoch size (seconds)", 10, 60, 10, 5)
    frequence_hz = st.selectbox(
        "Sampling rate (Hz)",
        options=[30, 40, 50, 60, 70, 80, 90, 100],
        index=3,
        help="Sampling rate of your wGT3X-BT device. Check this parameter in your recording settings."
    )
    fichier_agrege = st.checkbox(
        "File is already aggregated into epochs",
        value=False,
        help=(
            "Check if your CSV file already contains counts per epoch "
            "(e.g., AxisX/AxisY/AxisZ for 10-second epochs)."
        )
    )
    date_enregistrement = st.date_input(
        "Recording date",
        value=pd.to_datetime("2025-10-03")
    )

    st.divider()
    st.subheader("🚴 Cycling Detection")
    st.caption("GPS-free heuristic — calibrated for hip placement")
    activer_velo = st.toggle(
        "Enable cycling detection",
        value=True,
        disabled=fichier_agrege,
        help="Disabled if file is already aggregated (raw data required for FFT analysis)."
    )

    with st.expander("Advanced parameters"):
        st.caption(
            "Default values empirically calibrated on real-world data "
            "(hip placement, wGT3X-BT, 60 Hz)."
        )
        seuil_spectral = st.slider(
            "Spectral power threshold (FFT)",
            min_value=0.05, max_value=0.40, value=0.08, step=0.01,
            help=(
                "Minimum share of energy in the 0.5–2.0 Hz band (pedaling cadence). "
                "Default: 0.08 → sensitivity = 98.9%. Increase for stricter filtering."
            )
        )
        seuil_cv = st.slider(
            "Variability threshold (CV)",
            min_value=0.10, max_value=0.80, value=0.55, step=0.05,
            help=(
                "Coefficient of variation of the Vector Magnitude (VM) over 3 consecutive epochs. "
                "Default: 0.55 → sensitivity = 97.8%. Decrease for stricter filtering."
            )
        )

# ==============================================================================
# MAIN PROCESSING
# ==============================================================================

if fichier_upload:
    zone_chargement = st.empty()
    with zone_chargement.container():
        st.write("### 🔄 Processing data...")
        barre = st.progress(0)

    filename = getattr(fichier_upload, 'name', '')

    # If user uploaded an .agd (SQLite) file, precompute all days and allow navigation
    if filename.lower().endswith('.agd'):
        file_bytes = fichier_upload.read()
        file_hash = hashlib.sha256(file_bytes).hexdigest()
        cache_key = f'days_{file_hash}'
        if cache_key not in st.session_state:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.agd')
            try:
                tmp.write(file_bytes)
                tmp.flush()
                tmp.close()
                days = read_agd_and_aggregate(tmp.name, epoch_seconds=epoch_choisi)
            finally:
                for attempt in range(3):
                    try:
                        os.unlink(tmp.name)
                        break
                    except PermissionError:
                        gc.collect()
                        time.sleep(0.1)
            st.session_state[cache_key] = days
            st.session_state['agd_days'] = days
        else:
            st.session_state['agd_days'] = st.session_state[cache_key]
        days = st.session_state.get(cache_key, {})

        if not days:
            st.error('No data could be extracted from the file.')
            zone_chargement.empty()
            st.stop()

        available_days = sorted(days.keys())
        selected_day = st.selectbox('Select day to view', available_days)
        df_res = days[selected_day]
        df_brut = None
        fichier_agrege = True
    else:
        if 'agd_days' in st.session_state:
            del st.session_state['agd_days']
        df_brut = pd.read_csv(fichier_upload)
        df_res  = executer_pipeline_agcounts(
            df_brut,
            epoch_choisi,
            frequence_hz,
            deja_agrege=fichier_agrege
        )
        df_res  = ajouter_temps(
            df_brut,
            df_res,
            epoch_choisi,
            date_enregistrement,
            deja_agrege=fichier_agrege
        )
        df_res  = lisser_activite(df_res, epoch_choisi)
        df_res  = detecter_non_port(df_res, epoch_choisi)

    # Map French internal database activity labels to translated English display labels if necessary
    # Note: If your underlying pipeline scripts output variables in French, we handle them safely here
    for col in ['Activite_Lissee', 'Activite_Annotee']:
        if col in df_res.columns:
            df_res[col] = df_res[col].replace({
                'Non Porté': 'Non-Wear',
                'Sédentaire': 'Sedentary',
                'Légère': 'Light',
                'Vélo': 'Cycling',
                'Modérée (Marche)': 'Moderate (Walking)',
                'Vigoureuse (Course)': 'Vigorous (Running)'
            })

    # Cycling detection — direct integration into Activite_Lissee
    colonnes_diag = ['Diag_FFT_Ratio', 'Diag_CV', 'Diag_VM_OK', 'Diag_FFT_OK', 'Diag_CV_OK']
    if activer_velo and not fichier_agrege:
        df_velo_diag = detecter_velo(
            df_brut, df_res, epoch_choisi, frequence_hz,
            seuil_puissance_spectrale=seuil_spectral,
            seuil_cv=seuil_cv
        )
        for col in df_velo_diag.columns:
            df_res[col] = df_velo_diag[col].values
        df_res.loc[df_res['Flag_Velo'], 'Activite_Lissee'] = 'Cycling'
    else:
        if activer_velo and fichier_agrege:
            st.warning(
                "Cycling detection is unavailable for pre-aggregated files "
                "because raw data is required for FFT spectral analysis."
            )
        df_res['Flag_Velo'] = False
        for col in colonnes_diag:
            df_res[col] = np.nan

    # Clear the processing placeholder once done
    zone_chargement.empty()

    df_actif = df_res[df_res['Activite_Lissee'] != 'Non-Wear'].copy()

    ORDRE_ACTIVITES = [
        'Non-Wear', 'Sedentary', 'Light',
        'Cycling', 'Moderate (Walking)', 'Vigorous (Running)'
    ]

    # Fixed palette mapping per activity for visual coherence across plots
    PALETTE = {
        'Non-Wear': '#b3b3b3',            # Neutral grey
        'Sedentary': '#66c2a5',           # Teal
        'Light': '#fc8d62',               # Coral orange
        'Cycling': '#8da0cb',             # Lavender blue
        'Moderate (Walking)': '#e78ac3',  # Pink
        'Vigorous (Running)': '#a6d854',  # Lime green
        'Not annotated': 'white'          # Pure white for unannotated periods
    }
    COULEURS_ORDRE = [PALETTE[a] for a in ORDRE_ACTIVITES]

    # ==============================================================================
    # TABS AND CHARTS DEFINITION (WITH UNIQUE KEYS)
    # ==============================================================================
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📊 Time Profile", "🎯 Global Summary", "🧪 Statistics",
        "📋 Session Details", "🚴 Cycling Detected", "🔬 Cycling Diagnosis",
        "📁 Participant Summary", "🖼️ Figure Export"
    ])

    # ──────────────────────────────────────────────────────────────────
    # TAB 1: TIME PROFILE
    # ──────────────────────────────────────────────────────────────────
    with tab1:
        # CRITICAL FIX: Initialize session state variable immediately at the start of Tab 1
        if 'annotations' not in st.session_state:
            st.session_state.annotations = []

        st.subheader("Daily Activity Dynamics")
        precision = st.select_slider(
            "Time resolution", options=["10min", "30min", "1h"], value="30min", key="slider_precision_t1"
        )
        
        # CHART 1: AUTOMATED PIPELINE TIMELINE
        df_groupe = (
            df_res.copy().set_index('Heure')
            .groupby([pd.Grouper(freq=precision), 'Activite_Lissee'])
            .size().reset_index(name='Count')
        )
        total_par_bin            = df_groupe.groupby('Heure')['Count'].transform('sum')
        df_groupe['Percentage'] = (df_groupe['Count'] / total_par_bin) * 100
        df_groupe                = df_groupe.sort_values('Heure')
        df_groupe['Time_Label'] = df_groupe['Heure'].dt.strftime('%H:%M')
        ordre_labels             = df_groupe['Time_Label'].drop_duplicates().tolist()

        fig_bar = px.bar(
            df_groupe, x='Time_Label', y='Percentage', color='Activite_Lissee',
            color_discrete_map=PALETTE,
            category_orders={
                "Activite_Lissee": ORDRE_ACTIVITES,
                "Time_Label": ordre_labels
            },
            labels={'Time_Label': 'Time', 'Percentage': '% of Time', 'Activite_Lissee': 'Activity'}
        )
        fig_bar.update_traces(marker_line_width=1, marker_line_color="black")
        fig_bar.update_layout(barmode='stack', yaxis_range=[0, 100], hovermode="x unified", title="Automated Pipeline Classifications")
        st.plotly_chart(fig_bar, use_container_width=True, key="chart_t1_pipeline_bars")
        
        # CHART 2: MANUALLY LOGGED TIMELINE
        df_annot = None
        if st.session_state.annotations:
            # (Rest of your code for Chart 2 stays exactly the same...)
            raw_annotations_input = []
            for item in st.session_state.annotations:
                vals = list(item.values())
                raw_annotations_input.append({'Heure_Debut': vals[0], 'Heure_Fin': vals[1], 'Activite_Annotee': vals[2]})

            df_annot = creer_df_annotations(raw_annotations_input, df_res, epoch_choisi)
            if 'Activite_Annotee' in df_annot.columns:
                df_annot['Activite_Annotee'] = df_annot['Activite_Annotee'].replace({
                    'Non annoté': 'Not annotated', 'Sédentaire': 'Sedentary', 'Légère': 'Light',
                    'Vélo': 'Cycling', 'Modérée (Marche)': 'Moderate (Walking)', 'Vigoureuse (Course)': 'Vigorous (Running)'
                })

            df_tmp = df_annot.copy()
            df_tmp['Heure'] = df_res['Heure'].values

            df_groupe_annot = (
                df_tmp
                .set_index('Heure')
                .groupby([pd.Grouper(freq=precision), 'Activite_Annotee'])
                .size()
                .reset_index(name='Count')
            )
            
            total_par_bin_annot = df_groupe_annot.groupby('Heure')['Count'].transform('sum')
            df_groupe_annot['Percentage'] = (df_groupe_annot['Count'] / total_par_bin_annot) * 100
            df_groupe_annot = df_groupe_annot.sort_values('Heure')
            df_groupe_annot['Time_Label'] = df_groupe_annot['Heure'].dt.strftime('%H:%M')
            ordre_labels_annot = df_groupe_annot['Time_Label'].drop_duplicates().tolist()
            
            fig_bar_annot = px.bar(
                df_groupe_annot, x='Time_Label', y='Percentage', color='Activite_Annotee',
                color_discrete_map=PALETTE,
                category_orders={
                    "Activite_Annotee": ORDRE_ACTIVITES,
                    "Time_Label": ordre_labels_annot
                },
                labels={'Time_Label': 'Time', 'Percentage': '% of Time', 'Activite_Annotee': 'Activity'}
            )
            fig_bar_annot.update_traces(marker_line_width=1, marker_line_color="black")
            fig_bar_annot.update_layout(barmode='stack', yaxis_range=[0, 100], hovermode="x unified", title="Manually Logged Activity Overlays")
            st.plotly_chart(fig_bar_annot, use_container_width=True, key="chart_t1_manual_overlay_bars")

        # MANUAL ANNOTATION INTERFACE FORMS
        st.divider()
        st.subheader("✏️ Manual Activity Annotation Log Interface")
        st.caption("Input verification tags below to validate the matching statistics baseline:")
        
        col_debut, col_duree, col_act, col_btn = st.columns([2, 2, 4, 2])
        with col_debut:
            heure_debut_txt = st.text_input("🕐 Start Time (HH:MM)", value="09:00", key="heure_debut_txt")
            try:
                h_deb, m_deb = map(int, heure_debut_txt.split(':'))
                heure_debut = datetime.time(hour=h_deb, minute=m_deb)
                erreur_format_debut = False
            except ValueError:
                erreur_format_debut = True
                st.caption("⚠️ Required format: HH:MM")
        
        with col_duree:
            duree_txt = st.text_input("⏳ Duration (HH:MM)", value="01:00", key="input_duree_txt")
            try:
                h_dur, m_dur = map(int, duree_txt.split(':'))
                delta_duree = datetime.timedelta(hours=h_dur, minutes=m_dur)
                erreur_format_duree = False
            except ValueError:
                erreur_format_duree = True
                st.caption("⚠️ Required format: HH:MM")
        
        with col_act:
            activite_annotee = st.selectbox(
                "🏃 Activity to log",
                options=['Sedentary', 'Light', 'Cycling', 'Moderate (Walking)', 'Vigorous (Running)'],
                key="input_activite_select"
            )
        
        with col_btn:
            st.write("<br>", unsafe_allow_html=True)
            bouton_clique = st.button("➕ Add Log", use_container_width=True, key="btn_add_annotation")
            
        if bouton_clique:
            if erreur_format_debut or erreur_format_duree:
                st.error("❌ Cannot add log: Input formatting is incorrect.")
            else:
                datetime_debut = datetime.datetime.combine(datetime.date.today(), heure_debut)
                datetime_fin = datetime_debut + delta_duree
                heure_fin = datetime_fin.time()
                heure_debut_str = heure_debut.strftime('%H:%M')
                heure_fin_str = heure_fin.strftime('%H:%M')
                
                if datetime_fin > datetime_debut:
                    fr_map = {'Sedentary': 'Sédentaire', 'Light': 'Légère', 'Cycling': 'Vélo', 'Moderate (Walking)': 'Modérée (Marche)', 'Vigorous (Running)': 'Vigoureuse (Course)'}
                    st.session_state.annotations = ajouter_annotation(st.session_state.annotations, heure_debut_str, heure_fin_str, fr_map.get(activite_annotee, activite_annotee))
                    st.success(f"✅ Successfully added: {activite_annotee} from {heure_debut_str} to {heure_fin_str}")
                    st.rerun()
                else:
                    st.error("❌ Activity duration must be greater than 00:00.")

        if st.session_state.annotations and df_annot is not None:
            if "editor_annotations" in st.session_state:
                state_editor = st.session_state.editor_annotations
                if "edited_rows" in state_editor and state_editor["edited_rows"]:
                    for idx_ligne, colonnes_modifiees in state_editor["edited_rows"].items():
                        for nom_colonne, nouvelle_valeur in colonnes_modifiees.items():
                            st.session_state.annotations[idx_ligne][nom_colonne] = nouvelle_valeur
                    st.rerun()

            st.write("#### ✏️ Current Annotations Ledger (Double-click any field to modify):")
            df_annotations_table = pd.DataFrame(st.session_state.annotations)
            df_annotations_table.columns = ['Start Time', 'End Time', 'Logged Activity'] if len(df_annotations_table.columns) == 3 else df_annotations_table.columns
            
            st.data_editor(
                df_annotations_table,
                column_config={
                    "Start Time": st.column_config.TextColumn("Start Time (HH:MM)", max_chars=5),
                    "End Time": st.column_config.TextColumn("End Time (HH:MM)", max_chars=5),
                    "Logged Activity": st.column_config.SelectboxColumn("Activity", options=['Sedentary', 'Light', 'Cycling', 'Moderate (Walking)', 'Vigorous (Running)', 'Sédentaire', 'Légère', 'Vélo', 'Modérée (Marche)', 'Vigoureuse (Course)'], required=True)
                },
                hide_index=True, use_container_width=True, key="editor_annotations"
            )
            
            if st.button("🗑️ Clear All Annotations", key="btn_clear_annotations"):
                st.session_state.annotations = []
                st.rerun()

            st.write("#### 📊 Metrics & Match Validation Statistics:")
            df_res_match = df_res.copy()
            df_annot_match = df_annot.copy()
            rev_map = {'Sedentary': 'Sédentaire', 'Light': 'Légère', 'Cycling': 'Vélo', 'Moderate (Walking)': 'Modérée (Marche)', 'Vigorous (Running)': 'Vigoureuse (Course)', 'Non-Wear': 'Non Porté', 'Not annotated': 'Non annoté'}
            df_res_match['Activite_Lissee'] = df_res_match['Activite_Lissee'].replace(rev_map)
            df_annot_match['Activite_Annotee'] = df_annot_match['Activite_Annotee'].replace(rev_map)

            metriques = calculer_metriques_correspondance(df_res_match, df_annot_match)
            if metriques is not None:
                classes_selectionnees = df_annot[df_annot['Activite_Annotee'] != 'Not annotated']['Activite_Annotee'].nunique()
                c1, c2, c3 = st.columns(3)
                c1.metric("Logged Accuracy", f"{metriques['accuracy']*100:.1f}%")
                c2.metric("Logged Weighted F1", f"{metriques['f1_weighted']:.3f}")
    
                if classes_selectionnees <= 1:
                    c3.metric("Logged Cohen's Kappa", "N/A", help="Kappa calculation requires at least two distinct classified categories.")
                    st.warning("⚠️ **Note on Kappa:** You have only logged one activity type. Add another classification to evaluate Kappa.")
                else:
                    c3.metric(label="Logged Cohen's Kappa", value=f"{metriques['kappa']:.3f}", help="Cohen's Kappa (1960) measures inter-rater agreement accounting for chance.")
        
                st.write("**Confusion Matrix:**")
                df_cm = generer_df_confusion_matrix(metriques)
                df_cm.index = [list(rev_map.keys())[list(rev_map.values()).index(x)] if x in rev_map.values() else x for x in df_cm.index]
                df_cm.columns = [list(rev_map.keys())[list(rev_map.values()).index(x)] if x in rev_map.values() else x for x in df_cm.columns]
                st.dataframe(df_cm, use_container_width=True)
        
                st.write("**Activity Performance Breakdown:**")
                translated_classes = [list(rev_map.keys())[list(rev_map.values()).index(c)] if c in rev_map.values() else c for c in metriques['classes']]
                df_perf = pd.DataFrame({
                            'Activity': translated_classes, 'Precision': [metriques['precision'][c] for c in metriques['classes']],
                            'Recall': [metriques['recall'][c] for c in metriques['classes']], 'F1-Score': [metriques['f1'][c] for c in metriques['classes']],
                            "Epoch Count (10s)": [metriques['support'][c] for c in metriques['classes']]
                        })
                st.dataframe(df_perf.style.format({'Precision': '{:.3f}', 'Recall': '{:.3f}', 'F1-Score': '{:.3f}'}), use_container_width=True)

                total_epochs_annotees = df_perf["Epoch Count (10s)"].sum()
                temps_total_minutes = round((total_epochs_annotees * epoch_choisi) / 60, 1)
                st.info(f"📋 **Total for this day:** You manually annotated a total of **{total_epochs_annotees} epochs** (approx. **{temps_total_minutes} minutes**).")
        
                if classes_selectionnees > 1:
                    kappa_val = metriques['kappa']
                    interpretation = "❌ Very Poor Agreement" if kappa_val < 0 else ("⚠️ Slight Agreement" if kappa_val < 0.2 else ("🟡 Fair Agreement" if kappa_val < 0.4 else ("🟢 Moderate Agreement" if kappa_val < 0.6 else ("✅ Substantial Agreement" if kappa_val < 0.8 else "🟢 Almost Perfect Agreement"))))
                    st.info(f"**Interpretation:** {interpretation}")
            else:
                st.info("Log timeline interval inputs above to compute precision accuracy analytics profiles.")

    # ──────────────────────────────────────────────────────────────────
    # TAB 2: GLOBAL SUMMARY
    # ──────────────────────────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)
        df_repartition = df_res['Activite_Lissee'].value_counts(normalize=True).reset_index()
        df_repartition.columns = ['Activity', 'Percentage']
        df_repartition['Percentage'] *= 100

        with col1:
            st.write("#### Time Split Allocation (Donut Chart)")
            fig_donut = px.pie(df_repartition, names='Activity', values='Percentage', hole=0.6, color='Activity', color_discrete_map=PALETTE)
            fig_donut.update_traces(textinfo='percent+label', hovertemplate="<b>%{label}</b><br>Share : %{value:.1f}%")
            st.plotly_chart(fig_donut, use_container_width=True, key="chart_t2_global_donut")

        with col2:
            st.write("#### Motor Footprint (Radar Chart)")
            categories    = ['Sedentary', 'Light', 'Cycling', 'Moderate (Walking)', 'Vigorous (Running)']
            valeurs_radar = {cat: 0.0 for cat in categories}
            for _, row in df_repartition.iterrows():
                if row['Activity'] in valeurs_radar: valeurs_radar[row['Activity']] = row['Percentage']
            r_v = list(valeurs_radar.values()) + [list(valeurs_radar.values())[0]]
            t_v = list(valeurs_radar.keys())   + [list(valeurs_radar.keys())[0]]
            fig_radar = go.Figure(data=go.Scatterpolar(r=r_v, theta=t_v, fill='toself', line_color='#8da0cb', fillcolor='rgba(141,160,203,0.3)'))
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=False)
            st.plotly_chart(fig_radar, use_container_width=True, key="chart_t2_motor_radar")

    # ──────────────────────────────────────────────────────────────────
    # TAB 3: STATISTICS (WHERE THE CURRENT ERROR WAS TRIGGERED)
    # ──────────────────────────────────────────────────────────────────
    with tab3:
        st.subheader("Statistical Threshold Validation")
        fig_box = px.box(
            df_actif, x="Activite_Lissee", y="VM_Counts", color="Activite_Lissee",
            color_discrete_map=PALETTE, category_orders={"Activite_Lissee": ORDRE_ACTIVITES},
            labels={"VM_Counts": "VM Counts", "Activite_Lissee": "Activity Classification"}
        )
        st.plotly_chart(fig_box, use_container_width=True, key="chart_t3_threshold_box")

        st.divider()
        st.markdown("Mann-Whitney U non-parametric test comparison between **Light** and **Moderate (Walking)** intensity classifications to validate variance and threshold separability bounds.")
        val_moderee = df_actif[df_actif['Activite_Lissee'] == 'Moderate (Walking)']['VM_Counts']
        val_legere  = df_actif[df_actif['Activite_Lissee'] == 'Light']['VM_Counts']
        if not val_moderee.empty and not val_legere.empty:
            _, p = mannwhitneyu(val_moderee, val_legere)
            st.metric("Mann-Whitney U P-Value (Light vs Moderate)", f"{p:.4e}")
            if p < 0.05: st.success("✅ Statistically significant difference verified (p < 0.05)")
            else: st.warning("⚠️ Non-significant variance disparity difference (p ≥ 0.05)")
        else:
            st.info("Insufficient data available across selected filters to calculate statistical test metrics.")

    # ──────────────────────────────────────────────────────────────────
    # TAB 4: SESSION DETAILS
    # ──────────────────────────────────────────────────────────────────
    with tab4:
        st.subheader("Detailed Activity Session Log Inventory")
        c1, c2 = st.columns(2)
        with c1:
            choix_act = st.selectbox("Select designated activity class filter:", ['Moderate (Walking)', 'Vigorous (Running)', 'Cycling', 'Light', 'Sedentary'])
        with c2:
            seuil_min = st.number_input("Minimum continuous session cut-off rule duration (Minutes):", min_value=0.0, value=1.0, step=0.5)
            
        fr_session_query_map = {'Moderate (Walking)': 'Modérée (Marche)', 'Vigorous (Running)': 'Vigoureuse (Course)', 'Cycling': 'Vélo', 'Light': 'Légère', 'Sedentary': 'Sédentaire'}
        df_sessions = extraire_sessions(df_res, fr_session_query_map.get(choix_act, choix_act), epoch_choisi, seuil_min)
        
        if not df_sessions.empty:
            st.write(f"**{len(df_sessions)}** continuous block session(s) of **{choix_act}** qualified (≥ {seuil_min} Min)")
            colonnes_affichage = [c for c in df_sessions.columns if c != 'Heure_Debut_Raw']
            df_sessions_eng = df_sessions[colonnes_affichage].copy()
            df_sessions_eng.columns = ['Start Time', 'End Time', 'Duration (Min)', 'Mean Intensity (VM)'] if len(df_sessions_eng.columns) == 4 else df_sessions_eng.columns
            st.dataframe(df_sessions_eng, use_container_width=True)
            
            if choix_act == 'Vigorous (Running)':
                st.write("#### 📝 Automatic Generated Text Synthesis for Reporting Documents:")
                for idx, row in df_sessions.iterrows():
                    date_session = row['Heure_Debut_Raw'].strftime('%m/%d/%Y')
                    heure_debut = row[df_sessions.columns[0]] 
                    heure_fin = row[df_sessions.columns[1]]
                    duree = row[df_sessions.columns[2]]
                    st.info(f"📖 *\"On **{date_session}**, participant **P1** performed a running session of **{duree} minutes** starting at **{heure_debut}** and ending at **{heure_fin}** (with a 2-min Troiano tolerance rule applied).\"*")

    # ──────────────────────────────────────────────────────────────────
    # TAB 5: CYCLING DETECTED
    # ──────────────────────────────────────────────────────────────────
    with tab5:
        if not activer_velo:
            st.info("🚴 Cycling detection is currently disabled. Toggle setting parameter switch in sidebar panel to initialize.")
        else:
            st.subheader("🚴 Detected Cycling Activity Sessions")
            st.warning("**Heuristic Classification Result** — Detection profiling based on spectral analysis computation rules (FFT applied to Axis Z) and stability coefficient matching parameters of Vector Magnitude values (CV), independent of GPS logs. Hip-placement calibrated thresholds rules: FFT ≥ 0.08 (Sensitivity: 98.9%), CV < 0.55 (Sensitivity: 97.8%). Methodological reference source citation structure: Brage et al. (2005), *J Appl Physiol*.", icon="⚠️")
            df_velo        = df_res[df_res['Activite_Lissee'] == 'Cycling'].copy()
            n_velo         = len(df_velo)
            duree_velo_min = round((n_velo * epoch_choisi) / 60.0, 1)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Detected Cycling Epochs", n_velo)
            col_b.metric("Total Cycling Duration", f"{duree_velo_min} min")
            col_c.metric("% of Valid Wear Time Allocation", f"{100 * n_velo / max(1, len(df_actif)):.1f} %")

            if not df_velo.empty:
                fig_velo = px.scatter(df_velo, x='Heure', y='VM_Counts', color_discrete_sequence=[PALETTE['Cycling']], labels={'VM_Counts': 'VM Counts', 'Heure': 'Time Value'}, title="Temporal Scatter Distribution of Cycling Classified Epoch Intervals")
                fig_velo.update_traces(marker=dict(size=6, symbol='diamond'))
                st.plotly_chart(fig_velo, use_container_width=True, key="chart_t5_cycling_scatter")

                st.write("#### Continuous Cycling Block Intervals")
                seuil_min_velo = st.number_input("Minimum cycling duration interval threshold limit (Minutes):", min_value=0.0, value=1.0, step=0.5, key="seuil_velo")
                df_velo_sessions = extraire_sessions(df_res, 'Vélo', epoch_choisi, seuil_min_velo)
                if not df_velo_sessions.empty:
                    st.write(f"**{len(df_velo_sessions)}** sequence session interval block(s) isolated")
                    df_velo_sessions.columns = ['Start Time', 'End Time', 'Duration (Min)', 'Mean Intensity (VM)'] if len(df_velo_sessions.columns) == 4 else df_velo_sessions.columns
                    st.dataframe(df_velo_sessions, use_container_width=True)
                else:
                    st.info("No continuous sequence blocks matching current filter constraint criteria isolated.")

    # ──────────────────────────────────────────────────────────────────
    # TAB 6: CYCLING DIAGNOSIS
    # ──────────────────────────────────────────────────────────────────
    with tab6:
        if not activer_velo:
            st.info("🔬 Cycling detection is currently disabled. Toggle setting parameter switch in sidebar panel to initialize.")
        else:
            st.subheader("🔬 Diagnostic Breakdown — Step-by-Step Criterion Analysis")
            st.info("This visual log displays raw criterion metrics corresponding to each analytical epoch to isolate and identify limiting diagnostic variables. Select your comparison timeframe slice configuration parameters below.")

            if 'Heure' in df_res.columns:
                col_h1, col_h2 = st.columns(2)
                with col_h1: h_debut = st.time_input("Observed activity sequence start bounds selection marker", value=df_res['Heure'].min().time(), key="time_h1_t6")
                with col_h2: h_fin = st.time_input("Observed activity sequence end bounds selection marker", value=df_res['Heure'].max().time(), key="time_h2_t6")
                df_diag = df_res[(df_res['Heure'].dt.time >= h_debut) & (df_res['Heure'].dt.time <= h_fin)].copy()
            else:
                df_diag = df_res.copy()

            if df_diag.empty:
                st.warning("No records found inside selected bounding constraints parameters.")
            else:
                st.markdown("#### Criterion Pass-Rate Log Percentage Across Time Frame Selection Index")
                n_total = len(df_diag)
                n_fft   = int(df_diag['Diag_FFT_OK'].sum())
                n_cv    = int(df_diag['Diag_CV_OK'].sum())
                n_vm    = int(df_diag['Diag_VM_OK'].sum())
                n_velo  = int(df_diag['Flag_Velo'].sum())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("FFT Rule Constraint Pass", f"{n_fft}/{n_total}", f"{100*n_fft/n_total:.0f}%")
                c2.metric("CV Rule Constraint Pass", f"{n_cv}/{n_total}", f"{100*n_cv/n_total:.0f}%")
                c3.metric("VM Range Constraint Pass", f"{n_vm}/{n_total}", f"{100*n_vm/n_total:.0f}%")
                c4.metric("🚴 Cycling Verified Result", f"{n_velo}/{n_total}", f"{100*n_velo/n_total:.0f}%", delta_color="off")
                st.caption(f"Active Operational Thresholds Parameter Logs — FFT Ratio Limit ≥ {seuil_spectral:.2f}  |  CV Ratio Limit < {seuil_cv:.2f}  |  VM Count Span Range: {round(200*epoch_choisi/60)}–{round(6167*epoch_choisi/60)} Counts/Epoch")

                st.markdown("#### Spectral Ratio Value Distribution Matrix — Criterion Rule Index 1 (FFT)")
                fig_fft = px.histogram(df_diag.dropna(subset=['Diag_FFT_Ratio']), x='Diag_FFT_Ratio', nbins=40, color_discrete_sequence=['#66c2a5'], labels={'Diag_FFT_Ratio': 'FFT Spectral Energy Fraction [0.5–2 Hz]'})
                fig_fft.add_vline(x=seuil_spectral, line_dash="dash", line_color="red", annotation_text=f"Threshold = {seuil_spectral}", annotation_position="top right")
                st.plotly_chart(fig_fft, use_container_width=True, key="chart_t6_diag_fft_hist")

                st.markdown("#### Coefficient of Variation Range Matrix — Criterion Rule Index 2 (CV)")
                fig_cv = px.histogram(df_diag.dropna(subset=['Diag_CV']), x='Diag_CV', nbins=40, color_discrete_sequence=['#fc8d62'], labels={'Diag_CV': 'VM Count Coefficient of Variation Value (3-Epoch Window Block)'})
                fig_cv.add_vline(x=seuil_cv, line_dash="dash", line_color="red", annotation_text=f"Threshold = {seuil_cv}", annotation_position="top right")
                st.plotly_chart(fig_cv, use_container_width=True, key="chart_t6_diag_cv_hist")

                st.markdown("#### Metric Log Matrix Ledger Breakdown — Row Tracking Profile Index")
                cols_affich    = ['Heure', 'VM_Counts', 'Activite_Lissee', 'Diag_FFT_Ratio', 'Diag_FFT_OK', 'Diag_CV', 'Diag_CV_OK', 'Diag_VM_OK', 'Flag_Velo']
                cols_presentes = [c for c in cols_affich if c in df_diag.columns]
                st.dataframe(
                    df_diag[cols_presentes].rename(columns={'Heure': 'Timestamp', 'VM_Counts': 'VM Counts', 'Activite_Lissee': 'Assigned Activity Class', 'Diag_FFT_Ratio': 'FFT Ratio', 'Diag_FFT_OK': 'FFT ✓', 'Diag_CV': 'CV Value', 'Diag_CV_OK': 'CV ✓', 'Diag_VM_OK': 'VM Range ✓', 'Flag_Velo': '🚴 Cycling Flag'}).style.map(
                        lambda v: 'background-color: #c6efce' if v is True else ('background-color: #ffc7ce' if v is False else ''), subset=['FFT ✓', 'CV ✓', 'VM Range ✓', '🚴 Cycling Flag']
                    ), use_container_width=True
                )

    # ──────────────────────────────────────────────────────────────────
    # TAB 7: PARTICIPANT SUMMARY
    # ──────────────────────────────────────────────────────────────────
    # ──────────────────────────────────────────────────────────────────
    # TAB 7: PARTICIPANT SUMMARY (FULLY TRANSLATED & OPTIMIZED)
    # ──────────────────────────────────────────────────────────────────
    with tab7:
        st.subheader("📁 Participant Aggregate Statistics Summary Report")
        st.caption("Valid compliance monitoring day requirement: ≥ 10 Hours of sensor wear (excluding Non-Wear). Minimal sequence: ≥ 1 Minute.\nMorning: 06:00–12:00 / Afternoon: 12:00–18:00 / Evening: 18:00–22:00.")

        if 'agd_days' in st.session_state:
            all_days = list(st.session_state['agd_days'].values())
            df_all = pd.concat(all_days, ignore_index=True)
            resume = calculer_resume_participant(df_all, epoch_choisi)
        else:
            resume = calculer_resume_participant(df_res, epoch_choisi)

        n_total  = len(resume['details_jours'])
        n_valides = resume['jours_valides']

        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Registered Log Sequence Days", n_total)
        col_b.metric("Compliant Valid Sequence Days (≥ 10h)", n_valides)
        col_c.metric("Non-Compliant Invalid Days (< 10h)", n_total - n_valides)

        df_jours = resume['details_jours'].copy()
        df_jours['Status'] = df_jours['Valide'].map({True: '✅ Valid Compliant', False: '❌ Invalid Under-wear'})
        df_jours['Date'] = df_jours['Date'].astype(str)
        st.dataframe(df_jours[['Date', 'Heures_Port', 'Status']].rename(columns={'Heures_Port': 'Accumulated Wear (Hours)', 'Status': 'Compliance Tag'}), use_container_width=True, hide_index=True)

        fig_jours = px.bar(df_jours, x='Date', y='Heures_Port', labels={'Heures_Port': 'Sensor Wear Duration (Hours)', 'Date': 'Calendar Tracking Date', 'Status': 'Compliance Key Label'}, title="Wear-Time Volume Quantiles Plot (Green = Compliant / Grey = Non-Compliant)", color='Status', color_discrete_map={'✅ Valid Compliant': '#66c2a5', '❌ Invalid Under-wear': '#b3b3b3'})
        fig_jours.add_hline(y=10, line_dash="dash", line_color="red", annotation_text="Compliance Minimum Wear Bar (10h)", annotation_position="top right")
        fig_jours.update_layout(xaxis_tickangle=-45, legend_title_text="")
        st.plotly_chart(fig_jours, use_container_width=True, key="chart_t7_wear_days_bars")

        # ── Matrix Ledger Summary ──────────────────────────────────────────
        st.divider()
        st.markdown("#### Operational Categorized Activity Continuous Blocks Session Log Matrix")
        
        df_sessions_resume = resume['resume_sessions'].copy()
        
        # 1. Translate matrix cell labels safely FIRST while column is still named 'Activite'
        if 'Activite' in df_sessions_resume.columns:
            df_sessions_resume['Activite'] = df_sessions_resume['Activite'].replace({
                'Modérée (Marche)': 'Moderate (Walking)', 
                'Vigoureuse (Course)': 'Vigorous (Running)', 
                'Vélo': 'Cycling', 
                'Légère': 'Light', 
                'Sédentaire': 'Sedentary'
            })
            
        # 2. Rename columns structural labels for UI presentation layer
        df_sessions_resume.columns = [
            'Class Activity Segment Type', 
            'Session Count Summary Quantiles', 
            'Mean Block Duration Value (Min)', 
            'Median Block Duration Value (Min)', 
            'Total Net Accumulation Runtime Span (Min)'
        ]
        st.dataframe(df_sessions_resume, use_container_width=True, hide_index=True)

        # ── Frequency Counts Plot Fix ──────────────────────────────────────
        fig_sessions = px.bar(
            df_sessions_resume,                     # FIX: Use the clean, translated dataframe source
            x='Class Activity Segment Type',         # FIX: Use the new translated x-axis column
            y='Session Count Summary Quantiles',     # FIX: Use the new translated y-axis column
            color='Class Activity Segment Type',
            color_discrete_map={
                'Moderate (Walking)': PALETTE['Moderate (Walking)'], 
                'Vigorous (Running)': PALETTE['Vigorous (Running)'], 
                'Cycling': PALETTE['Cycling'], 
                'Light': PALETTE['Light'], 
                'Sedentary': PALETTE['Sedentary']
            },
            labels={
                'Class Activity Segment Type': 'Activity Class Assignment', 
                'Session Count Summary Quantiles': 'Absolute Session Count'
            }, 
            title="Total Session Frequency Counts"
        )
        fig_sessions.update_layout(showlegend=False)
        st.plotly_chart(fig_sessions, use_container_width=True, key="chart_t7_activity_frequency_bars")

        # ── Diurnal Phase Windows Summary ──────────────────────────────────
        st.divider()
        st.markdown("#### Dominant Temporal Phase Windows Metric Breakdown Table")
        df_creneaux = resume['creneaux'].copy()
        if 'Activite' in df_creneaux.columns:
            df_creneaux['Activite'] = df_creneaux['Activite'].replace({'Modérée (Marche)': 'Moderate (Walking)', 'Vigoureuse (Course)': 'Vigorous (Running)', 'Vélo': 'Cycling', 'Légère': 'Light', 'Sédentaire': 'Sedentary'})

        df_creneaux_long = df_creneaux.melt(id_vars='Activite', value_vars=['N_Sessions_Matin', 'N_Sessions_Apres_Midi', 'N_Sessions_Soir'], var_name='Phase Window', value_name='N_Sessions')
        df_creneaux_long['Phase Window'] = df_creneaux_long['Phase Window'].map({'N_Sessions_Matin': 'Morning Windows (06:00–12:00)', 'N_Sessions_Apres_Midi': 'Afternoon Windows (12:00–18:00)', 'N_Sessions_Soir': 'Evening Windows (18:00–22:00)'})
        
        fig_creneaux = px.bar(
            df_creneaux_long, x='Activite', y='N_Sessions', color='Phase Window', barmode='group', 
            color_discrete_sequence=['#fc8d62', '#8da0cb', '#66c2a5'], 
            labels={'Activite': 'Activity Categorization Label', 'N_Sessions': 'Block Summation Frequency'}, 
            title="Diurnal Phase Split Profiles"
        )
        fig_creneaux.update_layout(legend_title_text="Phase Window Breakdown")
        st.plotly_chart(fig_creneaux, use_container_width=True, key="chart_t7_diurnal_phase_bars")

        df_creneaux_affich = df_creneaux.rename(columns={'Activite': 'Target Activity Profile Category', 'Creneau_Dominant': 'Primary Dominant Peak Assignment', 'N_Sessions_Matin': 'Morning Vol (06:00–12:00)', 'N_Sessions_Apres_Midi': 'Afternoon Vol (12:00–18:00)', 'N_Sessions_Soir': 'Evening Vol (18:00–22:00)'})
        if 'Primary Dominant Peak Assignment' in df_creneaux_affich.columns:
            df_creneaux_affich['Primary Dominant Peak Assignment'] = df_creneaux_affich['Primary Dominant Peak Assignment'].replace({'Matin': 'Morning', 'Après-midi': 'Afternoon', 'Soir': 'Evening'})
        st.dataframe(df_creneaux_affich, use_container_width=True, hide_index=True)

    # ──────────────────────────────────────────────────────────────────
    # TAB 8: FIGURE EXPORT
    # ──────────────────────────────────────────────────────────────────
    with tab8:
        st.subheader("🖼️ Thesis Figure Export")
        st.caption(
            f"Publication-ready figures (300 DPI, white background) saved to `{FIGURES_DIR}/`. "
            "All figures are generated from the currently selected day's processed data."
        )

        col_fig1, col_fig2 = st.columns(2)
        with col_fig1:
            activites_fig = st.multiselect(
                "Activities to highlight in fig 5 & 6",
                options=['Moderate (Walking)', 'Vigorous (Running)',
                         'Cycling', 'Light', 'Sedentary'],
                default=['Moderate (Walking)', 'Vigorous (Running)', 'Cycling'],
                help="Select one or more activities. Leave empty to show all.",
                key="fig_activites_multiselect"
            )
        with col_fig2:
            day_label_fig = st.text_input(
                "Day label for figure titles",
                value=str(df_res['Heure'].iloc[0].date()) if 'Heure' in df_res.columns else "",
                key="fig_day_label"
            )

        st.markdown("#### Figures that will be generated")
        st.markdown(
            "| File | Content |\n"
            "|---|---|\n"
            "| `fig1_vm_timeseries.png` | VM counts time series — full day signal |\n"
            "| `fig2_vm_thresholds.png` | VM signal with classification thresholds (Sasaki 2011) |\n"
            "| `fig3_smoothing.png` | Before/after 60 s modal smoothing |\n"
            "| `fig4_nonwear.png` | Non-wear periods highlighted (Choi et al., 2011) |\n"
            "| `fig5_sessions_*.png` | Session extraction for selected activities |\n"
            "| `fig6_multiday_*.png` | Multi-day session recurrence per selected activity *(requires .agd with multiple days)* |"
        )

        if st.button("⬇️ Generate & Save All Figures", key="btn_generate_figures",
                     type="primary", use_container_width=True):
            with st.spinner("Generating figures…"):
                agd_days_fig = st.session_state.get('agd_days', None)
                try:
                    paths = generer_toutes_figures(
                        df                 = df_res,
                        epoch_seconds      = epoch_choisi,
                        day_label          = day_label_fig,
                        agd_days           = agd_days_fig,
                        activites_sessions = activites_fig if activites_fig else None,
                    )
                    st.success(f"✅ {len(paths)} figure(s) saved to `{FIGURES_DIR}/`")
                    for p in paths:
                        st.write(f"  • `{p}`")
                except Exception as e:
                    st.error(f"❌ Error generating figures: {e}")

        st.divider()
        st.markdown("#### Preview")
        st.caption("Run generation above first, then reload to see previews here.")

        import glob
        existing = sorted(glob.glob(f"{FIGURES_DIR}/fig*.png"))
        if existing:
            for png_path in existing:
                fname = os.path.basename(png_path)
                st.markdown(f"**`{fname}`**")
                st.image(png_path, use_container_width=True)
        else:
            st.info("No figures generated yet. Click the button above to create them.")

    # ──────────────────────────────────────────────────────────────────
    # CSV EXPORT ROUTINE
    # ──────────────────────────────────────────────────────────────────
    colonnes_export = [c for c in df_res.columns if not c.startswith('Diag_')]
    os.makedirs("results", exist_ok=True)
    nom_fichier = f"{os.path.splitext(fichier_upload.name)[0]}_processed.csv"
    chemin      = os.path.join("results", nom_fichier)
    df_res[colonnes_export].to_csv(chemin, index=False, encoding='utf-8-sig')
    st.success(f"✅ Extracted export payload ledger written successfully out to path destination: `{chemin}`")