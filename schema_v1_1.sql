PRAGMA foreign_keys = ON;
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA cache_size = -64000;

CREATE TABLE IF NOT EXISTS individus (
    id TEXT PRIMARY KEY,
    nom TEXT NOT NULL,
    prenom TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    sexe TEXT NOT NULL CHECK (sexe IN ('M', 'F')),
    date_naissance TEXT,
    role TEXT NOT NULL CHECK (role IN ('athlete', 'coach', 'technicien', 'autre')),
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1)),
    password_hash TEXT,
    last_login_at TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    updated_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE TABLE IF NOT EXISTS athletes (
    individu_id TEXT PRIMARY KEY REFERENCES individus(id) ON DELETE RESTRICT,
    categorie_poids TEXT,
    taille_cm REAL
);

CREATE TABLE IF NOT EXISTS reference_tests (
    id TEXT PRIMARY KEY,
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id) ON DELETE RESTRICT,
    date_enregistrement TEXT NOT NULL,
    fc_max_bpm INTEGER CHECK (fc_max_bpm BETWEEN 100 AND 250),
    fc_repos_bpm INTEGER CHECK (fc_repos_bpm BETWEEN 25 AND 100),
    hrr_1min_bpm INTEGER CHECK (hrr_1min_bpm BETWEEN 0 AND 150),
    hrr_2min_bpm INTEGER CHECK (hrr_2min_bpm BETWEEN 0 AND 150),
    vo2max_ml_kg_min REAL CHECK (vo2max_ml_kg_min BETWEEN 20 AND 100),
    sv1_pct_fcmax REAL CHECK (sv1_pct_fcmax BETWEEN 0 AND 100),
    sv2_pct_fcmax REAL CHECK (sv2_pct_fcmax BETWEEN 0 AND 100),
    poids_kg REAL CHECK (poids_kg BETWEEN 20 AND 200),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (athlete_id, date_enregistrement)
);

CREATE INDEX IF NOT EXISTS idx_reference_tests_athlete_date
    ON reference_tests (athlete_id, date_enregistrement DESC);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    date_seance TEXT NOT NULL,
    heure_debut_reelle TEXT,
    lieu TEXT,
    type_seance TEXT NOT NULL CHECK (type_seance IN (
        'randori_tw', 'randori_nw', 'randori_mixte', 'technique', 'autre'
    )),
    activity_family TEXT,
    activity_notes TEXT,
    operateur TEXT NOT NULL,
    is_archived INTEGER NOT NULL DEFAULT 0 CHECK (is_archived IN (0, 1)),
    archived_at TEXT,
    is_temporally_annotated INTEGER NOT NULL DEFAULT 0 CHECK (is_temporally_annotated IN (0, 1)),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_sessions_date ON sessions (date_seance DESC);

CREATE TABLE IF NOT EXISTS session_tatamis (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    tatami_id TEXT NOT NULL,
    camera_id TEXT,
    PRIMARY KEY (session_id, tatami_id)
);

CREATE TABLE IF NOT EXISTS session_athletes (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id) ON DELETE RESTRICT,
    rpe_global_seance INTEGER CHECK (rpe_global_seance BETWEEN 0 AND 10),
    rpe_global_timestamp TEXT,
    PRIMARY KEY (session_id, athlete_id)
);

CREATE TABLE IF NOT EXISTS phases_programmees (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    ordre INTEGER NOT NULL CHECK (ordre >= 1),
    type_phase TEXT NOT NULL CHECK (type_phase IN (
        'randori_tw', 'randori_nw', 'randori_mixte', 'recuperation',
        'echauffement', 'technique', 'retour_calme', 'autre'
    )),
    nb_repetitions INTEGER CHECK (nb_repetitions >= 1),
    duree_cible_s INTEGER CHECK (duree_cible_s > 0),
    recuperation_cible_s INTEGER CHECK (recuperation_cible_s >= 0),
    rpe_cible INTEGER CHECK (rpe_cible BETWEEN 0 AND 10),
    notes_entraineur TEXT,
    UNIQUE (session_id, ordre)
);

CREATE TABLE IF NOT EXISTS phases_realisees (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    phase_programmee_id TEXT REFERENCES phases_programmees(id) ON DELETE SET NULL,
    ordre INTEGER NOT NULL CHECK (ordre >= 1),
    type_phase TEXT NOT NULL CHECK (type_phase IN (
        'randori_tw', 'randori_nw', 'randori_mixte', 'recuperation',
        'echauffement', 'technique', 'retour_calme', 'autre'
    )),
    t_debut_utc_ms INTEGER NOT NULL,
    t_fin_utc_ms INTEGER NOT NULL,
    duree_reelle_s REAL GENERATED ALWAYS AS ((t_fin_utc_ms - t_debut_utc_ms) / 1000.0) STORED,
    source_annotation TEXT NOT NULL CHECK (
        source_annotation IN ('annotation_fc', 'observateur_manuel', 'video_ia')
    ),
    timing_annotation TEXT NOT NULL CHECK (timing_annotation IN ('in', 'post')),
    source_segment_index INTEGER,
    notes TEXT,
    CHECK (t_fin_utc_ms > t_debut_utc_ms),
    UNIQUE (session_id, ordre)
);

CREATE INDEX IF NOT EXISTS idx_phases_realisees_session
    ON phases_realisees (session_id, ordre);
CREATE INDEX IF NOT EXISTS idx_phases_realisees_programmee
    ON phases_realisees (phase_programmee_id);

CREATE TABLE IF NOT EXISTS randoris_programmes (
    id TEXT PRIMARY KEY,
    phase_programmee_id TEXT NOT NULL REFERENCES phases_programmees(id) ON DELETE CASCADE,
    numero INTEGER NOT NULL CHECK (numero >= 1),
    type_randori TEXT NOT NULL CHECK (type_randori IN ('randori_tw', 'randori_nw', 'randori_mixte')),
    duree_cible_s INTEGER NOT NULL CHECK (duree_cible_s > 0),
    UNIQUE (phase_programmee_id, numero)
);

CREATE TABLE IF NOT EXISTS pipeline_versions (
    version TEXT PRIMARY KEY,
    date_deploy TEXT NOT NULL,
    description TEXT,
    params_json TEXT,
    is_current INTEGER NOT NULL DEFAULT 0 CHECK (is_current IN (0, 1))
);

CREATE TABLE IF NOT EXISTS acquisitions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id) ON DELETE RESTRICT,
    pipeline_version TEXT REFERENCES pipeline_versions(version),
    source TEXT NOT NULL CHECK (source IN ('psl', 'bleak')),
    device_id TEXT,
    mac_address TEXT,
    adapter_id TEXT,
    start_time_utc TEXT NOT NULL,
    end_time_utc TEXT,
    nb_rr_bruts INTEGER CHECK (nb_rr_bruts >= 0),
    nb_points_hr INTEGER CHECK (nb_points_hr >= 0),
    duree_s REAL CHECK (duree_s >= 0),
    segments_count INTEGER CHECK (segments_count >= 0),
    gaps_count INTEGER CHECK (gaps_count >= 0),
    nb_deconnexions INTEGER CHECK (nb_deconnexions >= 0),
    batterie_pct_debut INTEGER CHECK (batterie_pct_debut BETWEEN 0 AND 100),
    fc_offset_s REAL,
    rr_raw_filepath TEXT NOT NULL,
    hr_raw_filepath TEXT,
    qualite_signal TEXT CHECK (qualite_signal IN ('bonne', 'moyenne', 'mauvaise')),
    statut TEXT NOT NULL CHECK (statut IN ('running', 'completed', 'interrupted')),
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%SZ', 'now')),
    UNIQUE (session_id, athlete_id)
);

CREATE INDEX IF NOT EXISTS idx_acquisitions_session ON acquisitions (session_id);
CREATE INDEX IF NOT EXISTS idx_acquisitions_athlete ON acquisitions (athlete_id);

CREATE TABLE IF NOT EXISTS rr_clean_meta (
    acquisition_id TEXT PRIMARY KEY REFERENCES acquisitions(id) ON DELETE CASCADE,
    export_timestamp TEXT NOT NULL,
    ok_rr_total INTEGER CHECK (ok_rr_total >= 0),
    global_quality_label TEXT CHECK (global_quality_label IN ('OK', 'WARNING', 'BAD'))
);

CREATE TABLE IF NOT EXISTS rr_clean_exploitabilite (
    acquisition_id TEXT PRIMARY KEY REFERENCES acquisitions(id) ON DELETE CASCADE,
    fc_ok_total INTEGER CHECK (fc_ok_total >= 0),
    fc_non_viable_total INTEGER CHECK (fc_non_viable_total >= 0),
    fc_non_ok_rate REAL CHECK (fc_non_ok_rate BETWEEN 0 AND 1),
    hrr_ok_total INTEGER CHECK (hrr_ok_total >= 0),
    hrr_non_viable_total INTEGER CHECK (hrr_non_viable_total >= 0),
    hrr_non_ok_rate REAL CHECK (hrr_non_ok_rate BETWEEN 0 AND 1),
    rmssd_ok_total INTEGER CHECK (rmssd_ok_total >= 0),
    rmssd_non_viable_total INTEGER CHECK (rmssd_non_viable_total >= 0),
    rmssd_non_ok_rate REAL CHECK (rmssd_non_ok_rate BETWEEN 0 AND 1)
);

CREATE TABLE IF NOT EXISTS rr_indicator_files (
    id TEXT PRIMARY KEY,
    acquisition_id TEXT NOT NULL REFERENCES acquisitions(id) ON DELETE CASCADE,
    pipeline_version TEXT REFERENCES pipeline_versions(version),
    indicator_type TEXT NOT NULL,
    filepath TEXT NOT NULL,
    params_json TEXT,
    computed_at TEXT NOT NULL,
    UNIQUE (acquisition_id, indicator_type)
);

CREATE TABLE IF NOT EXISTS import_log (
    id TEXT PRIMARY KEY,
    acquisition_id TEXT NOT NULL REFERENCES acquisitions(id) ON DELETE CASCADE,
    fichier_source TEXT NOT NULL,
    type_fichier TEXT NOT NULL CHECK (type_fichier IN ('psl_rr', 'psl_hr', 'bleak_csv', 'parquet_clean')),
    date_import TEXT NOT NULL,
    nb_lignes INTEGER,
    statut TEXT NOT NULL CHECK (statut IN ('ok', 'erreur', 'partiel')),
    message TEXT
);

CREATE TABLE IF NOT EXISTS indicators_fc (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id) ON DELETE RESTRICT,
    acquisition_id TEXT NOT NULL REFERENCES acquisitions(id) ON DELETE RESTRICT,
    pipeline_version TEXT NOT NULL REFERENCES pipeline_versions(version),
    fc_peak_bpm REAL,
    fc_mean_bpm REAL,
    fc_min_bpm REAL,
    computed_at TEXT NOT NULL,
    UNIQUE (session_id, athlete_id, pipeline_version)
);

CREATE INDEX IF NOT EXISTS idx_indicators_fc_session ON indicators_fc (session_id);
CREATE INDEX IF NOT EXISTS idx_indicators_fc_athlete ON indicators_fc (athlete_id);

CREATE TABLE IF NOT EXISTS randoris_realises (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE RESTRICT,
    phase_realisee_id TEXT REFERENCES phases_realisees(id) ON DELETE RESTRICT,
    randori_programme_id TEXT REFERENCES randoris_programmes(id),
    athlete_a_id TEXT NOT NULL REFERENCES athletes(individu_id) ON DELETE RESTRICT,
    athlete_b_id TEXT REFERENCES athletes(individu_id) ON DELETE RESTRICT,
    numero_randori INTEGER NOT NULL CHECK (numero_randori >= 1),
    tatami_id TEXT,
    t_hajime_utc_ms INTEGER NOT NULL,
    t_matte_utc_ms INTEGER NOT NULL,
    duree_reelle_s REAL GENERATED ALWAYS AS ((t_matte_utc_ms - t_hajime_utc_ms) / 1000.0) STORED,
    type_randori TEXT NOT NULL CHECK (type_randori IN ('randori_tw', 'randori_nw', 'randori_mixte')),
    source_segmentation TEXT NOT NULL CHECK (
        source_segmentation IN ('video_ia', 'observateur_manuel', 'annotation_app', 'annotation_fc')
    ),
    confiance_segmentation REAL CHECK (confiance_segmentation BETWEEN 0 AND 1),
    notes TEXT,
    CHECK (t_matte_utc_ms > t_hajime_utc_ms),
    CHECK (athlete_b_id IS NULL OR athlete_a_id != athlete_b_id),
    UNIQUE (phase_realisee_id),
    UNIQUE (session_id, numero_randori)
);

CREATE INDEX IF NOT EXISTS idx_randoris_realises_session
    ON randoris_realises (session_id);
CREATE INDEX IF NOT EXISTS idx_randoris_realises_athletes
    ON randoris_realises (athlete_a_id, athlete_b_id);

CREATE TABLE IF NOT EXISTS indicators_video (
    id TEXT PRIMARY KEY,
    randori_id TEXT NOT NULL UNIQUE REFERENCES randoris_realises(id) ON DELETE CASCADE,
    duree_kumi_kata_s REAL CHECK (duree_kumi_kata_s >= 0),
    nb_kumi_kata INTEGER CHECK (nb_kumi_kata >= 0),
    nb_attaques_a INTEGER CHECK (nb_attaques_a >= 0),
    nb_tentatives_a INTEGER CHECK (nb_tentatives_a >= 0),
    nb_defense_a INTEGER CHECK (nb_defense_a >= 0),
    nb_chute_a INTEGER CHECK (nb_chute_a >= 0),
    nb_attaques_b INTEGER CHECK (nb_attaques_b >= 0),
    nb_tentatives_b INTEGER CHECK (nb_tentatives_b >= 0),
    nb_defense_b INTEGER CHECK (nb_defense_b >= 0),
    nb_chute_b INTEGER CHECK (nb_chute_b >= 0),
    duree_ne_waza_s REAL CHECK (duree_ne_waza_s >= 0),
    nb_breaks INTEGER CHECK (nb_breaks >= 0),
    duree_breaks_s REAL CHECK (duree_breaks_s >= 0),
    duree_pre_contact_s REAL CHECK (duree_pre_contact_s >= 0),
    confiance_identification_moy REAL CHECK (confiance_identification_moy BETWEEN 0 AND 1),
    perte_identite_flag INTEGER NOT NULL DEFAULT 0 CHECK (perte_identite_flag IN (0, 1)),
    frames_filepath_a TEXT,
    frames_filepath_b TEXT,
    source_fichier_video TEXT,
    imported_at TEXT
);

CREATE TABLE IF NOT EXISTS rpe (
    id TEXT PRIMARY KEY,
    randori_id TEXT NOT NULL REFERENCES randoris_realises(id) ON DELETE CASCADE,
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id) ON DELETE RESTRICT,
    rpe_ressenti INTEGER NOT NULL CHECK (rpe_ressenti BETWEEN 0 AND 10),
    moment_collecte TEXT NOT NULL CHECK (moment_collecte IN ('post_randori', 'post_seance')),
    timestamp_collecte TEXT NOT NULL,
    UNIQUE (randori_id, athlete_id)
);

CREATE TABLE IF NOT EXISTS rr_manual_annotations (
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source_index INTEGER NOT NULL CHECK (source_index >= 0),
    t_offset_ms INTEGER,
    rr_interval_ms REAL,
    manual_flag TEXT NOT NULL DEFAULT 'manuel',
    created_at TEXT,
    updated_at TEXT,
    PRIMARY KEY (session_id, source_index)
);

CREATE INDEX IF NOT EXISTS idx_rr_manual_annotations_session
    ON rr_manual_annotations (session_id, source_index);

INSERT OR IGNORE INTO pipeline_versions (version, date_deploy, description, params_json, is_current)
VALUES (
    'v3.3',
    '2026-04-18',
    'Pipeline Lipponen adapte judo v3.3. Regle post-reconnexion : post_reconnect_deco exclue, post_reconnect_artefact visible FC non RMSSD.',
    '{"rr_min_ms":250,"rr_max_ms":1500,"alpha":5.2,"window_th":91,"window_median":11,"threshold_drr":1.0,"threshold_mrr":3.0,"seuil_run_court_max":6,"seuil_run_moyen_max":15,"seuil_deco_interpoler":15,"window_densite":20,"seuil_densite":0.5,"y_court":5,"tolerance_court":2,"y_moyen":6,"tolerance_moyen":3,"chauffe_gap_court":5,"chauffe_gap_long":8,"seuil_gap_duree_ms":15000,"seuil_qualite_court":0.05,"seuil_qualite_long":0.15,"duree_seuil_segment_ms":120000}',
    1
);
