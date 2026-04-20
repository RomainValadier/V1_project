CREATE TABLE individus (
    id TEXT PRIMARY KEY,
    nom TEXT NOT NULL,
    prenom TEXT NOT NULL,
    email TEXT NOT NULL UNIQUE,
    sexe TEXT NOT NULL,
    date_naissance TEXT,
    role TEXT NOT NULL,
    is_active INTEGER NOT NULL,
    password_hash TEXT,
    last_login_at TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE athletes (
    individu_id TEXT PRIMARY KEY REFERENCES individus(id),
    categorie_poids TEXT,
    taille_cm REAL
);

CREATE TABLE reference_tests (
    id TEXT PRIMARY KEY,
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id),
    date_enregistrement TEXT NOT NULL,
    fc_max_bpm INTEGER,
    fc_repos_bpm INTEGER,
    hrr_1min_bpm INTEGER,
    hrr_2min_bpm INTEGER,
    vo2max_ml_kg_min REAL,
    sv1_pct_fcmax REAL,
    sv2_pct_fcmax REAL,
    poids_kg REAL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    date_seance TEXT NOT NULL,
    heure_debut_reelle TEXT,
    lieu TEXT,
    type_seance TEXT NOT NULL,
    activity_family TEXT,
    activity_notes TEXT,
    operateur TEXT NOT NULL,
    is_archived INTEGER NOT NULL,
    archived_at TEXT,
    is_temporally_annotated INTEGER NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE session_tatamis (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    tatami_id TEXT NOT NULL,
    camera_id TEXT,
    PRIMARY KEY (session_id, tatami_id)
);

CREATE TABLE session_athletes (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id),
    rpe_global_seance INTEGER,
    rpe_global_timestamp TEXT,
    PRIMARY KEY (session_id, athlete_id)
);

CREATE TABLE phases_programmees (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    ordre INTEGER NOT NULL,
    type_phase TEXT NOT NULL,
    nb_repetitions INTEGER,
    duree_cible_s INTEGER,
    recuperation_cible_s INTEGER,
    rpe_cible INTEGER,
    notes_entraineur TEXT
);

CREATE TABLE phases_realisees (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    phase_programmee_id TEXT REFERENCES phases_programmees(id),
    ordre INTEGER NOT NULL,
    type_phase TEXT NOT NULL,
    t_debut_utc_ms INTEGER NOT NULL,
    t_fin_utc_ms INTEGER NOT NULL,
    duree_reelle_s REAL,
    source_annotation TEXT NOT NULL,
    timing_annotation TEXT NOT NULL,
    source_segment_index INTEGER,
    notes TEXT
);

CREATE TABLE randoris_programmes (
    id TEXT PRIMARY KEY,
    phase_programmee_id TEXT NOT NULL REFERENCES phases_programmees(id),
    numero INTEGER NOT NULL,
    type_randori TEXT NOT NULL,
    duree_cible_s INTEGER NOT NULL
);

CREATE TABLE pipeline_versions (
    version TEXT PRIMARY KEY,
    date_deploy TEXT NOT NULL,
    description TEXT,
    params_json TEXT,
    is_current INTEGER NOT NULL
);

CREATE TABLE acquisitions (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id),
    pipeline_version TEXT REFERENCES pipeline_versions(version),
    source TEXT NOT NULL,
    device_id TEXT,
    mac_address TEXT,
    adapter_id TEXT,
    start_time_utc TEXT NOT NULL,
    end_time_utc TEXT,
    nb_rr_bruts INTEGER,
    nb_points_hr INTEGER,
    duree_s REAL,
    segments_count INTEGER,
    gaps_count INTEGER,
    nb_deconnexions INTEGER,
    batterie_pct_debut INTEGER,
    fc_offset_s REAL,
    rr_raw_filepath TEXT NOT NULL,
    hr_raw_filepath TEXT,
    qualite_signal TEXT,
    statut TEXT NOT NULL,
    notes TEXT,
    created_at TEXT NOT NULL
);

CREATE TABLE rr_clean_meta (
    acquisition_id TEXT PRIMARY KEY REFERENCES acquisitions(id),
    export_timestamp TEXT NOT NULL,
    ok_rr_total INTEGER,
    global_quality_label TEXT
);

CREATE TABLE rr_clean_exploitabilite (
    acquisition_id TEXT PRIMARY KEY REFERENCES acquisitions(id),
    fc_ok_total INTEGER,
    fc_non_viable_total INTEGER,
    fc_non_ok_rate REAL,
    hrr_ok_total INTEGER,
    hrr_non_viable_total INTEGER,
    hrr_non_ok_rate REAL,
    rmssd_ok_total INTEGER,
    rmssd_non_viable_total INTEGER,
    rmssd_non_ok_rate REAL
);

CREATE TABLE rr_indicator_files (
    id TEXT PRIMARY KEY,
    acquisition_id TEXT NOT NULL REFERENCES acquisitions(id),
    pipeline_version TEXT REFERENCES pipeline_versions(version),
    indicator_type TEXT NOT NULL,
    filepath TEXT NOT NULL,
    params_json TEXT,
    computed_at TEXT NOT NULL
);

CREATE TABLE import_log (
    id TEXT PRIMARY KEY,
    acquisition_id TEXT NOT NULL REFERENCES acquisitions(id),
    fichier_source TEXT NOT NULL,
    type_fichier TEXT NOT NULL,
    date_import TEXT NOT NULL,
    nb_lignes INTEGER,
    statut TEXT NOT NULL,
    message TEXT
);

CREATE TABLE indicators_fc (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id),
    acquisition_id TEXT NOT NULL REFERENCES acquisitions(id),
    pipeline_version TEXT NOT NULL REFERENCES pipeline_versions(version),
    fc_peak_bpm REAL,
    fc_mean_bpm REAL,
    fc_min_bpm REAL,
    computed_at TEXT NOT NULL
);

CREATE TABLE randoris_realises (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    phase_realisee_id TEXT UNIQUE REFERENCES phases_realisees(id),
    randori_programme_id TEXT REFERENCES randoris_programmes(id),
    athlete_a_id TEXT NOT NULL REFERENCES athletes(individu_id),
    athlete_b_id TEXT REFERENCES athletes(individu_id),
    numero_randori INTEGER NOT NULL,
    tatami_id TEXT,
    t_hajime_utc_ms INTEGER NOT NULL,
    t_matte_utc_ms INTEGER NOT NULL,
    duree_reelle_s REAL,
    type_randori TEXT NOT NULL,
    source_segmentation TEXT NOT NULL,
    confiance_segmentation REAL,
    notes TEXT
);

CREATE TABLE indicators_video (
    id TEXT PRIMARY KEY,
    randori_id TEXT NOT NULL UNIQUE REFERENCES randoris_realises(id),
    duree_kumi_kata_s REAL,
    nb_kumi_kata INTEGER,
    nb_attaques_a INTEGER,
    nb_tentatives_a INTEGER,
    nb_defense_a INTEGER,
    nb_chute_a INTEGER,
    nb_attaques_b INTEGER,
    nb_tentatives_b INTEGER,
    nb_defense_b INTEGER,
    nb_chute_b INTEGER,
    duree_ne_waza_s REAL,
    nb_breaks INTEGER,
    duree_breaks_s REAL,
    duree_pre_contact_s REAL,
    confiance_identification_moy REAL,
    perte_identite_flag INTEGER NOT NULL,
    frames_filepath_a TEXT,
    frames_filepath_b TEXT,
    source_fichier_video TEXT,
    imported_at TEXT
);

CREATE TABLE rpe (
    id TEXT PRIMARY KEY,
    randori_id TEXT NOT NULL REFERENCES randoris_realises(id),
    athlete_id TEXT NOT NULL REFERENCES athletes(individu_id),
    rpe_ressenti INTEGER NOT NULL,
    moment_collecte TEXT NOT NULL,
    timestamp_collecte TEXT NOT NULL
);

CREATE TABLE rr_manual_annotations (
    session_id TEXT NOT NULL REFERENCES sessions(id),
    source_index INTEGER NOT NULL,
    t_offset_ms INTEGER,
    rr_interval_ms REAL,
    manual_flag TEXT NOT NULL,
    created_at TEXT,
    updated_at TEXT,
    PRIMARY KEY (session_id, source_index)
);
