# Contexte de session

## A lire en premier

Au debut de chaque nouvelle session, demander explicitement :

`Relis context.md et reprends le projet`

Rappel :

- `context.md` sert de point de reprise rapide
- `changelog.md` sert d'historique detaille
- en fin de session, `context.md` doit etre mis a jour

Derniere mise a jour : 2026-04-19

## But du document

Ce fichier sert de relais entre sessions pour reprendre rapidement le projet sans reconstituer tout l'historique.
Il complete `changelog.md` :

- `changelog.md` = historique detaille des modifications
- `context.md` = etat courant, conventions, points d'attention, prochaine reprise

## Resume du projet

Application Python / Streamlit pour exploiter des donnees Polar H10 :

- import de fichiers RR / HR Polar
- structuration des seances dans `data/raw`, `data/processed`, `data/clean`
- nettoyage RR avec pipeline metier dedie
- comparaison `FC clean` vs `FC Polar`
- gestion et annotation des activites
- visualisation finale des activites annotees

## Point d'entree et lancement

Lancement principal de l'interface :

```powershell
streamlit run streamlit_app.py
```

CLI disponible via :

```powershell
python main.py
```

Le CLI appelle `polar_app.cli.run_cli()` et peut importer ou afficher les seances.

## Structure utile du depot

- `streamlit_app.py` : page d'accueil / dashboard principal
- `pages/1_Import_Polar.py` : import des fichiers Polar
- `pages/2_Nettoyage_RR.py` : nettoyage RR et visualisation detaillee
- `pages/3_FC_Clean_vs_Polar.py` : comparaison FC nettoyee vs FC Polar
- `pages/4_Analyse_RR.py` : synthese et recompilation des exports clean
- `pages/5_Gestion_activites.py` : edition des metadonnees et segmentation temporelle
- `pages/6_Visualisation_activite.py` : restitution finale des activites annotees
- `polar_app/rr_pipeline.py` : logique metier principale du nettoyage RR
- `polar_app/clean_export.py` : export des donnees clean et version d'algo
- `polar_app/repository.py` : acces / mise a jour des sessions et metadonnees
- `polar_app/models.py` : dataclasses metier
- `polar_app/fc_segment_component.py` : integration du composant front de segmentation FC
- `frontend/fc_segment_editor/index.html` : composant front local pour l'edition des segments
- `liste_activite.txt` : referentiel des types d'activite
- `seance_judo_phase.txt` : referentiel des phases judo
- `changelog.md` : historique detaille des evolutions
- `schema_v1_1.sql` : schema SQLite cible v1.1 pour la migration BDD
- `migrate_to_db.py` : script de migration fichiers locaux -> SQLite

## Etat fonctionnel courant

### Import et stockage

- Les donnees Polar sont importees puis rangees dans :
  - `data/raw/<session_id>/`
  - `data/processed/<session_id>/`
  - `data/clean/<session_id>/`
- Les gaps entre segments importes sont representes par des placeholders `GAP`.
- La fusion des segments importes existe deja avec regles metier dediees.

### Pipeline RR

- La version courante de l'algo clean est `3.3` dans `polar_app/clean_export.py`.
- Le pipeline produit une serie `RR_clean` enrichie avec labels, flags, corrections et indicateurs d'exploitabilite.
- Les zones denses d'artefacts sont marquees separement.
- Les coupures de plus de 60 secondes ne sont pas interpolees.
- Les zones de chauffe post-reconnexion apres vraie deconnexion font partie des zones d'exclusion et ne doivent plus alimenter les indicateurs d'exploitabilite.
- Exception : la chauffe apres cassure sur serie d'artefacts reste visible dans `RR clean` et peut encore alimenter la FC, mais pas le RMSSD.
- Le recalage temporel du clean reutilise les timestamps Polar d'origine quand ils existent.
- Les segments trop degradÃ©s sont exclus selon des seuils dependants de leur duree.
- Les exports clean ecrivent au minimum :
  - `rr_clean.parquet`
  - `fc_clean.parquet`
  - `clean_meta.json`

### Analyse RR

- La page Analyse RR sait afficher les stats de qualite et recompiler les exports clean manquants ou obsoletes.
- La detection d'obsolescence s'appuie sur la version d'algo de nettoyage.
- La selection s'appuie maintenant sur ctivity_label, puis permet de choisir plusieurs seances d'une meme activite pour afficher leurs graphes RR en parallele.
- Les graphes peuvent montrer plusieurs seances en meme temps, tandis que les indicateurs et tableaux detailles restent pilotes par une seance active.

### Gestion des activites

- Une page dediee permet d'annoter les seances avec metadonnees metier.
- Les activites peuvent etre archivees logiquement, sans suppression physique des donnees.
- Les seances judo / randoris ont une logique specifique de phases, blocs randoris, RPE global et RPE par randori.
- Un composant front local Plotly permet l'edition interactive des segments temporels FC.
- La page `Gestion activites` est orientee edition.
- La page `Visualisation activite` est orientee restitution finale.

## Conventions importantes

- Si une modification change reellement la pipeline de nettoyage RR, il faut demander explicitement s'il faut augmenter la version de l'algo :
  - `+0.1` pour une evolution mineure
  - `+1` pour une evolution majeure
- `changelog.md` doit rester la source de verite de l'historique detaille.
- `context.md` doit etre mis a jour en fin de session pour capturer :
  - l'objectif en cours
  - l'etat reel atteint
  - les blocages
  - la prochaine action recommandee

## Dependances reperees

Dans `requirements.txt` :

- `pandas`
- `plotly`
- `streamlit`
- `streamlit-calendar`

## Etat de reprise recommande

Au debut d'une nouvelle session :

1. Lire `context.md`.
2. Lire la derniere section de `changelog.md`.
3. Verifier si le besoin du moment concerne :
   - import Polar
   - pipeline RR
   - exports clean
   - annotation activite
   - visualisation finale
   - autres
4. Si le besoin touche le nettoyage RR, verifier si la version `3.3` doit rester valide ou etre incrementee.

## A completer apres chaque session

### Objectif courant

- Basculer le modele de donnees vers SQLite DB-first avec une table `phases_realisees`, tout en conservant les JSON comme copie historique/fallback.
- Etat : `schema_v1_1.sql`, `polar_app/db.py`, `migrate_to_db.py` et `projet_i.db` sont en place. La migration reelle est idempotente et remplit 15 sessions, 15 acquisitions, 28 phases programmees, 117 phases realisees FC, 49 randoris realises, 12 RPE par randori et 43 annotations RR manuelles. Le clean orphelin `20260318_102514` reste ignore avec log.
- `ProcessedSessionRepository` lit maintenant SQLite en priorite pour la liste des seances, les metadonnees principales, les chemins Parquet, la segmentation temporelle, les RPE par randori et les annotations RR manuelles. Les ecritures metier conservent une copie JSON puis resynchronisent la seance dans SQLite.

### Dernieres decisions actives

- Pipeline RR versionnee en `3.3`.
- La migration SQLite conserve les fichiers Parquet/TXT comme source des signaux ; la BDD stocke metadonnees, chemins, stats clean, phases programmees, phases realisees, randoris realises, RPE et annotations RR manuelles.
- Les segments temporels observes sont stockes d'abord dans `phases_realisees` avec `source_annotation = annotation_fc` et `timing_annotation = post`.
- Les randoris derives des annotations FC sont des lignes enfant de `phases_realisees` via `randoris_realises.phase_realisee_id`, avec `source_segmentation = annotation_fc`.
- Les annotations RR manuelles sont stockees dans la table `rr_manual_annotations` car elles representent une decision utilisateur persistante distincte des artefacts Parquet clean.
- `athlete_b_id` peut rester `NULL` quand l'adversaire n'est pas renseigne.
- L'athlete principal migre est Romain Valadier (`romain_valadier`, 64 kg, 165 cm, M, 2002-07-20, `romanopic@gmail.com`).
- Les timestamps Polar sans timezone sont interpretes comme Europe/Paris puis convertis en UTC pour la BDD.
- `Gestion activites` sert a l'edition.
- `Visualisation activite` sert a la restitution finale.
- Les zones de chauffe apres vraie deconnexion sont des zones exclues a part entiere.
- Les zones `post_reconnect_artefact` ne sont pas exclues : elles restent visibles dans `RR clean`, contribuent a `fc_ok`, mais pas a `rmssd_ok`.
- Le graphe `RR bruts` affiche une ligne continue des RR bruts, avec mise en avant des seuls labels non `ok`, des labels iteratifs par pass quand le mode iteratif est actif, et des labels 2 bis non `aucun`.
- La post-classification 2 bis ne conserve plus que la passe A ; la passe B a ete retiree car elle reclassait a tort certains beats `court` en `ok`.
- L'ordre experimental de calcul sur la page `Nettoyage RR` est maintenant : pass 1 standard -> iteration Lipponen -> passe A 2 bis.
- Le flag `manuel` est une surcouche de revue locale persistante par seance ; il ne modifie pas l'algo automatique lui-meme et ne doit donc pas forcer de bump de version d'algo.
- Les points marques `manuel` sont affiches comme `manuel` dans l'analyse, mais convertis en `long` uniquement au moment de relancer la correction aval.
- La selection manuelle sur `RR bruts` est maintenant par lot : clics locaux sans rerun, puis validation explicite d'un ensemble de beats.
- Le bloc automatique amont est maintenant cache par combinaison `session + params v3 + params iteratif + params 2 bis`, afin d'eviter de recalculer l'iteratif et le 2 bis a chaque validation manuelle.
- `RR clean` n'utilise plus Altair pour l'affichage principal ; il passe par un composant Plotly local avec pan, scroll zoom, reset double-clic et plein ecran.

### Prochaine reprise conseillee

- Tester les pages Streamlit en interaction reelle : modification de metadonnees, segmentation FC, annotations RR, archive/restauration.
- Verifier apres chaque action que SQLite change bien, puis que les JSON restent une copie coherente.
- Si la bascule DB-first est stable, reduire progressivement les lectures JSON restantes aux seuls cas de fallback ou aux champs pas encore normalises en BDD.

### Blocages / points a surveiller

- Le Python systeme disponible dans le terminal ne charge pas correctement `numpy/pandas`; utiliser `.\.venv\Scripts\python.exe` pour les migrations et verifications Parquet.
- Les commandes Git doivent etre lancees avec `-c safe.directory=C:/5eme/stage/V1_project` dans ce terminal.
- Le cache de la page `Nettoyage RR` vit dans `st.session_state` et doit etre invalide par toute modification de seance ou de parametres ; si un comportement parait incoherent, verifier d'abord la cle de cache et les objets clones.
- La migration reelle lit `rr_clean.parquet` pour remplir `rr_clean_exploitabilite`; elle necessite donc un environnement Python ou `pandas.read_parquet` fonctionne.
