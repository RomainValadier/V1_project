# Contexte de session

## A lire en premier

Au debut de chaque nouvelle session, demander explicitement :

`Relis context.md et reprends le projet`

Rappel :

- `context.md` sert de point de reprise rapide
- `changelog.md` sert d'historique detaille
- en fin de session, `context.md` doit etre mis a jour

Derniere mise a jour : 2026-04-06

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

## Etat fonctionnel courant

### Import et stockage

- Les donnees Polar sont importees puis rangees dans :
  - `data/raw/<session_id>/`
  - `data/processed/<session_id>/`
  - `data/clean/<session_id>/`
- Les gaps entre segments importes sont representes par des placeholders `GAP`.
- La fusion des segments importes existe deja avec regles metier dediees.

### Pipeline RR

- La version courante de l'algo clean est `3.1` dans `polar_app/clean_export.py`.
- Le pipeline produit une serie `RR_clean` enrichie avec labels, flags, corrections et indicateurs d'exploitabilite.
- Les zones denses d'artefacts sont marquees separement.
- Les coupures de plus de 60 secondes ne sont pas interpolees.
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
4. Si le besoin touche le nettoyage RR, verifier si la version `3.1` doit rester valide ou etre incrementee.

## A completer apres chaque session

### Objectif courant

- Etendre Analyse RR pour filtrer par activite et visualiser plusieurs seances d'une meme activite dans des graphes separes.
- Etat : fonctionnalite implementee dans `pages/4_Analyse_RR.py`.

### Dernieres decisions actives

- Pipeline RR versionnee en `3.1`.
- `Gestion activites` sert a l'edition.
- `Visualisation activite` sert a la restitution finale.

### Prochaine reprise conseillee

- Verifier visuellement dans Streamlit le comportement de la multi-selection sur `Analyse RR`, notamment le changement d'activite, la seance active et les cas sans `activity_label`.

### Blocages / points a surveiller

- Aucun blocage explicite documente dans ce fichier pour l'instant.
- Le prochain travail devra preciser s'il concerne plutot le pipeline RR ou l'UX des pages activite.


