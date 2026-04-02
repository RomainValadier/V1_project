# Changelog

Ce fichier suit les modifications apportees au projet suite a tes demandes.

## 2026-03-30

### Pipeline RR / nettoyage

- Mise en place initiale d'une page Streamlit dediee au nettoyage RR avec affichage des RR bruts et nettoyes, des variables intermediaires de l'algo, des parametres, des compteurs d'erreurs par type, du total d'artefacts corriges, du pourcentage corrige, et des sequences de plus de 4 artefacts consecutifs.
- Integration du pseudo-code texte de nettoyage RR dans le moteur de nettoyage avec calcul de `dRR`, `Th1`, `dRR_norm`, `mRR`, `Th2`, `mRR_norm`, sous-espaces de decision, classification des artefacts, et correction differenciee.
- Clarification du flag qualite : separation entre le `quality_flag` base sur le pourcentage corrige et un flag specifique pour les clusters consecutifs d'artefacts.
- Ajout du comptage explicite des clusters consecutifs et de leur mise en evidence dans les sorties d'analyse.
- Retrait puis reintroduction du bloc de filtrage absolu (`RR < 250 ms` ou `RR > 1500 ms`) selon les demandes, avec reintegration de la categorie `artefact_absolu` dans les compteurs, tableaux et graphes.
- Correction de plusieurs ecarts vis-a-vis du pseudo-code : meilleure gestion des bords (`NaN` au lieu de `0`), mise a jour du statut apres fusion, clarification des seuils, et amelioration des sorties d'analyse.
- Remplacement de l'ancienne interpolation polynomiale locale par une vraie interpolation `CubicSpline`, puis remplacement ulterieur de toute interpolation cubique par `PchipInterpolator` avec renommage complet des libelles et dependances vers `interpolation_pchip`.
- Refactorisation majeure du moteur vers une pipeline RR plus riche dans `rr_pipeline.py`, avec nouveaux objets metier et nouvelles sorties d'analyse.
- Alignement du pipeline sur la representation actuelle des deconnexions importees (`pipeline_flag = GAP` et non `RR = 0` en base traitee).
- Simplification des flags post-reconnexion vers seulement `post_reconnect_court` et `post_reconnect_long` au niveau metier.
- Suppression du critere ectopique dans la version v3 du pipeline de nettoyage.
- Introduction des notions `deco_flag`, `label`, `run_flag`, `correction_flag` et production d'une serie `RR_clean` enrichie.
- Mise en place d'une logique de segmentation plus fine : segments initiaux, cassures, sous-segments, qualite segmentaire, et cassures artefact sur `run_long`.
- Changement de la regle sur les coupures longues : une coupure de plus de 60 secondes n'est plus interpolee et le segment reste casse.
- Changement de la gestion de la densite d'artefacts : la densite ne cree plus de cassure, les points restent corriges par l'algo mais sont identifies via un marquage dedie.
- Correction de la detection des zones denses : marquage de toute fenetre dense, fusion des fenetres denses qui se chevauchent ou se touchent, et creation d'une table de synthese par zone dense.
- Creation d'un flag dense separe `run_series_flag` afin de ne pas ecraser `run_flag`.
- Ajout de la qualification d'exploitabilite par indicateur (`fc_ok`, `hrr_ok`, `rmssd_ok`) dans la serie clean.
- Mise en place d'un critere d'exclusion segmentaire : seuil a 5 % pour les segments de moins de 120 s et a 15 % pour les segments de 120 s ou plus, avec exclusion des segments trop degrades des donnees nettoyees.
- Les `RR_NON_VIABLE` ne sont plus traces dans la courbe nettoyee.
- Correction du recalage temporel de la serie clean : abandon du temps reconstruit par cumul, et reutilisation des timestamps Polar d'origine quand ils existent afin d'eviter la derive croissante avec la FC Polar.

### Import / structuration des donnees RR

- Conservation de la logique d'import actuelle avec placeholders `GAP` entre segments fusionnes, au lieu de stocker des `RR = 0` dans les donnees traitees.
- Evolution de la logique de fusion des sessions importees : seuils `auto < 30 s`, confirmation utilisateur sur les gaps intermediaires, et non-fusion par defaut au-dela du seuil defini.
- Generation de placeholders de gap a granularite estimee sur les RR manquants au lieu d'un simple pas fixe grossier.
- Remplacement progressif de `quality_flag` par `pipeline_flag` dans les structures traitees et l'UI liee au nettoyage.

### Page `Nettoyage RR`

- Creation puis enrichissement de la page [2_Nettoyage_RR.py](C:/5eme/stage/V1_project/pages/2_Nettoyage_RR.py).
- Regroupement des parametres par familles dans des sous-fenetres dediees pour rendre la page plus lisible.
- Ajout, puis ajustements, de l'interaction de validation des interpolations sur grands gaps ; ensuite restriction de cette logique en relevant les seuils pour limiter le cout de calcul, puis simplification des regles de cassure longues.
- Ajout du zoom / pan sur les graphes RR.
- Affichage des artefacts sur le graphe RR brut avec une couleur selon la classe du point.
- Affichage des points corriges sur le graphe RR nettoye avec la meme logique de couleur par type de correction, puis ajout de formes distinctes par methode (`division`, `fusion`, `interpolation_pchip`, `interpolation_lineaire`).
- Mise en evidence des sequences de plus de 4 artefacts consecutifs, puis ajustement pour ne plus les afficher lorsqu'une zone dense recouvre la meme zone.
- Mise en evidence des zones denses sur le graphe `RR clean` a partir des vraies bornes des zones fusionnees, avec correction du positionnement des rectangles pour qu'ils collent a la serie clean.
- Ajout dans le graphe `RR clean` d'un affichage explicite des points nettoyes et des zones denses d'artefacts.
- Correction de l'affichage des points `interpolation_pchip` dans `RR clean` et mise a jour des libelles visibles.
- Ajout de tableaux plus riches : segments initiaux, segments finaux, qualite des segments, cassures, zones denses, variables intermediaires, RR bruts, RR nettoyes.
- Ajustement de la table `RR bruts` pour que les valeurs utiles et les flags attendus restent visibles suivant les demandes successives.
- Ajout des colonnes utiles dans les tableaux : methode de correction, statut post-correction, `run_series_flag`, densite, zones denses, indicateurs d'exploitabilite.

### Exports `clean` et page `FC clean vs Polar`

- Creation d'un stockage persistant `data/clean/<session_id>/` avec export automatique des donnees RR nettoyees.
- Ecriture de `rr_clean.parquet`, `fc_clean.parquet` et `clean_meta.json` pour chaque seance nettoyee.
- Calcul d'une FC clean sur fenetre centree de 5 battements, avec trou dans la courbe si moins de 2 RR viables sont presents dans la fenetre.
- Creation de la page [3_FC_Clean_vs_Polar.py](C:/5eme/stage/V1_project/pages/3_FC_Clean_vs_Polar.py) pour comparer `FC clean` et `FC Polar` sur un axe temporel absolu demarrant a l'heure de debut de seance.
- Correction de bugs d'affichage et de compatibilite de cette page : import Altair, correction de shapes Altair, prise en charge d'anciens exports clean ne contenant pas encore toutes les colonnes v3 (`fc_ok`, etc.).
- Adaptation des exports clean et de la page FC pour prendre en compte la version recente du pipeline RR et les flags d'exploitabilite.

### Page `Analyse RR`

- Creation de la page [4_Analyse_RR.py](C:/5eme/stage/V1_project/pages/4_Analyse_RR.py).
- Ajout d'un bouton pour compiler ou recompiler automatiquement les `RR_clean` manquants ou obsoletes.
- Ajout du suivi d'avancement pendant la recompilation avec barre de progression et indication de la seance en cours de nettoyage.
- Affichage, pour une seance selectionnee, du nombre de segments, du pourcentage d'artefacts (`RR non ok`), du pourcentage de RR non viables (`gap_deco`, `post_reconnect_deco`), du detail des zones denses, des segments, des cassures et du taux d'artefact par label.
- Ajout dans `Analyse RR` de deux visualisations du `RR clean` : une version annotee et une version sans annotation.

### Versionnement du clean

- Ajout de la version de l'algo de nettoyage dans les exports clean via `RR_CLEAN_ALGO_VERSION` et `clean_meta.json`.
- Affichage de la version de l'algo courante et de la version de l'export clean dans la page `Analyse RR`.
- Mise en place d'une logique de detection des exports clean a recompiler lorsque leur version est absente ou differente de la version courante.
- Regle de travail fixee pour les evolutions futures : a chaque demande modifiant reellement la pipeline de nettoyage RR, demander explicitement s'il faut faire passer la version de l'algo en `+0.1` (mineur) ou `+1` (majeur).

### Notes diverses

- Plusieurs correctifs d'UI et de robustesse ont ete appliques au fil de l'eau (imports manquants, erreurs Altair, compatibilite avec anciens exports, affichage dynamique des legendes, correction des tooltips et des tables).
- `changelog.md` etend maintenant fortement le niveau de detail du suivi par rapport a sa premiere version tres concise.

## 2026-04-01

### Gestion des activites

- Ajout de la dependance `streamlit-calendar` dans `requirements.txt` pour preparer une vraie vue calendrier cliquable dans Streamlit.
- Creation de la nouvelle page `pages/5_Gestion_activites.py` avec affichage calendrier, filtre par famille, filtre `annotees seulement`, prise en charge des archivees, panneau detail, edition des metadonnees d'activite, et affichage de la FC brute.
- Ajout d'un fallback de selection simple si `streamlit-calendar` n'est pas encore installe localement, afin que la page reste exploitable meme sans composant calendrier actif.

### Metadonnees de seance

- Extension du dataclass `ProcessedSession` avec les champs d'annotation metier, d'archivage logique, de type de seance judo, de phases judo, de blocs randoris, et de notes generales.
- Conservation de la compatibilite avec les anciennes `session_meta.json` grace a des champs optionnels et a la logique existante de `from_dict()`.
- Ajout d'un flag persistant `is_activity_annotated` et de son horodatage `activity_annotated_at`.

### Repository activites

- Extension de `ProcessedSessionRepository` avec `update_activity_metadata()`, `archive_session()`, `restore_session()`, `build_calendar_events()` et un filtrage des sessions archivees dans `list_sessions()`.
- Mise en place d'un archivage logique base sur `is_archived` / `archived_at`, sans suppression physique des donnees `raw`, `processed` ou `clean`.
- Generation d'evenements calendrier colores selon la famille d'activite et mis en evidence lorsqu'une seance n'est pas encore annotee.
- Conservation de `update_annotation()` en l'adossant maintenant a la nouvelle logique metier centralisee.

### References activite / judo

- Normalisation de `liste_activite.txt` en liste explicite de labels d'activites, incluant `judo > randoris > NW pure`.
- Remplacement de `phase_rando.txt` par `seance_judo_phase.txt` avec les phases judo v1 : `echauffement`, `technique`, `randoris TW`, `randoris NW pure`, `randoris libres`, `autres`.
- Prise en charge dans l'UI des blocs randoris avec nombre, duree d'un randori, repos entre randoris, et valeurs `NA` quand l'information n'est pas connue.
- Mise en evidence visuelle de l'activite actuellement selectionnee dans le calendrier de `Gestion activites`, avec une couleur dediee pour la reperer immediatement.
- Correction de la synchronisation de selection dans le calendrier de `Gestion activites` : un clic sur une activite declenche maintenant un rerun immediat pour que la couleur de selection et le panneau de detail se mettent a jour sans decalage.
- Simplification du formulaire d'annotation dans `Gestion activites` : l'utilisateur choisit d'abord la famille, puis les champs de precision n'apparaissent que lorsque necessaire (`prepa` detaillee, `judo` par type, et `autres` / `judo > technique` auto-renseignes).
- Ajout de la dependance `plotly` et d'un module d'annotations temporelles FC pour les seances `judo > randoris` dans `Gestion activites`.
- Ajout d'un stockage persistant `fc_phase_segments` ainsi que du flag `is_temporally_annotated` et de son horodatage, limites aux seules seances `judo > randoris`.
- Mise en place d'une auto-generation des segments temporels FC a partir du premier passage au-dessus de `150 bpm`, avec alternance `randori / recuperation` lorsque `randori_count` et `randori_duration_min` sont renseignes, et recuperation par defaut a `2 min` si seule cette valeur manque.
- Ajout d'un graphe Plotly annote pour la FC brute, avec edition directe des bornes de segments a partir d'un point clique sur le graphe, ajout/suppression manuelle de segments, et tableau resume synchronise des phases et de leurs durees.
- Refonte du module d'annotations temporelles FC pour les seances `judo > randoris` vers une HMI plus directe : suppression des boutons de confirmation, auto-save apres chaque modification valide, et mise en avant visuelle du segment actif dans le graphe Plotly.
- Retrait des faux segments `debut_seance` et `fin_seance` du modele temporel persiste ; le debut et la fin d'enregistrement restent des bornes implicites affichees dans le graphe.
- Evolution de l'auto-generation pour s'appuyer d'abord sur les `judo_phases` deja enregistrees, reutiliser les `judo_randori_blocks` quand ils sont complets, et detecter le debut de la zone randori a partir du premier passage au-dessus de `160 bpm`.
- Ajout d'une normalisation temporelle orientee timeline dans le repository : lorsqu'un segment est ajuste ou deplace, les segments voisins sont automatiquement recales pour garder une seance coherente sans recouvrement.
- Amelioration de l'edition dans `Gestion activites` avec modes de clic graphe (`placer debut`, `placer fin`, `deplacer segment`), micro-ajustements rapides (`+-5 s`, `+-15 s`), et normalisation automatique des anciennes segmentations contenant encore des segments de bord obsoletes.
- Ajustement de l'interaction graphe dans `Gestion activites` : une borne temporelle deja selectionnee peut maintenant etre appliquee immediatement a un autre segment ou a un autre mode d'edition sans exiger un nouveau clic distinct sur le graphe.
- Ajustement du flux d'edition temporelle : le changement de segment actif vide maintenant le point graphe precedemment selectionne pour imposer un processus clair `selection du segment puis modification de ce segment`.
- Ajout d'un composant front local `frontend/fc_segment_editor/` embarquant Plotly.js pour l'annotation manuelle interactive des segments FC dans `Gestion activites`.
- La selection du segment actif se fait maintenant au clic gauche dans une zone de segment, avec bandeau d'information mis a jour et mise en evidence visuelle du segment selectionne.
- Les bords internes des segments peuvent etre redimensionnes par drag direct, avec contiguite stricte des segments, duree minimale fixee a 10 secondes, ligne verticale pointillee pendant le drag, tooltip temporel `mm:ss`, et mise a jour en temps reel des durees affichees.
- Ajout d'un export CSV des segments temporels (`segment_id`, `type`, `t_debut_s`, `t_fin_s`, `duree_s`) et extension des labels de phases autorises a `retour_calme` et `autre`.
- Correction de l'integration Streamlit du composant d'edition FC : la declaration `declare_component` a ete deplacee hors de la page vers `polar_app/fc_segment_component.py` pour eviter l'erreur de contexte d'execution `module is None` au chargement de la page multipage.
- Evolution de l'auto-generation temporelle FC : les phases judo deja renseignees servent maintenant de squelette direct, les randoris sont detectes sur une montee rapide au-dessus de `160 bpm`, les recuperations demarrent lors d'une forte rechute vers `140 bpm`, et le dernier segment auto-genere est aligne par defaut sur la fin de la seance.
- Ajout de bornes d'edition explicites `debut` et `fin` dans le composant front d'annotation FC, avec survol nomme et redimensionnement direct de la fenetre de seance depuis le graphe.
- Ajustement de l'auto-generation des seances `judo > randoris` pour respecter `randori_count` : chaque bloc randori genere maintenant le bon nombre de segments `randori`, avec recuperations intermediaires basees sur la duree de repos renseignee ou sur la valeur par defaut.
- Ajout dans `Gestion activites` d'un controle d'insertion manuelle d'un segment apres le segment actif, avec choix du type de phase, puis d'un bouton `Valider segmentation` qui persiste la timeline courante et rafraichit le resume des phases juste en dessous.
- Ajustement de l'HMI de segmentation temporelle : les controles `Type du nouveau segment`, `Ajouter apres segment actif` et `Valider segmentation` sont maintenant places sous le graphique pour suivre le flux naturel d'edition visuelle.
- Ajout dans le panneau `Segment actif` d'un selecteur pour modifier le type de la phase selectionnee (`echauffement`, `technique`, `randori`, `recuperation`, `retour_calme`, `autre`) sans modifier ses bornes temporelles.
- Correction de la selection du dernier segment dans l'editeur FC : un clic sur la derniere zone ne bascule plus par erreur sur le segment precedent lorsque le curseur tombe au niveau d'une borne partagee.
- Refonte de l'HMI du panneau `Segment actif` : edition regroupee du type et des bornes, actions explicites `Ajouter a gauche`, `Ajouter a droite`, `Supprimer`, et recalage automatique de la timeline apres insertion ou suppression.
- Correction de la synchronisation entre le clic dans le graphe FC et le panneau `Segment actif` : un changement de segment selectionne declenche maintenant un rerun immediat pour afficher sans decalage les bonnes informations et actions d'edition.
- Ajout d'un `RPE global de seance` persistant dans les metadonnees d'activite, avec affichage dans le resume de la seance et edition directe depuis `Gestion activites`.
- Extension des blocs `judo > randoris` pour stocker un `RPE` et un petit commentaire pour chaque repetition de randori, selon le nombre de randoris renseigne dans la phase.
- Ajustement du flux d'edition des seances `judo > randoris` : l'utilisateur construit d'abord toute la sequence des phases, clique sur `Enregistrer`, puis seulement ensuite les formulaires de detail par phase randori apparaissent ; tant que la sequence courante n'est pas enregistree, les details fins restent masques pour eviter les reinitialisations pendant la construction de seance.
- Correction d'un decalage intermittent entre le graphe FC et le panneau `Segment actif` : le composant front renvoie maintenant aussi l'index du segment selectionne, ce qui stabilise la synchronisation avec le panneau d'edition Streamlit meme quand l'identifiant interne et la position visuelle divergent temporairement.
- Ajout d'une vue `Synthese finale` pour les seances totalement annotees : hero visuel avec illustration adaptee au type d'entrainement, resume des informations clefs, graphe FC segmente plus lisible, cartes de phases et detail du `RPE` global puis des `RPE` par randori.
- Separation des usages : la page `Gestion activites` revient a son role d'edition, tandis qu'une nouvelle page `Visualisation activite` presente une lecture finale plus propre des seances annotees.
- Creation du dossier `assets/activity_visuals/` avec un fichier guide `README.md` pour deposer ensuite des images locales par type d'entrainement et par phase.

- Nettoyage de Gestion activites : suppression des derniers blocs visuels de synthese encore presents dans la page d'edition, maintenant entierement reportes vers Visualisation activite.

- Simplification de Visualisation activite : suppression du resume illustre des phases de seance, et ajout dans le tableau des randoris de la duree manuellement renseignee pour chaque repetition.

- Le tableau RPE des randoris de Visualisation activite affiche maintenant la duree reelle de chaque combat a partir des segments andori effectivement annotes sur le graphe FC, plutot que la duree declaree dans les champs de saisie.

- Stabilisation des formulaires de Gestion activites : rechargement des valeurs a partir d'une signature persistante de la seance et re-association des blocs randori via phase_uid pour conserver les parametres enregistres lors des retours sur une activite annotee.
- Evolution de Visualisation activite : ajout d'un calendrier de selection comme dans Gestion activites, et remplacement du filtre seances completes par un filtre nnotees seulement afin d'afficher aussi les seances deja annotees meme si les RPE ne sont pas encore renseignes.
