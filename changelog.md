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
