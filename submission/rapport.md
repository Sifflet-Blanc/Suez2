# SUEZ 2 - Rapport synthese

## Objectif
Reconstituer des chroniques journalieres de niveaux de nappe et produire des intervalles de confiance (95%). Un objectif secondaire est l'estimation des niveaux moyens en periode d'etiage.

## Donnees exploitees
- 78 series temporelles de niveaux piezometriques (variable cible).
- Attributs statiques issus de `piezo_characs.csv`.

Note: les forcages hydrometeorologiques ne sont pas presentes dans le dossier fourni. Le baseline ci-dessous s'appuie donc sur la dynamique propre des niveaux (lags) et les attributs statiques.

## Methode
1. Preparation des series: conversion en frequence journaliere, agregation des doublons par date.
2. Features dynamiques (mode `ar`): lags (1, 7, 30, 90), statistiques glissantes (moyenne, ecart-type) et variables saisonnieres (sin/cos du jour de l'annee).
3. Features statiques (modes `static` / `static_min`): colonnes numeriques et categorielle(s) encodees ordinalement, avec retrait des colonnes a cardinalite trop elevee et des signatures derivees.
4. Modele: `HistGradientBoostingRegressor` avec imputation simple.
5. Intervalles 95%: quantiles des residus (option conformal via un jeu de calibration).
6. Evaluation: Nash-Sutcliffe et score de Winkler (validation group k-fold par BSS_ID).
7. Evaluation temporelle (mode `ar`): entrainement sur 80% du debut de chaque serie et test sur 20% de fin.

## Livrables
- `submission/code.py` : fonctions `train` et `reconstruct` via CLI.
- `submission/src/pipeline.py` : chargement des donnees, features, metriques.
- `submission/rapport.md` : ce rapport.

## Utilisation
Entraine le modele:
```
python3 code.py train --data-dir ../datasets --model-dir ./model
```
Reconstruit les series (predictions + intervalles) et calcule l'etiage:
```
python3 code.py reconstruct --data-dir ../datasets --model-dir ./model --out-dir ./outputs
```
Evaluation group k-fold (generalisation par station):
```
python3 code.py evaluate --data-dir ../datasets --out-dir ./outputs --mode static_min --strategy groupkfold --splits 5
```
Evaluation leave-one-out (generalisation par station, plus long):
```
python3 code.py evaluate --data-dir ../datasets --out-dir ./outputs --mode static_min --strategy loo
```
Evaluation temporelle (reconstitution avec historique disponible):
```
python3 code.py evaluate --data-dir ../datasets --out-dir ./outputs --mode ar --strategy timesplit --test-ratio 0.2 --min-obs 365
```
Option intervalles conformal (calibration):
```
python3 code.py evaluate --data-dir ../datasets --out-dir ./outputs --mode ar --strategy timesplit --test-ratio 0.2 --min-obs 365 --config ../config_conformal.json
```

Sorties principales:
- `predictions.csv` (niveau predit + bornes 95%).
- `etiage_means.csv` (moyenne Jul-Sep par annee).
- `groupkfold_static_min.csv` et `groupkfold_static_min_summary.json` (evaluation).
- `loo_static_min.csv` et `loo_static_min_summary.json` (evaluation).

## Resultats (group k-fold, mode static_min)
- NSE moyen: 0.989
- NSE median: 0.994
- Couverture moyenne 95%: 0.189
- Winkler moyen: 197.22

Note: ces scores peuvent rester optimistes si certains attributs statiques encodent indirectement les niveaux observes. Le mode `static_min` retire les signatures les plus evidentes, mais une verification metier est necessaire.

## Resultats (leave-one-out, mode static_min)
- NSE moyen: -405.90
- NSE median: -42.92
- Couverture moyenne 95%: 0.208
- Winkler moyen: 174.24

Interpretation: la generalisation inter-stations est faible sans forcages hydrometeorologiques ni historique propre a la station.

## Resultats (time split, mode ar)
- NSE global: 1.000
- NSE moyen par station: 0.897
- NSE median par station: 0.914
- Couverture 95% globale: 0.942
- Winkler global: 1.26

Interpretation: ce mode est adapte a la reconstitution ou la prevision a court terme lorsqu'un historique recent est disponible. Il ne permet pas d'extrapoler vers une station totalement inconnue sans forcages hydrometeorologiques.

## Intervalles (option conformal)
Si `interval_method=conformal` et `calibration_ratio=0.2`, les bornes 95% sont calculees sur un sous-ensemble de calibration, ce qui donne des intervalles plus fiables hors entrainement.
Resultats conformal (time split, mode ar):
- NSE global: 1.000
- NSE moyen par station: 0.881
- NSE median par station: 0.900
- Couverture 95% globale: 0.946
- Winkler global: 1.21

## Graphes generes
Les figures suivantes ont ete produites pour une station representative (BSS000UFTP):
- `submission/outputs/plot_timeseries.png` : serie temporelle (observations, predictions, IC 95%).
- `submission/outputs/plot_scatter.png` : dispersion y_obs vs y_pred.
- `submission/outputs/plot_residuals.png` : histogramme des residus.

## Limites et pistes
- Ajouter les forcages hydrometeorologiques si disponibles (pluie, debit, ETP, temperature).
- Ameliorer la generalisation inter-stations (features hydro, topographiques, proximite au reseau hydrographique).
- Integrer un modele probabiliste (quantile regression, conformal) pour des intervalles conditionnels.
