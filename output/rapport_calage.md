**STATUT : PROTOTYPE / NON DESTINÉ À LA DÉCISION HYDROLOGIQUE OPÉRATIONNELLE**

# Rapport de calage pluie–débit automatisé

## 1. Périmètre du prototype

Ce document décrit un **prototype d'ingénierie du calage** construit autour du modèle conceptuel pluie–débit **GR4J**. L'objectif est de démontrer l'exploration automatique de l'espace des paramètres, la séparation explicite calage / validation, un ensemble comportemental inspiré de l'approche GLUE, la communication transparente de l'incertitude et la reproductibilité complète.

Il ne s'agit **pas** d'un système opérationnel de prévision de crue ni d'un outil réglementaire d'aide à la décision.

## 2. Bassin versant et données

- **Code station :** H0203020
- **Nom de la station :** La Laignes à Molesme
- **Surface du bassin versant :** 615.0 km²
- **Centroïde :** (47.96178, 4.361)
- **Période d'analyse :** 2010-01-01 → 2015-12-31
- **Période de mise en route (warm-up) :** 2010-01-01 → 2010-12-31
- **Période de calage :** 2011-01-01 → 2013-12-31
- **Période de validation :** 2014-01-01 → 2015-12-31
- **Source de débit :** API Hub'Eau hydrométrie v2 (`obs_elab`, `QmnJ`, L/s)
- **Source précipitations / ET0 :** Open-Meteo Historical Weather API (journalière)
- **Résolution temporelle :** journalière
- **Fuseau horaire (agrégation météo) :** Europe/Paris
- **Précipitations manquantes :** 0 jours
- **ET0 manquante :** 0 jours
- **Débit manquant :** 0 jours
- **Observations exploitables (toutes variables présentes) :** 2191 jours

Les précipitations sont représentées par un point Open-Meteo unique au centroïde du bassin versant ; il ne s'agit pas d'une précipitation moyenne de bassin.

Conversion du débit : Q_mm/j = Q_L/s × 0.0864 / basin_area_km².

## 3. Modèle hydrologique

**GR4J** (Perrin, Michel & Andréassian, 2003) — modèle conceptuel à quatre paramètres.

GR4J est un modèle conceptuel. Les paramètres calés ne doivent pas être interprétés automatiquement comme des mesures physiques directes des propriétés du bassin versant.

| Paramètre | Signification | Unité | Bornes de démonstration |
| --- | --- | --- | --- |
| X1 | Capacité du réservoir de production | mm | [100, 1200] |
| X2 | Flux d'échange souterrain | mm | [-5, 3] |
| X3 | Capacité du réservoir de routage | mm | [20, 300] |
| X4 | Base temporelle des hydrogrammes unitaires (UH1/UH2) | jours | [1.1, 2.9] |

**Convention d'état initial :** réservoir de production à 30 % de X1, réservoir de routage à 50 % de X3, stocks d'hydrogrammes unitaires vides (fractions par défaut airGR).
**Durée de la période de mise en route (warm-up) :** 365 jours (2010-01-01 → 2010-12-31), exclue de toutes les métriques reportées.
**Continuité des états :** GR4J s'exécute en continu du début de la période de mise en route (warm-up) jusqu'à la fin de la validation, sans réinitialisation des états aux frontières de période.

## 4. Expérience de calage

- **Méthode d'échantillonnage :** Latin Hypercube Sampling (latin_hypercube)
- **N :** 5000
- **Graine aléatoire :** 42
- **Bornes des paramètres :** voir section 3
- **Objectif de classement :** KGE_cal (période de calage uniquement)
- **Règle d'isolement de la validation :** les métriques de validation sont purement diagnostiques
- **Temps de calcul total :** 898.59 s
- **Temps moyen par évaluation :** 0.1797 s

Les données de validation ne sont utilisées ni pour l'échantillonnage, ni pour le classement, ni pour la sélection des paramètres.

## 5. Référence non calée

Paramètres de démonstration fixes (non ajustés manuellement aux observations) : X1=350.0, X2=0.0, X3=90.0, X4=1.4.

| Métrique | Calage | Validation |
| --- | ---: | ---: |
| NSE | 0.4402 | 0.0600 |
| KGE | 0.4731 | 0.2026 |
| r | 0.8714 | 0.8936 |
| alpha | 1.2594 | 1.5013 |
| beta | 1.4403 | 1.6109 |
| log-NSE | 0.6633 | 0.5791 |
| Biais volumique | 0.4403 | 0.6109 |

## 6. Meilleur jeu de paramètres issu du calage

- **Identifiant d'exécution (run_id) :** 3058
- **X1–X4 :** 248.360, -3.368, 86.318, 2.699

**Métriques de calage :** NSE=0.8083, KGE=0.9027, r=0.9071, alpha=1.0269, beta=0.9897, log-NSE=0.7695, biais=-0.0103.

**Métriques de validation (diagnostiques uniquement) :** NSE=0.7923, KGE=0.8350, r=0.9171, alpha=1.1295, beta=1.0597, log-NSE=0.8705, biais=0.0597.

**Proximité des bornes (à moins de 2 % de l'amplitude configurée) :** aucun (tous les paramètres à plus de 2 % des bornes configurées).

## 7. Calage vs validation

| Métrique | Non calé (calage) | Non calé (validation) | Meilleur calage (calage) | Meilleur calage (validation) |
| --- | ---: | ---: | ---: | ---: |
| NSE | 0.4402 | 0.0600 | 0.8083 | 0.7923 |
| KGE | 0.4731 | 0.2026 | 0.9027 | 0.8350 |
| r | 0.8714 | 0.8936 | 0.9071 | 0.9171 |
| alpha | 1.2594 | 1.5013 | 1.0269 | 1.1295 |
| beta | 1.4403 | 1.6109 | 0.9897 | 1.0597 |
| log-NSE | 0.6633 | 0.5791 | 0.7695 | 0.8705 |
| Biais volumique | 0.4403 | 0.6109 | -0.0103 | 0.0597 |

Le meilleur jeu de paramètres issu du calage conserve l'essentiel de sa performance de calage en validation (KGE_cal = 0.9027, KGE_val = 0.8350). Cela suggère une généralisation split-sample raisonnable pour ce bassin pilote, sans revendiquer une robustesse opérationnelle.

## 8. Diagnostics de l'espace des paramètres

**Distribution de KGE_cal (N = 5000) :** min=-3.9547, médiane=0.4862, p90=0.7080, p95=0.7609, p99=0.8450, max=0.9027.

**Effectifs par seuil :**
- kge_cal_gt_0.5: 2342
- kge_cal_gt_0.6: 1271
- kge_cal_gt_0.7: 540
- kge_cal_gt_0.75: 280
- kge_cal_gt_0.8: 126
- kge_cal_gt_0.85: 45

**corr(KGE_cal, KGE_val) :** 0.9529

**Plages des paramètres comportementaux (KGE_cal > seuil officiel) :**

| Paramètre | min | médiane | max |
| --- | ---: | ---: | ---: |
| X1 | 180.700 | 291.893 | 688.265 |
| X2 | -4.997 | -3.104 | -0.571 |
| X3 | 31.514 | 95.396 | 209.448 |
| X4 | 1.117 | 1.996 | 2.897 |

**Équifinalité.** Plusieurs jeux de paramètres distincts atteignent des performances de calage similaires ; le jeu de plus haut score ne doit donc pas être interprété comme une vérité physique unique.

**Paramètres faiblement contraints dans l'ensemble comportemental :** X2, X3, X4.

## 9. Ensemble comportemental

- **Méthode :** ensemble comportemental inspiré de l'approche GLUE (et non une implémentation complète de GLUE)
- **Critère :** KGE_cal > 0.8
- **Taille de l'ensemble :** 126 membres
- **Le critère est configurable** et **n'est pas** un seuil d'acceptabilité hydrologique universel
- **La validation n'intervient pas dans l'appartenance à l'ensemble**

**Métriques de validation de q50 (diagnostiques) :** NSE=0.7893, KGE=0.8342, r=0.9120, alpha=1.0860, beta=1.1111, log-NSE=0.8754, biais=0.1111.

## 10. Diagnostics d'incertitude

- **Couverture empirique de validation de l'enveloppe comportementale (q05–q95) :** 59.9%
- **Largeur moyenne de l'enveloppe :** 0.2063 mm/j
- **Largeur médiane de l'enveloppe :** 0.1586 mm/j
- **Largeur p90 de l'enveloppe :** 0.4023 mm/j
- **Largeur moyenne / débit observé moyen en validation :** 0.5086

L'enveloppe q05–q95 est une enveloppe représentant uniquement l'incertitude paramétrique (dispersion des simulations comportementales retenues). Il ne s'agit ni d'un intervalle de confiance à 90 %, ni d'un intervalle de prédiction à 90 %, ni d'une probabilité de 90 %.

La sous-couverture empirique observée sur la période de validation est attendue, car les incertitudes sur les précipitations, sur les observations et sur la structure du modèle ne sont pas propagées dans ce prototype : la dispersion paramétrique seule ne suffit pas à représenter l'incertitude prédictive totale.

L'enveloppe exclut explicitement :
- l'incertitude sur les précipitations ;
- l'incertitude observationnelle ;
- l'incertitude de structure du modèle ;
- l'incertitude d'état initial.

**Tableau de sensibilité au seuil (diagnostique ; le seuil n'est pas choisi à partir de la validation) :**

| KGE_cal > | Membres | Couverture empirique de validation |
| --- | ---: | ---: |
| 0.70 | 540 | 71.8% |
| 0.75 | 280 | 67.1% |
| 0.80 | 126 | 59.9% |
| 0.85 | 45 | 47.1% |

**Comparaison q50 vs meilleur calage en validation (diagnostique) :**

| Métrique | q50 | Meilleur calage |
| --- | ---: | ---: |
| nse | 0.7893 | 0.7923 |
| kge | 0.8342 | 0.8350 |
| r | 0.9120 | 0.9171 |
| alpha | 1.0860 | 1.1295 |
| beta | 1.1111 | 1.0597 |
| lognse | 0.8754 | 0.8705 |
| bias | 0.1111 | 0.0597 |

## 11. Caractérisation hydrologique des périodes

### Synthèse annuelle (2010–2015)

| Année | Précipitations (mm) | Lame d'eau Q (mm) | Coefficient d'écoulement | Q moyen (mm/j) | Q max (mm/j) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2010 | 903.5 | 142.0 | 0.1572 | 0.3891 | 3.4588 |
| 2011 | 712.3 | 113.7 | 0.1596 | 0.3115 | 3.1396 |
| 2012 | 920.4 | 188.4 | 0.2047 | 0.5147 | 3.3464 |
| 2013 | 980.3 | 310.8 | 0.3171 | 0.8516 | 3.5026 |
| 2014 | 840.6 | 147.7 | 0.1757 | 0.4046 | 1.3578 |
| 2015 | 724.6 | 148.4 | 0.2048 | 0.4065 | 2.7543 |

### Agrégats calage vs validation

| Agrégat de période | Précipitations (mm) | Lame d'eau Q (mm) | Coefficient d'écoulement | Q moyen (mm/j) | Q max (mm/j) |
| --- | ---: | ---: | ---: | ---: | ---: |
| calage | 2613.0 | 612.9 | 0.2346 | 0.5592 | 3.5026 |
| validation | 1565.2 | 296.1 | 0.1892 | 0.4056 | 2.7543 |

Note : l'année 2014 présente un maximum journalier de débit observé plus bas que les années voisines dans ce jeu de données ; cette observation est rapportée à titre diagnostique et nécessite investigation, sans être qualifiée d'anomalie définitive.

## 12. Limites

- Résolution temporelle journalière uniquement
- Précipitations au centroïde plutôt qu'une précipitation moyenne de bassin
- Modèle conceptuel global plutôt qu'une représentation physique distribuée
- Pas d'ensemble de précipitations
- Pas de propagation d'incertitude de courbe de tarage
- Pas d'assimilation d'état
- Incertitude paramétrique seule dans l'enveloppe reportée
- Seuil comportemental spécifique au prototype et configurable
- Un seul bassin versant pilote (une configuration de station)

## 13. Reproductibilité

- **Généré le (UTC) :** 2026-08-25T16:05:55Z
- **SHA256 de la configuration :** `0b093d57f547f99b6ba191ad1903ccb368d62ed77bd202028c4b6e06f4ef7353`
- **Commit Git :** 66919b5dc13cbf504189fb46bcd9679752c35b55
- **Version Python :** 3.14.0
- **Version modèle / prototype :** gr4j-prototype-v0.1
- **Graine aléatoire :** 42
- **N simulations :** 5000

**Versions des paquets :**
- python: 3.14.0
- numpy: 2.3.5
- pandas: 2.3.3
- matplotlib: 3.11.0
- PyYAML: 6.0.3
- pytest: 8.4.2

## 14. Artéfacts

- `output\data\basin_daily.csv` — données journalières traitées
- `output\runs.csv` — 5000 expériences de calage
- `output\top20_calibration.csv` — meilleurs candidats de calage
- `output\behavioral_runs.csv` — jeux de paramètres comportementaux
- `output\ensemble_timeseries.csv` — séries temporelles des quantiles d'ensemble
- `output\ensemble_validation.png` — figure d'incertitude en validation
- `output\ensemble_full_validation.png` — diagnostic de validation complète
- `output\parameter_space_diagnostic.png` — diagnostic de l'espace des paramètres
- `output\hydrological_years.png` — figure de caractérisation hydrologique

## 15. Bannière de décision

**STATUT : PROTOTYPE / NON DESTINÉ À LA DÉCISION HYDROLOGIQUE OPÉRATIONNELLE**

Ce prototype ne doit pas être interprété comme une validation réglementaire, une prévision ou une validation hydrologique opérationnelle.
