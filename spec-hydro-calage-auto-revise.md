# SPEC — `hydro-calibration-harness` : prototype de harnais de calage automatisé pluie-débit

**But** : démontrer en ~4 minutes d'écran un harnais d'automatisation du calage — **modèle hydrologique conceptuel documenté et remplaçable** + exploration paramétrique sous contraintes + validation split-sample + ensemble d'incertitude paramétrique + rapport auto-généré et reproductible.

**Thèse de la démonstration** : le modèle hydrologique est volontairement simple ; le prototype porte sur **l'ingénierie du calage, la reproductibilité, la validation et la gestion explicite de l'incertitude**. Le succès de la démo n'est pas d'obtenir le meilleur KGE possible, mais de rendre visibles la performance, la robustesse et les limites du modèle.

**Périmètre** : script Python (notebook Jupyter facultatif pour exploration) + 3 PNG + `runs.csv` + rapport MD. **AUCUNE UI.** Python : numpy, pandas, matplotlib, requests ; scipy uniquement si nécessaire pour le Latin Hypercube. Budget : **2 soirées max**.

---

## 1. Données — réelles, scriptables, zéro clé API

- **Débit observé** : API Hub'Eau hydrométrie, endpoint `obs_elab`, avec `code_entite=<station>` et `grandeur_hydro_elab=QmnJ` (débit moyen journalier en L/s). La station est un **paramètre de configuration**. Avant de figer le bassin pilote, vérifier que l'endpoint expose effectivement une chronique continue suffisamment longue.
- Choisir de préférence un bassin moyen (~200–2000 km²), à régime majoritairement pluvial, avec **au moins 4 ans réellement exploitables** après contrôle qualité.
- **Surface du bassin** : valeur explicite dans la configuration, avec sa source documentée.
- Conversion du débit en lame d'eau :

  \[
  Q_{mm/j} = \frac{Q_{L/s} \times 0.0864}{A_{km^2}}
  \]

  La formule et les unités sont rappelées dans le rapport pour assurer la traçabilité.

- **Pluie + ETP** : API historique Open-Meteo au centroïde du bassin (`lat`, `lon` en config), variables journalières `precipitation_sum` et `et0_fao_evapotranspiration`.
- **Limitation explicite** : la pluie ponctuelle au centroïde n'est **pas** une pluie moyenne de bassin. L'ET₀ FAO-56 fournie par Open-Meteo n'est pas nécessairement l'ETP utilisée dans une chaîne GR4J opérationnelle. Ces choix sont acceptés uniquement pour le prototype.
- Alignement pluie / ETP / débit sur un index journalier commun.
- Les trous restent `NaN` ; aucune interpolation silencieuse. Les dates non exploitables sont exclues des métriques et comptabilisées.
- Générer un petit résumé QA : période brute, période exploitable, nombre et proportion de lacunes par variable.
- Cache disque des réponses API pour garantir des reruns rapides et limiter la dépendance réseau pendant la démo.

### Limite temporelle assumée

Le prototype fonctionne au **pas journalier**. Il démontre l'ingénierie du calage et **pas** une chaîne opérationnelle de prévision de crue sub-journalière à +24/+48 h. Une application temps réel devra travailler à un pas cohérent avec le temps de réponse du bassin et les données disponibles.

---

## 2. Moteur hydrologique : GR4J journalier (4 paramètres)

- Implémenter **GR4J tel que publié** (Perrin, Michel & Andréassian, 2003), avec la documentation INRAE/airGR comme référence de contrôle : réservoir de production (`X1`), échange souterrain (`X2`), réservoir de routage (`X3`), hydrogrammes unitaires UH1/UH2 pilotés par `X4`, split 90/10.
- Présenter explicitement GR4J comme un **modèle hydrologique conceptuel global**, et non comme un modèle physique distribué.
- Tests minimum :
  - comportement sous pluie nulle / décroissance cohérente des stockages ;
  - réponse à une impulsion ;
  - non-négativité du débit simulé ;
  - comparaison à une sortie de référence airGR si une référence reproductible est disponible dans le temps imparti.
- **Bornes de démonstration configurables, jamais en dur** :
  - `X1 ∈ [100, 1200] mm`
  - `X2 ∈ [-5, 3] mm`
  - `X3 ∈ [20, 300] mm`
  - `X4 ∈ [1.1, 2.9] j`
- Ces plages sont présentées comme **bornes de démonstration documentées, inspirées de plages utilisées dans des implémentations de GR4J**, et non comme des bornes physiques universelles ou comme des « bornes airGR » normatives.
- Dans un projet réel, l'espace paramétrique admissible doit être défini avec l'hydrologue selon le bassin, la structure du modèle et les données disponibles.
- **Warm-up : 1 an**, exclu des métriques.
- Ne pas réinitialiser naïvement les états à la frontière calibration/validation : la simulation reste continue afin que la validation hérite d'un état hydrologique cohérent.

---

## 3. Configuration reproductible

Tous les éléments définissant une expérience sont externalisés dans `config/basin.yaml` :

```yaml
station:
  code: "..."
  basin_area_km2: 0
  centroid_lat: 0.0
  centroid_lon: 0.0

periods:
  warmup: ["YYYY-MM-DD", "YYYY-MM-DD"]
  calibration: ["YYYY-MM-DD", "YYYY-MM-DD"]
  validation: ["YYYY-MM-DD", "YYYY-MM-DD"]

model:
  name: "GR4J"
  parameter_bounds:
    X1: [100, 1200]
    X2: [-5, 3]
    X3: [20, 300]
    X4: [1.1, 2.9]

calibration:
  sampler: "latin_hypercube"
  n_samples: 5000
  seed: 42
  behavioral_kge_threshold: 0.60
```

- Utiliser des **périodes calendaires explicites** plutôt qu'un split aléatoire ou un simple `60/40` automatique.
- Avant le run final, vérifier que calibration et validation couvrent des conditions hydrologiques suffisamment diverses (années sèches/humides, événements significatifs si disponibles).

---

## 4. Calage automatique

- Échantillonnage **Latin Hypercube**, `N = 2000–5000` jeux de paramètres dans les bornes configurées.
- Graine fixée pour reproductibilité.
- Le LHS est volontairement privilégié à un optimiseur sophistiqué pour cette V0 : il permet de **caractériser explicitement l'espace paramétrique** et produit naturellement une population de simulations utile pour l'analyse d'incertitude.
- Métriques calculées pour chaque jeu :
  - **NSE** ;
  - **KGE**, avec ses composantes `r`, `α`, `β` ;
  - **log-NSE** pour documenter le comportement sur les faibles débits ;
  - **biais volumique**.
- Toutes les métriques sont calculées séparément sur **calibration** et **validation**, mais la validation ne sert pas à optimiser les paramètres.

### Sélection et robustesse

- **Best calibration candidate** : jeu maximisant `KGE_calibration`.
- Conserver également les **Top N candidats de calibration** (ex. Top 20) et afficher ensuite leur performance en validation sans retuning.
- Objectif pédagogique : montrer explicitement que le meilleur score de calibration n'est pas nécessairement le modèle qui généralise le mieux.
- Ne pas sélectionner automatiquement le « meilleur modèle final » en maximisant la validation : cela transformerait la validation en seconde calibration cachée.

### Ensemble comportemental

- Ensemble comportemental = jeux satisfaisant un critère explicite, par exemple `KGE_calibration > 0.60`.
- Le présenter comme une approche **inspirée de GLUE / GLUE-inspired**, et non comme une implémentation complète de GLUE.
- Le seuil est configurable et documenté ; il n'est pas présenté comme universel.

### Hors scope V0

- Pas de `differential_evolution`, NSGA-II, Optuna ou autre raffinement tant que la chaîne de base n'est pas impeccable.
- Une V2 pourra benchmarker plusieurs stratégies d'optimisation en comparant qualité, stabilité, nombre d'appels au moteur et coût de calcul.

---

## 5. Journal des expériences — `runs.csv`

Chaque simulation constitue une expérience traçable.

`runs.csv` contient au minimum :

```text
run_id
X1
X2
X3
X4
nse_cal
kge_cal
kge_r_cal
kge_alpha_cal
kge_beta_cal
lognse_cal
bias_cal
nse_val
kge_val
kge_r_val
kge_alpha_val
kge_beta_val
lognse_val
bias_val
behavioral
```

Résumé visible dans le rapport :

```text
5 000 parameter sets evaluated
→ 412 behavioral sets
→ Top 20 calibration candidates inspected in validation
→ 1 best-calibration candidate reported
```

Le prototype doit permettre de retrouver **pourquoi** un jeu de paramètres a été retenu ou rejeté.

---

## 6. Les trois sorties visuelles

### 1. Hydrogramme observé vs simulé

- Meilleur jeu selon `KGE_calibration`.
- Pluie en barres inversées en haut (convention hydrologique).
- Débit observé et simulé sur la même figure.
- Périodes warm-up / calibration / validation clairement identifiées.
- Ne pas masquer une dégradation éventuelle en validation.

### 2. Tableau de métriques calibration | validation

Afficher proprement :

| Metric | Calibration | Validation |
|---|---:|---:|
| NSE | | |
| KGE | | |
| r | | |
| α | | |
| β | | |
| log-NSE | | |
| Volume bias | | |

Ajouter si possible un petit tableau des Top candidats de calibration avec leur `KGE_cal` et `KGE_val` pour matérialiser le risque d'overfitting / manque de robustesse.

### 3. Fourchette de l'ensemble comportemental

- Fenêtre d'environ 60 jours dans la **période de validation**.
- Afficher : observé + médiane + enveloppe `q05–q95` des simulations comportementales.
- Titre recommandé : **Parametric uncertainty — validation period**.
- Mention visible :

> **Parametric uncertainty only — does not include precipitation, observation, initial-state or structural uncertainty.**

- Ne pas interpréter automatiquement `q05–q95` comme un intervalle probabiliste calibré. Il s'agit ici de la dispersion de l'ensemble paramétrique retenu.

---

## 7. Rapport auto-généré — la signature

`rapport_calage.md` est régénéré à chaque run.

Il contient :

1. **Bassin et données**
   - station ;
   - surface ;
   - période ;
   - sources ;
   - conversion L/s → mm/j ;
   - couverture et lacunes.

2. **Configuration du modèle**
   - moteur et version du prototype ;
   - bornes ;
   - warm-up ;
   - périodes calibration / validation.

3. **Expérience de calage**
   - méthode d'échantillonnage ;
   - `N` ;
   - seed ;
   - seuil comportemental ;
   - nombre de jeux comportementaux.

4. **Résultats**
   - paramètres du meilleur candidat calibration ;
   - métriques calibration / validation ;
   - comparaison des principaux candidats ;
   - liens vers les trois figures.

5. **Reproductibilité**

```text
Git commit:       <hash>
Config SHA256:    <hash>
Random seed:      42
Generated at:     <UTC timestamp>
Python version:   <version>
Model version:    gr4j-prototype-v0.1
```

6. **Limites assumées**
   - pluie ponctuelle au centroïde, non moyennée sur le bassin ;
   - résolution journalière ;
   - QA hydrométrique simplifié ;
   - courbes de tarage non auditées ;
   - pas d'assimilation / mise à jour d'état ;
   - incertitude paramétrique uniquement ;
   - pas d'incertitude météorologique, observationnelle ou structurelle.

7. **Statut**

```text
STATUS: PROTOTYPE / DEMONSTRATION ONLY
NOT FOR OPERATIONAL HYDROLOGICAL FORECASTING
```

**Message implicite** : chaque calage produit automatiquement son dossier auditable — miniature du livrable « validation du comportement du modèle » de la mission.

---

## 8. README — positionnement écrit

### Paragraphe 1 — thèse

> This prototype tests a simple engineering thesis: calibration of a documented hydrological model can be made reproducible and auditable through explicit parameter bounds, systematic exploration, strict out-of-sample validation and transparent uncertainty reporting.

### Paragraphe 2 — ce que c'est / ce que ce n'est pas

> GR4J is intentionally used as a compact conceptual rainfall-runoff model. The purpose of the repository is not to propose an operational hydrological model for the target project, but to prototype the calibration harness around a replaceable deterministic model. Daily point precipitation and simplified data QA make the hydrology deliberately limited; those limitations are reported rather than hidden.

### Paragraphe 3 — transposition

> The same harness can be adapted to another scriptable hydrological engine: explicit configuration, constrained parameters, repeatable model execution, objective metrics, out-of-sample validation, experiment tracking and automatic reporting remain separate from the underlying model implementation.

- Ne pas mentionner WindSound dans le README ; garder cette analogie pour l'entretien oral.
- Licence MIT si le repo est rendu public.

---

## 9. Architecture du repository

```text
hydro-calibration-harness/
│
├── config/
│   └── basin.yaml
│
├── src/
│   ├── data.py
│   ├── gr4j.py
│   ├── metrics.py
│   ├── sampling.py
│   ├── calibration.py
│   ├── validation.py
│   ├── ensemble.py
│   └── report.py
│
├── tests/
│   └── test_gr4j.py
│
├── output/
│   ├── hydrograph.png
│   ├── metrics.png
│   ├── ensemble.png
│   ├── runs.csv
│   └── rapport_calage.md
│
├── run.py
├── requirements.txt
└── README.md
```

Commande cible :

```bash
python run.py --config config/basin.yaml
```

Cette commande doit reconstruire l'ensemble des résultats de manière reproductible à partir de la configuration.

---

## 10. Démo entretien — ~4 minutes

### 0:00–0:30 — Le problème

> « Je n'ai pas essayé de reproduire HEC-HMS. J'ai prototypé la couche qui me semble au cœur de votre besoin : l'automatisation du calage autour d'un moteur hydrologique remplaçable. »

### 0:30–1:00 — Configuration

Montrer `basin.yaml` : station, périodes, modèle, bornes, seed, seuil.

> « Tout ce qui définit une expérience est explicite et versionnable. »

### 1:00–2:00 — Calage

Montrer :

```text
5 000 parameter sets
→ GR4J
→ NSE / KGE / log-NSE / bias
→ behavioral ensemble
→ runs.csv
```

Puis l'hydrogramme.

### 2:00–3:00 — Validation + incertitude

- montrer calibration vs validation ;
- montrer qu'un excellent candidat calibration peut se dégrader hors échantillon ;
- montrer `q05–q95` ;
- préciser spontanément : **incertitude paramétrique uniquement**.

### 3:00–4:00 — Rapport auditable

Ouvrir `rapport_calage.md` et conclure :

> « Ce qui m'intéresse ici n'est pas le score de ce bassin de démonstration. C'est qu'à chaque calage, la chaîne produit automatiquement les paramètres testés, les données utilisées, les performances en calibration et validation, l'incertitude et les limites. Pour moi, c'est cette couche qu'il faut ensuite connecter à HEC-HMS et formaliser avec l'hydrologue. »

---

## 11. Découpage en 2 soirées

### Soirée 1 — moteur + données

- loaders Hub'Eau + Open-Meteo avec cache disque ;
- sélection et validation rapide de la station ;
- configuration YAML ;
- GR4J ;
- tests minimum ;
- run manuel avec un jeu de paramètres ;
- hydrogramme brut.

### Soirée 2 — harnais + démonstration

- Latin Hypercube ;
- métriques ;
- calibration / validation temporelles ;
- `runs.csv` ;
- ensemble comportemental GLUE-inspired ;
- trois figures ;
- rapport MD ;
- README ;
- repo public si tout est propre.

### Coupe si retard

Ordre de priorité :

1. chaîne reproductible `config → run → résultats` ;
2. calibration + validation ;
3. hydrogramme ;
4. `runs.csv` ;
5. ensemble q05–q95 ;
6. rapport ;
7. finitions README.

Si nécessaire : `N = 1000`. **Aucun optimiseur supplémentaire.**

Les figures et le rapport doivent rester : **ils sont la démonstration.**
