# 🔍 Analyse du Projet NBA Predictor - Critique Constructive

## 📋 Résumé Exécutif

Cette analyse identifie les problèmes de mémoire rencontrés lors de la génération du dataset **gold** à partir des données **silver**, et propose des solutions concrètes pour optimiser le pipeline.

---

## 🚨 Problèmes Identifiés

### 1. **Problème Principal : Explosion de RAM lors de la génération Gold**

#### Localisation du problème
Le problème se situe principalement dans `src/nba_predictor/datasets/training.py`, fonction `prepare_match_feature_frame()`.

#### Causes identifiées

**a) Jointures sur LazyFrames complets sans filtrage préalable**
```python
# Ligne 101-121 dans training.py
team_facts = _scan_parquet(TEAM_FACTS_ROOT, seasons=seasons, label="team_game_facts")
team_form = _scan_parquet(TEAM_FORM_ROOT, seasons=seasons, label="team_form_windowed")
h2h_features = _scan_parquet(H2H_FEATURES_ROOT, seasons=seasons, label="matchups_h2h_features")

joined = (
    team_facts.join(team_form, on=["season", "game_id", "team_id"], how="left")
    .join(h2h_features, on=["season", "game_id", "team_id", "opponent_team_id"], how="left")
)
```

**Problème** : Même si `seasons` est filtré, les jointures créent des DataFrames très larges en mémoire car :
- Les LazyFrames sont matérialisés lors des jointures
- Les jointures `left` peuvent créer des duplications importantes
- Aucun filtrage par date avant les jointures

**b) Matérialisation complète avant filtrage**
```python
# Ligne 144-162
home = joined.filter(pl.col("is_home")).rename(...)
away = joined.filter(~pl.col("is_home")).rename(...)
```

**Problème** : Le DataFrame `joined` est matérialisé en mémoire avant d'être filtré en `home` et `away`.

**c) Jointure finale sur des DataFrames larges**
```python
# Ligne 164-178
dataset = home.join(away, on=["season", "game_date", "game_id", ...], how="inner")
```

**Problème** : Cette jointure peut créer un DataFrame avec un nombre de colonnes très élevé (toutes les features home + away + diff).

---

### 2. **Problèmes dans les Transformations Silver**

#### a) Collect() sans streaming=True
Plusieurs transformations matérialisent tout en mémoire :

```python
# matchups_h2h_base.py:80
materialized = joined.collect()  # ❌ Pas de streaming

# matchups_h2h_features.py:152
materialized = enriched.select(output_columns).collect()  # ❌ Pas de streaming

# team_form_windowed.py:160
materialized = enriched.select(output_columns).collect()  # ❌ Pas de streaming
```

#### b) Concaténation de DataFrames en mémoire
Dans plusieurs transformations, tous les fichiers parquet sont chargés avant concaténation :

```python
# team_game_facts.py:39-84
frames: list[pl.DataFrame] = []
for path in paths:
    df = pl.read_parquet(path)  # ❌ Charge tout en mémoire
    frames.append(df)
combined = pl.concat(frames, how="diagonal_relaxed")
```

**Problème** : Si vous avez des centaines de fichiers parquet, tous sont chargés simultanément en mémoire.

#### c) Rolling windows sur des datasets complets
Les rolling windows sont calculés sur des datasets entiers :

```python
# team_form_windowed.py:125-151
for window in window_sizes:  # (3, 5, 10, 25, 50, 100, 200)
    for column in feature_columns:  # Potentiellement des dizaines de colonnes
        rolling_exprs.append(
            pl.col(column).rolling_mean(window_size=window, min_periods=1)
            .over("team_id")
        )
```

**Problème** : Avec 7 fenêtres × N colonnes, cela crée un nombre exponentiel de features, et tout est calculé en mémoire.

---

### 3. **Problèmes d'Architecture**

#### a) Pas de traitement par chunks dans les jointures
Le code gold traite par saison et par chunks de dates, mais les jointures se font sur des LazyFrames complets avant le chunking.

#### b) Absence de gestion explicite de la mémoire
- Pas de limites de mémoire configurées
- Pas de monitoring de l'utilisation mémoire
- Pas de stratégie de garbage collection

#### c) Pas de partitionnement optimal
Les données silver sont partitionnées par saison, mais les jointures ne profitent pas toujours de ce partitionnement.

---

## 💡 Recommandations et Solutions

### Solution 1 : Refactoriser `prepare_match_feature_frame()` avec traitement par chunks

**Stratégie** : Faire les jointures par chunks de dates plutôt que sur des datasets complets.

```python
def prepare_match_feature_frame_chunked(
    *,
    seasons: Sequence[str] | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> pl.LazyFrame:
    """
    Version optimisée qui filtre par dates AVANT les jointures.
    """
    # Filtrer par dates dès le scan
    team_facts = _scan_parquet(
        TEAM_FACTS_ROOT, 
        seasons=seasons, 
        label="team_game_facts"
    )
    
    if date_start and date_end:
        team_facts = team_facts.filter(
            (pl.col("game_date") >= pl.lit(date_start)) &
            (pl.col("game_date") <= pl.lit(date_end))
        )
    
    # Sélectionner uniquement les colonnes nécessaires AVANT les jointures
    team_facts = team_facts.select([
        "season", "game_date", "game_id", "team_id", 
        "opponent_team_id", "is_home", "is_win",
        "team_points", "opponent_points", "team_plus_minus",
        "team_ast", "team_reb", "team_tov", "team_pf",
    ])
    
    # Charger les autres datasets avec filtrage par dates
    team_form = _scan_parquet(TEAM_FORM_ROOT, seasons=seasons)
    if date_start and date_end:
        team_form = team_form.filter(
            (pl.col("game_date") >= pl.lit(date_start)) &
            (pl.col("game_date") <= pl.lit(date_end))
        )
    
    h2h_features = _scan_parquet(H2H_FEATURES_ROOT, seasons=seasons)
    if date_start and date_end:
        h2h_features = h2h_features.filter(
            (pl.col("game_date") >= pl.lit(date_start)) &
            (pl.col("game_date") <= pl.lit(date_end))
        )
    
    # Jointures avec filtrage préalable
    joined = (
        team_facts.join(
            team_form,
            on=["season", "game_id", "team_id"],
            how="left",
            suffix="_team_form",
        )
        .join(
            h2h_features,
            on=["season", "game_id", "team_id", "opponent_team_id"],
            how="left",
            suffix="_h2h",
        )
    )
    
    # Reste du code identique...
    # ...
```

**Modifier `build_training_dataset()` pour utiliser cette version chunkée** :

```python
def build_training_dataset(...):
    # ...
    for season in seasons:
        # Récupérer les dates une seule fois
        dates_df = (
            _scan_parquet(TEAM_FACTS_ROOT, seasons=[season])
            .select(pl.col("game_date").unique())
            .collect(streaming=True)
            .sort("game_date")
        )
        date_list = dates_df["game_date"].to_list()
        
        for start_date, end_date in _chunk_date_ranges(date_list, chunk_size):
            # Préparer le frame par chunk
            chunk_frame = prepare_match_feature_frame_chunked(
                seasons=[season],
                date_start=start_date,
                date_end=end_date,
            )
            
            # Filtrer et traiter home/away par chunk
            home = chunk_frame.filter(pl.col("is_home")).pipe(...)
            away = chunk_frame.filter(~pl.col("is_home")).pipe(...)
            
            # Joindre home/away
            dataset_chunk = home.join(away, ...).pipe(_compute_differences)
            
            # Écrire directement le chunk
            # ...
```

---

### Solution 2 : Utiliser `streaming=True` partout

**Remplacer tous les `.collect()` par `.collect(streaming=True)`** :

```python
# ❌ AVANT
materialized = enriched.select(output_columns).collect()

# ✅ APRÈS
materialized = enriched.select(output_columns).collect(streaming=True)
```

**Fichiers à modifier** :
- `src/nba_predictor/transformations/matchups_h2h_base.py:80`
- `src/nba_predictor/transformations/matchups_h2h_features.py:152`
- `src/nba_predictor/transformations/team_form_windowed.py:160`

---

### Solution 3 : Éviter les concaténations de DataFrames en mémoire

**Remplacer** :
```python
# ❌ AVANT
frames: list[pl.DataFrame] = []
for path in paths:
    df = pl.read_parquet(path)
    frames.append(df)
combined = pl.concat(frames, how="diagonal_relaxed")
```

**Par** :
```python
# ✅ APRÈS
combined = pl.concat(
    [pl.scan_parquet(path) for path in paths],
    how="diagonal_relaxed"
)
```

**Fichiers à modifier** :
- `src/nba_predictor/transformations/team_game_facts.py:39-84`
- `src/nba_predictor/transformations/team_boxscores_agg.py:78-111`
- `src/nba_predictor/transformations/player_availability.py:42-73, 88-129`

---

### Solution 4 : Traitement par saison dans les transformations silver

**Modifier les transformations pour traiter par saison** :

```python
def build_team_form_windowed(...):
    # ...
    team_boxscores = _load_team_boxscores()
    team_facts = _load_team_game_facts()
    
    # Récupérer les saisons
    seasons = (
        team_facts.select(pl.col("season").unique())
        .collect(streaming=True)["season"]
        .to_list()
    )
    
    # Traiter par saison
    for season in seasons:
        season_boxscores = team_boxscores.filter(pl.col("season") == season)
        season_facts = team_facts.filter(pl.col("season") == season)
        
        # Jointure et calculs pour cette saison uniquement
        joined = season_boxscores.join(season_facts, ...)
        # ...
        
        # Écrire directement la saison
        write_partitioned(materialized, OUTPUT_ROOT, ingest_ts, partition_cols=["season"])
```

---

### Solution 5 : Optimiser les rolling windows

**Problème** : Calculer toutes les fenêtres sur toutes les colonnes crée un nombre exponentiel de features.

**Solutions** :
1. **Réduire le nombre de fenêtres** : Garder seulement les plus importantes (3, 5, 10, 25, 50)
2. **Calculer les rolling windows par chunks** : Traiter par période de dates plutôt que sur tout le dataset
3. **Utiliser des rolling windows incrémentales** : Calculer seulement les nouvelles valeurs

```python
# Exemple : Calculer les rolling windows par chunks de 100 matchs
def _compute_rolling_windows_chunked(
    df: pl.LazyFrame,
    feature_columns: list[str],
    window_sizes: Sequence[int],
    chunk_size: int = 100,
) -> pl.LazyFrame:
    # Grouper par team_id et créer des chunks
    # Calculer les rolling windows par chunk
    # ...
```

---

### Solution 6 : Ajouter un monitoring de mémoire

**Créer un utilitaire de monitoring** :

```python
# src/nba_predictor/utils/memory.py
import psutil
import structlog

logger = structlog.get_logger("nba_predictor.memory")

def log_memory_usage(label: str) -> None:
    process = psutil.Process()
    mem_info = process.memory_info()
    mem_mb = mem_info.rss / 1024 / 1024
    logger.info("memory_usage", label=label, memory_mb=mem_mb)
    
    if mem_mb > 8000:  # Alerte si > 8GB
        logger.warning("high_memory_usage", label=label, memory_mb=mem_mb)
```

**Utiliser dans les fonctions critiques** :

```python
def prepare_match_feature_frame(...):
    log_memory_usage("before_scan")
    team_facts = _scan_parquet(...)
    log_memory_usage("after_scan")
    
    joined = team_facts.join(...)
    log_memory_usage("after_join")
    # ...
```

---

### Solution 7 : Configurer Polars pour optimiser la mémoire

**Ajouter des configurations Polars** :

```python
# Dans training.py ou config.py
import polars as pl

# Configurer Polars pour optimiser la mémoire
pl.Config.set_streaming_chunk_size(50_000)  # Taille des chunks en streaming
pl.Config.set_fmt_str_lengths(100)  # Limiter la longueur des strings
```

---

## 📊 Plan de Refactoring Recommandé

### Phase 1 : Corrections Immédiates (1-2 jours)
1. ✅ Remplacer tous les `.collect()` par `.collect(streaming=True)`
2. ✅ Remplacer les concaténations de DataFrames par des `pl.scan_parquet()` + `pl.concat()`
3. ✅ Ajouter le monitoring de mémoire

### Phase 2 : Optimisations Critiques (3-5 jours)
1. ✅ Refactoriser `prepare_match_feature_frame()` avec filtrage par dates
2. ✅ Modifier `build_training_dataset()` pour faire les jointures par chunks
3. ✅ Traiter les transformations silver par saison

### Phase 3 : Optimisations Avancées (1 semaine)
1. ✅ Optimiser les rolling windows (réduire le nombre ou calculer par chunks)
2. ✅ Ajouter des tests de performance et de mémoire
3. ✅ Documenter les optimisations

---

## 🎯 Points Positifs du Code Actuel

1. **Architecture en couches** : Bronze → Silver → Gold est bien structurée
2. **Utilisation de Polars** : Bon choix pour la performance
3. **Partitionnement par saison** : Bonne pratique
4. **Chunking dans `build_training_dataset()`** : Déjà présent, mais à améliorer
5. **Utilisation de LazyFrames** : Bonne approche, mais pas toujours optimale

---

## ⚠️ Points d'Attention

1. **Complexité des features** : Avec 7 fenêtres × N colonnes, le nombre de features explose
2. **Pas de tests de performance** : Difficile de mesurer l'impact des optimisations
3. **Documentation limitée** : Les choix d'architecture ne sont pas documentés
4. **Pas de gestion d'erreurs** : Si une transformation échoue, tout le pipeline s'arrête

---

## 🔧 Actions Immédiates Recommandées

### Priorité 1 (À faire maintenant)
1. Modifier `prepare_match_feature_frame()` pour filtrer par dates avant les jointures
2. Remplacer tous les `.collect()` par `.collect(streaming=True)`
3. Ajouter le monitoring de mémoire

### Priorité 2 (Cette semaine)
1. Refactoriser les transformations silver pour traiter par saison
2. Remplacer les concaténations de DataFrames en mémoire

### Priorité 3 (Ce mois)
1. Optimiser les rolling windows
2. Ajouter des tests de performance
3. Documenter les optimisations

---

## 📝 Conclusion

Le problème principal vient de la matérialisation de DataFrames trop larges en mémoire lors des jointures dans `prepare_match_feature_frame()`. Les solutions proposées devraient réduire significativement l'utilisation mémoire en :

1. Filtrage préalable par dates
2. Traitement par chunks plus agressif
3. Utilisation systématique de `streaming=True`
4. Éviter les concaténations en mémoire

Ces modifications devraient permettre de générer le dataset gold sans explosion de RAM, tout en maintenant les performances du pipeline.

