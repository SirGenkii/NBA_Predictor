# 🔍 Analyse Détaillée du Crash lors de la Génération Gold

## 📊 Problème Identifié

Le crash se produit lors de l'exécution de `make run-training-dataset` qui appelle `build_training_dataset()`.

## 🔬 Analyse du Flux d'Exécution

### 1. Point d'Entrée
```bash
make run-training-dataset
→ python -m nba_predictor.datasets.training build
→ build_training_dataset()
```

### 2. Flux dans `build_training_dataset()`

Pour chaque saison :
1. Récupère les dates (légère)
2. Pour chaque chunk de dates (5 dates) :
   - Appelle `prepare_match_feature_frame(seasons=[season], date_start, date_end)`
   - Écrit le chunk en parquet

### 3. Problème Critique : `prepare_match_feature_frame()`

Cette fonction fait plusieurs opérations qui peuvent exploser la mémoire :

#### a) `_suffix_features()` - LIGNE 76
```python
def _suffix_features(df: pl.LazyFrame, suffix: str, protected: Iterable[str]) -> pl.LazyFrame:
    protected_set = set(protected)
    column_names = df.collect_schema().names()  # ⚠️ PROBLÈME ICI
    rename_map = {col: f"{col}_{suffix}" for col in column_names if col not in protected_set}
    if rename_map:
        df = df.rename(rename_map)
    return df
```

**Problème** : `collect_schema()` peut matérialiser le schéma, mais surtout, si le LazyFrame a des milliers de colonnes, cela peut être coûteux.

#### b) `_compute_differences()` - LIGNE 85
```python
def _compute_differences(df: pl.LazyFrame) -> pl.LazyFrame:
    diff_exprs = []
    columns = df.columns  # ⚠️ PROBLÈME ICI
    for col in columns:
        if col.endswith("_home"):
            base = col[:-5]
            away_col = f"{base}_away"
            if away_col in columns:
                diff_exprs.append((pl.col(col) - pl.col(away_col)).alias(f"{base}{DIFF_SUFFIX}"))
    if diff_exprs:
        df = df.with_columns(diff_exprs)
    return df
```

**Problème** : `df.columns` sur un LazyFrame peut être coûteux avec des milliers de colonnes.

#### c) Jointure finale `home.join(away)` - LIGNE 200
```python
dataset = (
    home.join(
        away,
        on=["season", "game_date", "game_id", "home_team_id", "away_team_id"],
        how="inner",
    )
    .pipe(_compute_differences)
    ...
)
```

**PROBLÈME MAJEUR** : Même si c'est lazy, quand on a :
- `home` avec ~1000 colonnes (toutes les features avec suffix `_home`)
- `away` avec ~1000 colonnes (toutes les features avec suffix `_away`)
- La jointure crée un DataFrame avec ~2000 colonnes
- Même en lazy mode, Polars doit matérialiser des parties pour le join
- Avec des milliers de lignes × 2000 colonnes, ça peut facilement dépasser 32Go

## 🎯 Calcul de la Mémoire

Avec 7 fenêtres × ~50 colonnes de features :
- Features de base : ~50 colonnes
- Rolling windows : 50 × 2 (mean + std) × 7 = 700 colonnes
- Plus win_rate et wins : 2 × 7 = 14 colonnes
- **Total par dataset** : ~764 colonnes

Après `_suffix_features()` :
- `home` : ~764 colonnes
- `away` : ~764 colonnes
- Après join : ~1528 colonnes
- Après `_compute_differences()` : ~1528 + ~764 = **~2292 colonnes**

Avec 11Go de données silver et potentiellement des millions de lignes :
- 1 million de lignes × 2292 colonnes × 8 bytes (float64) = **~18.3 Go**
- Mais avec les overheads Python/Polars, ça peut facilement dépasser 32Go

## 🔍 Comparaison avec l'Ancien Système (backup/v2)

L'ancien système utilisait Pandas et :
1. Chargeait tout en mémoire d'un coup
2. Mais avait probablement moins de features (pas de rolling windows aussi nombreux)
3. Utilisait des CSV qui sont plus compressés que Parquet pour certaines opérations
4. Traitait par saison complète, pas par chunks

## 💡 Solutions

### Solution 1 : Éviter `collect_schema()` et `df.columns`

Utiliser le schéma lazy sans matérialisation :

```python
def _suffix_features(df: pl.LazyFrame, suffix: str, protected: Iterable[str]) -> pl.LazyFrame:
    protected_set = set(protected)
    # Utiliser schema() au lieu de collect_schema()
    schema = df.schema
    rename_map = {col: f"{col}_{suffix}" for col in schema if col not in protected_set}
    if rename_map:
        df = df.rename(rename_map)
    return df

def _compute_differences(df: pl.LazyFrame) -> pl.LazyFrame:
    # Construire les expressions sans accéder à df.columns
    # Utiliser un schéma pré-calculé ou une approche différente
    ...
```

### Solution 2 : Écrire directement sans matérialiser le join complet

Au lieu de faire `home.join(away)` puis `sink_parquet()`, écrire directement en streaming :

```python
# Écrire home et away séparément, puis les joindre lors de la lecture
# Ou utiliser un merge plus efficace
```

### Solution 3 : Réduire le nombre de colonnes avant le join

Sélectionner uniquement les colonnes nécessaires avant le join :

```python
# Sélectionner seulement les colonnes essentielles avant le join
essential_cols = ["season", "game_date", "game_id", "home_team_id", "away_team_id"]
home_essential = home.select(essential_cols + [col for col in home.columns if col.endswith("_home")])
away_essential = away.select(essential_cols + [col for col in away.columns if col.endswith("_away")])
```

### Solution 4 : Utiliser `sink_parquet()` directement sur le LazyFrame

Éviter toute matérialisation intermédiaire :

```python
# Au lieu de collecter puis écrire, écrire directement
chunk_frame.sink_parquet(...)  # Déjà fait, mais peut être optimisé
```

## 🚨 Le Vrai Problème

Le problème n'est PAS le chunking, mais la **matérialisation implicite** lors de :
1. `collect_schema()` dans `_suffix_features()`
2. `df.columns` dans `_compute_differences()`
3. La jointure `home.join(away)` qui peut matérialiser des parties même en lazy mode

Même avec streaming, Polars doit parfois matérialiser des parties pour certaines opérations, et avec 2000+ colonnes, ça explose.

