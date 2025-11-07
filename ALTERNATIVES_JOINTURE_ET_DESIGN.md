# 🔄 Alternatives pour la Jointure Home/Away + Analyse du Design

## 🎯 Alternatives pour la Jointure Home/Away

### Problème Actuel
La jointure `home.join(away)` avec ~2000 colonnes peut matérialiser des parties importantes en mémoire, même en lazy mode.

### Alternative 1 : **Pivot/Unpivot Pattern** (Recommandé)

Au lieu de créer deux DataFrames séparés (home/away) puis les joindre, utiliser un pivot :

```python
def prepare_match_feature_frame_pivot(
    *,
    seasons: Sequence[str] | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> pl.LazyFrame:
    """
    Alternative approach: Keep data in long format, then pivot by is_home.
    This avoids the expensive home.join(away) operation.
    """
    # ... load and join as before ...
    joined = team_facts.join(team_form, ...).join(h2h_features, ...)
    
    # Instead of filtering home/away separately, use pivot
    # Create a column that indicates if it's home or away features
    dataset = (
        joined
        .with_columns([
            pl.when(pl.col("is_home"))
            .then(pl.struct([pl.col(c).alias(f"{c}_home") for c in feature_cols]))
            .otherwise(pl.struct([pl.col(c).alias(f"{c}_away") for c in feature_cols]))
            .alias("features_by_side")
        ])
        # Group by game and pivot
        .group_by(["season", "game_date", "game_id", "team_id", "opponent_team_id"])
        .agg([
            pl.col("features_by_side").first().struct.field("home").first(),
            pl.col("features_by_side").first().struct.field("away").first(),
            # ... other aggregations
        ])
    )
    
    return dataset
```

**Avantages** :
- Évite la jointure coûteuse
- Traite les données en une seule passe
- Moins de mémoire intermédiaire

**Inconvénients** :
- Plus complexe à implémenter
- Nécessite de restructurer la logique

---

### Alternative 2 : **Écriture Directe en Streaming avec Merge Externe**

Écrire home et away séparément, puis les joindre lors de la lecture :

```python
def prepare_match_feature_frame_streaming(
    *,
    seasons: Sequence[str] | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """
    Return home and away as separate LazyFrames.
    The join happens later during sink_parquet with merge strategy.
    """
    # ... load and join as before ...
    
    home = (
        joined.filter(pl.col("is_home"))
        .pipe(_suffix_features, suffix="home", ...)
    )
    
    away = (
        joined.filter(~pl.col("is_home"))
        .pipe(_suffix_features, suffix="away", ...)
    )
    
    # Return separately - join happens in build_training_dataset
    return home, away

def build_training_dataset_streaming(...):
    # ...
    for chunk:
        home_lazy, away_lazy = prepare_match_feature_frame_streaming(...)
        
        # Write home and away to separate temp files
        home_path = tmpdir / f"home_{chunk_counter}.parquet"
        away_path = tmpdir / f"away_{chunk_counter}.parquet"
        
        home_lazy.sink_parquet(home_path)
        away_lazy.sink_parquet(away_path)
        
        # Read and join the smaller files
        home_df = pl.scan_parquet(home_path)
        away_df = pl.scan_parquet(away_path)
        
        # Now join the smaller DataFrames
        merged = home_df.join(away_df, ...)
        merged.sink_parquet(chunk_output)
```

**Avantages** :
- Écriture en streaming sans matérialisation complète
- Join sur des fichiers plus petits
- Contrôle total sur la mémoire

**Inconvénients** :
- Plus d'opérations I/O
- Plus complexe

---

### Alternative 3 : **Utiliser Feast pour la Jointure**

Déplacer la logique de jointure dans Feast :

```python
# Dans silver, garder les données en format long (une ligne par team/match)
# Feast fait la jointure lors de la materialization

# Dans training.py, utiliser Feast pour récupérer les features déjà jointes
from feast import FeatureStore

fs = FeatureStore(repo_path=".")
features = fs.get_historical_features(
    entity_df=entity_df,  # game_id, home_team_id, away_team_id
    features=[
        "team_aggregated_stats:*",
        "team_vs_opp_h2h:*",
    ]
)
```

**Avantages** :
- Feast gère la jointure de manière optimisée
- Point-in-time correctness automatique
- Séparation des responsabilités

**Inconvénients** :
- Nécessite de refactorer l'architecture
- Dépendance à Feast pour le training

---

### Alternative 4 : **Sélectionner Seulement les Colonnes Essentielles Avant le Join**

Réduire drastiquement le nombre de colonnes avant la jointure :

```python
def prepare_match_feature_frame_selective(
    *,
    seasons: Sequence[str] | None = None,
    date_start: date | None = None,
    date_end: date | None = None,
    feature_whitelist: set[str] | None = None,
) -> pl.LazyFrame:
    """
    Only select essential columns before the join to reduce memory.
    """
    # ... load as before ...
    
    # Select only essential columns + whitelisted features
    essential_cols = {
        "season", "game_date", "game_id", "team_id", "opponent_team_id",
        "is_home", "is_win", "team_points", "opponent_points"
    }
    
    if feature_whitelist:
        selected_cols = essential_cols | feature_whitelist
        team_form = team_form.select([c for c in team_form.schema if c in selected_cols])
        h2h_features = h2h_features.select([c for c in h2h_features.schema if c in selected_cols])
    
    # ... rest as before ...
```

**Avantages** :
- Simple à implémenter
- Réduit immédiatement la mémoire
- Permet de tester avec un sous-ensemble de features

**Inconvénients** :
- Perd des features potentiellement utiles
- Nécessite de définir une whitelist

---

### Alternative 5 : **Utiliser un Format de Stockage Différent**

Stocker les features home/away dans des colonnes structurées :

```python
# Au lieu de :
# team_points_home, team_points_away, team_ast_home, team_ast_away, ...

# Utiliser :
# home_features: struct{team_points, team_ast, ...}
# away_features: struct{team_points, team_ast, ...}

dataset = (
    joined
    .with_columns([
        pl.struct([...home features...]).alias("home_features"),
        pl.struct([...away features...]).alias("away_features"),
    ])
    .select([
        "season", "game_date", "game_id",
        "home_features", "away_features",
        # Compute diffs by accessing struct fields
        (pl.col("home_features").struct.field("team_points") - 
         pl.col("away_features").struct.field("team_points")).alias("team_points_diff")
    ])
)
```

**Avantages** :
- Moins de colonnes (2 structs au lieu de 2000 colonnes)
- Jointure plus simple
- Structure plus logique

**Inconvénients** :
- Nécessite de refactorer toute la logique
- Les modèles ML peuvent avoir du mal avec les structs (dépend du framework)

---

## 🏗️ Analyse du Design Global du Projet

### ✅ Points Forts

1. **Architecture en Couches (Bronze/Silver/Gold)**
   - ✅ Séparation claire des responsabilités
   - ✅ Data lake pattern bien implémenté
   - ✅ Partitionnement par saison

2. **Utilisation de Polars**
   - ✅ Bon choix pour la performance
   - ✅ Lazy evaluation bien utilisée (après nos optimisations)
   - ✅ Streaming activé

3. **Configuration Centralisée**
   - ✅ `Settings` avec Pydantic
   - ✅ Chemins centralisés dans `DataPaths`
   - ✅ Configuration via TOML/env

4. **Logging Structuré**
   - ✅ Structlog pour les logs JSON
   - ✅ Logging centralisé

5. **Feature Store (Feast)**
   - ✅ Bonne idée pour la gestion des features
   - ✅ Point-in-time correctness

### ⚠️ Points à Améliorer

1. **Explosion de Features**
   - ❌ 7 fenêtres × ~50 colonnes = ~700 features par dataset
   - ❌ Multiplié par home/away = ~1400 colonnes
   - ❌ Plus les diffs = ~2100 colonnes finales
   - 💡 **Recommandation** : Réduire à 3-4 fenêtres les plus importantes (3, 5, 10, 25)

2. **Manque de Sélection de Features**
   - ❌ Toutes les features sont générées, même celles qui ne sont pas utilisées
   - ❌ Pas de feature selection avant le training
   - 💡 **Recommandation** : Ajouter une étape de feature selection (mutual information, correlation, etc.)

3. **Jointure Home/Away Coûteuse**
   - ❌ La jointure avec 2000 colonnes est le goulot d'étranglement
   - 💡 **Recommandation** : Implémenter l'Alternative 1 (Pivot) ou Alternative 2 (Streaming)

4. **Pas de Caching Intermédiaire**
   - ❌ Les transformations silver sont recalculées à chaque fois
   - 💡 **Recommandation** : Ajouter un système de cache basé sur `ingest_ts`

5. **Gestion d'Erreurs Limitée**
   - ❌ Pas de retry logic
   - ❌ Pas de validation robuste des données
   - 💡 **Recommandation** : Ajouter des validations et retry logic

6. **Tests Absents**
   - ❌ Pas de tests unitaires visibles
   - ❌ Pas de tests d'intégration
   - 💡 **Recommandation** : Ajouter pytest avec tests unitaires et d'intégration

7. **Documentation du Code**
   - ⚠️ Certaines fonctions manquent de docstrings
   - ⚠️ Pas de documentation des schémas de données
   - 💡 **Recommandation** : Ajouter des docstrings complètes et documenter les schémas

8. **Duplication de Code**
   - ⚠️ `_load_games()`, `_load_boxscores()` dupliqués dans plusieurs fichiers
   - 💡 **Recommandation** : Créer un module `data_loaders.py` partagé

9. **Pas de Monitoring de Performance**
   - ❌ Pas de métriques de temps d'exécution
   - ❌ Pas de monitoring de mémoire
   - 💡 **Recommandation** : Ajouter des métriques (temps, mémoire, taille des datasets)

10. **Configuration des Fenêtres Hardcodée**
    - ⚠️ `feature_windows = (3, 5, 10, 25, 50, 100, 200)` est fixe
    - 💡 **Recommandation** : Permettre la configuration par dataset/transformation

### 🎯 Recommandations Prioritaires

1. **Court Terme** (Cette semaine)
   - ✅ Implémenter l'Alternative 2 (Streaming join) pour résoudre le crash
   - ✅ Réduire `feature_windows` à `(3, 5, 10, 25)` 
   - ✅ Ajouter des validations de données

2. **Moyen Terme** (Ce mois)
   - Ajouter une étape de feature selection
   - Implémenter un système de cache
   - Ajouter des tests unitaires pour les transformations critiques
   - Documenter les schémas de données

3. **Long Terme** (Ce trimestre)
   - Refactorer pour utiliser Feast pour la jointure (Alternative 3)
   - Ajouter du monitoring de performance
   - Implémenter des tests d'intégration
   - Créer un module de data loaders partagé

### 📊 Comparaison avec l'Ancien Système (backup/v2)

**Ancien Système** :
- ✅ Plus simple (Pandas, tout en mémoire)
- ✅ Moins de features (pas de rolling windows aussi nombreux)
- ❌ Moins scalable
- ❌ Pas de séparation bronze/silver/gold
- ❌ Pas de feature store

**Nouveau Système** :
- ✅ Plus scalable (Polars, lazy evaluation)
- ✅ Architecture moderne (data lake, feature store)
- ❌ Plus complexe
- ❌ Problèmes de mémoire avec trop de features
- ❌ Jointure home/away coûteuse

**Verdict** : Le nouveau système est meilleur architecturalement, mais souffre d'un problème de design : trop de features générées sans sélection. Il faut trouver un équilibre.

