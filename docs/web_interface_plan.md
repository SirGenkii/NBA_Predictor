## Interface web & stack Docker

### Objectifs
- Visualiser rapidement les runs générés dans `data/mass_prediction_nba/runs/**/nba-run-*.json`.
- Afficher la recommandation avec l'edge maximal et le Kelly associé.
- Montrer l'état en temps réel du watcher via `data/mass_prediction_nba/watcher_state.json`.
- Offrir un point d'upload (optionnel) pour pousser les captures depuis mobile quand le watcher tourne sur le VPS.

### Architecture proposée
| Composant | Description |
| --- | --- |
| `web-api` | Service FastAPI/Uvicorn (Python 3.11) qui lit directement les JSON, expose des endpoints REST et sert de BFF pour le front. |
| `web-ui` | Client Next.js ou SvelteKit + Tailwind fournissant la liste des matchs, les graphiques et le statut du watcher. |
| Volume partagé | Bind `./data/mass_prediction_nba` dans les deux conteneurs pour accéder aux runs, logs, screenshots et watcher_state. |
| Reverse proxy (optionnel) | Traefik/Caddy ou nginx pour router `/api` → web-api et `/` → web-ui quand déployé sur le VPS. |

### Docker Compose
```
services:
  web-api:
    build:
      context: ./web/api
      dockerfile: Dockerfile
    environment:
      NBA_MASS_DATA_DIR: /data/mass_prediction_nba
    volumes:
      - ./data/mass_prediction_nba:/data/mass_prediction_nba:ro
    ports:
      - "8000:8000"

  web-ui:
    build:
      context: ./web/ui
      dockerfile: Dockerfile
      args:
        VITE_API_BASE_URL: http://localhost:8000
    depends_on:
      - web-api
    ports:
      - "4173:4173"
```
> Les dossiers `web/api` et `web/ui` peuvent être initialisés plus tard (`uvicorn` côté API, `pnpm create next-app` côté front). La stack reste isolée du watcher qui continue de tourner via `make nba-mass-start`.

### Commandes Make disponibles
- `make web-api-dev`: installe `web/api/requirements.txt` dans `.venv` si besoin puis lance Uvicorn (`NBA_MASS_DATA_DIR` pointant vers `$(WEB_DATA_DIR)`).
- `make web-ui-dev`: installe les dépendances NPM dans `web/ui` (via `NODE_BIN`, par défaut `~/.local/share/node/...`) et démarre Vite (`VITE_API_BASE_URL` configurable, `4173` par défaut).
- `make web-ui-build`: build production du front (bundle static pour le Dockerfile ou un hébergement S3).
- `make web-stack-up` / `make web-stack-down`: orchestrent les deux conteneurs à partir de `docker-compose.web.yml`.
> Ajustez `NODE_BIN`, `WEB_API_PORT`, `WEB_UI_PORT`, `VITE_API_BASE_URL` via l'environnement ou la ligne de commande (`make WEB_API_PORT=9000 web-api-dev`) selon les besoins locaux ou VPS.

### API FastAPI (esquisse)
- `GET /runs`: pagination, filtres par date, source (`match.metadata.source`), booléen `best_only`.
- `GET /runs/{run_id}`: détails du run, matches, liens vers screenshots (`match.screenshot_path`), label du modèle.
- `GET /matches/{run_id}/{index}` ou par `game_id`: retourne `prediction.pivot_probabilities`, `evaluations`, `summary`.
- `GET /watcher-state`: expose tel quel `watcher_state.json` (nouveaux champs `processed_total`, `last_source`, `model_label`, etc.).
- `POST /uploads`: (optionnel) endpoint authentifié pour déposer une capture qui sera vue par le watcher.

### Front-end
- Page principale: cartes par match (logo équipes, cote, recommandation, edge/kelly). Bouton pour ouvrir la courbe cumulative (`pivot_probabilities`).
- Barre latérale "Top EV bets": tri sur `evaluation.over/under.edge`, mention du stake `kelly.scaled`.
- Bandeau statut watcher: badge vert/orange/rouge selon `state` et fraîcheur du `heartbeat`. Affiche `last_run_id`, `last_source`, `model_label`, `processed_total`.
- Section logs récents: lecture tail de `data/mass_prediction_nba/logs/nba_mass_prediction_watcher.log` exposée par l'API (optionnel).

### Watcher state amélioré
- `src/scripts/mass_prediction_watcher.py` maintient désormais `processed_total`, `last_run_id`, `last_source`, `model_label` et détecte les signaux (`SIGTERM`, `SIGQUIT`, `SIGHUP`) pour passer automatiquement à `state="stopped"`.
- En cas d'erreur, `last_error` contient `{message, source, at}` permettant d'afficher une alerte dans l'UI.
- Le front doit considérer le watcher "degradé" si `now - heartbeat > 2 * poll_interval` ou si `state != "running"`.

### Roadmap côté VPS
1. Déployer le repo (code + data) et lancer le watcher via `make nba-mass-start` (systemd ou tmux).
2. Utiliser `make model-export-prod MODEL_EXPORT_HOST=user@vps MODEL_EXPORT_PATH=/srv/nba/models` pour pousser les artefacts.
3. Installer docker compose + builder sur le VPS, publier les images (`docker compose -f docker-compose.web.yml up -d`).
4. Ajouter un certificat TLS et un reverse proxy si nécessaire, puis brancher la collecte des screenshots (SFTP, API upload ou montage Nextcloud).
