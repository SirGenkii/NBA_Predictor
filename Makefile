DC := docker compose  # Pour Docker Compose V2 (officiel)
# DC := docker-compose  # Pour compatibilité Docker Compose V1

up:
	$(DC) up -d

down:
	$(DC) down

logs:
	$(DC) logs -f

restart:
	$(DC) down && $(DC) up -d

clean:
	$(DC) down -v

ps:
	$(DC) ps

exec:
	$(DC) exec jupyter-gpu bash
