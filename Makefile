.PHONY: up down build logs status

# Export UID/GID for docker-compose
export UID := $(shell id -u)
export GID := $(shell id -g)

up: ## Start paperflow daemon in background
	docker compose up -d --build

down: ## Stop and remove paperflow container
	docker compose down

build: ## Build the Docker image without starting
	docker compose build

logs: ## Follow container logs
	docker compose logs -f

status: ## Show container status
	docker compose ps

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

.DEFAULT_GOAL := help
