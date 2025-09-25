.PHONY: help run dev build up down logs shell clean

help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

run: ## 🚀 Smart run - checks PDM, .venv, builds and starts development
	@echo "🔍 Checking environment..."
	@# Check if PDM is installed
	@if ! command -v pdm >/dev/null 2>&1; then \
		echo "❌ PDM not found. Installing..."; \
		pip install pdm; \
	else \
		echo "✅ PDM found"; \
	fi
	@# Check/create local .venv
	@if [ ! -d ".venv" ]; then \
		echo "📦 Creating local .venv..."; \
		pdm install --without dev --without test; \
	else \
		echo "✅ Local .venv exists"; \
	fi
	@echo "🏗️  Building and starting development container..."
	@docker compose build app-dev
	@docker compose --profile dev up -d app-dev
	@echo "✅ Development container started!"

dev: run ## Alias for run

build: ## Build production image
	@echo "🔨 Building production image..."
	@docker compose build --target production app

up: ## Start production container
	@echo "🚀 Starting production container..."
	@docker compose up -d app

down: ## Stop all containers
	@docker compose --profile dev down

logs: ## Show logs
	@if docker compose --profile dev ps app-dev 2>/dev/null | grep -q "Up"; then \
		docker compose --profile dev logs -f app-dev; \
	else \
		docker compose logs -f app; \
	fi

shell: ## Open shell in running container
	@if docker compose --profile dev ps app-dev 2>/dev/null | grep -q "Up"; then \
		docker compose --profile dev exec app-dev sh; \
	else \
		docker compose exec app sh; \
	fi

clean: ## Clean up containers and images
	@echo "🧹 Cleaning up..."
	@docker compose --profile dev down --volumes
	@docker system prune -f
