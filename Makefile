.PHONY: help build build-dev up up-dev down logs shell clean size

# Default target
help: ## Show available commands
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}'

# Production builds
build: ## Build optimized production image
	@echo "🔨 Building production image..."
	docker compose build app

up: ## Start production app
	@echo "🚀 Starting production app..."
	docker compose up -d app

# Development builds
build-dev: ## Build development image
	@echo "🔨 Building development image..."
	docker compose build app-dev

up-dev: ## Start development app (reuses local .venv)
	@echo "🚀 Starting development app..."
	@if [ -d ".venv" ]; then \
		echo "✅ Local .venv found - will be reused"; \
	else \
		echo "⚠️  No local .venv found - will create new one"; \
	fi
	docker compose --profile dev up -d app-dev

# Quick development start
dev: build-dev up-dev ## Build and start development app

# Management commands
down: ## Stop and remove containers
	docker compose --profile dev down

logs: ## Show app logs
	docker compose logs -f app 2>/dev/null || docker compose --profile dev logs -f app-dev

shell: ## Open shell in running container
	@if docker compose ps app | grep -q "Up"; then \
		docker compose exec app sh; \
	elif docker compose ps app-dev | grep -q "Up"; then \
		docker compose --profile dev exec app-dev sh; \
	else \
		echo "❌ No running containers found. Start with 'make up' or 'make up-dev'"; \
	fi

# Cleanup commands
clean: ## Remove containers and unused images
	@echo "🧹 Cleaning up..."
	docker compose --profile dev down --volumes
	docker system prune -f

full-clean: ## Complete cleanup including all images
	@echo "🧹 Full cleanup..."
	docker compose --profile dev down --volumes --rmi all
	docker system prune -a -f

# Utility commands
size: ## Show Docker image sizes
	@echo "📊 Image sizes:"
	@docker images | grep contentgen || echo "No images built yet"

config: ## Validate docker-compose configuration
	docker compose config

# Local environment setup
venv-setup: ## Create optimized local virtual environment
	@echo "🔧 Setting up local virtual environment..."
	@if command -v pdm >/dev/null 2>&1; then \
		pdm install --without dev --without test; \
	else \
		echo "❌ PDM not found. Install with: pip install pdm"; \
		exit 1; \
	fi

# Combined workflows
fresh: full-clean build up ## Fresh production build and start
fresh-dev: full-clean venv-setup dev ## Fresh development build with local .venv
