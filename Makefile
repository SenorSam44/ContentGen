.PHONY: build up down logs shell run setup help

include Makefile.setup

# Build the image(s)
build: start.sh ## Build the Docker images
	docker compose build

# Start the app
up: ## Start the app
	docker compose up -d

# Stop and remove containers
down: ## Stop and remove containers
	docker compose down

# View logs
logs: ## Tail logs from the app container
	docker compose logs -f app

# Open a shell inside the running app container (use sh instead of bash for Alpine)
shell: ## Open a shell in the app container
	docker compose exec app sh

# Build and run the app
run: down build up ## Build and start the app

# Remove all stopped containers and dangling images
clean: ## Remove stopped containers and dangling images
	@echo "🧹 Cleaning up Docker..."
	@docker compose down --volumes --remove-orphans
	@docker system prune

full-clean: ## Removes python venv, all containers & images
	@echo "🧹 Running full clean up..."
	@docker compose down --volumes --remove-orphans
	@docker system prune -a -f
	rm -rf .venv venv

fresh-run: ## removes all, reinstalls all
	full-clean run
	
# Show available commands
help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'


