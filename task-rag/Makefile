# Task Management RAG Template - Makefile
# Development and deployment commands

.PHONY: help install dev bot test fmt lint clean setup check-env

# Default target
help:
	@echo "Task Management RAG Template - Available Commands:"
	@echo ""
	@echo "Development:"
	@echo "  make install    - Install all dependencies"
	@echo "  make dev        - Start FastAPI development server"
	@echo "  make bot        - Start Telegram bot (polling mode)"
	@echo "  make setup      - Initial project setup"
	@echo ""
	@echo "Testing:"
	@echo "  make test       - Run all tests"
	@echo "  make test-cov   - Run tests with coverage report"
	@echo "  make test-watch - Run tests in watch mode"
	@echo ""
	@echo "Code Quality:"
	@echo "  make fmt        - Format code with ruff and black"
	@echo "  make lint       - Run linting checks"
	@echo "  make check      - Run all quality checks"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean      - Clean temporary files and caches"
	@echo "  make check-env  - Validate environment configuration"
	@echo "  make logs       - View application logs"

# Installation and setup
install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

setup: install
	@echo "Setting up project..."
	@if [ ! -f .env ]; then \
		echo "Creating .env file from template..."; \
		cp .env.example .env; \
		echo "Please edit .env file with your configuration"; \
	fi
	@mkdir -p data/uploads data/chroma logs
	@echo "Project setup complete!"
	@echo ""
	@echo "Next steps:"
	@echo "1. Edit .env file with your API keys and configuration"
	@echo "2. Run 'make check-env' to validate configuration"
	@echo "3. Run 'make dev' to start the development server"

# Development servers
dev:
	@echo "Starting FastAPI development server..."
	@echo "Server will be available at: http://localhost:8000"
	@echo "API documentation: http://localhost:8000/docs"
	@echo "Press Ctrl+C to stop"
	uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

bot:
	@echo "Starting Telegram bot in polling mode..."
	@echo "Press Ctrl+C to stop"
	python -m bots.telegram_bot

# Testing
test:
	@echo "Running tests..."
	python -m pytest tests/ -v --tb=short

test-cov:
	@echo "Running tests with coverage..."
	python -m pytest tests/ -v --cov=app --cov-report=html --cov-report=term-missing

test-watch:
	@echo "Running tests in watch mode..."
	python -m pytest tests/ -v --tb=short -f

# Code quality
fmt:
	@echo "Formatting code..."
	python -m ruff check --fix app/ bots/ tests/
	python -m black app/ bots/ tests/
	@echo "Code formatting complete!"

lint:
	@echo "Running linting checks..."
	python -m ruff check app/ bots/ tests/
	python -m black --check app/ bots/ tests/
	@echo "Linting complete!"

check: lint test
	@echo "All quality checks passed!"

# Utilities
clean:
	@echo "Cleaning temporary files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf htmlcov/ .coverage 2>/dev/null || true
	@echo "Cleanup complete!"

check-env:
	@echo "Validating environment configuration..."
	@python -c "\
import os; \
from pathlib import Path; \
\
if not Path('.env').exists(): \
    print('❌ .env file not found. Run: make setup'); \
    exit(1); \
\
from dotenv import load_dotenv; \
load_dotenv(); \
\
required_vars = ['OPENAI_API_KEY', 'TELEGRAM_BOT_TOKEN']; \
optional_vars = ['TELEGRAM_WEBHOOK_SECRET', 'TELEGRAM_WEBHOOK_URL', 'MODEL_NAME', 'EMBEDDINGS_MODEL', 'LOG_LEVEL']; \
\
missing_required = [var for var in required_vars if not os.getenv(var)]; \
missing_optional = [var for var in optional_vars if not os.getenv(var)]; \
\
if missing_required: \
    print('❌ Missing required environment variables:'); \
    [print(f'   - {var}') for var in missing_required]; \
    print('Please add these to your .env file'); \
    exit(1); \
\
if missing_optional: \
    print('⚠️  Optional environment variables not set (using defaults):'); \
    [print(f'   - {var}') for var in missing_optional]; \
\
print('✅ Environment configuration is valid!'); \
\
dirs = ['data/uploads', 'data/chroma', 'logs']; \
[Path(dir_path).mkdir(parents=True, exist_ok=True) for dir_path in dirs]; \
\
print('✅ Required directories created!');"

logs:
	@echo "Viewing application logs..."
	@if [ -f logs/app.log ]; then \
		tail -f logs/app.log; \
	else \
		echo "No log file found. Start the application first with 'make dev'"; \
	fi

# Docker support (optional)
docker-build:
	@echo "Building Docker image..."
	docker build -t task-management-rag .

docker-run:
	@echo "Running Docker container..."
	docker run -p 8000:8000 --env-file .env task-management-rag

# Production deployment helpers
prod-install:
	@echo "Installing production dependencies..."
	pip install -r requirements.txt --no-dev

prod-start:
	@echo "Starting production server..."
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4

# Database migration helpers (for future database integration)
migrate:
	@echo "Database migrations not implemented yet (using in-memory storage)"

# Backup and restore (for future implementation)
backup:
	@echo "Creating backup of data directory..."
	@timestamp=$(date +%Y%m%d_%H%M%S); \
	tar -czf "backup_$timestamp.tar.gz" data/ logs/ || true; \
	echo "Backup created: backup_$timestamp.tar.gz"

# Development utilities
shell:
	@echo "Starting Python shell with app context..."
	python -c "
import sys
sys.path.append('.')
from app.config import Settings
from app.services.rag_service import initialize_rag_service
from app.services.task_service import initialize_task_service
settings = Settings()
print('Task Management RAG Shell')
print('Available: settings, initialize_rag_service, initialize_task_service')
import code
code.interact(local=locals())
"

# Health check
health:
	@echo "Checking application health..."
	@curl -s http://localhost:8000/healthz | python -m json.tool || echo "Application not running or health check failed"

# Quick start for new developers
quickstart: setup check-env
	@echo ""
	@echo "🚀 Quick Start Complete!"
	@echo ""
	@echo "Your Task Management RAG Template is ready!"
	@echo ""
	@echo "To start developing:"
	@echo "1. Run 'make dev' to start the API server"
	@echo "2. Run 'make bot' in another terminal to start the Telegram bot"
	@echo "3. Visit http://localhost:8000/docs for API documentation"
	@echo ""
	@echo "Happy coding! 🎉"


