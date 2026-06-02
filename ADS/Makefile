.PHONY: install dev lint format typecheck test test-cov clean docker-build docker-up db-upgrade alembic-init

install:
	pip install -e ".[dev]"

dev:
	pip install -e ".[dev,docs]"

lint:
	ruff check src/ tests/

format:
	ruff check --fix src/ tests/
	ruff format src/ tests/

typecheck:
	mypy src/

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=ads --cov-report=term-missing --cov-report=html

test-fast:
	pytest tests/ -v -m "not slow" -n auto

test-model:
	pytest tests/model/ -v --timeout=300

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf dist/ build/ .coverage htmlcov/ mlruns/ logs/

docker-build:
	docker compose -f docker/docker-compose.yml build

docker-up:
	docker compose -f docker/docker-compose.yml up -d

docker-down:
	docker compose -f docker/docker-compose.yml down
