.PHONY: check lint format typecheck arch test api ui ui-build

lint:
	uv run ruff check src tests
	uv run ruff format --check src tests

format:
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck:
	uv run mypy src

arch:
	uv run lint-imports
	uv run pytest tests/test_architecture.py -v

test:
	uv run pytest -v

check: lint typecheck arch test

api:
	uv run llm-rag-faq-api --reload

ui:
	npm --prefix frontend run dev

ui-build:
	npm --prefix frontend run build
