.PHONY: install test preview notebook-check

install:
	python -m pip install --upgrade pip
	pip install -e ".[dev,notebooks]"

test:
	pytest

preview:
	python scripts/make_readme_preview.py

notebook-check:
	pytest --nbmake notebooks/01_bernoulli_exponential_tilting.ipynb
	pytest --nbmake notebooks/02_cramer_bernoulli_tail_risk.ipynb