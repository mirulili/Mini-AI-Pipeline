# Python version
PYTHON = python3.11

# Dependencies
REQUIREMENTS = requirements.txt

SUBSET10 = 10
SUBSET50 = 50
SUBSET100 = 100

.PHONY: all install run clean help

all: help

install: ## Install dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r $(REQUIREMENTS)

run: ## Run the application locally
	$(PYTHON) -m src.baseline
evaluate-all: ## Evaluate the model on the dev set
	$(PYTHON) -m src.evaluator

evaluate-10: ## Evaluate the model on the dev set with a subset of data 10
	$(PYTHON) -m src.evaluator --subset $(SUBSET10)
evaluate-50: ## Evaluate the model on the dev set with a subset of data 50
	$(PYTHON) -m src.evaluator --subset $(SUBSET50)
evaluate-100: ## Evaluate the model on the dev set with a subset of data 100
	$(PYTHON) -m src.evaluator --subset $(SUBSET100)

index: ## Build the search index from source documents
	$(PYTHON) -m src.preprocessing

clean: ## Clean up cache and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".coverage" -delete

help: ## Show this help message
	@echo "Usage: make [target]"
	@echo ""
	@echo "Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'
