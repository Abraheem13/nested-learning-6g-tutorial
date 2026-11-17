# Nested Learning for AI-Native 6G Networks
WORKERS ?= $(shell python3 -c "import os;print(max(1,(os.cpu_count() or 2)-1))")
PY      ?= python3
RUN      = $(PY) scripts/run_experiments.py
REPORT   = $(PY) scripts/make_report.py

.PHONY: help install test smoke quick full extended report all clean plan lint \
        robustness noise review chapter verify pdf reproduce

help:
	@echo "make install    install the package (editable) and dependencies"
	@echo "make test       run the test suite (~15 s)"
	@echo "make smoke      verify the pipeline end to end (~1 min)"
	@echo "make quick      indicative results (~8 min on 4 cores)"
	@echo "make full       the reported results (~20 min on 4 cores)"
	@echo "make extended   longer-horizon robustness check (~45 min)"
	@echo "make report     build tables, figures and REPORT.md"
	@echo "make all        test + full + report"
	@echo "make plan       print the run plan without training"
	@echo ""
	@echo "Chapter (run in this order, or 'make reproduce' for everything):"
	@echo "make robustness controls 1-3 and the powered ablation"
	@echo "make noise      control 4, the precondition probe"
	@echo "make review     partition sensitivity and drift structure"
	@echo "make chapter    regenerate every number, table and data figure"
	@echo "make verify     check the manuscript against the run logs"
	@echo "make pdf        flatten and compile in svmult (needs TeX Live)"
	@echo "make clean      delete results/ (they are all regenerable)"
	@echo ""
	@echo "detected workers: $(WORKERS)   override with: make full WORKERS=8"

install:
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest tests/ -q

smoke:
	$(RUN) --preset smoke --workers $(WORKERS)

quick:
	$(RUN) --preset quick --workers $(WORKERS)

full:
	$(RUN) --preset full --workers $(WORKERS)

extended:
	$(RUN) --preset extended --workers $(WORKERS)

plan:
	$(RUN) --preset full --dry-run

report:
	$(REPORT)

all: test full report

robustness:
	$(PY) scripts/run_robustness.py --workers $(WORKERS)

noise:
	$(PY) scripts/run_noise_probe.py --workers $(WORKERS)

review:
	$(PY) scripts/run_review_analyses.py --workers $(WORKERS)

chapter:
	$(PY) scripts/make_chapter_assets.py

verify:
	$(PY) scripts/verify_chapter.py

pdf:
	$(PY) scripts/build_chapter.py --zip

reproduce: test full report robustness noise review chapter verify

lint:
	ruff check src scripts tests

clean:
	rm -rf results
