SOURCES=$(shell find lequa2024 -name "*.py")

results_lequa2022.csv: $(SOURCES)
	venv/bin/python -m lequa2024.experiments.lequa2022 --n_jobs 12 --is_full_run --omit_testing $@

test: test_lequa2022.csv
test_lequa2022.csv: $(SOURCES)
	venv/bin/python -m lequa2024.experiments.lequa2022 --n_jobs 2 --is_test_run $@

unittest: $(SOURCES)
	venv/bin/python -m unittest

.PHONY: test unittest
