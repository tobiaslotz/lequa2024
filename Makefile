SOURCES=$(shell find lequa2024 -name "*.py")

val_lequa2022.csv: $(SOURCES)
	venv/bin/python -m lequa2024.experiments.lequa2022 --n_jobs 1 $@ trn_lequa2022.csv

test: test_lequa2022.csv
test_lequa2022.csv: $(SOURCES)
	venv/bin/python -m lequa2024.experiments.lequa2022 --n_jobs 1 --is_test_run $@ test_trn_lequa2022.csv

unittest: $(SOURCES)
	venv/bin/python -m unittest

.PHONY: test unittest
