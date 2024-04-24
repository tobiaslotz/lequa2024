## LeQua2024 Competition

Repository that contains our submission to the LeQua2024 competition.

```sh
# installation
python -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel
venv/bin/pip install certifi # bug-fix for quapy v0.1.8
venv/bin/pip install -e .

# testing
venv/bin/python testing.py

# evaluation on T1B of LeQua2022
venv/bin/python -m lequa2024.experiments.lequa2022 --n_jobs 5 results_lequa2022.csv
```
