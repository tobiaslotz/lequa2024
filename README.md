## LeQua2024 Competition

Repository that contains our submission to the LeQua2024 competition.

```sh
# installation
python -m venv venv
venv/bin/pip install --upgrade pip setuptools wheel
venv/bin/pip install certifi # bug-fix for quapy v0.1.8
venv/bin/pip install -e .

# testing of LeQua2024
venv/bin/python testing.py

# testing of a LeQua2022 experiment, which we use for development
make test

# unit testing
make unittest

# evaluation on T1B of LeQua2022
make
```

# Slurm

To start a Slurm job on our computer cluster, use

```sh
./srun.sh
```

Within the job, you must use the system-wide installation of python / pip to use GPU acceleration.

```sh
pip install --upgrade pip setuptools wheel
pip install certifi # bug-fix for quapy v0.1.8
pip install -e .
```
