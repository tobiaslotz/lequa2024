from setuptools import setup

def readme():
    with open("README.md") as f:
        return f.read()

setup(
    name="lequa2024",
    version="0.0.1-rc",
    description="Our submission to the LeQua 2024 competition",
    long_description=readme(),
    classifiers=[
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Development Status :: 3 - Alpha",
    ],
    keywords=[
        "machine-learning",
        "supervised learning",
        "quantification",
        "supervised prevalence estimation",
    ],
    url="https://github.com/tobiaslotz/lequa2024",
    author="Tobias Lotz",
    author_email="tobias.lotz@tu-dortmund.de",
    packages=setuptools.find_packages(),
    install_requires=[
        "jax[cpu]",
        "numpy",
        "quapy",
        "qunfold @ git+https://github.com/mirkobunse/qunfold@v0.1.3",
        "scikit-learn",
        "scipy",
    ],
    python_requires=">=3.11",
    include_package_data=True,
    zip_safe=False,
    test_suite="nose.collector",
    extras_require = {
        "tests": ["nose"],
        "plots": ["plotly==5.21.0", "notebook>=5.3", "ipywidgets>=7.5"]
    }
)
