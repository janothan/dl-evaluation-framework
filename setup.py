from setuptools import setup

setup(
    name="dl_evaluation_framework",
    version="0.1",
    packages=["dl_evaluation_framework"],
    url="www.jan-portisch.eu",
    license="MIT",
    author="Jan Portisch",
    author_email="jan@informatik.uni-mannheim.de",
    description="Evaluation program to evaluate vectors on description logics test sets.",
    package_data={"dl_evaluation_framework": ["log.conf"]},
    install_requires=[
        "scikit-learn>=1.0.2",
        "pandas>=1.4.0",
        "numpy>=1.21.2",
    ],
)
