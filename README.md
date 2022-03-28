# ML Evaluation Module for DL Gold Standard
[![Python CI](https://github.com/janothan/dl-evaluation-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/janothan/dl-evaluation-framework/actions/workflows/ci.yml)
[![Lint](https://github.com/janothan/dl-evaluation-framework/actions/workflows/black.yml/badge.svg)](https://github.com/janothan/dl-evaluation-framework/actions/workflows/black.yml)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Pre-Commit Enabled](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)

A framework to evaluate knowledge graph embeddings on description logics test cases.
You can generate or download the gold standard using [this project](https://github.com/janothan/DBpediaTestCaseGenerator/).

## Usage Notes


### Minimal Python Example
```python
from dl_evaluation_framework.evaluation_manager import EvaluationManager, VectorTuple

# provide the path to the test case directory
test_directory = ""

# provide vector_name and vector_path
vlist = [VectorTuple(vector_name="", vector_path="")]

# run
em = EvaluationManager(test_directory=test_directory)
em.evaluate(vector_names_and_files=vlist)

```

## Developer Notes
This project is tested with Python [3.8, 3.9, 3.10] on macOS and Linux.

- Testing framework: [pytest](https://docs.pytest.org/en/6.2.x/)
- Docstring format: <a href="https://numpy.org/doc/stable/docs/howto_document.html">NumPy/SciPy</a>
- Code formatting: <a href="https://github.com/psf/black">black</a>
