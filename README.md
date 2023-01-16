# Benchmarking Post-Hoc Interpretability Approaches for Transformer-based Misogyny Detection

**TL;DR**

In this paper, we benchmark four post-hoc explanation methods on two misogyny identification datasets across two languages, English and Italian. We evaluate explanations in terms of plausibility and faithfulness.
We demonstrate that not every token attribution method provides reliable insights and that attention cannot serve as explanation.

## Getting Started

You need a functioning Python 3.6+ interpreter. You suggest to create a dedicate virtual environment and install the dependencies as follows:

```bash
pip install -r requirements.txt
```

We base our evaluation on Kennedy's [implementation](https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations) of the Sampling and Occlusion algorithm. Hence, you will need to clone the library and either install it or add it to `sys.path` variable.

From the root of the repository, you can install the library with:

```bash
git clone https://github.com/BrendanKennedy/contextualizing-hate-speech-models-with-explanations
pip install -e contextualizing-hate-speech-models-with-explanations
```

or simply add it to your `sys.path` variable in python:

```python
SOC_DIR = "./contextualizing-hate-speech-models-with-explanations/"

import sys
if SOC_DIR not in sys.path:
    sys.path.append(SOC_DIR)
```
