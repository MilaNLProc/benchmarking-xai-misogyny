# Benchmarking Post-Hoc Interpretability Approaches for Transformer-based Misogyny Detection

In this paper, we benchmark four post-hoc explanation methods on two misogyny identification datasets across two languages, English and Italian. We evaluate explanations in terms of plausibility and faithfulness.
We demonstrate that not every token attribution method provides reliable insights and that attention cannot serve as explanation.

ACL bibkey: `attanasio-etal-2022-benchmarking`

```bibtex
@inproceedings{attanasio-etal-2022-benchmarking,
    title = "Benchmarking Post-Hoc Interpretability Approaches for Transformer-based Misogyny Detection",
    author = "Attanasio, Giuseppe  and
      Nozza, Debora  and
      Pastor, Eliana  and
      Hovy, Dirk",
    booktitle = "Proceedings of NLP Power! The First Workshop on Efficient Benchmarking in NLP",
    month = may,
    year = "2022",
    address = "Dublin, Ireland",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.nlppower-1.11",
    doi = "10.18653/v1/2022.nlppower-1.11",
    pages = "100--112",
    abstract = "Transformer-based Natural Language Processing models have become the standard for hate speech detection. However, the unconscious use of these techniques for such a critical task comes with negative consequences. Various works have demonstrated that hate speech classifiers are biased. These findings have prompted efforts to explain classifiers, mainly using attribution methods. In this paper, we provide the first benchmark study of interpretability approaches for hate speech detection. We cover four post-hoc token attribution approaches to explain the predictions of Transformer-based misogyny classifiers in English and Italian. Further, we compare generated attributions to attention analysis. We find that only two algorithms provide faithful explanations aligned with human expectations. Gradient-based methods and attention, however, show inconsistent outputs, making their value for explanations questionable for hate speech detection tasks.",
}
```

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
