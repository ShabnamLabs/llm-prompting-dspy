# KNN Few-Shot Prompting with DSPy

This project shows how to use Stanford's [DSPy](https://github.com/stanfordnlp/dspy) for dynamic few-shot prompting with KNN-based retrieval.  
Few-shot examples are selected at runtime using semantic similarity, making prompt construction flexible and adaptive.

---

## Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/ShabnamLabs/knn-fewshot-prompting-dspy.git
    cd knn-fewshot-prompting-dspy
    ```

2. **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3. **Run the demo notebook**
    - Open `knn_fewshot_demo.ipynb` in VS Code or Jupyter.
    - Follow the steps to load data, compile the DSPy pipeline, and run inference.

---

## Project Structure

```
knn-fewshot-prompting-dspy/
├── README.md
├── requirements.txt
├── knn_fewshot_demo.ipynb
├── src/
│   └── knn_pipeline.py
└── data/
    └── examples.json
```

---

## About DSPy

DSPy is a framework for building modular, compositional pipelines for language model applications.  
This repo demonstrates how to use DSPy signatures and modules with KNN-based few-shot retrieval for more robust prompting.

For more, see [DSPy on GitHub](https://github.com/stanfordnlp/dspy).

