# LLM Prompting with DSPy

This project demonstrates how to use Stanford's [DSPy](https://github.com/stanfordnlp/dspy) for dynamic few-shot prompting with KNN-based retrieval and prompt optimization.  
It includes modular pipelines and demo notebooks for both general few-shot retrieval and healthcare chatbot prompt optimization.

---

## Quick Start

1. **Clone the repository**
    ```bash
    git clone https://github.com/ShabnamLabs/knn-fewshot-prompting-dspy.git
    cd knn-fewshot-prompting-dspy
    ```

2. **Install dependencies**
    ```bash
    pip3 install -r requirements.txt
    ```

3. **Run the demo notebooks**
    - Open `demos/knn_fewshot/knn_fewshot_demo.ipynb` for KNN-based few-shot prompting.
    - Open `demos/prompt_optimization/prompt_optimization_demo.ipynb` for prompt optimization in a healthcare chatbot scenario.

---

## Project Structure

```
knn-fewshot-prompting-dspy/
├── README.md
├── requirements.txt
├── demos/
│   ├── knn_fewshot/
│   │   ├── knn_fewshot_demo.ipynb
│   │   └── data/
│   │       └── examples.json
│   └── prompt_optimization/
│       ├── prompt_optimization_demo.ipynb
│       └── data/
│           ├── train_examples.json
│           └── val_examples.json
├── src/
│   ├── knn_pipeline.py
│   └── prompt_optimization_pipeline.py
└── .devcontainer/
    └── devcontainer.json
```

---

## About DSPy

DSPy is a framework for building modular, compositional pipelines for language model applications.  
This repo demonstrates how to use DSPy signatures and modules with KNN-based few-shot retrieval and prompt optimization for robust prompting.

- **KNN Few-Shot Demo:** Shows dynamic selection of few-shot examples using semantic similarity.
- **Prompt Optimization Demo:** Uses DSPy and MIPROv2 to optimize prompts for safety and helpfulness in healthcare chatbot responses.

For more, see [DSPy on GitHub](https://github.com/stanfordnlp/dspy).

