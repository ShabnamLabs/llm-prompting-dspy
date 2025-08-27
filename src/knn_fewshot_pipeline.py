import json
import dspy
from dspy.teleprompt import KNNFewShot
from sentence_transformers import SentenceTransformer
from typing import List, Callable, Any

def load_examples(path: str) -> List[dspy.Example]:
    """
    Load examples from a JSON file and convert to dspy.Example.

    Args:
        path (str): Path to the JSON file.

    Returns:
        List[dspy.Example]: List of DSPy examples.
    """
    try:
        with open(path, "r") as f:
            raw_examples = json.load(f)
        examples = [
            dspy.Example(**ex).with_inputs(next(iter(ex)))
            for ex in raw_examples
        ]
        return examples
    except Exception as e:
        print(f"Error loading examples: {e}")
        return []

def get_vectorizer(model_name: str) -> Callable[[List[str]], Any]:
    """
    Get a sentence transformer vectorizer.

    Args:
        model_name (str): Name of the sentence-transformers model.

    Returns:
        Callable: Embedding function.
    """
    try:
        model = SentenceTransformer(model_name)
        return dspy.Embedder(model.encode)
    except Exception as e:
        print(f"Error loading embedder model: {e}")
        raise

def compile_knn_dspy_pipeline(
    trainset: List[dspy.Example],
    dspy_module: dspy.Module,
    k: int = 3,
    embedder_model_name: str = "all-MiniLM-L6-v2"
) -> Callable:
    """
    Compile the KNNFewShot DSPy pipeline.

    Args:
        trainset (List[dspy.Example]): Training examples.
        dspy_module (dspy.Module): DSPy module (e.g., dspy.Predict(SignatureClass)).
        k (int): Number of nearest neighbors to retrieve.
        embedder_model_name (str): Name of the sentence-transformers model.

    Returns:
        Callable: Compiled DSPy pipeline for inference.
    """
    vectorizer = get_vectorizer(embedder_model_name)
    knn_few_shot = KNNFewShot(k=k, trainset=trainset, vectorizer=vectorizer)
    compiled_module = knn_few_shot.compile(dspy_module)
    return compiled_module

def inference_knn_dspy_pipeline(
    compiled_dspy: Callable,
    **inputs
) -> str:
    """
    Run inference on a question using the compiled pipeline.

    Args:
        compiled_dspy (Callable): Compiled DSPy pipeline.
        **inputs: Input fields for the signature.

    Returns:
        str: Answer to the question (or output field).
    """
    result = compiled_dspy(**inputs)
    # Try to return the first output field if possible
    if isinstance(result, dict):
        # Return the first value
        return next(iter(result.values()))
    elif hasattr(result, "__dict__"):
        # Return the first attribute that is not private
        for k, v in vars(result).items():
            if not k.startswith("_"):
                return v
    return str(result)

# Example signature for question answering
class QuestionAnswer(dspy.Signature):
    """Answer the question."""
    question: str = dspy.InputField(desc="input text to answer the question")
    answer: str = dspy.OutputField(desc="the answer to the question")

if __name__ == "__main__":
    # Example usage: compile and inference are separate
    example_path = "data/examples.json"
    embedder_model_name = "all-MiniLM-L6-v2"
    k = 3
    question = "What is the capital of Belgium?"

    trainset = load_examples(example_path)
    if not trainset:
        print("No training examples loaded. Exiting.")
    else:
        qa_module = dspy.Predict(QuestionAnswer)
        compiled_qa = compile_knn_dspy_pipeline(trainset, qa_module, k, embedder_model_name)
        answer = inference_knn_dspy_pipeline(compiled_qa, question=question)
        print(f"Q: {question}")
        print(f"A: {answer}")