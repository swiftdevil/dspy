import asyncio

import numpy as np

from dspy.clients import Embedder
from dspy.dsp.utils import Settings
from dspy.primitives import Example


class KNN:
    def __init__(self, settings: Settings, k: int, trainset: list[Example], vectorizer: Embedder):
        """
        A k-nearest neighbors retriever that finds similar examples from a training set.

        Args:
            k: Number of nearest neighbors to retrieve
            trainset: List of training examples to search through
            vectorizer: The `Embedder` to use for vectorization

        Example:
            ```python
            import dspy
            from sentence_transformers import SentenceTransformer

            # Create a training dataset with examples
            trainset = [
                dspy.Example(input="hello", output="world"),
                # ... more examples ...
            ]

            # Initialize KNN with a sentence transformer model
            knn = KNN(
                k=3,
                trainset=trainset,
                vectorizer=dspy.Embedder(SentenceTransformer("all-MiniLM-L6-v2").encode)
            )

            # Find similar examples
            similar_examples = knn(input="hello")
            ```
        """
        self.k = k
        self.trainset = trainset
        self.embedding = vectorizer
        trainset_casted_to_vectorize = [
            " | ".join([f"{key}: {value}" for key, value in example.items() if key in example._input_keys])
            for example in self.trainset
        ]
        self.trainset_vectors = (asyncio.get_event_loop().run_until_complete(self.embedding(settings, trainset_casted_to_vectorize)).astype(np.float32))

    async def __call__(self, settings, **kwargs) -> list:
        input_example_vector = await self.embedding(settings, [" | ".join([f"{key}: {val}" for key, val in kwargs.items()])])
        scores = np.dot(self.trainset_vectors, input_example_vector.T).squeeze()
        nearest_samples_idxs = scores.argsort()[-self.k :][::-1]
        return [self.trainset[cur_idx] for cur_idx in nearest_samples_idxs]
