import pytest
from unittest.mock import Mock, patch
import numpy as np

import dspy
from dspy.clients.embedding import Embedder


# Mock response format similar to litellm's embedding response.
class MockEmbeddingResponse:
    def __init__(self, embeddings):
        self.data = [{"embedding": emb} for emb in embeddings]
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        self.model = "mock_model"
        self.object = "list"


async def test_litellm_embedding():
    model = "text-embedding-ada-002"
    inputs = ["hello", "world"]
    mock_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
    ]

    with patch("litellm.aembedding") as mock_litellm:
        # Configure mock to return proper response format.
        mock_litellm.return_value = MockEmbeddingResponse(mock_embeddings)

        # Create embedding instance and call it.
        embedding = Embedder(model)
        result = await embedding(dspy.settings, inputs)

        # Verify litellm was called with correct parameters.
        mock_litellm.assert_called_once_with(model=model, input=inputs, caching=True)

        assert len(result) == len(inputs)
        np.testing.assert_allclose(result, mock_embeddings)


async def test_callable_embedding():
    inputs = ["hello", "world", "test"]

    expected_embeddings = [
        [0.1, 0.2, 0.3],  # embedding for "hello"
        [0.4, 0.5, 0.6],  # embedding for "world"
        [0.7, 0.8, 0.9],  # embedding for "test"
    ]

    async def mock_embedding_fn(settings, texts):
        # Simple callable that returns random embeddings.
        return expected_embeddings

    # Create embedding instance with callable
    embedding = Embedder(mock_embedding_fn)
    result = await embedding(dspy.settings, inputs)

    np.testing.assert_allclose(result, expected_embeddings)


async def test_invalid_model_type():
    # Test that invalid model type raises ValueError
    with pytest.raises(ValueError):
        embedding = Embedder(123)  # Invalid model type
        await embedding(dspy.settings, ["test"])
