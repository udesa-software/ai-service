from unittest.mock import MagicMock, patch

import numpy as np

from src.embeddings.model import EmbeddingModel, _to_sentence_embedding


class TestToSentenceEmbedding:
    def test_mean_pools_token_vectors_and_normalizes(self):
        tokens = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        result = _to_sentence_embedding(tokens)
        assert result.shape == (2,)
        assert np.isclose(np.linalg.norm(result), 1.0)

    def test_keeps_already_pooled_vector_normalized(self):
        vector = np.array([3.0, 4.0], dtype=np.float32)
        result = _to_sentence_embedding(vector)
        assert np.isclose(result[0], 0.6)
        assert np.isclose(result[1], 0.8)


class TestEmbeddingModel:
    @patch("src.embeddings.model._get_client")
    def test_encode_calls_hf_feature_extraction(self, mock_get_client):
        client = MagicMock()
        client.feature_extraction.return_value = np.array([1.0, 0.0], dtype=np.float32)
        mock_get_client.return_value = client

        model = EmbeddingModel()
        result = model.encode("Hola mundo")

        client.feature_extraction.assert_called_once_with(
            "Hola mundo",
            model="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
            normalize=True,
        )
        assert result.shape == (2,)
        assert np.isclose(np.linalg.norm(result), 1.0)
