import unittest
import numpy as np
from semantic_tagger import (
    Vec,
    build_tag_matrix,
    build_text_matrix,
    preprocess_text,
    similarity_matrix_vector_class,
    similarity_matrix_numpy,
    rank_tags,
    SOME_TAGS,
)

TAGS = SOME_TAGS

TEXT_WORDS = ["manipal", "hospital", "innovative", "digital", "scientists", "campus", "research"]


class FakeModel:

    def __init__(self, words, dim=50, seed=7):
        rng = np.random.default_rng(seed)
        self.vector_size = dim
        self.key_to_index = {word: i for i, word in enumerate(words)}
        self.vectors = rng.normal(size=(len(words), dim)).astype(np.float32)

    def __getitem__(self, word):
        return self.vectors[self.key_to_index[word]]


class Test_semantic_tagger(unittest.TestCase):

    def setUp(self):
        vocab = TAGS + ["higher"] + TEXT_WORDS
        self.model = FakeModel(list(dict.fromkeys(vocab)))
        self.tokens = ["manipal", "research", "hospital", "innovative", "research", "campus"]
        self.tag_names, self.T = build_tag_matrix(self.model, TAGS)
        self.in_vocab, self.oov, self.W = build_text_matrix(self.model, self.tokens)
        self.S_vector = similarity_matrix_vector_class(self.W, self.T)
        self.S_numpy = similarity_matrix_numpy(self.W, self.T)

    def test_tag_matrix_shape(self):
        self.assertEqual(self.T.shape, (20, 50))
        self.assertEqual(len(self.tag_names), 20)

    def test_text_matrix_shape(self):
        n = len(self.in_vocab)
        self.assertEqual(n, 6)
        self.assertEqual(self.W.shape, (n, 50))

    def test_repeated_tokens_preserved(self):
        self.assertEqual(self.in_vocab.count("research"), 2)
        np.testing.assert_allclose(self.W[1], self.W[4])

    def test_similarity_shape(self):
        n = len(self.in_vocab)
        self.assertEqual(self.S_vector.shape, (n, 20))
        self.assertEqual(self.S_numpy.shape, (n, 20))

    def test_single_pair_agrees_with_numpy(self):
        i = self.in_vocab.index("hospital")
        j = self.tag_names.index("medicine")
        value = Vec(self.W[i].tolist()).cosine_similarity(Vec(self.T[j].tolist()))
        self.assertAlmostEqual(value, self.S_numpy[i, j], places=6)

    def test_allclose(self):
        self.assertTrue(np.allclose(self.S_vector, self.S_numpy, atol=1e-6))
        self.assertLess(np.max(np.abs(self.S_vector - self.S_numpy)), 1e-6)

    def test_identical_word_and_tag_similarity_is_one(self):
        i = self.in_vocab.index("research")
        j = self.tag_names.index("research")
        self.assertAlmostEqual(self.S_numpy[i, j], 1.0, places=6)

    def test_oov_token_excluded(self):
        in_vocab, oov, W = build_text_matrix(self.model, ["research", "xyzzyqwerty", "campus"])
        self.assertEqual(in_vocab, ["research", "campus"])
        self.assertEqual(oov, ["xyzzyqwerty"])
        self.assertEqual(W.shape, (2, 50))

    def test_empty_text_matrix(self):
        in_vocab, oov, W = build_text_matrix(self.model, ["xyzzyqwerty"])
        self.assertEqual(in_vocab, [])
        self.assertEqual(W.shape, (0, 50))

    def test_tag_oov_raises_error(self):
        bad_tags = TAGS[:19] + ["xyzzyqwerty"]
        with self.assertRaises(ValueError):
            build_tag_matrix(self.model, bad_tags)

    def test_multiword_tag_oov_component_raises_error(self):
        bad_tags = TAGS[:19] + ["higher xyzzyqwerty"]
        with self.assertRaises(ValueError):
            build_tag_matrix(self.model, bad_tags)

    def test_wrong_tag_count_raises_error(self):
        with self.assertRaises(ValueError):
            build_tag_matrix(self.model, TAGS[:19])

    def test_multiword_tag_is_mean(self):
        tags = TAGS[:19] + ["higher education"]
        _, T = build_tag_matrix(self.model, tags)
        expected = (self.model["higher"].astype(np.float64) + self.model["education"].astype(np.float64)) / 2
        np.testing.assert_allclose(T[19], expected)

    def test_vec_cosine_similarity(self):
        self.assertAlmostEqual(Vec([1, 0]).cosine_similarity(Vec([0, 1])), 0.0)
        self.assertAlmostEqual(Vec([1, 2, 3]).cosine_similarity(Vec([2, 4, 6])), 1.0)
        self.assertAlmostEqual(Vec([1, 2]).cosine_similarity(Vec([-1, -2])), -1.0)
        with self.assertRaises(ValueError):
            Vec([0, 0]).cosine_similarity(Vec([1, 2]))

    def test_zero_row_raises_error(self):
        W = self.W.copy()
        W[0] = 0.0
        with self.assertRaises(ValueError):
            similarity_matrix_numpy(W, self.T)

    def test_preprocessing(self):
        raw_count, tokens = preprocess_text("The Research, at #Manipal; is an ok 2026 AI!")
        self.assertEqual(raw_count, 9)
        self.assertEqual(tokens, ["research", "manipal", "2026"])

    def test_rank_tags(self):
        ranking = rank_tags(self.tag_names, self.S_numpy, self.in_vocab)
        scores = [score for _, score, _ in ranking]
        self.assertEqual(len(ranking), 20)
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(ranking[0][1], 1.0, places=6)
        for tag, score, word in ranking:
            j = self.tag_names.index(tag)
            self.assertAlmostEqual(score, self.S_numpy[:, j].max())
            self.assertEqual(word, self.in_vocab[int(np.argmax(self.S_numpy[:, j]))])

if __name__ == "__main__":
    unittest.main()