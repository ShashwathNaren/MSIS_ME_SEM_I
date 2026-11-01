import argparse
import math
import random
import sys
import unicodedata
import numpy as np
from typing import Self

MODEL_NAME = "glove-wiki-gigaword-50"
TAG_COUNT = 20
TOP_K = 8

SOME_TAGS = [
    "research", "innovation", "education", "university", "students",
    "faculty", "campus", "engineering", "medicine", "technology",
    "curriculum", "collaboration", "publication", "laboratory",
    "scholarship", "mentorship", "internship", "entrepreneurship",
    "accreditation", "alumni"
]

STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to",
    "in", "on", "for", "is", "are", "was", "were",
    "with", "at", "by", "from",
    "it", "its", "this", "that", "these", "those",
    "our", "we", "us", "you", "your", "they", "their", "them",
    "he", "she", "his", "her", "my", "me", "who", "whom", "which", "what",
    "be", "been", "being", "have", "has", "had", "will", "would", "can", "could",
    "as", "but", "not", "so", "if", "about", "into", "also", "more", "all",
    "there", "than", "then", "just", "very", "one", "out", "up", "how", "when",
    "where", "while", "over", "across", "through", "each", "such", "both"
}

class Vec:

    def __init__(self, src=None) -> None:
        if src is None:
            self.elements = []
        else:
            self.elements = list(src)

    def __add__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError("Type error - vectors must be of same dimensions")
        return Vec([round(x + y, 5) for x, y in zip(self.elements, t.elements)])

    def __radd__(self, other):
        if other == 0:
            return self
        return self.__add__(other)

    def __iadd__(self, other):
        if not isinstance(other, Vec):
            raise TypeError(f"Expected Vec: {type(other)}")
        if len(self.elements) != len(other):
            raise TypeError("Type error - vectors must be of same dimensions")
        for i, y in enumerate(other.elements):
            self.elements[i] = round(self.elements[i] + y, 5)
        return self

    def __sub__(self, t: Self) -> Self:
        if not isinstance(t, Vec):
            raise TypeError(f"Expected Vec: {type(t)}")
        if len(self.elements) != len(t):
            raise TypeError("Type error - vectors must be of same dimensions")
        return Vec([round(x - y, 5) for x, y in zip(self.elements, t.elements)])

    def __neg__(self) -> Self:
        return Vec([-x for x in self.elements])

    def __rmul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        return Vec([round(x * scalar, 5) for x in self.elements])

    def __imul__(self, scalar: int | float) -> Self:
        if not isinstance(scalar, (int, float)):
            raise TypeError(f"Vector multiplication with invalid type: {type(scalar)}")
        for i, val in enumerate(self.elements):
            self.elements[i] = round(val * scalar, 5)
        return self

    def __repr__(self) -> str:
        return repr(self.elements)

    def __len__(self) -> int:
        return len(self.elements)

    @staticmethod
    def zeros(n: int) -> Self:
        return Vec([0.0] * n)

    @staticmethod
    def ones(n: int) -> Self:
        return Vec([1.0] * n)

    @staticmethod
    def uniform(n: int) -> Self:
        return Vec([random.random() for _ in range(n)])

    def dot(self, other: Self) -> float:
        if len(self.elements) != len(other):
            raise ValueError("Vectors must have the same length.")
        return sum(x * y for x, y in zip(self.elements, other.elements))

    def norm(self) -> float:
        return math.sqrt(sum(x * x for x in self.elements))

    def cosine_similarity(self, other: Self) -> float:
        norm_self = self.norm()
        norm_other = other.norm()
        if norm_self == 0 or norm_other == 0:
            raise ValueError("Cosine similarity is undefined for a zero vector.")
        return self.dot(other) / (norm_self * norm_other)

def load_model():
    import gensim.downloader as api
    return api.load(MODEL_NAME)

def read_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()

def remove_punctuation(text):
    return "".join(ch for ch in text if unicodedata.category(ch)[0] not in ("P", "S"))

def preprocess_text(text):
    raw_count = len(text.split())
    lowered = text.lower()
    cleaned = remove_punctuation(lowered)
    tokens = cleaned.split()
    tokens = [token for token in tokens if token not in STOPWORDS]
    tokens = [token for token in tokens if len(token) > 2]
    return raw_count, tokens

def build_tag_matrix(model, tags):
    if len(tags) != TAG_COUNT:
        raise ValueError(f"Exactly {TAG_COUNT} tags are required, got {len(tags)}.")
    if len(set(tags)) != len(tags):
        raise ValueError("Tags must be distinct.")

    tag_names = []
    rows = []
    for tag in tags:
        words = tag.split()
        if not words:
            raise ValueError("Empty tag.")
        for word in words:
            if word not in model.key_to_index:
                raise ValueError(f"Tag not in vocabulary: {word}")
        vectors = np.array([model[word] for word in words], dtype=np.float64)
        rows.append(vectors.mean(axis=0))
        tag_names.append(tag)

    T = np.vstack(rows)
    return tag_names, T

def build_text_matrix(model, tokens):
    in_vocab_tokens = []
    oov_tokens = []
    rows = []
    for token in tokens:
        if token in model.key_to_index:
            in_vocab_tokens.append(token)
            rows.append(model[token])
        else:
            oov_tokens.append(token)

    if rows:
        W = np.array(rows, dtype=np.float64)
    else:
        W = np.empty((0, model.vector_size), dtype=np.float64)
    return in_vocab_tokens, oov_tokens, W

def similarity_matrix_vector_class(W, T):
    n_words = W.shape[0]
    n_tags = T.shape[0]
    S = np.zeros((n_words, n_tags))
    tag_vectors = [Vec(T[j].tolist()) for j in range(n_tags)]
    for i in range(n_words):
        word_vector = Vec(W[i].tolist())
        for j in range(n_tags):
            S[i, j] = word_vector.cosine_similarity(tag_vectors[j])
    return S

def similarity_matrix_numpy(W, T):
    W = np.asarray(W, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)

    W_norm = np.linalg.norm(W, axis=1, keepdims=True)
    T_norm = np.linalg.norm(T, axis=1, keepdims=True)

    if np.any(W_norm == 0):
        raise ValueError("W contains a row with zero norm.")
    if np.any(T_norm == 0):
        raise ValueError("T contains a row with zero norm.")

    W_hat = W / W_norm
    T_hat = T / T_norm

    return W_hat @ T_hat.T

def rank_tags(tag_names, S, in_vocab_tokens):
    S = np.asarray(S)
    if S.shape[0] == 0:
        raise ValueError("Similarity matrix has no rows.")
    best_rows = np.argmax(S, axis=0)
    scores = S[best_rows, np.arange(S.shape[1])]
    order = np.argsort(-scores, kind="stable")
    return [(tag_names[j], float(scores[j]), in_vocab_tokens[best_rows[j]]) for j in order]

def match_type(tag, word):
    return "identical" if tag == word else "different"

def print_token_statistics(raw_count, tokens, in_vocab_tokens, oov_tokens):
    print("Tokens before preprocessing :", raw_count)
    print("Tokens after preprocessing  :", len(tokens))
    print("In-vocabulary tokens        :", len(in_vocab_tokens))
    print("Out-of-vocabulary tokens    :", len(oov_tokens))
    print("Distinct OOV tokens         :", sorted(set(oov_tokens)))

def print_ranking(ranking, top_k=TOP_K):
    print(f"{'Rank':<6}{'Tag':<20}{'Score':<10}{'Best-matching text word':<26}{'Match'}")
    print("-" * 70)
    for rank, (tag, score, word) in enumerate(ranking[:top_k], start=1):
        print(f"{rank:<6}{tag:<20}{score:<10.4f}{word:<26}{match_type(tag, word)}")

def print_word_scores(in_vocab_tokens, tag_names, S):
    S = np.asarray(S)
    seen = set()
    rows = []
    for i, word in enumerate(in_vocab_tokens):
        if word in seen:
            continue
        seen.add(word)
        j = int(np.argmax(S[i]))
        rows.append((word, tag_names[j], float(S[i, j])))
    rows.sort(key=lambda r: -r[2])
    print(f"{'Word':<20}{'Closest tag':<20}{'Score'}")
    print("-" * 50)
    for word, tag, score in rows:
        print(f"{word:<20}{tag:<20}{score:.4f}")

def best_different_words(tag_names, S, in_vocab_tokens):
    S = np.asarray(S)
    results = []
    for j, tag in enumerate(tag_names):
        mask = np.array([word != tag for word in in_vocab_tokens])
        if not mask.any():
            results.append((tag, None, None))
            continue
        column = np.where(mask, S[:, j], -np.inf)
        i = int(np.argmax(column))
        results.append((tag, float(S[i, j]), in_vocab_tokens[i]))
    results.sort(key=lambda r: (r[1] is None, -(r[1] or 0)))
    return results

def print_different_matches(results):
    print(f"{'Tag':<20}{'Score':<10}{'Best different word'}")
    print("-" * 50)
    for tag, score, word in results:
        if word is None:
            print(f"{tag:<20}{'-':<10}{'-'}")
        else:
            print(f"{tag:<20}{score:<10.4f}{word}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", default="manipal_text.txt")
    args = parser.parse_args()

    model = load_model()
    tag_names, T = build_tag_matrix(model, SOME_TAGS)

    text = read_text(args.text)
    raw_count, tokens = preprocess_text(text)
    if raw_count > 400:
        print(f"Warning: input has {raw_count} words, the limit is 400.")

    in_vocab_tokens, oov_tokens, W = build_text_matrix(model, tokens)

    print_token_statistics(raw_count, tokens, in_vocab_tokens, oov_tokens)
    print(T.shape)
    print(W.shape)

    if W.shape[0] == 0:
        sys.exit("Error: the input contains no usable in-vocabulary words.")

    S_vector = similarity_matrix_vector_class(W, T)
    S_numpy = similarity_matrix_numpy(W, T)

    print("S shape :", S_numpy.shape)
    print("np.allclose(S_vector, S_numpy) :", np.allclose(S_vector, S_numpy, atol=1e-6))
    print("Max absolute difference :", np.max(np.abs(S_vector - S_numpy)))

    ranking = rank_tags(tag_names, S_numpy, in_vocab_tokens)
    print_ranking(ranking, top_k=len(ranking))

    print()
    print_different_matches(best_different_words(tag_names, S_numpy, in_vocab_tokens))

    print()
    print_word_scores(in_vocab_tokens, tag_names, S_numpy)

if __name__ == "__main__":
    main()