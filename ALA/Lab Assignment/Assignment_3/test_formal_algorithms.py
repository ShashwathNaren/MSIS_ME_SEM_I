import unittest
import numpy as np
from formal_algorithms import *

L = 7
rng = np.random.default_rng(1)


# ---- independent row-vector reference (tokens are ROWS, as in NumPy/PyTorch) ----
def ref_attention_rows(x, Wq, Wk, Wv, bq, bk, bv, mask=None):
    Q, K, V = x @ Wq + bq, x @ Wk + bk, x @ Wv + bv
    S = Q @ K.T / np.sqrt(Wq.shape[1])
    if mask is not None:                       # mask[i, j] True = query i may attend key j
        S = np.where(mask, S, -np.inf)
    S = S - S.max(axis=-1, keepdims=True)
    A = np.exp(S)
    A /= A.sum(axis=-1, keepdims=True)
    return A @ V


def ref_mha_rows(x, p, mask=None):
    heads = [ref_attention_rows(x, p["W_q"][h].T, p["W_k"][h].T, p["W_v"][h].T,
                                p["b_q"][h], p["b_k"][h], p["b_v"][h], mask)
             for h in range(p["W_q"].shape[0])]
    return np.concatenate(heads, axis=-1) @ p["W_o"].T + p["b_o"]


class TestHyperparameters(unittest.TestCase):
    def test_section2_and_param_count(self):
        self.assertEqual(H * D_MID, D_E)
        attn = H * 3 * (D_ATTN * D_E + D_ATTN) + D_E * D_E + D_E
        mlp = 2 * D_E * D_MLP + D_MLP + D_E
        total = N_V * D_E + L_MAX * D_E + N_LAYERS * (attn + 4 * D_E + mlp) + 2 * D_E
        self.assertEqual(attn, 2_362_368)
        self.assertEqual(total, 124_439_808)
        self.assertEqual(total + N_V * D_E, 163_037_184)


class TestEmbedding(unittest.TestCase):
    def test_columns_and_onehot(self):
        W = init_embedding()
        v = rng.integers(0, N_V, L)
        e = token_embedding(v, W)
        self.assertEqual(e.shape, (D_E, L))
        onehot = np.zeros((N_V, L))
        onehot[v, np.arange(L)] = 1
        np.testing.assert_allclose(e, W @ onehot)
        self.assertEqual(token_embedding(int(v[0]), W).shape, (D_E,))


class TestLayerNorm(unittest.TestCase):
    def test_stats_and_affine(self):
        e = rng.standard_normal((D_E, L)) * 3 + 5
        g, b = init_layer_norm()
        out = layer_norm(e, g, b)
        np.testing.assert_allclose(out.mean(0), 0, atol=1e-6)
        v = e.var(0)
        np.testing.assert_allclose(out.var(0), v / (v + 1e-5), atol=1e-6)
        g2, b2 = rng.standard_normal(D_E), rng.standard_normal(D_E)
        np.testing.assert_allclose(layer_norm(e, g2, b2),
                                   layer_norm(e, g, b) * g2[:, None] + b2[:, None])
        np.testing.assert_allclose(layer_norm(e[:, 0], g, b), out[:, 0])


class TestAttention(unittest.TestCase):
    def setUp(self):
        self.p = init_mha(seed=3)
        self.X = rng.standard_normal((D_E, L))
        self.args = lambda h: (self.p["W_q"][h], self.p["b_q"][h], self.p["W_k"][h],
                               self.p["b_k"][h], self.p["W_v"][h], self.p["b_v"][h])

    def test_mask_shape_and_weights(self):
        M = causal_mask(L, L)
        self.assertTrue(np.array_equal(M, np.triu(np.ones((L, L), bool))))
        out, A = attention(self.X, self.X, *self.args(0), mask=M, return_weights=True)
        self.assertEqual(out.shape, (D_ATTN, L))
        np.testing.assert_allclose(A.sum(0), 1)
        self.assertTrue((A[~M] == 0).all())

    def test_causality(self):
        M = causal_mask(L, L)
        t = 3
        X2 = self.X.copy()
        X2[:, t + 1:] = rng.standard_normal((D_E, L - t - 1))
        a = mhattention(self.X, self.X, self.p, M)
        b = mhattention(X2, X2, self.p, M)
        np.testing.assert_allclose(a[:, :t + 1], b[:, :t + 1], atol=1e-12)
        self.assertFalse(np.allclose(a[:, t + 1:], b[:, t + 1:]))

    def test_cross_attention_shapes(self):
        Z = rng.standard_normal((D_E, 11))
        self.assertEqual(mhattention(self.X, Z, self.p, None).shape, (D_E, L))

    def test_matches_row_convention_reference(self):
        """Column version == row version (weights transposed), with and without mask."""
        for M_col in (None, causal_mask(L, L)):
            M_row = None if M_col is None else M_col.T      # Mask[t_z,t_x] -> [query,key]
            mine = mhattention(self.X, self.X, self.p, M_col)
            ref = ref_mha_rows(self.X.T, self.p, M_row)
            np.testing.assert_allclose(mine, ref.T, atol=1e-10)


if __name__ == "__main__":
    unittest.main(verbosity=2)