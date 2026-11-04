import numpy as np

L_MAX = 1024
D_E = 768
N_LAYERS = 12
H = 12
D_MLP = 3072
D_ATTN = 64
D_MID = 64
N_V = 50257


def causal_mask(l_z, l_x):
    rows = np.arange(l_z).reshape(-1, 1)
    cols = np.arange(l_x).reshape(1, -1)
    return rows <= cols


def softmax_cols(scores):
    shifted = scores - np.max(scores, axis=0, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=0, keepdims=True)


def token_embedding(v, W_e):
    return W_e[:, v]


def layer_norm(e, gamma, beta, eps=1e-5):
    mean = np.mean(e, axis=0, keepdims=True)
    var = np.mean((e - mean) ** 2, axis=0, keepdims=True)
    shape = (-1,) + (1,) * (e.ndim - 1)
    normalized = (e - mean) / np.sqrt(var + eps)
    return normalized * gamma.reshape(shape) + beta.reshape(shape)


def attention(X, Z, W_q, b_q, W_k, b_k, W_v, b_v, mask=None, return_weights=False):
    d_attn = W_q.shape[0]
    query = np.dot(W_q, X) + b_q.reshape(-1, 1)
    key = np.dot(W_k, Z) + b_k.reshape(-1, 1)
    value = np.dot(W_v, Z) + b_v.reshape(-1, 1)
    scores = np.dot(key.T, query) / np.sqrt(d_attn)
    if mask is not None:
        scores = np.where(mask, scores, -np.inf)
    weights = softmax_cols(scores)
    output = np.dot(value, weights)
    if return_weights:
        return output, weights
    return output


def mhattention(X, Z, params, mask=None):
    num_heads = params["W_q"].shape[0]
    head_outputs = []
    for h in range(num_heads):
        out = attention(X, Z,
                        params["W_q"][h], params["b_q"][h],
                        params["W_k"][h], params["b_k"][h],
                        params["W_v"][h], params["b_v"][h], mask)
        head_outputs.append(out)
    Y = np.concatenate(head_outputs, axis=0)
    return np.dot(params["W_o"], Y) + params["b_o"].reshape(-1, 1)


def init_mha(d_x=D_E, d_z=D_E, d_attn=D_ATTN, d_mid=D_MID, d_out=D_E, num_heads=H, seed=0):
    rng = np.random.default_rng(seed)

    def rand(*shape):
        return rng.standard_normal(shape) * 0.02

    return {
        "W_q": rand(num_heads, d_attn, d_x),
        "b_q": np.zeros((num_heads, d_attn)),
        "W_k": rand(num_heads, d_attn, d_z),
        "b_k": np.zeros((num_heads, d_attn)),
        "W_v": rand(num_heads, d_mid, d_z),
        "b_v": np.zeros((num_heads, d_mid)),
        "W_o": rand(d_out, num_heads * d_mid),
        "b_o": np.zeros(d_out),
    }


def init_embedding(n_v=N_V, d_e=D_E, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((d_e, n_v)) * 0.02


def init_layer_norm(d_e=D_E):
    return np.ones(d_e), np.zeros(d_e)