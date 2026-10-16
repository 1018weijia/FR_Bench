# Copyright 2024 Big Vision Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Gemma model adapted for the Pi system, derived from the big_vision project.
This is a Gemma implementation that supports a mixture-of-experts architecture,
used mainly for the PI0 diffusion policy.

We follow the einsum axis naming convention below:
  B: batch size (batch)
  T: query sequence length (query length)
  S: key/value sequence length (k/v length)
  N: number of query heads (num query heads)
  K: number of key/value heads (num k/v heads)
  G: number of query heads per key/value head (num query heads per k/v head)
  H: head dimension (head dim)
  D: model dimension, i.e. feature dimension (d_model/"features")
"""

# Import required modules
from collections.abc import Sequence  # For sequence type annotations
import dataclasses  # For the dataclass decorator
from typing import Literal, TypeAlias  # For literal types and type aliases

import einops  # For tensor operations and dimension rearrangement
import flax.linen as nn  # Flax neural network library
import jax  # JAX machine learning framework
import jax.numpy as jnp  # JAX NumPy interface

import openpi.models.lora as lora  # LoRA (low-rank adaptation) module
import openpi.shared.array_typing as at  # Array type annotation utilities
import openpi.training.sharding as sharding  # Model sharding utilities for training

# PaliGemma vocabulary size, fixed at 257,152
PALIGEMMA_VOCAB_SIZE = 257_152


@dataclasses.dataclass
class Config:
    """Gemma model configuration, defining the model's hyperparameters"""
    width: int  # Model width, i.e. hidden dimension
    depth: int  # Transformer depth (number of layers)
    mlp_dim: int  # MLP (multilayer perceptron) hidden dimension
    num_heads: int  # Number of query heads in the attention mechanism
    num_kv_heads: int  # Number of key/value heads in the attention mechanism
    head_dim: int  # Dimension of each attention head
    # LoRA config dict for low-rank adaptation at different layers
    lora_configs: dict[str, lora.LoRAConfig] = dataclasses.field(default_factory=dict)


# Supported Gemma model variants
Variant = Literal["dummy", "gemma_300m", "gemma_2b", "gemma_2b_lora"]


def get_config(variant: Variant) -> Config:
    """Return the configuration for the specified Gemma variant

    Args:
        variant: Model variant name

    Returns:
        Config: Corresponding model configuration object

    Raises:
        ValueError: Raised when an unknown variant is passed
    """
    if variant == "dummy":
        # Tiny dummy model for testing and debugging
        return Config(
            width=64,        # Very small model width
            depth=4,         # Very few layers
            mlp_dim=128,     # Very small MLP dimension
            num_heads=8,     # Number of attention heads
            num_kv_heads=1,  # Number of key/value heads
            head_dim=16,     # Smaller head dimension
        )
    if variant == "gemma_300m":
        # Small model with 311 million parameters, suitable for resource-constrained environments
        return Config(
            width=1024,      # Model width
            depth=18,        # Number of layers
            mlp_dim=4096,    # MLP hidden dimension
            num_heads=8,     # Number of attention heads
            num_kv_heads=1,  # Number of key/value heads, enabling multi-query attention
            head_dim=256,    # Head dimension
        )
    if variant == "gemma_2b":
        # Standard model with 2 billion parameters
        return Config(
            width=2048,      # Larger model width
            depth=18,        # Same number of layers as the 300M model
            mlp_dim=16_384,  # Larger MLP dimension
            num_heads=8,     # Number of attention heads
            num_kv_heads=1,  # Number of key/value heads
            head_dim=256,    # Head dimension
        )
    if variant == "gemma_2b_lora":
        # 2-billion-parameter variant with LoRA fine-tuning
        return Config(
            width=2048,      # Same architecture as the standard 2B model
            depth=18,
            mlp_dim=16_384,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                # LoRA config for the attention layers: rank 16, alpha 16.0
                "attn": lora.LoRAConfig(rank=16, alpha=16.0),
                # LoRA config for the feed-forward network
                "ffn": lora.LoRAConfig(rank=16, alpha=16.0)
            },
        )
    if variant == "gemma_300m_lora":
        # 311-million-parameter variant with LoRA fine-tuning
        return Config(
            width=1024,
            depth=18,
            mlp_dim=4096,
            num_heads=8,
            num_kv_heads=1,
            head_dim=256,
            lora_configs={
                # The 300M model uses larger LoRA parameters: rank 32, alpha 32.0
                "attn": lora.LoRAConfig(rank=32, alpha=32.0),
                "ffn": lora.LoRAConfig(rank=32, alpha=32.0)
            },
        )
    raise ValueError(f"Unknown model variant: {variant}")


@at.typecheck
class RMSNorm(nn.Module):
    """RMSNorm (root mean square normalization) module.
    Implements root mean square normalization, a simpler normalization method than LayerNorm
    that normalizes by the root mean square without mean centering.
    """

    @nn.compact
    def __call__(self, x):
        """Apply RMSNorm

        Args:
            x: Input tensor of shape [..., dim]

        Returns:
            Normalized tensor, preserving the original shape and dtype
        """
        dtype = x.dtype  # Save the original dtype, which may be half precision
        # Create a learnable scale parameter, initialized to 0
        scale = self.param("scale", nn.initializers.zeros_init(), (x.shape[-1]))
        # Compute variance in float32 for numerical stability
        var = jnp.mean(jnp.square(x.astype(jnp.float32)), axis=-1, keepdims=True)
        # Normalize in float32: x / sqrt(var + eps)
        normed_inputs = jnp.asarray(x * jnp.reciprocal(jnp.sqrt(var + 1e-06)))
        # Apply the learnable scale: (1 + scale) * normalized_x
        normed_inputs = normed_inputs * (1 + scale)
        # Cast back to the original dtype
        return normed_inputs.astype(dtype)


@at.typecheck
class Embedder(nn.Module):
    """Token embedding module that maps token IDs to vectors and vectors to logits"""

    vocab_size: int  # Vocabulary size
    embed_dim: int   # Embedding dimension

    def setup(self):
        """Initialize module parameters"""
        # Create the token embedding table of shape [vocab_size, embed_dim]
        # Initialized from a normal distribution
        self.input_embedding_table = self.param(
            "input_embedding",
            nn.initializers.normal(),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        """Encode token IDs into embedding vectors

        Args:
            x: Token ID tensor of shape [...]

        Returns:
            Embedding vectors of shape [..., embed_dim]
        """
        # Look up the corresponding vectors in the embedding table
        x = self.input_embedding_table[(x, )]
        # Multiply by the square root of the embedding dimension, a common embedding scaling trick
        x *= jnp.sqrt(self.embed_dim).astype(x.dtype)
        return x

    def decode(self, x):
        """Decode embedding vectors into logits (a distribution over the vocabulary)

        Args:
            x: Embedding vectors of shape [..., embed_dim]

        Returns:
            Logits of shape [..., vocab_size]
        """
        # Multiply by the transpose of the embedding table to score similarity with every token
        return jnp.dot(x, self.input_embedding_table.T)


@at.typecheck
class Attention(nn.Module):
    """Multi-head attention module.
    Supports a mixture-of-experts architecture: each expert can have its own config
    while sharing the same attention parameters.
    """

    configs: Sequence[Config]  # Sequence of expert configs, one per expert

    @nn.compact
    def __call__(self, xs, positions, attn_mask, kv_cache):
        """Run multi-head attention

        Args:
            xs: List of input sequences for multiple experts; each element has shape [batch, seq_len, dim] or is None
            positions: Positional encoding of shape [batch, seq_len]
            attn_mask: Attention mask of shape [batch, seq_len, seq_len]
            kv_cache: KV cache used to speed up inference

        Returns:
            tuple: (list of attention outputs, updated KV cache)
        """
        # Verify that all experts share the same attention parameters, which multi-head attention requires
        assert all(config.head_dim == self.configs[0].head_dim for config in self.configs)
        assert all(config.num_heads == self.configs[0].num_heads for config in self.configs)
        assert all(config.num_kv_heads == self.configs[0].num_kv_heads for config in self.configs)

        # Get the input dtype (may be a half-precision float)
        dtype = next(x.dtype for x in xs if x is not None)

        qkvs = []  # Store Q, K, and V tensors for each expert
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is None:
                continue  # Skip experts with no input

            if config.num_kv_heads == config.num_heads:
                # When the number of query heads equals the number of key/value heads, compute Q, K, and V in one einsum
                qkv_einsum = lora.Einsum(
                    shape=(3, config.num_heads, config.width, config.head_dim),  # [Q,K,V, heads, dim, head_dim]
                    name=_name("qkv_einsum", i),  # Unique parameter name for each expert
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),  # Apply the LoRA config
                )
                # Einsum: [batch, seq, dim] x [3, heads, dim, head_dim] -> [3, batch, seq, heads, head_dim]
                qkvs.append(qkv_einsum("BSD,3KDH->3BSKH", x))
            else:
                # When query heads differ from key/value heads (multi-query attention), compute Q and K/V separately
                q_einsum = lora.Einsum(
                    shape=(config.num_heads, config.width, config.head_dim),  # [heads, dim, head_dim]
                    name=_name("q_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, )),
                    lora_config=config.lora_configs.get("attn"),
                )
                # Compute the query tensor: [batch, seq, dim] x [heads, dim, head_dim] -> [batch, seq, heads, head_dim]
                q = q_einsum("BTD,NDH->BTNH", x)

                kv_einsum = lora.Einsum(
                    shape=(2, config.num_kv_heads, config.width, config.head_dim),  # [K,V, kv_heads, dim, head_dim]
                    name=_name("kv_einsum", i),
                    init_fn=nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, 1)),
                    lora_config=config.lora_configs.get("attn"),
                )
                # Compute the key/value tensors: [batch, seq, dim] x [2, kv_heads, dim, head_dim] -> [2, batch, seq, kv_heads, head_dim]
                k, v = kv_einsum("BSD,2KDH->2BSKH", x)
                qkvs.append((q, k, v))

        # Concatenate Q, K, and V from all experts along the sequence dimension
        q, k, v = (jnp.concatenate(y, axis=1) for y in zip(*qkvs, strict=True))

        # Apply rotary position embeddings (RoPE) to the query tensor
        q = _apply_rope(q, positions=positions)
        # Scale the query tensor to avoid vanishing gradients
        q *= self.configs[0].head_dim**-0.5

        # Apply rotary position embeddings (RoPE) to the key tensor
        k = _apply_rope(k, positions=positions)

        # Ensure dtypes match (keep half precision for efficiency)
        assert q.dtype == k.dtype == v.dtype == dtype

        # If a KV cache exists, concatenate the newly computed K and V with the cache (for autoregressive inference)
        if kv_cache is not None:
            cache_k, cache_v = kv_cache
            k = jnp.concatenate([cache_k, k], axis=1)  # Concatenate along the sequence dimension
            v = jnp.concatenate([cache_v, v], axis=1)

        # Rearrange the query tensor to support multi-query attention:
        # from [batch, seq, (kv_heads * groups), head_dim] -> [batch, seq, kv_heads, groups, head_dim]
        q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=self.configs[0].num_kv_heads)

        # Compute attention scores: dot product of queries and keys
        # einsum: [batch, seq, kv_heads, groups, head_dim] x [batch, seq_len, kv_heads, head_dim]
        # -> [batch, kv_heads, groups, seq, seq_len]
        logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)

        # Check that the attention mask has the expected shape
        if attn_mask.shape != (q.shape[0], 1, q.shape[1], k.shape[1]):
            raise ValueError(
                f"Attention mask shape {attn_mask.shape} does not match query tensor {q.shape} and key tensor {k.shape}")

        # Use a very large negative number to mask out disallowed attention positions
        # This value comes from the original Gemma implementation and keeps the numerics stable
        big_neg = -2.3819763e38  # See gemma/modules.py
        # Apply the attention mask: keep allowed positions, set disallowed ones to the large negative value
        masked_logits = jnp.where(attn_mask[:, :, None, :, :], logits, big_neg)

        # Compute attention weights (a probability distribution) and cast back to the original dtype
        probs = jax.nn.softmax(masked_logits, axis=-1).astype(dtype)

        # Weighted sum of the value tensor using the attention weights
        # einsum: [batch, kv_heads, groups, seq, seq_len] x [batch, seq_len, kv_heads, head_dim]
        # -> [batch, seq, kv_heads, groups, head_dim]
        encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probs, v)
        # Rearrange back to the original shape: [batch, seq, kv_heads, groups, head_dim] -> [batch, seq, (kv_heads * groups), head_dim]
        encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")

        # Split the attention output back across experts
        out = []
        start = 0  # Track the start index of the current expert in the concatenated sequence
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # End index of the current expert in the concatenated sequence
                end = start + x.shape[1]

                # Output projection that maps the attention output back to the original dimension
                out_einsum = lora.Einsum(
                    shape=(config.num_heads, config.head_dim, config.width),  # [heads, head_dim, width]
                    name=_name("attn_vec_einsum", i),  # Unique parameter name for each expert
                    init_fn=nn.initializers.lecun_normal(in_axis=(-3, -2), out_axis=-1),
                    lora_config=config.lora_configs.get("attn"),  # Apply the LoRA config
                )
                # Projection: [batch, seq, heads, head_dim] x [heads, head_dim, width] -> [batch, seq, width]
                out.append(out_einsum("BTNH,NHD->BTD", encoded[:, start:end]))
                start = end  # Update the start index for the next expert
            else:
                out.append(None)  # Return None for experts with no input

        return out, (k, v)  # Return each expert's output and the updated KV cache


@at.typecheck
class FeedForward(nn.Module):
    """Feed-forward network using a gating mechanism (similar to LLaMA's SwiGLU)"""

    features: int      # Input and output feature dimension
    hidden_dim: int   # Hidden dimension

    @nn.compact
    def __call__(self, x):
        """Run the feed-forward computation

        Args:
            x: Input tensor of shape [..., features]

        Returns:
            Output tensor of shape [..., features]
        """
        dtype = x.dtype  # Save the original dtype

        # Create the gating weight matrix: [2, features, hidden_dim]
        # The first weight is for the gate and the second is for the linear transform
        w_gating = self.param(
            "gating_einsum",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1, batch_axis=(0, )),
            (2, self.features, self.hidden_dim),
        ).astype(dtype)

        # Gating path: x @ W_gate -> [..., hidden_dim]
        ff_gate = jnp.dot(x, w_gating[0])
        # Apply GELU as the gate value
        gate_value = nn.gelu(ff_gate)

        # Linear path: x @ W_linear -> [..., hidden_dim]
        ff1 = jnp.dot(x, w_gating[1])
        # Gating: gate_value * linear_value
        activations = gate_value * ff1

        # Output projection weights: [hidden_dim, features]
        w_linear = self.param(
            "linear",
            nn.initializers.lecun_normal(in_axis=-2, out_axis=-1),
            (self.hidden_dim, self.features),
        ).astype(dtype)

        # Final linear transform: activations @ W_out -> [..., features]
        outputs = jnp.dot(activations, w_linear)
        assert outputs.dtype == dtype  # Ensure the dtype is unchanged
        return outputs


@at.typecheck
class Block(nn.Module):
    """Transformer block with attention and a feed-forward network.
    Implements the standard pre-norm Transformer architecture and supports mixture of experts.
    """

    configs: Sequence[Config]  # Sequence of expert configs

    dropout: float = 0.0  # Dropout probability
    dropout_bdims: tuple[int, ...] = ()  # Batch dimensions for dropout

    @nn.compact
    def __call__(self, xs, kv_cache, positions, attn_mask, decode, deterministic=True):
        """Forward pass of the Transformer block

        Args:
            xs: List of multi-expert input sequences
            kv_cache: KV cache
            positions: Positional encoding
            attn_mask: Attention mask
            decode: Whether this is decode mode
            deterministic: Whether to use deterministic mode (affects dropout)

        Returns:
            tuple: (list of output sequences, updated KV cache)
        """
        # Apply activation sharding constraints for distributed training
        xs = sharding.activation_sharding_constraint(xs)
        # Create a dropout layer, or an identity function when the dropout probability is 0
        drop = nn.Dropout(self.dropout, self.dropout_bdims) if self.dropout else lambda x, _: x

        # Create the multi-head attention layer
        attn = Attention(configs=self.configs, name="attn")

        # ===== Attention sub-block (pre-norm + residual connection) =====
        # Pre-normalize each expert's input
        pre_attn = []
        for i, x in enumerate(xs):
            if x is not None:
                # Apply RMSNorm; each expert has its own normalization parameters
                x = RMSNorm(name=_name("pre_attention_norm", i))(x)
            pre_attn.append(x)

        # Apply the sharding constraint
        pre_attn = sharding.activation_sharding_constraint(pre_attn)
        # Run multi-head attention
        post_attn, kv_cache = attn(pre_attn, positions, attn_mask, kv_cache)
        # Apply dropout (training only)
        post_attn = jax.tree.map(lambda x: drop(x, deterministic), post_attn)
        post_attn = sharding.activation_sharding_constraint(post_attn)
        # Residual connection: input + attention output
        xs = jax.tree.map(lambda x, y: x + y, xs, post_attn)
        xs = sharding.activation_sharding_constraint(xs)

        # ===== Feed-forward sub-block (pre-norm + residual connection) =====
        out = []
        for i, (x, config) in enumerate(zip(xs, self.configs, strict=True)):
            if x is not None:
                # Pre-normalization for the feed-forward network
                x = RMSNorm(name=_name("pre_ffw_norm", i))(x)
                # Apply the feed-forward network (with LoRA support)
                x = lora.FeedForward(
                    features=config.width,
                    hidden_dim=config.mlp_dim,
                    name=_name("mlp", i),
                    lora_config=config.lora_configs.get("ffn"),
                )(x)
            out.append(x)

        # Apply the sharding constraint and dropout
        out = sharding.activation_sharding_constraint(out)
        out = jax.tree.map(lambda x: drop(x, deterministic), out)
        # Residual connection: attention output + feed-forward output
        xs = jax.tree.map(lambda x, y: x + y, xs, out)
        xs = sharding.activation_sharding_constraint(xs)

        return xs, kv_cache


KVCache: TypeAlias = tuple[at.Float[at.Array, "l b _t _k _h"], at.Float[at.Array, "l b _t _v _h"]]


@at.typecheck
class Module(nn.Module):
    """Transformer model that supports mixture of experts.
    This is the core PI0 model. It can use different expert weights for different tokens,
    mainly mixing a PaliGemma (vision-language) expert with an action expert.
    """

    configs: Sequence[Config]  # List of expert configs, one per expert
    embed_dtype: str           # Embedding dtype (e.g. "bfloat16")

    dropout: float = 0.0  # Global dropout probability
    dropout_bdims: tuple[int, ...] = ()  # Batch dimensions for dropout

    def setup(self):
        """Initialize model components"""
        # All experts must have the same number of layers
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        # Create the token embedder (only the first expert needs one, because all experts share the vocabulary)
        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # Use the first expert's width as the embedding dimension
            name="embedder",
        )

        # Create a rematerialized Block class to save memory
        # remat: recompute the forward pass, using less memory at the cost of extra compute
        block_cls = nn.remat(
            Block,
            prevent_cse=False,  # Do not prevent common subexpression elimination
            static_argnums=(5, ),  # Mark static arguments: 0=self, 5=deterministic
            policy=jax.checkpoint_policies.nothing_saveable,  # Checkpoint policy: do not save intermediates
        )

        # Stack Transformer layers with nn.scan
        # scan calls block_cls in a loop, passing updated parameters and state each time
        self.layers = nn.scan(
            block_cls,
            variable_axes={"params": 0},  # Stack parameters along axis 0 (each layer has its own parameters)
            split_rngs={  # Split RNGs per layer
                "params": True,   # Parameter-initialization RNG
                "dropout": True   # dropout RNG
            },
            in_axes=(0, nn.broadcast, nn.broadcast, nn.broadcast),  # Input axis mapping
            # 0=kv_cache (different per layer), 1=positions (broadcast), 2=mask (broadcast), 3=decode (broadcast)
            length=self.configs[0].depth,  # Number of layers
        )(
            configs=self.configs,
            dropout=self.dropout,
            dropout_bdims=self.dropout_bdims,
        )
        # Final normalization layer for each expert
        self.final_norms = [RMSNorm(name=_name("final_norm", i)) for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        """Convert token IDs to embedding vectors

        Args:
            tokens: Token ID tensor of shape [batch, seq_len]

        Returns:
            Embedding vectors of shape [batch, seq_len, embed_dim]
        """
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # List of embedding vectors per expert; None if that expert is not run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],      # Positional encoding
        mask: at.Bool[at.Array, "b t s"],        # Attention mask
        *,
        kv_cache: KVCache | None = None,         # KV cache
        deterministic: bool = True,              # Whether this is deterministic mode
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        """Forward pass of the Transformer

        Args:
            embedded: List of embedding vectors for each expert
            positions: Positional encoding
            mask: Attention mask
            kv_cache: KV cache (for autoregressive inference)
            deterministic: Whether to use deterministic mode

        Returns:
            tuple: (list of outputs per expert, updated KV cache)
        """
        # Ensure every embedding uses the specified dtype
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype), embedded)
        # Expand the attention mask: [batch, seq_len, seq_len] -> [batch, 1, seq_len, seq_len]
        mask = jnp.asarray(mask)[:, None, :, :]

        # Pass through all Transformer layers
        embedded, kv_cache = self.layers(embedded, kv_cache, positions, mask, deterministic)

        # Check that dtypes are consistent
        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in embedded if e is not None)

        # Apply each expert's final normalization layer
        return [f(e) if e is not None else e for f, e in zip(self.final_norms, embedded, strict=True)], kv_cache

    def init(self):
        """Convenience method that initializes all model parameters.
        Because of how Flax Linen works, parameters are created by explicitly running one forward pass.
        """
        # Initialize the embedder
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        # Initialize the Transformer layers
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],  # Dummy input for each expert
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32), # Dummy positional encoding
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool), # Dummy attention mask
        )


def _apply_rope(x, *, positions, max_wavelength=10_000):
    """Apply rotary position embeddings (RoPE) to the input tensor

    RoPE encodes position by rotating vectors. It preserves relative position
    and can extrapolate beyond the training length.

    Args:
        x: Input tensor of shape [batch, seq_len, num_heads, head_dim]
        positions: Position indices of shape [batch, seq_len]
        max_wavelength: Maximum wavelength, controlling the frequency range of the position encoding

    Returns:
        Tensor after applying RoPE, with the same shape as the input
    """
    # Frequency exponents, evenly spaced from low to high frequency
    freq_exponents = (2.0 / x.shape[-1]) * jnp.arange(x.shape[-1] // 2, dtype=jnp.float32)
    # Timescales, growing exponentially from low to high frequency
    timescale = max_wavelength**freq_exponents
    # Rotation angles: position / timescale
    radians = positions[..., None] / timescale[None, None, :]
    radians = radians[..., None, :]  # Add the head dimension
    assert radians.dtype == jnp.float32  # Keep the computation in full precision

    # Sine and cosine
    sin, cos = jnp.sin(radians), jnp.cos(radians)
    # Split the input tensor into two halves along the last dimension
    x1, x2 = jnp.split(x, 2, axis=-1)

    # Apply the 2D rotation:
    # [x1, x2] @ [cos, -sin; sin, cos] = [x1*cos - x2*sin, x2*cos + x1*sin]
    res = jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)
    assert res.dtype == jnp.float32

    # Cast back to the original dtype
    # The original bigvision implementation upcasts RoPE to float32, then immediately downcasts
    # back to the cache dtype in inference mode. Following the original DeepMind implementation
    # and the widely used transformers implementation, always downcasting back to bfloat16 here is fine.
    return res.astype(x.dtype)


def _name(name, i):
    """Generate a parameter name for an expert layer

    Layers are named this way because:
    1. The first expert's weights have no suffix (for example "attn"), so they load seamlessly from an existing PaliGemma checkpoint
    2. Later experts have a suffix (for example "attn_1"), and their weights are initialized from zero
    3. In practice we use only two experts: the PaliGemma expert and the action expert

    Args:
        name: Base layer name
        i: Expert index

    Returns:
        str: Generated parameter name
    """
    if i == 0:
        return name  # The first expert keeps the original name so pretrained weights can be loaded
    return f"{name}_{i}"  # Later experts get a numeric suffix
