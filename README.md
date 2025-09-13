# Geodesic Language Models: From Pairwise Interactions to Learned Paths

This repository contains an experimental language model architecture that explores a geometric interpretation of language. It begins with the **Manifold Modulation Model (M3)**, which treats tokens as points on a high-dimensional surface that influence each other through forces. It then evolves this concept into the **Geodesic State Model**, an efficient, path-based architecture that models context as a trajectory across this surface.

This work diverges from the standard all-to-all self-attention of Transformers, proposing a more structured, sequential, and geometrically-grounded approach to language modeling.

## Part 1: The Foundation - The Manifold Modulation Model (M3)

The first architecture explored is the Manifold Modulation Model, or M3.

### Core Concept: Language as a Deformable Surface

The M3 model is built on a simple but powerful geometric intuition:
1.  **Tokens are points:** Every token in a sequence is represented as a point (a vector) in a high-dimensional geometric space, or "manifold."
2.  **Context deforms the manifold:** The model learns how these points exert influence on each other, creating "displacements." The final, context-aware representation of a token is its new position after being pushed and pulled by every other token in the sequence.

### The M3 Mechanism: A System of Forces

To find its final position, each point `p_i` is displaced by a force aggregated from every other point `p_j`. This force is carefully calculated based on three factors:

1.  **Direction:** The force from `p_j` on `p_i` always pushes or pulls `p_i` along the line connecting them. The direction is simply the normalized vector `(p_i - p_j)`.
2.  **Relevance (Attention):** A standard attention mechanism calculates a weight `attn_ij`, determining how relevant point `p_j` is to `p_i`. This acts as a content-based filter.
3.  **Influence:** The magnitude of the force is determined by a combination of factors:
    *   **Distance:** Closer points exert a stronger influence. This is modeled with a Radial Basis Function (RBF) kernel, creating a localized effect.
    *   **Amplitude:** Each point `p_j` learns a scalar "amplitude," representing its inherent power to influence other points.

The total displacement for point `p_i` is the weighted sum of all these pairwise forces.

> **Analogy:** Imagine a set of charged particles on a rubber sheet. Each particle's final position is determined by the push and pull from every other particle, based on their charge, distance, and relevance.

### Limitations of M3

While conceptually elegant, this architecture has a significant drawback: **quadratic complexity**. To compute the force between every pair of points, the model must create massive intermediate tensors of shape `[Batch, SeqLen, SeqLen, Dimension]`. This consumes a large amount of VRAM and scales poorly with longer sequences.

## Part 2: The Evolution - The Geodesic State Model

The limitations of M3 led to a conceptual breakthrough: **What if context isn't a static, all-pairs interaction, but a dynamic path drawn across the manifold?**

This insight transforms the model from one of pairwise forces to one of **navigation and momentum**. The Geodesic State Model was born from this idea.

### Core Concept: Attention as Navigation

Instead of allowing every token to "teleport" and interact with every other token, the Geodesic State Model processes the sequence step-by-step, as if walking a path. The context for the current token is the memory of the path traveled so far. The model's task is to learn the "rules of the road": how to steer the path based on the next token it encounters.

This is the core idea of State-Space Models (SSMs), framed within the geometric M3 intuition.

### Architectural Heart: The `GeodesicStateBlock`

The central component is the `GeodesicStateBlock`, a recurrent mechanism that efficiently builds the path token by token. For each point `p_t` in the sequence, it does the following:

1.  **Maintains a Path State (`hidden_state`):** A single vector represents the cumulative history of the sequence—the "memory" of the path.
2.  **Uses Gating to Steer the Path:** A GRU-like gating mechanism intelligently updates the path state based on the current token `p_t`.
    *   **Reset Gate:** Decides how much of the old path is relevant for the new direction. This allows the model to learn to start new "sub-paths" (e.g., at the beginning of a new sentence).
    *   **Update Gate:** Blends the memory of the old path with the new direction proposed by `p_t`. This is the core steering wheel, learning when to maintain momentum and when to change course.
3.  **Calculates Displacement:** The final, context-aware position of a token is its resulting state vector. The "displacement" is the difference between this new state and the token's original embedding. This displacement is added as a residual, deforming the original point to its new, context-aware position on the learned path.

```python
# The GeodesicStateBlock learns a path on the manifold using a recurrent state.
class GeodesicStateBlock(nn.Module):
    def forward(self, points: torch.Tensor) -> torch.Tensor:
        B, S, D = points.shape
        
        # The "memory" of the path, initialized at the origin.
        hidden_state = torch.zeros(B, D, device=points.device)
        
        outputs: list[torch.Tensor] = []
        for t in range(S):
            p_t = points[:, t, :] # The current token's position
            
            # Use GRU-style gates to decide how p_t should alter the path.
            # This is where the model "learns the rules of the road."
            reset_gate, update_gate = ...
            candidate_state = ... # Propose a new direction
            
            # Steer the path by blending the old memory with the new direction.
            hidden_state = (1.0 - update_gate) * hidden_state + update_gate * candidate_state
            
            outputs.append(hidden_state)
            
        final_states = torch.stack(outputs, dim=1)
        
        # Displace the original points to their final positions on the path.
        displacement = final_states - points
        deformed_points = points + displacement
        
        return self.norm(deformed_points)
```

### Key Properties of the Geodesic State Model

*   **Linear Time Complexity:** Computation scales as `O(SequenceLength)`, allowing for much longer context windows than the quadratic M3 model.
*   **Constant Memory Generation:** During inference, only the last `hidden_state` needs to be kept in memory, making token generation extremely fast and memory-efficient.
*   **Causal by Design:** The sequential, state-based processing naturally ensures causality without any explicit masking.
*   **Performance Optimization:** The Python `for` loop, while intuitive, is slow. This implementation uses `torch.jit.script` to compile the recurrent block into a single, highly optimized GPU kernel, achieving high performance without changing the underlying architecture.

