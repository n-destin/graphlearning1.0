Given a network G = (V, E) (which can be directed/undirected or weighted/unweighted), our goal is to learn a mapping function:

   f -> R^d

that assigns each node a d-dimensional feature representation. Equivalently, f is a |V| x d matrix.

For each node u ∈ V, we define its neighborhood N_S(u) ⊆ V using a sampling strategy S.

This framework supports various downstream tasks such as node classification, link prediction, and community detection.
