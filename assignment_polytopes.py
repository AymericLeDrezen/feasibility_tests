# assignment_polytopes.py
from __future__ import annotations

import numpy as np
import cdd


def compute_vertices_from_halfspace_representation(
    number_of_variables: int,
    inequality_matrix: np.ndarray,
    inequality_vector: np.ndarray,
) -> np.ndarray:
    """
    Given an H-representation
        { x in R^n | inequality_matrix x <= inequality_vector },
    compute the vertices of the corresponding polytope using cdd.

    number_of_variables: n
    inequality_matrix: shape (M, n)
    inequality_vector: shape (M,)
    """

    halfspace_array = np.hstack(
        [inequality_vector.reshape(-1, 1), -inequality_matrix]
    )

    cdd_matrix = cdd.matrix_from_array(
        halfspace_array,
        rep_type=cdd.RepType.INEQUALITY,
    )
    polyhedron = cdd.polyhedron_from_matrix(cdd_matrix)
    generator_matrix = cdd.copy_generators(polyhedron)
    generator_array = np.array(generator_matrix.array, dtype=float)

    is_vertex_row = generator_array[:, 0] == 1.0
    vertex_array = generator_array[is_vertex_row][:, 1:]

    return vertex_array


def build_effect_assignment_halfspace_representation(
    number_of_effects: int,
    effect_identity_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the H-representation (inequality_matrix, inequality_vector) for the
    effect-assignment polytope in R^{number_of_effects}.

    Variables: effect_assignment_vector[k] = ξ_{e_k}(λ)

    Constraints:
      0 <= effect_assignment_vector[k] <= 1
      Σ_k alpha_k * effect_assignment_vector[k] = 0
        for each row alpha of effect_identity_matrix.
    """

    inequality_matrix_list = []
    inequality_vector_list = []

    # 0 <= x_k  <=>  -x_k <= 0
    inequality_matrix_list.append(-np.eye(number_of_effects))
    inequality_vector_list.append(np.zeros(number_of_effects))

    # x_k <= 1
    inequality_matrix_list.append(np.eye(number_of_effects))
    inequality_vector_list.append(np.ones(number_of_effects))

    for effect_identity_row in effect_identity_matrix:
        effect_identity_row = np.asarray(effect_identity_row, float).reshape(1, -1)

        # alpha · x = 0 as two inequalities
        inequality_matrix_list.append(effect_identity_row)
        inequality_vector_list.append(np.array([0.0]))

        inequality_matrix_list.append(-effect_identity_row)
        inequality_vector_list.append(np.array([0.0]))

    inequality_matrix = np.vstack(inequality_matrix_list)
    inequality_vector = np.concatenate(inequality_vector_list)

    return inequality_matrix, inequality_vector


def build_source_assignment_halfspace_representation(
    number_of_states: int,
    state_identity_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the H-representation (inequality_matrix, inequality_vector) for the
    source-assignment polytope in R^{number_of_states}.

    Variables: source_assignment_vector[s] = ξ_{s̄ = s}(λ)

    Constraints:
      source_assignment_vector[s] >= 0
      Σ_s source_assignment_vector[s] = 1
      Σ_s beta_s * source_assignment_vector[s] = 0
        for each row beta of state_identity_matrix.
    """

    inequality_matrix_list = []
    inequality_vector_list = []

    # y_s >= 0  <=>  -y_s <= 0
    inequality_matrix_list.append(-np.eye(number_of_states))
    inequality_vector_list.append(np.zeros(number_of_states))

    normalization_row = np.ones(number_of_states).reshape(1, -1)

    # Σ_s y_s = 1 as two inequalities
    inequality_matrix_list.append(normalization_row)
    inequality_vector_list.append(np.array([1.0]))

    inequality_matrix_list.append(-normalization_row)
    inequality_vector_list.append(np.array([-1.0]))

    for state_identity_row in state_identity_matrix:
        state_identity_row = np.asarray(state_identity_row, float).reshape(1, -1)

        # beta · y = 0 as two inequalities
        inequality_matrix_list.append(state_identity_row)
        inequality_vector_list.append(np.array([0.0]))

        inequality_matrix_list.append(-state_identity_row)
        inequality_vector_list.append(np.array([0.0]))

    inequality_matrix = np.vstack(inequality_matrix_list)
    inequality_vector = np.concatenate(inequality_vector_list)

    return inequality_matrix, inequality_vector


def construct_assignment_polytope(
    states: np.ndarray,
    effects: np.ndarray,
    operational_identities: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    Construct source- and effect-assignment polytopes from linear
    operational identities among states and effects.

    states: array of shape (number_of_states, dimension)
    effects: array of shape (number_of_effects, dimension)
    operational_identities:
        {
          "states":  array of shape (number_of_state_identities, number_of_states),
          "effects": array of shape (number_of_effect_identities, number_of_effects),
        }

    Returns:
        {
          "source_vertices": array of shape (number_of_source_vertices, number_of_states),
          "effect_vertices": array of shape (number_of_effect_vertices, number_of_effects),
        }
    """

    number_of_states = states.shape[0]
    number_of_effects = effects.shape[0]

    state_identity_matrix = operational_identities["states"]
    effect_identity_matrix = operational_identities["effects"]

    effect_inequality_matrix, effect_inequality_vector = (
        build_effect_assignment_halfspace_representation(
            number_of_effects=number_of_effects,
            effect_identity_matrix=effect_identity_matrix,
        )
    )

    source_inequality_matrix, source_inequality_vector = (
        build_source_assignment_halfspace_representation(
            number_of_states=number_of_states,
            state_identity_matrix=state_identity_matrix,
        )
    )

    effect_vertices = compute_vertices_from_halfspace_representation(
        number_of_variables=number_of_effects,
        inequality_matrix=effect_inequality_matrix,
        inequality_vector=effect_inequality_vector,
    )

    source_vertices = compute_vertices_from_halfspace_representation(
        number_of_variables=number_of_states,
        inequality_matrix=source_inequality_matrix,
        inequality_vector=source_inequality_vector,
    )

    return {
        "source_vertices": source_vertices,
        "effect_vertices": effect_vertices,
    }
