# BEGIN test_feasibility.py


# Not Reviewed GPT code



from __future__ import annotations

import numpy as np
from scipy.optimize import linprog


def build_linear_program_matrices(
    probability_table_p_k_given_s_t: np.ndarray,
    source_assignment_vertices: np.ndarray,
    effect_assignment_vertices: np.ndarray,
    transformation_identity_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the coefficient matrix and right-hand-side vector (M, b)
    for the PTM feasibility problem in Formulation F1.

    Unknowns:
        x[t, source_vertex_index, effect_vertex_index]
            = p(kappa_prime = effect_vertex_index,
                kappa = source_vertex_index | t)

    Arrays
    ------
    probability_table_p_k_given_s_t : array, shape (number_of_effects,
                                                   number_of_states,
                                                   number_of_transformations)
        Empirical probabilities p(k | s, t).

    source_assignment_vertices : array, shape (number_of_source_vertices,
                                               number_of_states)
        Each row is a source-assignment vertex Ψ_kappa.

    effect_assignment_vertices : array, shape (number_of_effect_vertices,
                                               number_of_effects)
        Each row is an effect-assignment vertex Φ_kappa'.

    transformation_identity_matrix : array, shape (number_of_transformation_identities,
                                                   number_of_transformations)
        Each row alpha^(c) encodes an operational identity
        Σ_t alpha_t^(c) T_t ≃ 0.

    Returns
    -------
    coefficient_matrix_M : array, shape (number_of_constraints,
                                         number_of_variables)
    right_hand_side_vector : array, shape (number_of_constraints,)
    """

    number_of_effects, number_of_states, number_of_transformations = (
        probability_table_p_k_given_s_t.shape
    )
    number_of_source_vertices, _ = source_assignment_vertices.shape
    number_of_effect_vertices, _ = effect_assignment_vertices.shape
    number_of_transformation_identities, _ = transformation_identity_matrix.shape

    number_of_variables = (
        number_of_transformations
        * number_of_source_vertices
        * number_of_effect_vertices
    )

    basis_transformation_matrix = np.eye(number_of_transformations)
    basis_source_vertex_matrix = np.eye(number_of_source_vertices)
    basis_effect_vertex_matrix = np.eye(number_of_effect_vertices)

    coefficient_row_list = []
    right_hand_side_list = []

    # Convenience vectors
    ones_over_source_and_effect_vertices = np.ones(
        number_of_source_vertices * number_of_effect_vertices
    )
    ones_over_effect_vertices = np.ones(number_of_effect_vertices)

    # ----- Block F1b: normalization for each transformation -----
    # Σ_{kappa, kappa'} p(kappa', kappa | t) = 1
    for transformation_index in range(number_of_transformations):
        transformation_basis_vector = basis_transformation_matrix[transformation_index]
        coefficient_row_vector = np.kron(
            transformation_basis_vector,
            ones_over_source_and_effect_vertices,
        )
        coefficient_row_list.append(coefficient_row_vector)
        right_hand_side_list.append(1.0)

    # ----- Block F1c: non-signalling from transformation to source -----
    # For each source vertex and each t > 0:
    # Σ_{kappa'} [p(kappa', kappa | t) - p(kappa', kappa | 0)] = 0
    reference_transformation_basis_vector = basis_transformation_matrix[0]

    for source_vertex_index in range(number_of_source_vertices):
        source_basis_vector = basis_source_vertex_matrix[source_vertex_index]
        source_effect_block = np.kron(source_basis_vector, ones_over_effect_vertices)
        for transformation_index in range(1, number_of_transformations):
            transformation_basis_vector = basis_transformation_matrix[
                transformation_index
            ]
            transformation_difference_vector = (
                transformation_basis_vector - reference_transformation_basis_vector
            )
            coefficient_row_vector = np.kron(
                transformation_difference_vector,
                source_effect_block,
            )
            coefficient_row_list.append(coefficient_row_vector)
            right_hand_side_list.append(0.0)

    # ----- Block F1d: operational identities among transformations -----
    # For each identity row alpha^(c) and each pair (kappa, kappa'):
    # Σ_t alpha_t^(c) p(kappa', kappa | t) = 0
    for transformation_identity_index in range(number_of_transformation_identities):
        transformation_identity_row = transformation_identity_matrix[
            transformation_identity_index
        ]
        for source_vertex_index in range(number_of_source_vertices):
            source_basis_vector = basis_source_vertex_matrix[source_vertex_index]
            for effect_vertex_index in range(number_of_effect_vertices):
                effect_basis_vector = basis_effect_vertex_matrix[effect_vertex_index]
                source_effect_block = np.kron(
                    source_basis_vector,
                    effect_basis_vector,
                )
                coefficient_row_vector = np.kron(
                    transformation_identity_row,
                    source_effect_block,
                )
                coefficient_row_list.append(coefficient_row_vector)
                right_hand_side_list.append(0.0)

    # ----- Block F1e: data reproduction -----
    # For each (k, s, t):
    # N Σ_{kappa, kappa'} Φ_{kappa'}(k) Ψ_{kappa}(s) p(kappa', kappa | t) = p(k | s, t)
    number_of_original_states = number_of_states
    normalization_factor_N = float(number_of_original_states)

    for transformation_index in range(number_of_transformations):
        transformation_basis_vector = basis_transformation_matrix[transformation_index]
        for state_index in range(number_of_states):
            source_vertex_column = source_assignment_vertices[:, state_index]
            for effect_index in range(number_of_effects):
                effect_vertex_column = effect_assignment_vertices[:, effect_index]

                source_effect_block = normalization_factor_N * np.kron(
                    source_vertex_column,
                    effect_vertex_column,
                )
                coefficient_row_vector = np.kron(
                    transformation_basis_vector,
                    source_effect_block,
                )
                coefficient_row_list.append(coefficient_row_vector)

                right_hand_side_value = probability_table_p_k_given_s_t[
                    effect_index,
                    state_index,
                    transformation_index,
                ]
                right_hand_side_list.append(right_hand_side_value)

    coefficient_matrix_M = np.vstack(coefficient_row_list)
    right_hand_side_vector = np.array(right_hand_side_list, dtype=float)

    return coefficient_matrix_M, right_hand_side_vector


def test_feasibility_of_noncategorical_model(
    probability_table_p_k_given_s_t: np.ndarray,
    source_assignment_vertices: np.ndarray,
    effect_assignment_vertices: np.ndarray,
    transformation_identity_matrix: np.ndarray,
) -> linprog:
    """
    Test whether a given PTM data table admits a noncontextual model.

    The test is:
        ∃ x ≥ 0 such that M x = b,

    where M and b are built from Formulation F1 and the given scenario.

    Returns
    -------
    linear_program_result : scipy.optimize.OptimizeResult
        Result object from `scipy.optimize.linprog`. The PTM data are
        noncontextual iff `linear_program_result.success` is True.
    """

    (
        coefficient_matrix_M,
        right_hand_side_vector,
    ) = build_linear_program_matrices(
        probability_table_p_k_given_s_t=probability_table_p_k_given_s_t,
        source_assignment_vertices=source_assignment_vertices,
        effect_assignment_vertices=effect_assignment_vertices,
        transformation_identity_matrix=transformation_identity_matrix,
    )

    number_of_effect_vertices = effect_assignment_vertices.shape[0]
    number_of_source_vertices = source_assignment_vertices.shape[0]
    number_of_transformations = probability_table_p_k_given_s_t.shape[2]

    number_of_variables = (
        number_of_transformations
        * number_of_source_vertices
        * number_of_effect_vertices
    )

    objective_coefficients = np.zeros(number_of_variables)

    linear_program_result = linprog(
        c=objective_coefficients,
        A_eq=coefficient_matrix_M,
        b_eq=right_hand_side_vector,
        bounds=(0.0, None),
        method="highs",
    )

    return linear_program_result


# END test_feasibility.py
