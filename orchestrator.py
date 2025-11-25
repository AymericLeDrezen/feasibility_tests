# BEGIN orchestrator.py
from __future__ import annotations

import numpy as np

from data_tables import (
    lump_transformations_into_states,
    lump_transformations_into_effects,
    compute_data_table,
)
from example import states, effects, transformations, identity
from operational_identities import gen_operational_identities
from assignment_polytopes import construct_assignment_polytope
from test_feasibility import (
    build_linear_program_matrices,
    test_feasibility_of_noncategorical_model,
)


# ------------------------------
# 1. Lumped preparations and measurements
# ------------------------------

lumped_state_array = lump_transformations_into_states(
    states=states,
    transformations=transformations,
)
lumped_effect_array = lump_transformations_into_effects(
    transformations=transformations,
    effects=effects,
)


# ------------------------------
# 2. Data tables
# ------------------------------

probability_table_p_k_given_s_t = compute_data_table(
    states=states,
    transformations=transformations,
    effects=effects,
)
probability_table_from_lumped_states = compute_data_table(
    states=lumped_state_array,
    transformations=identity,
    effects=effects,
)
probability_table_from_lumped_effects = compute_data_table(
    states=states,
    transformations=identity,
    effects=lumped_effect_array,
)


# ------------------------------
# 3. Operational identities
# ------------------------------

operational_identity_dictionary = gen_operational_identities(
    states=states,
    effects=effects,
    transformations=transformations,
)
operational_identity_states = operational_identity_dictionary["states"]
operational_identity_effects = operational_identity_dictionary["effects"]
operational_identity_transformations = operational_identity_dictionary[
    "transformations"
]


# ------------------------------
# 4. Assignment polytopes (source and measurement)
# ------------------------------

assignment_polytope_dictionary = construct_assignment_polytope(
    states=states,
    effects=effects,
    operational_identities=operational_identity_dictionary,
)
source_assignment_vertices = assignment_polytope_dictionary["source_vertices"]
effect_assignment_vertices = assignment_polytope_dictionary["effect_vertices"]


# ------------------------------
# 5. Build (M, b) and test feasibility
# ------------------------------

coefficient_matrix_M, right_hand_side_vector_b = build_linear_program_matrices(
    probability_table_p_k_given_s_t=probability_table_p_k_given_s_t,
    source_assignment_vertices=source_assignment_vertices,
    effect_assignment_vertices=effect_assignment_vertices,
    transformation_identity_matrix=operational_identity_transformations,
)

linear_program_result = test_feasibility_of_noncategorical_model(
    probability_table_p_k_given_s_t=probability_table_p_k_given_s_t,
    source_assignment_vertices=source_assignment_vertices,
    effect_assignment_vertices=effect_assignment_vertices,
    transformation_identity_matrix=operational_identity_transformations,
)


if __name__ == "__main__":
    print("Operational effect identities (OEM):")
    print(operational_identity_effects)

    print("\nSource-assignment polytope vertices shape:", source_assignment_vertices.shape)
    print("Measurement-assignment polytope vertices shape:", effect_assignment_vertices.shape)

    print("\nLinear program matrix M shape:", coefficient_matrix_M.shape)
    print("Right-hand side vector b shape:", right_hand_side_vector_b.shape)

    print("\nFeasibility of noncontextual model (linprog.success):")
    print(linear_program_result.success)
# END orchestrator.py
