"""Shape, ordering, symmetry, sparse-physics, and autograd tests for the MLP."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from heom import q_func
from heom.heom_rep import heom_state
from model import (
    HEOMMLP,
    HEOMPINNLoss,
    TrainingConfig,
    column_vector_to_matrix,
    conjugate_ado_permutation,
    hierarchy_fingerprint,
    hierarchy_coordinates,
    matrix_to_column_vector,
    normalized_adjoint_factors,
    solve_mlp,
    state_and_time_derivative,
    train_mlp,
)


def make_hierarchy(depth=2):
    h_system = np.array([[0.5, 0.1], [0.1, -0.5]], dtype=np.complex128)
    h_coupling = np.diag([1.0, -1.0]).astype(np.complex128)
    return heom_state(
        K=0,
        L=depth,
        H_s=h_system,
        H_c=h_coupling,
        C_list=np.array([0.2 + 0.1j], dtype=np.complex128),
        gamma_list=np.array([0.3 + 1.0j], dtype=np.complex128),
    )


def test_coordinates_and_partner_permutation_follow_bfs_order():
    hierarchy = make_hierarchy(depth=2)
    assert hierarchy.idx_to_node == [
        ((0,), (0,)),
        ((1,), (0,)),
        ((0,), (1,)),
        ((2,), (0,)),
        ((1,), (1,)),
        ((0,), (2,)),
    ]
    np.testing.assert_array_equal(
        conjugate_ado_permutation(hierarchy),
        np.array([0, 2, 1, 5, 4, 3]),
    )
    np.testing.assert_allclose(
        hierarchy_coordinates(hierarchy),
        np.array(
            [
                [0.0, 0.0],
                [0.5, 0.0],
                [0.0, 0.5],
                [1.0, 0.0],
                [0.5, 0.5],
                [0.0, 1.0],
            ]
        ),
    )


def test_partner_permutation_handles_real_and_mixed_poles():
    h_system = np.diag([0.5, -0.5]).astype(np.complex128)
    h_coupling = np.diag([1.0, -1.0]).astype(np.complex128)
    real_hierarchy = heom_state(
        K=0,
        L=2,
        H_s=h_system,
        H_c=h_coupling,
        C_list=np.array([0.2]),
        gamma_list=np.array([0.3]),
    )
    np.testing.assert_array_equal(
        conjugate_ado_permutation(real_hierarchy),
        np.arange(real_hierarchy.nADO),
    )

    mixed_hierarchy = heom_state(
        K=1,
        L=1,
        H_s=h_system,
        H_c=h_coupling,
        C_list=np.array([0.2 + 0.1j, 0.05]),
        gamma_list=np.array([0.3 + 1.0j, 2.0]),
    )
    np.testing.assert_array_equal(
        conjugate_ado_permutation(mixed_hierarchy),
        np.array([0, 3, 2, 1]),
    )


def test_normalized_phase_twisted_symmetry_is_preserved_by_liouvillian():
    h_system = np.array([[0.5, 0.1], [0.1, -0.5]], dtype=np.complex128)
    h_coupling = np.diag([1.0, -1.0]).astype(np.complex128)
    hierarchy = heom_state(
        K=0,
        L=2,
        H_s=h_system,
        H_c=h_coupling,
        C_list=np.array([0.2 + 0.1j]),
        gamma_list=np.array([0.3]),
    )
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    model = HEOMMLP(hierarchy, hidden_sizes=(5,), dtype=torch.float64)
    factors = normalized_adjoint_factors(hierarchy)
    assert not np.allclose(factors, 1.0)

    state_real = model(torch.tensor([0.4], dtype=torch.float64)).detach().numpy()[0]
    state = state_real[: model.state_size] + 1j * state_real[model.state_size :]
    derivative = liouvillian @ state
    derivative_matrices = derivative.reshape(
        hierarchy.nADO, 2, 2
    ).transpose(0, 2, 1)
    partners = conjugate_ado_permutation(hierarchy)
    expected = factors[:, None, None] * derivative_matrices[
        partners
    ].conj().transpose(0, 2, 1)
    np.testing.assert_allclose(derivative_matrices, expected, atol=2e-14)


def test_coordinate_minibatch_has_section_three_shape_and_time_column():
    hierarchy = make_hierarchy(depth=2)
    model = HEOMMLP(hierarchy, hidden_sizes=(4,), dtype=torch.float64)
    times = torch.tensor([0.25, 0.75], dtype=torch.float64)
    inputs = model.coordinate_inputs(times)
    assert inputs.shape == (2, hierarchy.nADO, 2 * hierarchy.K + 3)
    torch.testing.assert_close(
        inputs[:, :, -1],
        times[:, None].expand(-1, hierarchy.nADO),
    )
    torch.testing.assert_close(
        inputs[0, :, :-1],
        torch.as_tensor(hierarchy_coordinates(hierarchy)),
    )
    assert all(
        not key.startswith(
            (
                "ado_coordinates",
                "conjugate_indices",
                "adjoint_factor_real",
                "adjoint_factor_imag",
            )
        )
        for key in model.state_dict()
    )


def test_hierarchy_fingerprint_detects_same_shape_physics_changes():
    first = make_hierarchy(depth=1)
    second = make_hierarchy(depth=1)
    second.C_list = second.C_list * np.exp(0.2j)
    assert first.nADO == second.nADO
    assert first.system_size == second.system_size
    assert hierarchy_fingerprint(first) != hierarchy_fingerprint(second)


def test_column_major_vectorization_round_trip_catches_transpose_errors():
    vector = torch.tensor(
        [[1.0 + 2.0j, 3.0 + 4.0j, 5.0 + 6.0j, 7.0 + 8.0j]],
        dtype=torch.complex128,
    )
    matrix = column_vector_to_matrix(vector, 2)
    expected = torch.tensor(
        [[[1.0 + 2.0j, 5.0 + 6.0j], [3.0 + 4.0j, 7.0 + 8.0j]]],
        dtype=torch.complex128,
    )
    torch.testing.assert_close(matrix, expected)
    torch.testing.assert_close(matrix_to_column_vector(matrix), vector)


def test_symmetrization_enforces_adjoint_partners_exactly():
    torch.manual_seed(2)
    hierarchy = make_hierarchy(depth=2)
    model = HEOMMLP(hierarchy, hidden_sizes=(5,), dtype=torch.float64)
    raw = torch.randn(
        3,
        hierarchy.nADO,
        2 * hierarchy.system_size,
        dtype=torch.float64,
    )
    real_state, imaginary_state = model.symmetrize_raw(raw)
    complex_ados = torch.complex(real_state, imaginary_state).reshape(
        3, hierarchy.nADO, hierarchy.system_size
    )
    matrices = column_vector_to_matrix(complex_ados, 2)
    partners = torch.as_tensor(conjugate_ado_permutation(hierarchy))
    torch.testing.assert_close(
        matrices.index_select(1, partners),
        matrices.transpose(-2, -1).conj(),
        rtol=0.0,
        atol=0.0,
    )


def test_loss_builds_requested_liouvillian_and_sparse_rhs_matches_scipy():
    hierarchy = make_hierarchy(depth=1)
    rho0 = np.array([[1.0, 0.2j], [-0.2j, 0.0]], dtype=np.complex128)
    with patch.object(
        hierarchy,
        "build_Liouvillian",
        wraps=hierarchy.build_Liouvillian,
    ) as constructor:
        objective = HEOMPINNLoss(hierarchy, rho0, dtype=torch.float64)
    constructor.assert_called_once_with(
        markovian_terminator=False,
        normalized=True,
    )

    liouvillian = hierarchy.liouvillian
    rng = np.random.default_rng(4)
    state = rng.normal(size=liouvillian.shape[0]) + 1j * rng.normal(
        size=liouvillian.shape[0]
    )
    state_real = q_func.state_to_real(state)
    expected = q_func.state_to_real(liouvillian @ state)
    actual = objective.rhs(
        torch.as_tensor(state_real[None, :], dtype=torch.float64)
    )
    np.testing.assert_allclose(actual.detach().numpy()[0], expected, atol=1e-13)


def test_jvp_and_complete_loss_backpropagate_to_every_parameter():
    torch.manual_seed(3)
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(
        markovian_terminator=False,
        normalized=True,
    )
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    model = HEOMMLP(hierarchy, hidden_sizes=(8, 8), dtype=torch.float64)
    objective = HEOMPINNLoss(
        hierarchy,
        rho0,
        liouvillian=liouvillian,
        dtype=torch.float64,
    )
    times = torch.linspace(0.0, 1.0, 4, dtype=torch.float64)
    states, derivatives = state_and_time_derivative(model, times)
    assert states.shape == derivatives.shape == (4, 2 * model.state_size)
    epsilon = 1e-6
    finite_difference = (
        model(times + epsilon) - model(times - epsilon)
    ) / (2 * epsilon)
    torch.testing.assert_close(
        derivatives,
        finite_difference,
        rtol=1e-7,
        atol=1e-8,
    )

    terms = objective(model, times)
    terms.total.backward()
    parameters = tuple(model.parameters())
    assert parameters
    assert all(parameter.grad is not None for parameter in parameters)
    assert all(torch.isfinite(parameter.grad).all() for parameter in parameters)


def test_float32_pipeline_has_finite_sparse_autograd():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    model = HEOMMLP(hierarchy, hidden_sizes=(4,), dtype=torch.float32)
    objective = HEOMPINNLoss(
        hierarchy,
        rho0,
        liouvillian=liouvillian,
        dtype=torch.float32,
    )
    terms = objective(model, torch.tensor([0.0, 0.1], dtype=torch.float32))
    terms.total.backward()
    assert torch.isfinite(terms.total)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_loss_rejects_same_size_model_with_different_bfs_contract():
    complex_hierarchy = make_hierarchy(depth=1)
    real_hierarchy = heom_state(
        K=1,
        L=1,
        H_s=complex_hierarchy.H_s,
        H_c=complex_hierarchy.H_c,
        C_list=np.array([0.2, 0.1]),
        gamma_list=np.array([0.3, 0.7]),
    )
    assert complex_hierarchy.nADO == real_hierarchy.nADO == 3
    liouvillian = complex_hierarchy.build_Liouvillian(normalized=True)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    objective = HEOMPINNLoss(
        complex_hierarchy,
        rho0,
        liouvillian=liouvillian,
    )
    wrong_model = HEOMMLP(real_hierarchy, hidden_sizes=(4,))
    try:
        objective(wrong_model, torch.tensor([0.1]))
    except ValueError as error:
        assert "exact hierarchy instance" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("different same-size hierarchies were accepted")


def test_qx_rescaling_makes_partial_minibatches_match_full_objective():
    torch.manual_seed(11)
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    model = HEOMMLP(hierarchy, hidden_sizes=(5,), dtype=torch.float64)
    objective = HEOMPINNLoss(
        hierarchy,
        rho0,
        liouvillian=liouvillian,
        dtype=torch.float64,
    )
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    full = objective(model, times, q_x=times.numel())
    first = objective(model, times[:2], q_x=times.numel())
    second = objective(model, times[2:], q_x=times.numel())
    combined = tuple(
        (2 * first_term + 3 * second_term) / 5
        for first_term, second_term in zip(first, second)
    )
    for full_term, combined_term in zip(full, combined):
        torch.testing.assert_close(full_term, combined_term)


def test_loss_rejects_prebuilt_operator_with_wrong_truncation_options():
    hierarchy = make_hierarchy(depth=1)
    wrong_liouvillian = hierarchy.build_Liouvillian(normalized=False)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    try:
        HEOMPINNLoss(
            hierarchy,
            rho0,
            liouvillian=wrong_liouvillian,
        )
    except ValueError as error:
        assert "normalized hard-cutoff" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("an unnormalized Liouvillian was accepted")


def test_failed_liouvillian_rebuild_preserves_last_valid_cache_atomically():
    zero_operator = np.zeros((2, 2), dtype=np.complex128)
    hierarchy = heom_state(
        K=0,
        L=1,
        H_s=zero_operator,
        H_c=np.diag([1.0, -1.0]).astype(np.complex128),
        C_list=np.array([0.2]),
        gamma_list=np.array([0.0]),
    )
    valid = hierarchy.build_Liouvillian(
        markovian_terminator=False,
        normalized=True,
    )
    valid_options = hierarchy.liouvillian_options.copy()
    try:
        hierarchy.build_Liouvillian(
            markovian_terminator=True,
            normalized=True,
        )
    except ValueError as error:
        assert "singular" in str(error)
    else:  # pragma: no cover - assertion branch
        raise AssertionError("the singular Markovian terminator was accepted")
    assert hierarchy.liouvillian is valid
    assert hierarchy.liouvillian_options == valid_options


def test_short_training_and_solution_preserve_trajectory_layout():
    torch.manual_seed(5)
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(
        markovian_terminator=False,
        normalized=True,
    )
    rho0 = np.array([[1.0, 0.0], [0.0, 0.0]], dtype=np.complex128)
    model = HEOMMLP(hierarchy, hidden_sizes=(6,), dtype=torch.float64)
    objective = HEOMPINNLoss(
        hierarchy,
        rho0,
        liouvillian=liouvillian,
        dtype=torch.float64,
    )
    result = train_mlp(
        model,
        objective,
        TrainingConfig(
            t_start=0.0,
            t_stop=0.2,
            epochs=2,
            collocation_points=4,
            batch_size=2,
            log_every=1,
            seed=7,
        ),
        verbose=False,
    )
    assert len(result.history) == 2
    assert np.isfinite(result.final.total)

    times = np.linspace(0.0, 0.2, 3)
    solution = solve_mlp(model, times, batch_size=2)
    assert solution.y.shape == (model.state_size, times.size)
    assert solution.primary_ados.shape == (times.size, 2, 2)
    assert np.all(np.isfinite(solution.y))
    np.testing.assert_allclose(
        solution.primary_ados,
        solution.primary_ados.conj().transpose(0, 2, 1),
        atol=1e-13,
    )
