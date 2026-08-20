"""Scientific shape, ordering, residual, and autograd checks for the MLP."""

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
    hierarchy_coordinates,
    matrix_to_column_vector,
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
        C_list=np.array([0.2 + 0.1j]),
        gamma_list=np.array([0.3 + 1.0j]),
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


def test_coordinate_minibatch_has_section_three_shape():
    hierarchy = make_hierarchy()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(4,),
        t_start=2.0,
        t_stop=6.0,
    )
    times = torch.tensor([2.0, 4.0, 6.0], dtype=torch.float64)
    inputs = model.coordinate_inputs(times)
    assert inputs.shape == (3, hierarchy.nADO, 2 * hierarchy.K + 3)
    torch.testing.assert_close(
        inputs[:, :, -1],
        inputs.new_tensor([-1.0, 0.0, 1.0])[:, None].expand(
            -1, hierarchy.nADO
        ),
    )


def test_physical_time_derivative_includes_normalization_chain_rule():
    hierarchy = make_hierarchy(depth=1)
    short_interval = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        t_start=0.0,
        t_stop=2.0,
    )
    long_interval = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        t_start=0.0,
        t_stop=10.0,
    )
    long_interval.load_state_dict(short_interval.state_dict())

    short_state, short_derivative = state_and_time_derivative(
        short_interval,
        torch.tensor([1.0], dtype=torch.float64),
    )
    long_state, long_derivative = state_and_time_derivative(
        long_interval,
        torch.tensor([5.0], dtype=torch.float64),
    )
    torch.testing.assert_close(short_state, long_state)
    torch.testing.assert_close(short_derivative, 5.0 * long_derivative)


def test_column_major_vectorization_round_trip():
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


def test_symmetrization_enforces_adjoint_partners():
    hierarchy = make_hierarchy()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        t_start=0.0,
        t_stop=1.0,
    )
    raw = torch.randn(3, hierarchy.nADO, 8, dtype=torch.float64)
    real_state, imaginary_state = model.symmetrize_raw(raw)
    ados = torch.complex(real_state, imaginary_state).reshape(
        3, hierarchy.nADO, hierarchy.system_size
    )
    matrices = column_vector_to_matrix(ados, 2)
    partners = torch.as_tensor(conjugate_ado_permutation(hierarchy))
    torch.testing.assert_close(
        matrices[:, partners],
        matrices.transpose(-2, -1).conj(),
        rtol=0.0,
        atol=0.0,
    )


def test_requested_liouvillian_and_sparse_rhs_match_scipy():
    hierarchy = make_hierarchy(depth=1)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    with patch.object(
        hierarchy,
        "build_Liouvillian",
        wraps=hierarchy.build_Liouvillian,
    ) as constructor:
        objective = HEOMPINNLoss(hierarchy, rho0)
    constructor.assert_called_once_with(
        markovian_terminator=False,
        normalized=True,
    )

    rng = np.random.default_rng(4)
    state = rng.normal(size=hierarchy.liouvillian.shape[0]) + 1j * rng.normal(
        size=hierarchy.liouvillian.shape[0]
    )
    expected = q_func.state_to_real(hierarchy.liouvillian @ state)
    actual = objective.rhs(
        torch.as_tensor(q_func.state_to_real(state)[None], dtype=torch.float64)
    )
    np.testing.assert_allclose(actual.numpy()[0], expected, atol=1e-13)


def test_initial_condition_is_evaluated_at_physical_interval_start():
    hierarchy = make_hierarchy(depth=1)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        t_start=2.0,
        t_stop=6.0,
    )
    objective = HEOMPINNLoss(hierarchy, rho0)
    prediction = model(torch.tensor([2.0], dtype=torch.float64))[0]
    expected = (
        prediction - objective.initial_state
    ).square().sum() / objective.state_size
    torch.testing.assert_close(objective.initial_condition_loss(model), expected)


def test_jvp_and_loss_backpropagate_to_every_parameter():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(8, 8),
        t_start=0.0,
        t_stop=4.0,
    )
    objective = HEOMPINNLoss(hierarchy, rho0, liouvillian=liouvillian)
    times = torch.linspace(0.25, 3.75, 4, dtype=torch.float64)
    states, derivatives = state_and_time_derivative(model, times)
    finite_difference = (
        model(times + 1e-6) - model(times - 1e-6)
    ) / 2e-6
    torch.testing.assert_close(
        derivatives,
        finite_difference,
        rtol=1e-7,
        atol=1e-8,
    )

    objective(model, times).total.backward()
    assert states.shape == (4, 2 * model.state_size)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_partial_minibatches_match_the_full_objective():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        t_start=0.0,
        t_stop=1.0,
    )
    objective = HEOMPINNLoss(hierarchy, rho0, liouvillian=liouvillian)
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    full = objective(model, times, q_x=5)
    first = objective(model, times[:2], q_x=5)
    second = objective(model, times[2:], q_x=5)
    combined = [
        (2 * first_term + 3 * second_term) / 5
        for first_term, second_term in zip(first, second)
    ]
    for full_term, combined_term in zip(full, combined):
        torch.testing.assert_close(full_term, combined_term)


def test_short_training_and_solution_layout():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = np.diag([1.0, 0.0]).astype(np.complex128)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(6,),
        t_start=0.0,
        t_stop=0.2,
    )
    objective = HEOMPINNLoss(hierarchy, rho0, liouvillian=liouvillian)
    result = train_mlp(
        model,
        objective,
        TrainingConfig(
            t_start=0.0,
            t_stop=0.2,
            epochs=2,
            collocation_points=4,
            batch_size=2,
        ),
        verbose=False,
    )
    solution = solve_mlp(model, np.linspace(0.0, 0.2, 3), batch_size=2)
    assert len(result.history) == 2
    np.testing.assert_allclose(solution.t, np.linspace(0.0, 0.2, 3))
    assert solution.y.shape == (model.state_size, 3)
    np.testing.assert_allclose(
        solution.primary_ados,
        solution.primary_ados.conj().transpose(0, 2, 1),
        atol=1e-13,
    )
