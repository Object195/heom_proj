"""Scientific shape, ordering, residual, and autograd checks for the MLP."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch

from experiment_parameters import MLP
from heom import q_func
from heom.heom_rep import heom_state
from heom.heom_solver import prepare_heom_initial_state, solve_heom
from model import (
    EpochRecord,
    HEOMMLP,
    HEOMPINNLoss,
    TrainingConfig,
    column_vector_to_matrix,
    compute_heom_dynamical_scales,
    conjugate_ado_permutation,
    hierarchy_coordinates,
    matrix_to_column_vector,
    solve_mlp,
    state_and_time_derivative,
    train_mlp,
)
from model.train_mlp_model import (
    LiveLossPlot,
    build_argument_parser,
    build_optimizer,
    load_pretrained_network,
    load_saved_model,
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


def make_rho0():
    return np.diag([1.0, 0.0]).astype(np.complex128)


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
    np.testing.assert_allclose(
        hierarchy_coordinates(hierarchy, normalize=False),
        np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 0.0],
                [1.0, 1.0],
                [0.0, 2.0],
            ]
        ),
    )


def test_hierarchy_coordinate_normalization_can_be_disabled():
    hierarchy = make_hierarchy(depth=2)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(4,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
        normalize_hierarchy_coordinates=False,
    )

    torch.testing.assert_close(
        model.ado_coordinates,
        torch.as_tensor(
            hierarchy_coordinates(hierarchy, normalize=False),
            dtype=model.dtype,
        ),
    )
    assertion = TestCase()
    with assertion.assertRaisesRegex(
        TypeError,
        "normalize_hierarchy_coordinates must be a boolean",
    ):
        HEOMMLP(
            hierarchy,
            hidden_sizes=(4,),
            rho0=make_rho0(),
            t_start=0.0,
            t_stop=1.0,
            normalize_hierarchy_coordinates="no",
        )


def test_coordinate_minibatch_has_section_three_shape():
    hierarchy = make_hierarchy()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(4,),
        rho0=make_rho0(),
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
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=2.0,
    )
    long_interval = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
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


def test_exponential_switch_endpoints_extrapolation_and_time_derivative():
    hierarchy = make_hierarchy(depth=1)
    t_start = 2.0
    t_stop = 6.0
    time_constant = 0.75
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=t_start,
        t_stop=t_stop,
        time_switch="exponential",
        switch_time_constant=time_constant,
    )

    endpoint_times = torch.tensor([t_start, t_stop], dtype=torch.float64)
    torch.testing.assert_close(
        model.switching_function(endpoint_times),
        torch.tensor([0.0, 1.0], dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )
    late_switch = model.switching_function(
        torch.tensor([100.0], dtype=torch.float64)
    )
    asymptote = 1.0 / (-np.expm1(-(t_stop - t_start) / time_constant))
    torch.testing.assert_close(
        late_switch,
        torch.tensor([asymptote], dtype=torch.float64),
        rtol=1e-14,
        atol=1e-14,
    )

    initial_time = torch.tensor([t_start], dtype=torch.float64)
    _, initial_derivative = state_and_time_derivative(model, initial_time)
    denominator = -np.expm1(-(t_stop - t_start) / time_constant)
    expected_initial_derivative = model.state_correction(initial_time) / (
        time_constant * denominator
    )
    torch.testing.assert_close(
        initial_derivative,
        expected_initial_derivative,
        rtol=1e-12,
        atol=1e-12,
    )

    interior_time = torch.tensor([3.25], dtype=torch.float64)
    _, derivative = state_and_time_derivative(
        model,
        interior_time,
        create_graph=False,
    )
    step = 1e-6
    finite_difference = (
        model(interior_time + step) - model(interior_time - step)
    ) / (2.0 * step)
    torch.testing.assert_close(
        derivative,
        finite_difference,
        rtol=1e-7,
        atol=1e-8,
    )


def test_time_switch_validation_rejects_invalid_configuration():
    hierarchy = make_hierarchy(depth=1)
    common = {
        "hidden_sizes": (5,),
        "rho0": make_rho0(),
        "t_start": 0.0,
        "t_stop": 1.0,
    }
    with np.testing.assert_raises_regex(ValueError, "time_switch"):
        HEOMMLP(hierarchy, time_switch="quadratic", **common)
    with np.testing.assert_raises_regex(ValueError, "time_constant"):
        HEOMMLP(
            hierarchy,
            time_switch="exponential",
            switch_time_constant=0.0,
            **common,
        )


def test_later_start_uses_complete_sparse_evolved_heom_state():
    hierarchy = make_hierarchy(depth=1)
    rho0 = make_rho0()
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    t_start = 0.2
    prepared = prepare_heom_initial_state(
        hierarchy,
        rho0,
        t_start,
        liouvillian=liouvillian,
        rtol=1e-10,
        atol=1e-12,
    )
    direct = solve_heom(
        hierarchy,
        rho0,
        np.array([0.0, t_start]),
        liouvillian=liouvillian,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(prepared, direct.y[:, -1], atol=1e-13)
    assert np.linalg.norm(prepared[hierarchy.system_size :]) > 0.0

    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        initial_heom_state=prepared,
        t_start=t_start,
        t_stop=1.0,
    )
    model_at_start = model(torch.tensor([t_start], dtype=torch.float64))[0]
    torch.testing.assert_close(
        model_at_start,
        torch.as_tensor(q_func.state_to_real(prepared), dtype=torch.float64),
    )

    continued = solve_heom(
        hierarchy,
        None,
        np.array([t_start, 0.3]),
        initial_state=prepared,
        liouvillian=liouvillian,
        rtol=1e-10,
        atol=1e-12,
    )
    uninterrupted = solve_heom(
        hierarchy,
        rho0,
        np.array([0.0, 0.3]),
        liouvillian=liouvillian,
        rtol=1e-10,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        continued.y[:, -1],
        uninterrupted.y[:, -1],
        rtol=1e-9,
        atol=1e-11,
    )


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
        rho0=make_rho0(),
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
    with patch.object(
        hierarchy,
        "build_Liouvillian",
        wraps=hierarchy.build_Liouvillian,
    ) as constructor:
        objective = HEOMPINNLoss(hierarchy)
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


def test_output_enforces_initial_state_and_root_trace():
    hierarchy = make_hierarchy(depth=1)
    rho0 = make_rho0()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=rho0,
        t_start=2.0,
        t_stop=6.0,
    )
    times = torch.tensor([2.0, 3.0, 6.0], dtype=torch.float64)
    states = model(times)
    torch.testing.assert_close(states[0], model.initial_state)

    correction = model.state_correction(times)
    real_correction, imaginary_correction = correction.split(
        model.state_size, dim=-1
    )
    real_trace = real_correction.index_select(
        1, model.root_diagonal_indices
    ).sum(dim=1)
    imaginary_trace = imaginary_correction.index_select(
        1, model.root_diagonal_indices
    ).sum(dim=1)
    torch.testing.assert_close(real_trace, torch.zeros_like(real_trace))
    torch.testing.assert_close(
        imaginary_trace, torch.zeros_like(imaginary_trace)
    )

    raw_real, raw_imaginary = model.symmetrize_raw(model.raw_output(times))
    torch.testing.assert_close(
        real_correction[:, model.system_size :],
        raw_real[:, model.system_size :],
    )
    torch.testing.assert_close(
        imaginary_correction[:, model.system_size :],
        raw_imaginary[:, model.system_size :],
    )

    correction_ados = torch.complex(
        real_correction, imaginary_correction
    ).reshape(times.numel(), hierarchy.nADO, hierarchy.system_size)
    correction_matrices = column_vector_to_matrix(correction_ados, 2)
    partners = torch.as_tensor(conjugate_ado_permutation(hierarchy))
    torch.testing.assert_close(
        correction_matrices[:, partners],
        correction_matrices.transpose(-2, -1).conj(),
    )

    root_traces = torch.diagonal(
        model.root_density_matrices(times), dim1=-2, dim2=-1
    ).sum(dim=-1)
    torch.testing.assert_close(root_traces, torch.ones_like(root_traces))

    _, initial_derivative = state_and_time_derivative(model, times[:1])
    expected_initial_derivative = model.state_correction(times[:1]) / (
        model.t_stop - model.t_start
    )
    torch.testing.assert_close(
        initial_derivative, expected_initial_derivative
    )


def test_correction_scale_rescales_the_ansatz_without_entering_state_dict():
    hierarchy = make_hierarchy(depth=1)
    common = {
        "hidden_sizes": (5,),
        "rho0": make_rho0(),
        "t_start": 1.0,
        "t_stop": 3.0,
    }
    unscaled = HEOMMLP(hierarchy, correction_scale=1.0, **common)
    scaled = HEOMMLP(hierarchy, correction_scale=3.0, **common)
    scaled.load_state_dict(unscaled.state_dict())

    assert "correction_scale" not in unscaled.state_dict()
    torch.testing.assert_close(
        unscaled.complex_initial_state(),
        torch.as_tensor(
            hierarchy.build_initial_state(make_rho0(), as_sparse=False),
            dtype=torch.complex128,
        ),
    )
    times = torch.tensor([1.0, 1.75, 3.0], dtype=torch.float64)
    unscaled_state = unscaled(times)
    scaled_state = scaled(times)
    torch.testing.assert_close(unscaled_state[0], unscaled.initial_state)
    torch.testing.assert_close(scaled_state[0], scaled.initial_state)
    torch.testing.assert_close(
        scaled_state - scaled.initial_state,
        3.0 * (unscaled_state - unscaled.initial_state),
    )

    initial_time = times[:1]
    _, unscaled_derivative = state_and_time_derivative(
        unscaled,
        initial_time,
    )
    _, scaled_derivative = state_and_time_derivative(scaled, initial_time)
    torch.testing.assert_close(
        scaled_derivative,
        3.0 * unscaled_derivative,
    )

    scaled.set_correction_scale(2.0)
    torch.testing.assert_close(
        scaled(times) - scaled.initial_state,
        2.0 * (unscaled_state - unscaled.initial_state),
    )


def test_jvp_and_loss_backpropagate_to_every_parameter():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = make_rho0()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(8, 8),
        rho0=rho0,
        t_start=0.0,
        t_stop=4.0,
    )
    objective = HEOMPINNLoss(hierarchy, liouvillian=liouvillian)
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

    loss = objective(model, times)
    expected_loss = (
        derivatives - objective.rhs(states)
    ).square().sum() / (model.state_size * times.numel())
    torch.testing.assert_close(loss, expected_loss)
    loss.backward()
    assert states.shape == (4, 2 * model.state_size)
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_tier_normalized_loss_matches_equal_tier_average():
    hierarchy = make_hierarchy(depth=2)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        tier_normalized=True,
    )
    batch_size = 5
    generator = torch.Generator().manual_seed(123)
    state = torch.zeros(
        batch_size,
        2 * objective.state_size,
        dtype=torch.float64,
    )
    derivative = torch.randn(
        state.shape,
        dtype=torch.float64,
        generator=generator,
    )

    actual = objective.dynamics_loss(state, derivative)
    residual_u, residual_v = derivative.split(objective.state_size, dim=-1)
    energy = (
        residual_u.reshape(batch_size, hierarchy.nADO, hierarchy.system_size)
        .square()
        + residual_v.reshape(
            batch_size,
            hierarchy.nADO,
            hierarchy.system_size,
        ).square()
    )
    tier_means = []
    for tier in range(hierarchy.L + 1):
        indices = torch.as_tensor(
            [
                index
                for index, node in enumerate(hierarchy.idx_to_node)
                if hierarchy._tier(node) == tier
            ],
            dtype=torch.long,
        )
        tier_means.append(energy.index_select(1, indices).mean())
    expected = torch.stack(tier_means).mean()

    torch.testing.assert_close(actual, expected)


def test_lower_tier_weight_matches_two_group_tier_average():
    hierarchy = make_hierarchy(depth=2)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    beta = 0.75
    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        tier_normalized=True,
        lower_tier_cutoff=0,
        lower_tier_weight=beta,
    )
    batch_size = 5
    generator = torch.Generator().manual_seed(456)
    state = torch.zeros(
        batch_size,
        2 * objective.state_size,
        dtype=torch.float64,
    )
    derivative = torch.randn(
        state.shape,
        dtype=torch.float64,
        generator=generator,
    )

    actual = objective.dynamics_loss(state, derivative)
    residual_u, residual_v = derivative.split(objective.state_size, dim=-1)
    energy = (
        residual_u.reshape(batch_size, hierarchy.nADO, hierarchy.system_size)
        .square()
        + residual_v.reshape(
            batch_size,
            hierarchy.nADO,
            hierarchy.system_size,
        ).square()
    )
    tier_means = []
    for tier in range(hierarchy.L + 1):
        indices = torch.as_tensor(
            [
                index
                for index, node in enumerate(hierarchy.idx_to_node)
                if hierarchy._tier(node) == tier
            ],
            dtype=torch.long,
        )
        tier_means.append(energy.index_select(1, indices).mean())
    expected = beta * tier_means[0] + (1.0 - beta) * torch.stack(
        tier_means[1:]
    ).mean()

    torch.testing.assert_close(actual, expected)


def test_tier_loss_power_matches_normalized_power_law_average():
    hierarchy = make_hierarchy(depth=2)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    power = 2.0
    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        tier_normalized=True,
        tier_loss_power=power,
    )
    batch_size = 5
    generator = torch.Generator().manual_seed(789)
    state = torch.zeros(
        batch_size,
        2 * objective.state_size,
        dtype=torch.float64,
    )
    derivative = torch.randn(
        state.shape,
        dtype=torch.float64,
        generator=generator,
    )

    actual = objective.dynamics_loss(state, derivative)
    residual_u, residual_v = derivative.split(objective.state_size, dim=-1)
    energy = (
        residual_u.reshape(batch_size, hierarchy.nADO, hierarchy.system_size)
        .square()
        + residual_v.reshape(
            batch_size,
            hierarchy.nADO,
            hierarchy.system_size,
        ).square()
    )
    tier_means = []
    for tier in range(hierarchy.L + 1):
        indices = torch.as_tensor(
            [
                index
                for index, node in enumerate(hierarchy.idx_to_node)
                if hierarchy._tier(node) == tier
            ],
            dtype=torch.long,
        )
        tier_means.append(energy.index_select(1, indices).mean())
    weights = torch.arange(
        1,
        hierarchy.L + 2,
        dtype=torch.float64,
    ).pow(-power)
    expected = torch.sum(weights * torch.stack(tier_means)) / weights.sum()

    torch.testing.assert_close(actual, expected)


def test_lower_tier_weight_validates_the_two_group_configuration():
    hierarchy = make_hierarchy(depth=2)
    assertion = TestCase()

    with assertion.assertRaisesRegex(ValueError, "must be set together"):
        HEOMPINNLoss(
            hierarchy,
            tier_normalized=True,
            lower_tier_cutoff=0,
        )
    with assertion.assertRaisesRegex(ValueError, "requires tier_normalized"):
        HEOMPINNLoss(
            hierarchy,
            tier_normalized=False,
            lower_tier_cutoff=0,
            lower_tier_weight=0.5,
        )
    with assertion.assertRaisesRegex(ValueError, "maximum occupied tier"):
        HEOMPINNLoss(
            hierarchy,
            tier_normalized=True,
            lower_tier_cutoff=2,
            lower_tier_weight=0.5,
        )
    with assertion.assertRaisesRegex(ValueError, "between 0 and 1"):
        HEOMPINNLoss(
            hierarchy,
            tier_normalized=True,
            lower_tier_cutoff=0,
            lower_tier_weight=1.1,
        )
    with assertion.assertRaisesRegex(ValueError, "mutually exclusive"):
        HEOMPINNLoss(
            hierarchy,
            tier_normalized=True,
            lower_tier_cutoff=0,
            lower_tier_weight=0.5,
            tier_loss_power=1.0,
        )
    with assertion.assertRaisesRegex(ValueError, "requires tier_normalized"):
        HEOMPINNLoss(
            hierarchy,
            tier_normalized=False,
            tier_loss_power=1.0,
        )


def test_dynamical_scales_match_global_and_equal_tier_constant_losses():
    hierarchy = make_hierarchy(depth=2)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    initial_state = hierarchy.build_initial_state(
        make_rho0(),
        as_sparse=False,
    )
    rhs = np.asarray(liouvillian @ initial_state).reshape(
        hierarchy.nADO,
        hierarchy.system_size,
    )
    component_energy = np.abs(rhs) ** 2
    expected_global = float(component_energy.mean())
    expected_tiers = []
    for tier in range(hierarchy.L + 1):
        indices = [
            index
            for index, node in enumerate(hierarchy.idx_to_node)
            if hierarchy._tier(node) == tier
        ]
        expected_tiers.append(float(component_energy[indices].mean()))
    expected_tier_average = float(np.mean(expected_tiers))

    common = {
        "t_start": 2.0,
        "t_stop": 6.0,
        "normalization_floor": 1e-30,
    }
    global_scales = compute_heom_dynamical_scales(
        hierarchy,
        initial_state,
        liouvillian,
        tier_normalized=False,
        **common,
    )
    tier_scales = compute_heom_dynamical_scales(
        hierarchy,
        initial_state,
        liouvillian,
        tier_normalized=True,
        **common,
    )
    beta = 0.75
    grouped_scales = compute_heom_dynamical_scales(
        hierarchy,
        initial_state,
        liouvillian,
        tier_normalized=True,
        lower_tier_cutoff=0,
        lower_tier_weight=beta,
        **common,
    )
    power = 2.0
    power_scales = compute_heom_dynamical_scales(
        hierarchy,
        initial_state,
        liouvillian,
        tier_normalized=True,
        tier_loss_power=power,
        **common,
    )

    assert np.isclose(global_scales.constant_loss, expected_global)
    assert np.isclose(tier_scales.constant_loss, expected_tier_average)
    assert np.isclose(
        grouped_scales.constant_loss,
        beta * expected_tiers[0]
        + (1.0 - beta) * np.mean(expected_tiers[1:]),
    )
    power_weights = np.arange(1, hierarchy.L + 2, dtype=np.float64) ** (-power)
    assert np.isclose(
        power_scales.constant_loss,
        np.dot(power_weights, expected_tiers) / power_weights.sum(),
    )
    assert np.isclose(tier_scales.global_constant_loss, expected_global)
    assert np.isclose(global_scales.initial_switch_slope, 0.25)
    expected_linear_scale = 4.0 * np.sqrt(expected_global)
    assert np.isclose(global_scales.correction_scale, expected_linear_scale)
    # The ansatz scale is a property of the dynamics, not of loss weighting.
    assert np.isclose(tier_scales.correction_scale, expected_linear_scale)

    time_constant = 0.75
    exponential_scales = compute_heom_dynamical_scales(
        hierarchy,
        initial_state,
        liouvillian,
        tier_normalized=True,
        time_switch="exponential",
        switch_time_constant=time_constant,
        **common,
    )
    denominator = -np.expm1(-4.0 / time_constant)
    expected_slope = 1.0 / (time_constant * denominator)
    assert np.isclose(
        exponential_scales.initial_switch_slope,
        expected_slope,
    )
    assert np.isclose(
        exponential_scales.correction_scale,
        np.sqrt(expected_global) / expected_slope,
    )


def test_l_const_normalization_uses_the_same_configured_tier_loss():
    hierarchy = make_hierarchy(depth=2)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    initial_state = hierarchy.build_initial_state(
        make_rho0(),
        as_sparse=False,
    )
    real_initial_state = torch.as_tensor(
        q_func.state_to_real(initial_state)[None],
        dtype=torch.float64,
    ).repeat(3, 1)
    zero_derivative = torch.zeros_like(real_initial_state)

    loss_configurations = (
        (False, None, None, None),
        (True, None, None, None),
        (True, 0, 0.75, None),
        (True, None, None, 2.0),
    )
    for (
        tier_normalized,
        lower_tier_cutoff,
        lower_tier_weight,
        tier_loss_power,
    ) in loss_configurations:
        scales = compute_heom_dynamical_scales(
            hierarchy,
            initial_state,
            liouvillian,
            t_start=0.0,
            t_stop=1.0,
            tier_normalized=tier_normalized,
            lower_tier_cutoff=lower_tier_cutoff,
            lower_tier_weight=lower_tier_weight,
            tier_loss_power=tier_loss_power,
            normalization_floor=1e-30,
        )
        objective = HEOMPINNLoss(
            hierarchy,
            liouvillian=liouvillian,
            tier_normalized=tier_normalized,
            lower_tier_cutoff=lower_tier_cutoff,
            lower_tier_weight=lower_tier_weight,
            tier_loss_power=tier_loss_power,
            normalization_loss=scales.effective_constant_loss,
        )

        raw_loss = objective.raw_dynamics_loss(
            real_initial_state,
            zero_derivative,
        )
        normalized_loss = objective.dynamics_loss(
            real_initial_state,
            zero_derivative,
        )
        torch.testing.assert_close(
            raw_loss,
            raw_loss.new_tensor(scales.constant_loss),
        )
        torch.testing.assert_close(
            normalized_loss,
            normalized_loss.new_tensor(1.0),
        )


def test_zero_dynamics_uses_the_normalization_floor_without_nan():
    hierarchy = make_hierarchy(depth=2)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    zero_liouvillian = 0.0 * liouvillian
    initial_state = hierarchy.build_initial_state(
        make_rho0(),
        as_sparse=False,
    )
    floor = 4e-10
    time_span = 3.0
    scales = compute_heom_dynamical_scales(
        hierarchy,
        initial_state,
        zero_liouvillian,
        t_start=1.0,
        t_stop=1.0 + time_span,
        tier_normalized=True,
        normalization_floor=floor,
    )

    assert scales.constant_loss == 0.0
    assert scales.global_constant_loss == 0.0
    assert scales.effective_constant_loss == floor
    assert scales.effective_global_constant_loss == floor
    assert np.isclose(
        scales.correction_scale,
        time_span * np.sqrt(floor),
    )

    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=zero_liouvillian,
        tier_normalized=True,
        normalization_loss=scales.effective_constant_loss,
    )
    state = torch.as_tensor(
        q_func.state_to_real(initial_state)[None],
        dtype=torch.float64,
    )
    normalized_loss = objective.dynamics_loss(state, torch.zeros_like(state))
    assert torch.isfinite(normalized_loss)
    assert normalized_loss.item() == 0.0


def test_partial_minibatches_match_the_full_objective():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
    )
    objective = HEOMPINNLoss(hierarchy, liouvillian=liouvillian)
    times = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    full = objective(model, times)
    first = objective(model, times[:2])
    second = objective(model, times[2:])
    combined = (2 * first + 3 * second) / 5
    torch.testing.assert_close(full, combined)


def test_short_training_and_solution_layout():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    rho0 = make_rho0()
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(6,),
        rho0=rho0,
        t_start=0.0,
        t_stop=0.2,
    )
    objective = HEOMPINNLoss(hierarchy, liouvillian=liouvillian)
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
    assert np.isfinite(result.final.loss)
    np.testing.assert_allclose(solution.t, np.linspace(0.0, 0.2, 3))
    assert solution.y.shape == (model.state_size, 3)
    np.testing.assert_allclose(
        solution.primary_ados,
        solution.primary_ados.conj().transpose(0, 2, 1),
        atol=1e-13,
    )


def test_lbfgs_uses_fixed_full_batch_and_strong_wolfe():
    arguments = build_argument_parser().parse_args(
        ["--resume", "--plot-loss", "--optimizer", "lbfgs"]
    )
    assert arguments.resume
    assert arguments.plot_loss
    assert arguments.optimizer == "lbfgs"

    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=0.2,
        dtype=torch.float64,
    )
    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        dtype=torch.float64,
    )
    optimizer = build_optimizer(model, "lbfgs")
    parameters_before = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    config = TrainingConfig(
        t_start=0.0,
        t_stop=0.2,
        epochs=1,
        collocation_points=5,
        batch_size=2,
        resample_each_epoch=True,
    )

    with (
        patch.object(objective, "forward", wraps=objective.forward) as call,
        patch("builtins.print") as printed,
    ):
        result = train_mlp(
            model,
            objective,
            config,
            optimizer=optimizer,
        )

    expected_times = torch.linspace(0.0, 0.2, 5, dtype=torch.float64)
    assert isinstance(optimizer, torch.optim.LBFGS)
    assert optimizer.defaults["line_search_fn"] == "strong_wolfe"
    assert optimizer.defaults["max_eval"] == MLP.lbfgs_max_eval
    assert optimizer.defaults["tolerance_grad"] == MLP.lbfgs_tolerance_grad
    assert (
        optimizer.defaults["tolerance_change"]
        == MLP.lbfgs_tolerance_change
    )
    assert call.call_count >= 1
    for invocation in call.call_args_list:
        torch.testing.assert_close(invocation.args[1], expected_times)
    expected_parameter_change = max(
        (parameter.detach() - previous).abs().amax().item()
        for parameter, previous in zip(model.parameters(), parameters_before)
    )
    optimizer.zero_grad(set_to_none=True)
    final_loss = objective(model, expected_times)
    final_loss.backward()
    expected_gradient = max(
        parameter.grad.detach().abs().amax().item()
        for parameter in model.parameters()
        if parameter.grad is not None
    )
    assert np.isclose(
        result.final.parameter_change_inf,
        expected_parameter_change,
    )
    assert np.isclose(result.final.gradient_inf, expected_gradient)
    assert np.isclose(result.final.loss, final_loss.detach().item())
    message = printed.call_args.args[0]
    assert "g_inf=" in message
    assert "delta_theta_inf=" in message
    assert "lbfgs_iter=" in message
    assert "evals=" in message
    assert "history=" in message
    assert result.final.lbfgs_iterations is not None
    assert result.final.lbfgs_evaluations is not None
    assert result.final.lbfgs_curvature_pairs is not None
    assert np.isfinite(result.final.loss)


def test_lbfgs_optimizes_residual_sum_but_reports_mean_loss():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=0.2,
        dtype=torch.float64,
    )
    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        dtype=torch.float64,
    )
    optimizer = build_optimizer(model, "lbfgs")
    config = TrainingConfig(
        t_start=0.0,
        t_stop=0.2,
        epochs=1,
        collocation_points=5,
        batch_size=5,
    )
    observed_optimizer_loss = None

    def one_closure_step(closure):
        nonlocal observed_optimizer_loss
        observed_optimizer_loss = closure().detach().item()

    with patch.object(optimizer, "step", side_effect=one_closure_step):
        result = train_mlp(
            model,
            objective,
            config,
            optimizer=optimizer,
            verbose=False,
        )

    times = torch.linspace(0.0, 0.2, 5, dtype=torch.float64)
    mean_loss = objective(model, times).detach().item()
    expected_scale = objective.state_size * times.numel()
    assert np.isclose(result.final.loss, mean_loss)
    assert np.isclose(observed_optimizer_loss, expected_scale * mean_loss)


def test_l_const_normalized_lbfgs_uses_dimensionless_loss_directly():
    hierarchy = make_hierarchy(depth=1)
    liouvillian = hierarchy.build_Liouvillian(normalized=True)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=0.2,
        dtype=torch.float64,
    )
    normalization_loss = 0.125
    objective = HEOMPINNLoss(
        hierarchy,
        liouvillian=liouvillian,
        normalization_loss=normalization_loss,
        dtype=torch.float64,
    )
    optimizer = build_optimizer(model, "lbfgs")
    config = TrainingConfig(
        t_start=0.0,
        t_stop=0.2,
        epochs=1,
        collocation_points=5,
        batch_size=5,
    )
    observed_optimizer_loss = None

    def one_closure_step(closure):
        nonlocal observed_optimizer_loss
        observed_optimizer_loss = closure().detach().item()

    with patch.object(optimizer, "step", side_effect=one_closure_step):
        result = train_mlp(
            model,
            objective,
            config,
            optimizer=optimizer,
            verbose=False,
        )

    times = torch.linspace(0.0, 0.2, 5, dtype=torch.float64)
    normalized_loss = objective(model, times).detach().item()
    assert np.isclose(observed_optimizer_loss, normalized_loss)
    assert np.isclose(result.final.loss, normalized_loss)
    assert np.isclose(
        result.final.raw_loss,
        normalization_loss * normalized_loss,
    )


def test_saved_model_can_be_loaded_for_additional_training():
    hierarchy = make_hierarchy(depth=1)
    model = HEOMMLP(
        hierarchy,
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
    )
    expected = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    with TemporaryDirectory() as directory:
        path = Path(directory) / "mlp_state_dict.pt"
        torch.save(model.state_dict(), path)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        load_saved_model(model, path, torch.device("cpu"))

    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, expected[name])


def test_pretrained_network_loads_across_depths_without_initial_state():
    source = HEOMMLP(
        make_hierarchy(depth=1),
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
        normalize_hierarchy_coordinates=False,
    )
    destination = HEOMMLP(
        make_hierarchy(depth=2),
        hidden_sizes=(5,),
        rho0=make_rho0(),
        t_start=0.0,
        t_stop=1.0,
        normalize_hierarchy_coordinates=False,
    )
    expected_network = {
        name: value.detach().clone()
        for name, value in source.network.state_dict().items()
    }
    destination_initial_state = destination.initial_state.detach().clone()

    with TemporaryDirectory() as directory:
        path = Path(directory) / "mlp_state_dict.pt"
        torch.save(source.state_dict(), path)
        load_pretrained_network(destination, path, torch.device("cpu"))

    for name, value in destination.network.state_dict().items():
        torch.testing.assert_close(value, expected_network[name])
    torch.testing.assert_close(
        destination.initial_state,
        destination_initial_state,
    )
    assert source.initial_state.shape != destination.initial_state.shape


def test_live_loss_plot_uses_log_scale():
    import matplotlib

    matplotlib.use("Agg", force=True)
    plotter = LiveLossPlot(update_every=100, final_epoch=201)
    plotter(EpochRecord(epoch=1, loss=10.0))
    plotter(EpochRecord(epoch=2, loss=1.0))
    plotter(EpochRecord(epoch=100, loss=1.0))
    plotter(EpochRecord(epoch=101, loss=0.5))
    plotter(EpochRecord(epoch=201, loss=0.0))
    plotter.finish(show=False)

    assert plotter.axis.get_yscale() == "log"
    np.testing.assert_array_equal(plotter.line.get_xdata(), [1, 100, 201])
    assert np.all(np.asarray(plotter.line.get_ydata()) > 0.0)
    plotter.plt.close(plotter.figure)
