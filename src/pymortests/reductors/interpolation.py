# This file is part of the pyMOR project (https://www.pymor.org).
# Copyright pyMOR developers and contributors. All rights reserved.
# License: BSD 2-Clause License (https://opensource.org/licenses/BSD-2-Clause)

import numpy as np
import pytest

from pymor.algorithms.to_matrix import to_matrix
from pymor.models.iosys import LTIModel
from pymor.models.transfer_function import TransferFunction
from pymor.reductors.interpolation import LTIBHIReductor, TFBHIReductor

pytestmark = pytest.mark.builtin


@pytest.mark.parametrize(('dim_input', 'dim_output'), [(1, 1), (1, 2), (2, 1), (2, 3)])
@pytest.mark.parametrize('sigma', [np.array([0., 1., 2.]), np.array([0., 1. + 1.j, 1. - 1.j])],
                         ids=['real', 'complex'])
@pytest.mark.parametrize('projection', ['orth', 'biorth'])
def test_ltibhi_tangential_value_and_derivative_interpolation(dim_input, dim_output, sigma, projection):
    rng = np.random.default_rng(391)
    A = -np.diag(np.arange(1., 9.))
    B = rng.standard_normal((8, dim_input))
    C = rng.standard_normal((dim_output, 8))
    fom = LTIModel.from_matrices(A, B, C)
    b = rng.standard_normal((3, dim_input)) * np.arange(1, dim_input + 1)
    c = rng.standard_normal((3, dim_output)) * np.arange(1, dim_output + 1)
    if np.iscomplexobj(sigma):
        b = b.astype(complex)
        c = c.astype(complex)
        b[1] += 1.j * rng.standard_normal(dim_input)
        c[1] += 1.j * rng.standard_normal(dim_output)
        b[2] = b[1].conj()
        c[2] = c[1].conj()

    rom = LTIBHIReductor(fom).reduce(sigma, b, c, projection=projection)

    for node, right, left in zip(sigma, b, c, strict=True):
        full_value = fom.transfer_function.eval_tf(node)
        reduced_value = rom.transfer_function.eval_tf(node)
        full_derivative = fom.transfer_function.eval_dtf(node)
        reduced_derivative = rom.transfer_function.eval_dtf(node)
        np.testing.assert_allclose(reduced_value @ right, full_value @ right, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(left @ reduced_value, left @ full_value, rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(left @ reduced_derivative @ right, left @ full_derivative @ right,
                                   rtol=1e-10, atol=1e-12)


def test_tfbhi_value_and_derivative_interpolation():
    fom = TransferFunction(
        1,
        1,
        lambda s: np.array([[sum((i + 1) / (s + i + 1) for i in range(5))]]),
        dtf=lambda s: np.array([[-sum((i + 1) / (s + i + 1)**2 for i in range(5))]]),
    )
    sigma = np.array([0, 1j, -1j])
    directions = np.ones((3, 1))
    values = np.array([fom.eval_tf(node).item() for node in sigma])
    scale = 1 / np.sqrt(2)
    transformation = np.array([
        [1, 0, 0],
        [0, scale, scale],
        [0, -1j * scale, 1j * scale],
    ])

    rom = TFBHIReductor(fom).reduce(sigma, directions, directions)

    assert np.allclose(to_matrix(rom.B), (transformation @ values[:, np.newaxis]).real)
    assert np.allclose(to_matrix(rom.C), (values[np.newaxis] @ transformation.conj().T).real)
    for node in sigma:
        assert np.allclose(rom.transfer_function.eval_tf(node), fom.eval_tf(node))
        assert np.allclose(rom.transfer_function.eval_dtf(node), fom.eval_dtf(node))
