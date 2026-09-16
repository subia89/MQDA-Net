import math

import pytest
import torch

from mqda.models.quantum import (CircuitSpec, QuantumClassificationHead, TorchQuantumCircuit,
                                 apply_1q, expval_x, u3_matrix)


def test_single_qubit_u3_expectation():
    state = torch.zeros(1, 2, dtype=torch.cfloat)
    state[:, 0] = 1
    th, ph, lam = torch.tensor(0.7), torch.tensor(0.3), torch.tensor(-1.1)
    out = apply_1q(state, u3_matrix(th, ph, lam), 0, 1)
    # U3|0> = cos(th/2)|0> + e^{i ph} sin(th/2)|1>  ->  <X> = sin(th) cos(ph)
    assert torch.allclose(expval_x(out, 1), (torch.sin(th) * torch.cos(ph)).view(1, 1), atol=1e-6)


def test_u3_is_unitary():
    u = u3_matrix(torch.rand(5), torch.rand(5), torch.rand(5))
    eye = u @ u.conj().transpose(-1, -2)
    assert torch.allclose(eye, torch.eye(2, dtype=torch.cfloat).expand_as(eye), atol=1e-6)


def test_expectations_bounded_and_trainable_count():
    spec = CircuitSpec(12, 3, 2, 4)
    circ = TorchQuantumCircuit(spec)
    z = circ(torch.randn(3, 36), torch.rand(3, 3) * 2 - 1)
    assert z.shape == (3, 12)
    assert (z.abs() <= 1 + 1e-5).all()
    assert circ.weights.numel() == 72  # layers x qubits x 3


def test_parameter_shift_matches_autograd():
    torch.manual_seed(0)
    spec = CircuitSpec(6, 2, 2, 3)
    circ = TorchQuantumCircuit(spec)
    a = torch.randn(2, spec.n_angles)
    c = torch.rand(2, 2) * 2 - 1
    w = torch.randn(6)
    a1, a2 = a.clone().requires_grad_(), a.clone().requires_grad_()
    circ.diff_method = "backprop"
    (circ(a1, c) @ w).sum().backward()
    g1 = circ.weights.grad.clone()
    circ.weights.grad = None
    circ.diff_method = "parameter-shift"
    (circ(a2, c) @ w).sum().backward()
    assert torch.allclose(g1, circ.weights.grad, atol=1e-4)
    # includes the four-term rule for the controlled-U3 angles
    assert torch.allclose(a1.grad, a2.grad, atol=1e-4)


def test_pennylane_backend_matches_torch():
    pytest.importorskip("pennylane")
    from mqda.models.quantum import PennyLaneQuantumCircuit

    spec = CircuitSpec(6, 2, 2, 3)
    tq = TorchQuantumCircuit(spec)
    pq = PennyLaneQuantumCircuit(spec, diff_method="backprop")
    with torch.no_grad():
        pq.layer.weights.copy_(tq.weights)
    a, c = torch.randn(2, spec.n_angles), torch.rand(2, 2) * 2 - 1
    assert torch.allclose(tq(a, c), pq(a, c), atol=1e-4)


def test_head_output_and_gradients():
    head = QuantumClassificationHead(20, 4, n_qubits=8, n_pos=2, hidden_dims=[16])
    f = torch.randn(3, 20, requires_grad=True)
    logits, z = head(f, torch.zeros(3, 3))
    assert logits.shape == (3, 4) and z.shape == (3, 8)
    logits.sum().backward()
    assert f.grad is not None and head.circuit.weights.grad is not None
    assert head.quantum_parameter_count() == 2 * 8 * 3
    assert math.isfinite(float(logits.detach().sum()))
