"""Quantum-Enhanced Classification Head (paper Sec. 3.4).

Circuit on an ``n_qubits`` register (default 12):

1. Spatial register (first ``n_pos`` qubits): Hadamard, then an RZ phase that
   encodes one normalised tumor-centroid coordinate per qubit.
2. Feature register (remaining qubits): U3(theta, phi, lambda) per qubit, each
   triplet drawn from three consecutive elements of the projected vector.
3. Controlled-U3 gates from each spatial qubit to a feature qubit (angles are
   the remaining 3 * n_pos projected elements), followed by a recurrent
   controlled-Z ring that entangles the two registers. Together steps 2-3
   consume exactly 3 * n_qubits encoded angles, one triplet per qubit (Eq. 9).
4. ``n_layers`` parameterised quantum-convolution layers. Each layer applies
   ``n_kernels`` parallel kernels on disjoint qubit blocks; a kernel is a U3 on
   every qubit in the block followed by a CZ chain (Eq. 10). The block
   partition is shifted by half a block in odd layers so information crosses
   kernel borders. Trainable parameters: n_layers * n_qubits * 3 (= 72).
5. Pauli-X read-out z_g = <Psi|X_g|Psi>, then softmax(W_a z + b_a) (Eq. 11).

Two simulation back-ends are provided:

* ``torch`` - an exact batched state-vector simulator written in PyTorch
  (2^12 = 4096 amplitudes). Gradients come either from autograd
  (``diff_method="backprop"``) or from the parameter-shift rule
  (``diff_method="parameter-shift"``); both give the same values.
* ``pennylane`` - the same circuit on ``default.qubit`` through
  ``qml.qnn.TorchLayer`` (paper setting: PennyLane 0.35, parameter-shift).
"""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn

# --------------------------------------------------------------------------
# State-vector primitives
# --------------------------------------------------------------------------

_H = torch.tensor([[1, 1], [1, -1]], dtype=torch.cfloat) / math.sqrt(2)


def u3_matrix(theta, phi, lam):
    """Batched U3 matrices. Inputs broadcast to a common shape (...,)."""
    theta, phi, lam = torch.broadcast_tensors(theta, phi, lam)
    c = torch.cos(theta / 2).to(torch.cfloat)
    s = torch.sin(theta / 2).to(torch.cfloat)
    e_phi = torch.polar(torch.ones_like(phi), phi)
    e_lam = torch.polar(torch.ones_like(lam), lam)
    row0 = torch.stack([c, -e_lam * s], -1)
    row1 = torch.stack([e_phi * s, e_phi * e_lam * c], -1)
    return torch.stack([row0, row1], -2)  # (..., 2, 2)


def rz_matrix(angle):
    zero = torch.zeros_like(angle)
    e_m = torch.polar(torch.ones_like(angle), -angle / 2)
    e_p = torch.polar(torch.ones_like(angle), angle / 2)
    row0 = torch.stack([e_m, zero.to(torch.cfloat)], -1)
    row1 = torch.stack([zero.to(torch.cfloat), e_p], -1)
    return torch.stack([row0, row1], -2)


def apply_1q(state: torch.Tensor, gate: torch.Tensor, q: int, n: int) -> torch.Tensor:
    """state: (B, 2^n); gate: (2, 2) or (B, 2, 2)."""
    b = state.shape[0]
    psi = state.reshape(b, 2 ** q, 2, 2 ** (n - q - 1))
    if gate.dim() == 2:
        out = torch.einsum("ij,bajc->baic", gate, psi)
    else:
        out = torch.einsum("bij,bajc->baic", gate, psi)
    return out.reshape(b, -1)


def _bit_mask(q: int, n: int, device) -> torch.Tensor:
    idx = torch.arange(2 ** n, device=device)
    return ((idx >> (n - 1 - q)) & 1).bool()


def apply_controlled_1q(state, gate, control, target, n):
    applied = apply_1q(state, gate, target, n)
    mask = _bit_mask(control, n, state.device)
    return torch.where(mask.unsqueeze(0), applied, state)


def cz_sign(pairs: Sequence[tuple[int, int]], n: int, device) -> torch.Tensor:
    sign = torch.ones(2 ** n, device=device)
    for a, b in pairs:
        both = _bit_mask(a, n, device) & _bit_mask(b, n, device)
        sign = torch.where(both, -sign, sign)
    return sign.to(torch.cfloat)


def expval_x(state: torch.Tensor, n: int) -> torch.Tensor:
    """<X_g> for every qubit g. Returns (B, n) real."""
    b = state.shape[0]
    outs = []
    for g in range(n):
        psi = state.reshape(b, 2 ** g, 2, 2 ** (n - g - 1))
        outs.append(2 * (psi[:, :, 0].conj() * psi[:, :, 1]).real.sum((1, 2)))
    return torch.stack(outs, 1)


# --------------------------------------------------------------------------
# Circuit definition
# --------------------------------------------------------------------------

class CircuitSpec:
    def __init__(self, n_qubits=12, n_pos=3, n_layers=2, n_kernels=4):
        assert n_qubits % n_kernels == 0, "n_qubits must be divisible by n_kernels"
        assert 0 < n_pos < n_qubits
        self.n = n_qubits
        self.n_pos = n_pos
        self.n_feat = n_qubits - n_pos
        self.n_layers = n_layers
        self.n_kernels = n_kernels
        self.block = n_qubits // n_kernels
        # control (spatial qubit) -> target (feature qubit)
        stride = max(self.n_feat // n_pos, 1)
        self.cu3_pairs = [(i, n_pos + (i * stride) % self.n_feat) for i in range(n_pos)]
        self.ring = [(k, (k + 1) % n_qubits) for k in range(n_qubits)] if n_qubits > 2 else [(0, 1)]

    @property
    def n_angles(self) -> int:
        return 3 * self.n

    def kernels(self, layer: int):
        offset = (self.block // 2) * (layer % 2)
        order = [(q + offset) % self.n for q in range(self.n)]
        return [order[k * self.block:(k + 1) * self.block] for k in range(self.n_kernels)]

    def kernel_cz_pairs(self, layer: int):
        pairs = []
        for ker in self.kernels(layer):
            pairs += [(ker[i], ker[i + 1]) for i in range(len(ker) - 1)]
        return pairs


def simulate(spec: CircuitSpec, angles: torch.Tensor, coords: torch.Tensor,
             weights: torch.Tensor) -> torch.Tensor:
    """Run the circuit.

    angles:  (B, 3n) encoded rotation angles
    coords:  (B, n_pos) coordinates in [-1, 1]
    weights: (L, n, 3) trainable U3 angles
    returns: (B, n) Pauli-X expectations
    """
    n, dev = spec.n, angles.device
    b = angles.shape[0]
    state = torch.zeros(b, 2 ** n, dtype=torch.cfloat, device=dev)
    state[:, 0] = 1
    h = _H.to(dev)
    # 1) spatial register
    for q in range(spec.n_pos):
        state = apply_1q(state, h, q, n)
        state = apply_1q(state, rz_matrix(math.pi * coords[:, q]), q, n)
    # 2) feature register U3 encoding
    a = angles.reshape(b, n, 3)
    for i in range(spec.n_feat):
        q = spec.n_pos + i
        state = apply_1q(state, u3_matrix(a[:, i, 0], a[:, i, 1], a[:, i, 2]), q, n)
    # 3) controlled-U3 bridge + recurrent CZ ring
    for j, (c, t) in enumerate(spec.cu3_pairs):
        k = spec.n_feat + j
        state = apply_controlled_1q(state, u3_matrix(a[:, k, 0], a[:, k, 1], a[:, k, 2]), c, t, n)
    state = state * cz_sign(spec.ring, n, dev)
    # 4) parameterised quantum-convolution layers
    for layer in range(spec.n_layers):
        for q in range(n):
            w = weights[layer, q]
            state = apply_1q(state, u3_matrix(w[0], w[1], w[2]), q, n)
        state = state * cz_sign(spec.kernel_cz_pairs(layer), n, dev)
    # 5) Pauli-X read-out
    return expval_x(state, n)


# --------------------------------------------------------------------------
# Parameter-shift autograd function
# --------------------------------------------------------------------------

_S = math.pi / 2
# four-term rule for controlled rotations (Anselmetti et al., NJP 2021)
_C1 = (math.sqrt(2) + 1) / (4 * math.sqrt(2))
_C2 = (math.sqrt(2) - 1) / (4 * math.sqrt(2))


class _ParameterShift(torch.autograd.Function):
    @staticmethod
    def forward(ctx, angles, coords, weights, spec):
        ctx.spec = spec
        ctx.save_for_backward(angles, coords, weights)
        with torch.no_grad():
            return simulate(spec, angles, coords, weights)

    @staticmethod
    def backward(ctx, gz):
        angles, coords, weights = ctx.saved_tensors
        spec = ctx.spec
        run = lambda a, w: simulate(spec, a, coords, w)  # noqa: E731
        g_ang = g_w = None
        with torch.no_grad():
            if ctx.needs_input_grad[0]:
                g_ang = torch.zeros_like(angles)
                controlled = set(range(3 * spec.n_feat, 3 * spec.n))
                for i in range(angles.shape[1]):
                    e = torch.zeros_like(angles)
                    e[:, i] = 1
                    if i in controlled:
                        d = (_C1 * (run(angles + _S * e, weights) - run(angles - _S * e, weights))
                             - _C2 * (run(angles + 3 * _S * e, weights) - run(angles - 3 * _S * e, weights)))
                    else:
                        d = 0.5 * (run(angles + _S * e, weights) - run(angles - _S * e, weights))
                    g_ang[:, i] = (gz * d).sum(1)
            if ctx.needs_input_grad[2]:
                g_w = torch.zeros_like(weights)
                for idx in torch.cartesian_prod(*[torch.arange(s) for s in weights.shape]):
                    idx = tuple(idx.tolist())
                    e = torch.zeros_like(weights)
                    e[idx] = _S
                    d = 0.5 * (run(angles, weights + e) - run(angles, weights - e))
                    g_w[idx] = (gz * d).sum()
        return g_ang, None, g_w, None


# --------------------------------------------------------------------------
# Modules
# --------------------------------------------------------------------------

class TorchQuantumCircuit(nn.Module):
    def __init__(self, spec: CircuitSpec, diff_method="backprop"):
        super().__init__()
        assert diff_method in ("backprop", "parameter-shift")
        self.spec = spec
        self.diff_method = diff_method
        self.weights = nn.Parameter(0.1 * torch.randn(spec.n_layers, spec.n, 3))

    def forward(self, angles, coords):
        angles = angles.float()
        coords = coords.float().detach()
        with torch.autocast(device_type=angles.device.type, enabled=False):
            if self.diff_method == "parameter-shift":
                return _ParameterShift.apply(angles, coords, self.weights, self.spec)
            return simulate(self.spec, angles, coords, self.weights)


class PennyLaneQuantumCircuit(nn.Module):
    """Same circuit on PennyLane ``default.qubit`` via ``qml.qnn.TorchLayer``."""

    def __init__(self, spec: CircuitSpec, diff_method="parameter-shift",
                 device_name="default.qubit"):
        super().__init__()
        import pennylane as qml

        self.spec = spec
        dev = qml.device(device_name, wires=spec.n)
        n, n_pos, n_feat = spec.n, spec.n_pos, spec.n_feat

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            ang = inputs[..., : 3 * n]
            crd = inputs[..., 3 * n:]
            for q in range(n_pos):
                qml.Hadamard(wires=q)
                qml.RZ(math.pi * crd[..., q], wires=q)
            for i in range(n_feat):
                qml.U3(ang[..., 3 * i], ang[..., 3 * i + 1], ang[..., 3 * i + 2], wires=n_pos + i)
            for j, (c, t) in enumerate(spec.cu3_pairs):
                k = n_feat + j
                qml.ctrl(qml.U3, control=c)(ang[..., 3 * k], ang[..., 3 * k + 1], ang[..., 3 * k + 2], wires=t)
            for a, b in spec.ring:
                qml.CZ(wires=[a, b])
            for layer in range(spec.n_layers):
                for q in range(n):
                    qml.U3(weights[layer, q, 0], weights[layer, q, 1], weights[layer, q, 2], wires=q)
                for a, b in spec.kernel_cz_pairs(layer):
                    qml.CZ(wires=[a, b])
            return [qml.expval(qml.PauliX(g)) for g in range(n)]

        self.layer = qml.qnn.TorchLayer(circuit, {"weights": (spec.n_layers, n, 3)})

    @property
    def weights(self):
        return self.layer.weights

    def forward(self, angles, coords):
        x = torch.cat([angles.float(), coords.float().detach()], dim=-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self.layer(x).float()


class QuantumClassificationHead(nn.Module):
    """Projection -> quantum circuit -> Pauli-X vector z -> linear classifier."""

    def __init__(self, in_dim: int, num_classes: int = 4, n_qubits=12, n_pos=3,
                 n_layers=2, n_kernels=4, hidden_dims: Sequence[int] = (2048, 1920),
                 dropout=0.1, backend="torch", diff_method="backprop"):
        super().__init__()
        self.spec = CircuitSpec(n_qubits, n_pos, n_layers, n_kernels)
        dims = [in_dim, *hidden_dims]
        layers = []
        for i in range(len(dims) - 1):
            layers += [nn.Linear(dims[i], dims[i + 1]), nn.LayerNorm(dims[i + 1]),
                       nn.GELU(), nn.Dropout(dropout)]
        layers.append(nn.Linear(dims[-1], self.spec.n_angles))
        self.project = nn.Sequential(*layers)
        if backend == "torch":
            self.circuit = TorchQuantumCircuit(self.spec, diff_method)
        elif backend == "pennylane":
            self.circuit = PennyLaneQuantumCircuit(self.spec, diff_method)
        else:
            raise ValueError(f"unknown quantum backend {backend!r}")
        self.readout = nn.Linear(n_qubits, num_classes)  # W_a, b_a

    def forward(self, f_cls: torch.Tensor, coords: torch.Tensor):
        angles = math.pi * torch.tanh(self.project(f_cls))
        z = self.circuit(angles, coords)
        logits = self.readout(z.to(self.readout.weight.dtype))
        return logits, z

    def quantum_parameter_count(self) -> int:
        return self.spec.n_layers * self.spec.n * 3
