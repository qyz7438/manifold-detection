r"""Complex-valued linear/MLP layers and a spectral manifold autoencoder.

The modules operate on PyTorch complex tensors (``torch.cfloat`` /
``torch.cdouble``) and are designed so that ``ComplexSpectralManifold`` is
initialized as an approximate identity mapping when ``in_dim == latent_dim``.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _complex_uniform_(tensor: torch.Tensor, a: float = 0.0, b: float = 1.0) -> torch.Tensor:
    """Fill a complex tensor with uniform real/imag parts (in-place)."""
    with torch.no_grad():
        tensor.real.uniform_(a, b)
        tensor.imag.uniform_(a, b)
    return tensor


def _random_unitary_matrix(size: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    r"""Return a random unitary matrix of shape ``(size, size)``.

    Uses the QR decomposition of a random complex Gaussian matrix. The
    diagonal phase of ``R`` is absorbed so that the returned matrix is
    uniformly distributed on the unitary group (Haar measure).
    """
    A = torch.randn(size, size, dtype=dtype, device=device)
    Q, R = torch.linalg.qr(A)
    # Absorb the diagonal phase of R to obtain a Haar unitary.
    phases = torch.diag(R).angle()
    Q = Q * torch.exp(-1j * phases).unsqueeze(0)
    return Q


class modReLU(nn.Module):
    r"""modReLU activation for complex-valued features.

    Given a complex input :math:`z` and a real bias :math:`b` per feature,
    returns

    .. math::
        \mathrm{modReLU}(z, b) = \frac{z}{|z|} \, \max(0, |z| + b).

    When the bias is initialized to zero and the input is non-zero, modReLU
    acts as the identity at initialization.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))
        self._eps = 1e-12

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        mag = z.abs().clamp_min(self._eps)
        # Broadcast bias over arbitrary leading batch dimensions.
        bias_shape = [1] * (z.ndim - 1) + [-1]
        bias = self.bias.view(*bias_shape)
        scale = F.relu(mag + bias)
        return z * (scale / mag)


class ComplexLinear(nn.Module):
    r"""Fully-connected complex linear layer.

    Implements :math:`y = x W^{\top} + b` using complex arithmetic. The
    weight and bias tensors have ``dtype=torch.cfloat`` by default.

    Args:
        in_features: size of each input sample.
        out_features: size of each output sample.
        bias: whether to include a learnable bias term.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=torch.cfloat))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, dtype=torch.cfloat))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Glorot-like initialization for complex weights."""
        std = 1.0 / math.sqrt(self.in_features)
        _complex_uniform_(self.weight, -std, std)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @classmethod
    def from_real(cls, real: torch.Tensor, imag: torch.Tensor, bias: bool = True) -> "ComplexLinear":
        """Factory method building a complex weight from real/imag parts.

        Args:
            real: real part of shape ``(out_features, in_features)``.
            imag: imaginary part of shape ``(out_features, in_features)``.
            bias: whether to add a zero-initialized complex bias.

        Returns:
            A ``ComplexLinear`` layer whose weight is ``real + 1j * imag``.
        """
        if real.shape != imag.shape:
            raise ValueError("real and imag must have the same shape")
        out_features, in_features = real.shape
        layer = cls(in_features, out_features, bias=bias)
        with torch.no_grad():
            layer.weight.copy_(torch.complex(real, imag))
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        return layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class ComplexMLP(nn.Module):
    r"""Two-layer complex MLP with modReLU activation.

    Args:
        in_features: input dimension.
        out_features: output dimension.
        hidden_features: hidden dimension. Defaults to ``in_features``.
        identity_init: if ``True``, the layer is initialized so that
            ``forward(x) \approx x`` when ``in_features == out_features == hidden_features``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: int | None = None,
        identity_init: bool = False,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = hidden_features or in_features

        self.fc1 = ComplexLinear(in_features, self.hidden_features)
        self.activation = modReLU(self.hidden_features)
        self.fc2 = ComplexLinear(self.hidden_features, out_features)

        if identity_init:
            self._init_identity()

    def _init_identity(self) -> None:
        r"""Initialize weights so that ``forward(x) \approx x`` at start.

        For the square case we use a random unitary matrix :math:`Q`:

        .. math::
            W_1 = Q, \quad W_2 = Q^{H}, \quad \text{bias}=0.

        When dimensions differ we fall back to a best-effort sub-unitary
        initialization that preserves as much information as possible.
        """
        dtype = self.fc1.weight.dtype
        device = self.fc1.weight.device

        square = self.in_features == self.hidden_features == self.out_features
        if square:
            size = self.in_features
            Q = _random_unitary_matrix(size, dtype, device)
            QH = Q.conj().t()
            with torch.no_grad():
                self.fc1.weight.copy_(Q)
                self.fc2.weight.copy_(QH)
                if self.fc1.bias is not None:
                    nn.init.zeros_(self.fc1.bias)
                if self.fc2.bias is not None:
                    nn.init.zeros_(self.fc2.bias)
                nn.init.zeros_(self.activation.bias)
            return

        # Non-square best-effort: build a unitary of the largest dimension and
        # carve out compatible sub-matrices so that W2 @ W1 is a projection.
        max_dim = max(self.in_features, self.hidden_features, self.out_features)
        Q = _random_unitary_matrix(max_dim, dtype, device)
        QH = Q.conj().t()

        w1 = torch.zeros(self.hidden_features, self.in_features, dtype=dtype, device=device)
        w2 = torch.zeros(self.out_features, self.hidden_features, dtype=dtype, device=device)

        # Copy top-left blocks from Q / QH so that W2 @ W1 \approx projection.
        h1, w1_in = w1.shape
        h2, w2_hid = w2.shape
        block_h = min(h1, h2)
        block_w = min(w1_in, w2_hid, block_h)

        if block_w > 0:
            w1[:block_h, :block_w] = Q[:block_h, :block_w]
            if block_h == self.hidden_features:
                w2[:, :block_h] = QH[: self.out_features, :block_h]
            else:
                w2[:block_h, :block_h] = QH[:block_h, :block_h]

        with torch.no_grad():
            self.fc1.weight.copy_(w1)
            self.fc2.weight.copy_(w2)
            if self.fc1.bias is not None:
                nn.init.zeros_(self.fc1.bias)
            if self.fc2.bias is not None:
                nn.init.zeros_(self.fc2.bias)
            nn.init.zeros_(self.activation.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


class ComplexSpectralManifold(nn.Module):
    r"""Autoencoder that embeds complex spectral coefficients onto a latent manifold.

    The encoder maps :math:`F \in \mathbb{C}^{d}` to
    :math:`z \in \mathbb{C}^{k}`; the decoder maps back. When
    ``in_dim == latent_dim`` and ``hidden_dim == in_dim`` the module is
    initialized so that ``decode(encode(F)) \approx F``.

    Args:
        in_dim: dimensionality of the input spectral vector.
        latent_dim: dimensionality of the latent manifold.
        hidden_dim: hidden width of the encoder/decoder MLPs. Defaults to
            ``in_dim`` to allow an exact identity initialization.
    """

    def __init__(self, in_dim: int, latent_dim: int, hidden_dim: int | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim or in_dim

        self.encoder = ComplexMLP(
            in_dim, latent_dim, hidden_features=self.hidden_dim, identity_init=True
        )
        self.decoder = ComplexMLP(
            latent_dim, in_dim, hidden_features=self.hidden_dim, identity_init=True
        )

    def encode(self, F: torch.Tensor) -> torch.Tensor:
        """Encode a complex spectral tensor to latent coordinates.

        Args:
            F: tensor of shape ``(..., in_dim)`` with complex dtype.

        Returns:
            Latent tensor of shape ``(..., latent_dim)``.
        """
        if not torch.is_complex(F):
            raise ValueError("ComplexSpectralManifold expects a complex-valued input")
        return self.encoder(F)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent coordinates back to spectral space.

        Args:
            z: tensor of shape ``(..., latent_dim)`` with complex dtype.

        Returns:
            Reconstructed tensor of shape ``(..., in_dim)``.
        """
        if not torch.is_complex(z):
            raise ValueError("ComplexSpectralManifold latent variable must be complex-valued")
        return self.decoder(z)

    def forward(self, F: torch.Tensor) -> torch.Tensor:
        """Autoencoder forward: encode then decode.

        Args:
            F: tensor of shape ``(..., in_dim)`` with complex dtype.

        Returns:
            Reconstruction of shape ``(..., in_dim)``.
        """
        return self.decode(self.encode(F))

    @staticmethod
    def split_magnitude_phase(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Split a complex latent tensor into magnitude and phase.

        Returns:
            ``(rho, theta)`` where ``rho = |z|`` and ``theta = arg(z)``.
        """
        return z.abs(), z.angle()

    @staticmethod
    def combine_magnitude_phase(rho: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Combine magnitude and phase into a complex latent tensor.

        Returns:
            ``rho * exp(1j * theta)``.
        """
        return rho * torch.exp(1j * theta)
