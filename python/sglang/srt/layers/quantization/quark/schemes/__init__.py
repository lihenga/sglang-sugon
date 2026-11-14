# SPDX-License-Identifier: Apache-2.0

from importlib import import_module

from .quark_scheme import QuarkLinearScheme, QuarkMoEScheme

_SCHEMES = {
    "QuarkW4A4MXFP4": "quark_w4a4_mxfp4",
    "QuarkW4A4MXFp4MoE": "quark_w4a4_mxfp4_moe",
    "QuarkW4A8MXFp4MoE": "quark_w4a8_mxfp4_moe",
    "QuarkW8A8Fp8": "quark_w8a8_fp8",
    "QuarkW8A8FP8MoE": "quark_w8a8_fp8_moe",
}


def __getattr__(name):
    if name not in _SCHEMES:
        raise AttributeError(name)
    # An unused quantization scheme must not require newer optional GPU kernels.
    value = getattr(import_module(f"{__name__}.{_SCHEMES[name]}"), name)
    globals()[name] = value
    return value


__all__ = [
    "QuarkLinearScheme",
    "QuarkMoEScheme",
    "QuarkW4A4MXFP4",
    "QuarkW8A8Fp8",
    "QuarkW4A4MXFp4MoE",
    "QuarkW4A8MXFp4MoE",
    "QuarkW8A8FP8MoE",
]
