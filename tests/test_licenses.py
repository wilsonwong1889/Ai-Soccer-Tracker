"""Enforce docs/decisions.md T10: nothing installed may be copyleft or non-commercial.

This checks package metadata, so it catches a new dependency (or a new
transitive one) the moment it's installed. Model weights and datasets are
tracked by hand in docs/register.md.
"""

import importlib.metadata as md

import pytest

BLOCKED = ("agpl", "gpl", "affero", "non-commercial", "noncommercial", "cc-by-nc",
           "hippocratic", "sspl", "business source")
# Reviewed and fine for commercial use (see docs/register.md).
REVIEWED = {
    # NVIDIA's CUDA runtime libraries: proprietary but free to use and
    # redistribute. Installed on Linux only, as PyTorch's GPU dependencies.
    "cuda-toolkit", "nvidia-cublas", "nvidia-cuda-cupti", "nvidia-cuda-nvrtc",
    "nvidia-cuda-runtime", "nvidia-cudnn-cu13", "nvidia-cufft", "nvidia-cufile",
    "nvidia-curand", "nvidia-cusolver", "nvidia-cusparse", "nvidia-cusparselt-cu13",
    "nvidia-nccl-cu13", "nvidia-nvjitlink", "nvidia-nvshmem-cu13", "nvidia-nvtx",
    # BSD, but bundles libgfortran: GPL-3.0 with the GCC Runtime Library
    # Exception, which allows use in closed-source software.
    "scipy",
    # PSF-style, but bundles FreeType under "FTL OR GPL-2.0": we take the FTL.
    "matplotlib",
    # This project itself (proprietary, no license metadata).
    "soccer-recorder",
}


def licence_text(dist) -> str:
    meta = dist.metadata
    parts = [meta.get("License-Expression") or "", meta.get("License") or ""]
    parts += [c for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return " ".join(parts).lower()


def installed():
    return sorted(md.distributions(), key=lambda d: d.metadata["Name"].lower())


@pytest.mark.parametrize("dist", installed(), ids=lambda d: d.metadata["Name"])
def test_dependency_licence_is_free_for_commercial_use(dist):
    name = dist.metadata["Name"].lower()
    if name in REVIEWED:
        return
    text = licence_text(dist)
    # LGPL is weaker than GPL but still needs a review before it's allowed.
    blocked = [word for word in BLOCKED if word in text]
    assert not blocked, f"{name} looks {blocked[0].upper()}-licensed: review before using it"
    assert text.strip(), f"{name} declares no licence: check it and add it to REVIEWED"
