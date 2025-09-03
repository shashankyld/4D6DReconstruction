from __future__ import annotations

import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "src"))
from forward_renderer import (
    rgb_to_srgb,
    srgb_to_rgb,
    distribution_ggx,
    geometry_ggx,
    fresnel_schlick,
    brdf,
)


def test_rgb_to_srgb():
    rgb = torch.linspace(0, 1, 1000)
    srgb = rgb_to_srgb(rgb)
    rgb_rec = srgb_to_rgb(srgb)

    assert torch.allclose(rgb, rgb_rec)


def test_distribution_ggx():
    NdotH = torch.distributions.Uniform(0, 1).sample([100, 3])
    roughness = torch.zeros((100, 1))

    D = distribution_ggx(NdotH, roughness)

    assert torch.allclose(D, torch.zeros_like(D))

    NdotH = torch.distributions.Uniform(0, 1).sample([100, 3])
    roughness = torch.ones((100, 1))

    D = distribution_ggx(NdotH, roughness)

    assert torch.allclose(D, torch.full_like(D, 1.0 / torch.pi))

    NdotH = torch.distributions.Uniform(0, 1).sample([100, 3])
    roughness = torch.distributions.Uniform(0, 1).sample([100, 3])

    D = distribution_ggx(NdotH, roughness)

    assert torch.all(D > 0.0)

    NdotH = torch.distributions.Uniform(0, 1).sample([100, 3])
    roughness = torch.full((100, 3), 0.8)

    D = distribution_ggx(NdotH, roughness)
    expected = 0.4096 / (torch.pi * (NdotH**2 * (-0.5904) + 1.0) ** 2)

    assert torch.allclose(D, expected)


def test_geometry_schlick_ggx():
    roughness = torch.distributions.Uniform(0, 1).sample([100, 3])
    NdotV = torch.zeros((100, 1))

    G = geometry_ggx(NdotV, roughness)

    assert torch.all(torch.isfinite(G))

    NdotV = torch.ones((100, 1))

    G = geometry_ggx(NdotV, roughness)

    assert torch.allclose(G, torch.zeros_like(G), 1e-5)

    roughness = torch.distributions.Uniform(0, 1).sample([100, 3])
    NdotV = torch.distributions.Uniform(0, 1).sample([100, 3])

    G = geometry_ggx(NdotV, roughness)

    assert G.min() >= 0.0


def test_fresnel_schlick():
    F0 = torch.distributions.Uniform(0, 1).sample([2, 3])
    VdotH = torch.tensor([0.0, 1.0]).reshape(-1, 1)

    F = fresnel_schlick(F0, VdotH)

    assert torch.allclose(F[0], torch.ones_like(F[0]))
    assert torch.allclose(F[1], F0[1])

    F0 = torch.distributions.Uniform(0, 1).sample([100, 3])
    VdotH = torch.distributions.Uniform(0, 1).sample([100, 3])

    F = fresnel_schlick(F0, VdotH)

    assert F.min() >= 0.0
    assert F.max() <= 1.0

    F0 = torch.full((100, 3), 0.3)
    VdotH = torch.distributions.Uniform(0, 1).sample([100, 3])

    F = fresnel_schlick(F0, VdotH)
    expected = 0.3 + 0.7 * (1.0 - VdotH) ** 5.0

    assert torch.allclose(F, expected)


def test_bsdf():
    light_dirs = torch.distributions.Uniform(-1, 1).sample([100, 3])
    light_dirs = torch.nn.functional.normalize(light_dirs, dim=1)
    view_dirs = torch.distributions.Uniform(-1, 1).sample([100, 3])
    view_dirs = torch.nn.functional.normalize(view_dirs, dim=1)
    normals = torch.distributions.Uniform(-1, 1).sample([100, 3])
    normals = torch.nn.functional.normalize(normals, dim=1)

    NdotL = torch.sum(light_dirs * normals, dim=1)
    light_dirs[NdotL < 0] = -light_dirs[NdotL < 0]
    NdotV = torch.sum(view_dirs * normals, dim=1)
    view_dirs[NdotV < 0] = -view_dirs[NdotV < 0]

    diffuse = torch.distributions.Uniform(0, 1).sample([100, 3])
    metallic = torch.distributions.Uniform(0, 1).sample([100, 1])
    roughness = torch.distributions.Uniform(0, 1).sample([100, 1])

    values = brdf(light_dirs, view_dirs, normals, diffuse, metallic, roughness)

    assert torch.all(torch.isfinite(values))
    assert torch.all(values >= 0.0)

    values_rec = brdf(view_dirs, light_dirs, normals, diffuse, metallic, roughness)

    assert torch.allclose(values, values_rec)

    light_dir = torch.tensor([[1.0, 0.0, 0.0]])
    view_dir = torch.tensor([[1.0, 0.0, 0.0]])
    normal = torch.tensor([[0.0, 0.0, 1.0]])

    diffuse = torch.distributions.Uniform(0, 1).sample([1, 3])
    metallic = torch.distributions.Uniform(0, 1).sample([1, 1])
    roughness = torch.distributions.Uniform(0, 1).sample([1, 1])

    values = brdf(light_dir, view_dir, normal, diffuse, metallic, roughness)

    assert torch.all(torch.isfinite(values))
