from __future__ import annotations

import unittest

import torch
from torch import nn

from src.interventions import (
    FeatureInterventionHook,
    InterventionRequest,
    apply_feature_intervention,
)
from src.sae import TopKSAE


def identity_sae() -> TopKSAE:
    sae = TopKSAE(
        d_model=2,
        d_sae=2,
        k=1,
        device=torch.device("cpu"),
        subtract_pre_bias=False,
    )
    with torch.no_grad():
        sae.encoder_weight.copy_(torch.eye(2))
        sae.encoder_bias.zero_()
        sae.decoder_weight.copy_(torch.eye(2))
        sae.decoder_bias.zero_()
    return sae.eval().requires_grad_(False)


class ApplyFeatureInterventionTests(unittest.TestCase):
    def test_clamping_adds_only_activation_difference_along_direction(self) -> None:
        residual = torch.tensor([[2.0, 0.5]])

        modified = apply_feature_intervention(
            residual, identity_sae(), feature_id=0, mode="clamp", amount=5.0
        )

        torch.testing.assert_close(modified, torch.tensor([[5.0, 0.5]]))
        torch.testing.assert_close(residual, torch.tensor([[2.0, 0.5]]))

    def test_clamping_treats_a_feature_outside_top_k_as_zero(self) -> None:
        residual = torch.tensor([[2.0, 0.5]])

        modified = apply_feature_intervention(
            residual, identity_sae(), feature_id=1, mode="clamp", amount=3.0
        )

        torch.testing.assert_close(modified, torch.tensor([[2.0, 3.5]]))

    def test_additive_steering_adds_scaled_decoder_direction(self) -> None:
        residual = torch.tensor([[2.0, 0.5]])

        modified = apply_feature_intervention(
            residual,
            identity_sae(),
            feature_id=1,
            mode="additive",
            amount=-1.5,
        )

        torch.testing.assert_close(modified, torch.tensor([[2.0, -1.0]]))

    def test_intervention_converts_from_normalized_sae_coordinates(self) -> None:
        residual = torch.tensor([[20.0, 5.0]])

        clamped = apply_feature_intervention(
            residual,
            identity_sae(),
            feature_id=0,
            mode="clamp",
            amount=5.0,
            activation_scale=0.1,
        )
        steered = apply_feature_intervention(
            residual,
            identity_sae(),
            feature_id=0,
            mode="additive",
            amount=2.0,
            activation_scale=0.1,
        )

        torch.testing.assert_close(clamped, torch.tensor([[50.0, 5.0]]))
        torch.testing.assert_close(steered, torch.tensor([[40.0, 5.0]]))

    def test_hook_changes_layer_input_and_is_removed_after_context(self) -> None:
        layer = nn.Identity()
        residual = torch.tensor([[2.0, 0.5]])

        with FeatureInterventionHook(
            layer,
            identity_sae(),
            feature_id=0,
            mode="additive",
            amount=2.0,
            activation_scale=1.0,
        ):
            torch.testing.assert_close(layer(residual), torch.tensor([[4.0, 0.5]]))

        torch.testing.assert_close(layer(residual), residual)


class InterventionRequestTests(unittest.TestCase):
    def test_parses_each_intervention_parameter(self) -> None:
        clamp = InterventionRequest.from_payload(
            {
                "prompt": "Hello",
                "feature_id": 3,
                "mode": "clamp",
                "clamp_value": 8,
                "max_new_tokens": 12,
            },
            d_sae=10,
        )
        additive = InterventionRequest.from_payload(
            {
                "prompt": "Hello",
                "feature_id": 3,
                "mode": "additive",
                "alpha": -2.5,
                "max_new_tokens": 12,
            },
            d_sae=10,
        )

        self.assertEqual(clamp.amount, 8.0)
        self.assertEqual(clamp.intervention(), {"mode": "clamp", "clamp_value": 8.0})
        self.assertEqual(additive.amount, -2.5)
        self.assertEqual(
            additive.intervention(), {"mode": "additive", "alpha": -2.5}
        )

    def test_rejects_out_of_range_feature_and_token_count(self) -> None:
        base = {
            "prompt": "Hello",
            "feature_id": 10,
            "mode": "clamp",
            "clamp_value": 1.0,
            "max_new_tokens": 12,
        }
        with self.assertRaisesRegex(ValueError, "feature_id"):
            InterventionRequest.from_payload(base, d_sae=10)

        base["feature_id"] = 2
        base["max_new_tokens"] = 0
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            InterventionRequest.from_payload(base, d_sae=10)


if __name__ == "__main__":
    unittest.main()
