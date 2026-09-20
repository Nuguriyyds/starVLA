"""Decoder norm must preserve initialization and serve training AND generation."""
import unittest

from omegaconf import OmegaConf
import torch

from starVLA.model.modules.action_model.LayerwiseFM_ActionHeader import LayerwiseFlowmatchingActionHead


def head(norm=None):
    options = dict(action_dim=16, state_dim=16, action_horizon=16, num_inference_timesteps=4,
        num_target_vision_tokens=2, add_pos_embed=True, max_seq_len=64,
        noise_beta_alpha=1.5, noise_beta_beta=1.0, noise_s=0.999, num_timestep_buckets=1000,
        diffusion_model_cfg=dict(num_layers=2, input_embedding_dim=32, cross_attention_dim=32,
            num_attention_heads=2, attention_head_dim=16, dropout=0., final_dropout=False,
            interleave_self_attention=True, use_canonical_forward=True))
    if norm is not None:
        options['decoder_input_norm'] = norm
    return LayerwiseFlowmatchingActionHead(OmegaConf.create(dict(framework=dict(action_model=options))))


class DecoderNormTest(unittest.TestCase):
    def test_initial_weights_and_rng_unchanged(self):
        torch.manual_seed(42)
        original = head()
        original_rng = torch.get_rng_state()
        torch.manual_seed(42)
        normalized = head('layer_norm')
        self.assertTrue(torch.equal(original_rng, torch.get_rng_state()))
        self.assertEqual(list(original.state_dict()), list(normalized.state_dict()))
        for key, value in original.state_dict().items():
            self.assertTrue(torch.equal(value, normalized.state_dict()[key]), key)
        self.assertIsInstance(original.decoder_input_norm, torch.nn.Identity)

    def test_norm_used_in_train_and_each_sampling_step(self):
        torch.manual_seed(19)
        model = head('layer_norm')
        conditions = [torch.randn(2, 5, 32) * 1000 for _ in range(2)]
        state = torch.randn(2, 1, 16)
        actions = torch.randn(2, 16, 16)
        observed = []
        hook = model.action_decoder.register_forward_pre_hook(lambda module, args: observed.append(args[0].detach().clone()))
        try:
            loss = model(conditions, actions, state)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            grad = model.state_encoder.layer1.weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertTrue(bool(grad.any()))
            model.eval()
            generated = model.predict_action(conditions, state)
        finally:
            hook.remove()
        self.assertEqual(len(observed), 5)
        for value in observed:
            self.assertLess(float(value.mean(-1).abs().max()), 1e-5)
            self.assertTrue(torch.allclose(value.square().mean(-1), torch.ones(value.shape[:2]), atol=1e-4))
        self.assertEqual(tuple(generated.shape), (2, 16, 16))
        self.assertTrue(torch.isfinite(generated).all())

    def test_unknown_policy_fails(self):
        with self.assertRaisesRegex(ValueError, 'Unknown decoder_input_norm'):
            head('typo')


if __name__ == '__main__':
    unittest.main()
