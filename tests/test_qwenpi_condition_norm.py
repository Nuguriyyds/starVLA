"""Guard condition-only normalization and actual-input diagnostic semantics."""
from pathlib import Path
import sys
import unittest

import torch
from starVLA.model.modules.action_model.flow_matching_head.cross_attention_dit import DiT, BasicTransformerBlock
from test_qwenpi_decoder_norm import head

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'examples/umi_pretrain/tools'))
from diagnose_qwenpi_condition import probe_velocity


def dit(policy='none'):
    return DiT(num_layers=2,num_attention_heads=2,attention_head_dim=16,
               cross_attention_dim=32,dropout=0.,final_dropout=False,
               interleave_self_attention=True,positional_embeddings=None,
               cross_condition_norm=policy)


class ConditionNormTest(unittest.TestCase):
    def test_unchanged_initialization_and_self_attention(self):
        torch.manual_seed(42); old=dit(); rng=torch.get_rng_state()
        torch.manual_seed(42); new=dit('layer_norm')
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertEqual(list(old.state_dict()),list(new.state_dict()))
        for key,value in old.state_dict().items():
            self.assertTrue(torch.equal(value,new.state_dict()[key]),key)
        self.assertEqual(new.config.cross_condition_norm,'layer_norm')
        x=torch.randn(2,4,32); temb=torch.randn(2,32)
        self.assertTrue(torch.equal(old.transformer_blocks[1](x,temb=temb),
                                    new.transformer_blocks[1](x,temb=temb)))

    def test_same_normalized_condition_reaches_kv_and_keeps_gradient(self):
        model=dit('layer_norm')
        c=(torch.randn(2,5,32)*100+10).requires_grad_()
        x=torch.randn(2,4,32)
        observed={}
        hooks=[]
        for name in ('to_k','to_v'):
            hooks.append(getattr(model.transformer_blocks[0].attn1,name).register_forward_pre_hook(
                lambda m,a,n=name: observed.__setitem__(n,a[0])))
        try:
            out=model(x,[c,c],torch.tensor([0,250]),return_pre_output=True)
            out.square().mean().backward()
        finally:
            for h in hooks: h.remove()
        k,v=observed['to_k'],observed['to_v']
        self.assertIs(k,v)
        self.assertLess(float(k.mean(-1).abs().max()),1e-5)
        self.assertTrue(torch.allclose(k.square().mean(-1),torch.ones(2,5),atol=1e-4))
        self.assertTrue(torch.isfinite(c.grad).all() and bool(c.grad.any()))
        changed=c.detach().clone(); changed[:,4]+=500
        norm=model.transformer_blocks[0].cross_condition_norm
        self.assertTrue(torch.equal(norm(c)[:,:4],norm(changed)[:,:4]))

    def test_training_and_sampling_apply_norm(self):
        model=head('layer_norm')
        # Same implementation as DiT construction; test head's two shared paths.
        for block in model.model.transformer_blocks:
            block.cross_condition_norm=torch.nn.LayerNorm(32,elementwise_affine=False)
        calls=[]
        h=model.model.transformer_blocks[0].attn1.to_k.register_forward_pre_hook(
            lambda m,a: calls.append(a[0].detach().clone()))
        c=[torch.randn(2,5,32)*100 for _ in range(2)]
        state=torch.randn(2,1,16); action=torch.randn(2,16,16)
        try:
            model(c,action,state).backward()
            model.eval(); model.predict_action(c,state)
        finally: h.remove()
        self.assertEqual(len(calls),5)
        self.assertTrue(all(float(v.mean(-1).abs().max())<1e-5 for v in calls))
        self.assertTrue(bool(model.state_encoder.layer1.weight.grad.any()))

    def test_probe_uses_real_first_step_and_restores_hooks(self):
        model=head('layer_norm').eval()
        c=[torch.randn(1,5,32) for _ in range(2)]
        state=torch.randn(1,1,16)
        captured={}
        def encoder(m,a):
            if 'noise' not in captured: captured['noise']=a[0].detach().clone()
        def decoder(m,a,o):
            if 'velocity' not in captured: captured['velocity']=o[:,-16:].detach().clone()
        h1=model.action_encoder.register_forward_pre_hook(encoder)
        h2=model.action_decoder.register_forward_hook(decoder)
        try: model.predict_action(c,state)
        finally: h1.remove(); h2.remove()
        actual,_=probe_velocity(model,c,None,captured['noise'],state,0.,details=False)
        self.assertTrue(torch.equal(actual['velocity'],captured['velocity']))
        for m in model.modules():
            self.assertEqual(len(m._forward_hooks)+len(m._forward_pre_hooks),0)
        zero,report=probe_velocity(model,c,None,captured['noise'],state,.25,'zero_cross',False)
        self.assertTrue(torch.isfinite(zero['velocity']).all())

    def test_invalid_policy_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown cross_condition_norm'): dit('typo')
        with self.assertRaisesRegex(ValueError,'Unknown cross_condition_norm'):
            BasicTransformerBlock(32,2,16,cross_condition_norm='typo')


if __name__=='__main__': unittest.main()
