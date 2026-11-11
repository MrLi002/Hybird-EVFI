from types import SimpleNamespace
import pytest
import torch
from models.blocks import BackwardWarp,flow_to_scale
from models.query import FlowQuery
from models.model import HybridModel
from models.config import ModelConfig
from models.basr import make_basr_masks,safe_key_attention,DetailsSup,TransformerBlock,effective_repair_gate
from models.flow_corr import local_corr,FlowCorrelationRefiner
from models.losses import masked_charbonnier,masked_ssim_loss,HybridLoss
from models.softsplat import softsplat,torch_softsplat
from tests.reference_ofaf.model import MyModel as ReferenceOF_AF


def test_warp_coordinates_and_flow_units():
    image=torch.arange(8*10,dtype=torch.float32).view(1,1,8,10)
    zeros=torch.zeros(1,2,8,10)
    torch.testing.assert_close(BackwardWarp()(image,zeros),image,atol=1e-5,rtol=0)
    assert not torch.allclose(BackwardWarp(True)(image,zeros),image)
    flow=zeros.clone();flow[:,0]=3;flow[:,1]=-2
    warped,valid=BackwardWarp()(image,flow,True)
    torch.testing.assert_close(warped[:,:,2:,:-3],image[:,:,:-2,3:],atol=1e-5,rtol=0)
    assert valid[:,:,:2].sum()==0
    up=flow_to_scale(torch.ones(1,2,2,3),0)
    assert up.shape==(1,2,8,12) and torch.all(up==4)


@pytest.mark.parametrize('legacy',[False,True])
def test_query_nodes_and_between_nodes(legacy):
    names=('flows0t','flows1t','flowst0','flowst1','ctxst0','ctxst1')
    knots={name:[torch.tensor(float(k)) for k in range(17)] for name in names}
    for t in (.001,.125,.5,.577,.999):
        out=FlowQuery(16,legacy)(knots,t);index=16*t;i=int(index)
        for key,value in out.items():
            expected=2*i+1-index if legacy and key!='f_0t' else index
            assert float(value)==pytest.approx(expected,abs=1e-6)
    with pytest.raises(ValueError):FlowQuery()(knots,1.)


def test_configuration_validation():
    with pytest.raises(ValueError):ModelConfig.from_args(dict(nb_of_flow=15))
    with pytest.raises(ValueError):ModelConfig.from_args(dict(unit_dim=44,basr_heads=(8,8)))
    with pytest.raises(ValueError):ModelConfig.from_args(dict(splat_metric=True,w_flow=0))
    with pytest.raises(ValueError):ModelConfig.from_args(dict(w_ps=1,basr=False))


def test_legacy_parameter_keys_and_output_equivalence():
    torch.manual_seed(9)
    model=HybridModel(dict(refine='resblock',legacy_query=True,legacy_warp=True,splat_backend='torch')).eval()
    reference=ReferenceOF_AF(SimpleNamespace(voxel_bins=128,nb_of_flow=16)).eval()
    assert set(model.state_dict())==set(reference.state_dict())
    reference.load_state_dict(model.state_dict(),strict=True)
    images=[torch.rand(1,3,16,16),torch.rand(1,3,16,16)]
    events=torch.randn(1,128,16,16)*.05;times=torch.tensor([[.125,.5,.77]])
    with torch.no_grad():
        actual=model(*images,events,times)
        expected=reference(*images,events,times)
    torch.testing.assert_close(actual,expected,atol=1e-5,rtol=0)
    assert not hasattr(model,'refine_head') and not hasattr(model,'event_adapter')


@pytest.mark.parametrize('unit,basr',[(32,False),(44,True)])
def test_model_multiscale_zero_initialization_caching_and_backward(unit,basr):
    model=HybridModel(dict(voxel_bins=8,nb_of_flow=2,unit_dim=unit,basr=basr,
                           mask_in_refine=basr,splat_backend='torch'))
    count=dict(events=0,frames=0,global_event=0)
    def hook(key):
        def run(*_):count[key]+=1
        return run
    handles=[model.events_encoder.register_forward_hook(hook('events')),
             model.refine_head.encoder_f.register_forward_hook(hook('frames')),
             model.event_adapter.enc.register_forward_hook(hook('global_event'))]
    im0,im1=torch.rand(1,3,32,32),torch.rand(1,3,32,32)
    event=torch.randn(1,8,32,32)*.05
    out=model(im0,im1,event,torch.tensor([[.25,.5,.75]]),return_aux=True)
    assert out['pred'].shape==(1,3,3,32,32)
    assert [tuple(x.shape[-2:]) for x in out['pred_ms'][0]]==[(32,32),(16,16),(8,8)]
    assert count==dict(events=1,frames=2,global_event=1)
    for j in range(3):torch.testing.assert_close(out['pred'][:,j],out['aux'][j]['w']['fuse'],atol=0,rtol=0)
    criterion=HybridLoss(w_lpips=0,w_flow=.1,w_ps=.1 if basr else 0)
    loss,_=criterion(out,torch.rand_like(out['pred']),im0,im1)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.refine_head.transformer.last_conv2.weight.grad.abs().sum()>0
    assert model.bidir_flow0.conv2.weight.grad is not None
    for h in handles:h.remove()


def test_default_128_bins_16_steps_nodes_and_output():
    model=HybridModel(dict(refine='resblock',splat_backend='torch')).eval()
    with torch.no_grad():
        x=torch.rand(1,3,16,16);e=torch.zeros(1,128,16,16)
        cache=model.encode_pair(x,x,e)
        assert all(len(nodes)==17 for nodes in cache['knots'].values())
        result=model.render_time(cache,.5)
    assert result.shape==(1,3,16,16)


def test_reject_different_batch_times_and_padding():
    model=HybridModel(dict(voxel_bins=8,nb_of_flow=2,basr=True))
    with pytest.raises(ValueError,match='share'):
        model(torch.zeros(2,3,32,32),torch.zeros(2,3,32,32),torch.zeros(2,8,32,32),torch.tensor([[.2],[.3]]))
    with pytest.raises(ValueError,match='pad'):
        model.encode_pair(torch.zeros(1,3,34,32),torch.zeros(1,3,34,32),torch.zeros(1,8,34,32))


def test_repair_support_truth_table():
    bad0=torch.tensor([0.,0.,1.,1.]).view(4,1,1,1)
    bad1=torch.tensor([0.,1.,0.,1.]).view(4,1,1,1)
    make=lambda bad:dict(hole=bad,diff=bad,in_bounds=torch.ones_like(bad))
    masks=make_basr_masks(dict(fuse=torch.zeros(4,3,1,1),a=make(bad0),b=make(bad1)))
    assert masks['repair'].flatten().tolist()==[0,1,1,1]
    assert masks['support0'].flatten().tolist()==[0,1,0,0]
    assert masks['support1'].flatten().tolist()==[0,0,1,0]
    assert masks['repair_gate'].flatten().tolist()==[0,1,1,0]


def test_attention_empty_keys_and_masked_keys():
    logits=torch.tensor([[[[0.,1.,100.]]],[[[1.,2.,3.]]]])
    values=torch.tensor([[[[2.],[4.],[1000.]]],[[[2.],[4.],[1000.]]]])
    valid=torch.tensor([[True,False,False],[False,False,False]])
    result,has=safe_key_attention(logits,values,valid)
    assert result.flatten().tolist()==[2.,0.] and has.tolist()==[True,False]
    block=TransformerBlock(2,8,8,4)
    out,has=block(torch.rand(1,8,4,4),torch.rand(1,8,4,4),torch.zeros(1,1,4,4))
    assert torch.isfinite(out).all() and out.count_nonzero()==0 and not has.any()


def test_full_basr_gate_and_empty_source():
    block=DetailsSup(2,8,8,heads=4)
    # Remove zero-initialized delta so the test actually exercises output protection.
    torch.nn.init.normal_(block.fusion_block[-1].weight,std=.02)
    features=torch.rand(1,8,4,4);up=torch.rand(1,32,8,8)
    masks=dict(support0=torch.ones(1,1,8,8),support1=torch.zeros(1,1,8,8),repair_gate=torch.zeros(1,1,8,8))
    torch.testing.assert_close(block(features,features,features,up,masks),up,atol=0,rtol=0)

    masks['repair_gate'][:,:,2:6,2:6]=1
    output=block(features,features,features,up,masks)
    assert (output-up)[:,:,2:6,2:6].abs().sum()>0
    gate=masks['repair_gate'].expand_as(up).bool()
    torch.testing.assert_close(output[~gate],up[~gate],atol=0,rtol=0)
    masks['support0'].zero_()
    torch.testing.assert_close(block(features,features,features,up,masks),up,atol=0,rtol=0)


def test_regional_loss_gate_tracks_empty_key_bypass():
    support=torch.zeros(1,1,32,32);support[:,:,0,0]=1
    masks=dict(repair_gate=support.clone(),support0=support.clone(),support1=torch.zeros_like(support))
    # There is local repair/support, but no fully reliable 32x32 source patch.
    assert masks['repair_gate'].sum()>0
    assert effective_repair_gate(masks).sum()==0
    masks['support0'].fill_(1)
    assert effective_repair_gate(masks).sum()>0


def test_masked_losses_are_area_normalized_and_empty_safe():
    x=torch.zeros(1,3,16,16,requires_grad=True);y=torch.ones_like(x)*.2
    mask=torch.zeros(1,1,16,16);mask[:,:,:4,:4]=1
    small=masked_charbonnier(x,y,mask)
    full=masked_charbonnier(x,y,torch.ones_like(mask))
    torch.testing.assert_close(small,full)
    assert masked_charbonnier(x,y,torch.zeros_like(mask))==0
    assert masked_ssim_loss(x,y,torch.zeros_like(mask))==0
    assert masked_ssim_loss(y,y,torch.ones_like(mask)).abs()<1e-5
    small.backward();assert torch.isfinite(x.grad).all()


def test_lpips_frozen_network_keeps_prediction_gradients_and_normalization():
    class PerceptualStub(torch.nn.Module):
        def __init__(self):
            super().__init__();self.scale=torch.nn.Parameter(torch.tensor(1.));self.normalized=None
        def forward(self,x,y,normalize=False):
            self.normalized=normalize
            return self.scale*(x-y).square().mean((1,2,3),keepdim=True)
    network=PerceptualStub()
    criterion=HybridLoss(w_lpips=.1,lpips_model=network).train()
    prediction=torch.zeros(1,1,3,16,16,requires_grad=True)
    outputs=dict(pred=prediction,pred_ms=[[prediction[:,0]]],aux=[{}])
    loss,_=criterion(outputs,torch.ones_like(prediction),prediction[:,0],prediction[:,0])
    loss.backward()
    assert network.normalized is True and network.training is False
    assert network.scale.grad is None and not network.scale.requires_grad
    assert prediction.grad.abs().sum()>0


def test_flow_smoothness_uses_each_flow_grid(monkeypatch):
    import models.losses as losses
    images=[torch.rand(1,3,16,16) for _ in range(3)]
    guides=[]
    def smooth(flow,image,beta=1.):
        guides.append(image)
        return flow.sum()*0
    monkeypatch.setattr(losses,'cal_grad2_error',smooth)
    q={key:torch.zeros(1,2,4,4) for key in ('f_0t','f_1t','f_t0','f_t1')}
    branch=dict(coverage=torch.ones(1,1,16,16),in_bounds=torch.ones(1,1,16,16),im_fwd=images[0])
    criterion=HybridLoss(w_lpips=0,w_flow=.1)
    criterion.flow_loss(q,dict(a=branch,b=branch),*images)
    assert [id(x) for x in guides]==[id(images[0]),id(images[1]),id(images[2]),id(images[2])]


def test_checkpointed_refinement_backward():
    model=HybridModel(dict(voxel_bins=8,nb_of_flow=2,checkpoint_bifeb=True,
                           checkpoint_refine=True,splat_backend='torch'))
    x=torch.rand(1,3,16,16)
    result=model(x,x,torch.zeros(1,8,16,16),torch.tensor([[.5]]),True)
    loss=charbonnier_for_test(result['pred'],torch.rand_like(result['pred']))
    loss.backward()
    assert model.refine_head.transformer.last_conv2.weight.grad.abs().sum()>0


def charbonnier_for_test(x,y):
    return ((x-y).square()+1e-6).sqrt().mean()


def test_local_correlation_and_no_query_mutation():
    a=torch.rand(1,128,4,4)
    torch.testing.assert_close(local_corr(a,a,0),(a*a).mean(1,keepdim=True))
    q={name:torch.zeros(1,2,4,4) for name in ('f_0t','f_1t','f_t0','f_t1')}
    q.update(ctx0=a,ctx1=a)
    refiner=FlowCorrelationRefiner(radius=1)
    corrected=refiner(q,a,a)
    assert corrected is not q and corrected['f_0t'] is q['f_0t']
    torch.testing.assert_close(corrected['f_t0'],q['f_t0'])


def test_splat_identity_collision_metric_gradient_and_gradcheck():
    src=torch.rand(1,1,3,4)
    zero=torch.zeros(1,2,3,4)
    torch.testing.assert_close(softsplat(src,zero,backend='torch'),src,atol=2e-7,rtol=0)
    torch.testing.assert_close(softsplat(src,zero,torch.zeros(1,1,3,4),'soft',backend='torch'),
                               softsplat(src,zero,backend='torch'))
    flow=zero.clone();flow[:,0,:,1]=-1
    metric=torch.zeros(1,1,3,4,requires_grad=True)
    out=softsplat(src,flow,metric,'soft',backend='torch');out.sum().backward()
    assert torch.isfinite(metric.grad).all() and metric.grad.abs().sum()>0
    x=torch.rand(1,1,2,2,dtype=torch.double,requires_grad=True)
    f=torch.full((1,2,2,2),.2,dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda image,displacement:torch_softsplat(image,displacement,None,'avg'),
                                    (x,f),atol=1e-4,rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA device unavailable')
def test_optional_cupy_matches_torch_values_and_gradients():
    pytest.importorskip('cupy')
    x=torch.rand(1,3,8,8,device='cuda',requires_grad=True)
    flow=(torch.rand(1,2,8,8,device='cuda')*.4+.2).requires_grad_()
    a=softsplat(x,flow,None,'avg',backend='torch')
    b=softsplat(x,flow,None,'avg',backend='cupy')
    torch.testing.assert_close(a,b,atol=2e-5,rtol=1e-5)
    ga=torch.autograd.grad(a.square().mean(),(x,flow),retain_graph=True)
    gb=torch.autograd.grad(b.square().mean(),(x,flow))
    for first,second in zip(ga,gb):torch.testing.assert_close(first,second,atol=2e-5,rtol=1e-4)

