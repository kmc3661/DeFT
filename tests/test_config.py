import pytest
from deft import DeFTConfig


@pytest.mark.parametrize('p,alpha,expected',[
    (.8,.2,(360,200)),(.8,.3,(440,200)),(.9,.2,(280,100)),
    (.7,.2,(440,300)),(.8,0,(200,200)),(.8,1,(1000,200))])
def test_counts(p,alpha,expected):
    assert DeFTConfig(p,alpha).counts(1000)==expected


def test_rounding_and_boundaries():
    for n in range(1,2000):
        m,k=DeFTConfig().counts(n)
        assert 1<=k<=m<=n
    assert DeFTConfig().boundary(36)==17
    assert DeFTConfig(selection_depth=.25).boundary(36)==8
    assert DeFTConfig(selection_depth=.75).boundary(36)==26


@pytest.mark.parametrize('kwargs',[{'alpha':-1},{'alpha':2},{'prune_ratio':1},{'selection_depth':0}])
def test_invalid(kwargs):
    with pytest.raises(ValueError):DeFTConfig(**kwargs)


def test_matched_workload():
    reference=DeFTConfig().counts(1000)
    work=18*reference[0]+18*reference[1]
    for layer in (9,18,22,27):
        cfg=DeFTConfig(alpha=.2*18/layer,selection_depth=layer/36)
        m,k=cfg.counts(1000)
        assert abs(layer*m+(36-layer)*k-work)<=layer
