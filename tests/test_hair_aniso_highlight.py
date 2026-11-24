import pytest
import graphics.hair_aniso_highlight as mod

CASES = [
    ("tangent_continuity", (1,)),
    ("lod_highlight", (1,)),
    ("root_offset", (1,)),
    ("hair_shadow", (1,)),
    ("strand_order", (1,)),
    ("highlight_params", (1,)),
    ("card_blend", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("sort_strand", (1,)),
    ("metrics", (1,)),
    ("density", (1,)),
    ("alpha_edge", (1,)),
    ("determinism", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
