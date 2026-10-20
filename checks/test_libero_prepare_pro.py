"""HF local download cache must not be mistaken for evaluation metadata."""
from vpp2.libero.prepare_pro import prepare

def test_prepare_ignores_hf_cache(tmp_path):
    runtime=tmp_path/'runtime'
    core=runtime/'liberopro/liberopro'
    core.mkdir(parents=True)
    (core/'__init__.py').touch()
    assets=tmp_path/'assets'
    (assets/'scenes').mkdir(parents=True)
    for name in ['libero_tabletop_base_style.xml','libero_floor_base_style.xml','libero_kitchen_tabletop_base_style.xml','libero_living_room_tabletop_base_style.xml']:
        (assets/'scenes'/name).touch()
    snapshot=tmp_path/'snapshot'
    for name in ['init_files','bddl_files']:
        (snapshot/name).mkdir(parents=True)
        (snapshot/name/'real.txt').write_text('real metadata')
        (snapshot/'.cache/huggingface/download'/name).mkdir(parents=True)
        (snapshot/'.cache/huggingface/download'/name/'metadata').touch()
    result=prepare(runtime,assets,snapshot)
    assert result['revision']
    assert (core/'assets').resolve()==assets
    assert (core/'init_files/real.txt').read_text()=='real metadata'
    assert not (core/'init_files/metadata').exists()
