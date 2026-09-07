from surgground.cfg import box, cfg_hash, expand, load_cfg, provenance, set_seed


def test_load_default_resolves():
    cfg = load_cfg()
    assert cfg.seed == 0
    assert cfg.backend == "internvl3_2b"
    assert str(cfg.paths.data_root)          # env-interpolation resolved to a plain str
    assert cfg.data.regime.short_max_s == 2700
    assert cfg.ckpt_hub_repo == "DucAnhValentinoNguyen/surgground-ckpts"


def test_overrides_and_hash_change():
    a = load_cfg()
    b = load_cfg(overrides=["seed=7", "backend=internvl3_8b"])
    assert b.seed == 7 and b.backend == "internvl3_8b"
    assert cfg_hash(a) != cfg_hash(b)
    assert cfg_hash(a) == cfg_hash(load_cfg())      # deterministic


def test_provenance_shape():
    p = provenance(0, note="unit")
    for k in ("seed", "git_sha", "box", "python", "torch", "timestamp"):
        assert k in p
    assert p["note"] == "unit"


def test_helpers():
    assert expand("~/x").startswith("/")
    set_seed(0)
    assert box()  # some non-empty string
