from __future__ import annotations

from PIL import Image


def test_distortion_registry_apply():
    from wmbench.attacks.registry import build_default_registry

    reg = build_default_registry()
    atk = reg["Dist-Blur"]
    im = Image.new("RGB", (64, 64), color=(120, 44, 180))
    out = atk.apply(im, 0.5)
    assert out.size == im.size


def test_dct_roundtrip_bitscore():
    from wmbench.watermarks.dct import DCTAdapter

    ad = DCTAdapter(bit_length=64, seed=7, alpha=0.15)
    im = Image.new("L", (128, 128), color=200).convert("RGB")
    wm = ad.embed(im)
    s = ad.detect(wm, im)
    assert s == s
    assert s > 0.0


def test_dct_dwt_roundtrip_bitscore():
    from wmbench.watermarks import get_adapter

    ad = get_adapter("dct-dwt")
    im = Image.new("RGB", (512, 512), color=(160, 80, 200))
    wm = ad.embed(im)
    s = ad.detect(wm, None, meta=ad.payload_for_meta(), blind=True)
    assert s == s
    assert s > 0.5


def test_dwt_dct_svd_roundtrip_bitscore():
    from wmbench.watermarks import get_adapter

    ad = get_adapter("dwt-dct-svd")
    im = Image.new("RGB", (256, 256), color=(100, 150, 80))
    wm = ad.embed(im)
    s = ad.detect(wm, im, meta=ad.payload_for_meta(), blind=False)
    assert s == s
    assert s > 0.5
