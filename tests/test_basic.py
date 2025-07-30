
import numpy as np, pytest
import dask.array as da
from mifqc.entire_image import EntireImage

rng = np.random.default_rng(0)

@pytest.fixture
def dummy():
    arr = rng.random((3, 256, 256)).astype("float32")
    darr = da.from_array(arr, chunks=(1, 256, 256))
    return EntireImage.from_dask(darr, channel_names=["A", "B", "C"], name="dummy")

def test_per_channel_stats(dummy):
    df = dummy.per_channel_stats()
    assert set(df.index) == {"A", "B", "C"}
    # sanity: mean of channel A equals the actual mean
    np.testing.assert_allclose(df.loc["A", "mean"], dummy.pixels[0].mean().compute(), rtol=1e-6)

def test_tissue_fraction(dummy):
    with pytest.raises(ValueError):
        dummy.tissue_fraction("NONEXISTENT")

def test_cli_info(monkeypatch, capsys):
    from mifqc import cli
    monkeypatch.setattr(cli, "typer", __import__("types").SimpleNamespace(echo=print))
    cli.info()
    captured = capsys.readouterr().out
    assert "numpy" in captured and "dask" in captured
