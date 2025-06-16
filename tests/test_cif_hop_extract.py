import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import zipfile

import pandas as pd

from rail_data.rail_io import cif_hop_extractor as cif


def make_line(prefix: str, fields: dict[tuple[int, int], str]) -> str:
    line = list(" " * 80)
    line[0:2] = prefix
    for (start, length), val in fields.items():
        line[start - 1 : start - 1 + length] = val.ljust(length)
    return "".join(line)


def test_build_and_iter_hops(tmp_path):
    ti = make_line("TI", {(3, 7): "TIP1", (45, 5): "12345", (54, 3): "AAA"})
    bs = make_line(
        "BS",
        {
            (4, 6): "TRNID1",
            (22, 7): "1111111",
            (10, 6): "010120",
            (16, 6): "310120",
            (42, 8): "SERV1",
        },
    )
    lo = make_line("LO", {(3, 7): "TIP1", (11, 4): "0800"})
    li = make_line("LI", {(3, 7): "TIP1", (11, 4): "0830", (16, 4): "0900"})

    zip_path = tmp_path / "cif.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("file.MCA", "\n".join([ti, bs, lo, li]))

    lines = list(cif.iter_cif_lines(zip_path))
    t2c, t2s = cif.build_tiploc_maps(lines)
    assert t2c["TIP1"] == "AAA"
    assert t2s["TIP1"] == "12345"

    hops = list(cif.iter_hops(lines, t2s, progress=False))
    assert len(hops) == 1
    hop = hops[0]
    assert hop.train_id == "TRNID1"
    assert hop.stanox_dep == "12345"
    assert hop.dep_time == "0800"

    df = cif.write_hops(hops)
    assert isinstance(df, pd.DataFrame)
    assert not df.empty