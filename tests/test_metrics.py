import numpy as np
import pandas as pd

from gridfc.harness import QCOLS
from gridfc.metrics import score


def test_partial_quantiles_score_what_they_can():
    """TimesFM has no q05/q95: 80% coverage is scored, 90% coverage and pinball are not."""
    y = np.array([100.0, 110.0, 90.0, 105.0])
    d = pd.DataFrame({"y": y, "mean": 100.0, "q10": 95.0, "q25": 98.0, "q50": 100.0,
                      "q75": 102.0, "q90": 106.0, "q05": np.nan, "q95": np.nan})
    s = score(d)
    assert s["cover80_%"] == 50.0  # 100 and 105 are inside 95-106
    assert "cover90_%" not in s and "pinball_%" not in s
    full = d.assign(q05=90.0, q95=110.0)
    assert set(["cover90_%", "pinball_%"]) <= set(score(full).index)
    assert score(full)["cover90_%"] == 100.0
    assert list(full[QCOLS].columns) == QCOLS
