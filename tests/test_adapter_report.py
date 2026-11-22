import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from adapter_report import comparison_table, training_table

RUN = {
    "base": "org/parent-model",
    "runtime": {"minutes": {"total": 41.6}},
    "training": {"examples": 2500, "rank": 64, "target_modules": ["q_proj", "v_proj"], "learning_rate": 1e-4,
                 "selected_epoch": 2, "history": [{"validation_score": 60.0}, {"validation_score": 61.257}]},
    "test": {
        "ind_to_jav": {"n": 400, "before": {"chrf_pp": 44.0}, "after": {"chrf_pp": 60.25},
                       "scale_0.5": {"chrf_pp": 55.0}, "paired_sentence_delta_95_ci": [14.2, 18.1]},
        "senti_jav": {"n": 400, "before": {"accuracy": 70.0, "macro_f1": 60.0},
                      "after": {"accuracy": 82.5, "macro_f1": 80.0}, "accuracy_delta_95_ci": [8.0, 17.0]},
    },
}


def test_comparison_table_shows_parent_adapter_change_and_interval():
    other = {**RUN, "test": {"senti_jav": RUN["test"]["senti_jav"]}}
    table = comparison_table({"core": RUN, "other": other}).splitlines()
    assert table[0] == "| Key | Metric | n | Parent of core | core | Parent of other | other |"
    assert table[2] == "| `ind_to_jav` | chrf++ | 400 | 44.0 | 60.2 (+16.2 [+14.2, +18.1]) | — | — |"
    assert table[3] == ("| `senti_jav` | accuracy | 400 | 70.0 | 82.5 (+12.5 [+8.0, +17.0]) | 70.0 | "
                        "82.5 (+12.5 [+8.0, +17.0]) |")
    scaled = comparison_table({"core": RUN}, arm="scale_0.5").splitlines()
    assert scaled[2] == "| `ind_to_jav` | chrf++ | 400 | 44.0 | 55.0 (+11.0) |"
    assert scaled[3] == "| `senti_jav` |  |  | — | — |"


def test_training_table_lists_the_recipe_and_validation_by_epoch():
    row = training_table({"core": RUN}).splitlines()[2]
    assert row == "| core | parent-model | 2,500 | 64 | 2 | 0.0001 | 60.00, 61.26 | 2 | 42 |"
