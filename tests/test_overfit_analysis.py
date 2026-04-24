from pathlib import Path

from scripts.run_overfit_analysis import parse_train_log, summarize_overfit


def test_parse_and_summarize_train_log(tmp_path: Path) -> None:
    log_path = tmp_path / "train.log"
    log_path.write_text(
        "\n".join(
            [
                "step=1 train_loss=0.900000 lr=0.00010000",
                "step=250 val_loss=0.800000 wall=2.0s n=10",
                "step=500 train_loss=0.700000 lr=0.00005000",
                "step=500 val_loss=0.850000 wall=2.1s n=10",
            ]
        ),
        encoding="utf-8",
    )

    parsed = parse_train_log(log_path)
    summary = summarize_overfit(parsed)

    assert len(parsed["train"]) == 2
    assert len(parsed["val"]) == 2
    assert summary["best_val_step"] == 250
    assert summary["final_val_step"] == 500
    assert abs(summary["val_loss_delta"] - 0.05) < 1e-8
    assert abs(summary["val_loss_delta_pct"] - 6.25) < 1e-8
