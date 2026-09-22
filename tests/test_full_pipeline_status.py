import json

from scripts.full_pipeline_status import status


def write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def setup_campaign(tmp_path, complete=False):
    campaign = tmp_path / "campaign.json"
    output = tmp_path / "labels"
    write(campaign, {"condition_ids": ["a"], "candidate_names": ["x"]})
    if complete:
        write(output / "cases/a_x/record.json", {"status": "accepted"})
    return campaign, output


def test_label_generation(tmp_path):
    campaign, output = setup_campaign(tmp_path)
    result = status(campaign, output, tmp_path / "train", tmp_path / "physics")
    assert result["active_stage"] == "label_generation"
    assert result["campaign"]["observed_records"] == 0
    assert result["campaign"]["retrying"] == 0


def test_training_stage(tmp_path):
    campaign, output = setup_campaign(tmp_path, True)
    train = tmp_path / "train"
    write(train / "workflow.json", {"stage": "training"})
    write(
        train / "progress.json",
        {
            "status": "running",
            "epoch": 3,
            "planned_epochs": 10,
            "best_epoch": 2,
            "latest": {"train_loss": 0.2, "validation_loss": 0.3},
        },
    )
    result = status(campaign, output, train, tmp_path / "physics")
    assert result["active_stage"] == "training"
    assert result["training"]["current_epoch"] == 3
    assert result["training"]["train_loss"] == 0.2


def test_frozen_physics_and_complete(tmp_path):
    campaign, output = setup_campaign(tmp_path, True)
    physics = tmp_path / "physics"
    write(
        physics / "workflow.json",
        {
            "stage": "physics_evaluation",
            "planned": 10,
            "completed": 4,
            "accepted": 4,
            "unsettled": 0,
            "remaining": 6,
        },
    )
    running = status(campaign, output, tmp_path / "train", physics)
    assert running["active_stage"] == "frozen_physics"
    assert running["physics"]["completed"] == 4
    write(physics / "report.json", {"decision": "passes_frozen_physics_evaluation"})
    assert status(campaign, output, tmp_path / "train", physics)["active_stage"] == "complete"


def test_malformed_optional_state_is_tolerated(tmp_path):
    campaign, output = setup_campaign(tmp_path)
    (tmp_path / "train").mkdir()
    (tmp_path / "train/summary.json").write_text("{")
    result = status(campaign, output, tmp_path / "train", tmp_path / "physics")
    assert result["active_stage"] == "label_generation"


def test_waiting_physics_watcher_does_not_hide_label_generation(tmp_path):
    campaign, output = setup_campaign(tmp_path)
    physics = tmp_path / "physics"
    write(physics / "workflow.json", {"stage": "waiting_for_training"})

    result = status(campaign, output, tmp_path / "train", physics)

    assert result["active_stage"] == "label_generation"


def test_main_physics_hands_off_to_circle_generalization(tmp_path):
    campaign, output = setup_campaign(tmp_path, True)
    physics = tmp_path / "physics"
    generalization = tmp_path / "generalization"
    write(physics / "report.json", {"decision": "passes_frozen_physics_evaluation"})
    write(generalization / "workflow.json", {"stage": "circle_generalization_evaluation"})

    running = status(
        campaign,
        output,
        tmp_path / "train",
        physics,
        generalization_output=generalization,
    )
    assert running["active_stage"] == "circle_generalization"
    assert running["generalization"]["workflow"] == "circle_generalization_evaluation"

    write(
        generalization / "report.json",
        {"decision": "passes_circle_position_scale_generalization"},
    )
    complete = status(
        campaign,
        output,
        tmp_path / "train",
        physics,
        generalization_output=generalization,
    )
    assert complete["active_stage"] == "complete"
