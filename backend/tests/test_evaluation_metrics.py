from __future__ import annotations

import pytest


def rectangle(x1: float, y1: float, x2: float, y2: float) -> list[tuple[float, float]]:
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def test_crop_metrics_are_perfect_for_identical_instances() -> None:
    from app.evaluation.cropping import evaluate_cropping_dataset, evaluate_page_crops

    truth = [rectangle(0.1, 0.1, 0.4, 0.5), rectangle(0.55, 0.2, 0.9, 0.8)]
    page = evaluate_page_crops(truth, truth, width=800, height=600, accepted=[True, False])
    dataset = evaluate_cropping_dataset([page])

    assert page.true_positive == 2
    assert page.serious_failures == 0
    assert page.boundary_f_score == pytest.approx(1.0)
    assert dataset.detection_f1 == pytest.approx(1.0)
    assert dataset.polygon_iou == pytest.approx(1.0)
    assert dataset.automatic_acceptance_rate == pytest.approx(0.5)
    assert dataset.human_review_rate == pytest.approx(0.5)


def test_crop_metrics_measure_content_cutoff_and_background_independently() -> None:
    from app.evaluation.cropping import evaluate_page_crops

    truth = [rectangle(0.2, 0.2, 0.8, 0.8)]
    too_tight = [rectangle(0.3, 0.2, 0.8, 0.8)]
    too_wide = [rectangle(0.1, 0.2, 0.8, 0.8)]

    tight = evaluate_page_crops(truth, too_tight, width=600, height=400)
    wide = evaluate_page_crops(truth, too_wide, width=600, height=400)

    assert tight.content_cutoff > 0.15
    assert tight.background_retained == pytest.approx(0.0)
    assert wide.content_cutoff == pytest.approx(0.0)
    assert wide.background_retained > 0.1
    assert tight.serious_failures == 1


def test_crop_metrics_do_not_reuse_one_merged_detection_for_two_photos() -> None:
    from app.evaluation.cropping import evaluate_page_crops

    truth = [rectangle(0.1, 0.2, 0.4, 0.7), rectangle(0.5, 0.2, 0.8, 0.7)]
    merged = [rectangle(0.1, 0.2, 0.8, 0.7)]

    page = evaluate_page_crops(truth, merged, width=700, height=500, iou_threshold=0.4)

    assert page.true_positive == 1
    assert page.false_negative == 1
    assert page.serious_failures >= 1
