from rava.experiments.runner import _sample_with_stratification


def test_stratified_sampling_respects_counts():
    examples = []
    for i in range(90):
        examples.append(
            {
                "id": f"a-{i}",
                "metadata": {"demographics": {"gender": "group_0"}},
            }
        )
    for i in range(90, 150):
        examples.append(
            {
                "id": f"b-{i}",
                "metadata": {"demographics": {"gender": "group_1"}},
            }
        )

    sampled = _sample_with_stratification(
        dataset="bias_in_bios",
        examples=examples,
        dataset_limit=80,
        stratified_cfg={
            "field": "metadata.demographics.gender",
            "counts": {"group_0": 40, "group_1": 40},
        },
        seed=42,
    )
    assert len(sampled) == 80
    counts = {"group_0": 0, "group_1": 0}
    for row in sampled:
        g = str(row["metadata"]["demographics"]["gender"])
        if g in counts:
            counts[g] += 1
    assert counts["group_0"] == 40
    assert counts["group_1"] == 40


def test_paired_fixed_panel_is_deterministic_and_seeded():
    examples = [{"id": f"ex-{i}", "metadata": {}} for i in range(100)]
    sampled_a = _sample_with_stratification(
        dataset="medqa",
        examples=examples,
        dataset_limit=12,
        stratified_cfg=None,
        seed=42,
        sampling_mode="paired_fixed_panel",
    )
    sampled_b = _sample_with_stratification(
        dataset="medqa",
        examples=examples,
        dataset_limit=12,
        stratified_cfg=None,
        seed=42,
        sampling_mode="paired_fixed_panel",
    )
    sampled_c = _sample_with_stratification(
        dataset="medqa",
        examples=examples,
        dataset_limit=12,
        stratified_cfg=None,
        seed=123,
        sampling_mode="paired_fixed_panel",
    )
    assert [row["id"] for row in sampled_a] == [row["id"] for row in sampled_b]
    assert [row["id"] for row in sampled_a] != [row["id"] for row in sampled_c]


def test_stratified_sampling_strict_fails_when_group_counts_unavailable():
    examples = [{"id": f"ex-{i}", "metadata": {"demographics": {"gender": "group_0"}}} for i in range(10)]
    try:
        _sample_with_stratification(
            dataset="bias_in_bios",
            examples=examples,
            dataset_limit=10,
            stratified_cfg={
                "field": "metadata.demographics.gender",
                "counts": {"group_0": 5, "group_1": 5},
                "strict": True,
            },
            seed=42,
        )
    except RuntimeError as exc:
        assert "could not satisfy requested counts" in str(exc)
        return
    raise AssertionError("Expected strict stratified sampling to fail when a required group is missing.")
