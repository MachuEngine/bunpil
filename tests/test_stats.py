"""evals/stats.py — 부트스트랩 CI·가중 kappa 유닛테스트. LLM 호출 없음."""
import pytest

from evals.stats import cluster_bootstrap_ci, mean_ci, weighted_kappa


def test_mean_ci_reproducible_with_same_seed():
    values = [1, 2, 3, 4, 5, 6, 7, 8]
    clusters = ["a", "a", "b", "b", "c", "c", "d", "d"]
    result1 = mean_ci(values, clusters, n_boot=200, seed=42)
    result2 = mean_ci(values, clusters, n_boot=200, seed=42)
    assert result1 == result2


def test_mean_ci_reflects_cluster_level_variation():
    """클러스터 내부는 완전히 같은 값, 클러스터 간에만 값이 다르면 CI가 그 변동을 반영해야 한다."""
    # 클러스터 a는 전부 0, 클러스터 b는 전부 10 — 리샘플이 클러스터 단위로 섞이므로
    # 항목 단위로 뽑았을 때보다 훨씬 넓은 CI가 나와야 한다(절반 확률로 평균이 0 또는 10에 가까움).
    values = [0, 0, 0, 10, 10, 10]
    clusters = ["a", "a", "a", "b", "b", "b"]
    point, lo, hi = mean_ci(values, clusters, n_boot=500, seed=0)
    assert point == pytest.approx(5.0)
    assert hi - lo > 5.0


def test_cluster_bootstrap_ci_no_clusters_returns_none_bounds():
    point, lo, hi = cluster_bootstrap_ci([], [], lambda rows: None)
    assert point is None
    assert lo is None
    assert hi is None


def test_cluster_bootstrap_ci_paired_diff_usage():
    """짝지은 비교(A-B 차이) 사용법 — rows에 (a, b) 쌍을 넣고 stat_fn에서 평균 차이를 낸다."""
    rows = [(3, 1), (4, 2), (5, 3)]
    clusters = ["q1", "q2", "q3"]

    def mean_diff(sample):
        if not sample:
            return None
        return sum(a - b for a, b in sample) / len(sample)

    point, lo, hi = cluster_bootstrap_ci(rows, clusters, mean_diff, n_boot=200, seed=0)
    assert point == pytest.approx(2.0)
    assert lo is not None and hi is not None


def test_weighted_kappa_perfect_agreement():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    assert weighted_kappa(a, b, labels=[1, 2, 3, 4, 5]) == pytest.approx(1.0)


def test_weighted_kappa_all_same_value_returns_none():
    a = [3, 3, 3, 3]
    b = [3, 3, 3, 3]
    assert weighted_kappa(a, b, labels=[1, 2, 3, 4, 5]) is None


def test_weighted_kappa_value_outside_labels_raises():
    # 척도 밖 값은 조용히 건너뛰지 않고 입력 오류로 취급한다.
    a = [1, 2, 6]
    b = [1, 2, 3]
    with pytest.raises(ValueError, match=r"\[6\]"):
        weighted_kappa(a, b, labels=[1, 2, 3, 4, 5])


def test_weighted_kappa_sparse_observations_use_full_scale():
    """관측값이 {1,2,5}뿐이어도 labels=[1..5] 척도 전체를 기준으로 계산해야 한다 —
    관측값만으로 추정하면 2와 5가 인접 라벨로 취급돼 값이 달라진다."""
    sklearn = pytest.importorskip("sklearn")
    from sklearn.metrics import cohen_kappa_score

    a = [1, 1, 2, 2, 5, 5]
    b = [1, 2, 2, 5, 1, 5]

    ours = weighted_kappa(a, b, labels=[1, 2, 3, 4, 5], weights="quadratic")
    theirs = cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])
    assert ours == pytest.approx(theirs)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_kappa_matches_sklearn(seed):
    sklearn = pytest.importorskip("sklearn")
    from sklearn.metrics import cohen_kappa_score
    import random

    rng = random.Random(seed)
    a = [rng.randint(1, 5) for _ in range(30)]
    b = [rng.randint(1, 5) for _ in range(30)]

    ours = weighted_kappa(a, b, labels=[1, 2, 3, 4, 5], weights="quadratic")
    theirs = cohen_kappa_score(a, b, weights="quadratic", labels=[1, 2, 3, 4, 5])
    assert ours == pytest.approx(theirs)
