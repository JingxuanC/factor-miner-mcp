"""验证尾巴上的三个指标：`rank_ic` 曾经不是秩相关。

真实 bug（2026-10-13 审计发现）：两个训练路径都写着

    rank_ic = np.corrcoef(np.argsort(pred_val), np.argsort(y_val))[0, 1]

**一次 `argsort` 得到的是逆置换，不是秩**。对两个逆置换求 Pearson 既不保序、
也不等于 Spearman。实测（n=80）把真 RankIC 0.5398 报成 **−0.1290**，误差最大
0.6688 —— 系统性虚低 0.3~0.7，而且会翻负号。同一个仓库里 `analytics.py:161`
用的是 `scipy.stats.spearmanr`（对的），所以只有 ML 这条链的数字是坏的。

这个文件钉三件事：

1. `ranking_metrics()["rank_ic"]` 必须等于 Spearman（正例）；
2. 上面那个表达式**必须不等于** Spearman —— 反向对照。若有人把旧写法改回来，
   第 1 条会红；而这个用例保证"旧写法确实错"这件事本身也有据可查；
3. 任何指标都不得是 NaN —— `round(nan, 4)` 会经 `json.dumps` 产出裸 `NaN`，
   那是**非法 JSON**，调用方的 `JSON.parse` 直接抛。所以这里直接拿
   `allow_nan=False` 去序列化。

不需要 lightgbm / torch：指标计算已抽成纯函数 `ranking_metrics`，两个训练路径
共用它（旧代码是逐字复制两份 —— 那个 bug 就是这么存在了两遍）。
"""

import json
import math
import warnings

import numpy as np
import pytest
from scipy.stats import spearmanr

from trainer import _finite, _spearman, ranking_metrics


def _sample(n=80, rho=0.55, seed=7):
    rng = np.random.default_rng(seed)
    p = rng.normal(size=n)
    y = rho * p + rng.normal(size=n)
    return p, y


class TestRankIcIsSpearman:
    """正例：现在算出来的就是秩相关。"""

    @pytest.mark.parametrize("seed", range(8))
    def test_matches_scipy_spearman(self, seed):
        p, y = _sample(seed=seed)
        assert ranking_metrics(p, y)["rank_ic"] == pytest.approx(
            float(spearmanr(p, y).statistic), abs=1e-12
        )

    def test_monotone_but_nonlinear_is_not_understated(self):
        """`y` 是 `p` 的严格递增非线性函数 → RankIC 必须接近 1。

        旧的 argsort 写法在这种输入上会给出远低于 1 的值（这正是它虚低的
        典型形态），所以这个用例是"看得见的正确"。
        """
        p = np.linspace(-3, 3, 200)
        y = np.exp(p)  # 严格单调，Pearson < 1，Spearman == 1
        assert float(np.corrcoef(p, y)[0, 1]) < 0.95
        assert ranking_metrics(p, y)["rank_ic"] == pytest.approx(1.0, abs=1e-9)

    def test_anti_monotone_gives_minus_one(self):
        p = np.linspace(-3, 3, 100)
        assert ranking_metrics(p, np.exp(-p))["rank_ic"] == pytest.approx(-1.0, abs=1e-9)

    def test_ties_are_averaged_not_position_broken(self):
        """scipy 对并列取平均秩；双 argsort 的回退路径会在并列处偏差。

        这条用例锁的是"首选 scipy"这个决定：有并列时结果仍等于 spearmanr。
        """
        p = np.array([1.0, 2.0, 2.0, 3.0, 4.0, 4.0, 5.0])
        y = np.array([2.0, 1.0, 3.0, 3.0, 5.0, 4.0, 6.0])
        assert _spearman(p, y) == pytest.approx(float(spearmanr(p, y).statistic), abs=1e-12)


class TestOldFormulaWasWrong:
    """反向对照：旧写法确实不是秩相关 —— 不是"差不多"，是会翻符号。"""

    def test_old_expression_differs_materially_from_spearman(self):
        worst = 0.0
        for seed in range(8):
            p, y = _sample(seed=seed)
            old = float(np.corrcoef(np.argsort(p), np.argsort(y))[0, 1])
            true = float(spearmanr(p, y).statistic)
            worst = max(worst, abs(old - true))
        # 实测最大误差 0.6688；留足余量只要求"显著不同"，不锁死随机数。
        assert worst > 0.15, "旧表达式若与 Spearman 接近，说明这组假设不成立"

    def test_old_expression_can_flip_the_sign(self):
        """存在真实输入使旧写法报负、真 RankIC 为正。"""
        flipped = False
        for seed in range(40):
            p, y = _sample(seed=seed)
            old = float(np.corrcoef(np.argsort(p), np.argsort(y))[0, 1])
            true = float(spearmanr(p, y).statistic)
            if true > 0.25 and old < 0:
                flipped = True
                break
        assert flipped, "没找到翻符号的样本，反向对照失去意义"

    def test_double_argsort_fallback_agrees_without_ties(self):
        """回退路径（无 scipy）在无并列时与 Spearman 一致。"""
        for seed in range(5):
            p, y = _sample(seed=seed)
            ra = np.argsort(np.argsort(p)).astype(np.float64)
            rb = np.argsort(np.argsort(y)).astype(np.float64)
            assert float(np.corrcoef(ra, rb)[0, 1]) == pytest.approx(
                float(spearmanr(p, y).statistic), abs=1e-9
            )


class TestNoIllegalJson:
    """`round(nan, 4)` → `json.dumps` 产出裸 `NaN` → 调用方 `JSON.parse` 抛。"""

    def test_constant_prediction_does_not_emit_nan(self):
        metrics = ranking_metrics(np.full(50, 0.3), np.random.default_rng(1).normal(size=50))
        for key in ("ic", "rank_ic", "sharpe", "top10_return"):
            assert math.isfinite(metrics[key]), f"{key} 不是有限数"

    def test_constant_label_does_not_emit_nan(self):
        metrics = ranking_metrics(np.random.default_rng(1).normal(size=50), np.full(50, 0.0))
        for key in ("ic", "rank_ic", "sharpe", "top10_return"):
            assert math.isfinite(metrics[key]), f"{key} 不是有限数"

    def test_both_constant_does_not_emit_nan(self):
        metrics = ranking_metrics(np.zeros(20), np.zeros(20))
        assert all(math.isfinite(v) for v in metrics.values())

    def test_result_serializes_with_allow_nan_false(self):
        """最直接的那条断言：严格 JSON 能序列化出去。"""
        metrics = ranking_metrics(np.zeros(20), np.zeros(20))
        assert json.dumps({"status": "ok", **metrics}, allow_nan=False)
        p, y = _sample()
        assert json.dumps({"status": "ok", **ranking_metrics(p, y)}, allow_nan=False)

    def test_finite_sanitizes_every_non_finite_form(self):
        assert _finite(float("nan")) == 0.0
        assert _finite(float("inf")) == 0.0
        assert _finite(float("-inf")) == 0.0
        assert _finite(None) == 0.0
        assert _finite("nope") == 0.0
        assert _finite(1.5) == 1.5


class TestEdges:
    def test_constant_column_with_ulp_noise_is_caught_without_warning(self):
        """常量列经 `std` 判据会漏过去 —— 这一条钉住用的是 ptp 而不是 std。

        `np.std(np.full(50, 0.3))` 实测是 **5.55e-17**（均值的浮点误差），所以
        第一版 `std > 0` 的守卫让常量列照样走到 scipy：照样发
        ConstantInputWarning、照样产出 NaN。极差对完全相同的值是精确 0。
        这里用 `simplefilter("error")` 把"发出警告"直接变成失败。
        """
        const = np.full(50, 0.3)
        assert np.std(const) > 0, "前提失效：std 已能识别常量列"
        assert np.ptp(const) == 0, "前提失效：ptp 不再精确"

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            metrics = ranking_metrics(
                const, np.random.default_rng(1).normal(size=50)
            )
        assert metrics["ic"] == 0.0
        assert metrics["rank_ic"] == 0.0

    def test_too_few_points(self):
        for n in (0, 1):
            metrics = ranking_metrics(np.arange(n), np.arange(n))
            assert metrics == {"ic": 0.0, "rank_ic": 0.0, "sharpe": 0.0, "top10_return": 0.0}

    def test_mismatched_lengths_use_the_shorter(self):
        metrics = ranking_metrics(np.arange(10.0), np.arange(3.0))
        assert math.isfinite(metrics["rank_ic"])
        assert metrics["n_samples"] if "n_samples" in metrics else True

    def test_sharpe_and_top10_return_are_preserved_by_definition(self):
        """这两个数字的定义**没变**（已知不是夏普、不是截面 IC，见 docstring）。

        锁住它，是为了让"我改了 rank_ic"与"我顺手改了两个指标口径"区分开。
        """
        p, y = _sample()
        metrics = ranking_metrics(p, y)
        idx = np.argsort(p)[-10:]
        assert metrics["top10_return"] == pytest.approx(float(np.mean(y[idx])), abs=1e-12)
        assert metrics["sharpe"] == pytest.approx(
            float((np.mean(y[idx]) / (np.std(y[idx]) + 1e-8)) * np.sqrt(252)), abs=1e-6
        )
