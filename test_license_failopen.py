"""鉴权配置的故障方向：**配置了就必须生效，否则拒绝启动**。

真实安全洞（2026-10-13 审计发现）：`LicenseStore` 曾经是 fail-open。

    MCP_LICENSE_FILE 指向一个空文件（或缺失/写坏）
      → json.loads("") 抛错
      → reload() 只 logger.error 然后继续
      → _keys 留在 {}
      → enabled 返回 False
      → 鉴权**完全开放**，服务照常启动，/health 还报 "auth": false

运维以为已经启用，实际裸奔。仓库里 `licenses/licenses.json` 恰好就是 **0 字节**，
所以这不是理论路径。（同一个错误在 hub 侧也犯过一次，见
mcp-suite/AUDIT-2026-09-14.md 的 P0-8「鉴权 fail-open」。）

这个文件钉住修复后的契约：

* 未配置 `MCP_LICENSE_FILE` → 开放模式（本地/内网的**显式**选择）；
* 已配置 → 文件必须存在、是合法 JSON、`keys` 是非空 dict，否则构造即抛
  `LicenseConfigError`。**"配置坏了"永远不得变成"开放"。**
* 运行期重载失败 → 保留上一份可用的 key，继续按旧清单执法。
"""

import json

import pytest

from mcp_gateway import LicenseConfigError, LicenseStore

KEY = "ak_test_key"
GOOD = {"keys": {KEY: {"name": "t", "daily_quota": 0, "heavy_quota": 0}}}


def _write(tmp_path, text, name="licenses.json"):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


class TestOpenModeIsExplicit:
    """没配对文件 = 开放，且这是唯一得到开放模式的途径。"""

    def test_no_file_configured_is_open(self, tmp_path):
        store = LicenseStore("", domain="t", usage_dir=str(tmp_path))
        assert store.enabled is False
        assert store.check("anything") == (True, "")

    def test_open_store_never_touches_disk(self, tmp_path):
        store = LicenseStore("", domain="t", usage_dir=str(tmp_path))
        assert store.consume("anything", heavy=True) == {"mode": "open"}


class TestBrokenConfigRefusesToStart:
    """每一种"配了但不可用"，都必须炸掉，而不是静默开放。"""

    def test_missing_file(self, tmp_path):
        with pytest.raises(LicenseConfigError):
            LicenseStore(str(tmp_path / "nope.json"), domain="t", usage_dir=str(tmp_path))

    def test_zero_byte_file_is_the_real_production_shape(self, tmp_path):
        """仓库里 `licenses/licenses.json` 就是 0 字节 —— 这条是那次事故的复现。"""
        path = _write(tmp_path, "")
        assert path.stat().st_size == 0
        with pytest.raises(LicenseConfigError):
            LicenseStore(str(path), domain="t", usage_dir=str(tmp_path))

    def test_truncated_json(self, tmp_path):
        with pytest.raises(LicenseConfigError):
            LicenseStore(str(_write(tmp_path, '{"keys": {"ak_x"')),
                         domain="t", usage_dir=str(tmp_path))

    def test_json_that_is_not_an_object(self, tmp_path):
        with pytest.raises(LicenseConfigError):
            LicenseStore(str(_write(tmp_path, "[1, 2, 3]")),
                         domain="t", usage_dir=str(tmp_path))

    def test_object_without_keys(self, tmp_path):
        with pytest.raises(LicenseConfigError):
            LicenseStore(str(_write(tmp_path, '{"other": 1}')),
                         domain="t", usage_dir=str(tmp_path))

    def test_empty_keys_dict(self, tmp_path):
        with pytest.raises(LicenseConfigError):
            LicenseStore(str(_write(tmp_path, '{"keys": {}}')),
                         domain="t", usage_dir=str(tmp_path))

    def test_keys_is_not_a_dict(self, tmp_path):
        with pytest.raises(LicenseConfigError):
            LicenseStore(str(_write(tmp_path, '{"keys": ["ak_x"]}')),
                         domain="t", usage_dir=str(tmp_path))

    @pytest.mark.parametrize("bad", ["", "   ", "{}", '{"keys": null}', "not json", '{"keys": {}}'])
    def test_regression_no_broken_shape_ever_yields_open_mode(self, tmp_path, bad):
        """核心回归断言：坏配置的**唯一**结局是异常，绝不是 enabled=False。"""
        path = _write(tmp_path, bad)
        store = None
        try:
            store = LicenseStore(str(path), domain="t", usage_dir=str(tmp_path))
        except LicenseConfigError:
            return  # 预期
        pytest.fail(
            "坏配置换来了 open mode：enabled=%r，check('任一把 key')=%r"
            % (store.enabled, store.check("anything"))
        )

    def test_error_message_says_how_to_fix(self, tmp_path):
        path = _write(tmp_path, "")
        with pytest.raises(LicenseConfigError) as ei:
            LicenseStore(str(path), domain="t", usage_dir=str(tmp_path))
        msg = str(ei.value)
        assert "MCP_LICENSE_FILE" in msg
        assert str(path) in msg
        assert "开放模式" in msg  # 告诉运维另一条路是显式的


class TestValidConfigStillWorks:
    """修复不得误伤正常配置。"""

    def test_valid_file_enforces_auth(self, tmp_path):
        store = LicenseStore(str(_write(tmp_path, json.dumps(GOOD))),
                             domain="t", usage_dir=str(tmp_path))
        assert store.enabled is True
        assert store.check(KEY) == (True, "t")
        ok, why = store.check("ak_wrong")
        assert ok is False and "license key" in why

    def test_example_shape_from_the_repo(self, tmp_path):
        """用仓库里 licenses.example.json 的真实形状再验一次。"""
        example = {"keys": {"ak_REPLACE_WITH_GENERATED_KEY": {
            "name": "owner", "daily_quota": 0, "heavy_quota": 0}}}
        store = LicenseStore(str(_write(tmp_path, json.dumps(example))),
                             domain="t", usage_dir=str(tmp_path))
        assert store.enabled is True
        assert store.check("ak_REPLACE_WITH_GENERATED_KEY")[0] is True


class TestRuntimeReloadFailsClosed:
    """运行期把文件写坏，不得让门在此刻打开。"""

    def test_emptied_file_keeps_the_last_good_keys(self, tmp_path):
        path = _write(tmp_path, json.dumps(GOOD))
        store = LicenseStore(str(path), domain="t", usage_dir=str(tmp_path))
        assert store.check(KEY)[0] is True

        path.write_text("", encoding="utf-8")      # 账号服务写坏了
        store.reload(force=True)

        assert store.enabled is True, "重载失败把鉴权关掉了"
        assert store.check(KEY)[0] is True, "旧清单里的 key 不该突然失效"
        assert store.check("ak_wrong")[0] is False, "门开了"

    def test_deleted_file_keeps_the_last_good_keys(self, tmp_path):
        path = _write(tmp_path, json.dumps(GOOD))
        store = LicenseStore(str(path), domain="t", usage_dir=str(tmp_path))
        path.unlink()
        store.reload(force=True)
        assert store.enabled is True
        assert store.check("ak_wrong")[0] is False

    def test_emptied_keys_keeps_the_last_good_keys(self, tmp_path):
        path = _write(tmp_path, json.dumps(GOOD))
        store = LicenseStore(str(path), domain="t", usage_dir=str(tmp_path))
        path.write_text('{"keys": {}}', encoding="utf-8")
        store.reload(force=True)
        assert store.enabled is True
        assert store.check(KEY)[0] is True

    def test_a_newly_issued_key_still_picks_up(self, tmp_path):
        """重载的正向路径不能被上面的保守措施挡住。"""
        path = _write(tmp_path, json.dumps(GOOD))
        store = LicenseStore(str(path), domain="t", usage_dir=str(tmp_path))
        assert store.check("ak_new")[0] is False
        path.write_text(json.dumps({"keys": {
            KEY: {"name": "t"}, "ak_new": {"name": "new"}}}), encoding="utf-8")
        store.reload(force=True)
        assert store.check("ak_new") == (True, "new")
