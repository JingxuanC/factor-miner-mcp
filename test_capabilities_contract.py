"""`capabilities` 的自述不得与实际不符。

工具自述就是 Agent 的接口契约。审计实测（2026-10-13）：`capabilities` 发布的
`async_tools` 只有 **5** 个，而 `server.ASYNC_TOOLS` 是 **7** 个 —— 漏报了
`factor_evaluate` 与 `ml_train_rolling`。两处各存一份字面量，漂移了没人发现。

现在的结构是：`ASYNC_TOOLS` 只在 `tools.py` 定义一次，`capabilities` 渲染它、
`server.py` import 它。这个文件钉住那条不变量，以及第一个真实受害者
`factor_recent_ic`（它同步跑最长约 150s 占 HTTP 线程）。
"""

import json
import re
from pathlib import Path

import pytest

import server
import tools

ROOT = Path(__file__).resolve().parent

def _capabilities() -> dict:
    return json.loads(tools.HANDLERS["capabilities"]())


class TestSingleSourceOfTruth:
    def test_server_reexports_the_same_object(self):
        """不是"两份内容相同的字面量"，是同一个对象 —— 这才防得住漂移。"""
        assert server.ASYNC_TOOLS is tools.ASYNC_TOOLS

    def test_capability_card_matches(self):
        assert _capabilities()["async_tools"] == sorted(tools.ASYNC_TOOLS)

    def test_async_set_is_immutable(self):
        """frozenset：防止某个模块运行时往公共清单里塞一个工具。"""
        assert isinstance(tools.ASYNC_TOOLS, frozenset)


class TestAsyncListIsActionable:
    def test_every_async_tool_has_a_handler(self):
        """列成异步但网关找不到 handler，提交时会 500。"""
        missing = sorted(t for t in tools.ASYNC_TOOLS if t not in tools.HANDLERS)
        assert missing == [], f"async 清单里有没注册的工具: {missing}"

    def test_every_async_tool_is_advertised(self):
        """列成异步但 /tools 里查不到，调用方根本不知道能调它。"""
        advertised = set(tools.TOOLS) | set(tools.EXTRA_SCHEMAS)
        missing = sorted(t for t in tools.ASYNC_TOOLS if t not in advertised)
        assert missing == [], f"async 清单里有未对外声明的工具: {missing}"

    @pytest.mark.parametrize("name", ["factor_evaluate", "ml_train_rolling"])
    def test_the_two_the_card_used_to_omit(self, name):
        """回归：这两个曾经只躺在 server.py 里、没进 capabilities。"""
        assert name in tools.ASYNC_TOOLS
        assert name in _capabilities()["async_tools"]


class TestFactorRecentIcIsNotLight:
    def test_it_is_async(self):
        """真实事故：它整读 ~710MiB 全量面板（~27s）+ 沙箱 120s 超时。

        同步跑最长约 150s 占着一个 HTTP 线程，而 `server.py` 自己写的设计原则是
        「重负载工具……不占 HTTP 连接」。README:100 也已自认它与 `factor_execute`
        的开销同源 —— 而那个是异步的。
        """
        assert "factor_recent_ic" in tools.ASYNC_TOOLS

    def test_it_is_advertised_as_async(self):
        assert "factor_recent_ic" in _capabilities()["async_tools"]

    def test_gateway_would_enqueue_it(self):
        """网关的分支条件就是 `tool_name in ASYNC_TOOLS and job_queue`。"""
        assert "factor_recent_ic" in server.ASYNC_TOOLS


class TestToolCountIsNotLiedAbout:
    """README 与 tools.py docstring 都写死过数量，然后漂了。

    tools.py 说"7 个工具"（摘出来时是真的，后来 22 个）；README 说"17 个"、
    "/tools 应返回 16 个"。数字会漂 —— 所以这里不靠"记得改"，直接对账。
    """

    def test_tools_docstring_points_at_the_live_source_of_truth(self):
        doc = tools.__doc__ or ""
        assert "capabilities" in doc, "工具数量会漂；docstring 应指向 capabilities 而不是写死"

    def test_readme_advertised_count_matches_reality(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        live = len(tools.TOOLS) + len(tools.EXTRA_SCHEMAS)
        m = re.search(r"#\s*应返回\s*(\d+)\s*个工具", readme)
        assert m, "README 的 curl 示例里找不到工具数量断言"
        assert int(m.group(1)) == live, (
            "README 说 /tools 返回 %s 个，实际 %d 个（Tools=%d + ExtraSchemas=%d）"
            % (m.group(1), live, len(tools.TOOLS), len(tools.EXTRA_SCHEMAS)))

    def test_readme_heading_count_matches_total_callable(self):
        """总数 = /tools 的 + job_status（后者只在 /mcp 注册、不走 /tools）。"""
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        total = len(tools.TOOLS) + len(tools.EXTRA_SCHEMAS) + 1
        m = re.search(r"##\s*工具清单（(\d+)\s*个）", readme)
        assert m, "README 的『工具清单』小节找不到数量"
        assert int(m.group(1)) == total, (
            "README 标题说 %s 个，实际可调用 %d 个" % (m.group(1), total))

    def test_job_status_is_the_one_not_in_tools(self):
        """上面那条 +1 的理由要成立：job_status 确实不在 /tools 的清单里。"""
        advertised = set(tools.TOOLS) | set(tools.EXTRA_SCHEMAS)
        assert "job_status" not in advertised
        assert server.JOB_STATUS_SCHEMA["name"] == "job_status"
        assert server.JOB_STATUS_SCHEMA not in tools.TOOLS.values()
