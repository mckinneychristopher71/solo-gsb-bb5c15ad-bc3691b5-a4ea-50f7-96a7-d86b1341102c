"""端到端 API 与时间轴语义测试（native 与 python 两种内核均会覆盖）。"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.alignment import align_chars  # noqa: E402
from app.main import app  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402


def rc(items):
    return [{"char": c, "start": a, "end": b} for c, a, b in items]


@pytest.fixture()
def client():
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# 对齐内核：重复字符结合上下文选择匹配位置，同分优先较早原位置
# ---------------------------------------------------------------------------
def test_duplicate_chars_prefer_earlier_source():
    # aaa -> aa：保留前两个（较早原位置）
    assert align_chars("aaa", "aa") == [0, 1]
    # aa -> aaa：新增字符落在末尾
    assert align_chars("aa", "aaa") == [0, 1, -1]


def test_substitution_is_anchored():
    # 替换纠错优先锚定到识别字符时间槽，而不是全部走删除+新增
    mapping = align_chars("识别错误文字", "识别正确文字")
    assert mapping[2] == 2  # “正” 替换 “错”，锚定下标 2
    assert mapping[3] == 3  # “确” 替换 “误”


def test_insertion_and_deletion():
    mapping = align_chars("你好吗世界", "你好世界")
    assert mapping == [0, 1, 3, 4]  # “吗” 被删除


# ---------------------------------------------------------------------------
# 健康检查
# ---------------------------------------------------------------------------
def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["backend"] in ("native", "python")


# ---------------------------------------------------------------------------
# 基本对齐与新增字符空档均分
# ---------------------------------------------------------------------------
def test_basic_align_with_interpolation(client):
    payload = {
        "recognized": rc([
            ("你", 0, 100), ("好", 200, 300), ("世", 400, 500), ("界", 600, 700),
        ]),
        "total_duration": 1000,
        "revised_text": "你好呀世界",
    }
    r = client.post("/align", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    chars = body["chars"]
    assert [c["char"] for c in chars] == list("你好呀世界")
    assert [c["source"] for c in chars] == [
        "matched", "matched", "computed", "matched", "matched"]

    ni = chars[2]
    # “呀”位于“好”结束(300)与“世”开始(400)之间
    assert ni["start"] == 300 and ni["end"] == 400
    # matched 锚点时间原样保留
    assert (chars[0]["start"], chars[0]["end"]) == (0, 100)
    assert body["inserted_count"] == 1
    assert body["deleted_count"] == 0
    assert body["matched_count"] == 4


def test_multiple_insertions_evenly_split(client):
    # 锚点 a 与 c 之间没有任何识别字符，x、y 为两个纯新增字符，空档 [100,600]
    payload = {
        "recognized": rc([("a", 0, 100), ("c", 600, 700)]),
        "total_duration": 1000,
        "revised_text": "axyc",
    }
    r = client.post("/align", json=payload)
    assert r.status_code == 200, r.text
    x, y = r.json()["chars"][1], r.json()["chars"][2]
    assert x["source"] == "computed" and y["source"] == "computed"
    # [100,600] 分两段：100..350, 350..600
    assert (x["start"], x["end"]) == (100, 350)
    assert (y["start"], y["end"]) == (350, 600)


def test_substitution_tie_anchors_into_gap(client):
    # a b c -> a x y c：x 与 b 替换代价相同且优先锚定，y 落在 b 与 c 的空档
    payload = {
        "recognized": rc([("a", 0, 100), ("b", 200, 300), ("c", 600, 700)]),
        "total_duration": 1000,
        "revised_text": "axyc",
    }
    r = client.post("/align", json=payload)
    assert r.status_code == 200, r.text
    x, y = r.json()["chars"][1], r.json()["chars"][2]
    assert x["source"] == "matched" and x["recognized_char"] == "b"
    assert (x["start"], x["end"]) == (200, 300)
    assert y["source"] == "computed"
    assert (y["start"], y["end"]) == (300, 600)


def test_leading_and_trailing_insertions_use_recording_bounds(client):
    payload = {
        "recognized": rc([("b", 400, 500)]),
        "total_duration": 1000,
        "revised_text": "abc",
    }
    r = client.post("/align", json=payload)
    assert r.status_code == 200, r.text
    a, b, c = r.json()["chars"]
    # 前缀“a”在录音起点 0 到 b 开始 400 之间
    assert (a["start"], a["end"]) == (0, 400)
    # 后缀“c”在 b 结束 500 到录音总时长 1000 之间
    assert (c["start"], c["end"]) == (500, 1000)
    assert b["source"] == "matched"


def test_deleted_chars_not_output(client):
    payload = {
        "recognized": rc([("a", 0, 100), ("x", 200, 300), ("b", 400, 500)]),
        "total_duration": 600,
        "revised_text": "ab",
    }
    r = client.post("/align", json=payload)
    body = r.json()
    assert [c["char"] for c in body["chars"]] == ["a", "b"]
    assert body["deleted_count"] == 1


def test_substitution_marked_matched_with_recognized_char(client):
    payload = {
        "recognized": rc([("错", 0, 100)]),
        "total_duration": 200,
        "revised_text": "对",
    }
    r = client.post("/align", json=payload)
    body = r.json()
    ch = body["chars"][0]
    assert ch["char"] == "对"
    assert ch["source"] == "matched"
    assert ch["recognized_char"] == "错"
    assert (ch["start"], ch["end"]) == (0, 100)


def test_all_inserted_spans_whole_recording(client):
    payload = {"recognized": [], "total_duration": 300, "revised_text": "三"}
    r = client.post("/align", json=payload)
    ch = r.json()["chars"][0]
    assert ch["source"] == "computed"
    assert (ch["start"], ch["end"]) == (0, 300)


def test_empty_revised_text(client):
    payload = {
        "recognized": rc([("a", 0, 100)]),
        "total_duration": 200,
        "revised_text": "",
    }
    r = client.post("/align", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["chars"] == []
    assert body["deleted_count"] == 1


def test_timestamps_ordered_and_in_range(client):
    payload = {
        "recognized": rc([("a", 0, 10), ("b", 20, 30), ("c", 40, 50)]),
        "total_duration": 100,
        "revised_text": "xaxbxcx",
    }
    r = client.post("/align", json=payload)
    assert r.status_code == 200, r.text
    prev_end = 0
    for c in r.json()["chars"]:
        assert 0 <= c["start"] <= c["end"] <= 100
        assert c["start"] >= prev_end
        prev_end = c["end"]


def test_zero_duration(client):
    payload = {"recognized": rc([("a", 0, 0)]), "total_duration": 0,
               "revised_text": "a"}
    r = client.post("/align", json=payload)
    assert r.status_code == 200
    ch = r.json()["chars"][0]
    assert (ch["start"], ch["end"]) == (0, 0)


def test_duplicate_resolved_by_context_to_earlier_source(client):
    # 识别串里出现两个“对”。修订文本在“错”前补“对”、“错”后保留“对”：
    # 第一个“对”必须结合上下文匹配到更早的识别位置 0，而不是误配位置 2。
    payload = {
        "recognized": rc([
            ("对", 0, 100), ("错", 200, 300), ("对", 400, 500),
        ]),
        "total_duration": 600,
        "revised_text": "对错对",
    }
    r = client.post("/align", json=payload)
    assert r.status_code == 200, r.text
    chars = r.json()["chars"]
    assert [c["matched_index"] for c in chars] == [0, 1, 2]


def test_duplicate_identical_target_keeps_earlier_slots(client):
    # aaa（三个相同识别字） -> aa：删除多余重复字符时，保留最早的两个时间槽
    payload = {
        "recognized": rc([("a", 0, 10), ("a", 20, 30), ("a", 40, 50)]),
        "total_duration": 60,
        "revised_text": "aa",
    }
    r = client.post("/align", json=payload)
    chars = r.json()["chars"]
    assert [c["matched_index"] for c in chars] == [0, 1]
    assert (chars[0]["start"], chars[1]["start"]) == (0, 20)
    assert r.json()["deleted_count"] == 1


# ---------------------------------------------------------------------------
# 输入校验：时间无效时给出明确错误
# ---------------------------------------------------------------------------
def _post(client, payload):
    return client.post("/align", json=payload)


def test_negative_total_duration(client):
    r = _post(client, {"recognized": [], "total_duration": -1,
                       "revised_text": "a"})
    assert r.status_code == 422
    assert "非负" in r.json()["detail"]


def test_end_before_start(client):
    r = _post(client, {
        "recognized": rc([("a", 300, 100)]),
        "total_duration": 500, "revised_text": "a"})
    assert r.status_code == 422
    assert "结束时间" in r.json()["detail"]


def test_out_of_recording_range(client):
    r = _post(client, {
        "recognized": rc([("a", 100, 600)]),
        "total_duration": 500, "revised_text": "a"})
    assert r.status_code == 422
    assert "超出录音总时长" in r.json()["detail"]


def test_not_sorted(client):
    r = _post(client, {
        "recognized": rc([("b", 300, 400), ("a", 0, 100)]),
        "total_duration": 500, "revised_text": "ab"})
    assert r.status_code == 422
    assert "排序" in r.json()["detail"]


def test_overlapping_intervals(client):
    r = _post(client, {
        "recognized": rc([("a", 0, 200), ("b", 100, 300)]),
        "total_duration": 500, "revised_text": "ab"})
    assert r.status_code == 422
    assert "重叠" in r.json()["detail"]


def test_negative_timestamp(client):
    r = _post(client, {
        "recognized": rc([("a", -5, 100)]),
        "total_duration": 500, "revised_text": "a"})
    assert r.status_code == 422
    assert "负数" in r.json()["detail"]


def test_missing_field(client):
    r = client.post("/align", json={"recognized": [], "total_duration": 100})
    assert r.status_code == 422
    assert "revised_text" in r.json()["detail"]


def test_wrong_type_timestamp(client):
    r = client.post("/align", json={
        "recognized": [{"char": "a", "start": "0", "end": 100}],
        "total_duration": 500, "revised_text": "a"})
    assert r.status_code == 422
    assert "整数" in r.json()["detail"]


def test_multi_char_recognized_item(client):
    r = _post(client, {
        "recognized": [{"char": "ab", "start": 0, "end": 100}],
        "total_duration": 500, "revised_text": "ab"})
    assert r.status_code == 422
    assert "1 个字符" in r.json()["detail"]


def test_float_duration_rejected(client):
    r = _post(client, {"recognized": [], "total_duration": 1.5,
                       "revised_text": "a"})
    assert r.status_code == 422


def test_extra_field_rejected(client):
    r = _post(client, {"recognized": [], "total_duration": 5,
                       "revised_text": "a", "bogus": 1})
    assert r.status_code == 422
    assert "不允许" in r.json()["detail"]


def test_zero_duration_multiple_inserts(client):
    r = _post(client, {"recognized": [], "total_duration": 0,
                       "revised_text": "甲乙"})
    assert r.status_code == 200
    chars = r.json()["chars"]
    assert [(c["start"], c["end"]) for c in chars] == [(0, 0), (0, 0)]
