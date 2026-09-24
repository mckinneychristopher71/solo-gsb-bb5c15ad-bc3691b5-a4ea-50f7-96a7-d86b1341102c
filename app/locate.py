"""在修订文本中近似定位引语。

在修订文本里枚举与引语的 Levenshtein 编辑距离（插入/删除/替换）不超过
上限（0～2）的所有非空连续片段，再通过带权区间调度全局选出**互不重叠**
的命中集合：依次最大化命中数、最小化编辑次数总和，仍同分时取下标序列
字典序最小，按原文顺序输出。

每个命中的起止毫秒取自逐字时间轴首字符的 ``start`` 与末字符的 ``end``。
"""
from __future__ import annotations

from .schemas import RevisedCharTimeline

# 重复窗口的距离缓存上限（如高度重复/周期性文本，可避免重复计算）。
_CACHE_LIMIT = 2048


def _banded_distance(pattern: str, window: str, cap: int) -> int:
    """带对角带宽的 Levenshtein 距离；结果超过 cap 时直接返回 cap。

    任何编辑代价 ≤ cap-1 的对齐路径，其行列差始终落在 ±(cap-1) 之内，
    因此只需计算宽 2*cap-1 的对角带；cap = max_edits + 1 ≤ 3，开销很小。
    """
    max_edits = cap - 1
    k = len(pattern)
    prev_lo = 0
    prev_hi = min(k, max_edits)
    prev = [j for j in range(prev_hi + 1)]  # 第 0 行：空窗口前缀

    for r in range(1, len(window) + 1):
        lo = max(0, r - max_edits)
        hi = min(k, r + max_edits)
        cur = [cap] * (hi - lo + 1)
        wc = window[r - 1]
        alive = False

        for j in range(lo, hi + 1):
            best = cap

            # 匹配/替换：(r-1, j-1)
            if prev_lo <= j - 1 <= prev_hi:
                cost = prev[j - 1 - prev_lo] + (0 if wc == pattern[j - 1] else 1)
                if cost < best:
                    best = cost
            # 删除识别（窗口侧）字符：(r-1, j)
            if prev_lo <= j <= prev_hi:
                cost = prev[j - prev_lo] + 1
                if cost < best:
                    best = cost
            # 插入引语字符：(r, j-1)
            if j - 1 >= lo:
                cost = cur[j - 1 - lo] + 1
                if cost < best:
                    best = cost

            if best < cap:
                alive = True
            cur[j - lo] = best

        if not alive:
            return cap
        prev, prev_lo, prev_hi = cur, lo, hi

    if prev_lo <= k <= prev_hi:
        return prev[k - prev_lo]
    return cap


def locate_quotes(chars: list[RevisedCharTimeline], text: str,
                  quote: str, max_edits: int) -> list[dict]:
    """返回互不重叠的近似引语命中（见模块说明）。"""
    n = len(text)
    k = len(quote)
    # starts[s]：以 s 为起点的候选片段列表，按终点升序，元素 (end, distance)。
    starts: list[list[tuple[int, int]]] = [[] for _ in range(n)]

    if n > 0 and k > 0:
        if max_edits == 0:
            # 精确匹配：借助底层快速子串搜索枚举所有（可重叠）出现位置。
            pos = text.find(quote)
            while pos != -1:
                starts[pos].append((pos + k, 0))
                pos = text.find(quote, pos + 1)
        else:
            min_len = max(1, k - max_edits)
            max_len = min(k + max_edits, n)
            cap = max_edits + 1
            cache: dict[str, int] = {}

            for length in range(min_len, max_len + 1):
                for s in range(n - length + 1):
                    window = text[s:s + length]
                    distance = cache.get(window)
                    if distance is None:
                        distance = _banded_distance(quote, window, cap)
                        if len(cache) < _CACHE_LIMIT:
                            cache[window] = distance
                    if distance <= max_edits:
                        starts[s].append((s + length, distance))

    # 后缀最优：best[p] = 仅使用起点 >= p 的片段时，
    # （最大命中数, 最小编辑次数总和）。
    count = [0] * (n + 1)
    cost = [0] * (n + 1)
    for p in range(n - 1, -1, -1):
        best_count = count[p + 1]   # 跳过位置 p
        best_cost = cost[p + 1]
        for end, distance in starts[p]:
            cand_count = 1 + count[end]
            cand_cost = distance + cost[end]
            if (cand_count > best_count
                    or (cand_count == best_count and cand_cost < best_cost)):
                best_count, best_cost = cand_count, cand_cost
        count[p], cost[p] = best_count, best_cost

    # 贪心构造同分情况下字典序最小的下标序列：从左到右选择可行的最早
    # (起点, 终点) 片段；后缀 DP 保证其后仍能拼出全局最优。
    hits: list[dict] = []
    pos = 0
    while count[pos] > 0:
        target_count, target_cost = count[pos], cost[pos]
        s = pos
        while True:
            if count[s] == target_count and cost[s] == target_cost:
                chosen: tuple[int, int, int] | None = None
                for end, distance in starts[s]:
                    if (1 + count[end] == count[s]
                            and distance + cost[end] == cost[s]):
                        chosen = (s, end, distance)
                        break
                if chosen is not None:
                    start, end, distance = chosen
                    hits.append({
                        "start": start,
                        "end": end,
                        "text": text[start:end],
                        "edits": distance,
                        "start_ms": chars[start].start,
                        "end_ms": chars[end - 1].end,
                    })
                    pos = end
                    break
            s += 1

    return hits
