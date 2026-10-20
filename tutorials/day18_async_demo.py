"""day18：实测 FastAPI 并发 —— 为什么本项目路由用 def 而不是 async def。

直接跑（脚本会自动起临时服务、自动关，不碰你已经在跑的 8000 端口）：
    python tutorials/day18_async_demo.py

背景：你的 Agent 一次请求要 2~5 秒（LLM 调用）。多个用户同时问问题时，
写法不同，结局天差地别。三种写法各发 2 个并发请求，看总耗时：

① GET /sync       def + time.sleep(2)             → FastAPI 丢线程池，并行 ≈ 2 秒 ✅
② GET /blocking   async def + time.sleep(2)       → 卡死事件循环，排队 ≈ 4 秒 ❌
③ GET /proper     async def + await asyncio.sleep → 正确的异步，并行 ≈ 2 秒 ✅

结论（面试这么说）：
- async def 只有在「内部全程 await 非阻塞调用」时才有并发收益；
  里面塞任何一个阻塞调用（time.sleep / requests.get / 同步版 openai 客户端），
  都会把整个事件循环堵死，所有请求排队。
- 我们的 Agent 用的是同步阻塞的 openai 客户端，所以老老实实用 def：
  FastAPI 会自动把它丢进线程池，每个请求占一个线程，慢请求互不干扰。
- 想真上 async，要把 LLM 客户端换成异步版（AsyncOpenAI）并全程 await，
  那是另一个改造量级。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import uvicorn
from fastapi import FastAPI

PORT = 8001          # 临时端口，避开你正常跑的 8000
SLEEP = 2            # 模拟一次慢调用（类比 LLM 耗时）
BASE = f"http://127.0.0.1:{PORT}"

app = FastAPI()


# ---------------- 三种写法，对照用 ----------------


@app.get("/sync")
def sync_endpoint() -> dict:
    """普通 def + 阻塞调用：FastAPI 自动丢进线程池 → 多个请求并行。"""
    time.sleep(SLEEP)
    return {"mode": "def + time.sleep（线程池，推荐）"}


@app.get("/blocking")
async def blocking_endpoint() -> dict:
    """反模式：async def 里塞阻塞调用 → 堵死事件循环，所有请求排队。"""
    time.sleep(SLEEP)  # ← 罪魁祸首：这里的阻塞会让整个服务停摆
    return {"mode": "async def + time.sleep（反模式）"}


@app.get("/proper")
async def proper_endpoint() -> dict:
    """正确的异步写法：全程 await 非阻塞调用。"""
    await asyncio.sleep(SLEEP)
    return {"mode": "async def + await asyncio.sleep（正确异步）"}


# ---------------- 测量 ----------------


def wait_ready(timeout: float = 30) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            urllib.request.urlopen(f"{BASE}/sync", timeout=5)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def measure(path: str, n: int = 2) -> tuple[float, float]:
    """对同一接口并发发 n 个请求，返回（总耗时, 单个最长耗时）。"""

    def one(_: int) -> float:
        t0 = time.perf_counter()
        with urllib.request.urlopen(f"{BASE}{path}", timeout=60) as resp:
            json.loads(resp.read())
        return time.perf_counter() - t0

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        latencies = list(pool.map(one, range(n)))
    wall = time.perf_counter() - started
    return wall, max(latencies)


def main() -> int:
    print("=" * 66)
    print("FastAPI 并发实测：def vs async def（每个请求模拟耗时 2 秒）")
    print("=" * 66)

    config = uvicorn.Config(app, host="127.0.0.1", port=PORT, log_level="error")
    server = uvicorn.Server(config)
    t = threading.Thread(target=server.run, daemon=True)
    t.start()

    if not wait_ready():
        print(f"❌ 临时服务未能在 30 秒内就绪（端口 {PORT} 可能被占用）")
        server.should_exit = True
        return 1
    print(f"临时服务已就绪: {BASE}\n")

    cases = [
        ("/sync", "① def + 阻塞调用        （线程池，本项目用法）"),
        ("/blocking", "② async def + 阻塞调用  （反模式）"),
        ("/proper", "③ async def + await      （正确异步）"),
    ]

    results = []
    for path, label in cases:
        wall, longest = measure(path, n=2)
        results.append((label, wall, longest))
        # 判定基准必须是「不变量 SLEEP」，不能用观测到的单个耗时：
        # 请求一旦排队，客户端观测到的单条延迟也会跟着涨到 4 秒，
        # 拿它当基准就会把「排队」误判成「并行」（这个脚本的第一版就错了）。
        parallel = wall < SLEEP * 1.5
        print(f"{label}")
        print(f"    2 个并发请求总耗时 : {wall:.2f} 秒   （单个耗时 {longest:.2f} 秒）")
        print(f"    基准               : 并行约 {SLEEP:.0f} 秒 / 排队约 {SLEEP * 2:.0f} 秒")
        print(f"    判断               : {'并行 ✅' if parallel else '排队 ❌'}")
        print()

    server.should_exit = True

    print("=" * 66)
    print("结论：")
    print("- ② 的总耗时接近 ① 的两倍 —— async 里塞阻塞调用会让请求排队，")
    print("  这是 FastAPI 最常见的性能坑。")
    print("- ① 用普通 def，FastAPI 自动丢线程池，拿到和正确异步一样的并行效果，")
    print("  而我们不用改一行 Agent 代码。对本项目的同步 LLM 客户端，这就是最优解。")
    print("=" * 66)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
