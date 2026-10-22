"""配置层：所有凭据与外部依赖只从环境变量读取（硬约束 C6）。

为什么单独一个文件：凭据散落在业务代码里是安全事故的第一大来源。
集中到一处后，"有没有硬编码密钥"这个问题可以用一条 grep 回答。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def load_dotenv(path: str | None = None) -> None:
    """把项目根目录的 .env 逐行塞进环境变量。

    为什么放在配置层而不是某个入口脚本里：
    每个入口（CLI、评测脚本、测试）都需要它。曾经因为评测脚本漏掉这一句，
    在 Settings.from_env() 处抛「缺少 LLM_API_KEY」——复制两份加载逻辑
    必然会出现改一个忘一个的情况，所以只保留这一份。

    为什么用 setdefault：命令行显式设置的环境变量优先级更高，不被 .env 覆盖，
    方便临时换模型或注入预算做实验。

    默认路径基于本文件位置推导（agent/config.py 的上两级即项目根），
    因此从任意工作目录启动都能找到 .env。
    """
    p = Path(path) if path else Path(__file__).resolve().parent.parent / ".env"
    if not p.exists():
        return
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        os.environ.setdefault(key.strip(), val.strip())


def _env(name: str, default: str | None = None, required: bool = False) -> str:
    """读环境变量。required=True 时缺失直接抛错——早失败胜过线上 401。"""
    val = os.getenv(name, default)
    if required and not val:
        raise RuntimeError(f"缺少必需环境变量: {name}（请先 cp .env.example .env 并填写）")
    return val or ""


@dataclass(frozen=True)
class Settings:
    # —— LLM 接入：走 OpenAI 兼容协议，DeepSeek / 通义 / 智谱 / Kimi 均支持 ——
    llm_api_key: str          # 禁止硬编码，禁止打印
    llm_base_url: str         # 例如 https://api.deepseek.com/v1
    llm_model: str            # 主模型
    llm_fallback_model: str   # 降级模型，主模型连续失败时切换
    llm_temperature: float    # 采样温度：0=确定性，>0=允许随机；评测/生产关键路径建议 0
    memory_db_path: str       # 记忆库路径；评测用隔离库避免污染真实对话数据
    loop_impl: str            # 自主循环实现：plain=手写 for 循环（默认）/ graph=LangGraph 状态图

    # —— 超时（硬约束 C4：必须显式数值，禁止依赖客户端默认值）——
    llm_timeout_s: float        # 单次 LLM 请求超时
    tool_timeout_s: float       # 工具侧超时

    # —— 有界自主（硬约束 C5 四件套）——
    max_steps: int              # 自主循环步数上限
    max_tokens_budget: int      # 单次任务 token 预算
    wall_clock_budget_s: float  # 墙钟上限

    # —— 计费（用于成本观测；价格以官方为准，见 README 待核实清单）——
    price_per_1k_input: float
    price_per_1k_output: float

    # —— 输入校验上限 ——
    max_input_chars: int

    # —— 语义缓存（见 agent/cache.py：用意思相似复用历史答案，省 LLM 调用）——
    cache_enabled: bool          # 总开关：false=完全关闭，命中逻辑直接降级为未命中
    cache_backend: str          # memory=进程内（零依赖）/ upstash=云 Redis（零本地安装）
    cache_ttl_s: int            # 单条缓存存活秒数，到期自动失效，避免政策更新后吐旧答案
    cache_sim_threshold: float  # 相似度阈值：超过才算命中，越高越保守（防错答）
    upstash_url: str            # Upstash Redis REST 地址（CACHE_BACKEND=upstash 时必填）
    upstash_token: str          # Upstash Redis REST 令牌（禁止硬编码、禁止打印）

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            llm_api_key=_env("LLM_API_KEY", required=True),
            llm_base_url=_env("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            llm_model=_env("LLM_MODEL", "deepseek-flash"),
            llm_fallback_model=_env("LLM_FALLBACK_MODEL", "deepseek-flash"),
            llm_temperature=float(_env("LLM_TEMPERATURE", "0.2")),
            memory_db_path=_env("MEMORY_DB_PATH", "data/memory.db"),
            loop_impl=_env("AGENT_LOOP_IMPL", "plain"),
            llm_timeout_s=float(_env("LLM_TIMEOUT_S", "30")),
            tool_timeout_s=float(_env("TOOL_TIMEOUT_S", "5")),
            max_steps=int(_env("MAX_STEPS", "6")),
            max_tokens_budget=int(_env("MAX_TOKENS_BUDGET", "20000")),
            wall_clock_budget_s=float(_env("WALL_CLOCK_BUDGET_S", "45")),
            price_per_1k_input=float(_env("PRICE_PER_1K_INPUT", "0")),
            price_per_1k_output=float(_env("PRICE_PER_1K_OUTPUT", "0")),
            max_input_chars=int(_env("MAX_INPUT_CHARS", "2000")),
            cache_enabled=_env("CACHE_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            cache_backend=_env("CACHE_BACKEND", "memory"),
            cache_ttl_s=int(_env("CACHE_TTL_S", "86400")),
            cache_sim_threshold=float(_env("CACHE_SIM_THRESHOLD", "0.82")),
            upstash_url=_env("UPSTASH_REDIS_REST_URL", ""),
            upstash_token=_env("UPSTASH_REDIS_REST_TOKEN", ""),
        )
