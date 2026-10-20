# Dockerfile：把 ticket-agent 打包成「复制到哪儿跑都一样」的环境。
#
# 构建：  docker build -t ticket-agent .
# 运行：  docker run -p 8000:8000 -e LLM_API_KEY=sk-你的密钥 ticket-agent
#
# ⚠️ 安全铁律：LLM_API_KEY 绝对不能写进镜像，也绝不 COPY .env 进去。
#    密钥一旦进了镜像层，任何拿到镜像的人都能把它翻出来（镜像是可以被解包的）。
#    正确做法是「运行时注入环境变量」，镜像本身保持干净。
#    config.py 里用 os.environ.setdefault 读变量，所以运行时注入的真环境变量优先级最高。

# 基准镜像写精确小版本，不用 latest：
# 镜像必须可复现，latest 会随时间漂移，今天能跑明天就报错。
FROM python:3.11-slim

# PYTHONUNBUFFERED：让 print/日志立刻输出，否则容器里看不到实时日志
# PYTHONDONTWRITEBYTECODE：不生成 .pyc，镜像更干净
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# ---- 关键提速技巧：分层缓存 ----
# 先只复制依赖清单并安装依赖。这样只要 requirements.txt 没变，
# 这一层就命中缓存直接复用，不用每次 build 都重装一遍 openai/fastapi。
# 如果把 COPY . . 写在 pip install 前面，改一行代码就会导致整个依赖重装。
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 源码后复制：代码经常变，放最后才能保住上面那层依赖缓存
COPY agent/ ./agent/
COPY app.py server.py ./
COPY data/ ./data/
COPY eval/ ./eval/
COPY README.md ./

EXPOSE 8000

# 健康检查：复用我们在 server.py 里写的 /health 端点。
# slim 精简镜像里没有 curl，所以用 Python 自带的 urllib 发一次请求。
# 请求失败会抛异常 → 进程退出码非 0 → Docker 判定 unhealthy。
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8000/health')"

# ⚠️ 最常见的容器新手坑：必须绑 0.0.0.0，不能绑 127.0.0.1。
# 容器内的 127.0.0.1 只有容器自己能访问，宿主机的 -p 端口映射过去
# 永远连不上，表现是「容器在跑但访问报 connection refused」。
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000"]
