FROM python:3.11-slim

# 注意：pyqlib 不进镜像（PyPI 无 linux/aarch64 wheel，arm64 需源码编译），
# 回测类工具（factor_backtest/gen_data/update_data）在容器内不可用；
# lightgbm 的 linux wheel 自带 OpenMP 运行时，直接可用。

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

RUN useradd --uid 10001 --no-create-home --home-dir /app appuser \
    && mkdir -p /app/data /app/usage \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 50053

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "50053"]
