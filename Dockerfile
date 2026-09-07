FROM python:3.11-slim

# pyqlib 策略（WITH_QLIB build-arg）：
#   auto（默认）：amd64 装官方 manylinux wheel（生产服务器架构，秒装）；
#                 aarch64 跳过——PyPI 无 linux/aarch64 wheel 也无 sdist，
#                 factor_backtest/gen_data/update_data 返回明确错误，其余工具不受影响
#   1           ：aarch64 从 GitHub 源码编译（慢，一次性；需构建期可访问 GitHub）
#   0           ：任何架构都不装
# lightgbm 的 linux wheel 自带 OpenMP 运行时，直接可用。
ARG WITH_QLIB=auto
# WITH_QLIB=1（arm64 源码编译）时的 qlib 仓库地址，拉不动 GitHub 可换镜像
ARG QLIB_GIT_URL=https://github.com/microsoft/qlib.git
# 国内构建加速：--build-arg PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
ARG PIP_INDEX_URL
# 国内 apt 加速（仅 WITH_QLIB=1 需要装编译链）：
# --build-arg APT_MIRROR=mirrors.tuna.tsinghua.edu.cn
ARG APT_MIRROR

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt ./
RUN pip install -r requirements.txt \
    && arch="$(dpkg --print-architecture)" \
    && if [ "$WITH_QLIB" != "0" ]; then \
         if [ "$arch" = "amd64" ]; then \
           pip install "pyqlib>=0.9.6" jinja2 mlflow; \
         elif [ "$WITH_QLIB" = "1" ]; then \
           if [ -n "$APT_MIRROR" ]; then \
             sed -i "s|deb.debian.org|$APT_MIRROR|g" /etc/apt/sources.list.d/debian.sources; \
           fi \
           && apt-get update \
           && apt-get install -y --no-install-recommends build-essential git \
           && pip install "cython<3" "numpy>=1.26,<2" \
           && pip install "git+${QLIB_GIT_URL}@v0.9.7" jinja2 mlflow \
           && apt-get purge -y build-essential git \
           && apt-get autoremove -y \
           && rm -rf /var/lib/apt/lists/*; \
         else \
           echo "WITH_QLIB=auto: aarch64 无 pyqlib wheel，跳过（回测类工具容器内不可用）"; \
         fi; \
       fi

COPY . .

RUN useradd --uid 10001 --no-create-home --home-dir /app appuser \
    && mkdir -p /app/data /app/usage \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 50053

CMD ["python3", "server.py", "--host", "0.0.0.0", "--port", "50053"]
