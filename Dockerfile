FROM node:20-bookworm-slim AS webui-build

WORKDIR /app/webui

COPY webui/package.json ./
RUN npm install

COPY webui/ ./
RUN npm run build


FROM python:3.11-slim AS app

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=10000

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-webui.txt requirements-nn.txt requirements-spendee.txt ./
RUN pip install --no-cache-dir \
    -r requirements-webui.txt \
    -r requirements-nn.txt \
    -r requirements-spendee.txt \
    pybind11

COPY . .
COPY --from=webui-build /app/webui/dist ./webui/dist

RUN cmake -S /app -B /app/build \
    -DCMAKE_BUILD_TYPE=Release \
    -DSPLENDOR_BUILD_TEST_BINS=OFF
RUN cmake --build /app/build --target splendor_native -j

EXPOSE 10000

CMD ["sh", "-c", "uvicorn nn.webapp:app --host 0.0.0.0 --port ${PORT}"]
