# stripe-reconstructor

RAID-6 风格的双校验条带恢复服务。连续观测数据被写成 `N` 个数据分片加
`P`、`Q` 两个校验分片的存储条带；本服务在磁盘离线后恢复至多两个缺片，
并校验幸存分片未被静默改写。

纯 Python 标准库实现，无第三方运行时依赖。

## 校验数学

- 域：GF(2⁸)，本原多项式 `0x11d`，生成元 `2`（`app/gf256.py`）。
- `P[j] = D_0[j] ⊕ D_1[j] ⊕ … ⊕ D_{N-1}[j]`（同位置字节异或）。
- `Q[j] = Σ_i 2^i · D_i[j]`（GF(2⁸) 上求和，`i` 为从 0 开始的数据片索引，
  系数 `2^i` 为生成元幂）。
- 任意至多两个缺片均可恢复：单数据片缺失用 P（或 Q）方程；双数据片缺失
  联立 P、Q 两方程求解；校验片缺失则由数据片重算（`app/reed_solomon.py`）。

## API

### `POST /api/stripes/reconstruct`

请求体：

```json
{
  "dataShards": 4,
  "shardSize": 1024,
  "shards":  ["<base64>", null, "<base64>", "<base64>", "<base64>", null],
  "digests": ["<sha256-hex>", "... 共 dataShards+2 条 ..."]
}
```

- `dataShards`：2–16；`shardSize`：1–4096 字节。
- `shards` 长度必须为 `dataShards + 2`，按索引排列，末两片依次为 P、Q；
  缺片用 `null` 表示，现存片为标准 Base64，解码后长度须等于 `shardSize`。
- `digests` 为每片（含缺片）预期的 SHA-256 十六进制串。

成功 `200`：

```json
{
  "dataShards": 4,
  "shardSize": 1024,
  "shardCount": 6,
  "shards": ["<base64>", "… 全部 6 片，按原索引，规范 Base64 …"],
  "recoveredIndices": [1, 5],
  "digests": ["<每片实际 SHA-256>"]
}
```

失败响应（绝不输出貌似完整的条带，仅含错误信息）：

```json
{ "error": { "code": "SHARD_DIGEST_MISMATCH", "message": "…", "shardIndex": 3 } }
```

| 状态 | 错误码 | 含义 |
| --- | --- | --- |
| 422 | `INVALID_BODY` / `INVALID_JSON` / `INVALID_FIELD` | 请求体结构或取值非法 |
| 422 | `INVALID_SHARD_COUNT` / `INVALID_DIGEST_COUNT` | 数组长度与 `dataShards+2` 不符 |
| 422 | `INVALID_SHARD_ENCODING` / `SHARD_SIZE_MISMATCH` | 分片非规范 Base64 或长度不符 |
| 422 | `INVALID_DIGEST_FORMAT` | 摘要非 64 位十六进制 |
| 422 | `TOO_MANY_MISSING_SHARDS` | 缺片超过 2 个，超出恢复能力（附 `missingIndices`） |
| 409 | `SHARD_DIGEST_MISMATCH` | 现存片与预期摘要矛盾；**不会**被擅自当作缺片（附 `shardIndex`） |
| 409 | `RECONSTRUCTED_DIGEST_MISMATCH` | 重建片与预期摘要矛盾，幸存数据与归档元数据不一致（附 `shardIndex`） |
| 409 | `PARITY_RELATION_MISMATCH` | 拼合后的条带不满足 P/Q 校验关系（附 `parity`、`pIndex`、`qIndex`） |

### `GET /health`

返回 `200 {"status": "ok"}`，用于容器健康检查。

## 运行

### Docker Compose（推荐）

```bash
# 启动服务并运行一次性 verify（单测 + 可发布包构建 + 接口冒烟），
# verify 的退出码即整体结果，随后容器自行结束：
APP_PORT=8080 docker compose up --build --exit-code-from verify verify

# 仅启动服务（宿主机端口由 APP_PORT 决定，默认 8000）：
APP_PORT=8080 docker compose up --build app
```

`verify` 服务通过 `depends_on: condition: service_healthy` 等待应用健康
检查通过后才开始执行。

### 本地（无 Docker）

```bash
python3 -m unittest discover -s tests   # 单元测试
python3 -m app                          # 启动服务，PORT 环境变量可改端口，默认 8000
python3 verify.py                       # 对 http://localhost:8000 跑完整 verify 流程
```

## 布局

```
app/gf256.py          GF(2^8) 算术（0x11d，生成元 2）
app/reed_solomon.py   P/Q 计算、条带重建与校验关系核验
app/service.py        请求校验、摘要核验、错误分类（422/409）
app/server.py         标准库 HTTP 服务（/health、/api/stripes/reconstruct）
tests/                单元测试（穷举全部单/双缺片组合）
scripts/package_build.py  可发布包构建（python -m build，离线时回退 stdlib 构建器）
verify.py             一次性验证流水线：单测 → 包构建 → 接口冒烟
Dockerfile / docker-compose.yml
```
