# 延迟定义与离线验证

所有 `*_at` 是同一进程的 `time.perf_counter()` 时间点（秒），不是日期时间，不能跨进程或跨会话相减。毫秒指标为 `(结束点 - 开始点) * 1000`。缺失、非有限或逆序时间点报告 `null` / N/A，不补零。

## 用户应看到的两种整体延迟

- `endpoint_trigger_to_first_playback_ms`：断句触发 → 首次软件播放。旧名 `speech_end_to_first_playback_ms` 保留且同值，实际起点始终是 `endpoint_triggered_at`，不含断句等待，不能解释成“用户刚停口到出声”。
- `last_voice_to_first_playback_ms`：最后一次检测到语音 → 首次软件播放，包含断句等待。这更接近整体体验，但仍是软件估计，不是实际声学停口到扬声器出声的实测。

`last_voice_at` 是 VAD 在消费音频块时记录的末次语音时间；受到分块、输入缓冲、调度与检测误差影响。`endpoint_triggered_at` 是应用判断达到断句条件的时间。

## 首次播放的测量边界

ElevenLabs PCM 路径的 `first_playback_at` 是首次成功写入输出流那次 `write()` 的开始时间。它不是写入完成时间，也不是扬声器真正发声时间。声卡、驱动、虚拟设备、系统混音和接收软件都可能继续缓冲。双输出时也不是每个输出设备分别测量的声学首帧。

新 trace 记录：
- `latency_semantics_version: 2`
- `first_playback_is_acoustic: false`
- `first_playback_measurement: output_stream_write_start`

pyttsx3 降级时采用 `pyttsx3_started_word` 回调近似；没有回调时为 `pyttsx3_invocation_start` 调用起点近似，精度更低。未知来源标为 `unknown`，不要与 PCM write 首帧混成同一种声学测量。旧日志没有该标记，不能反推硬件出声时间。

## 分阶段指标

| 字段 | 开始 → 结束 | 注意 |
| --- | --- | --- |
| `endpoint_wait_ms` | `last_voice_at` → `endpoint_triggered_at` | 含静音阈值和调度影响 |
| `asr_stage_ms` / `speech_end_to_asr_result_ms` | `endpoint_triggered_at` → `asr_result_at` | 断句后阶段，不等于纯模型推理 |
| `translation_after_asr_ms` | `asr_result_at` → `translation_first_segment_at` | 含热词处理、排队和首段准备 |
| `tts_after_first_segment_ms` | `translation_first_segment_at` → `tts_first_audio_received_at` | 含 TTS 建连/调度/网络首音频，非纯服务端耗时 |
| `tts_queue_wait_ms` / `playback_queue_ms` | 实际首个播放块的 `audio_received_at` → `first_playback_at` | 接收后至 write，包括等待前句和输出流准备 |
| `translation_ttft_ms` | 翻译请求开始 → 首 token | 不从断句起算 |
| `translation_ttfs_ms` | 翻译请求开始 → 首翻译段 | 不从断句起算 |
| `network_first_audio_ms` / `first_audio_chunk_ms` | 首段 ready → 首音频块 | 旧命名保留，包含请求前调度等 |
| `first_playback_ms` | 首段 ready → 首次软件播放 | 不含 ASR/翻译首段之前时间 |

其余 `speech_end_to_translation_*` 和 `speech_end_to_tts_first_audio_ms` 旧字段均从 endpoint 起算。SenseVoice benchmark 的旧四元组和 `speech_end_to_result_latency_ms` 则从 last voice 起算，保留其历史结构，切勿仅凭相似名称混用。

同一条有效 trace 总满足：

`last_voice_to_first_playback_ms = endpoint_wait_ms + endpoint_trigger_to_first_playback_ms`

正常首块成功路径中，断句后整体近似等于 ASR + 翻译首段 + TTS 首音频 + 播放队列。丢块、播放失败、降级会破坏逐段相加条件。不能把累计的 endpoint→首段、endpoint→首音频再相加，否则重复计时。软件指标不包含已知或未知的硬件输出延迟。

## 日志与统计兼容

JSONL 仍为追加写入，保留旧字段，不重写历史日志。纯函数 `calculate_trace_latencies(trace)` 定义时间区间；`summarize_trace_records(records, session_id=...)` 强制指定运行会话，仅统计该会话 `trace_summary` 行。支持读取旧 endpoint 整体字段别名。每项单独给有效样本数、均值、中位数、nearest-rank P95；缺失项不参与计算。空会话返回 0 条和空统计，不冒充成功运行。

统计时还应说明机器、模型、输入输出设备、断句参数、成功/失败/降级数量，必要时分别统计。`trace_count` 是记录数，不保证所有请求成功，也不自动消除重复记录。没有 session_id 的历史数据需明确作为历史样本单独处理，不能归入当前运行。

历史“56 条”“2.11 秒”等结果不是本轮性能证据。本轮离线测试只验证定义、字段兼容与统计隔离，不能证明真实音频链路延迟改善。新性能结论需要同一设备、同一配置、代表性语料下新采集的会话；实际声学延迟需回环录音或外部测量。

## 离线测试

运行：`python -m unittest discover -s tests -p 'test_latency_semantics.py' -v`

仅使用合成时间点和临时 JSONL，不调用模型、外部服务、真实麦克风或扬声器。
