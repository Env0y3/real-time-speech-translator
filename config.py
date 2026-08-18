from core.paths import get_resource_path, get_resource_root, get_user_data_path


# V2 的 Microphone（麦克风）基础配置。
PREFERRED_SAMPLE_RATE = 16_000  # Sample Rate（采样率）：每秒 16000 个样本
CHANNELS = 1  # Channel（声道）：1 表示单声道
CHUNK_DURATION_SECONDS = 0.2  # Chunk（音频块）：每块大约 200 ms
VAD_RMS_THRESHOLD = 10.0  # RMS 高于此值时认为当前有人说话
ENDPOINT_BASE_SECONDS = 0.5  # Base Threshold（基础句尾静音阈值）
ENDPOINT_MIN_SECONDS = 0.4  # 动态阈值下限
ENDPOINT_MAX_SECONDS = 0.8  # 动态阈值上限
ENDPOINT_ADAPTIVE_ENABLED = True
ENDPOINT_SAFETY_MARGIN_SECONDS = 0.20
ENDPOINT_SMOOTHING_ALPHA = 0.3  # EMA（指数移动平均）的平滑系数
ENDPOINT_MIN_SPEECH_SECONDS = 0.6  # 短语音保护的有效发声时长界线
ENDPOINT_SHORT_UTTERANCE_EXTRA_WAIT_SECONDS = 0.2  # 疑似碎片额外确认时间
ENDPOINT_MIN_INTRA_PAUSE_SECONDS = 0.1  # 过滤过短的VAD停顿抖动
# 保留旧常量名，Vosk 和其他固定模式继续读取统一的 0.5 秒基线。
ENDPOINT_SILENCE_SECONDS = ENDPOINT_BASE_SECONDS
ASR_PROVIDER = "sensevoice"  # Benchmark 可选："vosk" 或 "sensevoice"
NORMAL_ASR_PROVIDER = "sensevoice"  # Normal Mode 可选："vosk" 或 "sensevoice"
NORMAL_HOTWORD_CORRECTION_ENABLED = True
INPUT_VALIDITY_GUARD_ENABLED = True
FALSE_TRIGGER_FILTER_ENABLED = True
# 仅过滤不超过 350 ms 的 filler；优先避免误删自然长度的真实短回答。
FALSE_TRIGGER_MAX_SPEECH_SECONDS = 0.35
FALSE_TRIGGER_FILLERS = {
    "嗯",
    "啊",
    "呃",
    "额",
    "嗯嗯",
}
VOSK_MODEL_NAME = "vosk-model-small-cn-0.22"
SENSEVOICE_MODEL_NAME = "iic/SenseVoiceSmall"
BENCHMARK_REPEATS = 3
TEST_SENTENCES = [
    "你好",
    "今天天气怎么样",
    "你叫什么名字",
    "我今天想去深圳",
    "我今天下午准备去深圳找朋友吃饭",
    "明天早上八点提醒我去上课",
    "广州",
    "深圳",
    "东莞",
    "贵州",
    "杭州",
    "我最近正在学习 LangGraph",
    "我用 DeepSeek 做翻译",
    "今天想休息",
    "今天想去西安",
]
HOTWORD_TEST_SENTENCES = [
    "我用 DeepSeek 做翻译",
    "我最近正在学习 LangGraph",
    "我最近在学 Python",
    "我用 Redis 做缓存",
    "这个项目使用 WebSocket",
    "我准备使用 Docker 部署项目",
    "我使用 FastAPI 搭建后端",
    "我通过 GitHub 管理代码",
    "我正在使用 OpenAI 的模型",
    "我正在使用 ChatGPT",
]
HOTWORD_CORRECTION_THRESHOLD = 0.80  # 最佳候选至少达到此相似度才纠错
HOTWORD_MARGIN_THRESHOLD = 0.10  # 最佳候选必须明显领先第二候选
HOTWORD_MAX_PHRASE_TOKENS = 3  # 最多合并三个相邻英文 token（词元）
PROJECT_ROOT = get_resource_root()
MODEL_PATH = (
    get_resource_path("models", VOSK_MODEL_NAME)
)
BENCHMARK_RESULTS_PATH = (
    get_user_data_path("logs", "asr_benchmark_results.jsonl")
)
RUNTIME_LATENCY_LOG_PATH = (
    get_user_data_path("logs", "runtime_latency_log.jsonl")
)
HOTWORDS_PATH = get_resource_path("data", "hotwords.json")
NORMALIZATION_PUNCTUATION = set(
    "，。！？；：、“”‘’,.!?;:\"'"
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"
STREAMING_TRANSLATION_ENABLED = True
STREAMING_TRANSLATION_MIN_CHARS = 20
STREAMING_TRANSLATION_TARGET_CHARS = 35
STREAMING_TRANSLATION_MAX_CHARS = 60
TTS_PROVIDER = "elevenlabs"  # 可选："pyttsx3" 或 "elevenlabs"
ELEVENLABS_VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"
ELEVENLABS_MODEL_ID = "eleven_flash_v2_5"
ELEVENLABS_OUTPUT_FORMAT = "pcm_24000"
ELEVENLABS_SAMPLE_RATE = 24_000
ELEVENLABS_CHANNELS = 1
ELEVENLABS_DTYPE = "int16"
ELEVENLABS_AUDIO_QUEUE_MAXSIZE = 50
# None 使用系统默认输出；也可填写 sounddevice 设备索引或唯一名称。
TRANSLATION_OUTPUT_DEVICE = 4
LOCAL_MONITOR_ENABLED = False
LOCAL_MONITOR_DEVICE = None

# System Prompt（系统提示词）只定义单一的中译英职责。

TRANSLATION_SYSTEM_PROMPT = (
    "你是一个实时中英翻译器。"
    "请把用户提供的中文准确、自然地翻译成英文。"
    "只返回英文翻译，不要解释，不要添加额外内容，不要道歉，"
    "不要询问用户提供内容，也不要以聊天助手身份回答。"
    "如果输入没有有效可翻译内容，返回空字符串。"
)
