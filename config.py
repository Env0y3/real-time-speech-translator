from pathlib import Path


# V2 的 Microphone（麦克风）基础配置。
PREFERRED_SAMPLE_RATE = 16_000  # Sample Rate（采样率）：每秒 16000 个样本
CHANNELS = 1  # Channel（声道）：1 表示单声道
CHUNK_DURATION_SECONDS = 0.2  # Chunk（音频块）：每块大约 200 ms
VAD_RMS_THRESHOLD = 10.0  # RMS 高于此值时认为当前有人说话
ENDPOINT_SILENCE_SECONDS = 0.6  # 连续静音达到此时长时认为一句话结束
ASR_PROVIDER = "sensevoice"  # Benchmark 可选："vosk" 或 "sensevoice"
NORMAL_ASR_PROVIDER = "sensevoice"  # Normal Mode 可选："vosk" 或 "sensevoice"
NORMAL_HOTWORD_CORRECTION_ENABLED = True
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
PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / VOSK_MODEL_NAME
)
BENCHMARK_RESULTS_PATH = (
    PROJECT_ROOT / "asr_benchmark_results.jsonl"
)
HOTWORDS_PATH = PROJECT_ROOT / "hotwords.json"
NORMALIZATION_PUNCTUATION = set(
    "，。！？；：、“”‘’,.!?;:\"'"
)
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-v4-flash"

# System Prompt（系统提示词）只定义单一的中译英职责。

TRANSLATION_SYSTEM_PROMPT = (
    "你是一个实时中英翻译器。"
    "请把用户提供的中文准确、自然地翻译成英文。"
    "只输出英文翻译，不要解释，不要添加额外内容。"
)
