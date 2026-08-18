TRANSLATIONS = {
    "zh": {
        "window_title": "实时语音翻译器",
        "language": "界面语言：",
        "audio_devices": "音频设备",
        "input_microphone": "输入麦克风：",
        "translation_output": "翻译输出：",
        "enable_local_monitor": "启用本地监听",
        "monitor_device": "监听设备：",
        "refresh_devices": "刷新设备",
        "pipeline": "处理流程",
        "asr": "语音识别：",
        "translation": "翻译：",
        "tts": "语音合成：",
        "chinese": "中文",
        "english": "英文",
        "latency": "延迟",
        "status": "状态",
        "start_translation": "开始翻译",
        "stop_translation": "停止翻译",
        "system_default": "系统默认",
        "chinese_placeholder": "等待识别中文语音…",
        "english_placeholder": "等待英文翻译…",
        "speech_end_first_playback": "语音结束 → 首次播放：{value}",
        "endpoint": "端点：{value}",
        "asr_latency": "识别：{value}",
        "translation_latency": "翻译：{value}",
        "tts_latency": "合成：{value}",
        "api_ready": "✓ 已就绪",
        "api_missing": "✗ 未配置",
        "initializing": "正在初始化…",
        "checking_model": "正在检查语音识别模型…",
        "loading_model": "正在下载/加载 SenseVoiceSmall…",
        "checking_audio": "正在检查音频设备…",
        "ready_detail": "SenseVoiceSmall 已加载，可以开始翻译。",
        "virtual_device_missing": "未检测到虚拟音频设备，Discord 虚拟麦克风功能不可用。",
        "saved_input_missing": "上次选择的输入设备不可用，已切换到系统默认。",
        "saved_output_missing": "上次选择的输出设备不可用，请重新选择。",
        "saved_monitor_missing": "上次选择的监听设备不可用，已切换到系统默认。",
        "device_refresh_failed": "刷新音频设备失败：{detail}",
        "missing_deepseek_api": "未配置 DEEPSEEK_API_KEY。",
        "missing_elevenlabs_api": "未配置 ELEVENLABS_API_KEY。",
        "model_not_found": "未找到语音识别模型。",
        "model_load_failed": "语音识别模型加载失败：{detail}",
        "pipeline_error": "翻译流程错误：{detail}",
        "status_idle": "空闲",
        "status_loading": "加载中",
        "status_ready": "已就绪",
        "status_listening": "正在监听",
        "status_recognizing": "正在识别",
        "status_translating": "正在翻译",
        "status_speaking": "正在播放",
        "status_stopping": "正在停止",
        "status_error": "错误",
    },
    "en": {
        "window_title": "Real-Time Speech Translator",
        "language": "Language:",
        "audio_devices": "Audio Devices",
        "input_microphone": "Input Microphone:",
        "translation_output": "Translation Output:",
        "enable_local_monitor": "Enable Local Monitor",
        "monitor_device": "Monitor Device:",
        "refresh_devices": "Refresh Devices",
        "pipeline": "Pipeline",
        "asr": "ASR:",
        "translation": "Translation:",
        "tts": "TTS:",
        "chinese": "Chinese",
        "english": "English",
        "latency": "Latency",
        "status": "Status",
        "start_translation": "Start Translation",
        "stop_translation": "Stop Translation",
        "system_default": "System Default",
        "chinese_placeholder": "Waiting for Chinese speech…",
        "english_placeholder": "Waiting for translation…",
        "speech_end_first_playback": "Speech End → First Playback: {value}",
        "endpoint": "Endpoint: {value}",
        "asr_latency": "ASR: {value}",
        "translation_latency": "Translation: {value}",
        "tts_latency": "TTS: {value}",
        "api_ready": "✓ Ready",
        "api_missing": "✗ Missing",
        "initializing": "Initializing…",
        "checking_model": "Checking the speech recognition model…",
        "loading_model": "Downloading/loading SenseVoiceSmall…",
        "checking_audio": "Checking audio devices…",
        "ready_detail": "SenseVoiceSmall is loaded. Translation can start.",
        "virtual_device_missing": (
            "No virtual audio device detected. "
            "Discord virtual microphone is unavailable."
        ),
        "saved_input_missing": (
            "The previously selected input device is unavailable. "
            "System Default is selected."
        ),
        "saved_output_missing": (
            "The previously selected output device is unavailable. "
            "Please select another device."
        ),
        "saved_monitor_missing": (
            "The previously selected monitor device is unavailable. "
            "System Default is selected."
        ),
        "device_refresh_failed": "Audio device refresh failed: {detail}",
        "missing_deepseek_api": "DEEPSEEK_API_KEY is not configured.",
        "missing_elevenlabs_api": "ELEVENLABS_API_KEY is not configured.",
        "model_not_found": "Speech recognition model was not found.",
        "model_load_failed": "Speech recognition model failed to load: {detail}",
        "pipeline_error": "Translation pipeline error: {detail}",
        "status_idle": "Idle",
        "status_loading": "Loading",
        "status_ready": "Ready",
        "status_listening": "Listening",
        "status_recognizing": "Recognizing",
        "status_translating": "Translating",
        "status_speaking": "Speaking",
        "status_stopping": "Stopping",
        "status_error": "Error",
    },
}


def normalize_language(language: str | None) -> str:
    return language if language in TRANSLATIONS else "zh"


def tr(key: str, language: str, **values) -> str:
    selected = TRANSLATIONS[normalize_language(language)]
    template = selected.get(key, TRANSLATIONS["en"].get(key, key))
    return template.format(**values) if values else template
