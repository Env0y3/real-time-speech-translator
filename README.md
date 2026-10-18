# 实时中文 → 英文语音翻译桌面程序

本轮维护单向：中文麦克风 → 本地识别 → DeepSeek 英译 → ElevenLabs 英文 PCM 播放。未实现英译中、自动语言识别、网页端或公网服务。

## 受支持的运行方式

推荐 **CPython 3.12、64 位**。本轮实际安装/测试平台：Debian 13 x86_64、CPython 3.12.14；Windows 10/11 64 位为目标桌面平台，尚未进行本轮真机验收。不要把 Linux offscreen 测试理解成 Windows/麦克风验收。其他 Python 版本未验证。

从仓库根目录执行，保留目录结构（含 data/hotwords.json）。

```sh
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS:
. .venv/bin/activate
python -m pip install -r requirements-gui.txt
python -m pip check
```

requirements.txt 是核心运行及离线测试依赖；requirements-gui.txt 包含核心和 PySide6。requirements-models.txt / docs/MODEL_SETUP.md 说明默认 SenseVoice 模型依赖，必须另行准备后才可真实识别。Vosk 包因现有入口导入需要保留，即使选择 SenseVoice。

Linux 还需系统 PortAudio（Debian/Ubuntu：libportaudio2）、Qt 所需图形库；pyttsx3 本地语音回退需要系统语音引擎（Linux espeak-ng；Windows 系统 SAPI5）。安装系统包需由有权限的用户完成。Windows sounddevice wheel 自带 PortAudio，仍需可用声卡驱动。无桌面的离线 GUI 测试可设 QT_QPA_PLATFORM=offscreen，它不创造真实音频设备。

## 模型和配置

默认 config.py 的 NORMAL_ASR_PROVIDER 为 sensevoice，首次加载可能下载 iic/SenseVoiceSmall，需允许访问模型官方分发站点及足够磁盘。先按 docs/MODEL_SETUP.md 准备。Vosk 选项需要把官方中文模型解压至 models/vosk-model-small-cn-0.22，并将 NORMAL_ASR_PROVIDER 改为 vosk；目录存在不等于模型加载验收。

只通过环境变量设置自己的密钥；程序**不会自动读取 .env 文件**：

```powershell
$env:DEEPSEEK_API_KEY = "在本机填写"
$env:ELEVENLABS_API_KEY = "在本机填写"
python gui_main.py
```

Linux 可用交互式 read -s 设置变量，不要把真实值写进脚本、截图、Git 或诊断附件。config.py 控制 DEEPSEEK_BASE_URL、DEEPSEEK_MODEL、TTS_PROVIDER、ELEVENLABS_VOICE_ID、ELEVENLABS_MODEL_ID。当前模型/声音必须由你的服务账号实际支持；本轮未发起收费请求验证其可用性。不要直接运行 scripts/elevenlabs_test.py、fish_tts_demo.py 等旧服务示例来做离线冒烟，它们可能直接调用服务。

启动：python gui_main.py。终端入口：python main.py（交互选择正常翻译/ASR benchmark）。正常桌面模式在模型初始化完成前禁用开始；失败时显示错误，修正安装/模型条件后重启。停止后可再次开始；关闭运行中的窗口会请求后台停止。模型下载/加载期间关闭需等待初始化线程结束，不保证立即退出。

## 音频路由

首次默认系统设备，不再固定索引 4。GUI 可选择麦克风、翻译输出和可选本地监听。已保存的名称优先于易变的数字索引；同名歧义或设备消失会保留“不可用”状态，要求重新选择，绝不静默改投另一输出。刷新失败不会覆盖保存的选择。

翻译开始/停止控制主输出；本地监听单独开关。监听关闭时，不可用的监听设备不会阻断主翻译；开启则必须重新选择可用设备。监听与主输出相同时不应重复播放。设备运行中拔出时按错误提示停止、刷新并重选后重启。

仅需直接听译音时选择普通扬声器/耳机。要让会议软件收到英译音，需要自行安装并配置受信任的虚拟声卡，翻译器选择其播放端（如 CABLE Input），会议软件选择对应录音端（如 CABLE Output）。本地监听选择耳机，避免反馈。虚拟声卡不是本程序附带功能；没有它不影响普通输出。

设置和新日志在用户目录 RealTimeSpeechTranslator 下：Windows %APPDATA%，Linux $XDG_CONFIG_HOME 或 ~/.config。REALTIME_TRANSLATOR_USER_DATA_DIR 可指定隔离测试目录。历史仓库 logs 文件保持不变。

## 测试（不调用收费服务）

```sh
# Windows无需offscreen；Linux CI/无桌面：
export QT_QPA_PLATFORM=offscreen
python -m unittest discover -s tests -v
# 可选：输出分类统计JSON
python scripts/run_tests.py test-results.json
```

缺 PySide6 时只跳过界面测试，其他导入错误必须暴露。完整桌面验收先安装 requirements-gui.txt；模型测试中有替身，不代表实际模型加载或语音质量。详见 docs/DESKTOP_ACCEPTANCE.md、docs/LATENCY.md 和本轮 evidence 报告。

## 延迟

界面“断句触发→首次软件播放”从程序确认句尾开始；“末次语音→首次软件播放”还包含断句等待。首次写入是软件输出调用近似值，不是扬声器实测出声。历史字段保留兼容；不将历史混合配置的 2.11 秒中位数称为本轮性能。详见 docs/LATENCY.md。

### 安全限制

设备名称重复时不能可靠识别重插后的设备，本轮保守地要求重选唯一名称/使用系统默认，不信任历史索引。pyttsx3 直接模式只支持系统默认输出且关闭本地监听；选择显式输出或开启监听会明确拒绝启动，避免配置被静默忽略。ElevenLabs 路由支持显式设备。云合成回退到 pyttsx3 时测量点可能是词回调或调用起点，因此界面统称“首次软件播放”，不能统一称为 PCM 写入或声学出声。

模型准备后可运行 python scripts/smoke_desktop.py --screenshot desktop-smoke.png 检查真实Qt窗口和模型初始化，再自动关闭；它不会开始翻译或调用收费服务。无麦克风时会清楚提示，但模型初始化可独立成功。
