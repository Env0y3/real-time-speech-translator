import os
from pathlib import Path

from elevenlabs.client import ElevenLabs
from elevenlabs.play import play

client = ElevenLabs(
    api_key=os.environ["ELEVENLABS_API_KEY"]
)

audio = client.text_to_speech.convert(
    text="Hello, this is a basic ElevenLabs API test.",
    voice_id="JBFqnCBsd6RMkjVDRZzb",
    model_id="eleven_flash_v2_5",
    output_format="mp3_44100_128",
)

output_path = Path(__file__).resolve().parent / "elevenlabs_test.mp3"
with output_path.open("wb") as f:
    for chunk in audio:
        f.write(chunk)

print(f"生成成功：{output_path}")
