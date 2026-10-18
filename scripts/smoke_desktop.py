"""Open the real Qt window, wait for real model init, capture, close. No translation starts."""
import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication
from gui.main_window import MainWindow

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--screenshot', default='desktop-smoke.png')
args = parser.parse_args()
app = QApplication([])
window = MainWindow()
window.show()
started = time.monotonic()
print(json.dumps({'initial_start_enabled': window.start_button.isEnabled()}), flush=True)
code = [1]

def check():
    if window.initialization_thread is not None:
        return
    details = {
        'visible': window.isVisible(), 'status': window._status,
        'start_enabled': window.start_button.isEnabled(),
        'no_audio_devices': 'no_audio_devices' in window._notice_keys,
        'model_ready': window._model_ready,
        'error_key': window._error_key,
        'elapsed_seconds': round(time.monotonic() - started, 2),
    }
    print(json.dumps(details), flush=True)
    window.grab().save(args.screenshot)
    code[0] = 0 if window._model_ready else 1
    timer.stop()
    window.close()
    app.quit()

timer = QTimer()
timer.timeout.connect(check)
timer.start(100)
app.exec()
raise SystemExit(code[0])
