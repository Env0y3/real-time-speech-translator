# Current audio-device choice regression

## Scope and result

An explicit current GUI selection among duplicate device names now survives an
unchanged pre-start enumeration. Persisted duplicate names and old indexes still
cannot establish identity and remain unavailable until the user selects again.

The live selection is accepted only while the complete `AudioDeviceInfo` snapshot
is unchanged. A failed enumeration, changed index/name/channel/rate, or other
snapshot change invalidates a duplicate choice. The unavailable placeholder stays
blocked across further refreshes, including when only one same-name peer remains.
Choosing a currently listed device or system default clears that placeholder.
The snapshot and reselect marker are not persisted as device identity.

## Offline reproduction and verification

- Added `tests/test_gui_device_selection.py` before changing the implementation.
- Before patch: 10 tests; 4 failures, 2 errors, 4 passes.
  See `patch-device-before.txt`.
- After patch: the same 10 tests pass.
  See `patch-device-after.txt`.
- Existing and new GUI tests: 37 pass; audio-device unit tests: 15 pass.
  See `patch-device-regressions.txt`.
- `git diff --check` passes.

Commands use `/tmp/translator-core-venv/bin/python`,
`QT_QPA_PLATFORM=offscreen`, and the prepared Qt library directory in
`LD_LIBRARY_PATH`. Enumeration, validation, worker construction, model loading,
and service calls are mocked where needed. No model downloads or paid API calls
are part of these checks.

## Limits

This is Linux offscreen/mock verification, not Windows or physical audio-device
acceptance. PortAudio metadata does not expose a persistent hardware identifier;
an unplug/replug or replacement that produces an identical snapshot cannot be
detected by this comparison. Persisted indexes are never used as that proof.
