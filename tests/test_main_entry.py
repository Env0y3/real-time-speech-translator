import unittest
from unittest.mock import AsyncMock, patch

import main


class MainEntryTests(unittest.IsolatedAsyncioTestCase):
    async def test_normal_mode_still_uses_shared_cli_pipeline(self) -> None:
        run_pipeline = AsyncMock()
        with (
            patch("main.choose_run_mode", return_value="1"),
            patch("main.run_normal_pipeline", run_pipeline),
        ):
            await main.main()
        run_pipeline.assert_awaited_once_with(interactive_stop=True)

    async def test_benchmark_mode_still_uses_existing_entry(self) -> None:
        run_benchmark = AsyncMock()
        with (
            patch("main.choose_run_mode", return_value="2"),
            patch("main.run_benchmark_mode", run_benchmark),
        ):
            await main.main()
        run_benchmark.assert_awaited_once_with()


if __name__ == "__main__":
    unittest.main()
