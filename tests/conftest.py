from collections.abc import Iterator
from pathlib import Path

import moderngl_window as mglw
import pytest
from moderngl import Context
from moderngl_window import resources
from moderngl_window.conf import settings
from moderngl_window.context.base import WindowConfig
from moderngl_window.context.base.window import BaseWindow


@pytest.fixture(scope="session")
def window() -> Iterator[BaseWindow]:
    settings.WINDOW["class"] = "moderngl_window.context.headless.Window"
    settings.WINDOW["size"] = (16, 16)
    settings.WINDOW["aspect_ratio"] = 1.0
    settings.WINDOW["gl_version"] = (4, 1)

    window = mglw.create_window_from_settings()
    resources.register_dir(Path("./resources").resolve())
    mglw.activate_context(window=window)
    yield window
    window.close()


@pytest.fixture(scope="session")
def ctx(window: BaseWindow) -> Context:  # noqa: ARG001
    return mglw.ctx()


@pytest.fixture(scope="session")
def window_config(window: BaseWindow, ctx: Context) -> WindowConfig:
    window.config = WindowConfig(ctx=ctx, wnd=window, timer=None)
    assert window.config
    return window.config
