"""Shared pytest fixtures."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def mock_environment():
    """Create a mock BaseEnvironment for testing."""
    env = MagicMock()
    env.exec = AsyncMock(return_value=MagicMock(stdout="", stderr="", return_code=0))
    env.upload_dir = AsyncMock()
    return env
