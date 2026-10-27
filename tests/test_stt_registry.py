import pytest
from unittest.mock import patch, MagicMock

from dictum.stt.registry import create_engine

def test_create_engine_unknown():
    with pytest.raises(KeyError, match="Unknown STT engine 'invalid_engine'"):
        create_engine("invalid_engine")

@patch("dictum.stt.registry.import_module")
def test_create_engine_import_error(mock_import):
    mock_import.side_effect = ImportError("No module named 'mlx'")

    with pytest.raises(ImportError, match=r"Engine 'whisper' needs an optional dependency.*underlying error: No module named 'mlx'"):
        create_engine("whisper")

@patch("dictum.stt.registry.import_module")
def test_create_engine_success(mock_import):
    mock_module = MagicMock()
    mock_class = MagicMock()

    setattr(mock_module, "WhisperMLX", mock_class)
    mock_import.return_value = mock_module

    result = create_engine("whisper", some_arg="value")

    mock_import.assert_called_once_with("dictum.stt.whisper_mlx")
    mock_class.assert_called_once_with(some_arg="value")
    assert result == mock_class.return_value
