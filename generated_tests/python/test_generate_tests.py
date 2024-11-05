# test_test_generator.py

import json
import pytest
from unittest.mock import mock_open, patch, MagicMock
from test_generator import TestGenerator

@pytest.fixture
def test_generator():
    with patch('os.getenv') as mock_getenv:
        mock_getenv.side_effect = lambda k, default=None: {'OPENAI_API_KEY': 'dummy_key', 'OPENAI_MODEL': 'gpt-4-turbo-preview', 'OPENAI_MAX_TOKENS': '2000'}.get(k, default)
        yield TestGenerator()

def test_init_successful(test_generator):
    assert test_generator.api_key == 'dummy_key'
    assert test_generator.model == 'gpt-4-turbo-preview'
    assert test_generator.max_tokens == 2000

def test_init_with_invalid_max_tokens():
    with patch('os.getenv', side_effect=lambda k, default=None: {'OPENAI_MAX_TOKENS': 'invalid'}.get(k, default)):
        with pytest.raises(ValueError):
            TestGenerator()

def test_load_prompt_config_missing_file(mocker):
    mock_path_exists = mocker.patch("pathlib.Path.exists", return_value=False)
    tg = TestGenerator()
    config = tg.load_prompt_config()
    assert config == {}
    mock_path_exists.assert_called_once()

def test_load_prompt_config_with_valid_config(mocker):
    mock_open = mocker.patch("builtins.open", mock_open(read_data='{"additional_instructions": "Use AAA pattern"}'))
    mocker.patch("pathlib.Path.exists", return_value=True)
    tg = TestGenerator()
    config = tg.load_prompt_config()
    assert config == {"additional_instructions": "Use AAA pattern"}
    mock_open.assert_called_once()

def test_detect_language_with_unsupported_extension(test_generator):
    language = test_generator.detect_language("example.unsupported")
    assert language == "Unknown"

def test_get_test_framework_for_python(test_generator):
    framework = test_generator.get_test_framework("Python")
    assert framework == "pytest"

def test_create_prompt_without_file(mocker, test_generator):
    mocker.patch("builtins.open", side_effect=FileNotFoundError)
    prompt = test_generator.create_prompt("missing.py", "Python")
    assert prompt is None

@pytest.mark.parametrize("file_name,expected_language", [
    ("example.py", "Python"),
    ("example.js", "JavaScript"),
    ("example.unknown", "Unknown"),
])
def test_detect_language(test_generator, file_name, expected_language):
    language = test_generator.detect_language(file_name)
    assert language == expected_language

def test_call_openai_api_successful_response(mocker, test_generator):
    mock_response = MagicMock()
    mock_response.json.return_value = {"choices": [{"text": "Test case"}]}
    mock_requests_post = mocker.patch("requests.post", return_value=mock_response)
    response = test_generator.call_openai_api("example prompt")
    assert response == "Test case"
    mock_requests_post.assert_called_once()

def test_call_openai_api_failure_response(mocker, test_generator):
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = Exception("API call failed")
    mocker.patch("requests.post", return_value=mock_response)
    with pytest.raises(Exception, match="API call failed"):
        test_generator.call_openai_api("example prompt")

def test_save_test_cases(mocker, test_generator):
    mock_open = mocker.patch("builtins.open", mock_open())
    test_cases = "Test case content"
    test_generator.save_test_cases(test_cases, "example_test.py")
    mock_open.assert_called_once_with("example_test.py", "w")
    mock_open().write.assert_called_once_with(test_cases)

def test_run_with_invalid_args(mocker, test_generator):
    mocker.patch("sys.argv", ["script_name", "missing.py"])
    mocker.patch("builtins.open", side_effect=FileNotFoundError)
    with pytest.raises(SystemExit) as e:
        test_generator.run()
    assert e.type == SystemExit
    assert e.value.code != 0

def test_run_successful(mocker, test_generator):
    mocker.patch("sys.argv", ["script_name", "example.py"])
    mocker.patch("builtins.open", mock_open(read_data='print("Hello, World!")'))
    mocker.patch.object(Path, 'exists', return_value=True)
    mock_create_prompt = mocker.patch.object(TestGenerator, "create_prompt", return_value="test prompt")
    mock_call_openai_api = mocker.patch.object(TestGenerator, "call_openai_api", return_value="test case")
    mock_save_test_cases = mocker.patch.object(TestGenerator, "save_test_cases")

    test_generator.run()

    mock_create_prompt.assert_called_once()
    mock_call_openai_api.assert_called_once_with("test prompt")
    mock_save_test_cases.assert_called_once_with("test case", "example_test.py")