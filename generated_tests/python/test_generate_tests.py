# test_testgenerator.py

import pytest
from testgenerator import TestGenerator
from unittest.mock import patch, mock_open
from requests.exceptions import RequestException

@pytest.fixture
def setup_env_vars(monkeypatch):
    """Fixture to set up environment variables for tests."""
    monkeypatch.setenv("OPENAI_API_KEY", "test_api_key")
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4-turbo-preview")
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "100")

@pytest.fixture
def generator_fixture():
    """Fixture to create a TestGenerator instance."""
    return TestGenerator()

def test_init_valid_max_tokens(setup_env_vars):
    """Test initialization with valid max tokens environment variable."""
    generator = TestGenerator()
    assert generator.max_tokens == 100

def test_init_invalid_max_tokens(monkeypatch):
    """Test initialization with invalid max tokens defaults to 2000."""
    monkeypatch.setenv("OPENAI_MAX_TOKENS", "not_a_number")
    with pytest.raises(ValueError):
        TestGenerator()

def test_detect_language_known_extension(generator_fixture):
    """Test language detection for known file extensions."""
    assert generator_fixture.detect_language("test.py") == "Python"

def test_detect_language_unknown_extension(generator_fixture):
    """Test language detection defaults to 'Unknown' for unknown extensions."""
    assert generator_fixture.detect_language("test.unknown") == "Unknown"

@pytest.mark.parametrize("file_name,expected", [
    ("test.py", "pytest"),
    ("test.js", "jest"),
    ("test.unknown", "unknown"),
])
def test_get_test_framework(file_name, expected, generator_fixture):
    """Test getting the correct test framework based on file extension."""
    language = generator_fixture.detect_language(file_name)
    assert generator_fixture.get_test_framework(language) == expected

@patch("builtins.open", new_callable=mock_open, read_data="def test_function(): pass")
def test_create_prompt_reads_file(mock_file, generator_fixture):
    """Test that create_prompt reads from the file correctly."""
    prompt = generator_fixture.create_prompt("test.py", "Python")
    assert "def test_function(): pass" in prompt
    mock_file.assert_called_with("test.py", "r", encoding="utf-8")

@patch("requests.post")
def test_call_openai_api_success(mock_post, setup_env_vars, generator_fixture):
    """Test successful API call."""
    mock_response = mock_post.return_value
    mock_response.raise_for_status.side_effect = None
    mock_response.json.return_value = {"choices": [{"message": {"content": "test content"}}]}
    result = generator_fixture.call_openai_api("prompt")
    assert result == "test content"

@patch("requests.post")
def test_call_openai_api_failure(mock_post, setup_env_vars, generator_fixture):
    """Test API call failure handled gracefully."""
    mock_post.side_effect = RequestException("API failure")
    result = generator_fixture.call_openai_api("prompt")
    assert result is None

@patch("builtins.open", mock_open(read_data="language: Python\ntest_framework: pytest"))
def test_load_prompt_config(generator_fixture):
    """Test loading prompt configuration from a file."""
    config = generator_fixture.load_prompt_config("config.yaml")
    assert config == {"language": "Python", "test_framework": "pytest"}

@patch("builtins.open", new_callable=mock_open)
def test_save_test_cases(mock_file, generator_fixture):
    """Test saving generated test cases to a file."""
    generator_fixture.save_test_cases("test content", "test.py")
    mock_file.assert_called_with("test_test.py", "w", encoding="utf-8")
    mock_file().write.assert_called_once_with("test content")

# This test simulates the full run method including reading a file, generating a prompt,
# calling the API, and saving the result. It uses mocks to simulate each step.
@patch("testgenerator.TestGenerator.save_test_cases")
@patch("testgenerator.TestGenerator.call_openai_api", return_value="test content")
@patch("testgenerator.TestGenerator.create_prompt", return_value="prompt")
@patch("builtins.open", new_callable=mock_open, read_data="def test_function(): pass")
def test_full_run(mock_file, mock_create_prompt, mock_call_api, mock_save_cases, generator_fixture):
    """Test the full run process from reading a file to saving test cases."""
    generator_fixture.run("test.py")
    mock_file.assert_called_with("test.py", "r", encoding="utf-8")
    mock_create_prompt.assert_called_once()
    mock_call_api.assert_called_with("prompt")
    mock_save_cases.assert_called_with("test content", "test.py")