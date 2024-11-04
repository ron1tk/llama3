import os
import pytest
from unittest.mock import patch, mock_open
from pytest import raises
from your_module import TestGenerator

# Fixture for environment setup
@pytest.fixture
def test_env_setup():
    os.environ['OPENAI_API_KEY'] = 'test_api_key'
    os.environ['OPENAI_MODEL'] = 'test_model'
    os.environ['OPENAI_MAX_TOKENS'] = '1000'
    yield
    os.environ.pop('OPENAI_API_KEY')
    os.environ.pop('OPENAI_MODEL')
    os.environ.pop('OPENAI_MAX_TOKENS')

# Fixture for TestGenerator instance
@pytest.fixture
def generator_instance(test_env_setup):
    return TestGenerator()

def test_init_success(generator_instance):
    """Test successful initialization with environment variables."""
    assert generator_instance.api_key == 'test_api_key'
    assert generator_instance.model == 'test_model'
    assert generator_instance.max_tokens == 1000

def test_init_failure_no_api_key():
    """Test initialization fails when OPENAI_API_KEY is not set."""
    os.environ.pop('OPENAI_API_KEY', None)  # Ensure API key env var is not set
    with raises(ValueError):
        TestGenerator()

@patch('sys.argv', ['script_name', 'file1.py file2.js'])
def test_get_changed_files_success(generator_instance):
    """Test successful retrieval of changed files."""
    assert generator_instance.get_changed_files() == ['file1.py', 'file2.js']

@patch('sys.argv', ['script_name'])
def test_get_changed_files_empty(generator_instance):
    """Test no changed files returns empty list."""
    assert generator_instance.get_changed_files() == []

@pytest.mark.parametrize("file_name,expected_language", [
    ('test.py', 'Python'),
    ('test.js', 'JavaScript'),
    ('unknown.ext', 'Unknown')
])
def test_detect_language(generator_instance, file_name, expected_language):
    """Test language detection based on file extension."""
    assert generator_instance.detect_language(file_name) == expected_language

@pytest.mark.parametrize("language,expected_framework", [
    ('Python', 'pytest'),
    ('JavaScript', 'jest'),
    ('Unknown', 'unknown')
])
def test_get_test_framework(generator_instance, language, expected_framework):
    """Test retrieving correct test framework based on language."""
    assert generator_instance.get_test_framework(language) == expected_framework

@patch('builtins.open', new_callable=mock_open, read_data='def test_function(): pass')
def test_create_prompt_success(mock_file, generator_instance):
    """Test successful prompt creation."""
    prompt = generator_instance.create_prompt('test.py', 'Python')
    assert 'Generate comprehensive unit tests for the following Python code using pytest.' in prompt

@patch('builtins.open', side_effect=Exception('File not found'))
def test_create_prompt_failure(mock_file, generator_instance):
    """Test prompt creation failure due to file read error."""
    prompt = generator_instance.create_prompt('nonexistent.py', 'Python')
    assert prompt is None

@patch('requests.post')
def test_call_openai_api_success(mock_post, generator_instance):
    """Test successful API call to generate test cases."""
    mock_response = mock_post.return_value
    mock_response.raise_for_status = lambda: None
    mock_response.json.return_value = {
        'choices': [
            {'message': {'content': 'test case content'}}
        ]
    }
    result = generator_instance.call_openai_api("prompt")
    assert result == 'test case content'

@patch('requests.post', side_effect=Exception('API request failed'))
def test_call_openai_api_failure(mock_post, generator_instance):
    """Test handling API request failure."""
    result = generator_instance.call_openai_api("prompt")
    assert result is None

@patch('pathlib.Path.mkdir')
@patch('builtins.open', new_callable=mock_open)
def test_save_test_cases_success(mock_file_open, mock_mkdir, generator_instance):
    """Test successful saving of generated test cases."""
    generator_instance.save_test_cases('test.py', 'test case content', 'Python')
    mock_file_open.assert_called()
    mock_mkdir.assert_called()

def test_run_no_changed_files(generator_instance):
    """Test run method with no changed files."""
    with patch('sys.argv', ['script_name']), \
         patch.object(generator_instance, 'get_changed_files', return_value=[]), \
         patch.object(generator_instance, 'call_openai_api') as mock_api_call:
        generator_instance.run()
        mock_api_call.assert_not_called()

def test_run_with_changed_files(generator_instance):
    """Test run method processes changed files successfully."""
    with patch('sys.argv', ['script_name', 'file1.py']), \
         patch.object(generator_instance, 'get_changed_files', return_value=['file1.py']), \
         patch.object(generator_instance, 'detect_language', return_value='Python'), \
         patch.object(generator_instance, 'create_prompt', return_value='prompt'), \
         patch.object(generator_instance, 'call_openai_api', return_value='test cases'), \
         patch.object(generator_instance, 'save_test_cases') as mock_save:
        generator_instance.run()
        mock_save.assert_called_with('file1.py', 'test cases', 'Python')
```
This test suite covers the initialization of the `TestGenerator` class, including the handling of environment variables, the detection of changed files and programming languages, the creation of prompts, the calling of the OpenAI API, the saving of generated test cases, and the main execution flow. Mocking is utilized to simulate file operations, API responses, and the presence of command-line arguments, ensuring that external dependencies do not affect the tests' outcomes.