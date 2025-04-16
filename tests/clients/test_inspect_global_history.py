import pytest
from dspy.utils.dummies import DummyLM
from dspy.clients.base_lm import GLOBAL_HISTORY
import dspy

@pytest.fixture(autouse=True)
def clear_history():
    GLOBAL_HISTORY.clear()
    yield

async def test_inspect_history_basic(capsys):
    # Configure a DummyLM with some predefined responses
    lm = DummyLM([{"response": "Hello"}, {"response": "How are you?"}, {"response": "That's great!"}])
    dspy.settings.configure(lm=lm)
    
    # Make some calls to generate history
    predictor = dspy.Predict("query: str -> response: str")
    with dspy.context() as settings:
        await predictor(settings, query="Hi")
    with dspy.context() as settings:
        await predictor(settings, query="What's up?")
    await predictor(dspy.settings, query="Just fixing some tests")
    
    # Test inspecting all history
    history = GLOBAL_HISTORY
    print(capsys)
    assert len(history) > 0
    assert isinstance(history, list)
    assert all(isinstance(entry, dict) for entry in history)
    assert all("messages" in entry for entry in history)

async def test_inspect_history_with_n(capsys):
    """Test that inspect_history works with n
    Random failures in this test most likely mean you are printing messages somewhere
    """
    lm = DummyLM([{"response": "One"}, {"response": "Two"}, {"response": "Three"}])
    dspy.settings.configure(lm=lm)
    
    # Generate some history
    predictor = dspy.Predict("query: str -> response: str")
    await predictor(dspy.settings, query="First")
    await predictor(dspy.settings, query="Second")
    await predictor(dspy.settings, query="Third")
    
    dspy.inspect_history(n=2)
    # Test getting last 2 entries
    out, err = capsys.readouterr()
    assert not "First" in out
    assert "Second" in out
    assert "Third" in out

def test_inspect_empty_history(capsys):
    # Configure fresh DummyLM
    lm = DummyLM([])
    dspy.settings.configure(lm=lm)
    
    # Test inspecting empty history
    dspy.inspect_history()
    history = GLOBAL_HISTORY
    assert len(history) == 0
    assert isinstance(history, list)

async def test_inspect_history_n_larger_than_history(capsys):
    lm = DummyLM([{"response": "First"}, {"response": "Second"}])
    dspy.settings.configure(lm=lm)
    
    predictor = dspy.Predict("query: str -> response: str")
    await predictor(dspy.settings, query="Query 1")
    await predictor(dspy.settings, query="Query 2")
    
    # Request more entries than exist
    dspy.inspect_history(n=5)
    history = GLOBAL_HISTORY
    assert len(history) == 2  # Should return all available entries
