import time

import pytest

import dspy
from dspy.utils.callback import ACTIVE_CALL_ID, BaseCallback, with_callbacks
from dspy.utils.dummies import DummyLM


@pytest.fixture(autouse=True)
def reset_settings():
    # Make sure the settings are reset after each test
    original_settings = dspy.settings.copy()

    yield

    dspy.settings.configure(**original_settings)


class MyCallback(BaseCallback):
    """A simple callback that records the calls."""

    def __init__(self):
        self.calls = []

    async def on_module_start(self, call_id, instance, settings, inputs):
        self.calls.append({"handler": "on_module_start", "instance": instance, "inputs": inputs})

    async def on_module_end(self, call_id, settings, outputs, exception):
        self.calls.append({"handler": "on_module_end", "outputs": outputs, "exception": exception})

    async def on_lm_start(self, call_id, instance, settings, inputs):
        self.calls.append({"handler": "on_lm_start", "instance": instance, "inputs": inputs})

    async def on_lm_end(self, call_id, settings, outputs, exception):
        self.calls.append({"handler": "on_lm_end", "outputs": outputs, "exception": exception})

    async def on_adapter_format_start(self, call_id, instance, settings, inputs):
        self.calls.append({"handler": "on_adapter_format_start", "instance": instance, "inputs": inputs})

    async def on_adapter_format_end(self, call_id, settings, outputs, exception):
        self.calls.append({"handler": "on_adapter_format_end", "outputs": outputs, "exception": exception})

    async def on_adapter_parse_start(self, call_id, instance, settings, inputs):
        self.calls.append({"handler": "on_adapter_parse_start", "instance": instance, "inputs": inputs})

    async def on_adapter_parse_end(self, call_id, settings, outputs, exception):
        self.calls.append({"handler": "on_adapter_parse_end", "outputs": outputs, "exception": exception})

    async def on_tool_start(self, call_id, instance, settings, inputs):
        self.calls.append({"handler": "on_tool_start", "instance": instance, "inputs": inputs})

    async def on_tool_end(self, call_id, settings, outputs, exception):
        self.calls.append({"handler": "on_tool_end", "outputs": outputs, "exception": exception})


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ([1, "2", 3.0], {}),
        ([1, "2"], {"z": 3.0}),
        ([1], {"y": "2", "z": 3.0}),
        ([], {"x": 1, "y": "2", "z": 3.0}),
    ],
)
async def test_callback_injection(args, kwargs):
    class Target(dspy.Module):
        @with_callbacks
        async def forward(self, settings, x: int, y: str, z: float) -> int:
            time.sleep(0.1)
            return x + int(y) + int(z)

    callback = MyCallback()
    dspy.settings.configure(callbacks=[callback])

    target = Target()
    with dspy.context() as settings:
        result = await target.forward(settings, *args, **kwargs)

    assert result == 6

    assert len(callback.calls) == 2
    assert callback.calls[0]["handler"] == "on_module_start"
    assert callback.calls[0]["inputs"] == {"x": 1, "y": "2", "z": 3.0}
    assert callback.calls[1]["handler"] == "on_module_end"
    assert callback.calls[1]["outputs"] == 6


async def test_callback_injection_local():
    class Target(dspy.Module):
        @with_callbacks
        async def forward(self, settings, x: int, y: str, z: float) -> int:
            time.sleep(0.1)
            return x + int(y) + int(z)

    callback = MyCallback()

    target_1 = Target(callbacks=[callback])
    
    with dspy.context() as settings:
        result = await target_1.forward(settings, 1, "2", 3.0)

    assert result == 6

    assert len(callback.calls) == 2
    assert callback.calls[0]["handler"] == "on_module_start"
    assert callback.calls[0]["inputs"] == {"x": 1, "y": "2", "z": 3.0}
    assert callback.calls[1]["handler"] == "on_module_end"
    assert callback.calls[1]["outputs"] == 6

    callback.calls = []

    target_2 = Target()

    with dspy.context() as settings:
        result = await target_2.forward(settings, 1, "2", 3.0)

    # Other instance should not trigger the callback
    assert not callback.calls


async def test_callback_error_handling():
    class Target(dspy.Module):
        @with_callbacks
        async def forward(self, settings, x: int, y: str, z: float) -> int:
            time.sleep(0.1)
            raise ValueError("Error")

    callback = MyCallback()

    target = Target()

    with pytest.raises(ValueError, match="Error"):
        with dspy.context() as settings:
            settings.configure(callbacks=[callback])
            await target.forward(settings, 1, "2", 3.0)

    assert len(callback.calls) == 2
    assert callback.calls[0]["handler"] == "on_module_start"
    assert callback.calls[1]["handler"] == "on_module_end"
    assert isinstance(callback.calls[1]["exception"], ValueError)


async def test_multiple_callbacks():
    class Target(dspy.Module):
        @with_callbacks
        async def forward(self, settings, x: int, y: str, z: float) -> int:
            time.sleep(0.1)
            return x + int(y) + int(z)

    callback_1 = MyCallback()
    callback_2 = MyCallback()

    target = Target()
    with dspy.context() as settings:
        settings.configure(callbacks=[callback_1, callback_2])
        result = await target.forward(settings, 1, "2", 3.0)

    assert result == 6

    assert len(callback_1.calls) == 2
    assert len(callback_2.calls) == 2


async def test_callback_complex_module():
    callback = MyCallback()

    cot = dspy.ChainOfThought("question -> answer", n=3)
    
    with dspy.context() as settings:
        settings.configure(
            lm=DummyLM({"How are you?": {"answer": "test output", "reasoning": "No more responses"}}),
            callbacks=[callback],
        )
        
        result = await cot(settings=settings, question="How are you?")
    
    assert result["answer"] == "test output"
    assert result["reasoning"] == "No more responses"

    assert len(callback.calls) == 14
    assert [call["handler"] for call in callback.calls] == [
        "on_module_start",
        "on_module_start",
        "on_adapter_format_start",
        "on_adapter_format_end",
        "on_lm_start",
        "on_lm_end",
        # Parsing will run per output (n=3)
        "on_adapter_parse_start",
        "on_adapter_parse_end",
        "on_adapter_parse_start",
        "on_adapter_parse_end",
        "on_adapter_parse_start",
        "on_adapter_parse_end",
        "on_module_end",
        "on_module_end",
    ]


async def test_tool_calls():
    callback = MyCallback()

    async def tool_1(settings, query: str) -> str:
        """A dummy tool function."""
        return "result 1"

    async def tool_2(settings, query: str) -> str:
        """Another dummy tool function."""
        return "result 2"

    class MyModule(dspy.Module):
        def __init__(self):
            self.tools = [dspy.Tool(tool_1), dspy.Tool(tool_2)]

        async def forward(self, settings, query: str) -> str:
            query = await self.tools[0](settings, query)
            return await self.tools[1](settings, query)

    module = MyModule()
    
    with dspy.context() as settings:
        settings.configure(callbacks=[callback])
        result = await module(settings, "query")

    assert result == "result 2"
    assert len(callback.calls) == 6
    assert [call["handler"] for call in callback.calls] == [
        "on_module_start",
        "on_tool_start",
        "on_tool_end",
        "on_tool_start",
        "on_tool_end",
        "on_module_end",
    ]


async def test_active_id():
    # Test the call ID is generated and handled properly
    class CustomCallback(BaseCallback):
        def __init__(self):
            self.parent_call_ids = []
            self.call_ids = []

        async def on_module_start(self, call_id, instance, settings, inputs):
            parent_call_id = ACTIVE_CALL_ID.get()
            self.parent_call_ids.append(parent_call_id)
            self.call_ids.append(call_id)

    class Parent(dspy.Module):
        def __init__(self):
            self.child_1 = Child()
            self.child_2 = Child()

        async def forward(self, settings):
            await self.child_1(settings)
            await self.child_2(settings)

    class Child(dspy.Module):
        async def forward(self, settings):
            pass

    callback = CustomCallback()

    parent = Parent()
    
    with dspy.context() as settings:
        settings.configure(callbacks=[callback])
        await parent(settings)

    assert len(callback.call_ids) == 3
    # All three calls should have different call ids
    assert len(set(callback.call_ids)) == 3
    parent_call_id = callback.call_ids[0]
    assert callback.parent_call_ids == [None, parent_call_id, parent_call_id]
