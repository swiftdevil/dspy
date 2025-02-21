import dspy

from dspy.utils.dummies import DummyLM


async def test_parallel_module():
    lm = DummyLM([
        {"output": "test output 1"},
        {"output": "test output 2"},
        {"output": "test output 3"},
        {"output": "test output 4"},
        {"output": "test output 5"},
    ])

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor2 = dspy.Predict("input -> output")

            self.parallel = dspy.Parallel(num_threads=2)

        async def forward(self, settings, input):
            return await self.parallel(settings, [
                (self.predictor, input),
                (self.predictor2, input),
                (self.predictor, input),
                (self.predictor2, input),
                (self.predictor, input),
            ])

    with dspy.context as settings:
        settings.configure(lm=lm)
        output = await MyModule()(settings, dspy.Example(input="test input").with_inputs("input"))

    assert output[0].output == "test output 1"
    assert output[1].output == "test output 2"
    assert output[2].output == "test output 3"
    assert output[3].output == "test output 4"
    assert output[4].output == "test output 5"


async def test_batch_module():
    lm = DummyLM([
        {"output": "test output 1"},
        {"output": "test output 2"},
        {"output": "test output 3"},
        {"output": "test output 4"},
        {"output": "test output 5"},
    ])
    res_lm = DummyLM([
        {"output": "test output 1", "reasoning": "test reasoning 1"},
        {"output": "test output 2", "reasoning": "test reasoning 2"},
        {"output": "test output 3", "reasoning": "test reasoning 3"},
        {"output": "test output 4", "reasoning": "test reasoning 4"},
        {"output": "test output 5", "reasoning": "test reasoning 5"},
    ])

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor2 = dspy.Predict("input -> output, reasoning")

            self.parallel = dspy.Parallel(num_threads=2)

        async def forward(self, settings, input):
            dspy.settings.configure(lm=lm)
            res1 = await self.predictor.batch([input] * 5)

            dspy.settings.configure(lm=res_lm)
            res2 = await self.predictor2.batch([input] * 5)

            return (res1, res2)
        
    result, reason_result = await MyModule()(dspy.Example(input="test input").with_inputs("input"))

    assert result[0].output == "test output 1"
    assert result[1].output == "test output 2"
    assert result[2].output == "test output 3"
    assert result[3].output == "test output 4"
    assert result[4].output == "test output 5"

    assert reason_result[0].output == "test output 1"
    assert reason_result[1].output == "test output 2"
    assert reason_result[2].output == "test output 3"
    assert reason_result[3].output == "test output 4"
    assert reason_result[4].output == "test output 5"

    assert reason_result[0].reasoning == "test reasoning 1"
    assert reason_result[1].reasoning == "test reasoning 2"
    assert reason_result[2].reasoning == "test reasoning 3"
    assert reason_result[3].reasoning == "test reasoning 4"
    assert reason_result[4].reasoning == "test reasoning 5"


async def test_nested_parallel_module():
    lm = DummyLM([
        {"output": "test output 1"},
        {"output": "test output 2"},
        {"output": "test output 3"},
        {"output": "test output 4"},
        {"output": "test output 5"},
    ])
    dspy.settings.configure(lm=lm)

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")
            self.predictor2 = dspy.Predict("input -> output")

            self.parallel = dspy.Parallel(num_threads=2)

        async def forward(self, settings, input):
            return await self.parallel(settings, [
                (self.predictor, input),
                (self.predictor2, input),
                (self.parallel, [
                    (self.predictor2, input),
                    (self.predictor, input),
                ]),
            ])
        
    output = await MyModule()(settings, dspy.Example(input="test input").with_inputs("input"))

    assert output[0].output == "test output 1"
    assert output[1].output == "test output 2"
    assert output[2][0].output == "test output 3"
    assert output[2][1].output == "test output 4"


async def test_nested_batch_method():
    lm = DummyLM([
        {"output": "test output 1"},
        {"output": "test output 2"},
        {"output": "test output 3"},
        {"output": "test output 4"},
        {"output": "test output 5"},
    ])
    dspy.settings.configure(lm=lm)

    class MyModule(dspy.Module):
        def __init__(self):
            super().__init__()
            self.predictor = dspy.Predict("input -> output")

        async def forward(self, settings, input):
            res = await self.predictor.batch([dspy.Example(input=input).with_inputs("input")]*2)

            return res
        
    result = await MyModule().batch([dspy.Example(input="test input").with_inputs("input")]*2)

    assert {result[0][0].output, result[0][1].output, result[1][0].output, result[1][1].output} \
            == {"test output 1", "test output 2", "test output 3", "test output 4"}
