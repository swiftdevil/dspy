import dspy
from dspy import Example
from dspy.predict import Predict
from dspy.teleprompt import BootstrapFewShotWithRandomSearch
from dspy.utils.dummies import DummyLM


class SimpleModule(dspy.Module):
    def __init__(self, signature):
        super().__init__()
        self.predictor = Predict(signature)

    async def forward(self, settings, **kwargs):
        return await self.predictor(settings, **kwargs)


async def simple_metric(settings, example, prediction, trace=None):
    return example.output == prediction.output


async def test_basic_workflow():
    """Test to ensure the basic compile flow runs without errors."""
    student = SimpleModule("input -> output")
    teacher = SimpleModule("input -> output")

    lm = DummyLM(
        [
            {
                "input": "Initial thoughts",
                "output": "Finish[blue]"
            },  # Expected output for both training and validation
        ]
    )
    dspy.settings.configure(lm=lm)

    optimizer = BootstrapFewShotWithRandomSearch(metric=simple_metric, max_bootstrapped_demos=1, max_labeled_demos=1)
    trainset = [
        Example(input="What is the color of the sky?", output="blue").with_inputs("input"),
        Example(input="What does the fox say?", output="Ring-ding-ding-ding-dingeringeding!").with_inputs("input"),
    ]
    await optimizer.compile(dspy.settings, student, teacher=teacher, trainset=trainset)
