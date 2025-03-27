import signal
import threading
from typing import Callable
from unittest.mock import patch
import pandas as pd

import pytest
from sqlalchemy.util import await_only

import dspy
from dspy.evaluate.evaluate import Evaluate
from dspy.evaluate.metrics import answer_exact_match
from dspy.predict import Predict
from dspy.utils.callback import BaseCallback
from dspy.utils.dummies import DummyLM


def new_example(question, answer):
    """Helper function to create a new example."""
    return dspy.Example(
        question=question,
        answer=answer,
    ).with_inputs("question")


def test_evaluate_initialization():
    devset = [new_example("What is 1+1?", "2")]
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    assert ev.devset == devset
    assert ev.metric == answer_exact_match
    assert ev.num_threads == len(devset)
    assert ev.display_progress == False


async def test_evaluate_call():
    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        )
    )
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    result = await program(dspy.settings, question="What is 1+1?")
    assert result.answer == "2"
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    score = await ev(dspy.settings, program)
    assert score == 100.0


def test_construct_result_df():
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
    )
    results = [
        (devset[0], {"answer": "2"}, 100.0),
        (devset[1], {"answer": "4"}, 100.0),
    ]
    result_df = ev._construct_result_table(results, answer_exact_match.__name__)
    pd.testing.assert_frame_equal(
        result_df,
        pd.DataFrame(
            {
                "question": ["What is 1+1?", "What is 2+2?"],
                "example_answer": ["2", "4"],
                "pred_answer": ["2", "4"],
                "answer_exact_match": [100.0, 100.0],
            }
        )
    )


async def test_multithread_evaluate_call():
    dspy.settings.configure(lm=DummyLM({"What is 1+1?": {"answer": "2"}, "What is 2+2?": {"answer": "4"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    answer = await program(dspy.settings, question="What is 1+1?")
    assert answer.answer == "2"
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
        num_threads=2,
    )
    score = await ev(dspy.settings, program)
    assert score == 100.0


async def test_evaluate_call_bad():
    dspy.settings.configure(lm=DummyLM({"What is 1+1?": {"answer": "0"}, "What is 2+2?": {"answer": "0"}}))
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    score = await ev(dspy.settings, program)
    assert score == 0.0


def get_predict_def(sig: str, key) -> Callable:
    async def x(settings, text: str):
        result = await Predict(sig)(settings, text=text)
        return result[key]
    
    return x

@pytest.mark.parametrize(
    "program_with_example",
    [
        (Predict("question -> answer"), new_example("What is 1+1?", "2")),
        # Create programs that do not return dictionary-like objects because Evaluate()
        # has failed for such cases in the past
        (
            get_predict_def("text: str -> entities: List[str]", "entities"),
            dspy.Example(text="United States", entities=["United States"]).with_inputs("text"),
        ),
        (
            get_predict_def("text: str -> entities: List[Dict[str, str]]", "entities"),
            dspy.Example(text="United States", entities=[{"name": "United States", "type": "location"}]).with_inputs(
                "text"
            ),
        ),
        (
            get_predict_def("text: str -> first_word: Tuple[str, int]", "first_word"),
            dspy.Example(text="United States", first_word=("United", 6)).with_inputs("text"),
        ),
    ],
)
@pytest.mark.parametrize("display_table", [True, False, 1])
@pytest.mark.parametrize("is_in_ipython_notebook_environment", [True, False])
async def test_evaluate_display_table(program_with_example, display_table, is_in_ipython_notebook_environment, capfd):
    program, example = program_with_example
    example_input = next(iter(example.inputs().values()))
    example_output = {key: value for key, value in example.toDict().items() if key not in example.inputs()}

    dspy.settings.configure(
        lm=DummyLM(
            {
                example_input: example_output,
            }
        )
    )
    
    async def metric(settings, example, pred, **kwargs):
        return example == pred

    ev = Evaluate(
        devset=[example],
        metric=metric,
        display_table=display_table,
    )
    assert ev.display_table == display_table

    with patch(
        "dspy.evaluate.evaluate.is_in_ipython_notebook_environment", return_value=is_in_ipython_notebook_environment
    ):
        await ev(dspy.settings, program)
        out, _ = capfd.readouterr()
        if not is_in_ipython_notebook_environment and display_table:
            # In console environments where IPython is not available, the table should be printed
            # to the console
            example_input = next(iter(example.inputs().values()))
            assert example_input in out

async def test_evaluate_callback():
    class TestCallback(BaseCallback):
        def __init__(self):
            self.start_call_inputs = None
            self.start_call_count = 0
            self.end_call_outputs = None
            self.end_call_count = 0

        def on_evaluate_start(
            self,
            call_id: str,
            instance,
            settings,
            inputs,
        ):
            self.start_call_inputs = inputs
            self.start_call_count += 1
        
        def on_evaluate_end(
            self,
            call_id: str,
            settings,
            outputs,
            exception = None,
        ):
            self.end_call_outputs = outputs
            self.end_call_count += 1

    callback = TestCallback()
    dspy.settings.configure(
        lm=DummyLM(
            {
                "What is 1+1?": {"answer": "2"},
                "What is 2+2?": {"answer": "4"},
            }
        ),
        callbacks=[callback]
    )
    devset = [new_example("What is 1+1?", "2"), new_example("What is 2+2?", "4")]
    program = Predict("question -> answer")
    answer = await program(dspy.settings, question="What is 1+1?")
    assert answer.answer == "2"
    ev = Evaluate(
        devset=devset,
        metric=answer_exact_match,
        display_progress=False,
    )
    score = await ev(dspy.settings, program)
    assert score == 100.0
    assert callback.start_call_inputs["program"] == program
    assert callback.start_call_count == 1
    assert callback.end_call_outputs == 100.0
    assert callback.end_call_count == 1
