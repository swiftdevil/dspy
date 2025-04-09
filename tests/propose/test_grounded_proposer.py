import pytest
import dspy
from dspy.propose.grounded_proposer import GroundedProposer
from dspy.predict import Predict
from dspy.utils.dummies import DummyLM


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
async def test_propose_instructions_for_program(demo_candidates):
    # Set large numner here so that lm always returns the same response
    prompt_model = DummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")
    trainset = []

    proposer = GroundedProposer(prompt_model=prompt_model, program=program, trainset=trainset, verbose=False)
    result = await proposer.propose_instructions_for_program(
        settings=dspy.settings, trainset=trainset, program=program, demo_candidates=demo_candidates, trial_logs={}, N=1, T=0.5
    )
    assert isinstance(result, dict)
    assert len(result) == len(program.predictors())
    for pred_instructions in result.values():
        assert pred_instructions == ["instruction"]


@pytest.mark.parametrize(
    "demo_candidates",
    [
        None,
        [[[dspy.Example(question="What is the capital of France?", answer="Paris")]]],
    ],
)
async def test_propose_instruction_for_predictor(demo_candidates):
    prompt_model = DummyLM([{"proposed_instruction": "instruction"}] * 10)
    program = Predict("question -> answer")

    proposer = GroundedProposer(prompt_model=prompt_model, program=program, trainset=[], verbose=False)
    result = await proposer.propose_instruction_for_predictor(
        settings=dspy.settings,
        program=program,
        predictor=None,
        pred_i=0,
        T=0.5,
        demo_candidates=demo_candidates,
        demo_set_i=0,
        trial_logs={},
        tip=None,
    )
    assert result == "instruction"
