from abc import ABC, abstractmethod

from litellm import ContextWindowExceededError

from dspy.utils.callback import with_callbacks


class Adapter(ABC):
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)

        # Decorate format() and parse() method with with_callbacks
        cls.format = with_callbacks(cls.format)
        cls.parse = with_callbacks(cls.parse)

    async def __call__(self, settings, lm, lm_kwargs, signature, demos, inputs):
        inputs_ = await self.format(settings, signature, demos, inputs)
        inputs_ = dict(prompt=inputs_) if isinstance(inputs_, str) else dict(messages=inputs_)

        outputs = await lm(settings, **inputs_, **lm_kwargs)
        values = []

        try:
            for output in outputs:
                output_logprobs = None

                if isinstance(output, dict):
                    output, output_logprobs = output["text"], output["logprobs"]

                value = await self.parse(settings, signature, output)

                if set(value.keys()) != set(signature.output_fields.keys()):
                    raise ValueError(
                        "Parsed output fields do not match signature output fields. "
                        f"Expected: {set(signature.output_fields.keys())}, Got: {set(value.keys())}"
                    )

                if output_logprobs is not None:
                    value["logprobs"] = output_logprobs

                values.append(value)

            return values

        except Exception as e:
            if isinstance(e, ContextWindowExceededError):
                # On context window exceeded error, we don't want to retry with a different adapter.
                raise e
            from dspy.adapters.json_adapter import JSONAdapter

            if not isinstance(self, JSONAdapter):
                return JSONAdapter()(lm, lm_kwargs, signature, demos, inputs)
            raise e

    @abstractmethod
    async def format(self, settings, signature, demos, inputs):
        raise NotImplementedError

    @abstractmethod
    async def parse(self, settings, signature, completion):
        raise NotImplementedError

    async def format_finetune_data(self, settings, signature, demos, inputs, outputs):
        raise NotImplementedError
