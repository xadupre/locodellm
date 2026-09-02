"""Connects ONNX Runtime GenAI models to LM Evaluation Harness."""

from __future__ import annotations

from typing import Any

from lm_eval.api.model import LM

from locodellm.generate.generate_from_model import get_session


class OnnxRuntimeGenAILM(LM):
    """Runs LM-Eval generation requests with an ONNX Runtime GenAI model."""

    def __init__(
        self,
        model: str,
        precision: str | None = None,
        provider: str | None = None,
        provider_options: dict[str, str] | None = None,
        chat_template: str | None = None,
        max_length: int = 2048,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.model_id = model
        self.max_length = int(max_length)
        self.session = get_session(
            model_id=model,
            precision=precision,
            provider=provider,
            provider_options=provider_options,
            chat_template=chat_template,
            verbose=int(verbose),
        )

    @property
    def tokenizer_name(self) -> str:
        """Returns the model identifier used to fingerprint LM-Eval requests."""
        return self.model_id

    def loglikelihood(self, requests: list[Any]) -> list[tuple[float, bool]]:
        """Rejects likelihood requests unsupported by ONNX Runtime GenAI."""
        raise NotImplementedError("ONNX Runtime GenAI only supports LM-Eval generation tasks.")

    def loglikelihood_rolling(self, requests: list[Any]) -> list[float]:
        """Rejects rolling likelihood requests unsupported by ONNX Runtime GenAI."""
        raise NotImplementedError("ONNX Runtime GenAI only supports LM-Eval generation tasks.")

    def generate_until(self, requests: list[Any]) -> list[str]:
        """Generates one continuation for every LM-Eval request."""
        responses = []
        for request in requests:
            context, generation_kwargs = request.args
            options = dict(generation_kwargs)
            until = options.pop("until", [])
            if until is None:
                until = []
            elif isinstance(until, str):
                until = [until]
            max_gen_toks = int(options.pop("max_gen_toks", 256))

            current = self.session.new_session()
            prompt_length = len(current.tokenizer.encode(current._wrap_prompt(context)))
            max_length = min(prompt_length + max_gen_toks, self.max_length)
            if max_length <= prompt_length:
                raise ValueError(
                    f"The prompt has {prompt_length} tokens and exceeds max_length="
                    f"{self.max_length}."
                )

            current.generate(context, max_length=max_length, **options)
            response = current.text
            stop_positions = [response.find(stop) for stop in until if stop and stop in response]
            if stop_positions:
                response = response[: min(stop_positions)]
            responses.append(response)
        return responses


def run_lm_eval(
    model: str,
    tasks: list[str],
    precision: str | None = None,
    provider: str | None = None,
    provider_options: dict[str, str] | None = None,
    chat_template: str | None = None,
    max_length: int = 2048,
    num_fewshot: int | None = None,
    limit: float | None = None,
    verbose: int = 0,
) -> dict[str, Any] | None:
    """Runs LM Evaluation Harness with an ONNX Runtime GenAI model."""
    from lm_eval.evaluator import simple_evaluate

    evaluator_model = OnnxRuntimeGenAILM(
        model=model,
        precision=precision,
        provider=provider,
        provider_options=provider_options,
        chat_template=chat_template,
        max_length=max_length,
        verbose=verbose,
    )
    return simple_evaluate(
        model=evaluator_model, tasks=tasks, num_fewshot=num_fewshot, limit=limit
    )
