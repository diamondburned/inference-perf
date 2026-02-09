import json
import logging
from typing import List, Optional, Union

import pydantic
from tqdm import tqdm

from inference_perf.apis import ChatCompletionAPIData, CompletionAPIData, InferenceAPIData
from inference_perf.datagen import DataGenerator, LazyLoadDataMixin
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)

TOKENS = list[int]


class MooncakeTrace(pydantic.BaseModel):
    timestamp: int  # ms
    input_length: int
    output_length: int
    level: int  # current turn index in the tree of thought generation
    token_ids: TOKENS


class MooncakeTraceGenerator:
    def __init__(self, tokenizer: CustomTokenizer):
        self.tokenizer = tokenizer

    def _extract_text(self, request_data: InferenceAPIData) -> str:
        if isinstance(request_data, CompletionAPIData):
            return request_data.prompt
        elif isinstance(request_data, ChatCompletionAPIData):
            # Combine all message content for tokenization
            # This is a simplification; actual tokenization might add role separators
            return "".join([m.content for m in request_data.messages])
        else:
            raise ValueError(f"Unsupported API data type: {type(request_data)}")

    def generate_from_datagen(
        self,
        data_generator: DataGenerator,
        num_requests: int,
        arrival_interval_ms: int = 0,
        output_length: int = 100,
    ) -> List[MooncakeTrace]:
        """
        Generates traces by pulling requests from an existing DataGenerator.

        Args:
            data_generator: The initialized DataGenerator to source prompts from.
            num_requests: Number of traces to generate.
            arrival_interval_ms: Time between requests in milliseconds (simple constant rate for now).
            output_length: Desired output length for the trace.
        """
        traces: List[MooncakeTrace] = []
        current_timestamp = 0

        # Get the generator iterator
        data_iter = data_generator.get_data()

        logger.info(f"Generating {num_requests} Mooncake traces...")

        for _ in tqdm(range(num_requests)):
            try:
                # Get next request data
                # We might need to handle StopIteration if generator runs out
                raw_data = next(data_iter)
            except StopIteration:
                logger.warning("DataGenerator exhausted before reaching requested num_requests.")
                break

            # Resolve lazy loading if necessary
            request_data = LazyLoadDataMixin.get_request(data_generator, raw_data)

            # Extract text and tokenize
            text = self._extract_text(request_data)

            # Use the underlying tokenizer to get token IDs
            # We assume the CustomTokenizer wraps a huggingface tokenizer or similar
            # that has an 'encode' or similar method, or we access the inner tokenizer.
            # CustomTokenizer has get_tokenizer() returning PreTrainedTokenizerBase
            hf_tokenizer = self.tokenizer.get_tokenizer()
            token_ids = hf_tokenizer.encode(text, add_special_tokens=False)

            trace = MooncakeTrace(
                timestamp=current_timestamp,
                input_length=len(token_ids),
                output_length=output_length,  # Fixed for now, or could come from request_data if available
                level=0,  # Default to 0 as we don't have tree info
                token_ids=token_ids,
            )
            traces.append(trace)

            current_timestamp += arrival_interval_ms

        return traces

    def save_traces(self, traces: List[MooncakeTrace], file_path: str):
        """Saves traces to a JSONL file."""
        logger.info(f"Saving {len(traces)} traces to {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            for trace in traces:
                f.write(trace.model_dump_json() + "\n")
