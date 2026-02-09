import csv
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Iterator, List, Tuple, Optional

from tqdm import tqdm

from inference_perf.apis import InferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.datagen import DataGenerator, LazyLoadDataMixin
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)


class AzurePublicDatasetTraceEntry:
    """Represents a single trace entry with timing and token information."""

    def __init__(
        self,
        timestamp: float,
        input_tokens: int,
        output_tokens: int,
        prompt: Optional[str] = None,
        completion: Optional[str] = None,
    ):
        self.timestamp = timestamp
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.prompt = prompt
        self.completion = completion


class TraceReader(ABC):
    """Abstract base class for streaming trace readers."""

    @abstractmethod
    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream trace entries one by one"""
        raise NotImplementedError

    @abstractmethod
    def load_traces(self, file_path: Path) -> List[Tuple[float, int, int]]:
        """Load traces from file."""
        raise NotImplementedError


class AzurePublicDatasetTraceReader(TraceReader):
    """Trace reader for Azure Public Dataset format."""

    def __init__(self) -> None:
        self.timestamp_format = "%Y-%m-%d %H:%M:%S.%f"
        self.traces = None

    def load_traces(self, file_path: Path) -> List[Tuple[float, int, int]]:
        """Load traces from file into memory."""
        if self.traces is not None:
            return self.traces
        logger.info(f"Loading traces from {file_path}")
        traces = []
        start_line = 1
        initial_timestamp: float = 0
        before = time.time()
        with open(file_path, "r", encoding="utf-8") as f:
            if self.has_header(file_path):
                start_line = 2
                next(f)
            for line_num, line in enumerate(f, start_line):
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(",")
                        timestamp = self.parse_timestamp(entry_data[0])
                        if line_num == start_line:
                            initial_timestamp = timestamp
                        traces.append((timestamp - initial_timestamp, int(entry_data[1].strip()), int(entry_data[2].strip())))
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        after = time.time()
        logger.info(f"Time taken to load traces: {after - before} seconds")
        return traces

    def stream_token_entries(self, file_path: Path) -> Iterator[Tuple[int, int]]:
        """Stream entries from AzurePublicDataset format"""
        start_line = 1
        with open(file_path, "r", encoding="utf-8") as f:
            if self.has_header(file_path):
                start_line = 2
                next(f)
            for line_num, line in enumerate(f, start_line):
                try:
                    if line.strip():  # Skip empty lines
                        entry_data = line.split(",")
                        yield int(entry_data[1].strip()), int(entry_data[2].strip())
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")

    def parse_timestamp(self, timestamp: str) -> float:
        """Parse timestamp from string to float."""

        raw_ts = timestamp.strip().strip('"')
        # Normalize to "YYYY-MM-DD HH:MM:SS.ff" in UTC
        ts = raw_ts.replace("T", " ").rstrip("Z").strip()
        if "." in ts:
            head, frac = ts.split(".", 1)
            # Keep only digits in fractional seconds and coerce to 2 digits
            frac_digits = "".join(ch for ch in frac if ch.isdigit())
            frac2 = (frac_digits[:2]).ljust(2, "0")
            ts_clean = f"{head}.{frac2}"
        else:
            ts_clean = f"{ts}.00"
        return datetime.strptime(ts_clean, self.timestamp_format).replace(tzinfo=timezone.utc).timestamp()

    def has_header(self, file_path: Path) -> bool:
        """Check if the file has a header."""
        with open(file_path, "r", encoding="utf-8") as f:
            sniffer = csv.Sniffer()
            has_header = sniffer.has_header(f.read(2048))
            f.seek(0)
            return has_header


class AzurePublicDatasetTraceGenerator:
    def __init__(self, tokenizer: CustomTokenizer):
        self.tokenizer = tokenizer

    def _extract_text_and_counts(self, request_data: InferenceAPIData) -> Tuple[int, int]:
        input_text = ""
        output_tokens = 0

        if isinstance(request_data, CompletionAPIData):
            input_text = request_data.prompt
            output_tokens = request_data.max_tokens
        elif isinstance(request_data, ChatCompletionAPIData):
            input_text = "".join([m.content for m in request_data.messages])
            output_tokens = request_data.max_tokens
        else:
            logging.warning(f"AzurePublicDatasetTraceGenerator: Unknown API data type: {type(request_data)}")

        input_tokens = self.tokenizer.count_tokens(input_text)
        return input_tokens, output_tokens

    def generate_from_datagen(
        self,
        data_generator: DataGenerator,
        num_requests: int,
        arrival_interval_ms: int = 0,
    ) -> List[Tuple[str, int, int]]:
        """
        Generates traces in Azure Public Dataset format.
        Returns a list of (timestamp_str, input_tokens, output_tokens).
        """
        traces = []
        current_time = datetime.now(timezone.utc)

        # Get the generator iterator
        data_iter = data_generator.get_data()

        logger.info(f"Generating {num_requests} Azure traces...")

        for _ in tqdm(range(num_requests)):
            try:
                raw_data = next(data_iter)
            except StopIteration:
                logger.warning("DataGenerator exhausted before reaching requested num_requests.")
                break

            request_data = LazyLoadDataMixin.get_request(data_generator, raw_data)

            input_tokens, output_tokens = self._extract_text_and_counts(request_data)

            # Format: 2023-01-01 12:00:00.000000
            timestamp_str = current_time.strftime("%Y-%m-%d %H:%M:%S.%f")

            traces.append((timestamp_str, input_tokens, output_tokens))

            current_time += timedelta(milliseconds=arrival_interval_ms)

        return traces

    def save_traces(self, traces: List[Tuple[str, int, int]], file_path: str):
        """Saves traces to a CSV file matching Azure Public Dataset format."""
        logger.info(f"Saving {len(traces)} traces to {file_path}")
        with open(file_path, "w", encoding="utf-8") as f:
            # Azure format often has no header or specific header.
            # Based on AzurePublicDatasetReader, it checks for header.
            # Let's write a header for clarity, the reader supports it.
            f.write("Timestamp,InputTokens,OutputTokens\n")
            for ts, inp, out in traces:
                f.write(f"{ts},{inp},{out}\n")
