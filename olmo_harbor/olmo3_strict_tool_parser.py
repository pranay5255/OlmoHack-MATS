"""vLLM plugin for the experiment's strict OLMo3 native-tool contract."""

from __future__ import annotations

from typing import Any

from vllm.entrypoints.openai.engine.protocol import (
    ExtractedToolCallInformation,
)
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
from vllm.tool_parsers.olmo3_tool_parser import (
    Olmo3PythonicToolParser,
)

from olmo3_tool_contract import normalize_olmo3_bash_call


@ToolParserManager.register_module("olmo3_strict")
class Olmo3StrictToolParser(Olmo3PythonicToolParser):
    """OLMo3 parser with narrow normalization of observed complete variants."""

    # The default generic required/named-tool path bypassed this parser and suppressed
    # Think-family reasoning in vLLM 0.21. The contract therefore uses auto.
    supports_required_and_named = False

    def extract_tool_calls(
        self, model_output: str, request: Any
    ) -> ExtractedToolCallInformation:
        parsed = super().extract_tool_calls(model_output, request)
        if parsed.tools_called:
            return parsed
        normalized = normalize_olmo3_bash_call(model_output)
        if normalized is None or normalized == model_output:
            return parsed
        return super().extract_tool_calls(normalized, request)
