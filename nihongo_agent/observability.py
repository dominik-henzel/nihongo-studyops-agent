"""Observability, distributed tracing, structured JSON logging, and PII redaction.

Implements four production observability pillars without any ``print()`` calls:
1. **PII Redaction**: ``PiiRedactionScrubber`` combines deterministic regex rules
   (email, phone, US SSN, Japanese 12-digit MyNumber, API keys) with optional
   ``google.cloud.dlp_v2.DlpServiceClient`` de-identification before data enters
   logs or persistent memory.
2. **Distributed Tracing**: OpenTelemetry ``TracerProvider`` and ``SimpleSpanProcessor``
   with ``trace_span()`` context management linking parent and child spans from
   coordinator request through tool execution.
3. **Structured JSON Logging**: ``StructuredJsonFormatter`` serializing log records
   as single-line JSON payloads with trace correlation and optional
   ``google-cloud-logging`` attachment.
4. **Intent vs. Outcome Capture**: ``log_agent_intent(phase="INTENT")``,
   ``log_agent_outcome(phase="OUTCOME")``, and the ``@observe_tool_execution``
   decorator capturing pre-execution intent, post-execution status, and latency.
"""

from __future__ import annotations

import contextlib
import functools
import json
import logging
import os
import re
import time
import uuid
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any, Callable, Iterator, Optional, TypeVar

try:
    from google.cloud import dlp_v2  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    dlp_v2 = None  # type: ignore[assignment]

try:
    from google.cloud import logging as cloud_logging  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    cloud_logging = None  # type: ignore[assignment]

try:
    from opentelemetry import trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import ReadableSpan, SpanProcessor, TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExporter, SpanExportResult

    OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    trace = None  # type: ignore[assignment]
    ReadableSpan = object  # type: ignore[assignment,misc]
    SpanProcessor = object  # type: ignore[assignment,misc]
    TracerProvider = object  # type: ignore[assignment,misc]
    SimpleSpanProcessor = None  # type: ignore[assignment]
    SpanExporter = object  # type: ignore[assignment,misc]
    SpanExportResult = None  # type: ignore[assignment]
    Resource = None  # type: ignore[assignment]
    OTEL_AVAILABLE = False

F = TypeVar("F", bound=Callable[..., Any])

# Context variables for correlating structured logs with active OpenTelemetry spans.
_CURRENT_TRACE_ID: ContextVar[str] = ContextVar("current_trace_id", default="")
_CURRENT_SPAN_ID: ContextVar[str] = ContextVar("current_span_id", default="")

# In-memory ring buffers so automated test suites can assert emitted JSON logs and spans.
_STRUCTURED_LOG_BUFFER: list[dict[str, Any]] = []
_RECORDED_SPANS_BUFFER: list[dict[str, Any]] = []


class PiiRedactionScrubber:
    """Scrubs sensitive PII from text and nested payloads via regex and optional Cloud DLP."""

    # Ordered from specific patterns to general patterns to prevent partial masking collisions.
    _REGEX_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
        (
            "[REDACTED_API_KEY]",
            re.compile(
                r"\b(?:AIza[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{20,}|ya29\.[0-9A-Za-z._-]+)\b"
            ),
        ),
        (
            "[REDACTED_EMAIL]",
            re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
        ),
        (
            "[REDACTED_SSN]",
            re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        ),
        (
            "[REDACTED_MY_NUMBER]",
            re.compile(r"\b\d{4}[-\s]\d{4}[-\s]\d{4}\b|\b\d{12}\b"),
        ),
        (
            "[REDACTED_PHONE]",
            re.compile(
                r"(?:\+?\d{1,3}[-\s])?(?:0\d{1,4}[-\s]\d{1,4}[-\s]\d{3,4}|\(\d{3}\)\s?\d{3}-\d{4}|\b\d{3}-\d{3}-\d{4}\b)"
            ),
        ),
    )

    def __init__(self, project_id: Optional[str] = None, enable_cloud_dlp: bool = False) -> None:
        self.project_id = project_id or os.environ.get("GOOGLE_CLOUD_PROJECT")
        self.enable_cloud_dlp = enable_cloud_dlp
        self._dlp_client: Optional[Any] = None

    def _get_dlp_client(self) -> Optional[Any]:
        """Lazily initialize ``DlpServiceClient`` when Cloud DLP is explicitly enabled."""
        if not self.enable_cloud_dlp or dlp_v2 is None or not self.project_id:
            return None
        if os.environ.get("NIHONGO_AGENT_FORCE_OFFLINE", "").lower() in {"1", "true", "yes"}:
            return None
        if self._dlp_client is None:
            try:
                self._dlp_client = dlp_v2.DlpServiceClient()
            except Exception:
                self._dlp_client = None
        return self._dlp_client

    def scrub_text(self, text: str) -> str:
        """Redact PII tokens from ``text`` using regex rules and optional Cloud DLP."""
        if not text:
            return text

        scrubbed = text
        for replacement, pattern in self._REGEX_RULES:
            scrubbed = pattern.sub(replacement, scrubbed)

        client = self._get_dlp_client()
        if client is not None and self.project_id:
            try:
                parent = f"projects/{self.project_id}/locations/global"
                inspect_config = {
                    "info_types": [
                        {"name": "EMAIL_ADDRESS"},
                        {"name": "PHONE_NUMBER"},
                        {"name": "JAPAN_INDIVIDUAL_NUMBER"},
                        {"name": "US_SOCIAL_SECURITY_NUMBER"},
                    ]
                }
                deidentify_config = {
                    "info_type_transformations": {
                        "transformations": [
                            {
                                "primitive_transformation": {
                                    "replace_with_info_type_config": {}
                                }
                            }
                        ]
                    }
                }
                response = client.deidentify_content(
                    request={
                        "parent": parent,
                        "inspect_config": inspect_config,
                        "deidentify_config": deidentify_config,
                        "item": {"value": scrubbed},
                    }
                )
                if getattr(response, "item", None) and response.item.value:
                    scrubbed = response.item.value
            except Exception:
                # Deterministic regex scrubbing has already been applied above.
                pass

        return scrubbed

    def scrub_payload(self, payload: Any) -> Any:
        """Recursively scrub strings within dicts, lists, and tuples."""
        if isinstance(payload, str):
            return self.scrub_text(payload)
        if isinstance(payload, dict):
            return {str(k): self.scrub_payload(v) for k, v in payload.items()}
        if isinstance(payload, list):
            return [self.scrub_payload(item) for item in payload]
        if isinstance(payload, tuple):
            return tuple(self.scrub_payload(item) for item in payload)
        return payload


_DEFAULT_SCRUBBER = PiiRedactionScrubber()


def scrub_pii(value: Any) -> Any:
    """Convenience function to redact PII from strings or structured collections."""
    return _DEFAULT_SCRUBBER.scrub_payload(value)


if OTEL_AVAILABLE:

    class _InMemorySpanExporter(SpanExporter):
        """Captures completed OpenTelemetry spans in memory for local inspection and tests."""

        def export(self, spans: Any) -> Any:
            for span in spans:
                ctx = span.get_span_context()
                parent_id = (
                    f"{span.parent.span_id:016x}" if getattr(span, "parent", None) else None
                )
                _RECORDED_SPANS_BUFFER.append(
                    {
                        "name": span.name,
                        "trace_id": f"{ctx.trace_id:032x}",
                        "span_id": f"{ctx.span_id:016x}",
                        "parent_span_id": parent_id,
                        "attributes": dict(span.attributes or {}),
                        "status": span.status.status_code.name if span.status else "UNSET",
                    }
                )
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            return None


def _configure_tracer() -> Any:
    """Initialize OpenTelemetry TracerProvider and SimpleSpanProcessor once per process."""
    if not OTEL_AVAILABLE:
        return None
    provider = trace.get_tracer_provider()
    if not isinstance(provider, TracerProvider):
        resource = Resource.create({"service.name": "nihongo-studyops-agent"})
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(_InMemorySpanExporter()))
        trace.set_tracer_provider(provider)
    elif not getattr(provider, "_nihongo_exporter_attached", False):
        provider.add_span_processor(SimpleSpanProcessor(_InMemorySpanExporter()))
        setattr(provider, "_nihongo_exporter_attached", True)
    return trace.get_tracer("nihongo_agent.tracer")


_TRACER = _configure_tracer()


@contextlib.contextmanager
def trace_span(
    span_name: str,
    attributes: Optional[dict[str, Any]] = None,
) -> Iterator[dict[str, Any]]:
    """Create a distributed trace span and propagate ``trace_id`` / ``span_id`` to logs."""
    safe_attributes = scrub_pii(attributes or {})
    if _TRACER is not None:
        with _TRACER.start_as_current_span(span_name) as span:
            for key, val in safe_attributes.items():
                if isinstance(val, (str, bool, int, float)):
                    span.set_attribute(key, val)
                else:
                    span.set_attribute(key, json.dumps(val, ensure_ascii=False))
            ctx = span.get_span_context()
            trace_id_hex = f"{ctx.trace_id:032x}"
            span_id_hex = f"{ctx.span_id:016x}"
            token_trace = _CURRENT_TRACE_ID.set(trace_id_hex)
            token_span = _CURRENT_SPAN_ID.set(span_id_hex)
            span_meta = {
                "name": span_name,
                "trace_id": trace_id_hex,
                "span_id": span_id_hex,
                "attributes": safe_attributes,
            }
            try:
                yield span_meta
            finally:
                _CURRENT_TRACE_ID.reset(token_trace)
                _CURRENT_SPAN_ID.reset(token_span)
    else:  # pragma: no cover - fallback if opentelemetry is unavailable
        trace_id_hex = _CURRENT_TRACE_ID.get() or uuid.uuid4().hex
        span_id_hex = uuid.uuid4().hex[:16]
        parent_id = _CURRENT_SPAN_ID.get() or None
        token_trace = _CURRENT_TRACE_ID.set(trace_id_hex)
        token_span = _CURRENT_SPAN_ID.set(span_id_hex)
        span_meta = {
            "name": span_name,
            "trace_id": trace_id_hex,
            "span_id": span_id_hex,
            "parent_span_id": parent_id,
            "attributes": safe_attributes,
        }
        try:
            yield span_meta
        finally:
            _RECORDED_SPANS_BUFFER.append(span_meta)
            _CURRENT_TRACE_ID.reset(token_trace)
            _CURRENT_SPAN_ID.reset(token_span)


class StructuredJsonFormatter(logging.Formatter):
    """Formats Python ``LogRecord`` instances into PII-scrubbed single-line JSON strings."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "severity": record.levelname,
            "logger": record.name,
            "message": _DEFAULT_SCRUBBER.scrub_text(record.getMessage()),
            "trace_id": getattr(record, "trace_id", None) or _CURRENT_TRACE_ID.get() or None,
            "span_id": getattr(record, "span_id", None) or _CURRENT_SPAN_ID.get() or None,
        }
        for extra_field in (
            "phase",
            "action_name",
            "agent_name",
            "status",
            "latency_ms",
            "metadata",
        ):
            value = getattr(record, extra_field, None)
            if value is not None:
                payload[extra_field] = scrub_pii(value)

        _STRUCTURED_LOG_BUFFER.append(payload)
        return json.dumps(payload, ensure_ascii=False)


def get_structured_logger(name: str = "nihongo_agent") -> logging.Logger:
    """Return a configured ``logging.Logger`` using ``StructuredJsonFormatter``."""
    logger = logging.getLogger(name)
    if not getattr(logger, "_nihongo_structured_configured", False):
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())
        logger.handlers = [handler]
        logger.propagate = False
        setattr(logger, "_nihongo_structured_configured", True)
    return logger


_LOGGER = get_structured_logger()


def log_agent_intent(
    action_name: str,
    intent_payload: dict[str, Any],
    agent_name: str = "nihongo_studyops_agent",
) -> dict[str, Any]:
    """Emit a structured JSON log entry with ``phase="INTENT"`` prior to tool/agent execution."""
    scrubbed_metadata = scrub_pii(intent_payload)
    _LOGGER.info(
        f"Agent intent recorded for {action_name}",
        extra={
            "phase": "INTENT",
            "action_name": action_name,
            "agent_name": agent_name,
            "status": "PLANNED",
            "metadata": scrubbed_metadata,
        },
    )
    return _STRUCTURED_LOG_BUFFER[-1]


def log_agent_outcome(
    action_name: str,
    outcome_payload: dict[str, Any],
    status: str = "SUCCESS",
    latency_ms: Optional[float] = None,
    agent_name: str = "nihongo_studyops_agent",
) -> dict[str, Any]:
    """Emit a structured JSON log entry with ``phase="OUTCOME"`` after tool/agent execution."""
    scrubbed_metadata = scrub_pii(outcome_payload)
    _LOGGER.info(
        f"Agent outcome recorded for {action_name}: {status}",
        extra={
            "phase": "OUTCOME",
            "action_name": action_name,
            "agent_name": agent_name,
            "status": status,
            "latency_ms": round(latency_ms, 3) if latency_ms is not None else 0.0,
            "metadata": scrubbed_metadata,
        },
    )
    return _STRUCTURED_LOG_BUFFER[-1]


def observe_tool_execution(
    tool_name: Optional[str] = None,
    agent_name: str = "nihongo_studyops_agent",
) -> Callable[[F], F]:
    """Decorator wrapping a tool in an OpenTelemetry span and ``INTENT``/``OUTCOME`` JSON logs."""

    def decorator(func: F) -> F:
        resolved_name = tool_name or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            call_Summary = {
                "args": [
                    arg for arg in args if isinstance(arg, (str, int, float, bool, list, dict))
                ],
                "kwargs": {
                    k: v
                    for k, v in kwargs.items()
                    if k != "tool_context"
                    and isinstance(v, (str, int, float, bool, list, dict))
                },
            }
            start_ns = time.perf_counter_ns()
            with trace_span(f"tool.{resolved_name}", attributes={"tool.name": resolved_name}):
                log_agent_intent(
                    action_name=resolved_name,
                    intent_payload=call_Summary,
                    agent_name=agent_name,
                )
                try:
                    result = func(*args, **kwargs)
                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                    result_status = (
                        result.get("status", "SUCCESS")
                        if isinstance(result, dict)
                        else "SUCCESS"
                    )
                    log_agent_outcome(
                        action_name=resolved_name,
                        outcome_payload=result if isinstance(result, dict) else {"result": str(result)},
                        status=str(result_status),
                        latency_ms=elapsed_ms,
                        agent_name=agent_name,
                    )
                    return result
                except Exception as exc:
                    elapsed_ms = (time.perf_counter_ns() - start_ns) / 1_000_000.0
                    log_agent_outcome(
                        action_name=resolved_name,
                        outcome_payload={"error": str(exc)},
                        status="ERROR",
                        latency_ms=elapsed_ms,
                        agent_name=agent_name,
                    )
                    raise

        return wrapper  # type: ignore[return-value]

    return decorator


def get_structured_log_entries() -> list[dict[str, Any]]:
    """Return a copy of all structured JSON log entries emitted in this process."""
    return list(_STRUCTURED_LOG_BUFFER)


def clear_structured_log_entries() -> None:
    """Clear the in-memory structured log buffer (used between isolated test cases)."""
    _STRUCTURED_LOG_BUFFER.clear()


def get_recorded_spans() -> list[dict[str, Any]]:
    """Return a copy of all OpenTelemetry spans recorded in this process."""
    return list(_RECORDED_SPANS_BUFFER)


def clear_recorded_spans() -> None:
    """Clear the in-memory recorded OpenTelemetry spans buffer."""
    _RECORDED_SPANS_BUFFER.clear()
