from __future__ import annotations

import json
import os
from typing import Any

import pandas as pd


AWS_REGION = os.environ.get("AWS_REGION", "us-west-2")
AWS_PROFILE_NAME = os.environ.get("AWS_PROFILE")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "google.gemma-3-4b-it")

AI_SCOUTING_MODEL_LABEL = BEDROCK_MODEL_ID
AI_MAX_INPUT_CHARS = 36_000
AI_MAX_OUTPUT_TOKENS = 800


def _clean_value(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def compact_records(df: pd.DataFrame, *, columns: list[str] | None = None, max_rows: int = 8) -> list[dict[str, Any]]:
    if df.empty:
        return []
    out = df.copy()
    if columns:
        keep = [col for col in columns if col in out.columns]
        out = out[keep]
    out = out.head(max_rows)
    records = []
    for row in out.to_dict("records"):
        records.append({str(key): _clean_value(value) for key, value in row.items()})
    return records


def compact_frame(df: pd.DataFrame, *, columns: list[str] | None = None, max_rows: int = 8) -> str:
    return json.dumps(compact_records(df, columns=columns, max_rows=max_rows), default=str, separators=(",", ":"))


def trim_prompt(text: str, max_chars: int = AI_MAX_INPUT_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    return (
        text[:max_chars]
        + "\n\n[INPUT TRIMMED BY APP TOKEN BUDGET: use the all-row summaries above as source of truth; "
        + "pitch-level logs may be partial.]"
    )


def converse(client: Any, model_id: str, messages: list[dict[str, Any]], max_tokens: int | None = 500) -> str:
    inference_config = {"temperature": 0}
    if max_tokens is not None:
        inference_config["maxTokens"] = max_tokens

    response = client.converse(
        modelId=model_id,
        messages=messages,
        inferenceConfig=inference_config,
    )
    return response["output"]["message"]["content"][0]["text"].strip()


def generate_scouting_brief(prompt: str, *, max_tokens: int | None = AI_MAX_OUTPUT_TOKENS) -> str:
    """Call AWS Bedrock with a fast, practical scouting budget."""
    import boto3

    if AWS_PROFILE_NAME:
        session = boto3.Session(profile_name=AWS_PROFILE_NAME, region_name=AWS_REGION)
    else:
        session = boto3.Session(region_name=AWS_REGION)
    client = session.client("bedrock-runtime")
    messages = [{"role": "user", "content": [{"text": trim_prompt(prompt)}]}]
    return converse(client=client, model_id=BEDROCK_MODEL_ID, messages=messages, max_tokens=max_tokens)
