"""Clients for the two models the harness talks to.

The distinction matters for the experimental design and is therefore encoded in
the type system rather than left implicit:

* :class:`GeneratorClient` wraps the *policy model under evaluation*. Its
  sampling temperature and top-p are what the controller searches over.
* :class:`CorrectorClient` wraps a *frozen auxiliary model* that is part of the
  fixed harness. It repairs broken programs and summarises error histories at a
  fixed low temperature, identically for every configuration, so that it cannot
  confound a comparison between policy models.

Neither client holds a credential of its own: both read them from the config
object, which in turn reads them from the environment.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import requests
from openai import OpenAI

from .config import CorrectorConfig, GeneratorConfig
from .logging_utils import get_logger
from .tokens import count_chat_tokens, estimate_completion_tokens, estimate_tokens_by_chars

LOGGER = get_logger(__name__)


class GeneratorClient:
    """The policy model, served behind an OpenAI-compatible endpoint."""

    def __init__(self, config: GeneratorConfig) -> None:
        self.config = config
        self._client = OpenAI(api_key=config.api_key, base_url=config.base_url)

    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float,
        top_p: float,
    ) -> tuple[str, float]:
        """Sample one completion.

        Returns:
            ``(response_text, total_tokens)`` where the token figure is the sum
            of the prompt count and an estimate of the completion count.
        """
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        prompt_tokens = count_chat_tokens(self.config.model, messages)

        response = self._client.chat.completions.create(
            model=self.config.model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            max_tokens=self.config.max_tokens,
        )
        text = response.choices[0].message.content or ""

        completion_tokens = estimate_completion_tokens(text)
        total_tokens = prompt_tokens + completion_tokens
        LOGGER.info(
            "Generator call: ~%.0f prompt + ~%.0f completion = ~%.0f tokens",
            prompt_tokens,
            completion_tokens,
            total_tokens,
        )
        return text, total_tokens


class CorrectorClient:
    """The frozen auxiliary model used for repair and error summarisation."""

    def __init__(self, config: CorrectorConfig) -> None:
        self.config = config

    def _headers(self) -> Dict[str, str]:
        if not self.config.api_key:
            raise ValueError(
                "No corrector API key configured. Set CORRECTOR_API_KEY in the "
                "environment (see .env.example); never hard-code a credential."
            )
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.config.api_key}",
            "HTTP-Referer": self.config.referer,
            "X-Title": self.config.title,
        }

    def complete(
        self,
        prompt: str,
        system_message: str,
        mode: str = "analysis",
    ) -> tuple[str, float]:
        """Call the auxiliary model.

        Args:
            prompt: The user message.
            system_message: The system message.
            mode: ``"correction"`` for program repair, which needs a large
                output budget and a near-deterministic temperature;
                ``"analysis"`` for short error summaries.

        Returns:
            ``(response_text, total_tokens)``. On a transport or API error the
            error text is returned in place of the response, matching the
            original behaviour: the caller falls back to the previous program
            rather than aborting the episode.
        """
        if mode == "correction":
            max_tokens = self.config.correction_max_tokens
            temperature = self.config.correction_temperature
        else:
            max_tokens = self.config.analysis_max_tokens
            temperature = self.config.analysis_temperature

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": prompt},
        ]
        prompt_tokens = estimate_tokens_by_chars(messages)

        payload: Dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            response = requests.post(
                self.config.api_url, headers=self._headers(), json=payload, timeout=300
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"Corrector API returned {response.status_code}: {response.text[:500]}"
                )

            text = response.json()["choices"][0]["message"]["content"]
            completion_tokens = estimate_completion_tokens(text)
            total_tokens = prompt_tokens + completion_tokens
            LOGGER.info(
                "Corrector call (%s): ~%.0f prompt + ~%.0f completion = ~%.0f tokens",
                mode,
                prompt_tokens,
                completion_tokens,
                total_tokens,
            )
            return text, total_tokens

        except Exception as exc:
            LOGGER.error("Corrector API call failed: %s", exc)
            return f"Corrector API call failed: {exc}", 0.0
