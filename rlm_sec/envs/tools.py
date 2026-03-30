"""HTTP-backed finance retrieval tools for Verifiers environments."""

from __future__ import annotations

import asyncio
import json
from loguru import logger
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import requests
from finance_data.filings.utils import company_to_ticker
from requests.adapters import HTTPAdapter

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


def _build_search_payload(
    query: str,
    ticker: str,
    year: str,
    filing_type: str,
    topk: int,
) -> dict[str, Any]:
    """Build the request body expected by the finance search server."""
    return {
        "ticker": ticker,
        "year": year,
        "filing_type": filing_type,
        "query": query,
        "top_k": topk,
    }


def call_search_api(
    retrieval_service_url: str,
    query: str,
    ticker: str,
    year: str,
    filing_type: str,
    topk: int = 3,
    timeout: int = DEFAULT_TIMEOUT,
    log_requests: bool = True,
    session: requests.Session | None = None,
) -> tuple[Any | None, str | None]:
    """Call the search API with retries and return parsed JSON."""
    request_id = str(uuid.uuid4())
    log_prefix = f"[Search Request ID: {request_id}] "

    payload = _build_search_payload(
        query=query,
        ticker=ticker,
        year=year,
        filing_type=filing_type,
        topk=topk,
    )
    headers = {"Content-Type": "application/json", "Accept": "application/json"}

    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    response: requests.Response | None = None
    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    "%sAttempt %s/%s: Calling search API at %s",
                    log_prefix,
                    attempt + 1,
                    MAX_RETRIES,
                    retrieval_service_url,
                )
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            if response.status_code in [500, 502, 503, 504]:
                last_error = (
                    f"{log_prefix}API Request Error: Server Error "
                    f"({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                )
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info("%sRetrying after %s seconds...", log_prefix, delay)
                    time.sleep(delay)
                continue

            response.raise_for_status()

            if log_requests:
                logger.info(
                    "%sSearch API call successful on attempt %s",
                    log_prefix,
                    attempt + 1,
                )

            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as error:
            last_error = f"{log_prefix}Connection Error: {error}"
        except requests.exceptions.Timeout as error:
            last_error = f"{log_prefix}Timeout Error: {error}"
        except requests.exceptions.RequestException as error:
            last_error = f"{log_prefix}API Request Error: {error}"
            break
        except json.JSONDecodeError as error:
            raw_response_text = response.text if response is not None else "N/A"
            last_error = (
                f"{log_prefix}API Response JSON Decode Error: {error}, "
                f"Response: {raw_response_text[:200]}"
            )
            break
        except Exception as error:  # pylint: disable=broad-except
            last_error = f"{log_prefix}Unexpected Error: {error}"
            break

        logger.warning(last_error)
        if attempt < MAX_RETRIES - 1:
            delay = INITIAL_RETRY_DELAY * (attempt + 1)
            logger.info("%sRetrying after %s seconds...", log_prefix, delay)
            time.sleep(delay)

    logger.error(
        "%sAPI Request Failed after %s attempts: %s",
        log_prefix,
        MAX_RETRIES,
        last_error,
    )

    if should_close_session:
        session.close()

    return None, last_error


def _vector_chunks_to_string(chunks: list[dict[str, Any]]) -> str:
    """Format vector search results for the model context."""
    lines: list[str] = []
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        lines.append(f"Doc {idx + 1}: {text}\n")
    return "".join(lines)


@dataclass
class SearchResult:
    """Structured search result and metadata payload."""

    text: str
    metadata: dict[str, Any]


class SearchClient:
    """Shared HTTP client for SEC filings and transcript search tools."""

    _session_pool: dict[str, requests.Session] = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        """Get or create a shared session for a base URL."""
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                adapter = HTTPAdapter(
                    pool_connections=20,
                    pool_maxsize=20,
                    max_retries=0,
                    pool_block=False,
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
            return cls._session_pool[base_url]

    def __init__(
        self,
        search_url: str,
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = False,
    ):
        self._search_url = search_url
        self._topk = topk
        self._timeout = timeout
        self._log_requests = log_requests

        parsed_url = urlparse(search_url)
        base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"
        self._session = self._get_shared_session(base_url)

    def search(
        self, query: str, ticker: str, year: str, filing_type: str
    ) -> SearchResult:
        """Perform a search request and return result text and metadata."""
        if not query or not ticker or not year or not filing_type:
            return SearchResult(text="", metadata={"status": "invalid_input"})

        api_response, error_msg = call_search_api(
            retrieval_service_url=self._search_url,
            query=query.strip(),
            ticker=ticker.strip(),
            year=year.strip(),
            filing_type=filing_type.strip(),
            topk=self._topk,
            timeout=self._timeout,
            log_requests=self._log_requests,
            session=self._session,
        )

        metadata: dict[str, Any] = {
            "query": query,
            "ticker": ticker,
            "year": year,
            "filing_type": filing_type,
            "api_request_error": error_msg,
            "api_response": api_response,
            "status": "unknown",
        }

        if error_msg is not None:
            metadata["status"] = "api_error"
            return SearchResult(text=f"Search error: {error_msg}", metadata=metadata)

        if isinstance(api_response, list) and api_response:
            metadata["status"] = "success"
            metadata["total_results"] = len(api_response)
            return SearchResult(
                text=_vector_chunks_to_string(api_response),
                metadata=metadata,
            )

        if isinstance(api_response, list):
            metadata["status"] = "no_results"
            metadata["total_results"] = 0
            return SearchResult(text="No search results found.", metadata=metadata)

        metadata["status"] = "invalid_response"
        return SearchResult(text="Unknown API state.", metadata=metadata)


class FinanceSearchTools:
    """Async tool surface consumed by the Prime RL Verifiers environment."""

    def __init__(
        self,
        sec_filing_tool_url: str,
        earnings_transcript_tool_url: str,
        topk: int,
        timeout: int,
        log_requests: bool,
    ):
        self._sec_filing_client = SearchClient(
            search_url=sec_filing_tool_url,
            topk=topk,
            timeout=timeout,
            log_requests=log_requests,
        )
        self._earnings_transcript_client = SearchClient(
            search_url=earnings_transcript_tool_url,
            topk=topk,
            timeout=timeout,
            log_requests=log_requests,
        )

    async def sec_filing(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        """Search SEC filings by query, ticker, year, and filing type."""
        result = await asyncio.to_thread(
            self._sec_filing_client.search,
            query,
            ticker,
            year,
            filing_type,
        )
        return result.text

    async def earnings_transcript(
        self,
        query: str,
        ticker: str,
        year: str,
        quarter: str,
    ) -> str:
        """Search earnings transcripts by query, ticker, year, and quarter."""
        result = await asyncio.to_thread(
            self._earnings_transcript_client.search,
            query,
            ticker,
            year,
            quarter,
        )
        return result.text

    async def company_name_to_ticker(self, name: str) -> str:
        """Resolve a company name to a stock ticker."""
        ticker = await asyncio.to_thread(company_to_ticker, name)
        if ticker is None:
            return f"No ticker found for company name: {name!r}"
        return ticker
