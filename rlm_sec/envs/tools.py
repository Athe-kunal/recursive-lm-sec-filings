import json
import logging
import threading
import time
import uuid
from typing import Any, Dict, Optional, Tuple
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from finance_data.filings.utils import company_to_ticker
from skyrl_gym.tools.core import tool, ToolGroup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TIMEOUT = 30
MAX_RETRIES = 10
INITIAL_RETRY_DELAY = 1


def _build_search_payload(
    query: str,
    ticker: str,
    year: str,
    filing_type: str,
    topk: int,
) -> Dict[str, Any]:
    """Build the request body expected by ``/vector_store/search_sec_filings``."""
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
    session: Optional[requests.Session] = None,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Calls a search-style POST API for one filing or transcript.

    Args:
        retrieval_service_url: The URL of the search API endpoint.
        query: Semantic search query text.
        ticker: Stock symbol for the filing.
        year: Filing year.
        filing_type: SEC filing type (e.g. 10-K, 10-Q1).
        topk: Number of chunks to return (sent as top_k in the JSON body).
        timeout: Request timeout in seconds.
        log_requests: Whether to log requests.
        session: Optional shared requests.Session.

    Returns:
        Parsed JSON body on success, or None on failure.
        error_msg: Error message if the request failed.
    """
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

    # Use provided session or create a new one for this request
    if session is None:
        session = requests.Session()
        should_close_session = True
    else:
        should_close_session = False

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            if log_requests:
                logger.info(
                    f"{log_prefix}Attempt {attempt + 1}/{MAX_RETRIES}: Calling search API at {retrieval_service_url}"
                )
            response = session.post(
                retrieval_service_url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )

            # Check for Gateway Timeout (504) and other server errors for retrying
            if response.status_code in [500, 502, 503, 504]:
                last_error = f"{log_prefix}API Request Error: Server Error ({response.status_code}) on attempt {attempt + 1}/{MAX_RETRIES}"
                logger.warning(last_error)
                if attempt < MAX_RETRIES - 1:
                    delay = INITIAL_RETRY_DELAY * (attempt + 1)
                    logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                    time.sleep(delay)
                continue

            # Check for other HTTP errors (e.g., 4xx)
            response.raise_for_status()

            # If successful (status code 2xx)
            if log_requests:
                logger.info(
                    f"{log_prefix}Search API call successful on attempt {attempt + 1}"
                )

            # Close session if we created it
            if should_close_session:
                session.close()

            return response.json(), None

        except requests.exceptions.ConnectionError as e:
            last_error = f"{log_prefix}Connection Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.Timeout as e:
            last_error = f"{log_prefix}Timeout Error: {e}"
            logger.warning(last_error)
            if attempt < MAX_RETRIES - 1:
                delay = INITIAL_RETRY_DELAY * (attempt + 1)
                logger.info(f"{log_prefix}Retrying after {delay} seconds...")
                time.sleep(delay)
            continue
        except requests.exceptions.RequestException as e:
            last_error = f"{log_prefix}API Request Error: {e}"
            break  # Exit retry loop on other request errors
        except json.JSONDecodeError as e:
            raw_response_text = response.text if "response" in locals() else "N/A"
            last_error = f"{log_prefix}API Response JSON Decode Error: {e}, Response: {raw_response_text[:200]}"
            break  # Exit retry loop on JSON decode errors
        except Exception as e:
            last_error = f"{log_prefix}Unexpected Error: {e}"
            break  # Exit retry loop on other unexpected errors

    # If we reach here, all attempts failed
    logger.error(
        f"{log_prefix}API Request Failed after {MAX_RETRIES} attempts: {last_error}"
    )

    # Close session if we created it
    if should_close_session:
        session.close()

    return None, last_error


def _vector_chunks_to_string(chunks: list[dict[str, Any]]) -> str:
    """Format vector search API chunk list (ChunkResult JSON) for the model."""
    lines: list[str] = []
    for idx, chunk in enumerate(chunks):
        text = chunk.get("text", "").strip()
        lines.append(f"Doc {idx + 1}: {text}\n")
    return "".join(lines)


class SearchToolGroup(ToolGroup):
    # Class-level session pool shared across all instances
    _session_pool: dict[str, requests.Session] = {}
    _session_lock = threading.Lock()

    @classmethod
    def _get_shared_session(cls, base_url: str) -> requests.Session:
        """Get or create a shared session for the given base URL"""
        with cls._session_lock:
            if base_url not in cls._session_pool:
                session = requests.Session()
                # Configure connection pooling
                adapter = HTTPAdapter(
                    pool_connections=20,  # Number of connection pools
                    pool_maxsize=20,  # Max connections per pool
                    max_retries=0,  # We handle retries ourselves
                    pool_block=False,  # Don't block if pool is full
                )
                session.mount("http://", adapter)
                session.mount("https://", adapter)
                cls._session_pool[base_url] = session
                logger.info(f"Created shared session pool for {base_url}")
            return cls._session_pool[base_url]

    def __init__(
        self,
        search_url: str,
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
        group_name: str = "SearchToolGroup",
    ):
        self.search_url = search_url
        self.topk = topk
        self.timeout = timeout
        self.log_requests = log_requests
        self.group_name = group_name
        self.last_metadata: Dict[str, Any] = {}

        # Extract base URL for session sharing
        parsed_url = urlparse(self.search_url)
        self.base_url = f"{parsed_url.scheme}://{parsed_url.netloc}"

        # Get shared session for this base URL
        self.session = self._get_shared_session(self.base_url)
        if self.log_requests:
            logger.info(
                "%s initialized using shared session pool for %s",
                self.group_name,
                self.base_url,
            )

        super().__init__(name=self.group_name)

    def _build_metadata(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
        error_msg: Optional[str],
    ) -> Dict[str, Any]:
        """Build the base metadata dictionary for a tool invocation."""
        return {
            "query": query,
            "ticker": ticker,
            "year": year,
            "filing_type": filing_type,
            "api_request_error": error_msg,
            "api_response": None,
            "status": "unknown",
            "total_results": 0,
            "formatted_result": None,
        }

    def _format_api_response(
        self,
        api_response: Any,
        metadata: Dict[str, Any],
    ) -> str:
        """Format an API response and update the metadata dictionary."""
        if isinstance(api_response, list) and api_response:
            result_text = _vector_chunks_to_string(api_response)
            metadata["status"] = "success"
            metadata["total_results"] = len(api_response)
            metadata["formatted_result"] = result_text
            if self.log_requests:
                logger.info(
                    "Batch search: Successful, got %s chunks", len(api_response)
                )
            return result_text
        if isinstance(api_response, list):
            metadata["status"] = "no_results"
            metadata["total_results"] = 0
            if self.log_requests:
                logger.info("Batch search: No results found")
            return "No search results found."
        metadata["status"] = "processing_error"
        return metadata["formatted_result"]

    def _call_search(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        if query is None or ticker is None or year is None or filing_type is None:
            return ""

        query = query.strip()
        ticker = ticker.strip()
        year = year.strip()
        filing_type = filing_type.strip()

        api_response = None
        error_msg = None
        try:
            api_response, error_msg = call_search_api(
                retrieval_service_url=self.search_url,
                query=query,
                ticker=ticker,
                year=year,
                filing_type=filing_type,
                topk=self.topk,
                timeout=self.timeout,
                log_requests=self.log_requests,
                session=self.session,
            )
        except Exception as e:
            error_msg = f"API Request Exception during batch search: {e}"
            logger.error(f"Batch search: {error_msg}")

        metadata = self._build_metadata(query, ticker, year, filing_type, error_msg)
        result_text = "Search request failed or timed out after retries."

        if error_msg:
            metadata["status"] = "api_error"
            result_text = f"Search error: {error_msg}"
            logger.error(f"Batch search: API error occurred: {error_msg}")
        elif api_response is not None:
            logger.debug(f"Batch search: API Response: {api_response}")
            metadata["api_response"] = api_response

            try:
                result_text = self._format_api_response(api_response, metadata)
            except Exception as e:
                error_msg = f"Error processing search results: {e}"
                result_text = error_msg
                metadata["api_request_error"] = error_msg
                metadata["status"] = "processing_error"
                logger.error(f"Batch search: {error_msg}")
        else:
            metadata["status"] = "unknown_api_state"
            result_text = "Unknown API state (no response and no error message)."
            logger.error("Batch search: Unknown API state.")

        self.last_metadata = metadata
        return result_text

    def get_last_metadata(self) -> Dict[str, Any]:
        """Return the metadata from the most recent tool invocation."""
        return self.last_metadata

    @tool
    def search(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        return self._call_search(query, ticker, year, filing_type)


class SECFilingToolGroup(SearchToolGroup):
    """HTTP-backed SEC filing tool group."""

    def __init__(
        self,
        tool_url: str,
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
    ):
        super().__init__(
            search_url=tool_url,
            topk=topk,
            timeout=timeout,
            log_requests=log_requests,
            group_name="SECFilingToolGroup",
        )

    @tool
    def sec_filing_to_markdown_embed_and_search(
        self,
        query: str,
        ticker: str,
        year: str,
        filing_type: str,
    ) -> str:
        """Call the SEC filing tool server endpoint."""
        return self._call_search(query, ticker, year, filing_type)


class EarningsTranscriptToolGroup(SearchToolGroup):
    """HTTP-backed earnings transcript tool group."""

    def __init__(
        self,
        tool_url: str,
        topk: int = 3,
        timeout: int = DEFAULT_TIMEOUT,
        log_requests: bool = True,
    ):
        super().__init__(
            search_url=tool_url,
            topk=topk,
            timeout=timeout,
            log_requests=log_requests,
            group_name="EarningsTranscriptToolGroup",
        )

    @tool
    def earnings_transcript_to_embed_and_search(
        self,
        query: str,
        ticker: str,
        year: str,
        quarter: str,
    ) -> str:
        """Call the earnings transcript tool server endpoint."""
        return self._call_search(query, ticker, year, quarter)


class CompanyNameToTickerToolGroup(ToolGroup):
    """Resolves a company name to its stock ticker symbol."""

    def __init__(self):
        super().__init__(name="CompanyNameToTickerToolGroup")

    @tool
    def company_name_to_ticker_tool(self, name: str) -> str:
        """Resolve a full or partial company name to its stock ticker symbol.

        Use this when the question provides a company name instead of a ticker.
        Skip this when a valid ticker is already known.
        """
        ticker = company_to_ticker(name)
        if ticker is None:
            raise ValueError(f"No ticker found for company name: {name!r}")
        logger.info(f"{name=} resolved to {ticker=}")
        return ticker
