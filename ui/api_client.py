"""
Unified API Client — EC-VeriQuery Frontend

Central HTTP communication layer between the Streamlit frontend and the
FastAPI backend.  All UI pages interact with the backend exclusively through
the functions defined here.

Key design decisions:
    - Connection pooling via @st.cache_resource Session singleton — TCP
      connections survive Streamlit script re-executions.
    - Health-check with 30 s TTL cache — avoids redundant /health requests
      on every user interaction.
    - Document list with @st.cache_data(ttl=30) — lightweight read cache.
    - SSE (Server-Sent Events) streaming for the chat endpoint — single HTTP
      request, incremental response delivery.

Consumed by:
    sidebar_nav.py, 1_Documents.py, 2_Chat.py, 3_Pinout.py,
    4_ERC.py, 5_Compare.py, 6_Circuit.py
"""
import json
import logging
import os
import time
import urllib.parse

import requests
import requests.adapters
import streamlit as st

logger = logging.getLogger(__name__)

API_URL = os.environ.get("API_URL", "http://localhost:8000")
API_HEALTH_TIMEOUT = 10
API_CHECK_TTL = 30


@st.cache_resource
def _get_session() -> requests.Session:
    """Return a global requests.Session with connection pooling (server-level singleton)."""
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(
        pool_connections=2,
        pool_maxsize=8,
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session


def check_api_connection(force_refresh: bool = False) -> bool:
    """Check backend reachability with a 30 s TTL cache.

    Uses time.monotonic() (not time.time()) so NTP adjustments cannot
    invalidate the TTL calculation.
    """
    now = time.monotonic()
    last_ts = st.session_state.get("api_check_ts", 0)

    if not force_refresh and now - last_ts < API_CHECK_TTL and "api_connected" in st.session_state:
        return bool(st.session_state.api_connected)

    session = _get_session()
    try:
        response = session.get(f"{API_URL}/health", timeout=API_HEALTH_TIMEOUT)
        if response.status_code == 200:
            st.session_state.api_url = API_URL
            st.session_state.api_connected = True
            st.session_state.api_check_ts = now
            return True
    except Exception:
        pass

    st.session_state.api_url = API_URL
    st.session_state.api_connected = False
    st.session_state.api_check_ts = now
    return False


def is_api_connected() -> bool:
    """Fast read of cached connection state (no HTTP request)."""
    return bool(st.session_state.get("api_connected", False))


def get_api_url() -> str:
    """Return the currently active backend URL."""
    return st.session_state.get("api_url", API_URL)


@st.cache_data(ttl=30)
def _fetch_documents_cached(api_url: str) -> list:
    """Fetch document list with a 30 s Streamlit data cache."""
    try:
        response = _get_session().get(f"{api_url}/api/v1/documents", timeout=30)
        if response.status_code == 200:
            data = response.json()
            return data if isinstance(data, list) else data.get("documents", [])
    except Exception:
        pass
    return []


def get_documents(use_cache: bool = True) -> list:
    """Return the list of uploaded documents.

    Args:
        use_cache: When False, clear the cache and re-fetch from the backend.
    """
    if not use_cache:
        _fetch_documents_cached.clear()
    return _fetch_documents_cached(get_api_url())


def upload_document(file, filename: str = None) -> dict:
    """Upload a PDF document to the backend.

    Args:
        file: File-like object (from st.file_uploader).
        filename: Optional override for the upload filename.
    """
    if not is_api_connected():
        return {"success": False, "error": "API未连接"}

    try:
        files = {"file": (filename or file.name, file, "application/pdf")}
        response = _get_session().post(
            f"{get_api_url()}/api/v1/documents/upload",
            files=files,
            timeout=300,
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": f"HTTP {response.status_code}: {response.text[:200]}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def delete_document(document_id: str) -> dict:
    """Delete a document by its ID."""
    if not is_api_connected():
        return {"success": False, "error": "API未连接"}

    try:
        response = _get_session().delete(
            f"{get_api_url()}/api/v1/documents/{document_id}",
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_document_status(document_id: str) -> dict:
    """Query the processing status of a single document (lightweight poll)."""
    if not is_api_connected():
        return {"success": False, "error": "API未连接"}

    try:
        response = _get_session().get(
            f"{get_api_url()}/api/v1/documents/{document_id}",
            timeout=30,
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_devices(query: str) -> dict:
    """Search devices by semantic query (POST for structured filter payload)."""
    if not is_api_connected():
        return {"devices": [], "success": False, "error": "API未连接"}

    try:
        response = _get_session().post(
            f"{get_api_url()}/api/v1/documents/search",
            json={"query": query, "filters": {}},
            timeout=30,
        )
        if response.status_code == 200:
            data = response.json()
            if data.get("success"):
                return {"devices": data.get("devices", []), "success": True}
            return {"devices": [], "success": False, "error": data.get("message", "搜索失败")}
        return {"devices": [], "success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"devices": [], "success": False, "error": str(e)}


def erc_check(driver_chip=None, receiver_chip=None, document_ids=None, temperature=None):
    """Run an Electrical Rule Compatibility check between two chips.

    Args:
        driver_chip: Driver device name (e.g. "SN74HC04").
        receiver_chip: Receiver device name (e.g. "SN74HCT04").
        document_ids: Optional document IDs to scope parameter extraction.
        temperature: Operating temperature in °C (0 is valid, hence `is not None`).
    """
    if not is_api_connected():
        return {"success": False, "error": "API未连接"}

    try:
        payload = {}
        if driver_chip:
            payload["driver_chip"] = driver_chip
        if receiver_chip:
            payload["receiver_chip"] = receiver_chip
        if document_ids:
            payload["document_ids"] = [
                str(d) for d in (document_ids if isinstance(document_ids, list) else [document_ids])
            ]
        if temperature is not None:
            payload["temperature"] = temperature

        response = _get_session().post(
            f"{get_api_url()}/api/v1/erc/check",
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_pinout(chip_name: str, document_ids=None, package=None):
    """Analyze chip pinout and generate an SVG visualization.

    Args:
        chip_name: Device name (e.g. "SN74HC04").
        document_ids: Optional document IDs (str or list accepted).
        package: Optional package type (e.g. "DIP-14").
    """
    if not is_api_connected():
        return {"success": False, "error": "API未连接"}

    try:
        payload = {"chip_name": chip_name}
        if document_ids:
            payload["document_ids"] = [
                str(d) for d in (document_ids if isinstance(document_ids, list) else [document_ids])
            ]
        if package:
            payload["package"] = package

        response = _get_session().post(
            f"{get_api_url()}/api/v1/pinout/",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def compare_devices_enhanced(document_ids, device_names=None):
    """Multi-device parameter comparison with three-layer scoring.

    Args:
        document_ids: Document IDs specifying the chips to compare.
        device_names: Optional device names for targeted comparison.
    """
    if not is_api_connected():
        return {"success": False, "error": "API未连接"}

    try:
        payload = {
            "devices": device_names if device_names else [],
            "document_ids": document_ids,
        }
        response = _get_session().post(
            f"{get_api_url()}/api/v1/compare/devices-enhanced",
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()
        return {"success": False, "error": response.text}
    except Exception as e:
        return {"success": False, "error": str(e)}


def search_circuits(query: str, top_k: int = 10, doc_ids=None):
    """Multi-modal circuit figure search (text + image retrieval).

    Args:
        query: Natural language search query.
        top_k: Maximum number of results.
        doc_ids: Optional document IDs to constrain the search scope.
    """
    if not is_api_connected():
        return {"success": False, "error": "API未连接", "circuits": [], "results": []}

    try:
        payload = {"query": query, "top_k": top_k}
        if doc_ids:
            payload["document_ids"] = [
                str(d) for d in (doc_ids if isinstance(doc_ids, list) else [doc_ids])
            ]
        response = _get_session().post(
            f"{get_api_url()}/api/v1/circuit/search",
            json=payload,
            timeout=180,
        )
        if response.status_code == 200:
            return response.json()
        return {
            "success": False,
            "error": f"HTTP {response.status_code}: {response.text}",
            "circuits": [],
            "results": [],
        }
    except Exception as e:
        return {"success": False, "error": str(e), "circuits": [], "results": []}


def chat_query_stream(query, document_ids=None, callback=None):
    """Send a streaming chat request via SSE (Server-Sent Events).

    The backend sends incremental ``data: {json}`` lines.  Two chunk formats
    are supported for backward compatibility:
        - type=chunk → data.chunk
        - type=token → data.token

    If the stream ends without a ``complete`` event but text chunks were
    received, the accumulated text is returned as a partial result so that
    no user-visible content is lost.

    Args:
        query: User question.
        document_ids: Optional document IDs to scope retrieval.
        callback: Called with each parsed SSE event dict.
    """
    if not is_api_connected():
        return {"success": False, "error": "API未连接"}

    complete_data = {
        "success": False,
        "response": "",
        "citations": [],
        "intent": "",
        "processing_time": 0,
        "extracted_data": None,
    }
    stream_response_text = []

    try:
        payload = {
            "query": query,
            "document_ids": document_ids if document_ids else [],
        }
        response = _get_session().post(
            f"{get_api_url()}/api/v1/chat/stream",
            json=payload,
            stream=True,
            timeout=180,
        )
        if response.status_code == 200:
            try:
                for line in response.iter_lines():
                    if not line:
                        continue
                    try:
                        line_text = line.decode("utf-8")
                    except UnicodeDecodeError:
                        continue

                    if line_text.startswith("data: "):
                        try:
                            data = json.loads(line_text[6:])
                        except json.JSONDecodeError:
                            continue
                    else:
                        try:
                            data = json.loads(line_text)
                        except json.JSONDecodeError:
                            continue

                    if callback:
                        try:
                            callback(data)
                        except Exception as cb_err:
                            logger.warning(f"Stream callback error: {cb_err}")

                    if data.get("type") in ("chunk", "token"):
                        inner = data.get("data", {}) or {}
                        chunk_text = inner.get("chunk") or inner.get("token") or ""
                        if chunk_text:
                            stream_response_text.append(chunk_text)

                    if data.get("type") == "complete":
                        data_inner = data.get("data", {}) or {}
                        complete_data["response"] = data_inner.get("response", "")
                        complete_data["citations"] = data_inner.get("citations", [])
                        complete_data["intent"] = data_inner.get("intent", "")
                        complete_data["processing_time"] = data_inner.get("processing_time", 0)
                        complete_data["extracted_data"] = data_inner.get("extracted_data")
                        complete_data["success"] = data_inner.get("success", True)

                    if data.get("type") == "error":
                        data_inner = data.get("data", {}) or {}
                        complete_data["error"] = data_inner.get("message", "未知错误")
                        if stream_response_text:
                            complete_data["response"] = "".join(stream_response_text)
                            complete_data["success"] = True

            except requests.exceptions.ChunkedEncodingError as chunk_err:
                logger.warning(f"Stream connection interrupted: {chunk_err}")
                if stream_response_text:
                    complete_data["response"] = "".join(stream_response_text)
                    complete_data["success"] = True
                else:
                    complete_data["error"] = f"连接中断: {chunk_err}"

            except requests.exceptions.ConnectionError as conn_err:
                logger.error(f"Stream connection error: {conn_err}")
                complete_data["error"] = f"连接失败: {conn_err}"

            if not complete_data["success"] and not complete_data.get("error") and stream_response_text:
                complete_data["response"] = "".join(stream_response_text)
                complete_data["success"] = True

            return complete_data
        else:
            return {
                "success": False,
                "error": f"HTTP {response.status_code}",
                "response": "",
                "citations": [],
            }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "response": "",
            "citations": [],
        }


def get_circuit_image_url(circuit_id) -> str:
    """Return the URL for a circuit figure identified by its database ID."""
    if not is_api_connected():
        return None
    base_url = get_api_url()
    return f"{base_url}/api/v1/circuit/{circuit_id}/image"


def get_circuit_image_by_path_url(image_path: str) -> str:
    """Return the URL for a circuit figure identified by its server file path."""
    if not is_api_connected():
        return None
    base_url = get_api_url()
    encoded = urllib.parse.quote(image_path, safe="")
    return f"{base_url}/api/v1/circuit/image/by-path?image_path={encoded}"


def get_page_image_url(document_id, page_number: int) -> str:
    """Return the URL for a document page screenshot."""
    if not is_api_connected():
        return None
    base_url = get_api_url()
    return f"{base_url}/api/v1/documents/{document_id}/pages/{page_number}/image"


def get_document_id_by_filename(filename: str, documents=None):
    """Look up a document ID by its filename.

    Checks both ``document_id`` and ``id`` field names for backend
    version compatibility.

    Args:
        filename: The document filename to search for.
        documents: Optional pre-fetched document list.

    Returns:
        The document ID string, or None if not found.
    """
    if documents is None:
        documents = get_documents()
    for doc in documents:
        if doc.get("filename") == filename:
            return doc.get("document_id") or doc.get("id")
    return None
