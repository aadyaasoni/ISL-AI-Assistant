import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from src.lead_pipeline.request_limiter import RequestLimiter
from web import server as web_server


@pytest.fixture
def api_url():
    http_server = ThreadingHTTPServer(("127.0.0.1", 0), web_server.AppHandler)
    thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{http_server.server_port}"
    yield url
    http_server.shutdown()
    thread.join()
    http_server.server_close()


def post(url, path, data, content_type="application/json"):
    request = urllib.request.Request(
        f"{url}{path}",
        data=data,
        headers={"Content-Type": content_type},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=5)


def test_oversized_request_body_is_rejected(api_url):
    body = b"x" * (web_server.MAX_UPLOAD_BYTES + 1)

    with pytest.raises(urllib.error.HTTPError) as error:
        post(api_url, "/api/frame", body, "application/octet-stream")

    assert error.value.code == 413


def test_text_to_sign_endpoint_is_rate_limited_per_client(api_url, monkeypatch):
    monkeypatch.setattr(
        web_server,
        "api_request_limiter",
        RequestLimiter(limit=1, window_seconds=60),
    )
    body = json.dumps({"text": "loud"}).encode("utf-8")

    with post(api_url, "/api/text_to_sign", body) as response:
        assert json.load(response) == {
            "glosses": ["1. loud"],
            "unmatched": [],
        }
    with pytest.raises(urllib.error.HTTPError) as error:
        post(api_url, "/api/text_to_sign", body)

    assert error.value.code == 429
