import json
import threading
import urllib.request
from http.server import ThreadingHTTPServer

from web.server import AppHandler


class TestTextToSignEndpoint:
    @classmethod
    def setup_class(cls):
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), AppHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}/api/text_to_sign"

    @classmethod
    def teardown_class(cls):
        cls.server.shutdown()
        cls.thread.join()
        cls.server.server_close()

    def post_text(self, text):
        request = urllib.request.Request(
            self.url,
            data=json.dumps({"text": text}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.load(response)

    def test_matched_words_return_dataset_glosses(self):
        assert self.post_text("Loud, big large!") == {
            "glosses": ["1. loud", "83. big large"],
            "unmatched": [],
        }

    def test_unsupported_words_are_returned(self):
        assert self.post_text("mystery") == {
            "glosses": [],
            "unmatched": ["mystery"],
        }

    def test_empty_text_returns_empty_lists(self):
        assert self.post_text("  ") == {"glosses": [], "unmatched": []}
