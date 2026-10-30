import io
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import requests
import budget_checker

KEY = "sk-test-SECRETVALUE-1234"
BASE = "http://127.0.0.1:9"


def fake_response(status=200, payload=None):
    response = requests.Response()
    response.status_code = status
    response._content = (b'{"info": {"spend": 0.25, "max_budget": 3.0}}' if payload is None else payload)
    response.url = f"{BASE}/key/info"
    return response


class TestKeyNeverExposed(unittest.TestCase):

    def run_main(self, side_effect):
        out = io.StringIO()
        with mock.patch.object(budget_checker, "API_KEY", KEY), \
             mock.patch.object(budget_checker, "LITELLM_BASE_URL", BASE), \
             mock.patch.object(budget_checker.requests, "get", side_effect=side_effect) as get, \
             mock.patch.object(sys, "argv", ["budget_checker.py"]), mock.patch.object(sys, "stdout", out):
            try:
                budget_checker.main()
            except SystemExit:
                pass
        return out.getvalue(), get

    def assertNoKeyFragment(self, text):
        for fragment in (KEY, KEY[:4], KEY[-4:], "SECRETVALUE"):
            self.assertNotIn(fragment, text)

    def test_successful_check(self):
        text, get = self.run_main([fake_response()])
        self.assertIn("Total Spend: $0.2500", text)
        self.assertNoKeyFragment(text)
        self.assertNotIn("params", get.call_args.kwargs)   # key is sent in the header only
        self.assertEqual(get.call_args.kwargs["headers"]["Authorization"], f"Bearer {KEY}")

    def test_http_error_does_not_print_the_url_or_key(self):
        def unauthorized(url, **kwargs):
            response = fake_response(401, b'{"error": "bad key"}')
            response.url = f"{url}?key={KEY}"   # what requests would put in its error message
            response.raise_for_status()
        text, _ = self.run_main(unauthorized)
        self.assertIn("HTTPError (HTTP 401)", text)
        self.assertNoKeyFragment(text)

    def test_connection_error_does_not_print_details(self):
        text, _ = self.run_main(requests.exceptions.ConnectionError(f"failed for {BASE}/key/info?key={KEY}"))
        self.assertIn("ConnectionError", text)
        self.assertNoKeyFragment(text)


if __name__ == "__main__":
    unittest.main()
