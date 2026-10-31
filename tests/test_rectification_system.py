import sys
import json
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import tempfile
import llm_client
import rectifier
import rectification_system
from llm_client import LLMError, LLMResponse
from article_utils import strip_annotations, find_annotation_leak
from test_validation import COMMENTARY_004

MAPPING = {a["article_id"]: a for a in json.loads((ROOT / "article_mapping.json").read_text())}


def load(article_id):
    read = lambda key: (ROOT / MAPPING[article_id][key]).read_text(encoding="utf-8")
    reference = ROOT / "reference_rectified" / f"{article_id}.txt"
    return (read("ai_generated_file"), read("source_file"),
            reference.read_text(encoding="utf-8") if reference.exists() else None)


def reply(edits):
    text = edits if isinstance(edits, str) else json.dumps({"edits": edits})
    return LLMResponse(text=text, finish_reason="stop", prompt_tokens=1000, completion_tokens=200,
                       reasoning_tokens=100, cost_usd=0.00027)


EVIDENCE_010 = ("Vedan, who has been granted anticipatory bail in the case by the Kerala High Court, "
                "turned up at the police station by 10 a.m. on Tuesday (September 9, 2025).")
EDITS_010 = [
    {"op": "replace", "find": "on Monday morning", "replace": "on Tuesday morning", "evidence": EVIDENCE_010},
    {"op": "replace", "find": "granted regular bail", "replace": "granted anticipatory bail", "evidence": EVIDENCE_010},
    {"op": "replace", "find": "Inspector Rajeev P. Menon", "replace": "Inspector Kiran C. Nair",
     "evidence": "He was being questioned by the investigation team led by Inspector Kiran C. Nair."},
]
EDITS_008 = [
    {"op": "replace", "find": "detained around 3,000 South Korean nationals",
     "replace": "detained around 300 South Korean nationals",
     "evidence": "Some 300 South Korean nationals were detained", "evidence_from": "source"},
    {"op": "replace", "find": "with Asiana Airlines", "replace": "with Korean Air",
     "evidence": "arranged a chartered flight with Korean Air", "evidence_from": "annotation"},
]


class RunTestCase(unittest.TestCase):

    def setUp(self):
        self.complete = mock.patch.object(llm_client, "complete",
                                          side_effect=AssertionError("unexpected API call")).start()
        self.addCleanup(mock.patch.stopall)

    def run_article(self, article_id, *replies):
        self.complete.side_effect = list(replies)
        ai, source, _ = load(article_id)
        return rectification_system.run(ai, source, article_id)


class TestRectified(RunTestCase):

    def test_source_supported_edits_reproduce_the_human_reference(self):
        result = self.run_article("article_010", reply(EDITS_010))
        self.assertEqual(result.status, "rectified")
        self.assertEqual(result.text, load("article_010")[2])
        self.assertEqual([a["support"] for a in result.applied], ["source_supported"] * 3)
        self.assertEqual((result.skipped, result.warnings), ([], []))

    def test_annotation_supported_edit_is_reported_separately(self):
        result = self.run_article("article_008", reply(EDITS_008))
        self.assertEqual(result.text, load("article_008")[2])
        self.assertEqual([a["support"] for a in result.applied], ["source_supported", "annotation_supported"])

    def test_source_and_ai_article_reach_the_prompt(self):
        self.run_article("article_010", reply([]))
        ai, source, _ = load("article_010")
        prompt = self.complete.call_args.args[0][1]["content"]
        self.assertIn(source.strip(), prompt)
        self.assertIn(strip_annotations(ai), prompt)
        self.assertIn('"error": "\\"on Monday morning\\""', prompt)

    def test_usage_is_summed_across_calls(self):
        result = self.run_article("article_010", reply("not json"), reply(EDITS_010))
        self.assertEqual(result.status, "rectified")
        self.assertEqual((result.usage["calls"], result.usage["prompt_tokens"]), (2, 2000))
        self.assertAlmostEqual(result.usage["cost_usd"], 0.00054)


class TestRejectedEdits(RunTestCase):

    def test_unsupported_ambiguous_and_hallucinated_edits_are_skipped(self):
        bad = [
            {"op": "replace", "find": "Inspector Rajeev P. Menon", "replace": "Inspector Arun Kumar",
             "evidence": "He was being questioned by the investigation team led by Inspector Kiran C. Nair."},
            {"op": "replace", "find": "the", "replace": "a", "evidence": EVIDENCE_010},
            {"op": "replace", "find": "text that is not in the article", "replace": "x", "evidence": EVIDENCE_010},
            {"op": "replace", "find": "granted regular bail", "replace": "granted interim bail",
             "evidence": "Vedan was granted interim bail by the Supreme Court of India."},
        ]
        result = self.run_article("article_010", reply(bad))
        ai = load("article_010")[0]
        self.assertEqual(result.status, "no_valid_edits")
        self.assertEqual(result.text, strip_annotations(ai))
        self.assertEqual([s["reason"].split("(")[0] for s in result.skipped],
                         ["fact_not_in_evidence", "ambiguous", "not_found", "evidence_not_found"])

    def test_overlapping_edits_are_both_skipped(self):
        overlapping = [EDITS_010[1], {**EDITS_010[1], "find": "regular bail by", "replace": "anticipatory bail by"}]
        result = self.run_article("article_010", reply(overlapping))
        self.assertEqual(result.status, "no_valid_edits")
        self.assertEqual([s["reason"] for s in result.skipped], ["overlap", "overlap"])

    def test_no_edits_proposed(self):
        result = self.run_article("article_010", reply([]))
        self.assertEqual((result.status, result.text), ("no_change", strip_annotations(load("article_010")[0])))


class TestFallback(RunTestCase):

    def assertFallback(self, result, reason_prefix):
        self.assertEqual(result.status, "fallback")
        self.assertTrue(result.reason.startswith(reason_prefix), result.reason)
        self.assertEqual(result.text, strip_annotations(load("article_010")[0]))

    def test_invalid_json_twice(self):
        result = self.run_article("article_010", reply("not json"), reply("still not json"))
        self.assertFallback(result, "parse_error")
        self.assertEqual(self.complete.call_count, 2)

    def test_llm_errors(self):
        for reason in ("truncated", "budget_exceeded", "llm_error:AuthenticationError", "disabled"):
            with self.subTest(reason=reason):
                self.assertFallback(self.run_article("article_010", LLMError(reason)), reason)

    def test_critical_output_failure(self):
        with mock.patch.object(rectification_system, "check_output", return_value=(["annotation_leak: x"], [])):
            result = self.run_article("article_010", reply(EDITS_010))
        self.assertFallback(result, "critical: annotation_leak")


class TestAnnotationsNeverLeak(RunTestCase):

    def test_annotation_notes_never_reach_the_saved_article(self):
        edits = [{"op": "replace", "find": f, "replace": r, "evidence": r, "evidence_from": "annotation"}
                 for f, r in COMMENTARY_004]
        edits.append({"op": "replace", "find": "only a handful of theaters refusing to screen it",
                      "replace": "most theaters refusing to screen it",
                      "evidence": "most theaters refusing to screen it", "evidence_from": "annotation"})
        result = self.run_article("article_004", reply(edits))
        self.assertEqual(result.status, "rectified")
        self.assertEqual([a["replace"] for a in result.applied], ["most theaters refusing to screen it"])
        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "article_004.txt"
            with mock.patch.object(rectifier, "get_article_mapping", return_value={"rectified_file": str(target)}):
                rectifier.save_rectified_article("article_004", result.text)
            target.read_bytes().decode("utf-8")   # strict UTF-8; raises on any invalid byte
            saved = target.read_text(encoding="utf-8")   # text mode, as every reader in the repo opens it
        self.assertEqual(saved, result.text)
        self.assertNotIn("\ufffd", saved)
        self.assertIsNone(find_annotation_leak(saved))
        for _, note in COMMENTARY_004:
            self.assertNotIn(note, saved)
        for phrase in ("serious/controversial", "in some references", "director's name is", "in that regard"):
            self.assertNotIn(phrase, saved)

    def test_outputs_for_every_status_are_free_of_annotation_text(self):
        attempts = [("article_008", [reply(EDITS_008)]), ("article_015", [reply([])]),
                    ("article_047", [LLMError("truncated")]), ("article_053", [reply("x"), reply("y")])]
        for article_id, replies in attempts:
            with self.subTest(article=article_id):
                result = self.run_article(article_id, *replies)
                self.assertIsNone(find_annotation_leak(result.text))


if __name__ == "__main__":
    unittest.main()
