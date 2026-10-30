import sys
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import evaluate
import llm_client
from evaluate import score, diff_regions, word_changes, gold_edits

AI = "The team won 3 matches in March at Lords. Coach Smith was pleased with the result."
REF = "The team won 5 matches in April at Lords. Coach Smith was pleased with the result."


class TestScore(unittest.TestCase):

    def test_perfect_candidate(self):
        s = score(AI, REF, REF)
        self.assertTrue(s["exact_match"])
        self.assertEqual((s["precision"], s["recall"], s["f1"]), (1.0, 1.0, 1.0))
        self.assertEqual((s["unintended_regions"], s["annotation_leak"]), (0, None))

    def test_uncorrected_baseline(self):
        s = score(AI, AI, REF)
        self.assertFalse(s["exact_match"])
        self.assertEqual((s["gold_regions"], s["pred_regions"]), (1, 0))   # "3 ... March" merges into one region
        self.assertEqual((s["precision"], s["recall"], s["f1"]), (None, 0.0, 0.0))
        self.assertEqual(s["words"], {"added": 0, "removed": 0, "modified": 0})
        self.assertGreater(s["similarity"], 0.85)   # similarity stays high although no fact was fixed

    def test_partial_fix(self):
        ai = "Alpha won 3 games. Beta scored 10 goals. Gamma lost 2 sets."
        ref = "Alpha won 4 games. Beta scored 10 goals. Gamma lost 7 sets."
        s = score(ai, ai.replace("won 3", "won 4"), ref)
        self.assertEqual((s["gold_regions"], s["correct_gold"], s["pred_regions"]), (2, 1, 1))
        self.assertEqual((s["precision"], s["recall"]), (1.0, 0.5))

    def test_wrong_fix_is_located_but_not_correct(self):
        s = score(AI, AI.replace("3 matches in March", "4 matches in March"), REF)
        self.assertEqual((s["located_gold"], s["correct_gold"]), (1, 0))
        self.assertEqual((s["precision"], s["recall"], s["f1"]), (0.0, 0.0, 0.0))
        self.assertEqual(s["unintended_regions"], 0)

    def test_unrelated_change_is_unintended(self):
        candidate = REF.replace("was pleased with", "was happy with")
        s = score(AI, candidate, REF)
        self.assertEqual((s["unintended_regions"], s["unintended_words"]), (1, 1))
        self.assertEqual((s["precision"], s["recall"]), (0.5, 1.0))

    def test_insertion_and_deletion_regions(self):
        ai = "The film was officially banned in the state."
        ref = "The film was not officially banned in the state."
        self.assertEqual(score(ai, ref, ref)["recall"], 1.0)
        self.assertEqual(score(ai, ai, ref)["recall"], 0.0)
        ai2 = "Apple launched phones at WWDC in June, the company said."
        ref2 = "Apple launched phones, the company said."
        self.assertEqual(score(ai2, ref2, ref2)["recall"], 1.0)

    def test_whitespace_only_difference(self):
        s = score(AI, REF, REF.replace("Lords. Coach", "Lords.  Coach"))
        self.assertEqual((s["exact_match"], s["exact_words"], s["recall"]), (False, True, 1.0))

    def test_annotation_leak_detected(self):
        s = score(AI, REF + "\n\n**Error Annotations:**\n[]", REF)
        self.assertEqual(s["annotation_leak"], "**Error Annotations:**")

    def test_nearby_hunks_merge_into_one_region(self):
        self.assertEqual(len(diff_regions("a b c d e f".split(), "a X c Y e f".split())), 1)
        self.assertEqual(len(diff_regions("a b c d e f g".split(), "a X c d e Y g".split())), 2)

    def test_word_change_counts(self):
        self.assertEqual(word_changes("a b c".split(), "a x y c d".split()), {"added": 2, "removed": 0, "modified": 1})
        self.assertEqual(word_changes("a b c d".split(), "a d".split()), {"added": 0, "removed": 2, "modified": 0})


class TestGoldEdits(unittest.TestCase):

    def test_edits_are_unique_single_line_and_reproduce_reference(self):
        body = "## Match\nThe team won 3 games.\n\nThe team won 3 games again."
        ref = "## Match\nThe team won 5 games.\n\nThe team won 3 games again."
        edits, not_derivable = gold_edits(body, ref, "The team won 5 games, officials said.", [])
        self.assertEqual(not_derivable, [])
        [edit] = edits
        # shortest unique context: "won 3 games." (the other occurrence is "won 3 games again.")
        self.assertEqual((edit["find"], edit["replace"]), ("won 3 games.", "won 5 games."))
        self.assertEqual(body.count(edit["find"]), 1)
        self.assertEqual(edit["evidence"], "The team won 5 games, officials said.")

    def test_evidence_prefers_sentence_with_new_value(self):
        edits, _ = gold_edits("Sales rose 3 percent last year.", "Sales rose 9 percent last year.",
                              "Sales rose 3 percent in 2023. Sales rose 9 percent last year, the firm said.", [])
        self.assertIn("9 percent", edits[0]["evidence"])

    def test_sentences_do_not_break_at_abbreviations_or_initials(self):
        text = ("He arrived by 10 a.m. on Tuesday (September 9). Team led by Inspector Kiran C. Nair, "
                "with Dr. Rao present. Police said so.\nNew line here too.")
        self.assertEqual(evaluate._sentences(text),
                         ["He arrived by 10 a.m. on Tuesday (September 9).",
                          "Team led by Inspector Kiran C. Nair, with Dr. Rao present.",
                          "Police said so.", "New line here too."])

    def test_evidence_ranked_by_article_line_not_just_the_changed_word(self):
        line = "Vedan appeared before the Thrikkakara police in Ernakulam on Monday morning."
        source = ("Published Date: Tuesday September 09, 2025\n"
                  "Vedan turned up at the Thrikkakara police station in Ernakulam on Tuesday morning.")
        evidence, origin = evaluate.pick_evidence("Monday", "Tuesday", line, source, [])
        self.assertEqual((evidence, origin), ("Vedan turned up at the Thrikkakara police station in Ernakulam on Tuesday morning.", "source"))

    def test_source_sentence_preferred_over_annotation_on_a_tie(self):
        notes = [{"error": "Inspector Rajeev P. Menon", "correction": "Inspector Kiran C. Nair"}]
        evidence, origin = evaluate.pick_evidence("Rajeev P. Menon.", "Kiran C. Nair.", "Team led by Inspector Rajeev P. Menon.",
                                                  "The team was led by Inspector Kiran C. Nair.", notes)
        self.assertEqual(origin, "source")

    def test_annotation_correction_used_when_source_is_silent(self):
        notes = [{"error": "with Asiana Airlines", "correction": "\"with Korean Air\""}]
        edits, _ = gold_edits("A flight with Asiana Airlines left.", "A flight with Korean Air left.",
                              "Officials confirmed the flight left on time.", notes)
        self.assertEqual((edits[0]["evidence"], edits[0]["evidence_from"]), ("with Korean Air", "annotation"))


class TestHarness(unittest.TestCase):

    def setUp(self):
        self.addCleanup(mock.patch.stopall)
        mock.patch.object(llm_client, "_completion", side_effect=AssertionError("no provider calls")).start()

    def test_full_run_writes_reports_without_provider_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary, articles = evaluate.evaluate(output_dir=Path(tmp))
            for name in ("summary.json", "article_results.json", "evaluation_report.md"):
                self.assertTrue((Path(tmp) / name).is_file())
            self.assertEqual(len(list((Path(tmp) / "gold_replay").glob("article_*.txt"))), 10)
            report = (Path(tmp) / "evaluation_report.md").read_text(encoding="utf-8")
        self.assertEqual([a["article_id"] for a in articles], evaluate.SPLIT["tuning"] + evaluate.SPLIT["holdout"])
        base = summary["baseline"]["all"]
        self.assertEqual((base["exact_match"], base["recall"], base["unintended_regions"], base["annotation_leaks"]),
                         (0, 0.0, 0, 0))
        replay = summary["gold_replay"]
        self.assertEqual(replay["all"]["annotation_leaks"], 0)
        self.assertEqual(replay["all"]["unintended_regions"], 0)
        self.assertEqual(replay["tuning"]["articles"] + replay["holdout"]["articles"], 10)
        self.assertEqual(replay["annotated"]["articles"] + replay["unannotated"]["articles"], 10)
        self.assertIn("It is not LLM performance", report)
        self.assertIn("Not a factual-accuracy score", report)

    def test_candidate_directory_is_scored(self):
        with tempfile.TemporaryDirectory() as cand, tempfile.TemporaryDirectory() as out:
            ref = (ROOT / "reference_rectified" / "article_010.txt").read_text(encoding="utf-8")
            (Path(cand) / "article_010.txt").write_text(ref, encoding="utf-8")
            summary, articles = evaluate.evaluate(candidate_dir=Path(cand), label="perfect_010", output_dir=Path(out))
        row = next(a for a in articles if a["article_id"] == "article_010")
        self.assertTrue(row["modes"]["perfect_010"]["exact_match"])
        self.assertTrue(next(a for a in articles if a["article_id"] == "article_001")["modes"]["perfect_010"]["missing"])
        self.assertEqual(summary["perfect_010"]["all"]["articles"], 1)

    def test_refuses_protected_output_directories(self):
        for target in (ROOT / "rectified_articles", ROOT / "reference_rectified" / "x"):
            with self.subTest(target=target), self.assertRaises(ValueError):
                evaluate.evaluate(output_dir=target)


if __name__ == "__main__":
    unittest.main()
