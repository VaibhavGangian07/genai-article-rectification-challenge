import sys
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from article_utils import split_annotations
from edits import Edit, locate
from validation import check_edit, check_output, facts

BODY = "## Results\nThe team, led by coach John Smith, won 3 matches in March 2024 at Lords stadium.\n\nFans celebrated outside."
SOURCE = ("The team, coached by John Smith, won 5 matches in April 2024 at Lords stadium, officials said. "
          "Ticket prices rose by 7 dollars for supporters this season.")
EVIDENCE = "The team, coached by John Smith, won 5 matches in April 2024 at Lords stadium, officials said."


def check(op, find, new="", evidence=EVIDENCE, body=BODY, source=SOURCE, annotations=()):
    edit = Edit(op, find, new, evidence, "source", "")
    item, reason = locate(body, edit)
    assert reason is None, reason
    return check_edit(edit, item.start, item.end, body, source, list(annotations))


class TestReplaceEvidence(unittest.TestCase):

    def test_source_supported_number_fix(self):
        self.assertEqual(check("replace", "won 3 matches", "won 5 matches"), ("source_supported", None))

    def test_source_supported_date_fix(self):
        self.assertEqual(check("replace", "in March 2024", "in April 2024"), ("source_supported", None))

    def test_new_value_must_be_in_the_quote(self):
        self.assertEqual(check("replace", "won 3 matches", "won 9 matches"), (None, "fact_not_in_evidence"))

    def test_quote_must_exist_in_source(self):
        self.assertEqual(check("replace", "won 3 matches", "won 5 matches", evidence="The team won 5 matches in total."),
                         (None, "evidence_not_found"))

    def test_quote_must_not_be_trivially_short(self):
        self.assertEqual(check("replace", "won 3 matches", "won 5 matches", evidence="5 matches"),
                         (None, "evidence_too_short"))

    def test_value_found_elsewhere_in_source_is_not_enough(self):
        # "7" is in the source, but in a sentence about ticket prices, not this match count
        result = check("replace", "won 3 matches", "won 7 matches",
                       evidence="Ticket prices rose by 7 dollars for supporters this season.")
        self.assertEqual(result, (None, "evidence_not_slot_aligned"))

    def test_evidence_that_also_states_the_old_value_is_not_a_contradiction(self):
        source = "The team won 3 matches in March and won 5 matches in April 2024 at Lords stadium."
        result = check("replace", "won 3 matches", "won 5 matches", evidence=source, source=source)
        self.assertEqual(result, (None, "no_contradiction"))

    def test_fact_type_must_match(self):
        self.assertEqual(check("replace", "in March 2024", "in Lords 2024"), (None, "type_mismatch"))

    def test_multi_word_entity_must_appear_as_a_phrase(self):
        body = "The probe was led by Inspector Rajeev P. Menon on Monday."
        source = "He was being questioned by the investigation team led by Inspector Kiran C. Nair."
        ok = check("replace", "Inspector Rajeev P. Menon", "Inspector Kiran C. Nair", evidence=source,
                   body=body, source=source)
        self.assertEqual(ok, ("source_supported", None))
        split = check("replace", "Inspector Rajeev P. Menon", "Inspector Nair Kiran", evidence=source,
                      body=body, source=source)
        self.assertEqual(split, (None, "fact_not_in_evidence"))

    def test_term_correction_needs_the_new_term_in_the_quote(self):
        body = "After being granted regular bail by the Kerala High Court, Vedan was questioned."
        source = "Vedan, who has been granted anticipatory bail in the case by the Kerala High Court, turned up."
        self.assertEqual(check("replace", "granted regular bail", "granted anticipatory bail",
                               evidence=source, body=body, source=source), ("source_supported", None))
        self.assertEqual(check("replace", "granted regular bail", "granted interim protective custodial bail",
                               evidence=source, body=body, source=source), (None, "unsupported_rewrite"))

    def test_negation_must_be_quoted(self):
        body = "Though officially banned in the state, the film still ran."
        source = "Though not officially banned in the state, the film was pulled from most theatres."
        self.assertEqual(check("replace", "Though officially banned", "Though not officially banned",
                               evidence=source, body=body, source=source), ("source_supported", None))
        self.assertEqual(check("replace", "Though officially banned", "Though never officially banned",
                               evidence=source, body=body, source=source), (None, "fact_not_in_evidence"))


class TestAnnotationEvidence(unittest.TestCase):

    SILENT_SOURCE = "The team played at Lords stadium this spring, officials said."
    ANNOTATION = {"error": "won 3 matches", "correction": "won 5 matches in April"}

    def test_linked_annotation_supports_edit_when_source_is_silent(self):
        result = check("replace", "won 3 matches", "won 5 matches", evidence="won 5 matches in April",
                       source=self.SILENT_SOURCE, annotations=[self.ANNOTATION])
        self.assertEqual(result, ("annotation_supported", None))

    def test_annotation_about_other_text_is_rejected(self):
        other = {"error": "in March 2024", "correction": "won 5 matches in April"}
        result = check("replace", "won 3 matches", "won 5 matches", evidence="won 5 matches in April",
                       source=self.SILENT_SOURCE, annotations=[other])
        self.assertEqual(result, (None, "annotation_unlinked"))

    def test_source_stating_the_old_value_overrides_the_annotation(self):
        source = "The team won 3 matches at Lords stadium, officials said."
        result = check("replace", "won 3 matches", "won 5 matches", evidence="won 5 matches in April",
                       source=source, annotations=[self.ANNOTATION])
        self.assertEqual(result, (None, "source_conflict"))


class TestDeletion(unittest.TestCase):

    BODY = "Apple unveiled the phones at WWDC in June 2025, the company said, with critics calling it bold.\nNext line."
    SOURCE = "Apple launched the phones at its Awe Dropping event on September 9, the company said."

    def test_linked_annotation_allows_deletion(self):
        note = {"error": "with critics calling it bold", "correction": "(not in source)"}
        self.assertEqual(check("delete", ", with critics calling it bold", body=self.BODY, source=self.SOURCE,
                               evidence="", annotations=[note]), ("annotation_supported", None))

    def test_contradicting_source_allows_deletion(self):
        self.assertEqual(check("delete", " at WWDC in June 2025", body=self.BODY, source=self.SOURCE,
                               evidence=self.SOURCE), ("source_supported", None))

    def test_replace_that_only_removes_text_is_treated_as_deletion(self):
        self.assertEqual(check("replace", "phones at WWDC in June 2025,", "phones,", body=self.BODY,
                               source=self.SOURCE, evidence=self.SOURCE), ("source_supported", None))

    def test_absence_from_source_is_not_enough(self):
        source = "Apple launched the phones, the company said."
        self.assertEqual(check("delete", " at WWDC in June 2025", body=self.BODY, source=source, evidence=source),
                         (None, "delete_unsupported"))

    def test_generic_claims_are_never_deleted_without_annotation(self):
        self.assertEqual(check("delete", ", with critics calling it bold", body=self.BODY, source=self.SOURCE,
                               evidence=self.SOURCE), (None, "delete_generic"))

    def test_whole_line_cannot_be_deleted(self):
        note = {"error": "Next line.", "correction": "-"}
        self.assertEqual(check("delete", "Next line.", body=self.BODY, source=self.SOURCE, evidence="",
                               annotations=[note]), (None, "delete_whole_line"))

    def test_long_deletion_needs_annotation(self):
        body = ("Intro. " + " ".join(f"Word{i} in 2025" for i in range(12)) + " end.\nTail.")
        find = " " + " ".join(f"Word{i} in 2025" for i in range(12))
        self.assertEqual(check("delete", find, body=body, source=self.SOURCE, evidence=self.SOURCE),
                         (None, "delete_too_long"))


class TestOutputChecks(unittest.TestCase):

    def test_clean_output_passes(self):
        out = BODY.replace("won 3", "won 5")
        self.assertEqual(check_output(BODY, out, [Edit("replace", "won 3", "won 5", "", "", "")]), ([], []))

    def test_annotation_leak_is_critical(self):
        for leak in ('\n\n**Error Annotations:**\n[]', ' "correction": "x"', ' ```'):
            with self.subTest(leak=leak):
                critical, _ = check_output(BODY, BODY + leak, [])
                self.assertTrue(any(c.startswith("annotation_leak") for c in critical))

    def test_structure_changes_are_critical(self):
        self.assertIn("line_count_changed", check_output(BODY, BODY + "\nextra", [])[0])
        self.assertIn("heading_marker_changed: line 1", check_output(BODY, BODY.replace("## Results", "Results"), [])[0])
        self.assertIn("empty_output", check_output(BODY, "  ", [])[0])

    def test_warnings_do_not_block(self):
        out = BODY.replace("## Results", "## Scores").replace("Fans celebrated outside.", "Fans.")
        critical, warnings = check_output(BODY, out, [Edit("replace", "won 3", "won 5", "", "", "")])
        self.assertEqual(critical, [])
        self.assertIn("heading_text_changed: line 1", warnings)
        self.assertTrue(any(w.startswith("length_ratio") for w in warnings))

    def test_possible_unfixed_duplicate_warning(self):
        body = "Karachi hosted. Later, Karachi again."
        out = "Lahore hosted. Later, Karachi again."
        _, warnings = check_output(body, out, [Edit("replace", "Karachi hosted", "Lahore hosted", "", "", "")])
        self.assertIn("possible_unfixed_duplicate: karachi", warnings)

    def test_human_references_pass_critical_checks(self):
        mapping = {a["article_id"]: a for a in json.loads((ROOT / "article_mapping.json").read_text())}
        for ref in sorted((ROOT / "reference_rectified").glob("article_*.txt")):
            with self.subTest(article=ref.stem):
                ai = (ROOT / mapping[ref.stem]["ai_generated_file"]).read_text(encoding="utf-8")
                body = split_annotations(ai)[0]
                self.assertEqual(check_output(body, ref.read_text(encoding="utf-8"), [])[0], [])


class TestFacts(unittest.TestCase):

    def test_typed_values(self):
        f = facts("Inspector Kiran C. Nair said Rs. 1,34,900 on Tuesday, September 9, not 5/19.")
        self.assertEqual(f["NUM"], {"134900", "9", "5/19"})
        self.assertEqual(f["DATE"], {"tuesday", "september"})
        self.assertEqual(f["NEG"], {"not"})
        self.assertIn(("inspector", "kiran", "c", "nair"), f["ENT"])


if __name__ == "__main__":
    unittest.main()
