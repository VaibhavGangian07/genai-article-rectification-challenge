import sys
import json
import random
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import edits
from edits import Edit, LocatedEdit, parse_edits, locate, resolve_overlaps, apply_edits, EditParseError


def replace(find, new, evidence="e"):
    return Edit("replace", find, new, evidence, "source", "")


def delete(find):
    return Edit("delete", find, "", "", "source", "")


class TestParseEdits(unittest.TestCase):

    VALID = {"op": "replace", "find": "3,000", "replace": "300", "evidence": "some 300 people", "evidence_from": "source"}

    def test_plain_fenced_and_wrapped_json(self):
        payload = json.dumps({"edits": [self.VALID]})
        for raw in (payload, f"```json\n{payload}\n```", f"Here are the edits:\n{payload}\nDone."):
            with self.subTest(raw=raw[:20]):
                found, rejected = parse_edits(raw)
                self.assertEqual([(e.find, e.replace) for e in found], [("3,000", "300")])
                self.assertEqual(rejected, [])

    def test_empty_edit_list(self):
        self.assertEqual(parse_edits('{"edits": []}'), ([], []))

    def test_unusable_responses_raise(self):
        for raw in ("", None, "no json here", '{"edits": [', '{"changes": []}', '[{"op": "replace"}]'):
            with self.subTest(raw=raw), self.assertRaises(EditParseError):
                parse_edits(raw)

    def test_schema_violations_are_rejected_individually(self):
        bad = [
            ({"op": "rewrite", "find": "a", "replace": "b"}, "schema_invalid: op"),
            ({"op": "replace", "find": "  ", "replace": "b"}, "schema_invalid: find"),
            ({"op": "replace", "find": "x" * 301, "replace": "b"}, "schema_invalid: find"),
            ({"op": "replace", "find": "a\nb", "replace": "c"}, "multiline_not_allowed"),
            ({"op": "replace", "find": "a", "replace": "b\nc"}, "multiline_not_allowed"),
            ({"op": "replace", "find": "same", "replace": "same"}, "schema_invalid: replace"),
            ({"op": "replace", "find": "a", "replace": ""}, "schema_invalid: replace"),
            ({"op": "replace", "find": "a", "replace": "## New heading"}, "forbidden_markup"),
            ({"op": "replace", "find": "a", "replace": "b ```"}, "forbidden_markup"),
            ({"op": "replace", "find": "a", "replace": "**Error Annotations:**"}, "forbidden_markup"),
            ({"op": "delete", "find": "a", "replace": "b"}, "schema_invalid: delete with replacement"),
            ("not an object", "schema_invalid: not an object"),
        ]
        found, rejected = parse_edits(json.dumps({"edits": [item for item, _ in bad] + [self.VALID]}))
        self.assertEqual(len(found), 1)
        self.assertEqual([reason for _, reason in rejected], [reason for _, reason in bad])

    def test_delete_without_replace_field(self):
        found, _ = parse_edits('{"edits": [{"op": "delete", "find": " at WWDC"}]}')
        self.assertEqual((found[0].op, found[0].replace), ("delete", ""))


class TestLocate(unittest.TestCase):

    BODY = "Hasan Ali took 5/30 in Karachi.\nAlibaba listed in Karachi, said “officials” — again."

    def test_unique_exact_match(self):
        item, reason = locate(self.BODY, replace("took 5/30 in Karachi", "x"))
        self.assertIsNone(reason)
        self.assertEqual(self.BODY[item.start:item.end], "took 5/30 in Karachi")

    def test_ambiguous_match_is_rejected(self):
        self.assertEqual(locate(self.BODY, replace("Karachi", "Lahore")), (None, "ambiguous(2)"))

    def test_not_found(self):
        self.assertEqual(locate(self.BODY, replace("Lahore", "x")), (None, "not_found"))

    def test_word_boundaries(self):
        item, reason = locate(self.BODY, replace("Ali", "Ally"))
        self.assertIsNone(reason)
        self.assertEqual(item.start, self.BODY.index("Ali "))

    def test_normalized_quotes_dashes_and_spaces(self):
        item, reason = locate(self.BODY, replace('said "officials" - again', "x"))
        self.assertIsNone(reason)
        self.assertEqual(self.BODY[item.start:item.end], "said “officials” — again")
        item, reason = locate("Some   spaced  words here.", replace("spaced words", "x"))
        self.assertEqual("Some   spaced  words here."[item.start:item.end], "spaced  words")

    def test_normalized_match_must_also_be_unique(self):
        body = "the “plan” and the “plan”"
        self.assertEqual(locate(body, replace('the "plan"', "x")), (None, "ambiguous(2)"))


class TestApplyEdits(unittest.TestCase):

    BODY = "## Title\nThe team won 3 matches at Lords, with critics calling it lucky.\n\nCoach Smith left."

    def located(self, edit):
        item, reason = locate(self.BODY, edit)
        self.assertIsNone(reason)
        return item

    def test_replace_changes_only_the_span(self):
        out = apply_edits(self.BODY, [self.located(replace("won 3 matches", "won 5 matches"))])
        self.assertEqual(out, self.BODY.replace("won 3 matches", "won 5 matches"))

    def test_delete_tidies_the_join(self):
        # Only the join is tidied; punctuation outside "find" is never removed
        out = apply_edits(self.BODY, [self.located(delete(" with critics calling it lucky"))])
        self.assertEqual(out.split("\n")[1], "The team won 3 matches at Lords,.")
        out = apply_edits(self.BODY, [self.located(delete(", with critics calling it lucky"))])
        self.assertEqual(out.split("\n")[1], "The team won 3 matches at Lords.")
        out = apply_edits("A big red car.", [LocatedEdit(delete("red "), 6, 10)])
        self.assertEqual(out, "A big car.")
        out = apply_edits("A big red car.", [LocatedEdit(delete(" red"), 5, 9)])
        self.assertEqual(out, "A big car.")

    def test_result_does_not_depend_on_edit_order(self):
        items = [self.located(replace("won 3", "won 5")), self.located(replace("Lords", "Eden")),
                 self.located(replace("Coach Smith", "Coach Jones"))]
        expected = apply_edits(self.BODY, items)
        for _ in range(5):
            random.shuffle(items)
            self.assertEqual(apply_edits(self.BODY, items), expected)
        self.assertEqual(expected, "## Title\nThe team won 5 matches at Eden, with critics calling it lucky.\n\nCoach Jones left.")

    def test_untouched_lines_are_byte_identical(self):
        out = apply_edits(self.BODY, [self.located(replace("Coach Smith", "Coach Jones"))])
        self.assertEqual(out.split("\n")[:3], self.BODY.split("\n")[:3])

    def test_untouched_line_check_detects_corruption(self):
        with self.assertRaises(AssertionError):
            edits._check_untouched_lines(self.BODY, self.BODY.replace("Title", "Tidle"), [])
        with self.assertRaises(AssertionError):
            edits._check_untouched_lines(self.BODY, self.BODY + "\nextra", [])


class TestResolveOverlaps(unittest.TestCase):

    def test_identical_duplicates_collapse(self):
        a, b = LocatedEdit(replace("x", "y"), 0, 1), LocatedEdit(replace("x", "y"), 0, 1)
        kept, rejected = resolve_overlaps([a, b])
        self.assertEqual((len(kept), rejected), (1, []))

    def test_overlapping_edits_are_both_rejected(self):
        a = LocatedEdit(replace("won 3 matches", "won 5 matches"), 10, 23)
        b = LocatedEdit(replace("3 matches at", "3 games at"), 14, 26)
        c = LocatedEdit(replace("Lords", "Eden"), 30, 35)
        kept, rejected = resolve_overlaps([a, b, c])
        self.assertEqual(kept, [c])
        self.assertEqual(sorted(r for _, r in rejected), ["overlap", "overlap"])

    def test_adjacent_edits_are_kept(self):
        a, b = LocatedEdit(replace("ab", "x"), 0, 2), LocatedEdit(replace("cd", "y"), 2, 4)
        kept, rejected = resolve_overlaps([b, a])
        self.assertEqual((kept, rejected), ([a, b], []))


if __name__ == "__main__":
    unittest.main()
