import sys
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from article_utils import split_annotations, parse_annotations
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

    def test_date_change_may_add_a_day_number(self):
        body = "Apple unveiled the phones at WWDC in June 2025, the company said.\nNext."
        source = "Apple launched the phones at its Awe Dropping event on September 9, the company said."
        self.assertEqual(check("replace", "at WWDC in June 2025", "at its Awe Dropping event on September 9, 2025",
                               evidence=source, body=body, source=source), ("source_supported", None))
        # a number replacing a non-date fact is still a type mismatch
        self.assertEqual(check("replace", "led by coach John Smith", "led by coach 42", evidence=EVIDENCE),
                         (None, "type_mismatch"))

    def test_markdown_emphasis_in_source_does_not_block_a_quote(self):
        body = "He was a 28-year-old man from Barwala, the paper said."
        source = "The deceased was identified as Kapil, a 26-year-old man from Brah Kalan, the _Hindustan Times_ reported."
        quote = "The deceased was identified as Kapil, a 26-year-old man from Brah Kalan, the Hindustan Times reported."
        self.assertEqual(check("replace", "a 28-year-old man", "a 26-year-old man", evidence=quote, body=body,
                               source=source), ("source_supported", None))
        self.assertEqual(check("replace", "a 28-year-old man", "a 26-year-old man", evidence=quote.replace("Kapil", "Raj"),
                               body=body, source=source), (None, "evidence_not_found"))

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


MAPPING = {a["article_id"]: a for a in json.loads((ROOT / "article_mapping.json").read_text())}


def check_article(article_id, find, new, evidence):
    """check() against a real article, its source and its parsed annotations."""
    read = lambda key: (ROOT / MAPPING[article_id][key]).read_text(encoding="utf-8")
    body, raw = split_annotations(read("ai_generated_file"))
    return check("replace", find, new, evidence=evidence, body=body, source=read("source_file"),
                 annotations=parse_annotations(raw))


# annotation-backed replacements the model proposed for article_004; each copies an editor's note into the article
COMMENTARY_004 = [
    ("with some critics even calling it a light-hearted comedy",
     "the film has been described as a serious/controversial work dealing with sensitive historical events"),
    ("portrayal of Gopal Mukherjee, a former British police officer, rather than Gopal Patha, a revolutionary leader",
     "portrayal of Gopal Patha (also referred to as Gopal Mukherjee in some references), a key figure purportedly "
     "protecting Hindus during the 1946 riots — not a British police officer"),
    ("His grandson praised the depiction and filed a letter of appreciation to producer Karan Johar, instead of "
     "lodging a complaint against director Vivek Ranjan.",
     "His grandson expressed disappointment over the portrayal and filed a complaint against director Vivek "
     "Agnihotri (not a praise letter to Karan Johar, and the director's name is Vivek Agnihotri)."),
    ("The family denied any legal action, and no complaints were recorded.",
     "A complaint was filed by the grandson; the family did not deny legal action in that regard."),
]


class TestAnnotationCommentary(unittest.TestCase):

    SILENT_SOURCE = "Officials confirmed the flight left on time."

    def test_editorial_notes_from_article_004_are_rejected(self):
        for find, new in COMMENTARY_004:
            with self.subTest(find=find[:40]):
                support, reason = check_article("article_004", find, new, evidence=new)
                self.assertIsNone(support)
                self.assertIn(reason, ("annotation_commentary", "annotation_not_minimal"))

    def test_minimal_annotation_corrections_are_still_accepted(self):
        self.assertEqual(check_article("article_004", "only a handful of theaters refusing to screen it",
                                       "most theaters refusing to screen it", evidence="most theaters refusing to screen it"),
                         ("annotation_supported", None))
        self.assertEqual(check_article("article_007", "Afghanistan, who were set an above-par total",
                                       "Afghanistan, who were set a below-par total",
                                       evidence="Afghanistan, who were set a below-par total"),
                         ("annotation_supported", None))
        self.assertEqual(check_article("article_007", "winning the 2024 ICC T20 World Cup",
                                       "reaching the 2024 ICC T20 World Cup semi-final",
                                       evidence="reaching the 2024 ICC T20 World Cup semi-final"),
                         ("annotation_supported", None))

    def test_wrong_name_replaced_by_the_annotated_name(self):
        body = "The detainees flew home on a flight with Asiana Airlines on Friday.\nNext."
        note = {"error": "with Asiana Airlines", "correction": "flew home on a flight with Korean Air"}
        self.assertEqual(check("replace", "with Asiana Airlines", "with Korean Air",
                               evidence="flew home on a flight with Korean Air",
                               body=body, source=self.SILENT_SOURCE, annotations=[note]), ("annotation_supported", None))

    def test_long_explanation_without_editorial_markers_is_rejected(self):
        body = "The team won 3 matches at Lords stadium this spring.\nNext."
        correction = "won 5 matches, which was later confirmed by the league after a lengthy review"
        note = {"error": "won 3 matches", "correction": correction}
        self.assertEqual(check("replace", "won 3 matches", correction, evidence=correction, body=body,
                               source=self.SILENT_SOURCE, annotations=[note]), (None, "annotation_not_minimal"))

    def test_editorial_wording_is_rejected_even_when_short(self):
        body = "The team won 3 matches at Lords stadium this spring.\nNext."
        for correction in ("actually won 5 matches", "won 5 matches (correct)", "won 5 matches; source"):
            with self.subTest(correction=correction):
                note = {"error": "won 3 matches", "correction": correction}
                self.assertEqual(check("replace", "won 3 matches", correction, evidence=correction, body=body,
                                       source=self.SILENT_SOURCE, annotations=[note]), (None, "annotation_commentary"))

    def test_source_supported_edits_are_not_affected(self):
        # parentheses and several new words are fine when the source sentence itself says so
        evidence = ("Global brokerage Bank of America Securities (BofA), in its latest note, raised Maruti Suzuki's "
                    "share price target to ₹17,000 per share from ₹14,000, while maintaining a "
                    "‘buy’ rating on the stock.")
        self.assertEqual(check_article("article_003", "Morgan Stanley", "Bank of America Securities (BofA)", evidence),
                         ("source_supported", None))
        self.assertEqual(check("replace", "won 3 matches", "won 5 matches"), ("source_supported", None))


# the 14 edits from the Phase 4 run that were rejected only because the quote ended with a full stop
# where the source sentence continues (article id, find, replace, evidence as the model sent them)
F1_EDITS = [
    ('article_003', 'from 18% to 12%',
     'from 28% to 18%',
     'With GST rates on affordable cars reduced to 18% from 28%.'),
    ('article_024', 'The director plans to decide which film to shoot next by March 2025, and both Coal Emperor and Gadar 3 are said to have completed scripts.',
     'A call on his immediate next will be taken by the end of this year, but the three scripts are presently being worked upon simultaneously.',
     'A call on his immediate next will be taken by the end of this year, but the three scripts are presently being worked upon simultaneously.'),
    ('article_025', 'It now features a 24MP Fusion main camera and another 12MP ultra-wide lens',
     'It now features a 48MP Fusion main camera and a 48MP ultra-wide lens',
     'For optics, the iPhone 17 comes with a dual rear camera system, headlined by a 48-megapixel Fusion main camera ... Accompanying it is a 48-megapixel Fusion Ultra-Wide camera.'),
    ('article_047', 'to get a 36-hour battery life',
     'to get an all-day ~18-hour battery life',
     'The S10 chip also enables SE 3 to get all-day 18-hour battery life.'),
    ('article_050', 'uniform GST of 28%',
     'uniform GST of 18%',
     'Additionally, all auto components will now carry a uniform GST of 18%.'),
    ('article_057', 'Ghana',
     'Nigeria',
     'The Times, citing UK officials, stated that India, Pakistan, and Nigeria are among the most resistant nations.'),
    ('article_057', '2,070,600 Indians overstaying their visas up to 2021',
     '20,706 Indians overstaying their visas in the year to 2020',
     'India tops the list of visa overstayers, with 20,706 Indians failing to leave in the year to 2020.'),
    ('article_060', 'Emily Blunt, an Oscar winner for Oppenheimer (2023),',
     'Emily Blunt, an Oscar nominee for Oppenheimer (2023),',
     'Emily Blunt, an Oscar nominee for “Oppenheimer” (2023) delivers an effortless turn as Kerr’s girlfriend Dawn Staples.'),
    ('article_063', 'now costs ₹68.1 lakh after a price cut of nearly ₹3.8 lakh',
     'now costs ₹67.1 lakh after a price cut of nearly ₹4.8 lakh',
     'The refreshed XC60 now costs ₹67.1 lakh after a price drop of nearly ₹4.8 lakh.'),
    ('article_065', 'heard on Monday',
     'heard on Tuesday',
     'The matter came up on Tuesday before Justice Tejas Karia.'),
    ('article_065', 'Justice Ritu Karia',
     'Justice Tejas Karia',
     'The matter came up on Tuesday before Justice Tejas Karia.'),
    ('article_098', 'The nine-time champions have',
     'The eight-time champions have',
     'The eight-time Asia Cup champions have been battling the hot and humid conditions in Dubai ahead of their tournament opener against the hosts on September 10.'),
    ('article_098', 'During their last visit to Dubai for the ICC World Cup in 2019, India employed a four-pronged pace attack.',
     'India successfully used a four-pronged spin attack during their last visit to Dubai at the ICC Champions Trophy event earlier this year.',
     'India successfully used a four-pronged spin attack during their last visit to Dubai at the ICC Champions Trophy event earlier this year.'),
    ('article_100', 'up to 40 million mobile phones simultaneously',
     'at least four million mobile phones simultaneously',
     'Pakistan‘s spy agencies can monitor at least 4 million mobile phones at a time through its Lawful Intercept Management System (LIMS).'),
]


class TestQuoteEndingEarly(unittest.TestCase):
    """A quote may stop with a full stop where the source sentence continues; every word must still match."""

    SOURCE = "The matter came up on Tuesday before Justice Tejas Karia, who orally indicated an interim order."
    BODY = "The matter was heard on Monday by Justice Tejas Karia in Delhi.\nNext."

    def edit(self, evidence):
        return check("replace", "heard on Monday", "heard on Tuesday", evidence=evidence, body=self.BODY, source=self.SOURCE)

    def test_edits_from_the_phase_4_run_are_accepted(self):
        self.assertEqual(len(F1_EDITS), 14)
        for article_id, find, new, evidence in F1_EDITS:
            with self.subTest(article=article_id, find=find[:30]):
                self.assertEqual(check_article(article_id, find, new, evidence), ("source_supported", None))

    def test_closing_mark_where_the_sentence_continues(self):
        for evidence in ("The matter came up on Tuesday before Justice Tejas Karia.",
                         "The matter came up on Tuesday before Justice Tejas Karia!",
                         "The matter came up on Tuesday before Justice Tejas Karia?"):
            with self.subTest(evidence=evidence):
                self.assertEqual(self.edit(evidence), ("source_supported", None))

    def test_words_must_still_match_exactly(self):
        for evidence in ("The matter came up on Tuesday before Justice Tejas Kariah.",     # changed word
                         "The matter came up Tuesday before Justice Tejas Karia.",           # dropped word
                         "The case came up on Tuesday before Justice Tejas Karia.",          # replaced word
                         "The matter came up on Tuesday before Justice Tejas Karia, who said so."):  # invented ending
            with self.subTest(evidence=evidence):
                self.assertEqual(self.edit(evidence), (None, "evidence_not_found"))

    def test_only_one_sentence_final_mark_is_ignored(self):
        self.assertEqual(self.edit("The matter came up on Tuesday before Justice Tejas Karia.."), (None, "evidence_not_found"))
        for evidence in ("The matter came up on Tuesday before Justice Tejas Karia;",
                         "The matter came up on Tuesday before Justice Tejas Karia:"):
            with self.subTest(evidence=evidence):
                self.assertEqual(self.edit(evidence), (None, "evidence_not_found"))

    def test_other_rules_still_apply_to_a_trimmed_quote(self):
        source = "The court heard arguments on Tuesday about ticket prices, officials said."
        self.assertEqual(check("replace", "heard on Monday", "heard on Tuesday", body=self.BODY, source=source,
                               evidence="The court heard arguments on Tuesday about ticket prices."),
                         (None, "evidence_not_slot_aligned"))
        self.assertEqual(self.edit("Tuesday before Karia."), (None, "evidence_not_found"))
        self.assertEqual(check("replace", "heard on Monday", "heard on Tuesday", body=self.BODY, source=self.SOURCE,
                               evidence="on Tuesday."), (None, "evidence_too_short"))


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
