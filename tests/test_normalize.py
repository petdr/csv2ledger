"""Normalisation rules, pinned against real descriptions from the bank CSVs.

Two failure directions matter and both are tested:
  * under-stripping, where variants of one merchant stay distinct (the whole problem)
  * over-stripping, where distinct merchants collapse into one (silently wrong)
"""

from csv2ledger.normalize import from_regex, normalize


class TestCollapsesVariants:
    """Variants of the same merchant must reach the same key."""

    def test_store_numbers_collapse(self):
        a = normalize("7-ELEVEN 1148            NORTH MELBOURVI")
        b = normalize("7-ELEVEN 3092            NORTH MELBOURVI")
        assert a == b

    def test_ing_receipt_numbers_collapse(self):
        a = normalize("Cleaning Katrien - Transfer to Lavanda Cleaning - Receipt 623295 To 063113 11267704")
        b = normalize("Cleaning Katrien - Transfer to Lavanda Cleaning - Receipt 343379 To 083543 404203832")
        assert a == b

    def test_card_and_timestamp_noise_collapses(self):
        a = normalize(
            "BUZA CHICKEN QUEEN STR - Visa Purchase - Receipt 172217In MELBOURNE "
            "Date 25 Oct 2025 Card 420274xxxxxx4526"
        )
        b = normalize(
            "BUZA CHICKEN QUEEN STR - Visa Purchase - Receipt 999001In MELBOURNE "
            "Date 03 Nov 2025 Card 420274xxxxxx4526"
        )
        assert a == b

    def test_aldi_cashout_variants_collapse(self):
        a = normalize(
            "ALDI STORES           WES T MELBOURNVI - EFTPOS Purchase with Cash Out - "
            "Receipt 98988 - Cash amount $50.00<BR/>Date 23 Jul 2026 Time 5:52PM "
            "Card 462263xxxxxx2739"
        )
        b = normalize(
            "ALDI STORES           WES T MELBOURNVI - EFTPOS Purchase with Cash Out - "
            "Receipt 11111 - Cash amount $20.00<BR/>Date 04 Jan 2026 Time 9:01AM "
            "Card 462263xxxxxx2739"
        )
        assert a == b
        assert "ALDI STORES" in a

    def test_foreign_currency_suffix_removed(self):
        a = normalize("FRANPRIX 4145            PARIS        FR21.10 EUR")
        b = normalize("FRANPRIX 4145            PARIS        FR9.80 EUR")
        assert a == b

    def test_receipt_without_separator_keeps_following_word(self):
        # "Receipt 132547Data Processors" -- the bank omits the space.
        result = normalize("Salary - Salary Deposit - Receipt 132547Data Processors  ADS116154   000213")
        assert "DATA PROCESSORS" in result


class TestPreservesDistinctions:
    """Merchants that differ must stay different."""

    def test_different_merchants_stay_distinct(self):
        assert normalize("COLES                    MELBOURNE    VI") != normalize(
            "WOOLWORTHS               MELBOURNE    VI"
        )

    def test_same_chain_different_purpose_stays_distinct(self):
        assert normalize("ALDI STORES WEST MELBOURNE") != normalize(
            "ALDI STORES WEST MELBOURNE - EFTPOS Purchase with Cash Out"
        )

    def test_transfer_directions_stay_distinct(self):
        assert normalize("EVERYDAY TO MAX") != normalize("MAX TO EVERYDAY")

    def test_paypal_merchants_stay_distinct(self):
        assert normalize("PAYPAL *WOTIF.COM        4029357733   AU") != normalize(
            "PAYPAL *YAHOO INC        4029357733   CA"
        )

    def test_merchant_name_survives(self):
        assert "SPICETEMPLE" in normalize("SPICETEMPLE MELBOURNE    Melbourne    VI")


class TestMechanics:
    def test_idempotent(self):
        raw = "QUEEN VIC MARKET/THERRY S TRETHERRY ST - Receipt 378782ATM owner fee of $0.00"
        once = normalize(raw)
        assert normalize(once) == once

    def test_empty_input(self):
        assert normalize("") == ""
        assert normalize("   ") == ""

    def test_whitespace_collapsed_and_uppercased(self):
        assert normalize("  Coles   Express  ") == "COLES EXPRESS"

    def test_no_runs_of_hash_placeholders(self):
        assert "# #" not in normalize("To 063113 11267704 12345 999")


class TestFromRegex:
    def test_strips_leading_wildcard(self):
        assert from_regex("/.*KATHMANDU PTY LIMITE/") == "KATHMANDU PTY LIMITE"

    def test_strips_slashes_only_pattern(self):
        assert from_regex("/BUNNINGS/") == "BUNNINGS"

    def test_strips_character_classes_and_quantifiers(self):
        result = from_regex("/Internal Transfer - Receipt [0-9]* - From Orange Everyday/")
        assert "ORANGE EVERYDAY" in result
        assert "[" not in result and "]" not in result and "*" not in result

    def test_plain_literal_passes_through(self):
        assert from_regex("Woolworths Gift Card") == "WOOLWORTHS GIFT CARD"
