import numpy as np
import pytest

from shiftwatch.shifts import (SHIFTS, abbreviate, add_typos, apply_shift,
                               mix_out_of_scope, style_shift)


class TestTypos:
    def test_deterministic_for_same_seed(self):
        t = "please check my account balance today"
        assert add_typos(t, 0.2, seed=7) == add_typos(t, 0.2, seed=7)

    def test_different_seeds_differ(self):
        t = "please check my account balance today and send a statement"
        a = add_typos(t, 0.3, seed=1)
        b = add_typos(t, 0.3, seed=2)
        assert a != b

    def test_severity_zero_is_identity(self):
        t = "check my balance"
        assert add_typos(t, 0.0) == t

    def test_more_severity_means_more_changes(self):
        t = "check my account balance and send the statement please now"
        n = lambda s: sum(c1 != c2 for c1, c2 in zip(t, add_typos(t, s, seed=3)))
        assert n(0.2) >= n(0.05)

    def test_preserves_word_count(self):
        t = "transfer the payment to my account today"
        assert len(add_typos(t, 0.3, seed=5).split()) == len(t.split())

    def test_short_words_untouched(self):
        # 3-letter words are skipped, so "the" survives at high severity
        out = add_typos("the cat sat on a mat", 1.0, seed=2)
        assert "the" in out.split()


class TestAbbreviate:
    def test_known_word_shrinks(self):
        out = abbreviate("my account balance is low", 1.0, seed=1)
        assert len(out.split()) == len("my account balance is low".split())

    def test_severity_zero_identity(self):
        t = "my account balance"
        assert abbreviate(t, 0.0) == t

    def test_punctuation_preserved_on_keyword(self):
        out = abbreviate("balance, please", 1.0, seed=4)
        assert "," in out

    def test_unknown_text_unchanged(self):
        t = "zzyzx quux frobnicate"
        assert abbreviate(t, 1.0, seed=1) == t


class TestStyle:
    def test_severity_one_lowercases_and_strips_punctuation(self):
        out = style_shift("Hello, world! Thanks.", 1, seed=1)
        assert out == out.lower()
        assert "," not in out and "!" not in out

    def test_severity_three_can_uppercase(self):
        assert any(style_shift("hello there", 3, seed=s).isupper() for s in range(20))

    def test_zero_is_identity(self):
        t = "Hello, world!"
        assert style_shift(t, 0) == t


class TestOutOfScope:
    def test_zero_ratio_injects_nothing(self):
        texts, gold = mix_out_of_scope(["a", "b", "c"], ["x"], 0.0)
        assert texts == ["a", "b", "c"]
        assert gold == [None, None, None]

    def test_ratio_injects_expected_count(self):
        texts, gold = mix_out_of_scope([f"t{i}" for i in range(100)], ["oos"], 0.4, seed=1)
        injected = sum(1 for t in texts if t == "oos")
        assert injected == 40

    def test_injection_deterministic(self):
        a, _ = mix_out_of_scope([f"t{i}" for i in range(50)], ["x", "y"], 0.2, seed=3)
        b, _ = mix_out_of_scope([f"t{i}" for i in range(50)], ["x", "y"], 0.2, seed=3)
        assert a == b

    def test_ratio_one_replaces_everything(self):
        texts, _ = mix_out_of_scope(["a", "b"], ["oos"], 1.0)
        assert texts == ["oos", "oos"]


class TestApplyShift:
    @pytest.mark.parametrize("name,sev", [(n, s) for n, s, _ in SHIFTS])
    def test_every_ladder_rung_applies(self, name, sev):
        out = apply_shift(["check my account balance please"], name, sev, seed=0)
        assert len(out) == 1 and isinstance(out[0], str)

    def test_unknown_shift_raises(self):
        with pytest.raises(KeyError):
            apply_shift(["x"], "nonexistent", 1)