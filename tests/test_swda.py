from dictum.swda import disfluent_and_clean


def test_repair_and_fillers():
    d, c = disfluent_and_clean("[ Sometimes, + {F um, }  usually  ] the reason I will turn it on is to hear the news. /")
    assert d.startswith("Sometimes, um, usually the reason")
    assert c == "usually the reason I will turn it on is to hear the news."


def test_nested_repair_and_discourse():
    d, c = disfluent_and_clean("<Laughter> [ [ I, + I, ] + I ] ended up watching a lot of these things on, {D you know, } repeats in the afternoons or something. /")
    assert "Laughter" not in d and d.startswith("I, I, I ended up")
    assert c == "I ended up watching a lot of these things on repeats in the afternoons or something."


def test_abandoned_repair_and_conjunction():
    d, c = disfluent_and_clean("<Swallowing> {C But, } {F uh, } I like to have them [ where + ] <child_crying> handy. /")
    assert c == "But, I like to have them handy."
    assert "where" in d and "uh" in d


def test_partial_word():
    d, c = disfluent_and_clean("how about [ nucle-, + nuclear ] energy.  /")
    assert c == "how about nuclear energy."
