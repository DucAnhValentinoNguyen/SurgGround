from surgground.data.regime import pick


def test_short():
    assert pick(20 * 60) == "short_singlepass"


def test_long_hier_for_general_query():
    assert pick(80 * 60, "when is the gallbladder dissected?") == "long_hier"


def test_long_retrieve_for_object_query_or_very_long():
    assert pick(80 * 60, "when is the stapler first fired?") == "long_retrieve"
    assert pick(150 * 60, "when does phase 3 start?") == "long_retrieve"
