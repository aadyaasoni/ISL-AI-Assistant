from web.server import avatar_playback


def test_mapped_gloss_resolves_to_a_browser_clip():
    result = avatar_playback(["1. loud"])

    assert result["status"] == "ready"
    assert result["glosses"] == ["1. loud"]
    assert result["clips"] == ["/avatar/clips/1._loud.json"]
    assert result["unmatched"] == []


def test_unsupported_gloss_is_reported_without_failing():
    result = avatar_playback(["please repeat"])

    assert result == {
        "status": "no_clip",
        "glosses": ["please repeat"],
        "clips": [],
        "unmatched": ["please repeat"],
    }
