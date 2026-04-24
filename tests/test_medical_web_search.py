"""Tests for the MEDICAL_WEB_SEARCH_TOOL tool definition.

The tool is deliberately domain-restricted to authoritative medical sources
so the bot cannot cite random wellness blogs when a user with a psychiatric
diagnosis asks about their medication. This locks in the contract."""

from __future__ import annotations

from insult.core.llm import MEDICAL_WEB_SEARCH_TOOL, WEB_SEARCH_TOOL


class TestMedicalWebSearchTool:
    def test_shape_matches_anthropic_web_search_tool_v1(self):
        """Structural sanity: the tool must have the type+name+max_uses Anthropic expects."""
        assert MEDICAL_WEB_SEARCH_TOOL["type"] == "web_search_20250305"
        assert MEDICAL_WEB_SEARCH_TOOL["name"] == "web_search"
        assert isinstance(MEDICAL_WEB_SEARCH_TOOL["max_uses"], int)
        assert MEDICAL_WEB_SEARCH_TOOL["max_uses"] > 0

    def test_allowed_domains_includes_nih_and_aemps(self):
        """Lock in the 'trusted sources' set. If any of these are removed,
        the bot can now cite something less authoritative — red flag."""
        domains = MEDICAL_WEB_SEARCH_TOOL["allowed_domains"]
        # MUST include these — they are the canonical Spanish-language
        # patient-info sources plus NIH/WHO.
        for required in ("medlineplus.gov", "cima.aemps.es", "nih.gov", "who.int"):
            assert required in domains, f"missing required source: {required}"

    def test_no_blocked_domains_configured(self):
        """We rely on allowlist, not blocklist. A blocklist would be a red
        flag (implies we're trying to filter general results rather than
        constrain to trusted sources)."""
        assert "blocked_domains" not in MEDICAL_WEB_SEARCH_TOOL

    def test_user_location_mexico(self):
        """Localized to Mexico so MedlinePlus Spanish + Mexican resources
        surface first. If this changes, the bot may start surfacing
        US-only hotline numbers to Mexican users."""
        loc = MEDICAL_WEB_SEARCH_TOOL["user_location"]
        assert loc["country"] == "MX"
        assert loc["timezone"] == "America/Mexico_City"

    def test_differs_from_general_tool(self):
        """The general WEB_SEARCH_TOOL must NOT have the medical allowlist —
        otherwise non-medical searches would be starved of results."""
        assert "allowed_domains" not in WEB_SEARCH_TOOL
        assert MEDICAL_WEB_SEARCH_TOOL != WEB_SEARCH_TOOL
