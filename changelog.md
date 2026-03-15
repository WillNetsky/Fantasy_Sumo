# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- Fixed `ModuleNotFoundError: No module named 'simulate_tournament_v2'` in `daily_matchup_app.py` by updating the import of `TECHNIQUE_CATEGORIES` to use the root `simulate_tournament_v4.py` instead of the archived `v2` version.
- Verified `daily_matchup_app.py` loads correctly for the March 2026 basho.
