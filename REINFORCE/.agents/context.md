<!--
Last modified time: 2026-09-24
Last modified content: Consolidate problem documentation and reproducible validation
Last modified by: OpenAI Codex
File design: Sequencing documentation
File purpose: Describe verified behavior, operation, and research boundaries
File creator: OpenAI Codex
-->

# Current project context

`src/distillation_reinforce/` contains the independent learner. The sibling `distillation-sequencing-env` owns textbook chemistry, deterministic transitions, shared evaluation records and graph traversal. Existing chemistry import paths are compatibility exports. No other learner is a runtime dependency.

The current implementation is fixed at four components. See `README.md` and `../docs/problem.md` for the authoritative current behavior, `../docs/component-scaling.md` for the unimplemented research plan, and `.agents/reports/` for historical verification evidence.

Maintain algorithm-specific update definitions and fixed-seed behavior. Focused tests, full pytest with coverage >=90%, and mini-linter with zero warnings are required before updating a stage report. Shared chemistry tests run separately; compatibility and learner tests remain here.
