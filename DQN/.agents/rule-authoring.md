<!--
Last modified time: 2026-09-21-02:01
Last modified content: Define how to extend mini-linter rules in the DQN project
Last modified by: OpenAI Codex
File design: Lint extension guidance
File purpose: Require narrowly scoped and tested future quality rules
File creator: OpenAI Codex
-->

# Rule Authoring

No custom mini-linter plugins are needed for stage 1. A future rule must have one clear responsibility, an actionable message and hint, and both passing and failing tests. Do not weaken the existing warning-as-failure gate to accommodate new learner code.
