# casefile

A workflow for reviewing healthcare authorization cases with AI helpers. It survives crashes, stays inside a budget, and never releases a decision without a human sign-off.

> **Status:** early. I'm at the design stage and there is no released code yet. The results table below stays empty until a script in this repo produces the numbers. All cases are synthetic; no real patient or claim data is used anywhere in this repo.

## The problem

Reviewing a prior authorization request means pulling facts out of documents, checking them against policy criteria, spotting what's missing and drafting a recommendation. Agent demos show this on the happy path. They skip what happens when the process dies halfway, when the budget runs out, when a tool call gets retried, and who actually makes the decision.

## What it will do

- Run each case as an event-sourced state machine, so a case resumes after a crash exactly where it stopped.
- Coordinate a supervisor and specialists (extract facts, check policy, find gaps) that hand off typed, validated output.
- Enforce a per-case budget in tokens, money, steps and time, and stop with a recorded reason instead of spiralling.
- Gate release on a recorded human approval, with a tamper-evident audit trail of what the reviewer saw.
- Show a timeline of every step with its inputs, outputs and cost.

## How I plan to know it works

Kill the process at every step boundary and check that the resumed case ends up identical to an uninterrupted run. Run property tests that try to reach "released" without an approval. And do the comparison most demos skip: the multi-agent design against a single agent and against a plain fixed workflow, on the same synthetic cases, measured in quality per dollar.

| Measurement | How | Result |
|---|---|---|
| Correct resume after a kill at every step | Crash matrix | not measured |
| Runs stopped at the budget with a recorded reason | Budget exhaustion tests | not measured |
| Quality per dollar: multi-agent vs single agent vs fixed workflow | Same synthetic cases | not measured |
| Any path to release without approval | Property tests over random event orders | not measured |

## Design notes

**The log is the state.** Every step appends an event, and a case's state is a fold over its events. Recovery is replay. Model outputs are recorded in the log, so replaying a case doesn't call the model again and gives the same result.

**Record the side effect before performing it.** An outbox entry is written first and executed after, with an idempotency key, so a retry after a crash can't send the same request for information twice.

**Ground truth by construction.** A small rules engine defines the policy criteria for a made-up procedure. A generator writes the clinical documents with controlled gaps and contradictions, and the correct recommendation is computed from the rules, so every case comes with a label.

**Honest baselines.** More agents isn't automatically better. If a fixed workflow matches the agentic version at lower cost, the write-up says so.

## Roadmap

- [ ] Case state machine, event log and crash-resume tests
- [ ] Approval gates and audit trail
- [ ] Synthetic case generator with a rules engine
- [ ] Supervisor and specialists, review screen and timeline
- [ ] Fault injection and baseline comparison
- [ ] Write-up

## Related projects

Part of a set of six projects about building LLM systems you can trust:
[evalgate](https://github.com/mozenjii/evalgate),
[policylens](https://github.com/mozenjii/policylens),
[docproof](https://github.com/mozenjii/docproof),
[tollgate](https://github.com/mozenjii/tollgate),
[redline](https://github.com/mozenjii/redline),
[casefile](https://github.com/mozenjii/casefile).

Built by Mohib Ahmad.
