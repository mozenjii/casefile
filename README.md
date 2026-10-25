# casefile

A workflow for reviewing healthcare authorization cases with AI helpers. It survives crashes, stays inside a budget, and never releases a decision without a human sign-off.

> **Status:** early, and working end to end on synthetic cases: generated case → specialists → rules → dossier → review screen → approval → release, on a durable event log. Out of the box the specialists run on a deterministic offline stand-in, so nothing here measures model quality yet. Real models plug in with two environment variables (see [Using a real model](docs/MODELS.md)). Budgets, fault injection on the full workflow and the baseline comparison come next. A result appears in the table below only once a script in this repo produces it. All cases are synthetic; no real patient or claim data is used anywhere in this repo.

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
| Correct resume after a kill at every step | Crash matrix (`scripts/crash_matrix.py`) | 15 of 15 crash points on the four-step toy case; the full workflow is not built yet |
| Runs stopped at the budget with a recorded reason | Budget exhaustion tests | not measured |
| Quality per dollar: multi-agent vs single agent vs fixed workflow | Same synthetic cases | not measured |
| Any path to release without approval | Property tests over random event orders (`tests/test_release_properties.py`) | none found in 300 generated sequences of up to 25 review events per run |

## Design notes

**The log is the state.** Every step appends an event, and a case's state is a fold over its events. Recovery is replay. Model outputs are recorded in the log, so replaying a case doesn't call the model again and gives the same result. The one exception can't be designed away: a call that returned but hadn't been recorded when the process died gets made again. The crash matrix counts those repeats instead of hiding them.

**Each event carries the hash of the one before it.** Edit one byte of a stored event, drop one, or reorder them, and the verifier names the first event where the chain breaks. The table also refuses `UPDATE` and `DELETE` at the database level.

**Crash for real.** The crash matrix runs the case once to list every point where a crash would leave work half-done. Then, for each point, it starts a fresh process, kills it there with `os._exit` (no cleanup, no flush), resumes it in a new process, and compares the log, the effects at the destination and the hash chain with the uninterrupted run.

**Record the side effect before performing it.** An outbox entry is written first and executed after, with an idempotency key, so a retry after a crash can't send the same request for information twice.

**An approval is bound to exactly what the reviewer saw.** The reviewer is shown the dossier along with its hash, and their decision records that hash. Release is allowed only when the latest decision on the *current* dossier's hash is an approval. If someone edits the dossier after it was approved, the hash changes and the old approval simply stops matching. There's no "reset the approval" code to forget on some new edit path. The rule lives in the fold that every read of a case goes through, and every event is checked against it before it's written, so a log that releases without a matching approval can't exist. A property test throws random revisions, stale and made-up decisions, and release attempts at it, and checks the result against a separately written statement of the rule.

**Synthetic data that's provably fake.** A scanner that flags every nine-digit number gets switched off within a week. This one knows the ranges that are never issued: SSNs with area 000 or 666, phone numbers 555-0100 to 555-0199, the example.com domains, the card networks' test numbers. The build fails on any identifier in the repo that falls outside them, and the failure names the file and line but never the value. On its first run it caught a real-looking phone number in one of my own tests.

**Unknown is not a pass.** The criteria for the made-up procedure (an MRI of one knee; invented, not clinical guidance) are evaluated with four outcomes: met, not met, unknown when nobody documented a fact, and conflict when two documents disagree. Unknowns flow through each criterion, and "a red flag *or* six weeks of therapy" uses three-valued logic, so a documented red flag settles it even when therapy is unknown. The recommendation is the most severe outcome: nothing unknown gets approved, and a definite deny isn't softened by a missing document elsewhere. The model never decides. It reads and writes; the recommendation comes from the rules.

**Ground truth by construction, checked twice.** A generator turns structured facts into a request form, a clinical note, an X-ray report and a referral letter, and records the exact character span where each fact is written. Every case carries one defect (a missing or stale X-ray, a missing referral, two documents disagreeing about therapy, a red flag, a recent prior MRI, an instruction planted in a note telling the system to approve) and declares the outcome that defect should produce. The generator refuses any case where the rules engine's label disagrees with that, so a bug in either shows up immediately.

**Any model, same guarantees.** The specialists talk to models through one interface. Behind it are the stand-in, Claude through its official SDK (structured JSON output, a server-side fallback when a request is refused), and any OpenAI-compatible endpoint, which covers the hosts that serve open-weights models cheaply. Every answer is recorded with its token count and cost in the case's log, so a crash never pays twice, and answers are cached so rerunning the same cases is free. All of this is tested through the real SDKs with the network replaced, so no key is ever needed to run the tests.

**Every fact must point at its source.** When the extractor reports a fact, it has to give the character span it read. The grounding check re-reads that span independently and keeps the fact only if those words really state the value. A yes/no fact also has to sit right after its field's label, since almost any words could be read as "yes". Ungrounded facts are rejected with a reason and shown to the reviewer, and the document gets asked about once more. On the review screen every criterion links to the highlighted spans it rests on.

**Honest baselines.** More agents isn't automatically better. If a fixed workflow matches the agentic version at lower cost, the write-up says so.

## Running it

Needs Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/check.py        # format, lint, types, tests with coverage
uv run python scripts/crash_matrix.py # the crash matrix as a table
uv run python scripts/demo.py         # generate a case per defect, review them at http://127.0.0.1:8000
```

**Using a real model.** Set `CASEFILE_PROVIDER` and a key (copy `.env.example` to `.env`). OpenRouter with an open-weights model is the recommended default, at roughly $0.0006 per case; Claude Opus 5.5 is the high-quality option; any OpenAI-compatible server works. `uv run python scripts/estimate_cost.py` prices a run before anything is sent. Details in [docs/MODELS.md](docs/MODELS.md).

The browser tests need Chromium once (`uv run playwright install chromium`), then `uv run python scripts/check.py --browser`.

![The review screen: the dossier's criteria on the left, the source documents with highlighted evidence in the middle, the decision and the case's timeline on the right.](docs/review-screen.png)

## Roadmap

- [x] Case state machine, event log and crash-resume tests
- [x] Approval gates and audit trail
- [x] Synthetic case generator with a rules engine
- [x] Supervisor and specialists, review screen and timeline
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
