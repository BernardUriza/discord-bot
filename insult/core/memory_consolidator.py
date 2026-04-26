"""Mem0-style user-fact consolidator.

After hundreds of conversation turns, the auto-extracted facts pile up
with overlap, contradictions, and stale entries. The fact-extraction
LLM in :mod:`insult.core.facts` runs per-turn and never looks at the
*global* picture; it can't notice that "Vive en México" and "Está
viviendo en CDMX" are the same fact at different granularity, or that
"Trabaja en X" was superseded six weeks ago by "Renunció a X".

This module runs OUT-OF-BAND (scheduled task, every 2 days) and:

1. Loads the live (``deleted_at IS NULL``) facts for one user.
2. Asks the summary model (Haiku) for an ADD/UPDATE/DELETE/NOOP plan
   over the entire fact set in a single call. The model sees the
   whole snapshot, not pair-wise comparisons, so it catches global
   redundancy.
3. Applies the plan transactionally:
   - DELETE → soft-delete (``deleted_at = now()``); recoverable for
     90 days, then hard-purged.
   - UPDATE → soft-delete the original AND insert the merged text.
   - ADD/NOOP → no DB writes (ADD is the extraction LLM's job).
4. Logs every decision in ``fact_consolidation_log`` for audit.

Reference: Mem0 paper (arxiv 2504.19413). The big simplification vs.
the paper: we use one LLM call over the full set instead of pair-wise
extract→resolve passes against a vector store. With ~80 facts/user
the prompt fits comfortably in Haiku's context window and one call is
~30x cheaper than N**2 pair comparisons.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import anthropic
import structlog

if TYPE_CHECKING:
    from insult.core.memory import MemoryStore

log = structlog.get_logger()

# 90 days in seconds — retention window for soft-deleted facts before
# the hard-purge query removes them. Tuned with operations: 90d is long
# enough that a misclassified DELETE is almost always caught by the
# next consolidation run plus operator review, but short enough that
# the table doesn't accumulate forever.
SOFT_DELETE_RETENTION_SECONDS = 90 * 86400


JUDGE_PROMPT = """\
You are a memory curator for a long-term assistant. You will be given the \
COMPLETE current set of stored facts about a single user. Some are \
duplicates, some are stale, some contradict newer entries. Your job is \
to produce a curation plan as JSON.

Each input fact has an integer "id" you must reference verbatim in your output.

For each fact, decide ONE of:
- "NOOP"   — keep as-is, no overlap with others
- "DELETE" — remove (duplicate of a kept fact, contradicted by newer fact, \
  or no longer accurate)
- "UPDATE" — supersede this fact with merged or corrected text. Use when \
  two or more facts cover the same topic and you want to fold them into \
  one cleaner sentence.

Hard rules:
- Treat the LATEST `updated_at` as authoritative when two facts conflict. \
  The newer one wins; the older one is DELETEd.
- Never DELETE a fact unless another fact in the input set covers the same \
  ground OR the fact is plainly contradicted. If a fact stands alone, NOOP it.
- An UPDATE consumes one or more original ids and produces ONE new merged \
  fact text. List the consumed ids in "merge_ids".
- Preserve language: if the original facts are in Spanish, the merged text \
  must be in Spanish. Same for English.
- Preserve concrete detail. Dates, place names, numbers, names of people \
  must survive. Don't generalize "el 20 abril 2026 lo catearon en aduana" \
  into "tuvo un problema en la frontera".
- Keep merged facts under 25 words.

Return ONLY a JSON array. Each element is an operation object:

  {"op": "NOOP",   "id": 12, "reason": "standalone fact"}
  {"op": "DELETE", "id": 17, "reason": "duplicate of id=12"}
  {"op": "UPDATE", "merge_ids": [3, 8], "new_fact": "...", "category": "...", "reason": "..."}

Every input fact id MUST appear in exactly one operation (as `id` for NOOP/DELETE \
or inside `merge_ids` for UPDATE). Do not invent new facts that aren't a merge of \
existing ones.

Return the JSON array and nothing else."""


@dataclass
class FactOperation:
    """One row to write to fact_consolidation_log + apply to user_facts."""

    op: str  # NOOP | DELETE | UPDATE | ADD
    fact_id_before: int | None = None
    fact_id_after: int | None = None
    fact_text_before: str | None = None
    fact_text_after: str | None = None
    reason: str = ""


@dataclass
class ConsolidationReport:
    """Summary of one consolidation run for a single user."""

    user_id: str
    facts_in: int
    facts_out: int
    ops: list[FactOperation] = field(default_factory=list)
    duration_ms: int = 0
    haiku_input_tokens: int = 0
    haiku_output_tokens: int = 0
    error: str | None = None

    def counts_by_op(self) -> dict[str, int]:
        out = {"NOOP": 0, "DELETE": 0, "UPDATE": 0, "ADD": 0}
        for o in self.ops:
            out[o.op] = out.get(o.op, 0) + 1
        return out


def _build_user_prompt(facts: list[dict]) -> str:
    """Render the user's fact set as a numbered list for the judge."""
    lines = []
    for f in facts:
        # updated_at is unix-seconds; format as ISO date for the model
        # so age comparisons are easier to reason about than raw floats.
        ts = f.get("updated_at") or 0
        when = time.strftime("%Y-%m-%d", time.gmtime(ts)) if ts else "unknown"
        lines.append(
            f'{{"id": {f["id"]}, "fact": {json.dumps(f["fact"], ensure_ascii=False)}, '
            f'"category": "{f.get("category", "general")}", "updated_at": "{when}"}}'
        )
    return "Current facts:\n[\n  " + ",\n  ".join(lines) + "\n]"


def _parse_judge_response(raw: str) -> list[dict] | None:
    """Strip markdown fences, parse JSON, return list[dict] or None on error."""
    raw = raw.strip()
    if raw.startswith("```"):
        # Markdown fence: drop opening fence + optional language tag, drop closing.
        raw = raw.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        plan = json.loads(raw)
    except json.JSONDecodeError as e:
        log.warning("consolidator_judge_parse_failed", error=str(e), raw=raw[:200])
        return None
    if not isinstance(plan, list):
        log.warning("consolidator_judge_not_array", raw=raw[:200])
        return None
    return plan


def _validate_plan(plan: list[dict], facts: list[dict]) -> list[dict]:
    """Drop malformed ops and warn. Every input fact must be referenced exactly once."""
    valid_ids = {f["id"] for f in facts}
    seen: set[int] = set()
    valid_ops: list[dict] = []
    for op in plan:
        if not isinstance(op, dict) or "op" not in op:
            continue
        kind = op["op"]
        if kind in ("NOOP", "DELETE"):
            fid = op.get("id")
            if fid not in valid_ids or fid in seen:
                continue
            seen.add(fid)
            valid_ops.append(op)
        elif kind == "UPDATE":
            ids = op.get("merge_ids") or []
            new_text = op.get("new_fact")
            if not isinstance(ids, list) or not new_text:
                continue
            ids_int = [i for i in ids if isinstance(i, int) and i in valid_ids and i not in seen]
            if not ids_int:
                continue
            seen.update(ids_int)
            op["merge_ids"] = ids_int
            valid_ops.append(op)
    # Any fact id the judge ignored gets an implicit NOOP — never silently lose a row.
    for f in facts:
        if f["id"] not in seen:
            valid_ops.append({"op": "NOOP", "id": f["id"], "reason": "implicit (judge omitted)"})
    return valid_ops


async def _call_judge(
    client: anthropic.AsyncAnthropic,
    model: str,
    facts: list[dict],
) -> tuple[list[dict] | None, int, int]:
    """Single Haiku call. Returns (plan, input_tokens, output_tokens)."""
    user_prompt = _build_user_prompt(facts)
    try:
        response = await client.messages.create(
            model=model,
            max_tokens=2048,
            system=JUDGE_PROMPT,
            messages=[{"role": "user", "content": user_prompt}],
        )
    except (anthropic.APIError, anthropic.APIConnectionError) as e:
        log.warning("consolidator_judge_call_failed", error=str(e))
        return None, 0, 0
    raw = response.content[0].text if response.content else ""
    plan = _parse_judge_response(raw)
    return plan, response.usage.input_tokens, response.usage.output_tokens


async def _apply_plan(
    db,
    user_id: str,
    facts: list[dict],
    plan: list[dict],
    run_ts: float,
) -> list[FactOperation]:
    """Translate the judge's plan to SQL ops + audit log entries."""
    by_id = {f["id"]: f for f in facts}
    applied: list[FactOperation] = []

    for op in plan:
        kind = op["op"]
        reason = op.get("reason", "")[:500]

        if kind == "NOOP":
            fid = op["id"]
            applied.append(
                FactOperation(
                    op="NOOP",
                    fact_id_before=fid,
                    fact_id_after=fid,
                    fact_text_before=by_id[fid]["fact"],
                    fact_text_after=by_id[fid]["fact"],
                    reason=reason,
                )
            )
            continue

        if kind == "DELETE":
            fid = op["id"]
            await db.execute(
                "UPDATE user_facts SET deleted_at = ? WHERE id = ?",
                (run_ts, fid),
            )
            applied.append(
                FactOperation(
                    op="DELETE",
                    fact_id_before=fid,
                    fact_id_after=None,
                    fact_text_before=by_id[fid]["fact"],
                    fact_text_after=None,
                    reason=reason,
                )
            )
            continue

        if kind == "UPDATE":
            ids = op["merge_ids"]
            new_text = op["new_fact"]
            category = op.get("category", "general")
            # Soft-delete originals
            for fid in ids:
                await db.execute(
                    "UPDATE user_facts SET deleted_at = ? WHERE id = ?",
                    (run_ts, fid),
                )
            # Insert merged
            cursor = await db.execute(
                "INSERT INTO user_facts (user_id, fact, category, updated_at, source) VALUES (?, ?, ?, ?, 'auto')",
                (user_id, new_text, category, run_ts),
            )
            new_id = cursor.lastrowid or 0
            for fid in ids:
                applied.append(
                    FactOperation(
                        op="UPDATE",
                        fact_id_before=fid,
                        fact_id_after=new_id,
                        fact_text_before=by_id[fid]["fact"],
                        fact_text_after=new_text,
                        reason=reason,
                    )
                )

    # Write audit rows
    for o in applied:
        await db.execute(
            "INSERT INTO fact_consolidation_log "
            "(run_ts, user_id, fact_id_before, fact_id_after, op, reason, "
            "fact_text_before, fact_text_after) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                run_ts,
                user_id,
                o.fact_id_before,
                o.fact_id_after,
                o.op,
                o.reason,
                o.fact_text_before,
                o.fact_text_after,
            ),
        )
    return applied


async def consolidate_user_facts(
    user_id: str,
    *,
    memory: MemoryStore,
    llm_client: anthropic.AsyncAnthropic,
    model: str,
    dry_run: bool = False,
) -> ConsolidationReport:
    """Run one consolidation pass for a single user. See module docstring."""
    started = time.monotonic()
    run_ts = time.time()
    facts = await memory.get_facts(user_id)

    report = ConsolidationReport(user_id=user_id, facts_in=len(facts), facts_out=len(facts))

    if not facts:
        report.duration_ms = int((time.monotonic() - started) * 1000)
        log.info("consolidator_user_skipped", user_id=user_id, reason="no_facts")
        return report

    if len(facts) < 3:
        # No meaningful overlap to detect — skip the LLM call.
        report.duration_ms = int((time.monotonic() - started) * 1000)
        log.info("consolidator_user_skipped", user_id=user_id, reason="below_min_facts", facts=len(facts))
        return report

    plan, in_toks, out_toks = await _call_judge(llm_client, model, facts)
    report.haiku_input_tokens = in_toks
    report.haiku_output_tokens = out_toks
    if plan is None:
        report.error = "judge_failed"
        report.duration_ms = int((time.monotonic() - started) * 1000)
        log.warning("consolidator_user_failed", user_id=user_id, reason="judge_failed")
        return report

    valid_plan = _validate_plan(plan, facts)

    if dry_run:
        # Build the report without touching the DB.
        by_id = {f["id"]: f for f in facts}
        for op in valid_plan:
            kind = op["op"]
            if kind == "NOOP":
                report.ops.append(
                    FactOperation(
                        op="NOOP",
                        fact_id_before=op["id"],
                        fact_id_after=op["id"],
                        fact_text_before=by_id[op["id"]]["fact"],
                        fact_text_after=by_id[op["id"]]["fact"],
                        reason=op.get("reason", ""),
                    )
                )
            elif kind == "DELETE":
                report.ops.append(
                    FactOperation(
                        op="DELETE",
                        fact_id_before=op["id"],
                        fact_text_before=by_id[op["id"]]["fact"],
                        reason=op.get("reason", ""),
                    )
                )
            elif kind == "UPDATE":
                for fid in op["merge_ids"]:
                    report.ops.append(
                        FactOperation(
                            op="UPDATE",
                            fact_id_before=fid,
                            fact_text_before=by_id[fid]["fact"],
                            fact_text_after=op["new_fact"],
                            reason=op.get("reason", ""),
                        )
                    )
        report.facts_out = sum(1 for o in report.ops if o.op == "NOOP") + sum(
            1 for op in valid_plan if op["op"] == "UPDATE"
        )
        report.duration_ms = int((time.monotonic() - started) * 1000)
        log.info(
            "consolidator_user_dry_run",
            user_id=user_id,
            facts_in=report.facts_in,
            facts_out=report.facts_out,
            ops=report.counts_by_op(),
            duration_ms=report.duration_ms,
        )
        return report

    # Apply the plan transactionally — single connection, single commit.
    db = memory._db
    try:
        report.ops = await _apply_plan(db, user_id, facts, valid_plan, run_ts)
        await db.commit()
    except Exception as e:
        log.exception("consolidator_apply_failed", user_id=user_id, error=str(e))
        report.error = f"apply_failed: {e}"
        report.duration_ms = int((time.monotonic() - started) * 1000)
        return report

    report.facts_out = await _count_live_facts(memory, user_id)
    report.duration_ms = int((time.monotonic() - started) * 1000)
    log.info(
        "consolidator_user_applied",
        user_id=user_id,
        facts_in=report.facts_in,
        facts_out=report.facts_out,
        ops=report.counts_by_op(),
        duration_ms=report.duration_ms,
        haiku_input_tokens=in_toks,
        haiku_output_tokens=out_toks,
    )
    return report


async def _count_live_facts(memory: MemoryStore, user_id: str) -> int:
    """Live (non-deleted) facts after applying a plan."""
    db = memory._db
    cursor = await db.execute(
        "SELECT COUNT(*) FROM user_facts WHERE user_id = ? AND deleted_at IS NULL",
        (user_id,),
    )
    row = await cursor.fetchone()
    return row[0] if row else 0


async def hard_purge_soft_deleted(
    memory: MemoryStore,
    *,
    retention_seconds: int = SOFT_DELETE_RETENTION_SECONDS,
) -> int:
    """Delete rows whose ``deleted_at`` is older than the retention window.

    Returns the number of rows actually purged. Runs at the END of every
    consolidation invocation in the same scheduled job.
    """
    cutoff = time.time() - retention_seconds
    db = memory._db
    cursor = await db.execute(
        "DELETE FROM user_facts WHERE deleted_at IS NOT NULL AND deleted_at < ?",
        (cutoff,),
    )
    purged = cursor.rowcount or 0
    await db.commit()
    log.info("consolidator_hard_purge", purged=purged, retention_seconds=retention_seconds)
    return purged


async def consolidate_all_users(
    *,
    memory: MemoryStore,
    llm_client: anthropic.AsyncAnthropic,
    model: str,
    dry_run: bool = False,
) -> list[ConsolidationReport]:
    """Run consolidation across every user that has facts.

    Sequential (per the v3.6.0 design decision): one Haiku call per user
    rather than one batched call covering all users. Cost difference is
    ~$0.15/month total at current usage and the failure mode of a batch
    call is much harder to debug.
    """
    all_facts = await memory.get_all_facts()
    user_ids = sorted({f["user_id"] for f in all_facts})
    log.info("consolidator_run_started", users=len(user_ids), dry_run=dry_run)

    reports: list[ConsolidationReport] = []
    for uid in user_ids:
        report = await consolidate_user_facts(uid, memory=memory, llm_client=llm_client, model=model, dry_run=dry_run)
        reports.append(report)

    if not dry_run:
        purged = await hard_purge_soft_deleted(memory)
        log.info("consolidator_run_complete", users=len(reports), hard_purged=purged)
    else:
        log.info("consolidator_run_complete_dry", users=len(reports))

    return reports
