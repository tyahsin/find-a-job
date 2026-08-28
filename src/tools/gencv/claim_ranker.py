# claim_ranker.py

from core.llm_client import get_llm_client
from core.llm_reviewer import confirm_decision, parse_decision_text, GroupDecision

MAX_CLAIMS_PER_POSITION = 5
MAX_CLAIMS_PER_PROJECT = 3

RANKER_SYSTEM_PROMPT = """You are a resume content curator. You will be given a job description and \
several candidate experience records that ALL belong to the SAME group (one position, or one \
project) and have ALREADY been judged relevant to this job -- your task is NOT to re-judge \
relevance, it is to decide which to keep, which to merge together, and which to drop, so the \
final set fits within {max_claims} record(s) while showing the candidate's range rather than \
repeating the same capability.

Prefer keeping records whose capabilities are DIFFERENT from each other (MECE: mutually \
exclusive, collectively exhaustive). When two or more records describe essentially the same \
underlying capability or accomplishment, propose merging them into one group rather than \
keeping duplicates or arbitrarily dropping all but one -- the actual bullet fusion happens in a \
later step, you are only grouping them here.

Note: records in this group may come from different projects under the same position. Two \
records sharing a similar capability LABEL are NOT automatically merge candidates -- if they \
come from different projects and each independently demonstrates that capability in a \
distinct context, keeping both shows breadth across engagements and is not redundancy. Only \
propose a merge when the records describe essentially the SAME underlying accomplishment or \
fact, not merely a similar-sounding label.

{steering_note_block}

HARD CONSTRAINT (non-negotiable, checked automatically -- violating it wastes this entire turn): \
the cap of {max_claims} does NOT count raw records, it counts FINAL RENDERED BULLETS:
    total = (number of records in KEEP) + (number of MERGE lines you write)
Each MERGE line counts as EXACTLY 1 toward this total, no matter how many records it groups \
together, because all records on one MERGE line collapse into a single fused bullet later. \
Records in DROP count as 0.
You MUST satisfy total <= {max_claims} in THIS response, on your FIRST attempt -- there is no \
follow-up turn to fix it. Before writing your final KEEP/MERGE/DROP lines, silently count \
len(KEEP) + <number of MERGE lines>. If that count would exceed {max_claims}, you MUST keep \
reducing it -- by merging more records together (even if the shared capability is only loosely \
related) and/or moving more records to DROP -- until it does not, BEFORE you output anything. \
An output with total > {max_claims} is an invalid response and will be discarded.

Your output must be exactly this shape, nothing else. Within a line, separate "index:label" \
pairs with A SEMICOLON (";"), NEVER a comma. Every label -- in KEEP, MERGE, or DROP -- must be \
AT MOST 3 WORDS and contain NO symbols or punctuation at all (no commas, hyphens, slashes, \
parentheses, quotes, numbers-as-references like "#2" -- letters and single spaces only).
Line 1: KEEP <semicolon-separated "index:label" pairs, or NONE> -- label is a short capability \
tag naming what THAT record demonstrates (e.g. "API Integration", "Stakeholder Alignment", \
"Data Analysis").
Following lines (zero or more): MERGE <semicolon-separated "index:label" pairs> -- EXACTLY 2 or \
more numbers per line, one merge group per line; members of the same MERGE line should share \
the same (or nearly the same) label, since that's the reason they're being grouped. Never write \
a MERGE line with only one number -- if a record has nothing to merge with, put it under KEEP \
(or DROP) instead.
Final line: DROP <semicolon-separated "index:label" pairs, or NONE> -- label is a short reason \
for WHY that record was dropped, same 3-word/no-symbol rule (e.g. "Redundant Coverage", \
"Weakest Evidence", "Off Topic").
Every record number shown to you must appear in EXACTLY ONE of KEEP, MERGE, or DROP. Never \
invent a record number that wasn't shown to you, and never invent facts not present in the \
records.

Example line: KEEP 1:API Integration; 4:Team Collaboration"""


def _build_records_block_and_index(candidate_ids, contexts):
    index_to_claim_id = {}
    lines = []
    for i, claim_id in enumerate(candidate_ids, 1):
        ctx = contexts.get(claim_id)
        if not ctx:
            continue
        index_to_claim_id[i] = claim_id
        lines.append(
            f"[{i}] Project context: {ctx.get('project_context') or '(not provided)'}\n"
            f"    Problem: {ctx.get('problem_description') or '(not provided)'}\n"
            f"    Action: {ctx.get('action_description') or '(not provided)'}\n"
            f"    Result: {ctx.get('claim_description') or '(not provided)'}"
        )
    return "\n\n".join(lines), index_to_claim_id


def _build_context_block(avoid_note, steering_note):
    parts = []
    if avoid_note:
        parts.append(
            "The following capabilities/accomplishments are ALREADY well represented elsewhere "
            "on this resume (from the work experience section, already finalized) -- avoid "
            "selecting or merging toward records that would just repeat these; prefer records "
            "that add a NEW facet not covered below (this is a preference, not a hard exclusion "
            "-- if a record is genuinely the strongest option despite overlapping, you may still "
            "keep it):\n" + avoid_note
        )
    if steering_note:
        parts.append(
            "The user was not satisfied with a previous automated selection for this group and "
            "gave this steering note -- treat it as an additional preference on top of MECE "
            "diversity (never as license to select a record that wasn't already judged relevant, "
            "or to invent facts not in the records):\n"
            f'"{steering_note}"'
        )
    return "\n\n".join(parts)


def _build_group_system_prompt(job_description, records_block, count, max_claims, context_block=""):
    return (
        RANKER_SYSTEM_PROMPT.format(max_claims=max_claims, steering_note_block=context_block)
        + f"\n\nJob description:\n{job_description}\n\n"
        + f"{count} candidate record(s) for this group "
        + f"(cap: len(KEEP) + number_of_MERGE_lines <= {max_claims}):\n\n{records_block}"
    )


def _propose_decision_with_retry(system_prompt, valid_indices, user_prompt, temperature, fallback_ids, attempts=2):
    """Shared by claim-level and project-level selection: ask the LLM for a
    KEEP/MERGE/DROP decision, retrying up to `attempts` times before falling
    back to keeping fallback_ids (in order) with an honest 'Fallback
    Selection' label -- never a fabricated reason."""
    decision = None
    for attempt in range(attempts):
        response = get_llm_client().generate_text(
            system_prompt=system_prompt, user_prompt=user_prompt, temperature=temperature,
        )
        decision = parse_decision_text(response.text, valid_indices)
        if decision is not None:
            return decision
        print(f"  [Notice] Attempt {attempt + 1}: could not parse a usable KEEP/MERGE/DROP "
              "selection from the LLM.")

    print(f"  [Notice] Falling back to the first {len(fallback_ids)} candidate(s) in input order -- "
          "no reason is available for this fallback selection.")
    return GroupDecision(
        keep=fallback_ids, merge_groups=[],
        labels={i: "Fallback Selection" for i in fallback_ids},
    )


def _decision_counts(decision, show_merge):
    """Small shared helper: derive the kept/merged/dropped counts used in
    both the combined header line and the body sections, so the two never
    drift out of sync with each other."""
    kept = len(decision.keep)
    if show_merge:
        merge_groups = len(decision.merge_groups)
        merged_claims = sum(len(g) for g in decision.merge_groups)
        return {"kept": kept, "merge_groups": merge_groups, "merged_claims": merged_claims,
                "total": kept + merge_groups}
    return {"kept": kept, "total": kept}


def _render_decision_sections(decision, valid_indices, name_fn, show_merge=True):
    """Shared renderer for both claim-level (KEEP/MERGE/DROP) and
    project-level (KEEP/DROP only) decisions. name_fn(i) returns the display
    text for record i; labels come from decision.labels. Returns only the
    body -- callers prepend their own combined header line."""
    def _labeled(i):
        label = decision.labels.get(i)
        suffix = f" ({label})" if label else ""
        return f"  [{i}] {name_fn(i)}{suffix}"

    lines = ["KEEP:"]
    kept = sorted(decision.keep)
    if not kept:
        lines.append("  (none)")
    lines.extend(_labeled(i) for i in kept)

    claimed = set(decision.keep)
    if show_merge:
        lines.append("MERGE:")
        if not decision.merge_groups:
            lines.append("  (none)")
        for group in decision.merge_groups:
            lines.append("  [new]")
            for i in group:
                label = decision.labels.get(i)
                suffix = f" ({label})" if label else ""
                lines.append(f"  +  [{i}] {name_fn(i)}{suffix}")
            claimed.update(group)

    dropped = sorted(valid_indices - claimed)
    lines.append("DROP:")
    if not dropped:
        lines.append("  (none)")
    lines.extend(_labeled(i) for i in dropped)

    return "\n".join(lines)


def build_avoid_note(kept_items, contexts):
    """Summarizes already-finalized items (typically Stage 2's work-track
    result) into a note for a later stage's system prompt, so it can avoid
    re-selecting/merging toward the same capabilities."""
    lines = []
    for item in kept_items:
        reasons = item.get("reasons", {})
        for cid in item["claim_ids"]:
            label = reasons.get(cid)
            desc = contexts[cid]["claim_description"]
            prefix = f"{label}: " if label else ""
            lines.append(f"- {prefix}{desc}")
    return "\n".join(lines)


def _decision_to_items(decision, index_to_claim_id, group_key, group_label, track):
    items = []
    for i in decision.keep:
        cid = index_to_claim_id.get(i)
        if cid:
            items.append({
                "kind": "single", "claim_ids": [cid],
                "group_key": group_key, "group_label": group_label, "track": track,
                "reasons": {cid: decision.labels.get(i)},
            })
    for group in decision.merge_groups:
        pairs = [(i, index_to_claim_id[i]) for i in group if i in index_to_claim_id]
        if len(pairs) >= 2:
            items.append({
                "kind": "merge", "claim_ids": [cid for _, cid in pairs],
                "group_key": group_key, "group_label": group_label, "track": track,
                "reasons": {cid: decision.labels.get(i) for i, cid in pairs},
            })
    return items


def _serialize_decision_as_text(decision, all_indices):
    def _tok(i):
        return f"{i}:{decision.labels.get(i, 'reason unavailable')}"

    lines = [f"KEEP {'; '.join(_tok(i) for i in sorted(decision.keep))}" if decision.keep else "KEEP NONE"]
    for group in decision.merge_groups:
        lines.append(f"MERGE {'; '.join(_tok(i) for i in group)}")

    claimed = set(decision.keep) | {i for group in decision.merge_groups for i in group}
    dropped = sorted(set(all_indices) - claimed)
    lines.append(f"DROP {'; '.join(_tok(i) for i in dropped)}" if dropped else "DROP NONE")
    return "\n".join(lines)


def _render_group_decision(decision, contexts, index_to_claim_id, group_label):
    """Renders a GroupDecision back into a human-readable preview, using the
    SAME record numbers (1..N over this group's raw candidate list) that
    appear in the KEEP/MERGE decision text itself. The header line combines
    the group label with the KEEP/MERGE/DROP counts in one line, e.g.
    '=== Proposed selection for X: 4 kept + 2 merged (1 group(s)) = 5 total ==='."""
    def name_fn(i):
        return contexts[index_to_claim_id[i]]["claim_description"]

    valid_indices = set(index_to_claim_id.keys())
    counts = _decision_counts(decision, show_merge=True)
    header = (f"=== Proposed selection for {group_label}: {counts['kept']} kept + "
              f"{counts['merged_claims']} merged ({counts['merge_groups']} group(s)) "
              f"= {counts['total']} total ===")
    body = _render_decision_sections(decision, valid_indices, name_fn, show_merge=True)
    return header + "\n" + body


def _log_decision(log_path, rendered_block):
    """Appends ONLY the final, human-confirmed KEEP/MERGE/DROP block (its own
    header line already names the group) to log_path -- no intermediate
    attempts, no debug noise. log_path is caller-supplied (not hardcoded), so
    where the log lives is entirely up to whoever calls narrow_position_group
    / narrow_all_claims."""
    if not log_path:
        return
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(rendered_block + "\n\n")


def _force_within_cap(decision, cap):
    """Deterministic, non-LLM last resort: if a decision still has more than
    `cap` final slots (KEEP entries + MERGE groups, each MERGE group = 1
    slot), forcibly keep only the FIRST `cap` slots in the order the LLM
    itself produced them, and demote everything after that to DROP with an
    honest 'Cap Overflow' label. This guarantees count_fn(...) <= cap with
    zero additional LLM calls and zero additional human prompts -- it never
    invents a reason, it only reflects that the slot didn't fit."""
    slots = [("keep", i) for i in decision.keep] + [("merge", g) for g in decision.merge_groups]
    if len(slots) <= cap:
        return decision

    kept_slots, overflow_slots = slots[:cap], slots[cap:]
    new_keep = [i for kind, i in kept_slots if kind == "keep"]
    new_merge = [g for kind, g in kept_slots if kind == "merge"]

    new_labels = dict(decision.labels)
    for kind, item in overflow_slots:
        for i in ([item] if kind == "keep" else item):
            new_labels[i] = "Cap Overflow"

    return GroupDecision(keep=new_keep, merge_groups=new_merge, labels=new_labels)


def _propose_confirm_and_recap(build_system_prompt, valid_indices, fallback_ids, render_fn,
                                cap, count_fn, over_cap_label, log_path, max_silent_retries=2):
    """Shared flow for BOTH claim-level (narrow_position_group) and
    project-level (select_top_projects) group decisions.

    Cap enforcement now happens BEFORE the human ever sees a draft, and
    fully silently (no extra human confirmation round per retry):
      1. Propose (LLM, with _propose_decision_with_retry's own 2-attempt
         parse-retry) -> if over cap, retry up to `max_silent_retries` more
         times with a steering note, with NO human involvement in between.
      2. If still over cap after those silent retries, deterministically
         force it within cap via _force_within_cap -- guaranteed, no LLM
         call, no human prompt.
      3. Only NOW is the (already-within-cap) proposal shown to the human
         via confirm_decision for a SINGLE confirmation. If the human uses
         'r' to regenerate inside confirm_decision and the regenerated
         result is somehow over cap again (confirm_decision only validates
         grammar, not our cap), the same deterministic trim is applied
         again rather than bouncing back into another full LLM round.
    Net effect: the human is asked to confirm at most once per group, and
    the decision they see is ALWAYS already <= cap -- there is no longer a
    visible "over cap, redoing the whole thing, please confirm again" loop.
    A human 'skip all' during confirm is treated as an empty GroupDecision,
    which always satisfies the cap. build_system_prompt(steering_note) lets
    each caller supply its own template/context (avoid_note, max_claims vs
    max_projects, etc.)."""
    steering_note = None
    decision = None
    system_prompt = None
    for attempt in range(max_silent_retries + 1):
        system_prompt = build_system_prompt(steering_note)
        decision = _propose_decision_with_retry(
            system_prompt, valid_indices,
            user_prompt="Provide your decision now, following the output format exactly.",
            temperature=0.0 if not steering_note else 0.3,
            fallback_ids=fallback_ids,
        )
        count = count_fn(decision)
        if count <= cap:
            break
        if attempt < max_silent_retries:
            print(f"  [Notice] LLM proposal had {count} {over_cap_label}, over the {cap} cap -- "
                  f"silently re-prompting the LLM (retry {attempt + 1}/{max_silent_retries}, "
                  "no confirmation needed from you)...")
        steering_note = f"Previous round kept {count} {over_cap_label}, over the {cap} cap -- select fewer."

    count = count_fn(decision)
    if count > cap:
        print(f"  [Notice] Still {count} {over_cap_label} after {max_silent_retries} silent "
              f"retries -- deterministically trimming to the first {cap} in the LLM's own order "
              "(no further LLM calls, no extra confirmation).")
        decision = _force_within_cap(decision, cap)

    # The human now confirms exactly once, over a proposal that ALREADY
    # satisfies the cap.
    draft = _serialize_decision_as_text(decision, valid_indices)
    final_decision = confirm_decision(draft, system_prompt, valid_indices, render_fn=render_fn)
    if final_decision is None:
        final_decision = GroupDecision(keep=[], merge_groups=[], labels={})

    count = count_fn(final_decision)
    if count > cap:
        # Only reachable if the human used 'r' inside confirm_decision and
        # accepted a regenerated draft that happens to be over cap again
        # (confirm_decision itself only checks KEEP/MERGE/DROP grammar, not
        # this cap). Trim deterministically instead of looping the human
        # back into another full proposal round.
        print(f"  [Notice] Confirmed decision still {count} {over_cap_label}, over the {cap} "
              "cap -- deterministically trimming (no further loop).")
        final_decision = _force_within_cap(final_decision, cap)

    _log_decision(log_path, render_fn(final_decision))
    return final_decision


def narrow_position_group(candidate_ids, contexts, job_description, group_key, group_label,
                           track="work", max_claims=MAX_CLAIMS_PER_POSITION, avoid_note=None,
                           log_path=None):
    records_block, index_to_claim_id = _build_records_block_and_index(candidate_ids, contexts)
    if not index_to_claim_id:
        return []
    valid_indices = set(index_to_claim_id.keys())

    def build_system_prompt(steering_note):
        context_block = _build_context_block(avoid_note, steering_note)
        return _build_group_system_prompt(
            job_description, records_block, len(valid_indices), max_claims, context_block
        )

    def render_fn(d):
        return _render_group_decision(d, contexts, index_to_claim_id, group_label)

    # Cap counts final rendered bullets, not raw claims -- a merge group
    # collapses to ONE bullet regardless of how many claims feed into it.
    final_decision = _propose_confirm_and_recap(
        build_system_prompt, valid_indices, fallback_ids=list(valid_indices)[:max_claims],
        render_fn=render_fn, cap=max_claims,
        count_fn=lambda d: _decision_counts(d, show_merge=True)["total"],
        over_cap_label="bullet(s)", log_path=log_path,
    )
    return _decision_to_items(final_decision, index_to_claim_id, group_key, group_label, track)


def _group_ctxs_by_key(ctxs, key_field, label_field):
    """Groups context dicts by key_field (e.g. 'display_id' or 'project_id'),
    preserving first-seen order. A missing key gets its own singleton group
    (mirrors the old __no_project__ fallback) rather than being silently
    merged with other missing-key records."""
    groups = {}
    order = []
    for ctx in ctxs:
        key = ctx.get(key_field)
        if key is None:
            key = f"__no_{key_field}__{ctx['claim_id']}"
        if key not in groups:
            groups[key] = {"label": ctx.get(label_field), "claim_ids": []}
            order.append(key)
        groups[key]["claim_ids"].append(ctx["claim_id"])
    return [(key, groups[key]["label"], groups[key]["claim_ids"]) for key in order]


def narrow_work_track_claims(work_ctxs, contexts, job_description, log_path=None):
    """Stage 2: work-track claims grouped by position (display_id)."""
    items = []
    for display_id, display_name, claim_ids in _group_ctxs_by_key(work_ctxs, "display_id", "display_name"):
        items.extend(narrow_position_group(
            claim_ids, contexts, job_description,
            group_key=display_id, group_label=display_name, track="work",
            max_claims=MAX_CLAIMS_PER_POSITION, log_path=log_path,
        ))
    return items


MAX_PROJECTS_IN_SECTION = 3

PROJECT_SELECT_SYSTEM_PROMPT = """You are a resume content curator selecting which PROJECTS \
belong in the "Projects" section of a resume, given a job description. Each candidate below is \
ONE project, shown by its project context plus its raw candidate claims -- none of these claims \
have been narrowed down yet, so judge each project as a WHOLE by how well its project context \
matches the job description, not by cherry-picking one impressive claim. Select AT MOST \
{max_projects} project(s), ranking the ones whose project context is the closest fit to the JD \
first. Never merge projects with each other -- claim-level narrowing within each selected \
project happens in a later step, not here.

{context_block}

HARD CONSTRAINT (non-negotiable): there is no MERGE at this stage, every project goes into \
either KEEP or DROP, so:
    total = (number of projects in KEEP)
You MUST satisfy total <= {max_projects} on your FIRST attempt -- there is no follow-up turn to \
fix it. Before writing your final KEEP/DROP lines, silently count the KEEP entries; if it would \
exceed {max_projects}, move the weakest-fit project(s) to DROP until it does not, BEFORE you \
output anything. An output with total > {max_projects} is an invalid response and will be \
discarded.

Your output must be exactly this shape, nothing else. Within a line, separate "index:label" \
pairs with A SEMICOLON (";"), NEVER a comma. Every label must be AT MOST 3 WORDS and contain NO \
symbols or punctuation (letters and single spaces only).
Line 1: KEEP <semicolon-separated "index:label" pairs, or NONE> -- AT MOST {max_projects} \
entries; label is a short reason this project was selected (e.g. "Strong JD Fit").
Final line: DROP <semicolon-separated "index:label" pairs, or NONE> -- label is a short reason \
(e.g. "Weaker JD Fit", "Redundant Coverage").
Every project number shown to you must appear in EXACTLY ONE of KEEP or DROP. Never invent a \
project number that wasn't shown to you, and never invent facts not present in the records."""


def _build_project_candidate_block(project_entries, contexts):
    """project_entries: list of (project_id, project_name, claim_ids) -- RAW,
    pre-narrowing. Shows each project's project_context (the primary signal
    for this selection) plus its raw candidate claims for extra context."""
    index_to_project = {}
    lines = []
    for i, (project_id, project_name, claim_ids) in enumerate(project_entries, 1):
        index_to_project[i] = project_id
        sample_ctx = contexts.get(claim_ids[0], {}) if claim_ids else {}
        project_context = sample_ctx.get("project_context") or "(not provided)"
        claim_lines = [f"    - {contexts[cid]['claim_description']}" for cid in claim_ids if cid in contexts]
        lines.append(
            f"[{i}] Project: {project_name}\n"
            f"    Project context: {project_context}\n"
            + "\n".join(claim_lines)
        )
    return "\n\n".join(lines), index_to_project


def _render_project_decision(decision, project_entries, max_projects):
    def name_fn(i):
        return project_entries[i - 1][1]

    valid_indices = set(range(1, len(project_entries) + 1))
    counts = _decision_counts(decision, show_merge=False)
    dropped = len(valid_indices) - counts["kept"]
    header = (f"=== Proposed project selection: {counts['kept']} kept / {dropped} dropped "
              f"(max {max_projects}) ===")
    body = _render_decision_sections(decision, valid_indices, name_fn, show_merge=False)
    return header + "\n" + body


def select_top_projects(project_entries, contexts, job_description,
                         max_projects=MAX_PROJECTS_IN_SECTION, log_path=None):
    """project_entries: list of (project_id, project_name, claim_ids) -- RAW,
    BEFORE any claim-level narrowing. If there are more projects than
    max_projects, the LLM proposes which whole project(s) to keep, ranked by
    how well each project's project_context matches the JD (same KEEP/DROP
    grammar as claim-level narrowing, no MERGE -- projects aren't merged),
    with a 2-attempt retry before falling back, then a human confirms via the
    same confirm_decision gate. Returns the selected SUBSET of project_entries
    (same shape) -- claim-level narrowing happens afterward, only for the
    projects that survive this cut."""
    if not project_entries:
        return []
    if len(project_entries) <= max_projects:
        return project_entries

    records_block, index_to_project = _build_project_candidate_block(project_entries, contexts)
    valid_indices = set(index_to_project.keys())

    def build_system_prompt(steering_note):
        context_block = _build_context_block(avoid_note=None, steering_note=steering_note)
        return (
            PROJECT_SELECT_SYSTEM_PROMPT.format(max_projects=max_projects, context_block=context_block)
            + f"\n\nJob description:\n{job_description}\n\n"
            + f"{len(project_entries)} candidate project(s):\n\n{records_block}"
        )

    def render_fn(d):
        return _render_project_decision(d, project_entries, max_projects)

    final_decision = _propose_confirm_and_recap(
        build_system_prompt, valid_indices, fallback_ids=list(valid_indices)[:max_projects],
        render_fn=render_fn, cap=max_projects,
        count_fn=lambda d: _decision_counts(d, show_merge=False)["total"],
        over_cap_label="project(s)", log_path=log_path,
    )
    return [project_entries[i - 1] for i in sorted(final_decision.keep)]


def narrow_degree_track_claims(academic_ctxs, contexts, job_description, avoid_note, log_path=None):
    """Stage 3: degree/project-section. Selection happens FIRST -- which
    projects make the cut (MAX_PROJECTS_IN_SECTION), ranked by how well each
    project's project_context matches the JD, via select_top_projects -- and
    only the SELECTED projects then go through per-project claim-level MECE
    narrowing (MAX_CLAIMS_PER_PROJECT), with Stage 2's avoid_note applied at
    that point."""
    project_entries = _group_ctxs_by_key(academic_ctxs, "project_id", "project_name")
    selected_projects = select_top_projects(
        project_entries, contexts, job_description,
        max_projects=MAX_PROJECTS_IN_SECTION, log_path=log_path,
    )

    items = []
    for project_id, project_name, claim_ids in selected_projects:
        items.extend(narrow_position_group(
            claim_ids, contexts, job_description,
            group_key=project_id, group_label=project_name, track="degree",
            max_claims=MAX_CLAIMS_PER_PROJECT, avoid_note=avoid_note, log_path=log_path,
        ))
    return items


def narrow_all_claims(selected, job_description, log_path=None):
    """Full Stage 2 + Stage 3 pipeline: work-track first (sequencing enforces
    track priority, not a shared budget), then degree-track with an
    avoid_note built from work-track's finalized result. log_path, if given,
    receives one appended block per group with ONLY the final confirmed
    KEEP/MERGE/DROP decision -- no intermediate attempts.

    Returns (items, contexts) -- contexts is the same claim_id -> full-row
    dict built here from `selected` (which claim_selector.py already
    populated with every field downstream stages need, including
    notes_for_llm/org_name/project_context/etc.). Callers such as
    experience_rewriter.rewrite_and_confirm() should reuse this SAME
    contexts dict instead of re-querying the DB -- claim_selector.py already
    did the one-and-only DB round trip for this whole pipeline."""
    contexts = {ctx["claim_id"]: ctx for ctx in selected["work_track"] + selected["academic_track"]}

    work_items = narrow_work_track_claims(selected["work_track"], contexts, job_description, log_path=log_path)
    avoid_note = build_avoid_note(work_items, contexts)
    degree_items = narrow_degree_track_claims(
        selected["academic_track"], contexts, job_description, avoid_note, log_path=log_path
    )

    return work_items + degree_items, contexts