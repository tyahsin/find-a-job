from core.db import get_session, JobORM
from gencv.claim_selector import select_claims
from gencv.claim_ranker import narrow_all_claims
from gencv.claim_rewriter import rewrite_and_confirm

company_name = input("company_name: ")
job_title = input("job_title: ")

with get_session() as session:
    job = session.query(JobORM).filter(
        JobORM.company_name == company_name, JobORM.job_title == job_title
    ).one_or_none()
    if job is None:
        raise SystemExit("Job not found.")

job_description = job.job_description  # capture before leaving the session block
selected = select_claims(job_description)

if not selected["work_track"] and not selected["academic_track"]:
    raise SystemExit("No claims selected for this job -- pick a different job to test with.")

final_items = narrow_all_claims(selected, job_description, log_path="/Users/yahsintsai/Downloads/claim_ranker.log")

if not final_items:
    raise SystemExit("No items survived narrowing -- nothing to rewrite.")

rewritten = rewrite_and_confirm(final_items, selected, job_description)

print("\n=== Final rewritten bullets ===")
import json
with open("/Users/yahsintsai/Downloads/example.txt", "w", encoding="utf-8") as f:
    # for item in rewritten:
        # print(f"[{item['track']}] {item['group_label']} ({item['kind']}): {item['description']}")
    json.dump(rewritten, f)