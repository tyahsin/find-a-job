# def select_or_create(
#     session: Session,
#     model: Type[ModelT],
#     match_filters: dict,
#     create_values: Optional[dict] = None,
# ) -> ModelT:

from core.db import (
    get_session,
    select_or_create,
    JobORM,
)

with get_session() as session:
    jobs = select_or_create(
        session,
        JobORM,
        {'job_title': 'Data Scientist',
         'company_name': 'Merck'},
    )
    print(jobs.job_id)