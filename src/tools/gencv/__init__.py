"""gencv package: Job Description -> Tailored CV PDF.

放在 src/tools/gencv/,與 core/、onboard/ 平級。
對外使用方式跟 onboard 一致,外層包一層 CLI(cli.py),日常操作走:

    python -m gencv.cli generate --job-description-file jd.txt --headline "Research Analyst"

或在程式裡直接呼叫:
    from gencv import generate_cv
    pdf_path, output_id = generate_cv(job_description=jd_text, headline="Research Analyst")

檔案結構:
    cli.py                  外層 CLI 入口(typer),日常操作走這裡,風格對齊 onboard/cli.py
    pipeline.py             內部主流程,串起以下所有模組(不對外暴露,不用 -m 直接執行)
    experience_matcher.py   判斷哪些經歷跟職缺相關(治理過濾 + 單次批次 LLM 判斷)+ 共用查詢工具
    experience_rewriter.py  逐條用 LLM 輕度改寫措辭貼近 JD 用語,每條都要人工按 Enter/n 確認
    resume_builder.py       資料組裝(cv.yaml) + 呼叫 rendercv CLI 排版編譯成 PDF
    application_log.py      投遞紀錄,記錄每次產出

前置需求:
    core/db.py 需補上 GovernanceORM / ClaimChunkORM / GeneratedOutputORM 三個
    ORM class(參考 db_additions_for_core_db.py)。
    templates/design.yaml、templates/locale.yaml、templates/settings.yaml
    三份凍結設定檔已就位(可從 test.yaml 拆出對應區塊)。
    pip install rendercv

實作備註(2026-08-10):
    對外唯一執行入口現在是 cli.py(python -m gencv.cli ...),pipeline.py 已不含
    __main__ 區塊、也不會被直接用 -m 執行,因此本檔案不再需要之前用來規避
    sys.modules 重複載入警告的 __getattr__ 延遲載入技巧,直接在頂層 import 即可。
"""

from gencv.pipeline import generate_cv

__all__ = ["generate_cv"]
