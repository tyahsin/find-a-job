此資料夾需自行放入以下檔案(本包未附):

templates/
├── design_harvard.yaml   (從 test.yaml 拆出 design 區塊)
├── locale_base.yaml      (從 test.yaml 拆出 locale 區塊)
├── settings_base.yaml    (從 test.yaml 拆出 settings 區塊)
└── harvard/
    ├── Header.j2.typ
    ├── Preamble.j2.typ
    ├── SectionBeginning.j2.typ
    ├── SectionEnding.j2.typ
    └── entries/
        ├── ExperienceEntry.j2.typ
        └── OneLineEntry.j2.typ

來源:RenderCV 官方 GitHub repo (src/rendercv/renderer/templater/templates/typst/)
