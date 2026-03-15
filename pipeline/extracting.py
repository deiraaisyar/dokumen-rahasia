import pandas as pd
import pdfplumber
import re
from pathlib import Path


def extract_excel():

    df_raw = pd.read_excel(
        './data/inputs/1. IVC DOE R2 (Input).xlsx',
        sheet_name="Risk Register",
        header=None
    )

    columns = [
        "Revision Date",
        "RBS Level 1",
        "RBS Level 2",
        "Risk Name",
        "TRL",
        "TPL",
        "Technology Life Phase",
        "Risk Owner",
        "Baseline +/-",
        "Baseline TYP",
        "Baseline SEV",
        "Baseline FRQ",
        "Baseline RPN",
        "Baseline Description",
        "Response Strategy",
        "Response Description",
        "Response Timing",
        "Residual SEV",
        "Residual FRQ",
        "Residual RPN",
        "Residual Description",
        "Secondary Risks",
        "Recommendations & Action Items",
        "Contingency Plan",
    ]

    records = []
    current_budget_period = None

    for i, row in df_raw.iterrows():

        val = str(row[0]).strip()

        if i < 4:
            continue

        if val.startswith("Budget Period"):
            current_budget_period = val
            continue

        if row.isna().all():
            continue

        record = list(row.values) + [current_budget_period]
        records.append(record)

    col_names = columns + ["Budget Period"]

    df1 = pd.DataFrame(records, columns=col_names)

    cols = ["Budget Period"] + [c for c in df1.columns if c != "Budget Period"]
    df1 = df1[cols]

    df1 = df1.reset_index(drop=True)

    df2 = pd.read_excel('./data/inputs/2. City of York Council (Input).xlsx')
    df3 = pd.read_excel('./data/inputs/3. Digital Security IT Sample Register (Input).xlsx')
    df4 = pd.read_excel('./data/inputs/4. Moorgate Crossrail Register (Input).xlsx')

    return df1, df2, df3, df4


def extract_pdf():

    COL_BOUNDS = {
        'ref': (26, 52),
        'risk_text': (52, 138),
        'impact1': (138, 162),
        'likeli1': (162, 185),
        'score1': (185, 210),
        'mitigation': (210, 300),
        'impact2': (300, 323),
        'likeli2': (323, 347),
        'score2': (347, 372),
        'owner': (372, 430),
        'actions': (430, 575),
        'comments': (575, 850),
    }

    def words_to_text(wds):

        if not wds:
            return ''

        wds = [w for w in wds if not re.match(r'^[\uf0b7\u2022\uf0a7]$', w['text'])]

        if not wds:
            return ''

        wds_sorted = sorted(wds, key=lambda w: (round(w['top'] / 6), w['x0']))

        lines = {}

        for w in wds_sorted:
            key = round(w['top'] / 6)
            lines.setdefault(key, []).append(w['text'])

        return ' '.join(' '.join(v) for v in lines.values()).strip()

    records = []

    with pdfplumber.open('./data/inputs/5. Corporate_Risk_Register (Input).pdf') as pdf:

        for page_num, page in enumerate(pdf.pages[:20]):

            words = page.extract_words(x_tolerance=4, y_tolerance=4)
            page_h = page.height

            ref_words = sorted(
                [w for w in words if re.match(r'^\d{1,2}$', w['text']) and 26 <= w['x0'] <= 35],
                key=lambda w: w['top']
            )

            if not ref_words:
                continue

            for idx, rw in enumerate(ref_words):

                y_start = rw['top'] - 8
                y_end = ref_words[idx + 1]['top'] - 8 if idx + 1 < len(ref_words) else page_h - 30

                row = {'Reference': int(rw['text'])}

                for col, (x0, x1) in COL_BOUNDS.items():

                    if col == 'ref':
                        continue

                    band = [
                        w for w in words
                        if x0 <= w['x0'] < x1 and y_start <= w['top'] <= y_end
                    ]

                    row[col] = words_to_text(band)

                records.append(row)

    df5 = pd.DataFrame(records).rename(columns={
        'risk_text': 'Risk_and_Effects',
        'impact1': 'Risk_if_No_Action_Impact',
        'likeli1': 'Risk_if_No_Action_Likelihood',
        'score1': 'Risk_if_No_Action_Score',
        'mitigation': 'Mitigation',
        'impact2': 'Current_Risk_Impact',
        'likeli2': 'Current_Risk_Likelihood',
        'score2': 'Current_Risk_Score',
        'owner': 'Risk_Owner',
        'actions': 'Actions_Being_Taken',
        'comments': 'Comments_and_Progress',
    })

    df5 = df5.sort_values('Reference').reset_index(drop=True)

    for col in [
        'Risk_if_No_Action_Impact',
        'Risk_if_No_Action_Likelihood',
        'Risk_if_No_Action_Score',
        'Current_Risk_Impact',
        'Current_Risk_Likelihood',
        'Current_Risk_Score'
    ]:
        df5[col] = pd.to_numeric(df5[col], errors='coerce')

    return df5


def split_risk_effects(df):

    risks = []
    effects = []

    for text in df["Risk_and_Effects"].fillna(""):

        text = str(text)

        risk_match = re.search(r"risk[:\-]\s*(.*?)\s*effects?[:\-]", text, re.IGNORECASE)
        effect_match = re.search(r"effects?[:\-]\s*(.*)", text, re.IGNORECASE)

        risk = risk_match.group(1).strip() if risk_match else ""
        effect = effect_match.group(1).strip() if effect_match else ""

        risks.append(risk)
        effects.append(effect)

    insert_loc = df.columns.get_loc("Risk_and_Effects")

    df.insert(insert_loc, "Risk", risks)
    df.insert(insert_loc + 1, "Effects", effects)

    df = df.drop(columns=["Risk_and_Effects"])

    return df


def preprocessing(df, name):

    df = df.dropna(thresh=df.shape[1] - 5)

    if name == "df5":
        df = split_risk_effects(df)

    return df


def save_extracted(dfs):

    output_dir = Path("./data/extracted_inputs")
    output_dir.mkdir(exist_ok=True)

    names = ["df1", "df2", "df3", "df4", "df5"]

    for df, name in zip(dfs, names):

        path = output_dir / f"{name}.csv"
        df.to_csv(path, index=False)


def main():

    df1, df2, df3, df4 = extract_excel()
    df5 = extract_pdf()

    df1 = preprocessing(df1, "df1")
    df2 = preprocessing(df2, "df2")
    df3 = preprocessing(df3, "df3")
    df4 = preprocessing(df4, "df4")
    df5 = preprocessing(df5, "df5")

    save_extracted([df1, df2, df3, df4, df5])


if __name__ == "__main__":
    main()