"""
Build the combined Bangla harmful-comment dataset (v1).

    python dataset/build_dataset.py                 # uses data/raw, writes data/
    python dataset/build_dataset.py --no-xlsx       # skip the Excel copy (faster)

Steps (each step is recorded in the output, so every decision can be traced):
  1. Download the 5 source datasets (skipped when the files are already in data/raw).
  2. Map every source label to our 5 classes (rules below, see guideline_v1.md).
  3. Clean the text, drop empty / too-short rows.
  4. Remove duplicates across all sources. The same comment with DIFFERENT labels in two
     places is dropped completely (direct evidence of label conflict).
  5. Split into train / val / test (stratified by source and label, natural class mix).
     Val and test are never filtered.
  6. Find probable label errors in TRAIN only (5-fold TF-IDF + logistic regression,
     confident learning) and remove the strongest ones, at most MAX_REMOVE_PER_CLASS per class.
  7. Select a balanced training core (version C): CORE_PER_CLASS rows per class,
     shared as evenly as possible between the sources.
  8. Export data/bangla_forensics_dataset_v1.csv (+ .xlsx) and dataset/stats_v1.md.
"""
import argparse
import glob
import html
import os
import re
import shutil
import sys
import unicodedata

import numpy as np
import pandas as pd

SEED = 42
LABELS = ["normal", "offensive", "cyberbully", "hate_speech", "violence"]
PRIORITY = ["violence", "hate_speech", "cyberbully", "offensive"]   # when a comment has several harmful labels

TEST_SIZE = 4000              # rows, natural class mix
VAL_SIZE = 2000
MAX_REMOVE_PER_CLASS = 0.15   # label cleaning never removes more than 15% of a class
CORE_PER_CLASS = 6000         # balanced training core (version C)

HERE = os.path.dirname(os.path.abspath(__file__))
RAW_URLS = {
    "banhate/Dataset.json": "https://huggingface.co/datasets/aplycaebous/BanHate/resolve/main/Dataset.json",
    "belal/Multi_labeled_toxic_comments.csv": "https://raw.githubusercontent.com/deepu099cse/"
        "Multi-Labeled-Bengali-Toxic-Comments-Classification/main/Dataset/Multi_labeled_toxic_comments.csv",
    "bangla_online_comments/bangla_online_comments_dataset.xlsx": "https://data.mendeley.com/public-files/datasets/"
        "9xjx8twk8p/files/d50be61b-4718-476a-a28d-fad2375bbef5/file_downloaded",
    "vitd/train.csv": "https://huggingface.co/datasets/kcrl/Violence/resolve/main/train.csv",
    "vitd/dev.csv": "https://huggingface.co/datasets/kcrl/Violence/resolve/main/dev.csv",
    "vitd/test.csv": "https://huggingface.co/datasets/kcrl/Violence/resolve/main/test.csv",
}


# ---------------------------------------------------------------- 1. download
def ensure_sources(raw):
    import requests
    for path, url in RAW_URLS.items():
        dest = os.path.join(raw, path)
        if os.path.exists(dest):
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        print("downloading", path)
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=300)
        r.raise_for_status()
        with open(dest, "wb") as f:
            f.write(r.content)
    bdshs = os.path.join(raw, "bdshs")
    if not all(os.path.exists(os.path.join(bdshs, f"{s}.csv")) for s in ("train", "val", "test")):
        import kagglehub
        print("downloading BD-SHS from Kaggle")
        folder = kagglehub.dataset_download("naurosromim/bdshs")
        os.makedirs(bdshs, exist_ok=True)
        for f in glob.glob(os.path.join(folder, "**", "*.csv"), recursive=True):
            shutil.copy(f, os.path.join(bdshs, os.path.basename(f)))


# ---------------------------------------------------------------- 2. load + map labels
def pick(harmful):
    """Several harmful labels -> the most serious one (PRIORITY order)."""
    for label in PRIORITY:
        if label in harmful:
            return label
    return "normal"

# BanHate 'Abusive/Violence' mixes plain swearing and real threats. The annotators' own
# explanation tells them apart: physical harm / threat words -> violence, otherwise offensive.
VIOLENCE_WORDS = re.compile(
    r"হুমকি|হত্যা|ফাঁসি|ফাসি|শারীরিক|মারধর|মেরে|মারার|মারতে|মারো|পিটা|নির্যাতন|জবাই|পুড়িয়ে|"
    r"কুপি|গুলি|ধর্ষণ|সহিংসতার উস্কানি|সহিংসতার আহ্বান|সরাসরি সহিংস|বের করে দ")

def load_banhate(raw):
    df = pd.read_json(os.path.join(raw, "banhate", "Dataset.json"))
    rows = []
    for r in df.itertuples(index=False):
        cats = [c.strip() for c in str(r[df.columns.get_loc("Hate Category")] or "").split(",") if c.strip()]
        if r[df.columns.get_loc("H/NH")] != "H" or not cats:
            label, rule = "normal", "NH -> normal"
        else:
            harmful = set()
            for c in cats:
                if c in ("Religious", "Gender", "Origin", "Political"):
                    harmful.add("hate_speech")
                elif c in ("Personal Offence", "Body Shaming"):
                    harmful.add("cyberbully")
                elif c == "Abusive/Violence":
                    expl = str(r[df.columns.get_loc("explanation")] or "")
                    harmful.add("violence" if VIOLENCE_WORDS.search(expl) else "offensive")
            label = pick(harmful)
            rule = "+".join(sorted(set(cats))) + " -> " + label
        rows.append({"id": f"banhate_{r[df.columns.get_loc('sample')]}",
                     "text": r[df.columns.get_loc("Comment")], "source": "banhate", "source_split": "all",
                     "source_label": ",".join(cats) if cats else "NH", "label": label, "mapping_rule": rule})
    return pd.DataFrame(rows)

def load_bdshs(raw):
    parts = []
    for split in ("train", "val", "test"):
        df = pd.read_csv(os.path.join(raw, "bdshs", f"{split}.csv"))
        labels, rules = [], []
        for t, target, hs in zip(df["type"], df["target"], df["hate speech"]):
            if hs == 0 or pd.isna(t):
                labels.append("normal"); rules.append("NH -> normal"); continue
            types = set(str(t).split("_"))
            person = str(target) in ("ind", "male", "female", "male_female", "ind_male", "ind_female")
            harmful = set()
            if "callToViolence" in types:
                harmful.add("violence")
            if "religion" in types:
                harmful.add("hate_speech")
            if "gender" in types:                   # sexual / gender abuse: at one person -> cyberbully
                harmful.add("cyberbully" if person else "hate_speech")
            if "slander" in types:                  # insult: at one person -> cyberbully, else offensive
                harmful.add("cyberbully" if person else "offensive")
            label = pick(harmful)
            labels.append(label); rules.append(f"{t}|{'person' if person else 'group'} -> {label}")
        parts.append(pd.DataFrame({
            "id": [f"bdshs_{split}_{i}" for i in range(len(df))], "text": df["sentence"], "source": "bdshs",
            "source_split": split, "source_label": df["type"].fillna("NH").astype(str) + "|" + df["target"].fillna("-").astype(str),
            "label": labels, "mapping_rule": rules}))
    return pd.concat(parts, ignore_index=True)

def load_belal(raw):
    df = pd.read_csv(os.path.join(raw, "belal", "Multi_labeled_toxic_comments.csv"))
    cats = ["vulgar", "hate", "religious", "threat", "troll", "Insult"]
    to_ours = {"threat": "violence", "hate": "hate_speech", "religious": "hate_speech",
               "troll": "cyberbully", "Insult": "cyberbully", "vulgar": "offensive"}
    src = df[cats].apply(lambda r: ",".join(c for c in cats if r[c] == 1) or "none", axis=1)
    labels = src.apply(lambda s: "normal" if s == "none" else pick({to_ours[c] for c in s.split(",")}))
    return pd.DataFrame({"id": [f"belal_{i}" for i in range(len(df))], "text": df["text"], "source": "belal",
                         "source_split": "all", "source_label": src, "label": labels,
                         "mapping_rule": src + " -> " + labels})

def load_boc(raw):
    df = pd.read_excel(os.path.join(raw, "bangla_online_comments", "bangla_online_comments_dataset.xlsx"))
    to_ours = {"not bully": "normal", "troll": "cyberbully", "sexual": "cyberbully",
               "religious": "hate_speech", "threat": "violence"}
    lab = df["label"].astype(str).str.strip()
    return pd.DataFrame({"id": [f"boc_{i}" for i in range(len(df))], "text": df["comment"],
                         "source": "boc", "source_split": "all", "source_label": lab,
                         "label": lab.map(to_ours), "mapping_rule": lab + " -> " + lab.map(to_ours).fillna("?")})

def load_vitd(raw):
    names = {0: "non-violence", 1: "passive violence", 2: "direct violence"}
    to_ours = {0: "normal", 1: None, 2: "violence"}     # passive = abuse OR justified violence: too mixed to map
    parts = []
    for split in ("train", "dev", "test"):
        df = pd.read_csv(os.path.join(raw, "vitd", f"{split}.csv"))
        parts.append(pd.DataFrame({
            "id": [f"vitd_{split}_{i}" for i in range(len(df))], "text": df["text"], "source": "vitd",
            "source_split": split, "source_label": df["label"].map(names),
            "label": df["label"].map(to_ours), "mapping_rule": df["label"].map(names) + " -> " + df["label"].map(to_ours).fillna("dropped")}))
    return pd.concat(parts, ignore_index=True)


# ---------------------------------------------------------------- 3. clean
def clean(text):
    if not isinstance(text, str):
        return ""
    text = html.unescape(text).replace("<br />", " ").replace("<br>", " ")
    text = unicodedata.normalize("NFC", text).replace("‌", "").replace("‍", "").replace("﻿", "")
    return re.sub(r"\s+", " ", text).strip()

LETTERS = re.compile(r"[^ঀ-৿a-z0-9]")
def dedup_key(text):
    """Letters and digits only, lower case, repeated characters collapsed: catches near-duplicates
    that differ only in punctuation, emojis, spacing or 'sooooo' style repeats."""
    key = LETTERS.sub("", text.lower())
    return re.sub(r"(.)\1+", r"\1", key)


# ---------------------------------------------------------------- 4. duplicates
KEEP_ORDER = {"belal": 0, "bdshs": 1, "banhate": 2, "boc": 3, "vitd": 4}

def resolve_duplicates(df):
    df = df.copy()
    df["_order"] = df["source"].map(KEEP_ORDER)
    groups = df[df["status"] == "kept"].groupby("dedup_key")
    for key, g in groups:
        if len(g) == 1:
            continue
        # Belal et al. re-labelled Bangla Online Comments by hand: prefer their label.
        if (g["source"] == "belal").any() and (g["source"] == "boc").any():
            drop = g.index[g["source"] == "boc"]
            df.loc[drop, ["status", "reason"]] = ["dropped", "overlap: relabelled copy in belal kept"]
            g = g.drop(drop)
            if len(g) == 1:
                continue
        if g["label"].nunique() > 1:
            df.loc[g.index, ["status", "reason"]] = ["dropped", "duplicate with conflicting labels"]
        else:
            keep = g.sort_values("_order").index[0]
            df.loc[g.index.drop(keep), ["status", "reason"]] = ["dropped", "duplicate"]
    return df.drop(columns="_order")


# ---------------------------------------------------------------- 5. split
def split(df):
    from sklearn.model_selection import train_test_split
    kept = df[df["status"] == "kept"]
    strata = kept["source"] + "|" + kept["label"]
    rare = strata.map(strata.value_counts()) < 10                 # tiny strata always go to train
    pool = kept[~rare]
    rest, test = train_test_split(pool.index, test_size=TEST_SIZE, stratify=strata[~rare], random_state=SEED)
    train, val = train_test_split(rest, test_size=VAL_SIZE, stratify=strata[rest], random_state=SEED)
    df["split"] = ""
    df.loc[kept.index, "split"] = "train"
    df.loc[val, "split"] = "val"
    df.loc[test, "split"] = "test"
    return df


# ---------------------------------------------------------------- 6. label issues (train only)
def find_label_issues(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline

    train = df[(df["status"] == "kept") & (df["split"] == "train")]
    y = train["label"].map({l: i for i, l in enumerate(LABELS)}).to_numpy()
    model = make_pipeline(
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=300_000, sublinear_tf=True),
        LogisticRegression(max_iter=2000, C=4.0))
    print("5-fold cross-validated predictions on", len(train), "training rows ...")
    probs = cross_val_predict(model, train["text"], y, method="predict_proba",
                              cv=StratifiedKFold(5, shuffle=True, random_state=SEED), n_jobs=-1)

    # Confident learning (Northcutt et al., 2021): class threshold = average confidence the model
    # gives to rows that carry that label. A row is a probable error when another class passes its
    # threshold and wins, while the given label does not pass its own.
    thresholds = np.array([probs[y == k, k].mean() for k in range(len(LABELS))])
    above = probs >= thresholds
    given_prob = probs[np.arange(len(y)), y]
    masked = np.where(above, probs, -1.0)
    suggested = masked.argmax(axis=1)
    issue = above.any(axis=1) & (suggested != y) & (given_prob < thresholds[y])

    df.loc[train.index, "cv_pred_label"] = [LABELS[i] for i in probs.argmax(axis=1)]
    df.loc[train.index, "cv_given_prob"] = given_prob.round(4)
    df.loc[train.index, "label_issue"] = issue
    df.loc[train.index, "issue_suggested_label"] = np.where(issue, [LABELS[i] for i in suggested], "")

    # Remove the most suspicious issues, never more than MAX_REMOVE_PER_CLASS of a class
    removed = 0
    for k, label in enumerate(LABELS):
        idx = train.index[(y == k) & issue]
        cap = int(MAX_REMOVE_PER_CLASS * (y == k).sum())
        worst = df.loc[idx, "cv_given_prob"].sort_values().index[:cap]
        df.loc[worst, "train_filtered_out"] = True
        removed += len(worst)
    print(f"label issues flagged: {int(issue.sum())} | removed from training: {removed}")
    return df, {l: round(float(t), 3) for l, t in zip(LABELS, thresholds)}


# ---------------------------------------------------------------- 7. balanced core
def select_core(df):
    rng = np.random.default_rng(SEED)
    usable = df[(df["status"] == "kept") & (df["split"] == "train") & ~df["train_filtered_out"]]
    for label in LABELS:
        rows = usable[usable["label"] == label]
        by_source = {s: g.index.to_numpy() for s, g in rows.groupby("source")}
        quota = {s: 0 for s in by_source}
        left = min(CORE_PER_CLASS, len(rows))
        # water filling: share the quota equally, sources that run out pass the rest on
        while left > 0:
            open_ = [s for s in by_source if quota[s] < len(by_source[s])]
            share = max(left // len(open_), 1)
            for s in open_:
                add = min(share, len(by_source[s]) - quota[s], left)
                quota[s] += add; left -= add
                if left == 0:
                    break
        for s, n in quota.items():
            df.loc[rng.choice(by_source[s], n, replace=False), "train_core"] = True
    return df


# ---------------------------------------------------------------- 8. export
def md_table(t):
    t = t.reset_index()
    cols = [str(c) for c in t.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in t.itertuples(index=False)]
    return "\n".join(lines)

def write_stats(df, thresholds, path):
    kept = df[df["status"] == "kept"]
    s = ["# Dataset v1 - statistics (generated by build_dataset.py)\n"]
    s += ["## Rows per source and label, after mapping (before cleaning)\n",
          md_table(pd.crosstab(df["source"], df["label"].fillna("(unmapped)"), margins=True, margins_name="total"))]
    s += ["\n## Dropped rows\n", md_table(df[df["status"] == "dropped"].groupby(["reason", "source"]).size()
                                         .unstack(fill_value=0).assign(total=lambda t: t.sum(axis=1)))]
    s += ["\n## Kept rows per split and label\n",
          md_table(pd.crosstab(kept["split"], kept["label"], margins=True, margins_name="total")[LABELS + ["total"]])]
    s += ["\n## Kept rows per source and label\n",
          md_table(pd.crosstab(kept["source"], kept["label"], margins=True, margins_name="total")[LABELS + ["total"]])]
    tr = kept[kept["split"] == "train"]
    s += ["\n## Label cleaning (train only)\n",
          f"Confident-learning thresholds: {thresholds}\n",
          md_table(pd.DataFrame({
              "train rows": tr.groupby("label").size(),
              "flagged as probable error": tr.groupby("label")["label_issue"].sum().astype(int),
              "removed (max 15%)": tr.groupby("label")["train_filtered_out"].sum().astype(int)}).reindex(LABELS)),
          "\nMost common suggested corrections (given -> suggested):\n",
          md_table(tr[tr["label_issue"]].groupby(["label", "issue_suggested_label"]).size()
                   .sort_values(ascending=False).head(12).rename("rows"))]
    core = tr[tr["train_core"]]
    s += ["\n## Balanced training core (version C)\n",
          md_table(pd.crosstab(core["source"], core["label"], margins=True, margins_name="total")[LABELS + ["total"]])]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

def merge_checks(df):
    """Step 9: add the checked test labels (dataset/test_label_check.csv) to the dataset."""
    df = df.drop(columns=[c for c in ("checked_label", "check_note") if c in df.columns])
    check_path = os.path.join(HERE, "test_label_check.csv")
    if not os.path.exists(check_path):
        df["checked_label"], df["check_note"] = np.nan, np.nan
        return df
    check = pd.read_csv(check_path, dtype=str, keep_default_na=False)
    check = check.drop_duplicates("id", keep="last")
    df = df.merge(check[["id", "checked_label", "check_note"]], on="id", how="left")
    done = df["checked_label"].notna()
    print(f"checked test labels merged: {int(done.sum())} rows")
    return df

def write_check_stats(df, path):
    """Agreement between the source labels and the checked labels (test set)."""
    t = df[df["checked_label"].isin(LABELS)]
    if t.empty:
        return
    s = ["\n## Step 9: checked test labels\n",
         f"Checked rows: {df['checked_label'].notna().sum()} (of which `unclear`: "
         f"{(df['checked_label'] == 'unclear').sum()}). Labelled blind: the checker did not see the source label.\n",
         f"Agreement source label = checked label: **{(t['label'] == t['checked_label']).mean():.1%}**\n",
         "\n### Agreement per source\n",
         md_table(t.groupby("source").apply(lambda g: pd.Series({
             "rows": len(g), "agreement": f"{(g['label'] == g['checked_label']).mean():.1%}"}), include_groups=False)),
         "\n### Source label (rows) vs checked label (columns)\n",
         md_table(pd.crosstab(t["label"], t["checked_label"], margins=True, margins_name="total")
                  .reindex(index=LABELS + ["total"], columns=LABELS + ["total"], fill_value=0))]
    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(s) + "\n")

def export(df, out, xlsx):
    os.makedirs(out, exist_ok=True)
    csv_path = os.path.join(out, "bangla_forensics_dataset_v1.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")      # BOM: Excel shows Bangla correctly
    print("wrote", csv_path, len(df), "rows")
    # Compressed copy (about 8 MB) that is committed to the private GitHub repo and read by the notebook
    df.to_csv(csv_path + ".gz", index=False, encoding="utf-8-sig",
              compression={"method": "gzip", "compresslevel": 9, "mtime": 0})
    print("wrote", csv_path + ".gz")
    if xlsx:
        from openpyxl.cell.cell import ILLEGAL_CHARACTERS_RE    # control characters Excel refuses
        df = df.copy()
        for col in df.columns:
            if df[col].map(lambda v: isinstance(v, str)).any():
                df[col] = df[col].map(lambda v: ILLEGAL_CHARACTERS_RE.sub("", v) if isinstance(v, str) else v)
        xlsx_path = os.path.join(out, "bangla_forensics_dataset_v1.xlsx")
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as xw:
            df[df["status"] == "kept"].to_excel(xw, sheet_name="data", index=False)
            df[df["status"] == "dropped"].to_excel(xw, sheet_name="dropped", index=False)
            kept = df[df["status"] == "kept"]
            pd.crosstab(kept["split"], kept["label"], margins=True).to_excel(xw, sheet_name="split x label")
            pd.crosstab(kept["source"], kept["label"], margins=True).to_excel(xw, sheet_name="source x label")
            (df.groupby(["source", "mapping_rule"]).size().rename("rows").reset_index()
               .to_excel(xw, sheet_name="label mapping", index=False))
            checked = df[df["checked_label"].notna()]
            if len(checked):
                checked.to_excel(xw, sheet_name="checked test labels", index=False)
        print("wrote", xlsx_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=os.path.join(HERE, "..", "data", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "data"))
    ap.add_argument("--no-xlsx", action="store_true")
    ap.add_argument("--merge-checks", action="store_true",
                    help="only merge dataset/test_label_check.csv into an existing dataset CSV (fast)")
    args = ap.parse_args()

    if args.merge_checks:
        csv_path = os.path.join(args.out, "bangla_forensics_dataset_v1.csv")
        df = merge_checks(pd.read_csv(csv_path, encoding="utf-8-sig"))
        stats = open(os.path.join(HERE, "stats_v1.md"), encoding="utf-8").read().split("\n## Step 9")[0]
        with open(os.path.join(HERE, "stats_v1.md"), "w", encoding="utf-8") as f:
            f.write(stats.rstrip() + "\n")
        write_check_stats(df, os.path.join(HERE, "stats_v1.md"))
        export(df, args.out, not args.no_xlsx)
        return

    ensure_sources(args.raw)
    df = pd.concat([load_banhate(args.raw), load_bdshs(args.raw), load_belal(args.raw),
                    load_boc(args.raw), load_vitd(args.raw)], ignore_index=True)
    print("loaded", len(df), "rows:", df["source"].value_counts().to_dict())

    df["text"] = df["text"].map(clean)
    df["dedup_key"] = df["text"].map(dedup_key)
    df["status"], df["reason"] = "kept", ""
    df.loc[df["label"].isna(), ["status", "reason"]] = ["dropped", "label cannot be mapped (VITD passive violence)"]
    short = (df["status"] == "kept") & (df["dedup_key"].str.len() < 4)
    df.loc[short, ["status", "reason"]] = ["dropped", "empty or too short"]
    df = resolve_duplicates(df)
    df = split(df)

    df["label_issue"], df["train_filtered_out"], df["train_core"] = False, False, False
    df["cv_pred_label"], df["cv_given_prob"], df["issue_suggested_label"] = "", np.nan, ""
    df, thresholds = find_label_issues(df)
    df = select_core(df)

    df = merge_checks(df)      # step 9: the checked test labels (separate file)
    cols = ["id", "text", "source", "source_split", "source_label", "label", "mapping_rule", "status", "reason",
            "split", "label_issue", "issue_suggested_label", "cv_pred_label", "cv_given_prob",
            "train_filtered_out", "train_core", "checked_label", "check_note"]
    df = df[cols]
    write_stats(df, thresholds, os.path.join(HERE, "stats_v1.md"))
    write_check_stats(df, os.path.join(HERE, "stats_v1.md"))
    export(df, args.out, not args.no_xlsx)


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
