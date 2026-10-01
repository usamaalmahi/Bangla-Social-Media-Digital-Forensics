# Datasheet — Bangla Forensics Dataset v1

A combined, cleaned 5-class dataset of Bangla social-media comments for the thesis classifier
(`normal`, `offensive`, `cyberbully`, `hate_speech`, `violence`).
Everything is produced by `dataset/build_dataset.py`; the numbers below come from `dataset/stats_v1.md`.

## Files

| File | In git? | What it is |
|---|---|---|
| `dataset/build_dataset.py` | yes | builds the dataset from the original sources (about 10 minutes on a laptop) |
| `dataset/guideline_v1.md` | yes | the label definitions and the source-label mapping |
| `dataset/test_label_check.csv` | yes | the checked test labels (IDs + labels only, no comment text) |
| `dataset/stats_v1.md` | yes | all counts and agreement tables |
| `data/bangla_forensics_dataset_v1.csv.gz` | **yes** | the dataset, compressed (about 8 MB, 141,268 rows incl. dropped ones) — read by the notebook |
| `data/bangla_forensics_dataset_v1.csv` | no | the same, uncompressed (UTF-8 with BOM) |
| `data/bangla_forensics_dataset_v1.xlsx` | no | the same, for viewing: sheets data, dropped, split x label, source x label, label mapping, checked test labels |

**The repository must stay private**: two sources (Belal et al., VITD) have no licence that allows publishing
them. The compressed dataset is committed; everything else in `data/` stays local and can be rebuilt with
`python dataset/build_dataset.py` (which also rewrites the `.csv.gz`).

**For Colab:** notebook cell 1.1 downloads the `.csv.gz` from the private repository with a read-only GitHub
token stored in Colab Secrets as `GITHUB_TOKEN` (setup steps in `README.md`). Without a token,
upload the `.csv.gz` to Colab by hand.

## Sources and licences

| Source | Platform | Rows | Licence | Reference |
|---|---|---|---|---|
| BanHate | YouTube | 19,203 | MIT | huggingface.co/datasets/aplycaebous/BanHate |
| BD-SHS (train + val + test) | social media | 50,281 | MIT (code repo) | Romim et al., LREC 2022 |
| Bangla Online Comments | Facebook | 44,001 | CC BY 4.0 | Ahmed et al., Mendeley 9xjx8twk8p |
| Belal et al. toxic comments | social media | 16,073 | none stated — research use with citation | Belal et al., ECCE 2023 |
| BLP-2023 VITD | YouTube | 6,046 | none stated — research use with citation | huggingface.co/datasets/kcrl/Violence |
| TB-OLID *(training only)* | Facebook, Romanized Bangla | 5,000 | AGPL-3.0 repo; paper CC BY-NC-ND 4.0 — non-commercial research | Raihan et al., BLP-2023; github.com/LanguageTechnologyLab/TB-OLID |
| Bangla vulgar corpus, drama reviews *(training only)* | YouTube | 664 | none stated in repo (paper CC BY) — research use with citation | Sazzed, PeerJ CS 2021; github.com/sazzadcsedu/Bangla-vulgar-corpus |

The last two sources were added to give `offensive` more real examples. They go to **training only**, and a
comment that already exists in one of the five original sources is dropped from them, so the validation and test
split is identical to the first build.

Note: Belal et al. re-labelled part of Bangla Online Comments by hand. Where the same comment appears in both,
the Belal copy is kept (1,118 BOC rows dropped).

## Columns

`id` (stable: source + original row) · `text` · `source` · `source_split` · `source_label` (original) ·
`label` (mapped, see guideline) · `mapping_rule` · `status` (kept / dropped) · `reason` (why dropped) ·
`split` (train / val / test) · `label_issue`, `issue_suggested_label`, `cv_pred_label`, `cv_given_prob`
(confident-learning results, train only) · `train_filtered_out` (removed as probable label error) ·
`train_core` (in the balanced core, version C) · `checked_label`, `check_note` (step 9, test only).

## Processing

1. **Label mapping** to 5 classes with one guideline (`guideline_v1.md`). A comment with several harmful labels
   gets the most serious one (violence > hate_speech > cyberbully > offensive).
2. **Cleaning:** HTML entities, `<br>`, zero-width characters and extra spaces removed; rows with fewer than
   4 letters dropped (293).
3. **Duplicates across sources** (same letters after removing punctuation, emojis and repeated characters):
   2,615 exact/near duplicates dropped; **473 comments that two sources label differently dropped completely**.
4. **Unmappable labels:** VITD *passive violence* (2,058) dropped — it mixes abuse and justified violence.
5. **Split** (seed 42, stratified by source × label, natural class mix): test 4,000, validation 2,000,
   train 123,047. Validation and test are never filtered or balanced.
6. **Label cleaning (train only):** 5-fold cross-validated TF-IDF (char 2–5-grams) + logistic regression, then
   confident learning (Northcutt et al., 2021). 13,519 probable errors flagged; 12,395 removed, never more than
   15% of a class.
7. **Balanced core (version C):** up to 6,000 per class, shared as evenly as possible between sources
   (water-filling). `offensive` has 3,701 usable real rows (2,466 before TB-OLID and the vulgar corpus were
   added), so the core has 27,701 rows. The notebook (cell 4.2, `OVERSAMPLE_TO = 1.0`) repeats `offensive`
   comments inside every client, so version C **trains on exactly 6,000 comments per class**.

## Class counts (kept rows)

| split | normal | offensive | cyberbully | hate_speech | violence | total |
|---|---|---|---|---|---|---|
| train | 59,946 | 4,353 | 38,270 | 14,990 | 11,076 | 128,635 |
| val | 932 | 48 | 603 | 238 | 179 | 2,000 |
| test | 1,864 | 94 | 1,206 | 477 | 359 | 4,000 |

`offensive` is small because the guideline sends every insult **aimed at one person** to `cyberbully`; only
vulgar language without a personal or group target stays `offensive`.

## Step 9 — checked test labels

All 4,000 test comments were labelled again, **blind** (the checker saw only the text, never the source label),
by Claude (Anthropic, Opus 5.5) following `guideline_v1.md`, in 40 chunks of 100. Seven were marked `unclear`.
**These are LLM labels from one annotator, not human gold labels.**

Agreement between the source label and the checked label: **67.8%** overall.

| Source label → | rows | agree |
|---|---|---|
| normal | 1,859 | 84.7% |
| cyberbully | 1,204 | 65.9% |
| violence | 359 | 64.3% |
| offensive | 94 | 33.0% |
| hate_speech | 477 | **16.1%** |

The weakest mappings (test rows):

| Source label | rows | agree | what the checked labels say |
|---|---|---|---|
| BOC `religious` → hate_speech | 225 | 6% | cyberbully 101, normal 94 — these are comments *about religion* in one controversy (insults at a person, or religious advice), not attacks on a religious group |
| Belal `hate` → hate_speech | 54 | 17% | normal 33 |
| Belal `troll` → cyberbully | 36 | 17% | normal 25 |
| BD-SHS `slander` at a group → offensive | 54 | 43% | offensive 23, cyberbully 20 |
| BanHate `Personal Offence` → cyberbully | 86 | 45% | cyberbully 39, normal 32 |

**What this means:** the source datasets use different ideas of "hate", "religious" and "troll". The
`hate_speech` class is the least reliable one. Scores should be reported on both the source labels and the
checked labels (notebook cell 6.5 does this).

## Recommended for v2 (not applied)

- Map Bangla Online Comments `religious` to *dropped* (or relabel it) instead of `hate_speech`.
- Review Belal `hate`/`troll` and BanHate `Personal Offence` rows flagged by the label cleaning first.
- Keep the v1 split fixed for existing IDs, so the checked test labels stay valid.

## Limitations

- Checked labels come from one LLM annotator; a human check of a sample (e.g. 300 rows, two people,
  Cohen's kappa) would make the test labels much stronger for the thesis.
- The label cleaning uses a simple TF-IDF model; it also removes some hard but correct examples (sarcasm,
  borderline comments). The 15% cap limits this.
- The test set keeps the natural mix of the combined sources (47% normal), not the real Facebook mix, where
  normal comments are far more common.
- Sources differ in platform (YouTube vs Facebook) and topic (one BOC controversy dominates many BOC comments).
