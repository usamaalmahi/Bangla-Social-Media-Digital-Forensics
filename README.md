# Bangla Social-Media Digital Forensics

**Federated learning, blockchain evidence and user-level forensic profiling for harmful Bangla social-media content**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/usamaalmahi/Bangla-Social-Media-Digital-Forensics/blob/main/Bangla_Forensics_FL_UserProfiling.ipynb)

The main deliverable is the notebook [`Bangla_Forensics_FL_UserProfiling.ipynb`](Bangla_Forensics_FL_UserProfiling.ipynb).
It implements a complete digital-forensic (DF) pipeline for public social-media accounts:

1. **Text normalisation** restores abusive words hidden with `*` and converts Romanized Bangla to Bangla script.
2. A **BanglaBERT classifier**, trained with **federated learning** (three clients, FedAvg), assigns each text to one of five classes:
   `normal`, `offensive`, `cyberbully`, `hate_speech`, `violence`.
3. A **blockchain ledger** (SHA-256 hash chain with proof-of-work) stores model updates, harmful-content evidence and
   user profiles in tamper-evident form.
4. **User profiling** turns the activity time series of an account (status, share, comment) into a 5-aspect
   profile vector and a risk level (Low / Medium / High).

---

## 1. Problem statement

**Requirement: DF analysis of social-media user data.** An investigator monitors a *public* Facebook account and
records its About information and its activities (status posts, shares, comments) with timestamps. The account is
then profiled along five aspects, and the result is an activity profile with a risk level.

| # | Aspect | Question it answers |
|---|---|---|
| 1 | Activity level | How active is the account, and when? |
| 2 | Harmful ratio | What share of the activities is harmful? |
| 3 | Harm type & severity | What kind of harm, and how serious? |
| 4 | Time trend | Is the harmful behaviour escalating over time? |
| 5 | Behaviour channel | Does the user write (status), spread (share) or attack others (comment)? |

**Novel contribution: hidden and Romanized harmful comments.** Users disguise abuse by masking letters
(`kutt*r bacc*`, `কুত্*ার বাচ্চা`) or by writing Bangla in Latin letters. The pipeline restores both cases to
standard Bangla before classification:

```
kutt*r bacc*  --unmask-->  kuttar baccha  --transliterate-->  কুত্তার বাচ্চা  --classify-->  offensive  -->  EVIDENCE block
কুত্*ার বাচ্চা --unmask-->  কুত্তার বাচ্চা                       --classify-->  offensive  -->  EVIDENCE block
```

The work also addresses two problems in the training data:

* **Problem 1, class imbalance:** the data contains far more `normal` comments than harmful ones.
* **Problem 2, per-class F1:** the F1 score of every class should improve, not only the overall accuracy.

## 2. Pipeline

```
 Public account: About info + activities (status / share / comment, timestamped)
        |
        v   for every activity
 Clean -> Unmask (*) -> Transliterate (Roman -> Bangla)            Section 2
        |
        v
 BanglaBERT classifier, federated training, tuned class thresholds  Sections 4-6
        |
        +-- harmful and confidence >= 0.60 --> EVIDENCE block       Sections 3, 7
        +-- harmful but less confident     --> "uncertain (review)"
        v
 5-aspect profile -> profile vector -> risk level --> USER_PROFILE block   Section 8
        |
        v
 Ledger verification + tamper test, requirement checklist           Section 9
```

## 3. Methods

| Component | Method |
|---|---|
| **Unmasking** | Each `*` matches 1–3 unknown characters. The masked word *and its neighbour* are first matched against known abusive phrases (Nirmol, ToxLex), then single words are matched against a vocabulary built from the dataset and BanTH, with known abusive words and then frequent words preferred. The method needs no training, works for both scripts, and every decision can be explained. |
| **Transliteration** | A Roman→Bangla word dictionary is learned from aligned BanTH sentence pairs. A *spelling key* (`bh→v`, `sh→s`, double letters → single, …) merges variant spellings. A fine-tuned **BanglaT5** model translates fully Romanized sentences that still contain unknown words. |
| **Classifier** | `csebuetnlp/banglabert` with a 5-class head, trained with mixed precision and gradient clipping. |
| **Federated learning** | Three clients train locally on their own data each round; the server combines the weights with **FedAvg**, weighted by client size. The round with the best validation macro F1 is kept. Every client update and every global model is fingerprinted (SHA-256 of all weights) in the ledger. |
| **Class imbalance** | (a) a balanced training core in dataset version C; (b) over-sampling of rare harmful classes *inside each client* (at most 3× a class's own size); (c) no class-weighted loss, which over-corrected in combination with re-sampling in preliminary runs. |
| **Per-class F1** | One additive log-probability bias per class, tuned on the **validation** set by coordinate search to maximise macro F1. The test set keeps its natural class mix and is used once. |
| **Evidence ledger** | Blocks store the original text, its SHA-256, the normalised text, the prediction, the confidence and the context (user, activity type, time). A harmful prediction below 0.60 confidence is marked for human review and is not stored as evidence. |
| **Profiling** | Weekly time series per account. Risk score = 100 × (0.5 · harmful ratio + 0.3 · harmful ratio · severity + 0.2 · rise). The cut-offs correspond to 20 % (Medium) and 50 % (High) harmful activity, because the classifier itself flags roughly 10–15 % of normal comments as harmful. |

## 4. Dataset

`dataset/build_dataset.py` combines seven public Bangla datasets into one 5-class dataset using a single written
labelling guideline ([`dataset/guideline_v1.md`](dataset/guideline_v1.md)). The script removes duplicates, drops
comments that two sources label differently, fixes the train / validation / test split and flags probable label
errors with confident learning. Full provenance, licences and statistics are in
[`dataset/DATASHEET.md`](dataset/DATASHEET.md) and [`dataset/stats_v1.md`](dataset/stats_v1.md).

| Source | Platform | Rows | Use |
|---|---|---|---|
| BanHate | YouTube | 19,203 | all splits |
| BD-SHS | social media | 50,281 | all splits |
| Bangla Online Comments | Facebook | 44,001 | all splits |
| Belal et al. toxic comments | social media | 16,073 | all splits |
| BLP-2023 VITD | YouTube | 6,046 | all splits |
| TB-OLID (Romanized Bangla) | Facebook | 5,000 | training only |
| Bangla vulgar corpus | YouTube | 664 | training only |

| Split | normal | offensive | cyberbully | hate_speech | violence | total |
|---|---|---|---|---|---|---|
| train | 59,946 | 4,353 | 38,270 | 14,990 | 11,076 | 128,635 |
| validation | 932 | 48 | 603 | 238 | 179 | 2,000 |
| test | 1,864 | 94 | 1,206 | 477 | 359 | 4,000 |

The validation and test sets are identical for all three **training-data versions** (`DATA_VERSION` in cell 1.1):

| Version | Training data |
|---|---|
| A | BanHate + BD-SHS only |
| B | all sources |
| C | all sources, probable label errors removed, balanced core of up to 6,000 comments per class (default) |

**Label quality.** All 4,000 test comments were re-labelled blind against the guideline (a single LLM annotator,
documented in the datasheet). The source labels agree with these checked labels for **67.8 %** of the comments, and
`hate_speech` is the least consistent class across sources. The notebook therefore reports scores on both the
source labels and the checked labels (cell 6.5).

Additional resources: **BanTH** (Romanized ↔ Bangla sentence pairs, transliteration), **Nirmol** and **ToxLex**
(Bangla abusive words and phrases, unmasking).

## 5. Running the notebook

The notebook runs on **Google Colab with a T4 GPU** (`Runtime → Change runtime type → T4 GPU`).

1. **Dataset access.** The dataset file is stored in this private repository. Create a fine-grained GitHub token
   with *Contents: Read-only* access to this repository, and add it in Colab under *Secrets* (key icon) as
   `GITHUB_TOKEN` with notebook access enabled. Alternatively, upload `data/bangla_forensics_dataset_v1.csv.gz` to the
   Colab session.
2. **Configuration** (cell 0.2 and cell 1.1):

   | Setting | Values | Meaning |
   |---|---|---|
   | `RUN_MODE` | `FULL` / `QUICK` | `FULL`: all training data, 7 federated rounds, BanglaT5 fine-tuning, about 2 h. `QUICK`: 6,000 training comments, 3 rounds, about 15 min (functional check only). |
   | `SAVE_TO_DRIVE` | `True` / `False` | Save all outputs to Google Drive (`MyDrive/bangla_forensics_outputs`) or to the local `outputs/` folder. |
   | `SEED` | 42, 43, 44 | Random seed. Dataset versions are compared over these three seeds (cell 6.6). |
   | `DATA_VERSION` | `A` / `B` / `C` | Training-data version (see above). |

3. `Runtime → Run all`.

To analyse real accounts instead of the simulated ones, provide lawfully collected public activity as a CSV with
the columns `user_id, timestamp, activity_type, text` and set `USER_ACTIVITY_CSV` in cell 8.1.

## 6. Notebook structure

| Section | Content | Main outputs |
|---|---|---|
| 0 | Setup and configuration | – |
| 1 | Datasets: forensic dataset, BanTH, Nirmol/ToxLex; class distribution | `1_class_distribution.png` |
| 2 | Clean → unmask → transliterate (BanglaT5 fine-tuning in FULL mode) | worked examples |
| 3 | Blockchain ledger | `forensic_blockchain.json` |
| 4 | Fixed split, over-sampling inside each client | `2_balancing.png` |
| 5 | Federated training loop (FedAvg, best round by validation macro F1) | `CLIENT_UPDATE` / `GLOBAL_MODEL` blocks |
| 6 | Baseline vs improved: threshold tuning, per-class F1, confusion matrices, per-source and checked-label scores, dataset-version comparison | `per_class_f1.csv`, `3_per_class_f1.png`, `4_rounds_and_confusion_matrix.png`, `runs/*.json`, `dataset_version_comparison.csv` |
| 7 | End-to-end analysis of single comments, evidence blocks | `EVIDENCE` blocks |
| 8 | User accounts: activities, weekly time series, 5-aspect profile, risk level, forensic report | `5_timeline_*.png`, `6_user_risk_scores.png`, `user_profiles.json`, `user_profiles_summary.csv` |
| 9 | Ledger verification and tamper test, list of outputs, requirement checklist | `requirement_checklist.csv` |

The requirement checklist (cell 9.3) is filled in automatically from the results of the run, including the checks
that fail. Each simulated account was built to show one behaviour (Low risk, escalating trend, harasser role), and
the checklist confirms whether the profile detects it.

## 7. Repository contents

| Path | Content |
|---|---|
| `Bangla_Forensics_FL_UserProfiling.ipynb` | the complete pipeline |
| `dataset/build_dataset.py` | builds the combined dataset from the original sources |
| `dataset/guideline_v1.md` | label definitions and the mapping of every source label |
| `dataset/DATASHEET.md` | provenance, licences, processing steps, limitations |
| `dataset/stats_v1.md` | counts, dropped rows, label-cleaning and label-agreement tables |
| `dataset/test_label_check.csv` | checked test labels (IDs and labels only) |
| `data/bangla_forensics_dataset_v1.csv.gz` | the combined dataset, read by the notebook |

## 8. Ethics and licensing

* Only **public** data may be analysed, collected lawfully by an investigator with the appropriate authority.
  The notebook does not scrape any platform. The accounts in Section 8 are **simulated** from test-set comments and
  are marked as simulated in every output.
* Harmful predictions with low confidence are sent to human review rather than stored as evidence.
* Two sources (Belal et al., VITD) carry no licence that permits redistribution, and TB-OLID is restricted to
  non-commercial research. The repository must therefore remain **private**. See the datasheet for per-source
  licences.

## 9. Limitations

* The monitored accounts are simulated; no real account data was analysed.
* Federated learning is simulated on one GPU with three clients of similar class mix (IID).
* The ledger is a single local file with low proof-of-work difficulty. It detects modified blocks, but the latest
  block hash would have to be replicated or published to prevent the holder from rebuilding the chain.
* Label noise between sources (67.8 % agreement on the checked test set) limits per-class F1, especially for
  `hate_speech`.
* The unmasker can only restore words that occur in its vocabulary.
* The risk-score weights and cut-offs are set by design, not calibrated on labelled real accounts.

## 10. References

* Bhattacharjee et al. *BanglaBERT: Language Model Pretraining and Benchmarks for Low-Resource Language Understanding
  Evaluation in Bangla.* Findings of NAACL 2022.
* Bhattacharjee et al. *BanglaNLG and BanglaT5: Benchmarks and Resources for Evaluating Low-Resource Natural Language
  Generation in Bangla.* Findings of EACL 2023.
* McMahan et al. *Communication-Efficient Learning of Deep Networks from Decentralized Data.* AISTATS 2017 (FedAvg).
* Northcutt, Jiang and Chuang. *Confident Learning: Estimating Uncertainty in Dataset Labels.* JAIR 2021.
* Dataset sources and their references are listed in [`dataset/DATASHEET.md`](dataset/DATASHEET.md).
