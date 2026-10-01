# Bangla Social-Media Digital Forensics — সহজ বাংলায় পুরো নোটবুকের ব্যাখ্যা

> ফাইল: `Bangla_Forensics_FL_UserProfiling.ipynb`
> এই লেখায় আছে: **কী চাওয়া হয়েছিল (requirement)**, **কী করা হয়েছে**, **কীভাবে করা হয়েছে**, আর **কেন এভাবে করা হয়েছে**।
> টেকনিক্যাল শব্দগুলো (model, dataset, F1, blockchain ইত্যাদি) ইংরেজিতেই রাখা হয়েছে, কারণ থিসিসে ও মিটিংয়ে এগুলো ইংরেজিতেই বলা হয়। শেষে একটা ছোট শব্দকোষ দেওয়া আছে।

---

## সূচিপত্র

0. [Handover-এর আগে যা করতে হবে (নিজের জন্য নির্দেশনা)](#০-handover-এর-আগে-যা-করতে-হবে-নিজের-জন্য-নির্দেশনা)
1. [এক নজরে পুরো কাজ](#১-এক-নজরে-পুরো-কাজ)
2. [Requirement: কী চাওয়া হয়েছিল](#২-requirement-কী-চাওয়া-হয়েছিল)
3. [পুরো সিস্টেমের ছবি (pipeline)](#৩-পুরো-সিস্টেমের-ছবি-pipeline)
4. [নোটবুক কীভাবে চালাতে হয়](#৪-নোটবুক-কীভাবে-চালাতে-হয়)
5. [Section 0 — Setup](#৫-section-0--setup)
6. [Section 1 — Dataset লোড করা](#৬-section-1--dataset-লোড-করা)
7. [Section 2 — Text normalisation: clean → unmask → transliterate (Novel)](#৭-section-2--text-normalisation-novel)
8. [Section 3 — Blockchain (evidence ledger)](#৮-section-3--blockchain-evidence-ledger)
9. [Section 4 — Class imbalance ঠিক করা (Problem 1)](#৯-section-4--class-imbalance-ঠিক-করা-problem-1)
10. [Section 5 — Federated Learning classifier](#১০-section-5--federated-learning-classifier)
11. [Section 6 — Per-class F1 ফলাফল (Problem 2)](#১১-section-6--per-class-f1-ফলাফল-problem-2)
12. [Section 7 — একটা comment-এর পুরো বিশ্লেষণ + Live demo](#১২-section-7--একটা-comment-এর-পুরো-বিশ্লেষণ--live-demo)
13. [Section 8 — Facebook user account-এর DF analysis (মূল Requirement)](#১৩-section-8--facebook-user-account-এর-df-analysis-মূল-requirement)
14. [Section 9 — Blockchain যাচাই, ফাইল সেভ, Requirement checklist](#১৪-section-9--blockchain-যাচাই-ফাইল-সেভ-requirement-checklist)
15. [গুরুত্বপূর্ণ সিদ্ধান্তগুলো আর তার কারণ](#১৫-গুরুত্বপূর্ণ-সিদ্ধান্তগুলো-আর-তার-কারণ)
16. [ফলাফল নিয়ে সৎ মূল্যায়ন ও সীমাবদ্ধতা](#১৬-ফলাফল-নিয়ে-সৎ-মূল্যায়ন-ও-সীমাবদ্ধতা)
17. [Supervisor-এর সম্ভাব্য প্রশ্ন ও উত্তর](#১৭-supervisor-এর-সম্ভাব্য-প্রশ্ন-ও-উত্তর)
18. [শব্দকোষ (সহজ ভাষায়)](#১৮-শব্দকোষ-সহজ-ভাষায়)

---

## ০. Handover-এর আগে যা করতে হবে (নিজের জন্য নির্দেশনা)

নোটবুকটা professor-কে দেওয়ার মতো করে গোছানো হয়েছে: demo guide, "মিটিংয়ে কী দেখাতে হবে", token setup-এর ধাপ, "thesis-এর সংখ্যা এখান থেকে নাও" — এ ধরনের নিজের জন্য লেখা নির্দেশনা নোটবুক থেকে সরিয়ে এখানে (আর setup-এর অংশটা `README.md`-তে) আনা হয়েছে। নোটবুকে এখন শুধু পদ্ধতির ব্যাখ্যা, কোড আর একটা **Limitations** অংশ আছে।

### ০.১ সবচেয়ে জরুরি: নোটবুক FULL mode-এ আবার চালানো

> ⚠️ **নোটবুকের সব output মুছে (clear) দেওয়া হয়েছে।** কারণ: সেভ করা output-গুলো ছিল **পুরনো pipeline-এর** — পুরনো dataset (শুধু BanHate + BD-SHS, 59,410 comment), পুরনো 80/20 split (test 11,882), আর under-sampling। কিন্তু কোড এখন নতুন dataset (৭টা উৎস, fixed 4,000 test), version C আর over-sampling ব্যবহার করে। ফলে টেবিলে `undersampled_F1` দেখাচ্ছিল অথচ কোডে `oversampled_F1` — professor দেখলে প্রথমেই এটা ধরবেন। পুরনো output-গুলো হারায়নি: git history-তে আছে (commit `2c4cf94`-এর নোটবুক)।

করণীয়:
1. Colab-এ নোটবুক খোলা → `Runtime → Change runtime type → T4 GPU`।
2. Cell 0.2-এ `RUN_MODE = "FULL"`, `SEED = 42`; cell 1.1-এ `DATA_VERSION = "C"`।
3. `Runtime → Run all` (প্রায় ২ ঘণ্টা)।
4. শেষ হলে `File → Save a copy in GitHub` (বা নোটবুক download করে repo-তে রাখা) — যাতে **নতুন output সহ** নোটবুক repo-তে যায়।
5. Cell 9.3-এর checklist দেখা: কোন কোন ✘ আছে। ✘ থাকলে লুকানোর দরকার নেই (নিচে ০.৫ দেখো), কিন্তু professor-কে দেওয়ার আগে নিজে জানা থাকা দরকার।
6. (ঐচ্ছিক, কিন্তু ভালো) নতুন সংখ্যাগুলো দিয়ে `README.md`-তে একটা ছোট **Results** অংশ যোগ করা: per-class F1 টেবিল (baseline বনাম improved) আর ৩ user-এর risk level। এখন README-তে কোনো ফলাফলের সংখ্যা নেই, কারণ পুরনো সংখ্যাগুলো বর্তমান কোডের সাথে মেলে না।

### ০.২ Dataset version তুলনা (cell 6.6) — ৯টা run

Cell 6.6 dataset version A, B, C তুলনা করে, প্রতিটা ৩টা seed (42, 43, 44) দিয়ে — মোট ৯টা run। প্রতিটা run নিজের ফলাফল `runs/` ফোল্ডারে (Google Drive-এ) রাখে, আর 6.6 সব মিলিয়ে mean ± std দেখায়।
- সময় বাঁচাতে এই extra run-গুলোতে cell 6.1-এ `RUN_BASELINE = False` দেওয়া যায় (baseline শুধু একবার, মূল run-এ লাগে)।
- **কেন ৩টা seed?** একই setup দুবার চালালেই macro F1 ~0.002–0.005 বদলায় (GPU-র randomness)। একটা run দিয়ে "A-র চেয়ে C ভালো" বলা যায় না; পার্থক্য যদি seed-গুলোর মধ্যের ছড়ানোর (std) চেয়ে স্পষ্টভাবে বড় হয়, তবেই সেটা আসল।
- `SAVE_TO_DRIVE = True` রাখতে হবে, নইলে Colab session বন্ধ হলে `runs/` হারিয়ে যাবে।
- **Sidenote — seed:** random সংখ্যা তৈরির শুরুর বিন্দু। একই seed = একই shuffle, একই client ভাগ, তাই ফলাফল পুনরায় পাওয়া যায় (reproducible)।

### ০.৩ GitHub token আর ToxLex

**GITHUB_TOKEN (dataset পড়ার জন্য, একবারই করতে হয়):**
1. GitHub → Settings → Developer settings → **Fine-grained tokens** → *Generate new token*: repository access = শুধু এই repo, permission **Contents: Read-only**।
2. Colab-এ বাঁ পাশের **চাবি (key) আইকন** → *Add new secret* → নাম `GITHUB_TOKEN`, token paste, *Notebook access* চালু।
- Token নোটবুকে কখনো লেখা হয় না, তাই নোটবুক GitHub-এ গেলেও token ফাঁস হয় না।
- Token ছাড়াও চলে: `bangla_forensics_dataset_v1.csv.gz` হাতে Colab-এর Files panel-এ upload করলে।

**ToxLex এখনো লোড হয় না** (Mendeley Colab-কে 403 দেয়)। কোড এখন বদলানো হয়েছে: Mendeley ব্যর্থ হলে একই GITHUB_TOKEN দিয়ে private repo-র `data/ToxLex.xlsx` থেকে নেয়। (আগের কোড `raw.githubusercontent.com` থেকে token ছাড়া নিত — private repo-তে সেটা কখনোই কাজ করত না।) করণীয়:
1. ব্রাউজারে Mendeley dataset `9pz8ssmc49` থেকে ToxLex Excel ফাইলটা নামানো।
2. `data/ToxLex.xlsx` নামে রাখা (`.gitignore`-এ এর জন্য ছাড় যোগ করা হয়েছে) আর commit + push করা।
3. তাহলে unmasker 1,182টা Nirmol entry-র সাথে আরও ~1,959টা ToxLex bigram পাবে — বিশেষ করে বাক্যাংশ-ভিত্তিক unmask ভালো হবে।

### ০.৪ Professor-এর access

- Repo **private** (দুটো উৎসের license প্রকাশ করার অনুমতি দেয় না)। Professor-কে GitHub-এ **collaborator** হিসেবে যোগ করতে হবে, নইলে README, নোটবুক বা Colab badge কিছুই খুলবে না।
- Professor নিজে চালাতে চাইলে তাঁর নিজের token লাগবে (README-র Section 5-এ ধাপগুলো আছে), অথবা `.csv.gz` ফাইলটা আলাদাভাবে দেওয়া যায়।
- `SAVE_TO_DRIVE = True` থাকলে চালানোর সময় Google Drive-এর অনুমতি চাইবে — professor-কে জানিয়ে রাখা ভালো, অথবা তাঁর জন্য `False` (তখন output Colab-এর `outputs/` ফোল্ডারে যায়)।
- নোটবুকের শিরোনামে নিজের নাম / supervisor-এর নাম / বিভাগ যোগ করতে চাইলে cell 1-এর একদম উপরে যোগ করা যায় (এখন নাম নেই)।
- এই সব পরিবর্তন এখনো **commit করা হয়নি** — দেখে নিয়ে commit আর push করতে হবে।

### ০.৫ Supervisor মিটিংয়ে দেখানোর ক্রম (নোটবুক থেকে সরানো demo guide)

মিটিংয়ের আগে FULL mode-এ "Run all" করে রাখা (output সহ)। ছোট demo-র জন্য `RUN_MODE = "QUICK"` (~১৫ মিনিট) — কিন্তু QUICK-এর সংখ্যা থিসিসে দেওয়া যাবে না।

| ধাপ | Cell | কী দেখায় |
|---|---|---|
| 1 | **1.2** | Problem 1: class distribution (training data) |
| 2 | **2.5** | Novelty: `kutt*r bacc*` → `kuttar baccha` → `কুত্তার বাচ্চা`, আর `কুত্*ার বাচ্চা` → `কুত্তার বাচ্চা` |
| 3 | **7.1** | পুরো chain: লুকানো লেখা → class → blockchain-এর EVIDENCE block |
| 4 | **7.2** | Live: `MY_COMMENT`-এ যেকোনো comment (বাংলা, Romanized, `*` সহ) লিখে চালানো |
| 5 | **6.3** | Problem 2: প্রতিটা class-এর F1, baseline বনাম improved |
| 6 | **6.5** | উৎস অনুযায়ী score আর checked label-এ score (label noise দেখানো) |
| 7 | **8.3 – 8.5** | Requirement: account profile (৫ aspect, time series, risk level) |
| 8 | **9.1** | Blockchain বদলানো block সাথে সাথে ধরে ফেলে |
| 9 | **9.3** | Checklist: প্রতিটা requirement-এর এই run-এর ফলাফল |

**Checklist-এ ✘ এলে কী বলবে:** checklist ইচ্ছা করে run-এর আসল ফলাফল থেকে পূরণ হয়, হাতে লেখা নয় — তাই ✘ লুকানো হয় না। ✘-এর পেছনের কারণ (যেমন label noise) নিচে ১৬ আর ১৭ নম্বর অংশে আছে।

### ০.৬ একটা সৎ সতর্কতা: checked test label

`dataset/test_label_check.csv`-এর ৪,০০০টা "checked label" কোনো মানুষ দেয়নি — **একটা LLM (Claude) guideline দেখে দিয়েছে** (DATASHEET-এ স্পষ্ট লেখা আছে, নোটবুক আর README-তেও উল্লেখ করা হয়েছে)। Professor জিজ্ঞেস করলে এটা সরাসরি বলতে হবে। DATASHEET-এর পরামর্শ: ~৩০০টা row দুজন মানুষ দিয়ে label করিয়ে **Cohen's kappa** হিসাব করা — তাহলে test label অনেক বেশি বিশ্বাসযোগ্য হয়।
- **Sidenote — Cohen's kappa:** দুজন annotator কতটা একমত, সেটা মাপার সংখ্যা — কিন্তু কাকতালীয়ভাবে একমত হওয়ার অংশটা বাদ দিয়ে। ১ = পুরো মিল, ০ = শুধু কাকতালীয় মিল।

> 📌 **এই ফাইলের বাকি অংশে "সেভ করা run" / "শেষ FULL run" লেখা সংখ্যাগুলো সব পুরনো pipeline-এর** (পুরনো dataset, under-sampling)। এগুলো রাখা হয়েছে ইতিহাস আর যুক্তি বোঝার জন্য — নতুন FULL run-এর পর সংখ্যা বদলাবে।

---

## ১. এক নজরে পুরো কাজ

এই নোটবুকটা থিসিসের **final pipeline**। এক কথায়:

> একজন investigator একটা **public Facebook account**-এর কার্যকলাপ (status, share, comment) সময় ধরে (time series) দেখবে, প্রতিটা লেখা **ক্ষতিকর (harmful) কিনা** তা একটা AI model দিয়ে বের করবে, তারপর ওই account-এর একটা **profile** আর **risk level (Low / Medium / High)** বানাবে। সব প্রমাণ (evidence) একটা **blockchain**-এ রাখা হবে যাতে পরে কেউ বদলাতে না পারে।

এর সাথে দুটো **নতুনত্ব (novelty)** আছে:

1. মানুষ খারাপ শব্দ `*` দিয়ে লুকায় (যেমন `kutt*r bacc*`) বা বাংলা ইংরেজি অক্ষরে লেখে (Romanized Bangla)। সাধারণ model এগুলো ধরতে পারে না। আমাদের সিস্টেম আগে এগুলোকে **সাধারণ বাংলায় ফিরিয়ে আনে**, তারপর classify করে।
2. Dataset-এ `normal` comment অনেক বেশি (class imbalance), তাই model সব কিছুকে "normal" বলতে চায়। এটা ঠিক করে **প্রতিটা class-এর F1 score** ভালো করার চেষ্টা করা হয়েছে।

Model train করা হয়েছে **Federated Learning** দিয়ে (৩টা client, কেউ কারো data দেখে না), আর model-এর প্রতিটা ধাপও blockchain-এ লেখা থাকে।

---

## ২. Requirement: কী চাওয়া হয়েছিল

Requirement এসেছে দুটো হাতে লেখা নোট থেকে (ফোল্ডারের দুটো WhatsApp ছবি)।

### ২.১ প্রথম নোট — "DF Analysis from social media user data"

নোটে যা লেখা:

```
Facebook
  └── User account: Account profile public
        ├── About / personal info
        └── 1. Monitoring activities
               → status
               → share
               → comments
               → Time series data
                     ↓
               Profiling → Aspect: 1, 2, 3, 4, 5
                     ↓
               Result: # Activity
```

সহজ ভাষায় চাওয়া হয়েছিল:

| চাহিদা | মানে |
|---|---|
| **Public account** | শুধু public প্রোফাইল নিয়ে কাজ হবে (private নয়)। |
| **About / personal info** | প্রোফাইলের public তথ্য (কবে join করেছে, followers কত ইত্যাদি) রাখতে হবে। |
| **Monitoring activities** | user কী কী করছে তা দেখতে হবে: নিজে **status** লেখা, অন্যের পোস্ট **share** করা, অন্যের পোস্টে **comment** করা। |
| **Time series data** | প্রতিটা কাজের তারিখ/সময় আছে, তাই সময়ের সাথে আচরণ কীভাবে বদলাচ্ছে সেটা দেখতে হবে। |
| **Profiling with 5 aspects** | ৫টা দিক থেকে user-কে বিশ্লেষণ করতে হবে। |
| **Result** | শেষে user-এর activity profile আর একটা ফলাফল (risk level) দিতে হবে। |

নোটের ডান পাশে "Nobelity" (novelty) লেখা একটা ঘর আছে, যেখানে লুকানো (`*`) শব্দকে পূর্ণ করার ছবি আঁকা — এটাই দ্বিতীয় নোটে বিস্তারিত।

### ২.২ দ্বিতীয় নোট — Novelty আর দুটো Problem

নোটে যা লেখা:

```
Process / Output
kutt*r bacc*
   ↓
kuttar bacca
   ↓
কুত্তার বাচ্চা
   ↓
offensive  → can be saved in the blockchain

Output
কুত্*ার বাচ্চা
   ↓
কুত্তার বাচ্চা
   ↓
offensive  → saved into blockchain

Problem
১। Dataset has more bangla comments with normal level
২। We need to improve F1 score for every class in the database
```

সহজ ভাষায়:

- **Novelty:** লুকানো অক্ষর আন্দাজ করা (unmask) → ইংরেজি অক্ষরের বাংলাকে বাংলা অক্ষরে আনা (transliterate) → classify করা → ক্ষতিকর হলে blockchain-এ সেভ করা।
- **Problem 1:** Dataset-এ `normal` comment বেশি (imbalance)।
- **Problem 2:** শুধু মোট accuracy না, **প্রতিটা class-এর F1** ভালো করতে হবে।

### ২.৩ নোটবুকে ৫টা aspect যা ঠিক করা হয়েছে

হাতের নোটে aspect-গুলোর নাম লেখা ছিল না (শুধু 1–5 নম্বর)। নোটবুকে এগুলো এভাবে সংজ্ঞায়িত করা হয়েছে:

| # | Aspect | কোন প্রশ্নের উত্তর দেয় |
|---|---|---|
| 1 | Activity level | account কতটা active, আর কখন? |
| 2 | Harmful ratio | মোট কাজের কত ভাগ ক্ষতিকর? |
| 3 | Harm type & severity | কী ধরনের ক্ষতি (offensive, hate, violence…) আর কতটা গুরুতর? |
| 4 | Time trend | সময়ের সাথে ক্ষতিকর আচরণ কি বাড়ছে (escalating)? |
| 5 | Behaviour channel | user নিজে লেখে (status), ছড়ায় (share), নাকি অন্যকে আক্রমণ করে (comment)? |

**কেন এই ৫টা?** একজন forensic investigator-এর মূল প্রশ্নগুলো হলো: লোকটা কতটা active, কতটা খারাপ কাজ করে, কী ধরনের খারাপ, সেটা কি বাড়ছে, আর কোন পথে করে। এই ৫টা aspect ঠিক এই প্রশ্নগুলোর উত্তর দেয়, আর requirement-এর তিনটা activity type (status/share/comment) আর time series দুটোই কাজে লাগায়।

---

## ৩. পুরো সিস্টেমের ছবি (pipeline)

```
                   ┌──────────── Public Facebook account ────────────┐
                   │  About info (joined, followers …)                │
                   │  Activities: status / share / comment + সময়     │
                   └──────────────────────┬───────────────────────────┘
                                          │ প্রতিটা লেখা
                                          ▼
   ┌──────────── Novel part: text normalisation (Section 2) ────────────┐
   │  Clean  →  Unmask (* পূরণ)  →  Transliterate (Roman → বাংলা)       │
   └──────────────────────┬──────────────────────────────────────────────┘
                          ▼
   ┌──── Classifier: BanglaBERT, Federated Learning-এ train করা (Section 5, 6) ────┐
   │  ৫টা class: normal / offensive / cyberbully / hate_speech / violence          │
   └──────────────────────┬─────────────────────────────────────────────────────────┘
                          ▼
         ক্ষতিকর আর confidence ≥ 0.60 ?  ──হ্যাঁ──►  EVIDENCE block (blockchain)
                          │ না / কম নিশ্চিত
                          ▼
                 normal / "uncertain (review)"
                          │
                          ▼
   ┌──────── Profiling (Section 8): ৫টা aspect → profile vector → risk level ────────┐
   └──────────────────────┬────────────────────────────────────────────────────────────┘
                          ▼
                 USER_PROFILE block (blockchain)
```

**দুটো অংশ কীভাবে জোড়া লাগে:** Requirement অংশের (Facebook account) প্রতিটা activity আগে novel pipeline দিয়ে যায়। ফলে লুকানো বা Romanized খারাপ comment-ও ধরা পড়ে। ক্ষতিকর activity হয় `EVIDENCE` block, আর শেষ profile হয় `USER_PROFILE` block — পুরো তদন্তের রেকর্ড পরে আর বদলানো যায় না।

---

## ৪. নোটবুক কীভাবে চালাতে হয়

1. Google Colab-এ খুলে `Runtime → Change runtime type → T4 GPU` দিতে হবে।
2. Cell **0.2**-এ `RUN_MODE` বেছে নিতে হবে:

| Mode | কী হয় | সময় (T4 GPU) | কখন ব্যবহার |
|---|---|---|---|
| `"QUICK"` | ৬,০০০ training comment, ৩ federated round, BanglaT5 train হয় না | ~১৫ মিনিট | কোড ঠিকঠাক চলছে কিনা দেখা / ছোট demo |
| `"FULL"` | সব training comment, ৭ round, BanglaT5 train হয় | ~২ ঘণ্টা | **থিসিসের সংখ্যা এখান থেকে নিতে হবে** |

3. Cell **1.1**-এ `DATA_VERSION` (`"C"` default) — নিচে ৬ নম্বর অংশে ব্যাখ্যা।
4. `Runtime → Run all`।

Dataset পড়ার জন্য `GITHUB_TOKEN` secret লাগে (০.৩ দেখো)। মিটিংয়ে দেখানোর ক্রম ০.৫-এ।

---

## ৫. Section 0 — Setup

### কী করা হয়েছে
- **0.1:** দরকারি library install (`transformers`, `sentencepiece`, `datasets`, `kagglehub`, `scikit-learn`, `seaborn`, `openpyxl`)।
- **0.2:** সব import আর মূল setting। সাধারণত শুধু এই cell (আর 1.1-এর `DATA_VERSION`) বদলাতে হয়।

### মূল setting-গুলো

| Setting | মান | মানে ও কারণ |
|---|---|---|
| `RUN_MODE` | `"FULL"` | উপরে ব্যাখ্যা করা হয়েছে। |
| `SAVE_TO_DRIVE` | `True` | ফলাফল Google Drive-এ (`MyDrive/bangla_forensics_outputs`) সেভ হয়, Colab বন্ধ হলেও হারায় না। `False` দিলে Colab-এর `outputs/` ফোল্ডারে (session শেষে মুছে যায়)। |
| `SEED` | `42` | প্রতিবার চালালে একই ফলাফল আসার জন্য (reproducibility)। Dataset version তুলনায় 42, 43, 44 (০.২ দেখো)। |
| `DEVICE` | GPU থাকলে `cuda` | GPU ছাড়া training খুব ধীর, তাই warning দেখায়। |
| `USE_AMP` | GPU-তে চালু | Mixed precision — GPU-তে training প্রায় ২ গুণ দ্রুত হয়। |
| `LABELS` | `normal, offensive, cyberbully, hate_speech, violence` | ৫টা class। list-এ অবস্থানই class নম্বর। |
| `CLASSIFIER_MODEL` | `csebuetnlp/banglabert` | বাংলার জন্য pretrained BERT — বাংলা ভাষা আগে থেকেই "বোঝে"। |
| `T5_MODEL` | `csebuetnlp/banglat5` | Transliteration-এর জন্য বাংলা T5 model। |
| `NUM_CLIENTS` | `3` | Federated learning-এ ৩টা প্রতিষ্ঠান (client)। |
| `LEARNING_RATE` | `3e-5` | BERT fine-tuning-এর জন্য প্রচলিত নিরাপদ মান। |
| `EVIDENCE_MIN_CONFIDENCE` | `0.60` | Model অন্তত ৬০% নিশ্চিত হলেই ক্ষতিকর লেখা evidence হবে। কম নিশ্চিত হলে "uncertain", মানুষ দেখবে। |

**Mode অনুযায়ী setting:**

| | QUICK | FULL |
|---|---|---|
| `MAX_ROWS` | 6000 | সব |
| `FL_ROUNDS` | 3 | 7 |
| `LOCAL_EPOCHS` | 2 | 1 |
| `MAX_LEN` (token) | 64 | 128 |
| `BATCH_SIZE` | 16 | 16 |
| `TRAIN_BANGLAT5` | না | হ্যাঁ |

**কেন FULL-এ ৭ round?** আগের একটা run-এ ৫ round-এর শেষেও validation F1 বাড়ছিল — মানে model আরও শিখতে পারত। তাই round বাড়ানো হয়েছে।

**কেন `EVIDENCE_MIN_CONFIDENCE` 0.60?** আগে এটা 0.40 ছিল। তখন কম নিশ্চিত "ক্ষতিকর" অনুমানও evidence হয়ে যাচ্ছিল (false positive)। Forensic কাজে ভুল করে কাউকে দোষী দেখানো বিপজ্জনক, তাই কম নিশ্চিত ক্ষেত্রে মানুষের review-তে পাঠানো হয়।

সব figure `outputs/figures/`-এ 300 dpi PNG হিসেবে সেভ হয় (থিসিস ডকুমেন্টে বসানোর জন্য)।

(এই ফাইলে `outputs/` লেখা মানে output ফোল্ডার — `SAVE_TO_DRIVE = True` হলে সেটা আসলে `MyDrive/bangla_forensics_outputs/`।)

---

## ৬. Section 1 — Dataset লোড করা

### উৎসগুলো এক নজরে

| Dataset | কী কাজে |
|---|---|
| **Forensic dataset** (৭টা public উৎস মিলিয়ে, নিচে বিস্তারিত) | ৫-class classifier train / validation / test |
| **BanTH** (Romanized বাংলা ↔ বাংলা) | Transliteration শেখা |
| **Nirmol + ToxLex** (বাংলা গালি শব্দ ও বাক্যাংশ) | লুকানো খারাপ শব্দ unmask করা |

### ১.১ Forensic dataset: ৭টা উৎস → একটা ফাইল → ৫ class

আগে শুধু BanHate + BD-SHS ছিল। এখন `dataset/build_dataset.py` নামে আলাদা একটা script **৭টা public dataset** মিলিয়ে একটা ফাইল বানায়: `data/bangla_forensics_dataset_v1.csv.gz` (~8 MB)। নোটবুক শুধু এই তৈরি ফাইলটা পড়ে।

| Dataset | Platform | Row | ব্যবহার |
|---|---|---|---|
| BanHate | YouTube | 19,203 | train / val / test |
| BD-SHS | social media | 50,281 | train / val / test |
| Bangla Online Comments (BOC) | Facebook | 44,001 | train / val / test |
| Belal et al. toxic comments | social media | 16,073 | train / val / test |
| BLP-2023 VITD | YouTube | 6,046 | train / val / test |
| TB-OLID (Romanized বাংলা) | Facebook | 5,000 | **শুধু train** |
| Bangla vulgar corpus | YouTube | 664 | **শুধু train** |

**শেষ দুটো কেন শুধু train-এ?** এগুলো যোগ করা হয়েছে ছোট `offensive` class-কে আরও আসল উদাহরণ দিতে। Validation আর test-এ না দেওয়ায় test set আগের build-এর সাথে হুবহু একই থাকে — পুরনো আর নতুন ফলাফল তুলনা করা যায়।

**`build_dataset.py` কী কী করে (ক্রমানুসারে):**
1. **Label mapping:** প্রতিটা উৎসের নিজস্ব label (যেমন BOC-এর `religious`, BD-SHS-এর `slander`) একটা লিখিত নিয়ম-বই (`dataset/guideline_v1.md`) অনুযায়ী আমাদের ৫টা class-এ আনা হয়। একাধিক ক্ষতিকর label থাকলে সবচেয়ে গুরুতরটা: violence > hate_speech > cyberbully > offensive।
   - **মূল নিয়ম (guideline):** শারীরিক ক্ষতির হুমকি/আহ্বান = `violence`; কোনো **গোষ্ঠী-পরিচয়** (ধর্ম, লিঙ্গ, জাতি, দল) আক্রমণ = `hate_speech`; **একজন নির্দিষ্ট মানুষকে** আক্রমণ = `cyberbully`; কাউকে নির্দিষ্ট না করে অশ্লীল/গালি = `offensive`; বাকি সব (সমালোচনা, প্রশ্ন, প্রশংসা) = `normal`।
2. **Cleaning:** HTML চিহ্ন, অদৃশ্য অক্ষর বাদ; ৪টার কম অক্ষরের লেখা বাদ (293)।
3. **Duplicate বাদ** (উৎসগুলোর মধ্যেও): 2,615টা। আর যে **473টা comment দুই উৎসে আলাদা label পেয়েছে, সেগুলো পুরো বাদ** — কোনটা ঠিক জানা নেই।
   - **কেন duplicate বাদ?** একই comment train আর test দুটোতে থাকলে model "মুখস্থ" উত্তর দেয়, ফলাফল মিথ্যা ভালো দেখায়।
4. **Fixed split** (seed 42, উৎস × label অনুযায়ী stratified): **test 4,000, validation 2,000**, বাকি train। Validation আর test কখনো filter বা balance করা হয় না।
5. **Label cleaning (শুধু train):** **confident learning** দিয়ে সম্ভাব্য ভুল label খুঁজে বাদ দেওয়া (প্রতিটা class থেকে সর্বোচ্চ 15%)।
   - **Sidenote — confident learning:** একটা সহজ model (TF-IDF + logistic regression) 5-fold cross-validation-এ প্রতিটা comment-এর জন্য "এটা কোন class হওয়ার সম্ভাবনা কত" বলে। যদি model খুব নিশ্চিতভাবে অন্য class বলে আর দেওয়া label-এর সম্ভাবনা খুব কম হয়, তবে label-টা সম্ভবত ভুল। (Northcutt et al., 2021)
6. **Balanced core (version C):** প্রতিটা class থেকে সর্বোচ্চ 6,000টা, উৎসগুলো থেকে যতটা সম্ভব সমানভাবে।
7. **Checked test labels:** test-এর ৪,০০০টা comment আবার guideline দেখে label করা (source label না দেখে, "blind") — এটা LLM দিয়ে করা (০.৬ দেখো)।

**Kept row (split অনুযায়ী):**

| split | normal | offensive | cyberbully | hate_speech | violence | মোট |
|---|---|---|---|---|---|---|
| train | 59,946 | 4,353 | 38,270 | 14,990 | 11,076 | 128,635 |
| validation | 932 | 48 | 603 | 238 | 179 | 2,000 |
| test | 1,864 | 94 | 1,206 | 477 | 359 | 4,000 |

**`offensive` এত ছোট কেন?** Guideline অনুযায়ী **একজন মানুষকে লক্ষ্য করা** যেকোনো অপমান `cyberbully`-তে যায়; শুধু লক্ষ্যহীন গালি `offensive` থাকে।

### তিনটা dataset version (`DATA_VERSION`, cell 1.1)

Validation আর test তিনটাতেই **একই** — শুধু training data আলাদা:

| Version | Training data | কেন |
|---|---|---|
| **A** | শুধু BanHate + BD-SHS | পুরনো ২-উৎসের setup — তুলনার ভিত্তি |
| **B** | সব উৎস, সব train row | বেশি data কি সাহায্য করে? |
| **C** (default) | সব উৎস, সম্ভাব্য ভুল label বাদ, **balanced core** (প্রতি class ≤ 6,000) | পরিষ্কার আর balanced data কি সাহায্য করে? |

তিনটা version-এর তুলনা cell 6.6-এ, প্রতিটা ৩টা seed দিয়ে (০.২ দেখো)।

**Data কোথা থেকে আসে:** dataset ফাইলটা **private** GitHub repo-তে, কারণ দুটো উৎসের (Belal, VITD) license প্রকাশের অনুমতি দেয় না। Cell 1.1 Colab Secret-এ রাখা read-only `GITHUB_TOKEN` দিয়ে GitHub API থেকে ফাইলটা নামায়। Token না থাকলে পরিষ্কার error দেয় (আগের মতো synthetic data বানায় না)।

QUICK mode-এ train থেকে ৬,০০০টা নেওয়া হয় **stratified** ভাবে (class-এর অনুপাত একই রেখে); validation আর test সবসময় পুরো থাকে।

### ১.২ Class distribution (Problem 1 দেখানো)

চার্টটা **বেছে নেওয়া version-এর training data** দেখায় (`1_class_distribution.png`)। `build_dataset.py`-এর হিসাব অনুযায়ী:

| Class | Version B (সব train) | Version C (balanced core) |
|---|---|---|
| normal | 59,946 | 6,000 |
| offensive | 4,353 | 3,701 |
| cyberbully | 38,270 | 6,000 |
| hate_speech | 14,990 | 6,000 |
| violence | 11,076 | 6,000 |
| **normal ÷ সবচেয়ে ছোট class** | **≈ 13.8x** | **≈ 1.6x** |

**মানে:** পুরো data-য় Problem 1 খুব স্পষ্ট (normal, offensive-এর ~১৪ গুণ)। Version C-তে dataset বানানোর সময়ই imbalance-এর বেশিরভাগ ঠিক হয়ে যায়; বাকিটা (offensive 3,701 → 6,000) Section 4-এর over-sampling ঠিক করে। ফলে version C-তে model **প্রতি class-এ ঠিক ৬,০০০টা** comment দিয়ে train হয়।

(পুরনো ২-উৎসের run-এ: normal 35,151 (59.2%), cyberbully 2,874 (4.8%) — normal সবচেয়ে ছোট class-এর 12.2 গুণ।)

### ১.৩ Transliteration dataset: BanTH

একই বাক্য Romanized আর বাংলা অক্ষরে — **37,350 জোড়া**। যেমন: `Janwar ta k dore bichar kora hok` ↔ `জানাওয়ার টা কে ধরে বিচার করা হোক`। Download ব্যর্থ হলে ৫টা ছোট built-in উদাহরণ ব্যবহার হয়।

### ১.৪ গালির তালিকা: Nirmol + ToxLex

- **Nirmol** (GitHub): প্রতি লাইনে একটা বাংলা slang শব্দ/বাক্যাংশ — **1,182টা**।
- **ToxLex** (Mendeley Data): 1,959টা বিষাক্ত বাংলা bigram (দুই শব্দের বাক্যাংশ), Excel ফাইলে।

**কেন দরকার?** কেউ যখন শব্দ `*` দিয়ে লুকায়, সেটা প্রায় সবসময় একটা খারাপ শব্দ। তাই unmask করার সময় খারাপ শব্দ/বাক্যাংশকে অগ্রাধিকার দেওয়া হয়।

**ToxLex download-এর ব্যবস্থা:** Mendeley প্রায়ই Colab-কে block করে (403)। তাই কোড প্রথমে Mendeley, তারপর private repo-র `data/ToxLex.xlsx` থেকে (cell 1.1-এর একই token দিয়ে) চেষ্টা করে। ব্রাউজারের মতো User-Agent পাঠানো হয় (কিছু সার্ভার Python-এর default User-Agent block করে)। ফাইলটা আসলেই Excel কিনা তা দেখা হয় — প্রতিটা `.xlsx` ফাইল `PK` দিয়ে শুরু হয়।

> ⚠️ **সেভ করা run-এ ToxLex লোড হয়নি:** Mendeley দিয়েছে 403, আর repo-তে `data/ToxLex.xlsx` নেই। তাই শুধু Nirmol-এর 1,182টা entry ব্যবহার হয়েছে। সমাধান ০.৩-এ। ToxLex না পেলেও নোটবুক থামে না — শুধু Nirmol দিয়ে চলে।

এছাড়া কয়েকটা **seed শব্দ** হাতে লেখা আছে (`kutta → কুত্তা`, `kuttar → কুত্তার`, `baccha → বাচ্চা`, `shala → শালা`, `harami → হারামি` ইত্যাদি), যাতে download ব্যর্থ হলেও novelty-র উদাহরণগুলো সবসময় কাজ করে।

---

## ৭. Section 2 — Text normalisation (Novel)

তিনটা ধাপ: **Clean → Unmask → Transliterate**।

### ২.১ Clean (পরিষ্কার করা)

`clean_text()` যা সরায়:
- HTML চিহ্ন (`&amp;` → `&`)
- Link (`http…`, `www…`)
- Emoji
- অদৃশ্য অক্ষর (zero-width space ইত্যাদি)
- `@name` / `#tag` → শুধু `name` / `tag`
- বারবার বিরামচিহ্ন (`!!!` → `!`)
- অতিরিক্ত space

উদাহরণ: `দারুণ ভিডিও!!! 😂😂 https://fb.com/xyz   #Dhaka` → `দারুণ ভিডিও! Dhaka`

**কেন?** এগুলো classification-এ কোনো অর্থ যোগ করে না, শুধু model-কে বিভ্রান্ত করে।

কিছু ছোট helper-ও আছে: `split_words` (শব্দে ভাগ), `split_punctuation` (যেমন `"kutt*r!` → `"`, `kutt*r`, `!` — যাতে বিরামচিহ্ন lookup নষ্ট না করে), `has_bangla`, `has_roman`।

### ২.২ Unmask (লুকানো অক্ষর আন্দাজ করা)

**মূল ধারণা:** `kutt*r` কে একটা খোঁজার pattern বানানো হয় — প্রতিটা `*` = **১ থেকে ৩টা অজানা অক্ষর**। তারপর জানা সব শব্দের মধ্যে যেগুলো এই pattern-এ মেলে সেগুলো খোঁজা হয়।

**প্রস্তুতি:**
- গালির entry-গুলোকে দুই ভাগ করা হয়: **একক খারাপ শব্দ (BAD_WORDS)** আর **বহু-শব্দের খারাপ বাক্যাংশ (BAD_PHRASES)**।
  - **কেন ভাগ?** `কুত্তার বাচ্চা` কে শব্দে ভাঙলে `বাচ্চা` বা `ছেলে` এর মতো নিরীহ শব্দও "খারাপ শব্দ" হয়ে যেত। তাই বাক্যাংশ বাক্যাংশ হিসেবেই থাকে।
- **VOCAB:** dataset, BanTH-এর দুই দিক, আর সব গালি — এখানে দেখা প্রতিটা শব্দ আর কতবার দেখা গেছে। (সেভ করা run: **139,772** শব্দ, 737 খারাপ শব্দ, 450 খারাপ বাক্যাংশ)
- শব্দগুলো **দৈর্ঘ্য অনুযায়ী গ্রুপ** করা হয় যাতে খোঁজা দ্রুত হয় (শুধু সম্ভাব্য দৈর্ঘ্যের শব্দগুলো দেখা হয়)।

**Unmask-এর নিয়ম (ক্রম অনুযায়ী):**

1. **বাক্যাংশ আগে:** লুকানো শব্দ আর তার **পাশের শব্দ** মিলিয়ে খারাপ বাক্যাংশের তালিকায় খোঁজা হয়। যেমন `কু*ার বা*া` মেলে `কুত্তার বাচ্চা`-র সাথে। একাধিক মিললে যেটার শব্দগুলো সবচেয়ে বেশি দেখা যায় সেটা নেওয়া হয়।
   - **কেন?** পাশের শব্দ context দেয়। শুধু `বা*া` থেকে অনেক কিছু হতে পারে (বাবা, বাসা…), কিন্তু `কু*ার`-এর পাশে থাকলে প্রায় নিশ্চিতভাবে `বাচ্চা`।
2. **তারপর একক শব্দ:** মেলা শব্দগুলোর মধ্যে আগে **খারাপ শব্দ**, তারপর **সবচেয়ে বেশি ব্যবহৃত শব্দ** বেছে নেওয়া হয়।
   - **কেন খারাপ শব্দকে অগ্রাধিকার?** মানুষ সাধারণত খারাপ শব্দই লুকায়।
3. **`k-u-t-t-a` ধরনের লেখা** (অক্ষরের মাঝে `-` বা `/`) শুধু জোড়া লাগানো হয় → `kutta`।
4. জানা অক্ষর ২টার কম হলে আন্দাজ করা হয় না (তথ্য খুব কম)।

**ফলাফল (সেভ করা run):**

| Input | Output |
|---|---|
| `kutt*r bacc*` | `kuttar baccha` |
| `কুত্*ার বাচ্চা` | `কুত্তার বাচ্চা` |
| `কু*ার বা*া` | `কুত্তার বাচ্চা` |
| `তুই একটা হা*ামি` | `তুই একটা হারামি` |
| `k-u-t-t-a` | `kutta` |

**কেন এই পদ্ধতি (কোনো AI model নয়)?** সহজ, দ্রুত, কোনো training লাগে না, আর **বাংলা ও Romanized দুটোতেই** একইভাবে কাজ করে। ব্যাখ্যা করাও সহজ — forensic কাজে "কেন এই সিদ্ধান্ত" বলতে পারা গুরুত্বপূর্ণ।

### ২.৩ Transliterate — শব্দ-অভিধান (BanTH থেকে শেখা)

**অভিধান কীভাবে তৈরি হয়:** BanTH-এ একই বাক্য দুই লিপিতে আছে। যখন দুই দিকে **শব্দ-সংখ্যা সমান**, তখন ধরে নেওয়া হয় Roman-এর i-তম শব্দ = বাংলার i-তম শব্দ। হাজার হাজার বাক্যে এই জোড়াগুলো গুনে প্রতিটা Roman শব্দের জন্য সবচেয়ে বেশিবার আসা বাংলা শব্দটা নেওয়া হয়। Seed শব্দগুলোকে +1000 দেওয়া হয় যাতে সেগুলো সবসময় জেতে।

**Spelling key:** একই বাংলা শব্দ মানুষ অনেকভাবে লেখে — `bhalobasi`, `valobashi`, `bhalobashi`। তাই কিছু নিয়ম দিয়ে সবগুলোকে একটা "key"-তে আনা হয়:

- `bh→v`, `sh→s`, `ph→f`, `kh→k`, `gh→g`, `th→t`, `dh→d`, `ch→c`, `jh→j`, `z→j`, `w→o`, `y→i`, `ee→i`, `oo→u`, `aa→a`
- দ্বিগুণ অক্ষর → একটা (`tt → t`)

ফলে তিনটাই হয়ে যায় `valobasi`, আর একই বাংলা শব্দ পায়।

**সেভ করা run:** 38,589টা Romanized শব্দ, 29,916টা spelling key।

**কীভাবে একটা বাক্য transliterate হয়:**
1. বাক্যে কোনো ইংরেজি অক্ষর না থাকলে যেমন আছে তেমন ফেরত।
2. প্রতিটা শব্দ আগে সরাসরি অভিধানে, না পেলে spelling key দিয়ে খোঁজা হয়।
3. কিছু শব্দ তবুও অজানা থাকলে, **আর বাক্যটা পুরোপুরি Romanized হলে**, BanglaT5 পুরো বাক্যটা অনুবাদ করে (শুধু FULL mode-এ)।

উদাহরণ:
| Input | Output |
|---|---|
| `kuttar baccha` | `কুত্তার বাচ্চা` |
| `Ami tomake bhalobasi` | `আমি তোমাকে ভালোবাসি` |
| `tui ekta kharap manush` | `তুই একটা খারাপ মানুষ` |

**কেন আগে অভিধান, পরে T5?** অভিধান দ্রুত, নির্ভরযোগ্য, আর ভুল করলেও কোথায় ভুল তা বোঝা যায়। T5 অজানা শব্দ সামলাতে পারে কিন্তু ধীর, আর মাঝে মাঝে বানিয়ে লেখে (hallucinate)। তাই T5 শুধু শেষ ভরসা।

### ২.৪ BanglaT5 fine-tune (শুধু FULL mode)

- BanglaT5 একটা **sequence-to-sequence** model: Romanized বাক্য পড়ে বাংলা বাক্য লেখে।
- BanTH থেকে **20,000 জোড়া** (২০০ অক্ষরের কম), **2 epoch**, learning rate `3e-4`, input-এর শুরুতে `"transliterate: "` যোগ করা হয়।
- Padding-কে loss থেকে বাদ রাখা হয় (`-100`)।
- Output তৈরিতে **beam search (4 beams)** — কয়েকটা সম্ভাব্য বাক্য একসাথে বিবেচনা করে সবচেয়ে ভালোটা নেয়।
- একবার train হলে `outputs/banglat5_transliterator`-এ সেভ হয়; পরের run-এ আবার train না করে লোড হয়।
- সেভ করা run: loss 7.09 → 4.78, আর `Ami tomake bhalobasi` → `আমি তোমাকে ভালোবাসি` ✔

**কেন 20,000 জোড়া?** ভালো ফলাফলের জন্য যথেষ্ট, আবার training সময়ও যুক্তিসঙ্গত থাকে।

### ২.৫ পুরো normalisation pipeline

`normalize_text()` = clean → unmask → transliterate, আর প্রতিটা ধাপের ফলাফলও ফেরত দেয় (demo-তে দেখানোর জন্য)।

| input | cleaned | unmasked | transliterated |
|---|---|---|---|
| `kutt*r bacc*` | `kutt*r bacc*` | `kuttar baccha` | `কুত্তার বাচ্চা` |
| `কুত্*ার বাচ্চা` | `কুত্*ার বাচ্চা` | `কুত্তার বাচ্চা` | `কুত্তার বাচ্চা` |
| `Ami tomake bhalobasi` | একই | একই | `আমি তোমাকে ভালোবাসি` |
| `tui ekta h*rami` | একই | `tui ekta harami` | `তুই একটা হারামি` |

**হাতের নোটের দুটো উদাহরণ ঠিক এভাবেই কাজ করছে।**

তারপর **পুরো forensic dataset একইভাবে normalise করা হয়** (`clean_text` কলামে)।
**কেন?** Training-এর সময় model যে ধরনের লেখা দেখে, পরে ব্যবহারের সময়ও সেই ধরনের লেখা পেতে হবে — না হলে model বিভ্রান্ত হয়।

**একটা লক্ষ্য করার বিষয়:** পুরনো ২-উৎসের dataset-এ মাত্র **123টা** comment-এ `*` ছিল আর মাত্র **2টা** পুরোপুরি Romanized (নতুন dataset-এ TB-OLID-এর কারণে Romanized comment অনেক বেশি থাকবে — নতুন run-এ cell 2.5 সংখ্যাটা দেখাবে)। অর্থাৎ training data-তে এগুলো কম; novel অংশের আসল কাজ হলো **বাস্তব social media থেকে আসা নতুন input**-এ (যেখানে এগুলো অনেক বেশি)।

---

## ৮. Section 3 — Blockchain (evidence ledger)

### কী এটা, সহজ ভাষায়

Blockchain হলো **block-এর একটা তালিকা**। প্রতিটা block-এ থাকে কিছু data **আর আগের block-এর hash (আঙুলের ছাপ)**। কেউ পুরনো কোনো block বদলালে তার hash বদলে যায়, পরের block-এর সাথে link ভেঙে যায়, আর বদলটা ধরা পড়ে।

### প্রতিটা block-এ কী থাকে

| Field | মানে |
|---|---|
| `index` | block নম্বর |
| `timestamp` | কখন তৈরি (UTC) |
| `type` | কী ধরনের block |
| `data` | আসল তথ্য |
| `previous_hash` | আগের block-এর hash (প্রথম block-এ 64টা `0`) |
| `nonce` | proof-of-work-এর জন্য একটা সংখ্যা |
| `hash` | উপরের সব কিছুর SHA-256 |

### কীভাবে কাজ করে

- **SHA-256:** যেকোনো লেখা থেকে ৬৪ অক্ষরের একটা ছাপ। লেখায় এক অক্ষর বদলালেও ছাপ পুরো বদলে যায়।
- **Proof-of-work:** block-এর hash `00` দিয়ে শুরু না হওয়া পর্যন্ত `nonce` বাড়ানো হয়। **কেন?** এতে পুরো chain নতুন করে লেখা কষ্টসাধ্য হয় (এখানে difficulty ছোট, শুধু ধারণা দেখানোর জন্য)।
- **`copy.deepcopy(data)`:** block-এ data-র একটা আলাদা কপি রাখা হয়, যাতে পরে বাইরের variable বদলালে block না বদলায়।
- প্রতিটা নতুন block-এর পর পুরো chain `outputs/forensic_blockchain.json`-এ সেভ হয়।
- **`verify()`:** প্রতিটা block-এর hash আবার হিসাব করে মেলায়, আর প্রতিটা link ঠিক আছে কিনা দেখে।
- **Model fingerprint:** model-এর সব weight থেকে একটা SHA-256 — কোনো একটা weight বদলালেও ছাপ বদলায়।

### ৪ ধরনের block (+ GENESIS)

| Block type | কখন তৈরি হয় |
|---|---|
| `GENESIS` | শুরুতে একবার — chain-এর প্রথম block |
| `CLIENT_UPDATE` | কোনো federated client local training শেষ করলে (তার weight-এর fingerprint) |
| `GLOBAL_MODEL` | server FedAvg শেষ করলে (fingerprint + validation score) |
| `EVIDENCE` | কোনো comment যথেষ্ট নিশ্চয়তার সাথে ক্ষতিকর হলে |
| `USER_PROFILE` | কোনো user account-এর profile তৈরি হলে |

**কেন model-এর ধাপগুলোও blockchain-এ?** আদালতে প্রশ্ন উঠতে পারে "কোন model দিয়ে এই সিদ্ধান্ত নেওয়া হয়েছে, আর সেটা কি পরে বদলানো হয়েছে?" — fingerprint দিয়ে প্রমাণ করা যায়।

---

## ৯. Section 4 — Class imbalance ঠিক করা (Problem 1)

### ৪.১ Train / Validation / Test ভাগ (dataset ফাইলেই স্থির)

ভাগটা নোটবুকে হয় না — `build_dataset.py` একবারই করে ফাইলে লিখে রাখে (`split` কলাম)। নোটবুক শুধু পড়ে:

| | সংখ্যা | কাজ |
|---|---|---|
| **Train** | version অনুযায়ী (C-তে ~27,700) | model শেখানো |
| **Validation** | 2,000 | সেরা federated round বাছাই + threshold tuning |
| **Test** | 4,000 | একদম শেষে একবার — সৎ মূল্যায়ন |

**কেন ফাইলে স্থির?** তিনটা dataset version (A, B, C) আর সব seed-এ validation/test **হুবহু একই** থাকে — তাই তুলনা ন্যায্য। নোটবুকে প্রতিবার নতুন করে ভাগ করলে প্রতিটা run আলাদা test set পেত।

**কেন stratified (উৎস × label)?** প্রতিটা ভাগে প্রতিটা উৎস আর class-এর অনুপাত একই থাকে, যাতে ছোট class (offensive: test-এ মাত্র 94টা) কোনো ভাগে হারিয়ে না যায়।

**কেন test set অছোঁয়া?** বাস্তবে normal comment-ই বেশি। Test set balance করলে score কৃত্রিমভাবে ভালো দেখাত — এটা অসৎ হতো।

**Validation set কেন?** সেরা round বাছাই আর threshold tuning — এসব সিদ্ধান্ত test set দেখে নিলে test score আর নিরপেক্ষ থাকে না।

### ৪.২ Over-sampling (শুধু training set-এ, প্রতিটা client-এর ভেতরে)

- প্রথমে training data ৩টা client-এ ভাগ করা হয় (`split_into_clients`)। Baseline আর improved — দুই run-ই **একই client ভাগ** ব্যবহার করে।
- `split_into_clients`: আগে label অনুযায়ী সাজিয়ে তাস বাটার মতো একে একে দেওয়া হয় — ফলে প্রতিটা client-এর class mix প্রায় একই।
- প্রতিটা client **নিজের data-তেই** balance করে — কোনো comment এক client থেকে অন্য client-এ যায় না (federated learning-এর নিয়ম)।
- **কোনো comment বাদ দেওয়া হয় না।** ছোট ক্ষতিকর class-এর comment copy করে বারবার রাখা হয়, যতক্ষণ না সেটা **সবচেয়ে বড় ক্ষতিকর class-এর সমান** হয় (`OVERSAMPLE_TO = 1.0`) — কিন্তু কোনো class তার নিজের আকারের **৩ গুণের বেশি** হয় না (`MAX_OVERSAMPLE = 3`)।

**Version C-তে কী হয় (হিসাব):** প্রতিটা client পায় ~2,000 normal, ~1,234 offensive, ~2,000 করে বাকি class। Offensive 1,234 → 2,000 (×1.6, ৩ গুণের নিচে)। ফলে **প্রতি class ঠিক সমান** — imbalance 1.6x → 1.0x।

**Version B-তে:** cyberbully সবচেয়ে বড় ক্ষতিকর class (38,270)। Offensive (4,353) সর্বোচ্চ ৩ গুণ = 13,059 পর্যন্ত বাড়ে — পুরো সমান হয় না, কারণ ৮ গুণ copy করলে model ওই কয়েকটা comment মুখস্থ করে ফেলত (overfitting)। এজন্যই `MAX_OVERSAMPLE`।

**কেন আগের under-sampling বাদ দেওয়া হলো?** আগে `normal` comment কমিয়ে ফেলা হতো (25,309 → 14,130)। পুরনো dataset-এর দুটো FULL run-এই এতে macro F1 baseline-এর সমান থেকে গেছে (0.6286 বনাম 0.6277, আর 0.6308 বনাম 0.6302)। কারণ: model ~11,000টা normal উদাহরণ হারাত, আর threshold tuning সেই ঝোঁক প্রায় পুরোটা উল্টে দিত (সব ক্ষতিকর class-এর bias ঋণাত্মক এসেছিল)। Over-sampling-এ কোনো data হারায় না। (নোটবুকে এটা এখন এক লাইনে "preliminary runs" হিসেবে লেখা।)

**`OVERSAMPLE_TO` 0.75 থেকে 1.0 কেন?** Version C-তে offensive-কে পুরো সমান করতে মাত্র ×1.6 copy লাগে — মুখস্থ হওয়ার ঝুঁকি কম। আর মুখস্থ ঠেকানোর কাজটা এখন `MAX_OVERSAMPLE = 3` করে।

### Class weight কেন বন্ধ (`USE_CLASS_WEIGHTS = False`)

Class weight মানে loss হিসাবের সময় ছোট class-এর ভুলকে বেশি গুরুত্ব দেওয়া। কোডে এটা আছে (`sqrt(total / (5 × class_count))`, প্রতিটা client **শুধু নিজের data** থেকে হিসাব করে), কিন্তু বন্ধ রাখা হয়েছে।

**কারণ:** re-sampling-এর সাথে class weight একসাথে দিলে imbalance **দুবার** ঠিক করা হয়ে যায়। FULL run-এ দেখা গেছে: model অনেক বেশি comment-কে ক্ষতিকর বলছিল (recall অনেক বেশি, precision কম), `normal`-এর F1 কমেছিল, আর macro F1 বাড়েনি। চাইলে `True` দিয়ে পরীক্ষাটা আবার করা যায়।

---

## ১০. Section 5 — Federated Learning classifier

### ধারণা, সহজ ভাষায়

ধরা যাক ৩টা প্রতিষ্ঠান (যেমন ৩টা তদন্ত সংস্থা) একসাথে একটা model বানাতে চায়, কিন্তু **নিজেদের গোপন data কাউকে দেবে না**। Federated learning-এ data কখনো client ছেড়ে যায় না, শুধু **model-এর weight** যায়।

### প্রতিটা round-এ

1. Server বর্তমান **global model** প্রতিটা client-কে পাঠায়।
2. প্রতিটা client **নিজের data**-তে কিছুক্ষণ train করে।
3. প্রতিটা client শুধু **weight** ফেরত পাঠায় (data নয়)। তার fingerprint blockchain-এ (`CLIENT_UPDATE`)।
4. Server weight-গুলোর গড় করে (**FedAvg**, বেশি data-র client বেশি গুরুত্ব পায়) → নতুন global model, সেটাও blockchain-এ (`GLOBAL_MODEL`)।

সব round শেষে যে round-এ **validation macro F1 সবচেয়ে বেশি** ছিল, সেই model রাখা হয়।

### ৫.১ Helper function-গুলো

| Function | কাজ |
|---|---|
| `make_loader` | comment-কে token id-র batch বানায় (BanglaBERT tokenizer)। |
| `new_model` | BanglaBERT + একটা ৫-class classification layer। |
| `train_local` | এক client-এর local training: AdamW optimizer, mixed precision, **gradient clipping (1.0)** যাতে হঠাৎ খুব বড় update না হয়। |
| `fed_avg` | নতুন weight = Σ (client-এর weight × client-এর data / মোট data)। Integer buffer (যেমন position id) গড় করা হয় না। |
| `predict_proba` | লেখার জন্য প্রতিটা class-এর probability। |
| `adjust_probs` | প্রতিটা class-এর decision threshold সরানো (Section 6.2-তে ব্যাখ্যা)। |
| `evaluate` | accuracy, macro F1, আর প্রতিটা class-এর F1। |

**কেন সব client-এ একই model (BanglaBERT)?** FedAvg-এ weight গড় করতে হলে সবার model-এর গঠন একই হতে হবে।

### ৫.২ Federated training loop (`run_federated`)

প্রতিটা client প্রতিটা round-এ global model-এর একটা কপি নেয় → train করে → weight পাঠায় → blockchain-এ লেখা হয় → server গড় করে → validation-এ মূল্যায়ন → blockchain-এ লেখা → সেরা হলে মনে রাখা। শেষে সেরা round-এর model ফেরত।

GPU memory বাঁচাতে প্রতিটা client-এর পর local model মুছে cache খালি করা হয়।

---

## ১১. Section 6 — Per-class F1 ফলাফল (Problem 2)

### দুটো model তুলনা

| Run | Training data | Loss | সিদ্ধান্ত |
|---|---|---|---|
| **Baseline** | মূল (imbalanced) | সাধারণ cross-entropy | সবচেয়ে বেশি probability-র class |
| **Improved** | ছোট ক্ষতিকর class over-sampled | সাধারণ cross-entropy | validation set-এ tune করা **per-class threshold** |

দুটোই **একই federated setup**-এ train, আর **একই অছোঁয়া test set**-এ পরীক্ষা — তাই তুলনা ন্যায্য।

### ৬.১ Baseline আর Improved train করা

`RUN_BASELINE = True` হলে আগে baseline train হয়, test-এ মূল্যায়ন, তারপর GPU memory খালি করে improved model train হয়। Threshold tuning-এর **আগের** ফলাফলও রাখা হয় (`oversampled_result`), যাতে প্রতিটা ধাপের প্রভাব আলাদা করে দেখা যায়।

### ৬.২ Threshold tuning (validation set-এ)

**সমস্যা:** Imbalanced data-য় train করা model `normal`-কে বেশি probability দেয়।

**সমাধান:** প্রতিটা class-কে একটা ছোট **bias** দেওয়া হয় (log-probability-র সাথে যোগ)। Bias > 0 মানে ওই class বেশিবার predict হবে।

**কীভাবে bias খোঁজা হয় (coordinate search):**
- `normal`-এর bias 0-তে স্থির (শুধু পার্থক্যটাই গুরুত্বপূর্ণ)।
- বাকি ৪টা class-এর প্রতিটার জন্য −2 থেকে +2 পর্যন্ত **41টা মান** একটা একটা করে চেষ্টা করা হয়, যেটায় validation macro F1 সবচেয়ে বেশি সেটা রাখা হয়।
- পুরো প্রক্রিয়া **3 বার** ঘোরানো হয় (একটা class বদলালে অন্যটার সেরা মান বদলাতে পারে)।

**কেন validation set-এ?** Test set দেখে tune করলে test score মিথ্যা ভালো দেখাত। Test set শুধু একেবারে শেষে একবার ব্যবহার হয়।

Final model আর তার bias সেভ হয় `outputs/final_classifier/`-এ (`class_bias.json` সহ)।

### ৬.৩ Per-class F1 টেবিল ও চার্ট

টেবিলে থাকে: `baseline_F1`, `oversampled_F1`, `improved_F1`, আর `change` (improved − baseline), সাথে MACRO F1 আর ACCURACY। সেভ হয় `per_class_f1.csv` আর `3_per_class_f1.png`।

**কেন macro F1, শুধু accuracy না?** Accuracy বিভ্রান্তিকর: test set-এর ~৪৭% normal (পুরনো dataset-এ ৫৯%), তাই সবকিছুকে "normal" বললেও accuracy ~৪৭%! Macro F1 হলো ৫টা class-এর F1-এর সাধারণ গড় — ছোট class খারাপ করলে এটা কমে যায়। Problem 2 ঠিক এটাই চায়: **প্রতিটা class** ভালো করা।

### ৬.৪ Round-ভিত্তিক macro F1 আর confusion matrix

- প্রতিটা round-এর পর validation macro F1-এর line chart (baseline বনাম improved)।
- Test set-এ confusion matrix — কোন class-কে কোন class ভেবে ভুল করছে তা দেখায়।
- চার্ট: `4_rounds_and_confusion_matrix.png`।

### ৬.৫ উৎস অনুযায়ী score + checked label-এ score + run সেভ

তিনটা কাজ:
- **(a) উৎস অনুযায়ী macro F1:** test set-এর প্রতিটা উৎসে (BanHate, BD-SHS, BOC, Belal, VITD) আলাদা score। যে উৎসে যে class আছে শুধু সেগুলো ধরা হয় (যেমন VITD-তে শুধু normal আর violence)। **কেন?** কোন উৎসে model দুর্বল তা দেখা যায় — সাধারণত যে উৎসের label নিয়ম guideline থেকে বেশি আলাদা।
- **(b) Checked label-এ score:** একই prediction-কে দুইভাবে মাপা হয় — উৎসের দেওয়া label দিয়ে, আর guideline দেখে আবার দেওয়া (checked) label দিয়ে। সাথে দেখায় দুই label কত ভাগ মেলে (build-এর হিসাবে **67.8%**)।
  - **কেন দরকার?** উৎসের label-ই যদি এক-তৃতীয়াংশ ক্ষেত্রে guideline-এর সাথে না মেলে, তবে model "ভুল" করছে মনে হলেও আসলে label-টাই হয়তো ভুল। Checked label-এ score বেশি এলে বোঝা যায় সীমাবদ্ধতা model-এ নয়, data-য়।
  - সবচেয়ে কম মিল `hate_speech`-এ (মাত্র 16.1%) — যেমন BOC-এর `religious` comment-গুলো আসলে ধর্ম নিয়ে কথা বা ব্যক্তিকে অপমান, ধর্মীয় গোষ্ঠীকে আক্রমণ নয় (বিস্তারিত `dataset/DATASHEET.md`)।
- **(c) Run সেভ:** `runs/{version}_seed{seed}_{mode}.json` — 6.6-এর তুলনার জন্য।

### ৬.৬ Dataset version তুলনা (A / B / C)

`runs/` ফোল্ডারের সব সেভ করা run পড়ে (শুধু একই RUN_MODE-এর — QUICK আর FULL কখনো মেশানো হয় না), প্রতিটা version-এর জন্য macro F1 আর প্রতিটা class-এর F1-এর **mean ± std** দেখায়। ফাইল: `dataset_version_comparison.csv`।
- **Sidenote — mean ± std:** mean = গড়; std (standard deviation) = seed বদলালে ফলাফল গড় থেকে কতটা ছড়িয়ে যায়। উদাহরণ: C = 0.70 ± 0.004 আর B = 0.69 ± 0.005 হলে পার্থক্য (0.01) ছড়ানোর প্রায় দ্বিগুণ — মোটামুটি আসল। কিন্তু 0.002 পার্থক্য হলে সেটা noise।

### সেভ করা run-এ ফলাফল (পুরনো dataset)

> ⚠️ **গুরুত্বপূর্ণ:** নিচের সংখ্যাগুলো **পুরনো ২-উৎসের dataset (BanHate + BD-SHS) আর under-sampling-এর শেষ FULL run-এর** (৭ round, class weight বন্ধ, threshold tuning চালু)। এখন dataset, split আর balancing সব বদলেছে — তাই **FULL mode-এ আবার চালিয়ে নতুন সংখ্যা** থিসিসে ব্যবহার করতে হবে (০.১)।

শেষ run-এর test set ফলাফল:

| Class | Baseline F1 | Under-sampling | + threshold (improved) | পরিবর্তন |
|---|---|---|---|---|
| normal | 0.8606 | 0.8537 | 0.8583 | −0.0023 |
| offensive | 0.7581 | 0.7490 | 0.7520 | −0.0061 |
| cyberbully | 0.3464 | 0.3430 | 0.3487 | +0.0023 |
| hate_speech | 0.5910 | 0.6045 | 0.6019 | +0.0109 |
| violence | 0.5981 | 0.5954 | 0.5900 | −0.0081 |
| **MACRO F1** | 0.6308 | 0.6291 | 0.6302 | −0.0006 |
| **ACCURACY** | 0.7638 | 0.7585 | 0.7625 | −0.0013 |

(তার আগের run — under-sampling + class weight, ৫ round: macro F1 0.6286 → 0.6277।)

**কীভাবে পড়তে হবে:**
- দুই run-এই পরিবর্তনগুলো ±0.01-এর মধ্যে। একই setup দুবার চালালেই baseline 0.6286 আর 0.6308 এসেছে — অর্থাৎ এত ছোট পার্থক্য **random noise**, উন্নতি বা অবনতি কোনোটাই না। **Problem 2 এখনো সমাধান হয়নি** (checklist-এ ✘)।
- Threshold tuning-এ সব ক্ষতিকর class-এর bias **ঋণাত্মক** এসেছে (−0.2 থেকে −0.6)। মানে under-sampling model-কে বেশি "ক্ষতিকর"-এর দিকে ঠেলে দিয়েছিল, আর tuning সেটা প্রায় পুরো ফিরিয়ে দিয়েছে। শেষে ফল = baseline, শুধু ~11,000 কম normal উদাহরণ দিয়ে। এজন্যই এখন over-sampling।
- **Confusion matrix থেকে আসল সমস্যা:** baseline প্রায় সবকিছুকে "normal" বলে **না** — normal-এর recall 0.85। বড় ভুলগুলো হলো (১) normal comment-কে ক্ষতিকর বলা (7,030-এর মধ্যে 1,086), আর (২) cyberbully আর hate_speech গুলিয়ে ফেলা। Cyberbully-র ক্ষেত্রে normal থেকে ভুল করে আসা 263টা, ঠিক ধরা মাত্র 225টা। এটা সম্ভবত দুই dataset (BanHate, BD-SHS)-এর label দেওয়ার নিয়ম আলাদা হওয়ার ফল — শুধু imbalance নয়।

---

## ১২. Section 7 — একটা comment-এর পুরো বিশ্লেষণ + Live demo

### ৭.১ `analyze_texts()` — পুরো chain

**clean → unmask → transliterate → classify (tune করা threshold সহ) → ক্ষতিকর হলে blockchain**

প্রতিটা লেখার জন্য সিদ্ধান্ত:

| অবস্থা | সিদ্ধান্ত |
|---|---|
| prediction = normal | `normal` |
| ক্ষতিকর আর confidence ≥ 0.60 | `EVIDENCE` → blockchain block তৈরি |
| ক্ষতিকর কিন্তু confidence < 0.60 | `uncertain (review)` → মানুষ দেখবে, সেভ হবে না |

**EVIDENCE block-এ যা থাকে:**
- `original_text` — মূল লেখা (যেমন `kutt*r bacc*`)
- `text_sha256` — মূল লেখার hash। **কেন?** পরে কেউ দাবি করলে যে লেখাটা অন্যরকম ছিল, hash মিলিয়ে প্রমাণ করা যায়।
- `normalized_text` — `কুত্তার বাচ্চা`
- `prediction`, `confidence`
- context (user id, activity type, সময়) — না থাকলে `source: manual_input`

**সেভ করা run-এর ফলাফল:**

| Input | Transliterated | Prediction | Confidence | Decision |
|---|---|---|---|---|
| `kutt*r bacc*` | `কুত্তার বাচ্চা` | offensive | 0.9765 | EVIDENCE (block 41) |
| `কুত্*ার বাচ্চা` | `কুত্তার বাচ্চা` | offensive | 0.9765 | EVIDENCE (block 42) |
| `Ami tomake bhalobasi` | `আমি তোমাকে ভালোবাসি` | normal | 0.9940 | normal |
| `আজ আবহাওয়া খুব সুন্দর` | একই | normal | 0.9943 | normal |

**হাতের নোটের পুরো novelty chain এখানে প্রমাণিত** — দুটো উদাহরণই offensive হয়ে blockchain-এ সেভ হয়েছে।

### ৭.২ Live demo

`MY_COMMENT`-এ যেকোনো লেখা দিয়ে cell চালালে প্রতিটা ধাপ দেখায়। `SAVE_AS_EVIDENCE` দিয়ে blockchain-এ সেভ করা হবে কিনা ঠিক করা যায় (demo-তে সাধারণত `False`, যাতে ledger পরিষ্কার থাকে)।

উদাহরণ: `tui ekta sh*la` → `tui ekta shala` → `তুই একটা শালা` → offensive (0.97)।

চেষ্টা করার মতো আরও: `এই শু*রের বাচ্চা`, `kemon acho bhai`।

---

## ১৩. Section 8 — Facebook user account-এর DF analysis (মূল Requirement)

এই অংশটাই হাতের প্রথম নোটের requirement সরাসরি পূরণ করে।

### Input format

একটা CSV ফাইল, কলাম: `user_id, timestamp, activity_type, text` (`activity_type` = `status` / `share` / `comment`)। Cell 8.1-এ `USER_ACTIVITY_CSV`-তে ফাইলের নাম দিতে হয়।

### Ethics (নৈতিকতা)

- শুধু **public** data, আইনসম্মতভাবে সংগ্রহ করা (যেমন আইনি ক্ষমতাসম্পন্ন investigator)।
- নোটবুক **Facebook scrape করে না।**
- CSV না থাকলে test set-এর **আসল comment** দিয়ে **simulated (কৃত্রিম)** account তৈরি করে, স্পষ্টভাবে "simulated" চিহ্নিত।

**কেন simulation?** আসল মানুষের data অনুমতি ছাড়া নেওয়া নৈতিক ও আইনি সমস্যা। আর simulation-এ আমরা জানি সত্যিকারের উত্তর কী, তাই সিস্টেম ঠিকঠাক ধরছে কিনা যাচাই করা যায়।

### ৮.১ Simulated account

`simulate_user()` কীভাবে কাজ করে:
- শুরু: 2026-01-05, **12 সপ্তাহ**।
- প্রতি সপ্তাহে পোস্ট সংখ্যা **Poisson(5)** — বাস্তবের মতো কখনো বেশি কখনো কম।
- ক্ষতিকর হওয়ার সম্ভাবনা প্রথম সপ্তাহ থেকে শেষ সপ্তাহ পর্যন্ত **সরলরেখায় বদলায়** (`harm_start` → `harm_end`)।
- লেখাগুলো test set-এর আসল comment থেকে নেওয়া।
- প্রতিটা user-এর আলাদা কিন্তু স্থির seed — প্রতিবার একই ফলাফল।
- `true_label` রাখা হয় (শুধু simulation বলে জানা) — যাচাইয়ের জন্য।

| User | আচরণ | activity type-এর সম্ভাবনা (status/share/comment) | প্রত্যাশিত ফল |
|---|---|---|---|
| `user_A` | সাধারণ user, ~5% ক্ষতিকর | 0.5 / 0.3 / 0.2 | **Low** risk |
| `user_B` | ক্ষতিকর 5% থেকে 70%-এ বাড়ে (offensive, hate, violence) | 0.4 / 0.4 / 0.2 | **escalating** trend |
| `user_C` | অর্ধেক ক্ষতিকর (cyberbully, offensive), মূলত অন্যের পোস্টে comment, আর শেষ সপ্তাহে `kutt*r bacc*`, `কুত্*ার বাচ্চা` | 0.1 / 0.1 / 0.8 | **harasser** role |

**About info** (simulated): user_A — joined 2018-04, 320 followers; user_B — 2024-11, 1450; user_C — 2025-08, 45। এগুলো রিপোর্টে context হিসেবে থাকে, score-এ ব্যবহার হয় না।

সেভ করা run-এ activity সংখ্যা:

| user | comment | share | status |
|---|---|---|---|
| user_A | 12 | 15 | 31 |
| user_B | 9 | 24 | 21 |
| user_C | 60 | 2 | 7 |

### ৮.২ প্রতিটা activity বিশ্লেষণ

সব activity `analyze_texts()` দিয়ে যায় (context হিসেবে user id, type, সময়)। ক্ষতিকরগুলো EVIDENCE block হয়। প্রতিটা activity-কে **সপ্তাহ** অনুযায়ী ভাগ করা হয় — এটাই time series-এর ধাপ।

সেভ করা run:
- **181টা activity**, **56টা evidence**, **15টা uncertain**।
- Simulation-এ সত্যিকারের উত্তর জানা, তাই যাচাই: **85.1%** activity-তে ক্ষতিকর/normal সিদ্ধান্ত সঠিক।

| | ক্ষতিকর ধরা পড়েছে | normal ধরা পড়েছে |
|---|---|---|
| **আসলে ক্ষতিকর** | 47 | 18 |
| **আসলে normal** | 9 | 107 |

### ৮.৩ Forensic profile — ৫টা aspect

**Severity:** normal=0, offensive=1, cyberbully=2, hate_speech=3, violence=4 (analyst চাইলে বদলাতে পারেন)।

| Aspect | কীভাবে হিসাব |
|---|---|
| **1. Activity level** | প্রতি সপ্তাহে গড় পোস্ট; কোন ঘণ্টায় সবচেয়ে বেশি active (peak hour); মোট activity |
| **2. Harmful ratio** | ক্ষতিকর activity ÷ মোট activity |
| **3. Harm type & severity** | কোন ধরনের ক্ষতি কতবার; প্রধান ধরন; গড় severity ÷ 4 (০ থেকে ১) |
| **4. Time trend** | প্রতি সপ্তাহের harmful ratio-র মধ্য দিয়ে একটা সরলরেখা (linear fit)। `rise` = প্রথম থেকে শেষ সপ্তাহে ratio কতটা বেড়েছে। rise > 0.15 → **escalating**, < −0.15 → **decreasing**, নইলে **stable** |
| **5. Behaviour channel** | ক্ষতিকর কন্টেন্ট কোন পথে বেশি: status → **author** (নিজে লেখে), share → **spreader** (ছড়ায়), comment → **harasser** (অন্যকে আক্রমণ করে) |

**Profile vector (৭টা সংখ্যা):**
`[posts/week, harm ratio, severity, rise, status ভাগ, share ভাগ, comment ভাগ]`
**কেন vector?** সংখ্যায় প্রকাশ করলে user-দের তুলনা করা, clustering বা পরে অন্য model-এ ব্যবহার করা সহজ।

**Risk score (০–১০০):**

```
harm load = harmful ratio × severity          (প্রতিটা activity-র গড় severity)
risk = 100 × ( 0.5 × harmful ratio  +  0.3 × harm load  +  0.2 × rise (০ থেকে ১-এর মধ্যে সীমিত) )
```

| Score | Level | মানে (stable user, মাঝারি severity 0.5 ধরে) |
|---|---|---|
| < 13 | **Low** | 20%-এর কম activity ক্ষতিকর |
| 13 – 32.5 | **Medium** | 20% থেকে 50% |
| ≥ 32.5 | **High** | অর্ধেক বা তার বেশি activity ক্ষতিকর |

**কেন এই ওজন?** কতটা ক্ষতিকর কাজ করে (ratio) সবচেয়ে গুরুত্বপূর্ণ (৫০%), কতটা গুরুতর (৩০%), আর বাড়ছে কিনা (২০%)। শুধু **বাড়া** গোনা হয় (কমা নয়) — কারণ কমতে থাকা আচরণ ঝুঁকি বাড়ায় না।

**কেন severity-কে harmful ratio দিয়ে গুণ করা হয় (harm load)?** আগের সূত্রে ছিল `0.3 × severity` — ক্ষতিকর পোস্ট কতগুলো তা না দেখে। তখন ১০০টা পোস্টের মধ্যে মাত্র ১টা violence পোস্ট থাকলেও user 30 পয়েন্ট পেত (Medium)। শেষ FULL run-এ সাধারণ user_A এই কারণেই Medium হয়ে গিয়েছিল (21.4, সীমা ছিল 20)। Harm load-এ অল্প ক্ষতিকর পোস্ট = অল্প পয়েন্ট।

**কেন সীমা 20% আর 50%?** Classifier নিজেই test set-এ প্রায় ১৫% normal comment-কে ভুল করে ক্ষতিকর বলে (confusion matrix)। তাই কারো 20%-এর কম activity ক্ষতিকর দেখালে সেটা model-এর ভুল থেকে আলাদা করা যায় না → Low। কোডে সীমাগুলো সরাসরি এই অনুপাত থেকে হিসাব হয় (`RISK_MEDIUM = risk_score(0.20, 0.5, 0)`, `RISK_HIGH = risk_score(0.50, 0.5, 0)`), হাতে বসানো সংখ্যা নয়।

উদাহরণ (user_B, শেষ run): 100 × (0.5×0.278 + 0.3×0.278×0.633 + 0.2×0.765) = 100 × (0.139 + 0.053 + 0.153) ≈ **34.5 → High**।

**শেষ FULL run-এর ফলাফল** (profile-এর সংখ্যাগুলো run থেকে; risk দুই সূত্রেই হিসাব করা):

| user | harmful ratio | প্রধান ধরন | severity | trend | role | পুরনো সূত্র | নতুন সূত্র | প্রত্যাশা |
|---|---|---|---|---|---|---|---|---|
| user_A | 0.138 | offensive | 0.469 | stable | author | 21.4 Medium ✘ | **9.3 Low** ✔ | Low |
| user_B | 0.278 | hate_speech | 0.633 | **escalating** | author | 48.2 High | 34.5 High | escalating ✔ |
| user_C | 0.507 | offensive | 0.400 | stable | **harasser** | 39.7 Medium | 33.7 High | harasser ✔ |

(user_A-কে 5% ক্ষতিকর ধরে বানানো হলেও 13.8% ধরা পড়েছে — classifier-এর false positive। নতুন সূত্রে এখন সে সীমা থেকে অনেক দূরে: 9.3 বনাম 13। নতুন FULL run-এ model বদলাবে, তাই সংখ্যাগুলো একটু বদলাতে পারে।)

### ৮.৪ Time-series চার্ট

প্রতিটা user-এর জন্য দুটো চার্ট:
- **বামে:** প্রতি সপ্তাহে কতগুলো activity, detected class অনুযায়ী রঙ করা (stacked bar)।
- **ডানে:** প্রতি সপ্তাহের harmful ratio + trend line (কালো dashed)।

আর সব user-এর risk score-এর একটা bar chart (Low/Medium/High সীমারেখা সহ)।
ফাইল: `5_timeline_user_A.png`, `…_B.png`, `…_C.png`, `6_user_risk_scores.png`।

### ৮.৫ Profile blockchain-এ + forensic report

প্রতিটা user-এর profile একটা **USER_PROFILE block** হয়, যেখানে থাকে: About info, সময়কাল, ৫টা aspect, profile vector, risk, আর **তার সব EVIDENCE block-এর নম্বর** (link)।
**কেন link?** Profile-এর প্রতিটা দাবির পেছনের প্রমাণ সরাসরি খুঁজে পাওয়া যায়।

রিপোর্টের উদাহরণ (user_C):

```
FORENSIC PROFILE: user_C      (blockchain block #101, hash 00135292ddfc8d8f...)
About            : profile: public, joined: 2025-08, followers: 45, simulated: True
Period           : 2026-01-05 to 2026-03-29
1. Activity      : 69 activities, 5.75 per week, most active at 14:00
2. Harmful ratio : 47.8% of the activities are harmful
3. Harm type     : mostly offensive {'offensive': 22, 'cyberbully': 5, 'hate_speech': 3, 'violence': 3} (severity 0.402)
4. Trend         : stable (harmful ratio changed by -0.07 over the period)
5. Channel       : harasser (harmful comments on others) {'comment': 0.88, 'share': 0.06, 'status': 0.06}
RISK             : 36.0 -> Medium
Evidence blocks  : 33 blocks
```

সব profile `user_profiles.json`-এ সেভ হয়।

---

## ১৪. Section 9 — Blockchain যাচাই, ফাইল সেভ, Requirement checklist

### ৯.১ Integrity check + tamper test

- আসল ledger যাচাই: শেষ FULL run-এ **120টা block, সব valid** (GENESIS 1, CLIENT_UPDATE 42, GLOBAL_MODEL 14, EVIDENCE 60, USER_PROFILE 3)।
- **Tamper test:** chain-এর একটা **কপিতে** কেউ প্রমাণ লুকানোর চেষ্টা করে — block 57-এর prediction `offensive` → `normal` করে দেওয়া হয়।
- ফলাফল: `Block 57 was changed (hash does not match its data)` — **সাথে সাথে ধরা পড়ে।**

**কেন কপিতে?** আসল ledger নষ্ট না করে প্রমাণ দেখানো।

### ৯.২ Output ফাইল

| ফাইল | কী আছে |
|---|---|
| `outputs/forensic_blockchain.json` | পুরো blockchain |
| `outputs/per_class_f1.csv` | Per-class F1 টেবিল |
| `outputs/runs/*.json` | প্রতিটা run-এর score (version, seed, mode অনুযায়ী) |
| `outputs/dataset_version_comparison.csv` | Version A/B/C-এর mean ± std |
| `outputs/user_activity_analysed.csv` | প্রতিটা activity-র বিশ্লেষণ |
| `outputs/user_profiles.json` | সব user profile |
| `outputs/user_profiles_summary.csv` | Profile সারাংশ টেবিল |
| `outputs/requirement_checklist.csv` | Checklist |
| `outputs/figures/*.png` | সব চার্ট (300 dpi) |
| `outputs/final_classifier/` | Final model + class bias |
| `outputs/banglat5_transliterator/` | Train করা BanglaT5 |

### ৯.৩ Requirement checklist

এই run-এর আসল ফলাফল থেকে স্বয়ংক্রিয়ভাবে পূরণ হয় (হাতে লেখা নয়):

| অংশ | Requirement | সেভ করা run |
|---|---|---|
| Requirement | Public account + About info | ✔ |
| Requirement | Monitoring: status / share / comment | ✔ |
| Requirement | Activity-গুলো time series হিসেবে | ✔ (12 সপ্তাহ) |
| Requirement | ৫ aspect-এ profiling → profile vector | ✔ |
| Requirement | Result: activity profile + risk level | ✔ (A: Medium, B: High, C: Medium) |
| Novel | `kutt*r bacc*` → `কুত্তার বাচ্চা` | ✔ |
| Novel | `কুত্*ার বাচ্চা` → `কুত্তার বাচ্চা` | ✔ |
| Novel | দুটোই ক্ষতিকর আর blockchain-এ সেভ | ✔ (offensive 0.98) |
| Novel | Problem 1: imbalance কমেছে | ✔ (12.2x → 6.8x) |
| Novel | Problem 2: macro F1 বেড়েছে | ✘ (0.631 → 0.630) |
| Novel | Problem 2: সব ক্ষতিকর class-এর F1 বেড়েছে | ✘ (offensive, violence কমেছে) |
| Shared | Federated learning (data share হয় না) | ✔ |
| Shared | Blockchain valid + tampering ধরা পড়ে | ✔ |
| Requirement | Simulated user_A → Low | ✘ (Medium, 21.4 — পুরনো risk সূত্র) |
| Requirement | Simulated user_B → escalating | ✔ |
| Requirement | Simulated user_C → harasser | ✔ |

**Checklist ✘ দেখালেও লুকায় না** — এটা ইচ্ছাকৃত, যাতে থিসিসে সৎ ফলাফল যায়। এই তিনটা ✘ under-sampling আর পুরনো risk সূত্রের শেষ run-এর। কোড এখন বদলানো হয়েছে (over-sampling, harm load সূত্র) — FULL mode-এ আবার চালিয়ে দেখতে হবে। নতুন কোডে "Problem 1" check-টা `normal / সবচেয়ে ছোট class` অনুপাত তুলনা করে (over-sampling-এ normal-এর সংখ্যা কমে না)।

---

## ১৫. গুরুত্বপূর্ণ সিদ্ধান্তগুলো আর তার কারণ

| সিদ্ধান্ত | কেন |
|---|---|
| BanglaBERT classifier হিসেবে | বাংলা ভাষায় pretrained, অল্প fine-tuning-এ ভালো ফল দেয়। |
| Unmask-এ rule/pattern, AI না | দ্রুত, training লাগে না, দুই লিপিতে কাজ করে, আর ব্যাখ্যা করা যায় (forensic-এ জরুরি)। |
| Unmask-এ বাক্যাংশ আগে, পাশের শব্দ সহ | Context থাকলে সঠিক উত্তর বেশি আসে। |
| খারাপ শব্দ আর খারাপ বাক্যাংশ আলাদা | বাক্যাংশ ভাঙলে নিরীহ শব্দ "খারাপ" তালিকায় চলে যেত। |
| Transliteration-এ অভিধান আগে, T5 শেষে | অভিধান দ্রুত ও নির্ভরযোগ্য; T5 ধীর আর মাঝে মাঝে বানিয়ে লেখে। |
| Spelling key | একই শব্দের নানা বানান (bh/v, sh/s…) এক করা। |
| পুরো dataset-ও normalise | Training আর ব্যবহারের সময় লেখা একই রকম থাকতে হবে। |
| ৭টা উৎস, একটা লিখিত guideline | একটা ছোট dataset-এর চেয়ে বেশি আর বৈচিত্র্যময় data; সব উৎসের label একই নিয়মে আনা। |
| দুই উৎসে আলাদা label পাওয়া comment পুরো বাদ | কোন label ঠিক জানা নেই — রাখলে model-কে পরস্পরবিরোধী শিক্ষা দেওয়া হতো। |
| Duplicate বাদ (উৎসগুলোর মধ্যেও) | Train আর test-এ একই comment থাকলে ফলাফল মিথ্যা ভালো দেখায়। |
| নতুন দুই উৎস শুধু train-এ | Offensive-এর উদাহরণ বাড়ে, কিন্তু test set আগের মতোই থাকে। |
| Split dataset ফাইলে স্থির, test অছোঁয়া | সব version আর seed-এ একই test — ন্যায্য তুলনা আর সৎ মূল্যায়ন। |
| Confident learning দিয়ে ভুল label বাদ (শুধু train, ≤15%) | ভুল label model-কে ভুল শেখায়; test-এ হাত দিলে মূল্যায়ন অসৎ হতো। |
| ৩টা dataset version × ৩টা seed | কোন data সিদ্ধান্ত আসলে কাজে লাগে, তা noise থেকে আলাদা করা। |
| Checked label-এও score | Label noise কতটা, আর সেটা score-কে কতটা নামায় — দেখানো। |
| শুধু training-এ over-sampling, প্রতিটা client-এর ভেতরে | ছোট class-কে বেশি দেখানো, কোনো data না হারিয়ে; data client ছেড়ে যায় না; test সৎ থাকে। |
| Under-sampling বাদ | দুই FULL run-এ macro F1 বাড়েনি, আর ~11,000 normal উদাহরণ হারাত। |
| `OVERSAMPLE_TO = 1.0`, কিন্তু `MAX_OVERSAMPLE = 3` | Version C-তে সব class পুরো সমান হয়; আর কোনো ছোট class ৩ গুণের বেশি copy হয় না, তাই মুখস্থ (overfitting) হয় না। |
| Class weight বন্ধ | Re-sampling-এর সাথে দিলে দ্বিগুণ সংশোধন → অনেক false "harmful"। |
| Threshold tuning validation-এ | Imbalance-এর বাকি প্রভাব ঠিক করা, test নিরপেক্ষ রেখে। |
| Macro F1 দিয়ে সেরা round বাছাই | Problem 2 প্রতিটা class নিয়ে; accuracy imbalance-এ বিভ্রান্তিকর। |
| FULL-এ ৭ round | ৫ round-এও F1 বাড়ছিল। |
| Federated learning | প্রতিষ্ঠানগুলো গোপন data share না করেই একসাথে model বানাতে পারে। |
| Client-দের একই class mix | প্রতিটা client-এর data যেন একই রকম হয়, training স্থিতিশীল থাকে। |
| Gradient clipping, mixed precision | Training স্থিতিশীল ও দ্রুত। |
| Risk-এ severity × harmful ratio (harm load) | অল্প কয়েকটা গুরুতর পোস্টে সাধারণ user Medium হয়ে যেত না। |
| Risk সীমা 20% / 50% ক্ষতিকর activity থেকে | Classifier নিজেই ~১৫% normal-কে ক্ষতিকর বলে; তার নিচে Low। |
| `EVIDENCE_MIN_CONFIDENCE = 0.60` | কম নিশ্চিত অভিযোগ evidence না করে মানুষের review-তে পাঠানো। |
| মূল লেখার SHA-256 evidence-এ | মূল লেখা পরে বদলানো হয়নি তা প্রমাণ। |
| Model fingerprint blockchain-এ | কোন model দিয়ে সিদ্ধান্ত, আর সেটা বদলানো হয়নি — প্রমাণ। |
| Profile-এ evidence block-এর link | প্রতিটা দাবির পেছনের প্রমাণ খুঁজে পাওয়া যায়। |
| Simulated user, scraping না | নৈতিক ও আইনি; আর উত্তর জানা থাকায় যাচাই করা যায়। |
| Checklist ফলাফল থেকে স্বয়ংক্রিয় | হাতে লিখে "সব ঠিক" দেখানোর সুযোগ নেই — সৎ। |

---

## ১৬. ফলাফল নিয়ে সৎ মূল্যায়ন ও সীমাবদ্ধতা

**যা ভালো কাজ করছে:**
- Novelty chain (unmask → transliterate → classify → blockchain) হাতের নোটের দুটো উদাহরণেই ঠিক কাজ করছে।
- Imbalance কমেছে (under-sampling-এ 12.2x → 6.8x; over-sampling-এ হিসাব অনুযায়ী ≈ 4.8x)।
- user_B-এর escalating trend আর user_C-এর harasser role দুই run-এই ধরা পড়েছে।
- Blockchain পরিবর্তন সাথে সাথে ধরে ফেলে।

**যা মনে রাখতে হবে:**
1. **নোটবুকে এখন কোনো output নেই** (পুরনো output বর্তমান কোডের সাথে মিলত না)। FULL mode-এ আবার চালিয়ে output সহ সেভ করতে হবে (~২ ঘণ্টা, ০.১)।
2. **পুরনো dataset-এ কোনো imbalance পদ্ধতিতে macro F1 বাড়েনি।** Under-sampling + class weight: 0.6286 → 0.6277; under-sampling + threshold: 0.6308 → 0.6302 — দুটোই noise-এর মধ্যে। নতুন dataset (version C) আর over-sampling এখনো FULL-এ পরীক্ষা হয়নি। এবারও না বাড়লে থিসিসে সৎভাবে লিখতে হবে যে সীমাবদ্ধতা imbalance নয়, **label noise** — আর সেটার প্রমাণ এখন হাতে আছে: উৎসের label আর checked label মাত্র 67.8% মেলে।
3. **পুরনো run-এ cyberbully-র F1 কম ছিল (~0.35)।** normal থেকে ভুল করে আসা (263) ঠিক ধরার (225) চেয়ে বেশি ছিল, আর hate_speech-এর সাথে গুলিয়ে যেত। নতুন guideline-এ "একজন মানুষকে আক্রমণ = cyberbully" নিয়মটা স্পষ্ট করা হয়েছে; নতুন dataset-এ cyberbully অনেক বড় (38,270 train) — নতুন run-এ বদলাবে বলে আশা।
4. **ToxLex এখনো লোড হয় না** — `data/ToxLex.xlsx` repo-তে যোগ করতে হবে (০.৩)।
5. **User-রা simulated।** আসল public data (আইনসম্মত) দিয়ে পরীক্ষা করলে থিসিস আরও শক্তিশালী হবে।
6. **Federated learning একটা GPU-তেই simulate করা**, আর client-দের data সমান ভাগ করা (IID)। বাস্তবে প্রতিষ্ঠানগুলোর data আলাদা ধরনের হতে পারে (non-IID), তখন ফল খারাপ হতে পারে।
7. **Blockchain একটা local ফাইল, একটাই node।** যার হাতে ফাইল আছে সে চাইলে পুরো chain নতুন করে বানাতে পারে (difficulty মাত্র "00")। বাস্তবে একাধিক পক্ষের কাছে কপি রাখা বা শেষ hash-টা কোথাও নিরাপদে প্রকাশ করা দরকার।
8. **Training data-য় লুকানো comment খুব কম** (পুরনো dataset-এ 123টা)। Novel অংশের আসল উপকার নতুন বাস্তব input-এ।
11. **Checked test label একটা LLM-এর দেওয়া, মানুষের নয়** (০.৬)। থিসিসে এটা স্পষ্ট লিখতে হবে।
9. **Unmasker VOCAB-নির্ভর** — একদম নতুন শব্দ (vocab-এ নেই) unmask হবে না।
10. **Risk score-এর ওজন (0.5 / 0.3 / 0.2) হাতে ঠিক করা।** সীমাগুলো (13, 32.5) 20% আর 50% ক্ষতিকর activity থেকে হিসাব করা, আর 20%-এর কারণ classifier-এর মাপা false positive হার — তবুও আসল labelled user data দিয়ে calibrate করা নয়।

---

## ১৭. Supervisor-এর সম্ভাব্য প্রশ্ন ও উত্তর

**প্রশ্ন: Accuracy কমলে তবুও improved বলবে কেন?**
উত্তর: test set-এর প্রায় অর্ধেক (~৪৭%) normal, তাই accuracy মূলত normal-এর ফল দেখায়। আমাদের লক্ষ্য প্রতিটা class — তাই macro F1 আর per-class F1 দেখি। তবে সতর্কতা: এই data-য় baseline সবকিছুকে "normal" বলে **না** (পুরনো run-এ normal recall 0.85) — তাই macro F1 না বাড়লে শুধু accuracy কমা দিয়ে improved দাবি করা যাবে না।

**প্রশ্ন: Imbalance ঠিক করেও macro F1 বাড়েনি কেন?**
উত্তর: Confusion matrix দেখায়, বড় ভুলগুলো class-এর সংখ্যার কারণে নয় — normal comment-কে ক্ষতিকর বলা আর cyberbully/hate_speech গুলিয়ে ফেলা। দুই dataset (BanHate, BD-SHS) আলাদা নিয়মে label দিয়েছে, তাই একই ধরনের comment দুই জায়গায় আলাদা label পেতে পারে। Pretrained BanglaBERT আর macro F1 দিয়ে সেরা round বাছাই — এই দুটোই মাঝারি imbalance (12x) নিজেই অনেকটা সামলে নেয়।

**প্রশ্ন: Test set-এ over-sampling করোনি কেন?**
উত্তর: Test set বাস্তবের প্রতিনিধি। সেটা বদলালে score মিথ্যা ভালো দেখাত।

**প্রশ্ন: Threshold tuning কি cheating না?**
উত্তর: না, কারণ এটা শুধু validation set-এ করা হয়েছে; test set শেষে একবারই দেখা হয়েছে।

**প্রশ্ন: Unmask-এর জন্য ML model না কেন?**
উত্তর: Training data নেই (লুকানো শব্দ আর তার আসল রূপের জোড়া), rule-based পদ্ধতি দ্রুত, দুই লিপিতে কাজ করে, আর প্রতিটা সিদ্ধান্ত ব্যাখ্যা করা যায় — forensic-এ এটা জরুরি।

**প্রশ্ন: Blockchain-এর দরকার কী, একটা database-ই তো যথেষ্ট?**
উত্তর: Database-এ কেউ চুপচাপ রেকর্ড বদলাতে পারে। Blockchain-এ বদলালে hash মেলে না, সাথে সাথে ধরা পড়ে (Section 9.1-এ দেখানো)। Evidence-এর chain of custody প্রমাণ হয়।

**প্রশ্ন: Federated learning কেন?**
উত্তর: বিভিন্ন তদন্ত সংস্থা বা প্ল্যাটফর্ম আইনি কারণে user data share করতে পারে না। FL-এ শুধু model weight যায়, data নয়।

**প্রশ্ন: Facebook data কোথা থেকে?**
উত্তর: নোটবুক scrape করে না। আইনসম্মতভাবে সংগ্রহ করা public data CSV-তে দিলে কাজ করে; demo-তে test set-এর আসল comment দিয়ে simulated account।

**প্রশ্ন: এতগুলো dataset মেলালে label-এর মিল থাকে কীভাবে?**
উত্তর: একটা লিখিত guideline (`dataset/guideline_v1.md`) দিয়ে প্রতিটা উৎসের প্রতিটা label-কে আমাদের ৫ class-এ আনা হয়েছে; দুই উৎসে আলাদা label পাওয়া comment বাদ; train-এ সম্ভাব্য ভুল label confident learning দিয়ে বাদ। তবুও মিল নিখুঁত না — checked test label-এর সাথে 67.8% মেলে, আর সেটা আমরা লুকাই না, cell 6.5-এ মেপে দেখাই।

**প্রশ্ন: Checked label কে দিয়েছে?**
উত্তর: একটা LLM (Claude), guideline দেখে, উৎসের label না দেখে (blind)। তাই এটা "gold" label না, দ্বিতীয় একটা মত। পরের ধাপ: ~৩০০টা row দুজন মানুষ দিয়ে label করিয়ে Cohen's kappa দেখা।

**প্রশ্ন: Version C-তে তো data কম (~27,700 বনাম ~128,600) — তবু কেন default?**
উত্তর: কম কিন্তু পরিষ্কার আর balanced। কোনটা ভালো সেটা অনুমান নয় — cell 6.6-এ A/B/C তিনটাই ৩টা seed দিয়ে তুলনা করা হয়। ফলাফলে B ভালো এলে default বদলানো উচিত।

**প্রশ্ন: Version C তো এমনিতেই balanced — তাহলে Problem 1 কোথায়?**
উত্তর: আসল data-য় (version B) normal, offensive-এর ~১৪ গুণ — Problem 1 সেখানে স্পষ্ট। Version C হলো তার একটা সমাধান (data স্তরে), আর over-sampling বাকিটা (offensive 3,701 → 6,000) ঠিক করে। Test set-এ কিন্তু আসল অনুপাতই থাকে।

**প্রশ্ন: Model কম নিশ্চিত হলে কী হয়?**
উত্তর: 0.60-এর কম confidence-এ ক্ষতিকর prediction evidence হয় না, "uncertain (review)" হয় — মানুষ দেখবে। নির্দোষ কাউকে ভুল প্রমাণে দোষী দেখানো এড়াতে।

---

## ১৮. শব্দকোষ (সহজ ভাষায়)

| শব্দ | মানে |
|---|---|
| **DF (Digital Forensics)** | ডিজিটাল প্রমাণ সংগ্রহ, বিশ্লেষণ আর সংরক্ষণ — যাতে আদালতে ব্যবহার করা যায়। |
| **Class / Label** | যে ভাগে একটা comment পড়ে (normal, offensive …)। |
| **Class imbalance** | কোনো class-এ অনেক বেশি উদাহরণ, অন্যগুলোতে কম। |
| **Over-sampling** | ছোট class-এর উদাহরণ copy করে বারবার রাখা, যাতে model সেগুলো বেশি দেখে। কিছু বাদ যায় না। |
| **Under-sampling** | বড় class থেকে কিছু উদাহরণ বাদ দেওয়া (আগের version-এ ব্যবহার হতো)। |
| **Stratified split** | ভাগ করার সময় প্রতিটা ভাগে class-এর অনুপাত একই রাখা। |
| **Train / Validation / Test** | শেখা / সিদ্ধান্ত নেওয়া (tuning) / শেষ পরীক্ষা। |
| **Precision** | Model যেগুলোকে "X" বলেছে তার কতগুলো আসলেই X। |
| **Recall** | আসল X-গুলোর কতগুলো model ধরতে পেরেছে। |
| **F1 score** | Precision আর recall-এর মিলিত গড় (০ থেকে ১)। |
| **Macro F1** | সব class-এর F1-এর সাধারণ গড় — ছোট class-কেও সমান গুরুত্ব দেয়। |
| **Confusion matrix** | কোন class-কে কোন class ভেবে কতবার ভুল হয়েছে তার টেবিল। |
| **Confidence** | Model কতটা নিশ্চিত (০ থেকে ১)। |
| **Threshold / bias** | কোন class বলার জন্য কতটা নিশ্চিত হতে হবে — সরিয়ে কোনো class-কে সহজ বা কঠিন করা যায়। |
| **BERT / BanglaBERT** | ভাষা বোঝার জন্য আগে থেকে শেখানো বড় AI model; BanglaBERT বাংলার জন্য। |
| **T5 / BanglaT5** | লেখা পড়ে নতুন লেখা তৈরি করে (যেমন অনুবাদ) এমন model। |
| **Fine-tune** | আগে শেখানো model-কে আমাদের নির্দিষ্ট কাজে আরও একটু শেখানো। |
| **Token** | Model যে ছোট টুকরোয় লেখা পড়ে (শব্দ বা শব্দাংশ)। |
| **Epoch** | পুরো training data একবার দেখা। |
| **Transliteration** | এক লিপি থেকে আরেক লিপিতে লেখা (`ami` → `আমি`)। অনুবাদ নয়, ভাষা একই থাকে। |
| **Romanized Bangla** | ইংরেজি অক্ষরে বাংলা লেখা। |
| **Unmask** | `*` দিয়ে লুকানো অক্ষর আন্দাজ করা। |
| **Regex / pattern** | লেখার মধ্যে নির্দিষ্ট ধরন খোঁজার নিয়ম। |
| **Federated Learning (FL)** | Data না সরিয়ে একাধিক জায়গায় একসাথে model শেখানো। |
| **Client / Server** | FL-এ data-র মালিক প্রতিষ্ঠান / যে weight গড় করে। |
| **Round** | FL-এর একটা পুরো চক্র (পাঠানো → local training → গড়)। |
| **FedAvg** | Client-দের weight-এর ভারযুক্ত গড়। |
| **Weight** | Model-এর শেখা সংখ্যাগুলো। |
| **Hash (SHA-256)** | যেকোনো data-র একটা অনন্য "আঙুলের ছাপ"; সামান্য বদলেও পুরো বদলে যায়। |
| **Blockchain** | Block-এর শিকল, প্রতিটা আগেরটার hash ধরে রাখে — বদলালে ধরা পড়ে। |
| **Proof-of-work / nonce** | Hash নির্দিষ্ট রূপে (`00…`) না আসা পর্যন্ত একটা সংখ্যা (nonce) বদলানো — chain বদলানো কঠিন করে। |
| **Genesis block** | Blockchain-এর প্রথম block। |
| **Time series** | সময় অনুযায়ী সাজানো data। |
| **Profile vector** | একজন user-কে কয়েকটা সংখ্যায় প্রকাশ। |
| **Linear fit / slope** | Data-র মধ্য দিয়ে সবচেয়ে মানানসই সরলরেখা / তার ঢাল (বাড়ছে না কমছে)। |
| **Guideline (labelling)** | কোন comment কোন class পাবে, তার লিখিত নিয়ম-বই — সব উৎসে একই নিয়ম প্রয়োগের জন্য। |
| **Confident learning** | একটা model-এর cross-validated prediction দেখে সম্ভাব্য ভুল label খুঁজে বের করার পদ্ধতি। |
| **Cross-validation (5-fold)** | Data ৫ ভাগ করে, প্রতিবার ৪ ভাগে শিখে বাকি ১ ভাগে predict — যাতে প্রতিটা row-এর prediction এমন model থেকে আসে যে ওই row দেখেনি। |
| **Label noise** | Data-র কিছু label ভুল বা অসঙ্গত — model যত ভালোই হোক, score এর উপরে ওঠা কঠিন। |
| **Datasheet** | একটা dataset-এর "পরিচয়পত্র": উৎস, license, প্রক্রিয়া, সীমাবদ্ধতা। |
| **Seed** | Random সংখ্যার শুরুর বিন্দু; একই seed = একই ফলাফল। |
| **Mean ± std** | কয়েকটা run-এর গড় ± সেগুলো কতটা ছড়ানো। |
| **IID / non-IID** | সব client-এর data একই ধরনের (IID) / আলাদা ধরনের (non-IID)। |
| **GitHub token / Colab Secret** | Private repo পড়ার চাবি / Colab-এ চাবিটা নোটবুকে না লিখে লুকিয়ে রাখার জায়গা। |
| **Synthetic / Simulated data** | কৃত্রিমভাবে বানানো data — কোড পরীক্ষা বা demo-র জন্য। |
| **False positive** | নিরীহ জিনিসকে ভুল করে ক্ষতিকর বলা। |
| **Mixed precision (AMP)** | GPU-তে কম precision-এর সংখ্যা ব্যবহার করে দ্রুত training। |
