# Bangla Social-Media Digital Forensics — সহজ বাংলায় পুরো নোটবুকের ব্যাখ্যা

> ফাইল: `Bangla_Forensics_FL_UserProfiling.ipynb`
> এই লেখায় আছে: **কী চাওয়া হয়েছিল (requirement)**, **কী করা হয়েছে**, **কীভাবে করা হয়েছে**, আর **কেন এভাবে করা হয়েছে**।
> টেকনিক্যাল শব্দগুলো (model, dataset, F1, blockchain ইত্যাদি) ইংরেজিতেই রাখা হয়েছে, কারণ থিসিসে ও মিটিংয়ে এগুলো ইংরেজিতেই বলা হয়। শেষে একটা ছোট শব্দকোষ দেওয়া আছে।

---

## সূচিপত্র

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
| `"QUICK"` | ৬,০০০ comment, ৩ federated round, BanglaT5 train হয় না | ~১০–১৫ মিনিট | ছোট demo |
| `"FULL"` | সব comment, ৭ round, BanglaT5 train হয় | ~১.৫–২ ঘণ্টা | **থিসিসের সংখ্যা এখান থেকে নিতে হবে** |

3. `Runtime → Run all`।

**Supervisor মিটিংয়ে দেখানোর ক্রম** (নোটবুকের demo guide অনুযায়ী):

| ধাপ | Cell | কী দেখায় |
|---|---|---|
| 1 | 1.2 | Problem 1: normal comment অনেক বেশি |
| 2 | 2.5 | Novelty: `kutt*r bacc*` → `kuttar baccha` → `কুত্তার বাচ্চা` |
| 3 | 7.1 | পুরো chain: লুকানো লেখা → class → blockchain evidence |
| 4 | 7.2 | Live demo: যেকোনো comment টাইপ করে বিশ্লেষণ |
| 5 | 6.3 | Problem 2: প্রতিটা class-এর F1, baseline বনাম improved |
| 6 | 8.3 – 8.5 | Requirement: account profile (৫ aspect, time series, risk) |
| 7 | 9.1 | Blockchain বদলানো block ধরে ফেলে |
| 8 | 9.3 | Checklist: প্রতিটা requirement-এর ফলাফল |

---

## ৫. Section 0 — Setup

### কী করা হয়েছে
- **0.1:** দরকারি library install (`transformers`, `sentencepiece`, `datasets`, `kagglehub`, `scikit-learn`, `seaborn`, `openpyxl`)।
- **0.2:** সব import আর মূল setting। এটাই একমাত্র cell যেটা সাধারণত বদলাতে হয়।

### মূল setting-গুলো

| Setting | মান | মানে ও কারণ |
|---|---|---|
| `RUN_MODE` | `"FULL"` | উপরে ব্যাখ্যা করা হয়েছে। |
| `SAVE_TO_DRIVE` | `False` | `True` দিলে ফলাফল Google Drive-এ সেভ হয়, Colab বন্ধ হলেও হারায় না। |
| `SEED` | `42` | প্রতিবার চালালে একই ফলাফল আসার জন্য (reproducibility)। |
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

---

## ৬. Section 1 — Dataset লোড করা

### ৪টা public উৎস

| Dataset | ভাষা | কী কাজে |
|---|---|---|
| **BanHate** (YouTube comment) | বাংলা | ৫-class classifier train |
| **BD-SHS** (social media hate speech, Kaggle) | বাংলা | ৫-class classifier train |
| **BanTH** | Romanized বাংলা ↔ বাংলা | Transliteration শেখা |
| **Nirmol + ToxLex** | বাংলা গালি শব্দ ও বাক্যাংশ | লুকানো খারাপ শব্দ unmask করা |

### ১.১ Forensic dataset: BanHate + BD-SHS → ৫ class

দুটো dataset-এর নিজস্ব label আলাদা, তাই সেগুলোকে আমাদের ৫টা class-এ রূপান্তর (mapping) করা হয়েছে:

**BanHate mapping:**
| মূল category-তে যা আছে | আমাদের class |
|---|---|
| category নেই (খালি) | `normal` |
| "violence" | `violence` |
| religious / gender / origin / political | `hate_speech` |
| personal / body shaming | `cyberbully` |
| বাকি সব | `offensive` |

**BD-SHS mapping:**
| মূল type | আমাদের class |
|---|---|
| callToViolence (এবং এর রূপভেদ) | `violence` |
| gender, religion_slander, gender_religion_slander | `hate_speech` |
| slander | `offensive` |
| খালি | `normal` |

**কেন এই mapping?** গোষ্ঠী (ধর্ম, লিঙ্গ, জাতি, রাজনীতি) লক্ষ্য করে ঘৃণা = hate speech; ব্যক্তিকে লক্ষ্য করে অপমান (চেহারা, ব্যক্তিগত) = cyberbully; সহিংসতার আহ্বান = violence; সাধারণ গালি/অপবাদ = offensive।

তারপর দুটো মিলিয়ে খালি লেখা আর **duplicate বাদ** দেওয়া হয়েছে (একই comment দুবার থাকলে train আর test দুটোতেই চলে যেতে পারে, তাতে ফলাফল কৃত্রিমভাবে ভালো দেখায়)।

**সেভ করা run-এর ফলাফল:** BanHate 19,203 + BD-SHS 40,224 → duplicate বাদে **59,410** comment।

QUICK mode-এ ৬,০০০টা নেওয়া হয় **stratified** ভাবে — মানে class-গুলোর অনুপাত একই থাকে।

**Download ব্যর্থ হলে:** একটা ছোট **synthetic (নকল)** dataset তৈরি হয় যাতে কোড অন্তত চলে, আর বড় করে warning দেখায়। **Synthetic data-র ফলাফল কখনো থিসিসে দেওয়া যাবে না।**

### ১.২ Class distribution (Problem 1 দেখানো)

| Class | সংখ্যা | শতাংশ |
|---|---|---|
| normal | 35,151 | 59.2% |
| offensive | 9,812 | 16.5% |
| cyberbully | 2,874 | 4.8% |
| hate_speech | 5,703 | 9.6% |
| violence | 5,870 | 9.9% |

**`normal` সবচেয়ে ছোট class (cyberbully)-এর চেয়ে 12.2 গুণ বেশি।** এটাই Problem 1। চার্ট সেভ হয় `1_class_distribution.png` নামে।

### ১.৩ Transliteration dataset: BanTH

একই বাক্য Romanized আর বাংলা অক্ষরে — **37,350 জোড়া**। যেমন: `Janwar ta k dore bichar kora hok` ↔ `জানাওয়ার টা কে ধরে বিচার করা হোক`। Download ব্যর্থ হলে ৫টা ছোট built-in উদাহরণ ব্যবহার হয়।

### ১.৪ গালির তালিকা: Nirmol + ToxLex

- **Nirmol** (GitHub): প্রতি লাইনে একটা বাংলা slang শব্দ/বাক্যাংশ — **1,182টা**।
- **ToxLex** (Mendeley Data): 1,959টা বিষাক্ত বাংলা bigram (দুই শব্দের বাক্যাংশ), Excel ফাইলে।

**কেন দরকার?** কেউ যখন শব্দ `*` দিয়ে লুকায়, সেটা প্রায় সবসময় একটা খারাপ শব্দ। তাই unmask করার সময় খারাপ শব্দ/বাক্যাংশকে অগ্রাধিকার দেওয়া হয়।

**ToxLex download-এর ব্যবস্থা:** Mendeley প্রায়ই Colab-কে block করে (403)। তাই কোড প্রথমে Mendeley, তারপর GitHub repo-র `data/ToxLex.xlsx` থেকে চেষ্টা করে। ব্রাউজারের মতো User-Agent পাঠানো হয় (কিছু সার্ভার Python-এর default User-Agent block করে)। ফাইলটা আসলেই Excel কিনা তা দেখা হয় — প্রতিটা `.xlsx` ফাইল `PK` দিয়ে শুরু হয়।

> ⚠️ **সেভ করা run-এ ToxLex লোড হয়নি:** Mendeley দিয়েছে 403, আর GitHub-এ `data/ToxLex.xlsx` নেই (404)। তাই শুধু Nirmol-এর 1,182টা entry ব্যবহার হয়েছে। সমাধান: ব্রাউজারে ফাইলটা নামিয়ে repo-তে `data/ToxLex.xlsx` নামে রাখা, অথবা Colab-এ `ToxLex.xlsx` নামে upload করা।

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

**একটা লক্ষ্য করার বিষয়:** real dataset-এ মাত্র **123টা** comment-এ `*` আছে আর মাত্র **2টা** পুরোপুরি Romanized। অর্থাৎ training data-তে এগুলো কম; novel অংশের আসল কাজ হলো **বাস্তব social media থেকে আসা নতুন input**-এ (যেখানে এগুলো অনেক বেশি)।

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

### ৪.১ Train / Validation / Test ভাগ (stratified)

1. **80% train / 20% test।** Test set-এ **আসল class distribution** থাকে, কিছুই বদলানো হয় না।
2. Train-এর **10%** আলাদা করে **validation set** রাখা হয়।

সেভ করা run: **Train 42,775 | Validation 4,753 | Test 11,882**।

**কেন stratified?** প্রতিটা ভাগে class-গুলোর অনুপাত একই থাকে, যাতে ছোট class (cyberbully) কোনো ভাগে হারিয়ে না যায়।

**কেন test set অছোঁয়া?** বাস্তবে social media-তে normal comment-ই বেশি। Test set বদলালে score কৃত্রিমভাবে ভালো দেখাত — এটা অসৎ হতো।

**Validation set কেন?** সেরা federated round বাছাই আর threshold tuning — এসব সিদ্ধান্ত test set দেখে নিলে test score আর নিরপেক্ষ থাকে না। তাই আলাদা validation set।

### ৪.২ Under-sampling (শুধু training set-এ)

- সব ক্ষতিকর comment রাখা হয়।
- `normal` রাখা হয় সর্বোচ্চ **2 × (সবচেয়ে বড় ক্ষতিকর class)**। (`NORMAL_RATIO = 2.0`)

| Class | আগে | পরে |
|---|---|---|
| normal | 25,309 | **14,130** (= 2 × 7,065) |
| offensive | 7,065 | 7,065 |
| cyberbully | 2,069 | 2,069 |
| hate_speech | 4,106 | 4,106 |
| violence | 4,226 | 4,226 |

**Imbalance: 12.2x → 6.8x।** চার্ট: `2_balancing.png`।

**কেন normal পুরো সমান করা হয়নি?** পুরো সমান করলে অনেক normal উদাহরণ হারাত, model normal ভালোভাবে চিনত না, আর নিরীহ লেখাকেও ক্ষতিকর বলত। আগে `NORMAL_RATIO = 1.5` ছিল — তখন QUICK run-এ সাধারণ user `user_A` কে ভুল করে Medium risk দেখাচ্ছিল। তাই 2.0 করা হয়েছে (normal-এর recall বেশি থাকে, false positive কমে)।

### Class weight কেন বন্ধ (`USE_CLASS_WEIGHTS = False`)

Class weight মানে loss হিসাবের সময় ছোট class-এর ভুলকে বেশি গুরুত্ব দেওয়া। কোডে এটা আছে (`sqrt(total / (5 × class_count))`, প্রতিটা client **শুধু নিজের data** থেকে হিসাব করে), কিন্তু বন্ধ রাখা হয়েছে।

**কারণ:** under-sampling-এর সাথে class weight একসাথে দিলে imbalance **দুবার** ঠিক করা হয়ে যায়। FULL run-এ দেখা গেছে: model অনেক বেশি comment-কে ক্ষতিকর বলছিল (recall অনেক বেশি, precision কম), `normal`-এর F1 কমেছিল, আর macro F1 বাড়েনি। চাইলে `True` দিয়ে পরীক্ষাটা আবার করা যায়।

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
| `split_into_clients` | data ৩ ভাগ করে। আগে label অনুযায়ী সাজিয়ে তাস বাটার মতো একে একে দেওয়া হয় — ফলে প্রতিটা client-এর class mix প্রায় একই। |
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
| **Improved** | normal under-sampled | সাধারণ cross-entropy | validation set-এ tune করা **per-class threshold** |

দুটোই **একই federated setup**-এ train, আর **একই অছোঁয়া test set**-এ পরীক্ষা — তাই তুলনা ন্যায্য।

### ৬.১ Baseline আর Improved train করা

`RUN_BASELINE = True` হলে আগে baseline train হয়, test-এ মূল্যায়ন, তারপর GPU memory খালি করে improved model train হয়। Threshold tuning-এর **আগের** ফলাফলও রাখা হয় (`undersampled_result`), যাতে প্রতিটা ধাপের প্রভাব আলাদা করে দেখা যায়।

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

টেবিলে থাকে: `baseline_F1`, `undersampled_F1`, `improved_F1`, আর `change` (improved − baseline), সাথে MACRO F1 আর ACCURACY। সেভ হয় `per_class_f1.csv` আর `3_per_class_f1.png`।

**কেন macro F1, শুধু accuracy না?** Accuracy বিভ্রান্তিকর: ৫৯% data normal, তাই সবকিছুকে "normal" বললেও accuracy ৫৯%! Macro F1 হলো ৫টা class-এর F1-এর সাধারণ গড় — ছোট class খারাপ করলে এটা কমে যায়। Problem 2 ঠিক এটাই চায়: **প্রতিটা class** ভালো করা।

### ৬.৪ Round-ভিত্তিক macro F1 আর confusion matrix

- প্রতিটা round-এর পর validation macro F1-এর line chart (baseline বনাম improved)।
- Test set-এ confusion matrix — কোন class-কে কোন class ভেবে ভুল করছে তা দেখায়।
- চার্ট: `4_rounds_and_confusion_matrix.png`।

### সেভ করা run-এ ফলাফল

> ⚠️ **গুরুত্বপূর্ণ:** নোটবুকে এখন যে output সেভ আছে সেটা **একটা পুরনো run-এর** — তখন ৫ round ছিল, class weight চালু ছিল, আর threshold tuning ছিল না। বর্তমান কোড (৭ round, class weight বন্ধ, threshold tuning চালু) দিয়ে আবার **FULL mode-এ চালিয়ে নতুন সংখ্যা** থিসিসে ব্যবহার করতে হবে।

পুরনো run-এর test set ফলাফল:

| Class | Baseline F1 | Improved F1 | পরিবর্তন |
|---|---|---|---|
| normal | 0.8616 | 0.8423 | −0.0193 |
| offensive | 0.7560 | 0.7471 | −0.0089 |
| cyberbully | 0.3479 | 0.3629 | **+0.0150** |
| hate_speech | 0.5907 | 0.5915 | **+0.0008** |
| violence | 0.5869 | 0.5948 | **+0.0079** |
| **MACRO F1** | 0.6286 | 0.6277 | −0.0009 |
| **ACCURACY** | 0.7661 | 0.7444 | −0.0217 |

**কীভাবে পড়তে হবে:**
- তিনটা ছোট ক্ষতিকর class (cyberbully, hate_speech, violence)-এর F1 বেড়েছে — imbalance ঠিক করার প্রভাব।
- কিন্তু normal আর offensive একটু কমেছে, তাই macro F1 প্রায় একই (সামান্য কম)। অর্থাৎ **ওই run-এ Problem 2 পুরোপুরি সমাধান হয়নি** (checklist-এ ✘)।
- Accuracy কমা প্রত্যাশিত: baseline প্রায় সবকিছুকে "normal" বলে বেশি accuracy পায়।
- এই ফলাফল দেখেই পরে class weight বন্ধ করা হয় (দ্বিগুণ সংশোধন হচ্ছিল), round ৭ করা হয়, আর **threshold tuning** যোগ করা হয়। নতুন run-এ এগুলোর প্রভাব দেখতে হবে।

Improved model-এর বিস্তারিত (পুরনো run): normal-এর precision 0.91 কিন্তু recall 0.78; ক্ষতিকর class-গুলোর recall precision-এর চেয়ে বেশি (যেমন cyberbully: precision 0.31, recall 0.44)। মানে model ক্ষতিকর জিনিস বেশি ধরছে, কিন্তু কিছু নিরীহ জিনিসকেও ক্ষতিকর বলছে।

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
risk = 100 × ( 0.5 × harmful ratio  +  0.3 × severity  +  0.2 × rise (০ থেকে ১-এর মধ্যে সীমিত) )
```

| Score | Level |
|---|---|
| < 20 | **Low** |
| 20 – 45 | **Medium** |
| ≥ 45 | **High** |

**কেন এই ওজন?** কতটা ক্ষতিকর কাজ করে (ratio) সবচেয়ে গুরুত্বপূর্ণ (৫০%), কতটা গুরুতর (৩০%), আর বাড়ছে কিনা (২০%)। শুধু **বাড়া** গোনা হয় (কমা নয়) — কারণ কমতে থাকা আচরণ ঝুঁকি বাড়ায় না।

উদাহরণ (user_B): 100 × (0.5×0.296 + 0.3×0.688 + 0.2×0.831) = 100 × (0.148 + 0.206 + 0.166) ≈ **52.1 → High**।

**সেভ করা run-এর ফলাফল:**

| user | posts/week | harmful ratio | প্রধান ধরন | severity | trend | role | risk | level | প্রত্যাশা মিলেছে? |
|---|---|---|---|---|---|---|---|---|---|
| user_A | 4.83 | 0.121 | offensive | 0.429 | stable | author | 19.1 | **Low** | ✔ |
| user_B | 4.50 | 0.296 | hate_speech | 0.688 | **escalating** | author | 52.1 | High | ✔ |
| user_C | 5.75 | 0.478 | offensive | 0.402 | stable | **harasser** | 36.0 | Medium | ✔ |

**তিনটা simulated user-ই যেটা দেখানোর জন্য বানানো হয়েছিল, profile ঠিক সেটাই খুঁজে পেয়েছে।**

(লক্ষ্য করুন: user_A-কে 5% ক্ষতিকর ধরে বানানো হলেও 12.1% ধরা পড়েছে — কিছু false positive। তবুও Low-তেই আছে, অল্পের জন্য: 19.1।)

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

- আসল ledger যাচাই: সেভ করা run-এ **102টা block, সব valid** (GENESIS 1, CLIENT_UPDATE 30, GLOBAL_MODEL 10, EVIDENCE 58, USER_PROFILE 3)।
- **Tamper test:** chain-এর একটা **কপিতে** কেউ প্রমাণ লুকানোর চেষ্টা করে — block 41-এর prediction `offensive` → `normal` করে দেওয়া হয়।
- ফলাফল: `Block 41 was changed (hash does not match its data)` — **সাথে সাথে ধরা পড়ে।**

**কেন কপিতে?** আসল ledger নষ্ট না করে প্রমাণ দেখানো।

### ৯.২ Output ফাইল

| ফাইল | কী আছে |
|---|---|
| `outputs/forensic_blockchain.json` | পুরো blockchain |
| `outputs/per_class_f1.csv` | Per-class F1 টেবিল |
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
| Requirement | Result: activity profile + risk level | ✔ (A: Low, B: High, C: Medium) |
| Novel | `kutt*r bacc*` → `কুত্তার বাচ্চা` | ✔ |
| Novel | `কুত্*ার বাচ্চা` → `কুত্তার বাচ্চা` | ✔ |
| Novel | দুটোই ক্ষতিকর আর blockchain-এ সেভ | ✔ (offensive 0.98) |
| Novel | Problem 1: imbalance কমেছে | ✔ (12.2x → 6.8x) |
| Novel | Problem 2: macro F1 বেড়েছে | ✘ (0.629 → 0.628) |
| Novel | Problem 2: সব ক্ষতিকর class-এর F1 বেড়েছে | ✘ (offensive কমেছে) |
| Shared | Federated learning (data share হয় না) | ✔ |
| Shared | Blockchain valid + tampering ধরা পড়ে | ✔ |
| Requirement | Simulated user_A → Low | ✔ |
| Requirement | Simulated user_B → escalating | ✔ |
| Requirement | Simulated user_C → harasser | ✔ |

**Checklist ✘ দেখালেও লুকায় না** — এটা ইচ্ছাকৃত, যাতে থিসিসে সৎ ফলাফল যায়। দুটো ✘ পুরনো run-এর; নতুন কোড (threshold tuning সহ) FULL mode-এ চালিয়ে আবার দেখতে হবে।

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
| Duplicate বাদ | Train আর test-এ একই comment থাকলে ফলাফল মিথ্যা ভালো দেখায়। |
| Stratified split, test অছোঁয়া | সৎ মূল্যায়ন — বাস্তব distribution-এ পরীক্ষা। |
| শুধু training-এ under-sampling | Model-কে normal-এর দিকে ঝুঁকতে না দেওয়া, কিন্তু test সৎ রাখা। |
| `NORMAL_RATIO = 2.0` (1.5 না) | 1.5-এ normal user-কে ভুল করে Medium দেখাচ্ছিল। |
| Class weight বন্ধ | Under-sampling-এর সাথে দিলে দ্বিগুণ সংশোধন → অনেক false "harmful"। |
| Threshold tuning validation-এ | Imbalance-এর বাকি প্রভাব ঠিক করা, test নিরপেক্ষ রেখে। |
| Macro F1 দিয়ে সেরা round বাছাই | Problem 2 প্রতিটা class নিয়ে; accuracy imbalance-এ বিভ্রান্তিকর। |
| FULL-এ ৭ round | ৫ round-এও F1 বাড়ছিল। |
| Federated learning | প্রতিষ্ঠানগুলো গোপন data share না করেই একসাথে model বানাতে পারে। |
| Client-দের একই class mix | প্রতিটা client-এর data যেন একই রকম হয়, training স্থিতিশীল থাকে। |
| Gradient clipping, mixed precision | Training স্থিতিশীল ও দ্রুত। |
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
- Imbalance 12.2x থেকে 6.8x-এ নেমেছে।
- তিনটা simulated user-এর প্রত্যাশিত আচরণ (Low, escalating, harasser) সঠিকভাবে ধরা পড়েছে।
- Blockchain পরিবর্তন সাথে সাথে ধরে ফেলে।

**যা মনে রাখতে হবে:**
1. **নোটবুকের সেভ করা output পুরনো run-এর।** বর্তমান কোড FULL mode-এ আবার চালিয়ে নতুন সংখ্যা নিতে হবে।
2. **পুরনো run-এ macro F1 বাড়েনি** (0.6286 → 0.6277)। ছোট class-গুলো বেড়েছে, কিন্তু normal/offensive কমেছে। Threshold tuning এটা ঠিক করার জন্যই যোগ করা হয়েছে — নতুন run-এ যাচাই করতে হবে।
3. **Cyberbully-র F1 কম (~0.35)।** এটা সবচেয়ে ছোট class, আর অন্য ক্ষতিকর class-এর সাথে গুলিয়ে যায়।
4. **ToxLex লোড হয়নি** — `data/ToxLex.xlsx` repo-তে যোগ করতে হবে।
5. **User-রা simulated।** আসল public data (আইনসম্মত) দিয়ে পরীক্ষা করলে থিসিস আরও শক্তিশালী হবে।
6. **Federated learning একটা GPU-তেই simulate করা**, আর client-দের data সমান ভাগ করা (IID)। বাস্তবে প্রতিষ্ঠানগুলোর data আলাদা ধরনের হতে পারে (non-IID), তখন ফল খারাপ হতে পারে।
7. **Blockchain একটা local ফাইল, একটাই node।** যার হাতে ফাইল আছে সে চাইলে পুরো chain নতুন করে বানাতে পারে (difficulty মাত্র "00")। বাস্তবে একাধিক পক্ষের কাছে কপি রাখা বা শেষ hash-টা কোথাও নিরাপদে প্রকাশ করা দরকার।
8. **Training data-য় লুকানো/Romanized comment খুব কম** (123 আর 2)। Novel অংশের আসল উপকার নতুন বাস্তব input-এ।
9. **Unmasker VOCAB-নির্ভর** — একদম নতুন শব্দ (vocab-এ নেই) unmask হবে না।
10. **Risk score-এর ওজন আর সীমা (20, 45) হাতে ঠিক করা** — analyst বদলাতে পারেন, কিন্তু বৈজ্ঞানিকভাবে calibrate করা নয়।

---

## ১৭. Supervisor-এর সম্ভাব্য প্রশ্ন ও উত্তর

**প্রশ্ন: Accuracy কমল কেন, তবুও improved বলছ কেন?**
উত্তর: Baseline প্রায় সবকিছুকে "normal" বলে accuracy বাড়ায়, কারণ ৫৯% data normal। আমাদের লক্ষ্য প্রতিটা class — তাই macro F1 আর per-class F1 দেখি। ছোট ক্ষতিকর class-গুলোর F1 বেড়েছে।

**প্রশ্ন: Test set-এ under-sampling করোনি কেন?**
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

**প্রশ্ন: Model কম নিশ্চিত হলে কী হয়?**
উত্তর: 0.60-এর কম confidence-এ ক্ষতিকর prediction evidence হয় না, "uncertain (review)" হয় — মানুষ দেখবে। নির্দোষ কাউকে ভুল প্রমাণে দোষী দেখানো এড়াতে।

---

## ১৮. শব্দকোষ (সহজ ভাষায়)

| শব্দ | মানে |
|---|---|
| **DF (Digital Forensics)** | ডিজিটাল প্রমাণ সংগ্রহ, বিশ্লেষণ আর সংরক্ষণ — যাতে আদালতে ব্যবহার করা যায়। |
| **Class / Label** | যে ভাগে একটা comment পড়ে (normal, offensive …)। |
| **Class imbalance** | কোনো class-এ অনেক বেশি উদাহরণ, অন্যগুলোতে কম। |
| **Under-sampling** | বড় class থেকে কিছু উদাহরণ বাদ দেওয়া। |
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
| **Synthetic / Simulated data** | কৃত্রিমভাবে বানানো data — কোড পরীক্ষা বা demo-র জন্য। |
| **False positive** | নিরীহ জিনিসকে ভুল করে ক্ষতিকর বলা। |
| **Mixed precision (AMP)** | GPU-তে কম precision-এর সংখ্যা ব্যবহার করে দ্রুত training। |
