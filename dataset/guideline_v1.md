# Label guideline v1 — Bangla harmful comments (5 classes)

One label per comment. Read the whole comment, decide **who or what is attacked** and **how**,
then use the first rule that matches (most serious first).

| Priority | Label | The comment … | Examples |
|---|---|---|---|
| 1 | `violence` | threatens, wishes or calls for **physical harm**: killing, beating, hanging, burning, rape, expelling people by force, "he should be shot" | হাত-পা কেটে দাও · এদের গুলি করে হত্যা করা হোক · টেন্ডারে এমন পিটা দেওয়ার দরকার · ফাঁসি দেওয়া হোক |
| 2 | `hate_speech` | attacks a **group identity**: religion, gender as a group, ethnicity / origin / country people, political party or its supporters | মালুর বাচ্চা মালুরা · আওয়ামী লীগ মানেই নিকৃষ্ট · হিজরা সমাজের কীট · মেয়েরা সব … |
| 3 | `cyberbully` | attacks **one specific person** (named, addressed as তুই/তুমি, or clearly identifiable): insults, humiliation, body shaming, sexual remarks, mocking/trolling, defamation — with or without swear words | ইভানার কোন চেহারা হইলো · তুই খবিচ তুই ইতর · পাপন কুত্তার বাচ্চা · ময়ুরীর দুদ … · তিশা খুব চালাক শয়তান |
| 4 | `offensive` | uses **vulgar, obscene or abusive language** that is **not** aimed at one specific person or an identity group: swearing at a video/news/situation, at an unnamed crowd or at an institution | বালের নিউজ · সব পাগল ছাগল বসায় রাখছে · পুলিশ জে বিশ্ব মাদার*চোৎ |
| 5 | `normal` | none of the above: opinions, criticism without insults, questions, praise, news, prayers | পাপন এর পাপে বাংলাদেশর ক্রিকেট শেষ · মাশাআল্লাহ কন্ঠ নাকি বাশির সুর |

## Rules for difficult cases

- **Criticism is not harm.** Strong criticism of a person, party or government without insults,
  threats or slurs is `normal` ("দুর্নীতি হচ্ছে", "এই সিদ্ধান্ত ভুল").
- **Calls for legal punishment** ("বিচার চাই", "শাস্তি হোক", "গ্রেফতার করা হোক") are `normal`.
  Calls for **execution or physical punishment** ("ফাঁসি দেওয়া হোক", "পিটিয়ে মারা উচিত") are `violence`.
- **Wishes and curses are not threats.** `violence` needs a threat ("I/we will …") or a call/demand
  ("should be killed / beaten / hanged"). Death wishes and curses ("তুই মরি যা", "আল্লাহর গজব পড়ুক",
  "ধ্বংস হোক") are `cyberbully` when aimed at a person, `hate_speech` when aimed at an identity group,
  otherwise `offensive`.
- **Idioms** for removing someone from a post ("ধাক্কা দিয়ে বের করে দেওয়া হোক" about an official) are not
  physical threats: judge them by the insult they contain.
- **Person inside a group:** an insult to one named politician is `cyberbully`; an insult to the whole
  party or its supporters is `hate_speech`. A few identifiable people (the judges of a show, the actors of
  a drama) count as specific persons (`cyberbully`); an unnamed crowd ("মানুষ", "সবাই") does not (`offensive`).
- **Institutions and professions** (police, journalists, a TV channel) are not identity groups:
  vulgar abuse of them is `offensive`, threats against them are `violence`.
- **Sexual remarks** about a specific person (body comments, propositions, and formulaic obscenities such as
  "তোরে চুদি", "তোর মারে …") are `cyberbully`. A **described sexual assault** — with force, objects or animals —
  or a **call** for one ("… should be done to her", "কুত্তা দিয়া …") is `violence`. Sexist statements about women
  in general are `hate_speech`.
- **Slurs for an identity** used against one person ("শালা মালাউন", "বিহারি জারজ") are `hate_speech`:
  the attack is on the identity.
- **Forced expulsion** of an identity group ("হিন্দুদের দেশ থেকে বের করে দেওয়া হোক") is `violence`;
  a boycott call without force is `hate_speech`.
- **Masked swear words** (`চু*`, `মাদার*চোৎ`, `kutt*r`) count as the full word.
- **Unclear:** too short to judge, not Bangla/Banglish, or meaning cannot be understood — mark `unclear`
  (only used when checking labels, step 9; such rows are excluded from the checked evaluation).

## How the source labels were mapped to these rules (build_dataset.py)

| Source | Original label | Our label |
|---|---|---|
| BanHate | NH | normal |
| | Religious, Gender, Origin, Political | hate_speech |
| | Personal Offence, Body Shaming | cyberbully |
| | Abusive/Violence | violence if the annotators' explanation mentions physical harm or a threat, otherwise offensive |
| BD-SHS | NH | normal |
| | callToViolence (any combination) | violence |
| | religion | hate_speech |
| | gender | cyberbully when the target is a person (ind/male/female), otherwise hate_speech |
| | slander | cyberbully when the target is a person, otherwise offensive |
| Belal et al. | no label | normal |
| | threat | violence |
| | hate, religious | hate_speech |
| | troll, insult | cyberbully |
| | vulgar (only) | offensive |
| Bangla Online Comments | not bully | normal |
| | troll, sexual | cyberbully |
| | religious | hate_speech |
| | threat | violence |
| VITD (BLP-2023) | non-violence | normal |
| | direct violence | violence |
| | passive violence | dropped (mixes abuse and justified violence) |
| TB-OLID (training only) | not offensive | normal |
| | offensive, untargeted | offensive |
| | offensive, at an individual | cyberbully |
| | offensive, at a group | hate_speech if the group is an identity (religion, nation, ethnicity, party, women, LGBT — keyword list in `build_dataset.py`), otherwise offensive (a team, the police, fans) |
| Bangla vulgar corpus (training only) | vulgar drama review | offensive |

A comment with several harmful labels gets the most serious one:
`violence` > `hate_speech` > `cyberbully` > `offensive`.
