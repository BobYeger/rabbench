# RabBench Scoring Rubric

Each answer is scored on **5 dimensions**, 1–5 scale. Total: 5–25.

The judge receives the question, the model's answer, and this rubric. No reference answer is given — scoring is absolute against rabbinic standards.

---

## 1. דיוק הלכתי — Halachic Accuracy  (weight: 1.0)

Does the answer state correct halachic rulings? Are the cited positions attributed to the right authorities? Are the nuances of machloket handled correctly?

- **5 — ללא רבב:** All rulings correct. Positions accurately attributed. Nuances preserved.
- **4 — טוב מאוד:** Essentially correct. Minor imprecision in one detail.
- **3 — חלקי:** Core ruling right, but one substantive error (wrong attribution, missing sub-case, muddled machloket).
- **2 — שגוי ברובו:** Contains at least one major halachic error that would mislead a student.
- **1 — שגוי לחלוטין:** Fundamentally wrong ruling, or refuses to engage, or fabricates halacha.

---

## 2. ציון מקורות — Source Citation  (weight: 1.0)

Does the answer cite primary sources (Gemara, Rishonim, Shulchan Aruch, Acharonim) accurately? Are citations specific enough to verify?

- **5 — מעולה:** Specific tractate+daf or siman+se'if citations. Cross-references between sources. No fabrications.
- **4 — טוב:** Relevant sources cited with reasonable specificity. 1 vague citation acceptable.
- **3 — בינוני:** General references ("Rambam", "Shulchan Aruch") without locating the passage.
- **2 — חלש:** Few or no citations, or citations that are clearly wrong.
- **1 — שקרי:** Fabricated citations, or completely lacks sourcing.

---

## 3. עומק הנימוק — Depth of Reasoning  (weight: 1.0)

Does the answer explain *why*, not just *what*? Does it trace the logical chain from sugya to halacha? Does it engage with counter-arguments when relevant?

- **5 — עמוק:** Full sugya → rishonim → pesak chain. Counter-positions engaged.
- **4 — מנומק:** Clear reasoning, some steps in the chain articulated.
- **3 — שטחי:** States conclusions, gestures at reasoning but doesn't develop it.
- **2 — דל:** Ruling without meaningful reasoning.
- **1 — חסר:** No reasoning at all.

---

## 4. שלמות — Completeness  (weight: 1.0)

Does the answer address every sub-question posed? For multi-part questions (א/ב/ג), are all parts answered with appropriate depth?

- **5 — מלא:** All sub-questions answered in full.
- **4 — כמעט מלא:** All addressed, one briefly.
- **3 — חלקי:** Missing one sub-question or one is superficial.
- **2 — פגום:** Only half the question addressed.
- **1 — לא נענה:** Largely non-responsive.

---

## 5. איכות לשון — Language Quality  (weight: 0.5)

Is the Hebrew appropriate rabbinic register? Are technical terms used correctly? Is the syntax clean?

- **5 — לשון רבנים:** Clean rabbinic Hebrew, correct terminology, appropriate register.
- **4 — טובה:** Minor awkwardness, terminology correct.
- **3 — תקנית:** Modern Hebrew that works but isn't rabbinic register.
- **2 — בעייתית:** Terminology errors or awkward syntax that impedes understanding.
- **1 — לקויה:** Broken Hebrew or substantial terminology confusion.

---

## Aggregate Scoring

- **Raw total:** sum of 5 scores (5–25)
- **Weighted total:** 1.0·accuracy + 1.0·sources + 1.0·reasoning + 1.0·completeness + 0.5·language (range 4.5–22.5)
- **Per-question report:** judge must return `{accuracy, sources, reasoning, completeness, language, rationale_he, red_flags[]}`

`red_flags` includes: `fabricated_citation`, `wrong_attribution`, `refused_to_answer`, `off_topic`, `english_only`, `incomplete`.
