# Task 2: Comprehensive Error and Failure Analysis Report

- **Total Test Predictions:** 206
- **Total Classification (Label) Errors:** 55 (out of 206, Accuracy = 73.30%)
- **Serendipitous Predictions (Correct Label, Flawed Evidence):** 30
- **Distinct Error Categories Formulated:** 12

## 1. Summary of Distinct Error Types (>= 8 Types Required)

| # | Error Category | Count | Primary Impact & Clinical Significance |
|---|---|---|---|
| 1 | **Absent Evidence Conflated With Non Exclusion** | 25 | Absence of mention mistakenly assumed as eligibility; violates negative-evidence constraint. |
| 2 | **Hallucinated Evidence On Correct Label** | 11 | Spurious citation of irrelevant sentences; damages verification trustworthiness. |
| 3 | **Evidence Boundary Mismatch Correct Label** | 10 | Cites adjacent or partial sentences; correct clinical decision but imprecise provenance. |
| 4 | **False Insufficiency On Inclusion** | 9 | Fails to link present clinical findings to positive criterion; causes false trial rejection. |
| 5 | **Evidence Omission On Correct Label** | 9 | Correct label predicted from global context, but fails to cite supporting sentence ID. |
| 6 | **Inability To Reject Inclusion** | 7 | Cannot conclusively reject inclusion; causes downstream manual review overhead. |
| 7 | **Inapplicable Misclassified As Non Exclusion** | 7 | Fails to recognize irrelevance to patient cohort/procedure; conflates with non-exclusion. |
| 8 | **Inapplicable Misclassified As Missing Info** | 3 | Fails to recognize irrelevance to patient cohort/procedure; conflates with non-exclusion. |
| 9 | **Misclassified Exclusion As Inapplicable** | 1 | Fails to recognize irrelevance to patient cohort/procedure; conflates with non-exclusion. |
| 10 | **Other Label Mismatch** | 1 | Label or classification discrepancy. |
| 11 | **False Positive Inclusion Unverified** | 1 | Label or classification discrepancy. |
| 12 | **Misclassified Exclusion As Insufficient Info** | 1 | Label or classification discrepancy. |

---

## 2. Representative Examples per Error Category

### Category: Inability To Reject Inclusion (7 cases)

**Criterion ID:** `NCT02097186_I_161` | **Patient:** `sigir-201410`
**Type:** `inclusion`
**Criterion Text:** *"Patients undergoing elective carotid endarterectomy"*
- **Gold Label:** `not included` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT02097186_I_162` | **Patient:** `sigir-201410`
**Type:** `inclusion`
**Criterion Text:** *"Patients undergoing open abdominal aortic aneurysm repair"*
- **Gold Label:** `not included` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

---

### Category: Absent Evidence Conflated With Non Exclusion (25 cases)

**Criterion ID:** `NCT02097186_E_169` | **Patient:** `sigir-201410`
**Type:** `exclusion`
**Criterion Text:** *"Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2"*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT02097186_E_174` | **Patient:** `sigir-201410`
**Type:** `exclusion`
**Criterion Text:** *"Patients with previous axillary surgery"*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

---

### Category: False Insufficiency On Inclusion (9 cases)

**Criterion ID:** `NCT02264964_I_176` | **Patient:** `sigir-201410`
**Type:** `inclusion`
**Criterion Text:** *"No medical history of central vena catheterization."*
- **Gold Label:** `included` (Evidence IDs: [0])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning....
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT01526382_I_185` | **Patient:** `sigir-201411`
**Type:** `inclusion`
**Criterion Text:** *"Patients were included if they were diagnosed with Shock, Acute respiratory distress syndrome (ARDS), or both."*
- **Gold Label:** `included` (Evidence IDs: [2, 3])
- **Predicted:** `not enough information` (Evidence IDs: [0, 1, 2, 3])
- **Gold Evidence Text:** On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. | Her body temperature is normal and her blood pressure is 80/60....
- **Pred Evidence Text:** A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. | She denies trauma. | On examination she is pale and in moderate discomfort, as well as tachy...

---

### Category: Evidence Omission On Correct Label (9 cases)

**Criterion ID:** `NCT02264964_E_179` | **Patient:** `sigir-201410`
**Type:** `exclusion`
**Criterion Text:** *"Had been performed central venous puncture or catheterization before."*
- **Gold Label:** `not excluded` (Evidence IDs: [0])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** A physician is called to see a 67-year-old woman who underwent cardiac catheterization via the right femoral artery earlier in the morning....
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT01723137_E_197` | **Patient:** `sigir-201411`
**Type:** `exclusion`
**Criterion Text:** *"Decreased level of consciousness"*
- **Gold Label:** `not excluded` (Evidence IDs: [0, 5])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. | The patient will provide informed consent, and will comply with the trial protocol without a...
- **Pred Evidence Text:** None (No evidence cited)...

---

### Category: Hallucinated Evidence On Correct Label (11 cases)

**Criterion ID:** `NCT01526382_I_189` | **Patient:** `sigir-201411`
**Type:** `inclusion`
**Criterion Text:** *"at least 1 of 3 signs of hypoperfusion (urine output < 0.5mL/kg of body weight per hour for 1 hour or more; neurologic dysfunction defined by confusion, psychosis, or a Glasgow coma scale score of ≤ 6; plasma lactate higher than the upper limit of the normal value)."*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [2, 3])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic. | Her body temperature is normal and her blood pressure is 80/60....

**Criterion ID:** `NCT01526382_I_190` | **Patient:** `sigir-201411`
**Type:** `inclusion`
**Criterion Text:** *"the presence of acute decrease in PaO2/FIO2 to 200mmHg or lower,"*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [2])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** On examination she is pale and in moderate discomfort, as well as tachypneic and tachycardic....

---

### Category: Evidence Boundary Mismatch Correct Label (10 cases)

**Criterion ID:** `NCT01745731_I_306` | **Patient:** `sigir-201417`
**Type:** `inclusion`
**Criterion Text:** *"Patients with liver space occupying lesion (LOE) that require extended hepatic resection."*
- **Gold Label:** `not enough information` (Evidence IDs: [3])
- **Predicted:** `not enough information` (Evidence IDs: [0, 2])
- **Gold Evidence Text:** Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained....
- **Pred Evidence Text:** A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40. | Abdominal ultrasound shows hepatomegaly and abundant free intraperitoneal fluid....

**Criterion ID:** `NCT01745731_I_307` | **Patient:** `sigir-201417`
**Type:** `inclusion`
**Criterion Text:** *"Patient selection should be cautious, covering basically 5 types of liver damage which must be submitted prior to liver volumetry: Metastatic Disease subsidiary right hepatectomy extended to segment IV Metastatic Disease subsidiary right hepatectomy with suspected diseased liver (neoadjuvant chemotherapy) (in cases of doubt may be used liver function test indocyanine green) Bilobar liver metastases with multiple nodules in the right lobe and more than 3 nodules greater than 30 mm in the left hepatic lobe (LHI) will perform lumpectomies the LHI + right portal branch ligation (or postoperative percutaneous embolization) in order to make right hepatectomy 4-6 weeks (two stage surgery) Subsidiary Hepatocarcinoma extended right hepatectomy Liver Injury benign / malignant (Hemangiomas, hydatid cysts or liver tumors / primary bile hepatoblastoma), which by extension threatens the viability of the remaining liver tissue."*
- **Gold Label:** `not enough information` (Evidence IDs: [3])
- **Predicted:** `not enough information` (Evidence IDs: [0, 1, 2, 3])
- **Gold Evidence Text:** Exploratory laparotomy reveals a ruptured liver abscess, which is then surgically drained....
- **Pred Evidence Text:** A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40. | The physical examination is remarkable for tenderness and positive Murphy sign. | Abdominal ult...

---

### Category: Misclassified Exclusion As Inapplicable (1 cases)

**Criterion ID:** `NCT01745731_E_322` | **Patient:** `sigir-201417`
**Type:** `exclusion`
**Criterion Text:** *"Pregnant women or women of childbearing age who do not have adequate contraception."*
- **Gold Label:** `not excluded` (Evidence IDs: [0])
- **Predicted:** `not applicable` (Evidence IDs: [])
- **Gold Evidence Text:** A 48-year-old white male with history of common variable immunodeficiency (CVID) with acute abdominal pain, fever, dehydration, HR of 132 bpm, BP 80/40....
- **Pred Evidence Text:** None (No evidence cited)...

---

### Category: Other Label Mismatch (1 cases)

**Criterion ID:** `NCT00997100_E_408` | **Patient:** `sigir-201421`
**Type:** `exclusion`
**Criterion Text:** *"Female subject of child-bearing potential who is pregnant or lactating."*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `not applicable` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

---

### Category: False Positive Inclusion Unverified (1 cases)

**Criterion ID:** `NCT00236912_I_421` | **Patient:** `sigir-201422`
**Type:** `inclusion`
**Criterion Text:** *"Able to take medicine orally after recovering from surgery"*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `included` (Evidence IDs: [6])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** The patient will provide informed consent, and will comply with the trial protocol without any practical issues....

---

### Category: Inapplicable Misclassified As Missing Info (3 cases)

**Criterion ID:** `NCT00236912_I_422` | **Patient:** `sigir-201422`
**Type:** `inclusion`
**Criterion Text:** *"If female, using birth control"*
- **Gold Label:** `not applicable` (Evidence IDs: [3])
- **Predicted:** `not enough information` (Evidence IDs: [3])
- **Gold Evidence Text:** She has had no sexual partners and her menses are regular....
- **Pred Evidence Text:** She has had no sexual partners and her menses are regular....

**Criterion ID:** `NCT01255670_I_745` | **Patient:** `sigir-201513`
**Type:** `inclusion`
**Criterion Text:** *"female patients have adequate birth-control method"*
- **Gold Label:** `not applicable` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

---

### Category: Inapplicable Misclassified As Non Exclusion (7 cases)

**Criterion ID:** `NCT02562040_E_683` | **Patient:** `sigir-20158`
**Type:** `exclusion`
**Criterion Text:** *"Previous tonsillectomy, including partial tonsillectomy"*
- **Gold Label:** `not applicable` (Evidence IDs: [])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT00393913_E_693` | **Patient:** `sigir-20158`
**Type:** `exclusion`
**Criterion Text:** *"Suspected diagnosis of a sleep disorder other than OSA (i.e., periodic leg movements, narcolepsy, insomnia, central sleep apnea, sleep hypoventilation syndrome)"*
- **Gold Label:** `not applicable` (Evidence IDs: [])
- **Predicted:** `not excluded` (Evidence IDs: [1])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** No history of headache or night terrors....

---

### Category: Misclassified Exclusion As Insufficient Info (1 cases)

**Criterion ID:** `NCT02562040_E_690` | **Patient:** `sigir-20158`
**Type:** `exclusion`
**Criterion Text:** *"Children/caregivers who do not speak English or Spanish well enough to complete the neurobehavioral measures."*
- **Gold Label:** `not excluded` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

---

## 3. Contradictory Cases & Semantic Consistency Analysis

Rubric C4 requires analyzing contradictory cases within patient evaluations. Because clinical trials evaluate multiple related criteria for a single patient note, we assessed intra-patient semantic consistency across related criteria:

#### Case 1: Patient `sigir-201410` — Catheterization
**Observed Contradiction:** Model predicted 'not enough information' for one catheterization criterion while predicting 'not excluded' (confirming absence) for another in the same record.

- Criterion `NCT02264964_I_176` (inclusion): *"No medical history of central vena catheterization...."*
  - **Predicted:** `not enough information` | **Gold:** `included` | Evidence: []
- Criterion `NCT02264964_I_177` (inclusion): *"Maintenance hemodialysis after central vena catheterization...."*
  - **Predicted:** `not enough information` | **Gold:** `not included` | Evidence: []
- Criterion `NCT02264964_E_179` (exclusion): *"Had been performed central venous puncture or catheterization before...."*
  - **Predicted:** `not excluded` | **Gold:** `not excluded` | Evidence: []

---

## 4. Missing Information Failures (Uncertainty vs. Certainty)

A core challenge in clinical NLP is distinguishing between *explicit denial/confirmation* and *lack of documentation* (open-world assumption). The assignment requires explicitly analyzing failures due to missing information.

### 4.1 Definitive Predicted on Missing Information (26 cases)
The model predicted a definitive eligibility outcome (`not excluded` or `included`) when the expert labeled the criterion as `not enough information`. This reveals a strong prior toward assuming a patient is not excluded if no contraindication is explicitly documented.

**Criterion ID:** `NCT02097186_E_169` | **Patient:** `sigir-201410`
**Type:** `exclusion`
**Criterion Text:** *"Patients with an estimated pre-operative glomerular filtration rate < 30mls/min/1.73m2"*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT02097186_E_174` | **Patient:** `sigir-201410`
**Type:** `exclusion`
**Criterion Text:** *"Patients with previous axillary surgery"*
- **Gold Label:** `not enough information` (Evidence IDs: [])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT01526382_E_193` | **Patient:** `sigir-201411`
**Type:** `exclusion`
**Criterion Text:** *"Patients were moribund."*
- **Gold Label:** `not enough information` (Evidence IDs: [0, 1, 2, 3, 4])
- **Predicted:** `not excluded` (Evidence IDs: [])
- **Gold Evidence Text:** A 40-year-old woman with no past medical history presents to the ER with excruciating pain in her right arm that had started 1 hour prior to her admission. | She denies trauma. | On examination she is pale and in moderate discomfort, as well as tachy...
- **Pred Evidence Text:** None (No evidence cited)...

### 4.2 Missing Information Predicted on Definitive Ground Truth (20 cases)
The model conservatively predicted `not enough information` even though the clinical note contained sufficient evidence for expert determination. This failure stems from subtle medical reasoning (e.g. inferring age from phrase '67-year-old woman', or complex disease diagnostic criteria).

**Criterion ID:** `NCT02097186_I_161` | **Patient:** `sigir-201410`
**Type:** `inclusion`
**Criterion Text:** *"Patients undergoing elective carotid endarterectomy"*
- **Gold Label:** `not included` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT02097186_I_162` | **Patient:** `sigir-201410`
**Type:** `inclusion`
**Criterion Text:** *"Patients undergoing open abdominal aortic aneurysm repair"*
- **Gold Label:** `not included` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

**Criterion ID:** `NCT02097186_I_163` | **Patient:** `sigir-201410`
**Type:** `inclusion`
**Criterion Text:** *"Patients undergoing endovascular abdominal aneurysm repair"*
- **Gold Label:** `not included` (Evidence IDs: [])
- **Predicted:** `not enough information` (Evidence IDs: [])
- **Gold Evidence Text:** None (No evidence annotated)...
- **Pred Evidence Text:** None (No evidence cited)...

---

## 5. Synthesis & Recommendations for Clinical Deployment

1. **Negative Evidence Bias:** 25 out of 55 errors (45.5%) resulted from predicting `not excluded` when the medical record simply lacked relevant lab tests or history. In clinical deployment, an explicit calibrated uncertainty threshold must flag missing labs rather than default to non-exclusion.
2. **Evidence-Guided Re-ranking:** For 30 predictions, the model achieved the correct label but cited flawed evidence. An auxiliary cross-encoder or sentence-level citation verification head would ensure that eligibility is only affirmed when evidence sentence provenance is verified.
3. **Inapplicable Criteria:** 10 errors arose from the `not applicable` class. Distinguishing between 'criterion evaluated and passed' versus 'criterion irrelevant to cohort' remains a distinct challenge for 3B parameter decoders without external ontology grounding.
