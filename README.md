
# Responsible AI: FairLens
> *"See the bias before it decides."*

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io)

**Responsible AI: FairLens** is a production-grade algorithmic bias and fairness auditing platform. It empowers data scientists, machine learning engineers, and compliance officers to diagnose discriminatory disparities, uncover covert proxy features using SHAP, apply in-depth mitigation algorithms (pre-, in-, and post-processing), run counterfactual stress tests, and generate certified PDF audit reports and standardized Model Cards.

---

## 🔒 Cardinal Rule: Zero Demo Data & Zero Fake Output
- **No Pre-loaded Data:** FairLens contains **no sample datasets, demo CSVs, placeholder figures, or fake pre-filled metrics**.
- **Empty States:** Before any dataset is uploaded, every evaluation page displays an authoritative empty state guidance card.
- **100% Session-Computed:** Every metric, score, chart, SHAP value, narrative sentence, and exported PDF line is computed dynamically from data provided by the user in that active session.
- **Scientific Integrity:** When mitigating disparities, FairLens states **"bias reduced"**, never "bias removed". Small demographic subgroups ($N < 30$) trigger explicit statistical uncertainty alerts.

---

## 📐 Mathematical Formulation of Fairness Score (0–100)

FairLens synthesizes statistical parity, disparate impact, and error rate equality into a unified, transparent composite metric:

$$\text{Fairness Score} = 100 \times \left( w_1 \cdot \text{DIR\_Score} + w_2 \cdot (1 - \text{DPD}) + w_3 \cdot (1 - \text{EOD}) \right)$$

### Components & Weights
1. **Disparate Impact Ratio Score ($\text{DIR\_Score}$, Weight $w_1 = 0.40$):**
   $$\text{DIR} = \frac{\min_{g} P(\hat{Y}=1 \mid A=g)}{\max_{g} P(\hat{Y}=1 \mid A=g)}$$
   $$\text{DIR\_Score} = \min\left(1.0, \frac{\text{DIR}}{0.80}\right)$$
   Normalized against the statutory **EEOC 80% (Four-Fifths) Rule**. When $\text{DIR} \ge 0.80$, the component reaches full points (1.0).

2. **Demographic Parity Component ($(1 - \text{DPD})$, Weight $w_2 = 0.35$):**
   $$\text{DPD} = \max_{g} P(\hat{Y}=1 \mid A=g) - \min_{g} P(\hat{Y}=1 \mid A=g)$$
   Measures absolute gap in selection rates. Ideal is $0.0$.

3. **Equal Opportunity Component ($(1 - \text{EOD})$, Weight $w_3 = 0.25$):**
   $$\text{EOD} = \max_{g} \text{TPR}_g - \min_{g} \text{TPR}_g$$
   Where $\text{TPR}_g = P(\hat{Y}=1 \mid Y=1, A=g)$. Measures equity in favorable decisions among qualified candidates.

### Score Interpretations
- **80 – 100 (Green / Fair):** Model complies with EEOC four-fifths guideline and exhibits balanced error rates.
- **60 – 79 (Amber / Caution):** Moderate disparity detected; requires algorithmic mitigation or feature audit.
- **0 – 59 (Red / Critical Bias):** Severe disparate impact or parity violation; non-compliant with standard anti-discrimination thresholds.

---

## 📖 Key Metrics in Plain English

| Metric | Simple Explanation | Statutory / Industry Benchmark |
| :--- | :--- | :--- |
| **Selection Rate** | Percentage of applicants within a demographic subgroup receiving a favorable decision. | Compared across subgroups. |
| **Disparate Impact Ratio (DIR)** | Ratio of the least-selected group's acceptance rate to the most-selected group's rate. | **$\ge 0.80$** (US EEOC Uniform Guidelines). |
| **Demographic Parity Difference** | Maximum percentage difference in approval rates between any two demographic groups. | **$\le 10.0\%$** for balanced selection. |
| **Equal Opportunity Difference** | Gap in True Positive Rates (Recall) between groups. Shows whether qualified applicants have equal chances. | **$\le 10.0\%$** across protected groups. |
| **Equalized Odds Difference** | Combined maximum difference in both True Positive and False Positive rates. | **$\le 10.0\%$** across protected groups. |

---

## 🛠️ Architecture & Features

```text
fairlens/
├── app.py                    # Entry point, dark glassmorphism styling, navigation
├── requirements.txt          # Pinned compatible versions
├── README.md                 # Documentation and mathematical formulas
├── .streamlit/config.toml    # Modern dark theme and server configurations
├── pages/
│   ├── 1_Home.py             # Website landing page, hero, pitch, how-it-works
│   ├── 2_Upload_and_Setup.py # File loader, column configuration, validation
│   ├── 3_Data_Audit.py       # Demographic breakdowns, missing values, raw rates
│   ├── 4_Model_Performance.py# Logistic Regression & Random Forest, disaggregated metrics
│   ├── 5_Fairness_Report.py  # Fairness gauge (0-100), 4/5ths rule, intersectional heatmap
│   ├── 6_Why_Is_It_Biased.py # SHAP explanations, proxy feature detection
│   ├── 7_Fix_It.py           # Pre/in/post-processing mitigations & trade-off curves
│   ├── 8_What_If_Simulator.py# Counterfactual record flip, live decision threshold
│   ├── 9_Report_and_Export.py# ReportLab PDF report generation, model card export
│   └── 10_About_Responsible_AI.py # Educational ethics, fairness criteria, legal context
├── core/
│   ├── state.py              # Session state schema, initialization & page guards
│   ├── data_loader.py        # Safe CSV/Excel parsing, type detection, validation
│   ├── audit.py              # Group distribution, missingness, baseline bias
│   ├── model_trainer.py      # Automated training, evaluation, threshold tuning
│   ├── fairness_metrics.py   # Demographic Parity, Disparate Impact, Eq Odds, Fairness Score
│   ├── explain.py            # SHAP explainer wrapper, proxy correlation scanner
│   ├── mitigation.py         # Fairlearn Reweighing, ExponentiatedGradient, ThresholdOptimizer
│   ├── counterfactual.py     # Single-instance perturbation & flip verification
│   ├── report.py             # ReportLab PDF builder with custom layout & styling
│   └── utils.py              # Formatting helpers, color palettes, sanitizers
├── ui/
│   ├── styles.css            # Dark glassmorphism styles, glowing cards, badges
│   └── components.py         # Reusable UI widgets: metric cards, badges, empty states
└── tests/
    ├── conftest.py           # In-memory synthetic fixtures (NEVER saved to disk)
    ├── test_metrics.py       # Hand-verified fairness metric unit tests
    ├── test_pipeline.py      # End-to-end audit -> train -> fix -> report pipeline tests
    └── test_no_demo_data.py  # Linter verifying no CSV/XLSX demo data exists
```

---

## 🚀 Installation & Local Execution

FairLens runs locally with standard Python and does not require external or paid APIs.

### 1. Clone & Set Up Environment
```bash
git clone <repository-url>
cd fairlens

# Create virtual environment (Python 3.11 recommended)
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🧪 Running Automated Tests
All unit and integration tests execute with in-memory fixtures:
```bash
pytest tests/ -v
```
Included test suites:
- `tests/test_metrics.py`: Mathematical verification against hand-calculated ground truth.
- `tests/test_pipeline.py`: Full end-to-end workflow validation.
- `tests/test_no_demo_data.py`: Zero demo data quality gate enforcement.

---

## 🛡️ Ethical Limitations & Future Roadmap
- **Historical Label Bias:** If historical outcomes (e.g., credit defaults or hiring reviews) reflect discriminatory human bias, statistical parity alone cannot resolve systemic injustice.
- **Unmeasured Confounders:** Proxy feature detection identifies correlations within available columns; uncollected variables cannot be detected.
- **Future Roadmap:** Multi-class classification extensions, individual fairness metric bounds, and continuous drift monitoring webhooks.

