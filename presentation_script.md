# FairLens: 2-Minute Presentation Pitch Script

**Title:** Responsible AI: FairLens  
**Tagline:** *"See the bias before it decides."*  
**Audience:** Technical Leadership, Compliance Executives, AI Product Managers  
**Duration:** ~2 minutes (approx. 275 words, 130 WPM)

---

### [0:00 - 0:25] The Problem & Hook
*"Good morning. Machine learning models now decide who gets a mortgage, who gets an interview, and who receives insurance coverage. But when models train on historical data, they often inherit historical prejudice.*

*Worse, teams often fall into the **'Fairness Through Unawareness'** trap—they drop gender or race from the training features, assuming that solves discrimination. It doesn't. Complex models simply reconstruct protected groups through covert proxies like zip codes or job titles."*

---

### [0:25 - 0:55] Introducing FairLens & Zero-Demo Commitment
*"That is why we built **Responsible AI: FairLens**—with the mission: **'See the bias before it decides.'** 

FairLens is an end-to-end, locally hosted bias auditing and mitigation platform. 

A foundational rule of FairLens: **Zero demo data, zero fake output.** Every metric, curve, and report is computed in real time from the user's uploaded data. If data is un-uploaded, our pages show clean empty states. No hardcoded scores, no hand-waving."*

---

### [0:55 - 1:35] Core Capabilities Walkthrough
*"Here is how it works:
1. **Audit:** Upload your dataset. FairLens immediately scans class balance and baseline disparities, flagging demographic subgroups with fewer than 30 records to prevent false statistical confidence.
2. **Performance vs Fairness:** We train Logistic Regression and Random Forest models side-by-side, disaggregating accuracy per group to prove where high overall accuracy hides subgroup failure.
3. **Statutory Standards:** FairLens calculates the EEOC 80% Disparate Impact Ratio, Demographic Parity, and Equal Opportunity gaps, combining them into an animated **0–100 Fairness Score**.
4. **Root Cause & Fix:** Using SHAP and proxy detectors, FairLens explains *why* the model is biased. Then, through Fairlearn pre-, in-, and post-processing mitigations, it safely reduces bias and visualizes the Pareto trade-off curve against accuracy."*

---

### [1:35 - 2:00] Counterfactuals, Governance & Closing
*"Before deployment, our **What-If Simulator** performs counterfactual flip tests—verifying whether changing ONLY an applicant's demographic attribute flips their approval decision.

Finally, with one click, FairLens exports a certified **PDF Audit Report** and an IEEE-standard **Model Card**.

FairLens gives organizations the visibility and governance they need to deploy AI with confidence. Thank you."*

