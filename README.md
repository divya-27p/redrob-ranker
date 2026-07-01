##  Problem Statement
Recruiters face an overwhelming challenge: parsing through thousands of resumes to find the best fit. **Espresso** is an intelligent candidate ranking system developed for the INDIA.RUNS hackathon. It evaluates, scores, explains, and ranks applicants to identify the Top 100 most suitable candidates for a given job description (JD).

##  Key Features
* **Explainable AI:** Every candidate recommendation includes a clear, transparent explanation for their score, preventing "black-box" hiring.
* **Deterministic Ranking:** Ensures stable results with mathematically sound tie-breaking (via Candidate ID).
* **Robust Data Handling:** Gracefully manages missing values, invalid fields, incomplete resumes, and duplicate information.
* **High-Scale Performance:** Successfully tested and optimized to process datasets of up to 100,000 candidates efficiently.

##  System Architecture
Our end-to-end pipeline ensures accurate feature extraction and fair scoring:

1.  Data Loader: Ingests the candidate dataset (CSV/JSON).
2. Feature Extractor:Pulls key signals: Skills, Experience, Education, Certifications, and Projects.
3. Reasoning Engine: Evaluates extracted features against JD requirements.
4. Scoring Module:Applies custom rule-based weighted scoring.
5. Ranking Engine:Sorts by highest score and applies tie-breakers.
6.  Validation: Ensures no hallucinations and formatting compliance.
7. Submission File: Outputs the final Top 100 in CSV/XLSX format.

## Tech stack
* **Language:** Python
* **Data Processing:** JSON, CSV, XLSX
* **Logic:** Custom Rule-Based Scoring Engine / Feature Extraction Algorithms
* **Version Control:** Git & GitHub
