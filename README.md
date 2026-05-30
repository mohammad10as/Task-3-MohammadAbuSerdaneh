# Tech Stack Recommender 

An end-to-end Content-Based Recommendation Engine that analyzes a user's technical skills and aligns them with ideal career paths. The project bypasses third-party ML frameworks to implement mathematical vector space models completely from scratch.

## ⚙️ How the Pipeline Works

The recommendation system follows a strict 4-step pipeline:
1. **Ingestion:** Parses, cleans, and normalizes user-submitted skills via a dynamic frontend tag interface.
2. **TF-IDF Weighting:** Computes Term Frequency (TF) and Inverse Document Frequency (IDF) over the job role matrix. Specific niche skills score higher, while over-saturated skills are mathematically penalized.
3. **Cosine Similarity:** Maps vectors to a shared vocabulary space and calculates the angular distance:
   $$cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}$$
4. **Filtering:** Sorts results in descending order and returns a filtered Top-N recommendation match percentage.

## 🛠️ Tech Stack & Architecture

* **Backend:** Python (Flask, `math`, `collections.Counter`) — acts as the REST API processing vector calculations.
* **Frontend:** Modern HTML5/CSS3 (Vanilla JS, CSS Variables, Plus Jakarta Sans typography) — featuring an interactive tag input system, pre-configured suggestion chips, and responsive visual progress bars.
* **Fallback Mode:** Includes an automatic Client-Side JavaScript/CLI fallback loop if the Flask server is disconnected.

## 🚀 Getting Started

### 1. Run via Flask (Web Mode)
Ensure you have Flask installed:
```bash
pip install Flask
