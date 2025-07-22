# 🎬 Netflix User Behavior Analysis & Recommendation System

An interactive web-based application for exploring Netflix content and generating personalized movie and TV show recommendations. Includes advanced visualizations, content-based filtering (TF-IDF), and interactive global insights.

🔗 **Live App**: [fiddayy-netflix.streamlit.app](https://fiddayy-netflix.streamlit.app)  
📂 **Dataset**: [Netflix Movies and TV Shows Dataset, Country Coordinates Dataset]

---

## 📌 Features

### 🔍 Exploratory Data Analysis (EDA)
- Total count of Movies and TV Shows
- Content rating distribution
- Top genres and directors
- Yearly content release trends
- Country-wise production distribution

### 🌍 Interactive Globe
- Pydeck-powered interactive globe visualization
- Hover to view country name, number of Movies, and TV Shows
- Covers all countries in the dataset with accurate coordinates

### 🤖 Recommendation System
- **TF-IDF based Content Recommendation**:
  - Recommends similar titles based on title and description similarity
- **Rule-based Filtering**:
  - Genre-based and rating-based recommendation
- Handles edge cases: unknown titles, empty input, etc.

---

## 🛠️ Tech Stack

| Category         | Tools Used                                |
|------------------|--------------------------------------------|
| **Programming**   | Python 3.10                                |
| **Libraries**     | Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Altair, Pydeck |
| **Deployment**    | Streamlit Cloud                           |
| **Visualization** | Altair, Matplotlib, Seaborn, Pydeck       |

---

## 🚀 How to Run Locally

1. Clone the repository  
   ```bash
   git clone https://github.com/fiddayy/netflix_recommender.git
   cd netflix_recommender

2. Install dependencies
   pip install -r requirements.txt

3. Run the app in the command prompt
   streamlit run app.py

   
