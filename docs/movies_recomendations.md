### **Dataset Overview (ml-32m)**  
- **Source**: [MovieLens](http://movielens.org)  
- **Contents**:  
  - 32,000,204 ratings  
  - 2,000,072 tag applications  
  - 87,585 movies  
  - 200,948 users (randomly selected, each rated ≥20 movies)  
- **Timeframe**: Data collected from January 09, 1995, to October 12, 2023  
- **Generated**: October 13, 2023  
- **Files Included**:  
  - `links.csv`, `movies.csv`, `ratings.csv`, `tags.csv`  
- **User Data**:  
  - Only user IDs provided; no demographic information.  
- **Availability**: Publicly downloadable at [GroupLens Datasets](http://grouplens.org/datasets/).  

---
### **1. `movies.csv`**  
- Contains **87,585 movies**.  
- Each movie has:  
  - A unique **Movie ID** (e.g., `1`).  
  - A **title** (e.g., *Toy Story (1995)*).  
  - **Genres** (pipe-separated, e.g., `Adventure|Animation|Children`).  

### **2. `ratings.csv`**  
- Contains **32,000,204 ratings** from **200,948 users**.  
- Each rating includes:  
  - **User ID** (anonymous).  
  - **Movie ID** (matches `movies.csv`).  
  - **Rating** (1–5 stars).  
  - **Timestamp** (when the rating was made).  

### **3. `tags.csv`**  
- Contains **2,000,072 tag applications** (user-generated keywords).  
- Each tag includes:  
  - **User ID** and **Movie ID**.  
  - **Tag text** (e.g., *"sci-fi"*, *"classic"*).  
  - **Timestamp** (when the tag was applied).  

### **4. `links.csv`**  
- Maps MovieLens **Movie IDs** to external databases:  
  - **IMDb** (e.g., `tt0114709` for *Toy Story*).  
  - **TMDb** (The Movie Database, e.g., `862`).  

### **Key Notes**  
- **No user demographics** (only anonymized IDs).  
- **Minimum 20 ratings per user** (ensures meaningful data).  
- **Files are interlinked** via shared IDs (e.g., `ratings.csv` uses `Movie ID` from `movies.csv`).  

This structure supports **recommender systems research**, enabling analysis of user preferences, movie attributes, and collaborative filtering.  

For more details: [MovieLens Dataset Page](http://grouplens.org/datasets/movielens/).

### **Usage License**  
1. **No Endorsement**: Users may not imply endorsement by the University of Minnesota or GroupLens.  
2. **Attribution**: Required in publications using the dataset.  
3. **Redistribution**: Permitted under identical license terms.  
4. **Commercial Use**: Prohibited without explicit permission from GroupLens.  
5. **Disclaimer**:  
   - Data correctness/suitability not guaranteed.  
   - Software scripts provided "as is" (no warranties).  
   - No liability for damages arising from use.