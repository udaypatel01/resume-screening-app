# 🎯 AI Resume Screening System

An intelligent resume classification and analysis system powered by machine learning. Categorize resumes, extract skills, analyze experience, and score candidates automatically.

## ✨ Features

- **Single Resume Analysis**: Upload and analyze individual resumes
- **Batch Processing**: Process multiple resumes from ZIP files (50-100+ resumes)
- **Skill Extraction**: Automatically extract technical and soft skills
- **Experience Detection**: Identify years of experience, companies, and current positions
- **Resume Scoring**: Score resumes from 0-100 based on multiple criteria
- **Advanced Filtering**: Filter by category, skills, experience level, and search within results
- **Visual Analytics**: Interactive charts for skill frequency and category distribution
- **Excel Export**: Export results with multiple sheets organized by category

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Virtual environment (recommended)

### Installation Steps

1. **Activate Virtual Environment** (if using one):
   
   On Windows:
   ```bash
   env\Scripts\activate
   ```
   
   On Linux/Mac:
   ```bash
   source env/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
   **Note**: If you encounter issues with spaCy, you may need to download the English model:
   ```bash
   python -m spacy download en_core_web_sm
   ```

3. **Verify Model Files**:
   Ensure these files exist in the `models/` folder:
   - `clf.pkl` (classifier model)
   - `tfidf.pkl` (TF-IDF vectorizer)
   - `encoder.pkl` (label encoder)

### Running the Application

**Start the Streamlit app:**
```bash
streamlit run app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

If it doesn't open automatically, you can manually navigate to the URL shown in the terminal.

## 📖 Usage Guide

### Single Resume Analysis

1. Go to the **"🧾 Single Resume"** tab
2. Upload a resume file (PDF, DOCX, or TXT)
3. Optionally paste a job description for skills matching
4. View the analysis results:
   - Category prediction
   - Skills extraction (technical & soft)
   - Experience analysis
   - Detailed scoring breakdown

### Batch Processing

1. Go to the **"📦 Batch (ZIP)"** tab
2. Upload a ZIP file containing multiple resumes
3. Optionally paste a job description for comparison
4. Watch the progress bar as resumes are processed
5. View results with:
   - Skill frequency charts
   - Category distribution
   - Top 10 candidates
   - Advanced filtering options
6. Export results as CSV or Excel

### Advanced Features

#### Filtering & Search
- **Filter by Category**: Select specific job categories
- **Filter by Experience**: Choose experience levels (Fresher, Junior, Mid-Level, Senior, Expert)
- **Filter by Skills**: Select technical or soft skills
- **Minimum Score**: Set a minimum overall score threshold
- **Search**: Search within file names, skills, companies, and titles

#### Export Options
- **CSV Export**: Download filtered or all results as CSV
- **Excel Export**: Multi-sheet Excel file with:
  - All Results
  - Sheets organized by category
  - Top 10 candidates
  - Summary statistics

## 📁 Project Structure

```
resume_classifier_app/
├── app.py                 # Main Streamlit application
├── batch_processor.py     # Batch processing utilities
├── skill_extractor.py     # Skill extraction logic
├── experience_detector.py # Experience analysis
├── resume_scorer.py       # Scoring algorithms
├── utils.py              # Utility functions
├── config.py             # Configuration settings
├── models/               # ML model files
│   ├── clf.pkl
│   ├── tfidf.pkl
│   └── encoder.pkl
├── outputs/              # Exported results
├── skills_database.json  # Skills database
└── requirements.txt      # Python dependencies
```

## 🔧 Troubleshooting

### Models Not Loading
- Ensure model files exist in the `models/` folder
- Click "🔄 Clear Cache & Reload Models" in the sidebar
- Check the error details in the expandable error section

### OCR Issues (Scanned PDFs)
- Install OCR dependencies (optional):
  ```bash
  pip install pdf2image pytesseract
  ```
- Install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki

### Large Batch Processing
- For batches with 100+ resumes, processing may take several minutes
- The progress bar will show real-time updates
- Results are saved automatically to the `outputs/` folder

### Port Already in Use
If port 8501 is busy, use a different port:
```bash
streamlit run app.py --server.port 8502
```

## 📊 Supported File Formats

- **PDF**: Text-based and scanned PDFs (with OCR)
- **DOCX**: Microsoft Word documents
- **TXT**: Plain text files

**File Size Limit**: 20 MB per file

## 🎯 Scoring Criteria

Resumes are scored (0-100) based on:
- **Skills (40%)**: Number of skills and match with job description
- **Experience (30%)**: Years of experience and level
- **Education (20%)**: Educational qualifications
- **Format (10%)**: Resume structure and formatting quality

## 📝 Notes

- Files are processed securely and not stored permanently
- Results are saved to the `outputs/` folder for batch processing
- The app uses cached models for faster loading

## 🤝 Support

For issues or questions:
1. Check the error details in the expandable sections
2. Verify all dependencies are installed correctly
3. Ensure model files are present and not corrupted

---

**Happy Resume Screening! 🎉**
