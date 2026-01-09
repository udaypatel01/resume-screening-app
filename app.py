import streamlit as st
import pickle
import docx
import PyPDF2
import re
import os
import io
import zipfile
from collections import Counter

import pandas as pd
import plotly.express as px

from skill_extractor import extract_skills, match_skills_against_job
from experience_detector import analyze_experience
from resume_scorer import score_resume
from batch_processor import (
    process_single_resume,
    build_results_dataframe,
    iterate_zip_members,
    extract_text_from_bytes,
)
from config import ensure_output_dir

# Try to import OCR libraries (optional)
try:
    from pdf2image import convert_from_bytes
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

# Set page config
st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.75rem;
        font-size: 1.1rem;
        border-radius: 8px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .upload-section {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #ddd;
        text-align: center;
        margin: 1rem 0;
    }
    .result-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    .category-badge {
        font-size: 2rem;
        font-weight: bold;
        margin-top: 1rem;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Load models with error handling
@st.cache_resource
def load_models():
    """
    Load the trained classifier, TF-IDF vectorizer, and label encoder.
    Always load from the 'models' folder to match the training setup.
    """
    try:
        svc_model = pickle.load(open('models/clf.pkl', 'rb'))
        tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
        le = pickle.load(open('models/encoder.pkl', 'rb'))
        return svc_model, tfidf, le
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure 'models/clf.pkl', 'models/tfidf.pkl', and 'models/encoder.pkl' exist.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        import traceback
        with st.expander("🔍 View Error Details"):
            st.code(traceback.format_exc())
        return None, None, None

svc_model, tfidf, le = load_models()

# Text cleaning function
def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

# Extract text from PDF with OCR fallback
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        
        # Try regular text extraction first
        for i, page in enumerate(pdf_reader.pages):
            try:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + '\n'
            except Exception:
                continue
        
        # If no text extracted and OCR is available, try OCR
        if (not text.strip()) and OCR_AVAILABLE:
            st.info("🔍 Detected scanned PDF. Using OCR to extract text...")
            file.seek(0)  # Reset file pointer
            
            try:
                # Convert PDF to images
                images = convert_from_bytes(file.read())
                
                # Extract text from each image using OCR
                for i, image in enumerate(images):
                    st.write(f"Processing page {i + 1}/{len(images)}...")
                    ocr_text = pytesseract.image_to_string(image)
                    text += ocr_text + '\n'
                
                if text.strip():
                    st.success("✅ Text extracted successfully using OCR!")
                    
            except Exception as ocr_error:
                raise ValueError(f"OCR extraction failed: {str(ocr_error)}")
        
        # If still no text, raise error
        if not text.strip():
            if OCR_AVAILABLE:
                raise ValueError("Could not extract any text from the PDF. The file might be corrupted or empty.")
            else:
                raise ValueError("This appears to be a scanned PDF. Please install OCR libraries (pdf2image, pytesseract) or upload a text-based PDF.")
        
        return text
        
    except Exception as e:
        raise ValueError(f"PDF extraction failed: {str(e)}")

# Extract text from DOCX
def extract_text_from_docx(file):
    doc = docx.Document(file)
    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
    return text

# Extract text from TXT
def extract_text_from_txt(file):
    try:
        text = file.read().decode('utf-8')
    except UnicodeDecodeError:
        file.seek(0)
        text = file.read().decode('latin-1')
    return text

# File size check (20 MB limit)
def check_file_size(file, max_size_mb=20):
    file_size_mb = file.size / (1024 * 1024)
    return file_size_mb <= max_size_mb, file_size_mb

# Handle file upload
def handle_file_upload(uploaded_file):
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif file_extension == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif file_extension == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type.")

# Prediction function
def predict_category(input_resume):
    """
    Use the loaded models to predict the resume category.
    Assumes the TF-IDF vectorizer in 'models/tfidf.pkl' is already fitted.
    """
    if tfidf is None or svc_model is None or le is None:
        raise ValueError("Models are not loaded properly. Please check model files.")

    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    predicted_category_name = le.inverse_transform(predicted_category)
    return predicted_category_name[0]

# Main app
def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/resume.png", width=80)
        st.title("📋 About")
        st.markdown("""
        This AI-powered tool analyzes resumes and predicts the most suitable job category.
        
        **Supported Formats:**
        - PDF (.pdf) - Text-based and Scanned
        - Word (.docx)
        - Text (.txt)
        
        **File Size Limit:** 20 MB
        
        **How it works:**
        1. Upload your resume
        2. AI analyzes the content
        3. Get instant category prediction
        """)
        
        st.markdown("---")
        st.markdown("🔒 **Privacy:** Files are processed securely and not stored.")
    
    # Main content
    st.title("🎯 AI Resume Screening System")
    st.markdown("### Instantly categorize and score resumes with machine learning")

    # Add cache clear button in sidebar
    with st.sidebar:
        if st.button("🔄 Clear Cache & Reload Models"):
            st.cache_resource.clear()
            st.rerun()

    # Check if models are loaded
    if svc_model is None or tfidf is None or le is None:
        st.error("⚠️ Models not loaded. Please check if model files exist.")
        st.info("💡 Try clicking 'Clear Cache & Reload Models' in the sidebar, or use Streamlit's menu: ☰ → 'Clear cache'")
        return

    # Tabs: Single vs Batch
    tab_single, tab_batch = st.tabs(["🧾 Single Resume", "📦 Batch (ZIP)"])

    # ---------- SINGLE RESUME TAB ----------
    with tab_single:
        # Option to paste text directly
        input_method = st.radio(
            "Choose input method:",
            ["📄 Upload File", "📝 Paste Text"],
            horizontal=True,
            key="input_method"
        )
        
        resume_text = None
        
        if input_method == "📄 Upload File":
            st.markdown('<div class="upload-section">', unsafe_allow_html=True)
            st.markdown("### 📤 Upload Resume")
            uploaded_file = st.file_uploader(
                "Drag and drop or click to upload",
                type=["pdf", "docx", "txt"],
                help="Maximum file size: 20 MB",
                key="single_uploader",
            )
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("### 📝 Paste Resume Text")
            resume_text = st.text_area(
                "Paste your resume text here:",
                placeholder="Paste the complete resume text here...",
                height=300,
                key="pasted_text"
            )
            uploaded_file = None
        
        job_description = st.text_area(
            "📄 Job Description (optional, used for skills matching and scoring)",
            placeholder="Paste the JD here to see how well the resume matches...",
            height=120,
        )

        # Process either uploaded file or pasted text
        resume_text_to_process = None
        
        if input_method == "📄 Upload File" and uploaded_file is not None:
            # File size check
            is_valid_size, file_size = check_file_size(uploaded_file)

            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.info(f"📄 **File:** {uploaded_file.name}")
                st.info(f"📊 **Size:** {file_size:.2f} MB")

            if not is_valid_size:
                st.markdown('<div class="warning-box">', unsafe_allow_html=True)
                st.warning(
                    f"⚠️ File size ({file_size:.2f} MB) exceeds 20 MB limit. Please upload a smaller file."
                )
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                with st.spinner("🤖 Extracting text from file..."):
                    try:
                        # Extract text from uploaded file
                        resume_text_to_process = handle_file_upload(uploaded_file)
                    except Exception as e:
                        st.error(f"❌ Error extracting text: {str(e)}")
                        st.info("💡 Tip: Try using 'Paste Text' option instead, or ensure the file is not corrupted.")
                        resume_text_to_process = None
        elif input_method == "📝 Paste Text" and resume_text:
            resume_text_to_process = resume_text.strip()
        
        # Process resume text (from file or paste)
        if resume_text_to_process and resume_text_to_process.strip():
            with st.spinner("🤖 AI is analyzing the resume..."):
                try:
                    resume_text = resume_text_to_process
                    
                    if not resume_text or not resume_text.strip():
                        st.error("❌ No text provided.")
                    else:
                            # Show success message
                            st.success("✅ Text extracted successfully!")

                            # Show text preview option
                            with st.expander("📝 View Extracted Text"):
                                st.text_area(
                                    "Resume Content",
                                    resume_text[:2000] + "..."
                                    if len(resume_text) > 2000
                                    else resume_text,
                                    height=300,
                                )

                            # Predict category
                            category = predict_category(resume_text)

                            # Skills
                            skills_info = extract_skills(resume_text)
                            matched_count, total_skills, matched_list = match_skills_against_job(
                                skills_info["all"], job_description or ""
                            )

                            # Experience
                            exp_info = analyze_experience(resume_text)

                            # Scoring
                            scores = score_resume(
                                text=resume_text,
                                experience_level=str(exp_info.get("level") or "Unknown"),
                                total_resume_skills=total_skills,
                                matched_job_skills=matched_count,
                                has_job_description=bool(job_description),
                            )

                            # Display main result
                            st.markdown(
                                '<div class="result-box">', unsafe_allow_html=True
                            )
                            st.markdown("### 🎉 Analysis Complete!")
                            st.markdown(
                                f'<div class="category-badge">📌 {category}</div>',
                                unsafe_allow_html=True,
                            )
                            st.markdown(
                                f"**Overall Score:** {scores['overall_score']} / 100"
                            )
                            st.markdown("</div>", unsafe_allow_html=True)

                            # Stats
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Words Analyzed", len(resume_text.split()))
                            with col2:
                                st.metric("Characters", len(resume_text))
                            with col3:
                                st.metric("Matched Skills", matched_count)

                            # Detailed panels
                            col_left, col_right = st.columns(2)

                            with col_left:
                                st.subheader("🧠 Skills Analysis")
                                
                                # Technical Skills with badges
                                st.markdown("**Technical Skills:**")
                                if skills_info["technical"]:
                                    # Display as badges
                                    tech_badges = " ".join([f"`{skill}`" for skill in skills_info["technical"]])
                                    st.markdown(tech_badges)
                                    
                                    # Skill match percentage if JD provided
                                    if job_description:
                                        tech_matched = [s for s in skills_info["technical"] if s.lower() in job_description.lower()]
                                        if tech_matched:
                                            match_pct = (len(tech_matched) / len(skills_info["technical"])) * 100
                                            st.metric("Technical Skills Match", f"{match_pct:.1f}%", f"{len(tech_matched)}/{len(skills_info['technical'])}")
                                else:
                                    st.write("No technical skills detected.")

                                st.markdown("---")
                                
                                # Soft Skills with badges
                                st.markdown("**Soft Skills:**")
                                if skills_info["soft"]:
                                    soft_badges = " ".join([f"`{skill}`" for skill in skills_info["soft"]])
                                    st.markdown(soft_badges)
                                else:
                                    st.write("No soft skills detected.")

                                st.markdown("---")
                                
                                if job_description:
                                    st.markdown("**Skills Matched to Job Description:**")
                                    if matched_list:
                                        matched_badges = " ".join([f"`{skill}`" for skill in matched_list])
                                        st.markdown(matched_badges)
                                        st.success(f"✅ {matched_count} out of {total_skills} skills match the job description!")
                                    else:
                                        st.write("No direct matches found in JD.")

                            with col_right:
                                st.subheader("📈 Experience & Scoring")
                                
                                # Experience Level with badge
                                exp_level = exp_info.get('level', 'Unknown')
                                level_colors = {
                                    'Fresher': '🔵',
                                    'Junior': '🟢',
                                    'Mid-Level': '🟡',
                                    'Senior': '🟠',
                                    'Expert': '🔴',
                                    'Unknown': '⚪'
                                }
                                st.markdown(
                                    f"**Experience Level:** {level_colors.get(exp_level, '⚪')} **{exp_level}**"
                                )
                                
                                years = exp_info.get("years")
                                if years is not None:
                                    st.metric(
                                        "Estimated Years of Experience", 
                                        f"{years:.1f} years"
                                    )
                                
                                companies = exp_info.get("companies") or []
                                if companies:
                                    st.markdown("**Previous Companies:**")
                                    for company in companies[:5]:  # Show max 5
                                        st.write(f"🏢 {company}")
                                    if len(companies) > 5:
                                        st.caption(f"... and {len(companies) - 5} more")
                                
                                current_title = exp_info.get("current_title")
                                if current_title:
                                    st.markdown("**Current / Recent Title:**")
                                    st.info(f"💼 {current_title}")

                                st.markdown("---")
                                st.markdown("**Score Breakdown:**")
                                
                                # Visual score bars
                                score_col1, score_col2 = st.columns(2)
                                
                                with score_col1:
                                    st.metric(
                                        "Skills Score", f"{scores['skills_score']:.1f} / 100"
                                    )
                                    st.progress(scores['skills_score'] / 100)
                                    
                                    st.metric(
                                        "Experience Score",
                                        f"{scores['experience_score']:.1f} / 100",
                                    )
                                    st.progress(scores['experience_score'] / 100)
                                
                                with score_col2:
                                    st.metric(
                                        "Education Score",
                                        f"{scores['education_score']:.1f} / 100",
                                    )
                                    st.progress(scores['education_score'] / 100)
                                    
                                    st.metric(
                                        "Format Score",
                                        f"{scores['format_score']:.1f} / 100",
                                    )
                                    st.progress(scores['format_score'] / 100)
                                
                                # Overall score visualization
                                st.markdown("---")
                                overall_score = scores['overall_score']
                                score_color = (
                                    '🟢' if overall_score >= 80 else
                                    '🟡' if overall_score >= 60 else
                                    '🟠' if overall_score >= 40 else
                                    '🔴'
                                )
                                st.markdown(f"### {score_color} Overall Score: **{overall_score:.1f} / 100**")
                                st.progress(overall_score / 100)

                except Exception as e:
                    st.error(f"❌ Error processing resume: {str(e)}")
                    st.info(
                        "💡 Tip: Ensure the file is not corrupted and contains readable text, or try using 'Paste Text' option."
                    )
                    import traceback

                    with st.expander("🔍 View Error Details"):
                        st.code(traceback.format_exc())

    # ---------- BATCH TAB ----------
    with tab_batch:
        st.markdown("### 📦 Upload ZIP of Resumes")
        zip_file = st.file_uploader(
            "Upload a .zip file containing multiple resumes (pdf, docx, txt)",
            type=["zip"],
            key="batch_uploader",
        )

        batch_job_description = st.text_area(
            "📄 Job Description for Batch (optional)",
            placeholder="Paste the JD here to compare all resumes against the same role...",
            height=120,
            key="batch_jd",
        )

        if zip_file is not None:
            import zipfile
            import io

            try:
                with zipfile.ZipFile(zip_file) as zf:
                    members = iterate_zip_members(zf)

                    if not members:
                        st.warning(
                            "No supported files found in ZIP. Please include pdf, docx, or txt files."
                        )
                        return

                    st.info(f"Found {len(members)} resume files in the ZIP.")
                    
                    if len(members) > 100:
                        st.warning(f"⚠️ Large batch detected ({len(members)} files). Processing may take several minutes.")

                    results = []
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    stats_container = st.empty()

                    for i, info in enumerate(members, start=1):
                        # Update progress
                        progress_percent = i / len(members)
                        progress_bar.progress(progress_percent)
                        
                        # Update status
                        status_text.markdown(
                            f"**Processing:** {i}/{len(members)} ({progress_percent*100:.1f}%) - `{info.filename}`"
                        )
                        
                        # Show stats
                        if i > 1:
                            avg_score = sum(r.get('score_overall', 0) for r in results) / len(results)
                            stats_container.markdown(
                                f"📊 **Progress Stats:** Processed: {len(results)} | "
                                f"Avg Score: {avg_score:.1f} | "
                                f"Categories Found: {len(set(r.get('category', '') for r in results))}"
                            )
                        data = zf.read(info.filename)

                        try:
                            # Reuse single-resume processing logic
                            text = extract_text_from_bytes(info.filename, data)
                            skills_info = extract_skills(text)
                            matched_count, total_skills, matched_list = match_skills_against_job(
                                skills_info["all"], batch_job_description or ""
                            )
                            exp_info = analyze_experience(text)
                            scores = score_resume(
                                text=text,
                                experience_level=str(
                                    exp_info.get("level") or "Unknown"
                                ),
                                total_resume_skills=total_skills,
                                matched_job_skills=matched_count,
                                has_job_description=bool(batch_job_description),
                            )

                            category = predict_category(text)

                            results.append(
                                {
                                    "file_name": info.filename,
                                    "category": category,
                                    "skills_technical": ", ".join(
                                        skills_info["technical"]
                                    ),
                                    "skills_soft": ", ".join(
                                        skills_info["soft"]
                                    ),
                                    "skills_all": ", ".join(
                                        skills_info["all"]
                                    ),
                                    "skills_matched_to_job": ", ".join(
                                        matched_list
                                    ),
                                    "skills_matched_count": matched_count,
                                    "skills_total": total_skills,
                                    "experience_years": exp_info.get("years"),
                                    "experience_level": exp_info.get("level"),
                                    "companies": ", ".join(
                                        exp_info.get("companies", [])
                                    ),
                                    "current_title": exp_info.get(
                                        "current_title"
                                    ),
                                    "score_overall": scores["overall_score"],
                                    "score_skills": scores["skills_score"],
                                    "score_experience": scores[
                                        "experience_score"
                                    ],
                                    "score_education": scores[
                                        "education_score"
                                    ],
                                    "score_format": scores["format_score"],
                                }
                            )
                        except Exception as e:
                            st.warning(
                                f"Skipping file {info.filename} due to error: {e}"
                            )

                    # Final status update
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Batch processing complete! Processed {len(results)} out of {len(members)} resumes.")
                    stats_container.empty()

                    if results:
                        df = build_results_dataframe(results)
                        
                        # Store in session state for filtering
                        st.session_state['batch_results_df'] = df
                        st.session_state['batch_job_description'] = batch_job_description

                        # ========== SKILL FREQUENCY CHARTS ==========
                        st.subheader("📊 Skill Frequency Analysis")
                        
                        # Collect all skills
                        all_technical_skills = []
                        all_soft_skills = []
                        for skills_str in df['skills_technical'].dropna():
                            if skills_str:
                                all_technical_skills.extend([s.strip() for s in skills_str.split(',') if s.strip()])
                        for skills_str in df['skills_soft'].dropna():
                            if skills_str:
                                all_soft_skills.extend([s.strip() for s in skills_str.split(',') if s.strip()])
                        
                        col_chart1, col_chart2 = st.columns(2)
                        
                        with col_chart1:
                            if all_technical_skills:
                                tech_counter = Counter(all_technical_skills)
                                top_tech = dict(tech_counter.most_common(15))
                                tech_df = pd.DataFrame(list(top_tech.items()), columns=['Skill', 'Frequency'])
                                fig_tech = px.bar(
                                    tech_df, 
                                    x='Frequency', 
                                    y='Skill', 
                                    orientation='h',
                                    title='Top 15 Technical Skills',
                                    labels={'Frequency': 'Count', 'Skill': 'Technical Skill'},
                                    color='Frequency',
                                    color_continuous_scale='Blues'
                                )
                                fig_tech.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig_tech, use_container_width=True)
                            else:
                                st.info("No technical skills found in batch.")
                        
                        with col_chart2:
                            if all_soft_skills:
                                soft_counter = Counter(all_soft_skills)
                                top_soft = dict(soft_counter.most_common(15))
                                soft_df = pd.DataFrame(list(top_soft.items()), columns=['Skill', 'Frequency'])
                                fig_soft = px.bar(
                                    soft_df, 
                                    x='Frequency', 
                                    y='Skill', 
                                    orientation='h',
                                    title='Top 15 Soft Skills',
                                    labels={'Frequency': 'Count', 'Skill': 'Soft Skill'},
                                    color='Frequency',
                                    color_continuous_scale='Greens'
                                )
                                fig_soft.update_layout(height=400, showlegend=False)
                                st.plotly_chart(fig_soft, use_container_width=True)
                            else:
                                st.info("No soft skills found in batch.")
                        
                        # Category distribution chart
                        if 'category' in df.columns:
                            category_counts = df['category'].value_counts()
                            fig_cat = px.pie(
                                values=category_counts.values,
                                names=category_counts.index,
                                title='Resume Distribution by Category',
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig_cat.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_cat, use_container_width=True)
                        
                        # Experience level distribution
                        if 'experience_level' in df.columns:
                            exp_counts = df['experience_level'].value_counts()
                            fig_exp = px.bar(
                                x=exp_counts.index,
                                y=exp_counts.values,
                                title='Experience Level Distribution',
                                labels={'x': 'Experience Level', 'y': 'Count'},
                                color=exp_counts.values,
                                color_continuous_scale='Viridis'
                            )
                            fig_exp.update_layout(showlegend=False)
                            st.plotly_chart(fig_exp, use_container_width=True)

                        # ========== TOP 10 CANDIDATES WITH DETAILED INFO ==========
                        st.subheader("🏆 Top 10 Candidates (Ranked by Overall Score)")
                        df_ranked = df.copy()
                        df_ranked['rank'] = df_ranked['score_overall'].rank(ascending=False, method='min').astype(int)
                        top10 = df_ranked.nlargest(10, 'score_overall')
                        
                        # Display top 10 with more details
                        display_cols = ['rank', 'file_name', 'category', 'score_overall', 
                                       'experience_level', 'skills_total', 'skills_matched_count']
                        try:
                            # Try to use styled dataframe with gradient
                            styled_df = top10[display_cols].style.background_gradient(subset=['score_overall'], cmap='YlOrRd')
                            st.dataframe(styled_df, use_container_width=True)
                        except ImportError:
                            # Fallback to regular dataframe if matplotlib not available
                            st.dataframe(top10[display_cols], use_container_width=True)
                        
                        # Expandable detailed view for top 10
                        with st.expander("📋 View Detailed Top 10 Information"):
                            for idx, row in top10.iterrows():
                                with st.container():
                                    st.markdown(f"### 🥇 Rank #{int(row['rank'])}: {row['file_name']}")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Overall Score", f"{row['score_overall']:.1f}/100")
                                        st.metric("Skills Score", f"{row['score_skills']:.1f}/100")
                                    with col2:
                                        st.metric("Experience Score", f"{row['score_experience']:.1f}/100")
                                        st.metric("Education Score", f"{row['score_education']:.1f}/100")
                                    with col3:
                                        st.metric("Format Score", f"{row['score_format']:.1f}/100")
                                        st.metric("Category", row['category'])
                                    
                                    if pd.notna(row['experience_level']):
                                        st.info(f"**Experience:** {row['experience_level']}")
                                    if pd.notna(row['experience_years']):
                                        st.info(f"**Years:** {row['experience_years']:.1f}")
                                    if pd.notna(row['current_title']) and row['current_title']:
                                        st.info(f"**Current Title:** {row['current_title']}")
                                    if pd.notna(row['companies']) and row['companies']:
                                        st.info(f"**Companies:** {row['companies']}")
                                    if pd.notna(row['skills_technical']) and row['skills_technical']:
                                        st.write(f"**Technical Skills:** {row['skills_technical']}")
                                    if pd.notna(row['skills_soft']) and row['skills_soft']:
                                        st.write(f"**Soft Skills:** {row['skills_soft']}")
                                    st.markdown("---")

                        # ========== ADVANCED SEARCH & FILTERING ==========
                        st.subheader("🔍 Advanced Search & Filtering")
                        
                        filter_col1, filter_col2, filter_col3 = st.columns(3)
                        
                        with filter_col1:
                            # Filter by category
                            categories = ['All'] + sorted(df['category'].unique().tolist())
                            selected_category = st.selectbox(
                                "Filter by Category",
                                categories,
                                key="filter_category"
                            )
                        
                        with filter_col2:
                            # Filter by experience level
                            exp_levels = ['All'] + sorted([x for x in df['experience_level'].dropna().unique() if x])
                            selected_exp_level = st.selectbox(
                                "Filter by Experience Level",
                                exp_levels,
                                key="filter_exp_level"
                            )
                        
                        with filter_col3:
                            # Filter by minimum score
                            min_score = st.slider(
                                "Minimum Overall Score",
                                min_value=0.0,
                                max_value=100.0,
                                value=0.0,
                                step=5.0,
                                key="filter_min_score"
                            )
                        
                        # Search within results
                        search_query = st.text_input(
                            "🔎 Search in Results (searches file names, skills, companies, titles)",
                            placeholder="Enter keywords to search...",
                            key="search_query"
                        )
                        
                        # Filter by skills
                        st.markdown("**Filter by Skills:**")
                        skill_filter_col1, skill_filter_col2 = st.columns(2)
                        with skill_filter_col1:
                            filter_technical_skill = st.selectbox(
                                "Technical Skill",
                                ['All'] + sorted(set(all_technical_skills)),
                                key="filter_tech_skill"
                            )
                        with skill_filter_col2:
                            filter_soft_skill = st.selectbox(
                                "Soft Skill",
                                ['All'] + sorted(set(all_soft_skills)),
                                key="filter_soft_skill"
                            )
                        
                        # Apply filters
                        filtered_df = df.copy()
                        
                        if selected_category != 'All':
                            filtered_df = filtered_df[filtered_df['category'] == selected_category]
                        
                        if selected_exp_level != 'All':
                            filtered_df = filtered_df[filtered_df['experience_level'] == selected_exp_level]
                        
                        filtered_df = filtered_df[filtered_df['score_overall'] >= min_score]
                        
                        if search_query:
                            search_lower = search_query.lower()
                            mask = (
                                filtered_df['file_name'].str.lower().str.contains(search_lower, na=False) |
                                filtered_df['skills_all'].str.lower().str.contains(search_lower, na=False) |
                                filtered_df['companies'].str.lower().str.contains(search_lower, na=False) |
                                filtered_df['current_title'].str.lower().str.contains(search_lower, na=False)
                            )
                            filtered_df = filtered_df[mask]
                        
                        if filter_technical_skill != 'All':
                            filtered_df = filtered_df[
                                filtered_df['skills_technical'].str.contains(filter_technical_skill, case=False, na=False)
                            ]
                        
                        if filter_soft_skill != 'All':
                            filtered_df = filtered_df[
                                filtered_df['skills_soft'].str.contains(filter_soft_skill, case=False, na=False)
                            ]
                        
                        # Display filtered results
                        st.markdown(f"**📊 Filtered Results: {len(filtered_df)} out of {len(df)} resumes**")
                        
                        if len(filtered_df) > 0:
                            # Sort filtered results by score
                            filtered_df_sorted = filtered_df.sort_values('score_overall', ascending=False)
                            try:
                                # Try to use styled dataframe with gradient
                                styled_filtered = filtered_df_sorted.style.background_gradient(subset=['score_overall'], cmap='YlOrRd')
                                st.dataframe(styled_filtered, use_container_width=True, height=400)
                            except ImportError:
                                # Fallback to regular dataframe if matplotlib not available
                                st.dataframe(filtered_df_sorted, use_container_width=True, height=400)
                            
                            # Export filtered results
                            st.markdown("### 💾 Export Filtered Results")
                            
                            col_exp1, col_exp2 = st.columns(2)
                            
                            with col_exp1:
                                csv_filtered = filtered_df_sorted.to_csv(index=False).encode("utf-8")
                                st.download_button(
                                    "⬇️ Download Filtered Results as CSV",
                                    data=csv_filtered,
                                    file_name="filtered_results.csv",
                                    mime="text/csv",
                                )
                            
                            with col_exp2:
                                excel_buffer_filtered = io.BytesIO()
                                with pd.ExcelWriter(excel_buffer_filtered, engine="openpyxl") as writer:
                                    filtered_df_sorted.to_excel(writer, index=False, sheet_name="Filtered Results")
                                excel_buffer_filtered.seek(0)
                                st.download_button(
                                    "⬇️ Download Filtered Results as Excel",
                                    data=excel_buffer_filtered,
                                    file_name="filtered_results.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                )
                        else:
                            st.warning("No results match the selected filters. Please adjust your filters.")

                        # ========== EXPORT ALL RESULTS ==========
                        st.subheader("💾 Export All Results")
                        
                        # Enhanced Excel export with categories as separate sheets
                        excel_buffer = io.BytesIO()
                        with pd.ExcelWriter(excel_buffer, engine="openpyxl") as writer:
                            # Main results sheet
                            df.to_excel(writer, index=False, sheet_name="All Results")
                            
                            # Sheets by category
                            for category in df['category'].unique():
                                cat_df = df[df['category'] == category].sort_values('score_overall', ascending=False)
                                sheet_name = category[:31]  # Excel sheet name limit
                                cat_df.to_excel(writer, index=False, sheet_name=sheet_name)
                            
                            # Top 10 sheet
                            top10.to_excel(writer, index=False, sheet_name="Top 10")
                            
                            # Summary statistics sheet
                            summary_data = {
                                'Metric': [
                                    'Total Resumes',
                                    'Categories Found',
                                    'Average Score',
                                    'Highest Score',
                                    'Lowest Score',
                                    'Fresher',
                                    'Junior',
                                    'Mid-Level',
                                    'Senior',
                                    'Expert'
                                ],
                                'Value': [
                                    len(df),
                                    df['category'].nunique(),
                                    f"{df['score_overall'].mean():.2f}",
                                    f"{df['score_overall'].max():.2f}",
                                    f"{df['score_overall'].min():.2f}",
                                    len(df[df['experience_level'] == 'Fresher']),
                                    len(df[df['experience_level'] == 'Junior']),
                                    len(df[df['experience_level'] == 'Mid-Level']),
                                    len(df[df['experience_level'] == 'Senior']),
                                    len(df[df['experience_level'] == 'Expert']),
                                ]
                            }
                            summary_df = pd.DataFrame(summary_data)
                            summary_df.to_excel(writer, index=False, sheet_name="Summary")
                        
                        excel_buffer.seek(0)
                        
                        col_exp_all1, col_exp_all2 = st.columns(2)
                        
                        with col_exp_all1:
                            st.download_button(
                                "⬇️ Download All Results as CSV",
                                data=df.to_csv(index=False).encode("utf-8"),
                                file_name="batch_results.csv",
                                mime="text/csv",
                            )
                        
                        with col_exp_all2:
                            st.download_button(
                                "⬇️ Download All Results as Excel (Multi-Sheet)",
                                data=excel_buffer,
                                file_name="batch_results.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            )
                        
                        # Also save to outputs folder
                        out_dir = ensure_output_dir()
                        excel_path = os.path.join(out_dir, "batch_results.xlsx")
                        csv_path = os.path.join(out_dir, "batch_results.csv")
                        
                        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                            df.to_excel(writer, index=False, sheet_name="All Results")
                            for category in df['category'].unique():
                                cat_df = df[df['category'] == category].sort_values('score_overall', ascending=False)
                                sheet_name = category[:31]
                                cat_df.to_excel(writer, index=False, sheet_name=sheet_name)
                            top10.to_excel(writer, index=False, sheet_name="Top 10")
                            summary_df.to_excel(writer, index=False, sheet_name="Summary")
                        
                        df.to_csv(csv_path, index=False)
                        
                        st.success(f"✅ Results exported to Excel and CSV in the 'outputs' folder. ({len(df)} resumes processed)")

            except Exception as e:
                st.error(f"❌ Error processing ZIP file: {e}")
                import traceback

                with st.expander("🔍 View Error Details (Batch)"):
                    st.code(traceback.format_exc())

if __name__ == "__main__":
    main()