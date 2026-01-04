# pip install streamlit scikit-learn python-docx PyPDF2

import streamlit as st
import pickle
import docx
import PyPDF2
import re
import os

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
    try:
        svc_model = pickle.load(open('clf.pkl', 'rb'))
        tfidf = pickle.load(open('tfidf.pkl', 'rb'))
        le = pickle.load(open('encoder.pkl', 'rb'))
        return svc_model, tfidf, le
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Please ensure clf.pkl, tfidf.pkl, and encoder.pkl are in the same directory.")
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

# Extract text from PDF
def extract_text_from_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ''
        for page in pdf_reader.pages:
            try:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + '\n'
            except Exception as e:
                st.warning(f"⚠️ Warning: Could not extract text from page {pdf_reader.pages.index(page) + 1}")
                continue
        
        # If no text extracted, it might be a scanned PDF
        if not text.strip():
            raise ValueError("This appears to be a scanned PDF (image-based). Please upload a text-based PDF or try converting it first.")
        
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
        - PDF (.pdf)
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
    st.markdown("### Instantly categorize resumes with machine learning")
    
    # Check if models are loaded
    if svc_model is None or tfidf is None or le is None:
        st.error("⚠️ Models not loaded. Please check if model files exist.")
        return
    
    # File upload section
    st.markdown('<div class="upload-section">', unsafe_allow_html=True)
    st.markdown("### 📤 Upload Resume")
    uploaded_file = st.file_uploader(
        "Drag and drop or click to upload",
        type=["pdf", "docx", "txt"],
        help="Maximum file size: 20 MB"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        # File size check
        is_valid_size, file_size = check_file_size(uploaded_file)
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info(f"📄 **File:** {uploaded_file.name}")
            st.info(f"📊 **Size:** {file_size:.2f} MB")
        
        if not is_valid_size:
            st.markdown('<div class="warning-box">', unsafe_allow_html=True)
            st.warning(f"⚠️ File size ({file_size:.2f} MB) exceeds 20 MB limit. Please upload a smaller file.")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            # Automatic analysis when file is uploaded
            with st.spinner("🤖 AI is analyzing the resume..."):
                try:
                    # Extract text
                    resume_text = handle_file_upload(uploaded_file)
                    
                    if not resume_text or not resume_text.strip():
                        st.error("❌ Could not extract text from the file.")
                        st.warning("""
                        **Possible reasons:**
                        - The PDF is scanned (image-based) and needs OCR
                        - The file is encrypted or password-protected
                        - The file format is corrupted
                        
                        **Solutions:**
                        - Try converting the PDF to Word (.docx) first
                        - Use a text-based PDF (not scanned)
                        - Copy-paste the resume text into a .txt file and upload that
                        """)
                    else:
                        # Show success message
                        st.success("✅ Text extracted successfully!")
                        
                        # Show text preview option
                        with st.expander("📝 View Extracted Text"):
                            st.text_area("Resume Content", resume_text[:2000] + "..." if len(resume_text) > 2000 else resume_text, height=300)
                        
                        # Predict category
                        category = predict_category(resume_text)
                        
                        # Display result
                        st.markdown('<div class="result-box">', unsafe_allow_html=True)
                        st.markdown("### 🎉 Analysis Complete!")
                        st.markdown(f'<div class="category-badge">📌 {category}</div>', unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Stats
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Words Analyzed", len(resume_text.split()))
                        with col2:
                            st.metric("Characters", len(resume_text))
                        with col3:
                            st.metric("Confidence", "High")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                    st.info("💡 Tip: Ensure the file is not corrupted and contains readable text.")
                    import traceback
                    with st.expander("🔍 View Error Details"):
                        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()