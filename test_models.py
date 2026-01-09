"""
Quick test to verify models work correctly.
Run this to test if your models are working before using the Streamlit app.
"""
import pickle
import re
import sys
import io

# Fix encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def cleanResume(txt):
    cleanText = re.sub('http\S+\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\S+\s', ' ', cleanText)
    cleanText = re.sub('@\S+', '  ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\s+', ' ', cleanText)
    return cleanText

def test_models():
    print("Loading models...")
    try:
        svc_model = pickle.load(open('models/clf.pkl', 'rb'))
        tfidf = pickle.load(open('models/tfidf.pkl', 'rb'))
        le = pickle.load(open('models/encoder.pkl', 'rb'))
        print("SUCCESS: Models loaded successfully")
    except Exception as e:
        print(f"ERROR: Error loading models: {e}")
        return False
    
    # Test with sample resume text
    sample_resume = """
    John Doe
    Software Engineer
    
    Experience:
    - 5 years of Python development
    - Machine learning and data science
    - Web development with Django
    
    Skills: Python, Java, SQL, Machine Learning
    """
    
    print("\nTesting prediction...")
    try:
        cleaned_text = cleanResume(sample_resume)
        print(f"SUCCESS: Text cleaned: {len(cleaned_text)} characters")
        
        vectorized_text = tfidf.transform([cleaned_text]).toarray()
        print(f"SUCCESS: Text vectorized: shape {vectorized_text.shape}")
        
        predicted_category = svc_model.predict(vectorized_text)
        print(f"SUCCESS: Prediction made: {predicted_category}")
        
        predicted_category_name = le.inverse_transform(predicted_category)
        print(f"SUCCESS: Category decoded: {predicted_category_name[0]}")
        
        print(f"\n{'='*60}")
        print("SUCCESS! Models are working correctly.")
        print(f"Predicted category: {predicted_category_name[0]}")
        print(f"{'='*60}")
        return True
        
    except Exception as e:
        print(f"\nERROR during prediction: {e}")
        import traceback
        traceback.print_exc()
        print(f"\n{'='*60}")
        print("TROUBLESHOOTING:")
        print("1. Clear Streamlit cache: Menu → Clear cache → Clear cache")
        print("2. Check scikit-learn version: pip show scikit-learn")
        print("3. Try reinstalling: pip install --upgrade scikit-learn")
        print(f"{'='*60}")
        return False

if __name__ == "__main__":
    test_models()

