"""
Script to check if model files are properly saved and fitted.
Run this to diagnose model issues.
"""
import pickle
import os
from sklearn.utils.validation import check_is_fitted

def check_model_file(filepath, model_name):
    """Check if a model file exists and is properly fitted."""
    print(f"\n{'='*60}")
    print(f"Checking {model_name}...")
    print(f"File: {filepath}")
    
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return False
    
    try:
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        
        print(f"SUCCESS: File loaded successfully")
        print(f"   Type: {type(model)}")
        
        # Check if it's a TF-IDF vectorizer (has both transform and idf_ attribute)
        if hasattr(model, 'transform') and hasattr(model, 'idf_'):
            try:
                check_is_fitted(model, attributes=["idf_"], msg="idf vector is not fitted")
                print(f"SUCCESS: Model is properly fitted")
                if hasattr(model, 'idf_'):
                    print(f"   IDF vector shape: {model.idf_.shape if hasattr(model.idf_, 'shape') else 'N/A'}")
                return True
            except Exception as e:
                print(f"ERROR: Model is NOT fitted properly")
                print(f"   Error: {str(e)}")
                print(f"\nSOLUTION: You need to re-train and save the TF-IDF model.")
                print(f"   The model must be fitted before saving with pickle.dump()")
                return False
        else:
            # For other models (classifier, encoder), check if they have required attributes
            if hasattr(model, 'classes_') or hasattr(model, 'predict'):
                # It's a classifier or encoder, check if fitted
                try:
                    if hasattr(model, 'classes_'):
                        check_is_fitted(model, attributes=["classes_"])
                    elif hasattr(model, 'predict'):
                        # For OneVsRestClassifier, check if it has estimators
                        if hasattr(model, 'estimators_'):
                            check_is_fitted(model, attributes=["estimators_"])
                        else:
                            check_is_fitted(model)
                    print(f"SUCCESS: Model is properly fitted")
                except Exception as e:
                    print(f"WARNING: Could not verify fitting: {str(e)}")
            else:
                print(f"SUCCESS: Model loaded (fitting check not applicable)")
            return True
            
    except Exception as e:
        print(f"ERROR loading file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    import sys
    # Fix encoding for Windows console
    if sys.platform == 'win32':
        import io
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    
    print("="*60)
    print("MODEL DIAGNOSTIC TOOL")
    print("="*60)
    
    # Check models in models/ folder
    models_dir = 'models'
    if os.path.exists(models_dir):
        print(f"\nChecking models in '{models_dir}/' folder:")
        clf_path = os.path.join(models_dir, 'clf.pkl')
        tfidf_path = os.path.join(models_dir, 'tfidf.pkl')
        encoder_path = os.path.join(models_dir, 'encoder.pkl')
    else:
        print(f"\nChecking models in current directory:")
        clf_path = 'clf.pkl'
        tfidf_path = 'tfidf.pkl'
        encoder_path = 'encoder.pkl'
    
    results = []
    results.append(check_model_file(clf_path, "Classifier (clf.pkl)"))
    results.append(check_model_file(tfidf_path, "TF-IDF Vectorizer (tfidf.pkl)"))
    results.append(check_model_file(encoder_path, "Label Encoder (encoder.pkl)"))
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    
    if all(results):
        print("SUCCESS: All models are properly loaded and fitted!")
    else:
        print("ERROR: Some models have issues. Please fix them before running the app.")
        print("\nTIP: If TF-IDF is not fitted, you need to:")
        print("   1. Re-run your training script")
        print("   2. Make sure tfidf.fit() is called before pickle.dump()")
        print("   3. Example:")
        print("      from sklearn.feature_extraction.text import TfidfVectorizer")
        print("      tfidf = TfidfVectorizer()")
        print("      tfidf.fit(X_train)  # <- Must fit before saving!")
        print("      pickle.dump(tfidf, open('models/tfidf.pkl', 'wb'))")

if __name__ == "__main__":
    main()

