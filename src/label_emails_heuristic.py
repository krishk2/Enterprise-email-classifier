import pandas as pd
import re
from tqdm import tqdm

def label_emails_heuristic():
    # Load the dataset
    input_path = 'cleaned_data_sets/cleaned_email_priority_dataset.csv'
    output_path = 'cleaned_data_sets/labeled_email_dataset.csv'
    
    print(f"Loading dataset from {input_path}...")
    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: File not found at {input_path}")
        return

    # Define keywords for each category
    
    spam_keywords = [
        r'\bwinner\b', r'\blottery\b', r'\bprize\b', r'\bmillion\b', r'\btransfer funds\b', 
        r'\binheritance\b', r'\bclick here\b', r'\bverify your account\b', r'\bcongratulations\b'
    ]
    
    complaint_keywords = [
        r'\bissue\b', r'\bproblem\b', r'\bfail\b', r'\berror\b', r'\bdisappointed\b', 
        r'\bslow\b', r'\bdown\b', r'\bcrash\b', r'\bbug\b', r'\bstuck\b', r'\bbroken\b', 
        r'\bunable\b', r'\bcannot\b', r'\bnot working\b', r'\bstopped\b', r'\bbreach\b',
        r'\bunauthorized\b', r'\bconcern\b', r'\bdiscrepanc\b', r'\boutage\b', r'\bvulnerabilit\b',
        r'\bdisruption\b', r'\binterruption\b', r'\bglitch\b', r'\bmalfunction\b', r'\bcritical\b',
        r'\bincorrect\b', r'\bindisposed\b', r'\bdifficult\b', r'\bchallenge\b'
    ]
    
    feedback_keywords = [
        r'\bsuggest\b', r'\benhancement\b', r'\bfeature\b', r'\bfeedback\b', r'\bopinion\b', 
        r'\bproposal\b', r'\breview\b', r'\bidea\b', r'\bthought\b', r'\brating\b', r'\bcomment\b'
    ]
    
    request_keywords = [
        r'\brequest\b', r'\brequesting\b', r'\bneed\b', r'\bcan you\b', r'\bplease\b', r'\bassist\b', 
        r'\bhelp\b', r'\bsupport\b', r'\bguidance\b', r'\binquire\b', r'\bquestion\b', 
        r'\bhow to\b', r'\bwant to\b', r'\binterested\b', r'\bseeking\b', r'\badvice\b', 
        r'\brecommend\b', r'\brequire\b', r'\brequirement\b', r'\bquery\b', r'\bsearching for\b',
        r'\blooking for\b', r'\bushould\b'
    ]

    def get_label(text):
        if not isinstance(text, str):
            return "Request" # Default for empty/non-string
        
        text_lower = text.lower()
        
        # Check Spam
        for keyword in spam_keywords:
            if re.search(keyword, text_lower):
                return "Spam"
        
        # Check Complaint
        for keyword in complaint_keywords:
            if re.search(keyword, text_lower):
                return "Complaint"
        
        # Check Request (Prioritized over Feedback now to catch "requesting feature" properly)
        for keyword in request_keywords:
            if re.search(keyword, text_lower):
                return "Request"
                
        # Check Feedback
        for keyword in feedback_keywords:
            if re.search(keyword, text_lower):
                return "Feedback"
                
        return "Request" # Default fall-through

    print("Classifying emails using heuristics...")
    tqdm.pandas()
    df['category'] = df['body'].progress_apply(get_label)

    # Save labeled dataset
    print(f"Saving labeled dataset to {output_path}...")
    df.to_csv(output_path, index=False)
    print("Done!")
    print(df[['body', 'category']].head(10))
    print("\nLabel Distribution:")
    print(df['category'].value_counts())

if __name__ == "__main__":
    label_emails_heuristic()
