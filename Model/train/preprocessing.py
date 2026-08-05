import re
import string

# Ingredient analysis must work offline.  The earlier version downloaded NLTK data
# at import time, which breaks production servers and makes OCR/API startup slow.
STOP_WORDS = {"and", "or", "of", "the", "with", "contains", "may", "be", "from"}


# ======================================
# REMOVE PUNCTUATION
# ======================================

def remove_punctuation(text):
    return text.translate(str.maketrans('', '', string.punctuation))


# ======================================
# REMOVE NUMBERS
# ======================================

def remove_numbers(text):
    return re.sub(r'\d+', '', text)


# ======================================
# REMOVE EXTRA SPACES
# ======================================

def remove_extra_spaces(text):
    return re.sub(r'\s+', ' ', text).strip()


# ======================================
# REMOVE STOPWORDS
# ======================================

def remove_stopwords(tokens):
    return [word for word in tokens if word not in STOP_WORDS]


# ======================================
# STEMMING
# ======================================

def stemming(tokens):
    # Stemming makes important aliases harder to match (e.g. "whey"), so retain
    # the original ingredient tokens for inference and model training.
    return tokens


# ======================================
# MAIN CLEANING FUNCTION
# ======================================

def clean_text(text):

    # FIX 2: handle NaN / non-string values safely
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove punctuation
    text = remove_punctuation(text)

    # Remove numbers
    text = remove_numbers(text)

    # Remove extra spaces
    text = remove_extra_spaces(text)

    # Tokenization
    tokens = text.split()

    # Remove stopwords
    tokens = remove_stopwords(tokens)

    # Stemming
    tokens = stemming(tokens)

    # Join back
    text = " ".join(tokens)

    return text


# ======================================
# BUILD ingredients_text COLUMN
# ======================================

# FIX 3: create the missing 'ingredients_text' column before calling clean_text
def build_ingredients_text(df):
    cols = ['Main Ingredient', 'Sweetener', 'Fat/Oil', 'Seasoning', 'Allergens']
    # Only use columns that actually exist in the dataframe
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("CSV has no supported ingredient columns.")

    temp = df[cols].copy()

    # FIX 4: dataset stores missing values as string "None" — replace with empty string
    temp = temp.replace('None', '')
    temp = temp.fillna('')

    # FIX 5: Allergens column has comma-separated values like "Almonds, Wheat, Dairy"
    # Replace commas with spaces so they tokenize as separate words
    if 'Allergens' in temp.columns:
        temp['Allergens'] = temp['Allergens'].str.replace(',', ' ', regex=False)

    df['ingredients_text'] = temp.astype(str).agg(' '.join, axis=1).str.strip()
    return df
