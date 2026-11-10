from pathlib import Path
import sys

from langdetect import detect_langs
import pandas as pd
import re
import emoji
import contractions
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import ast
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import os
import joblib

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_DIR = PROJECT_ROOT / "datasets"
raw_data_path_en = PROJECT_ROOT / "data" / "raw_data_en.csv"
raw_data_path_ar = PROJECT_ROOT / "data" / "raw_data_ar.csv"
cleaned_data_path_en = PROJECT_ROOT / "data" / "cleaned_data_en"
cleaned_data_path_ar = PROJECT_ROOT / "data" / "cleaned_data_ar"
BASE_MODEL_DIR = PROJECT_ROOT / "saved_models"


def require_nltk_data():
    """Check local resources without downloading anything during startup."""
    missing = []
    for resource, package in (
        ("tokenizers/punkt_tab/english/", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
    ):
        try:
            nltk.data.find(resource)
        except LookupError:
            missing.append(package)
    if missing:
        raise RuntimeError(
            "Missing NLTK resources: " + ", ".join(missing) + ".\n"
            "Install them once with this environment's Python:\n"
            f'  "{sys.executable}" -m nltk.downloader punkt_tab stopwords\n'
            "In PowerShell, put & before the quoted executable path."
        )


require_nltk_data()


RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


VOCAB_SIZE = 30000   # limit vocab size (top 30k words)
OOV_TOKEN = "<OOV>"  # token for unknown words

TEXT_COL = "clean_text"
LABEL_COL = "label"

slang_dict = {
    "u": "you", 
    "ur": "your", 
    "r": "are", 
    "ya": "you",
    "tho": "though", 
    "thx": "thanks", 
    "tbh": "to be honest",
    "idk": "i do not know", 
    "idc": "i do not care", 
    "imo": "in my opinion",
    "imho": "in my humble opinion", 
    "lol": "laughing out loud",
    "lmao": "laughing my ass off", 
    "omg": "oh my god",
    "wtf": "what the fuck", 
    "wth": "what the hell",
    "brb": "be right back", 
    "btw": "by the way", 
    "irl": "in real life",
    "gg": "good game", 
    "np": "no problem", 
    "pls": "please",
    "plz": "please", 
    "ty": "thank you", 
    "smh": "shaking my head",
    "jk": "just kidding", 
    "afaik": "as far as i know",
    "nvm": "never mind", 
    "bc": "because", 
    "cuz": "because",
    "coz": "because", 
    "lmk": "let me know", 
    "omw": "on my way",
    "fr": "for real", 
    "fomo": "fear of missing out"
}

stop_words_en = set(stopwords.words('english'))
stop_words_ar = set(stopwords.words('arabic'))

important_words_en = {
    "please", "click", "free", "now", "check",
    "subscribe", "follow", "join", "download", "buy", "visit"
}

important_words_ar = {
    "مجانا",
    "مجاناً",
    "مجانية",
    "رابح",
    "اربح",
    "اربح الآن",
    "اربح المال",
    "اربح جوائز",
    "فرصة",
    "فرصة ذهبية",
    "لا تفوت الفرصة",
    "عرض",
    "عرض خاص",
    "عرض حصري",
    "عرض محدود",
    "لفترة محدودة",
    "خصم",
    "تخفيض",
    "تخفيضات",
    "هدية",
    "جوائز",
    "جائزة",
    "ربح مضمون",
    "مضمون ١٠٠٪",
    "اضغط",
    "اضغط هنا",
    "اضغط على الرابط",
    "اضغط الرابط",
    "اضغط للاشتراك",
    "اشترك",
    "اشترك الآن",
    "اشترك في القناة",
    "سجل",
    "سجل الآن",
    "تسجيل سريع",
    "تابعنا",
    "تابع الحساب",
    "تابع صفحتنا",
    "حمّل",
    "حمّل الآن",
    "تحميل مجاني",
    "رابط",
    "الرابط في الوصف",
    "الرابط في البايو",
    "افتح الرابط",
    "ادخل الرابط",
    "تواصل معنا",
    "راسلنا",
    "اطلب الآن",
    "اطلبه الآن",
    "اطلب المنتج",
    "اشتري الآن",
    "اشتري قبل النفاد",
    "سعر خاص",
    "أقل سعر",
    "أفضل سعر",
    "استثمر",
    "استثمار مضمون",
    "دخل إضافي",
    "ربح سريع",
    "تسويق",
    "تسويق بالعمولة",
    "متابعين",
    "زيادة متابعين",
    "مشاهدات",
    "زيادة مشاهدات",
    "لايكات",
    "زيادة لايكات",
}

stop_words_en = stop_words_en - important_words_en
stop_words_ar = stop_words_ar - important_words_ar

useless_tokens_en = {",", ".", ":", ";", ")", "(", "[", "]", "{", "}", "'", '"', "-", "_"}
useless_tokens_ar = {"،",",", ".", ":", ";", ")", "(", "[", "]", "{", "}", "'", '"', "-", "_"} #added the arabic comma

# =============================================================================
# reusable function DOES NOT NEED CHANGING PARAMETERS
# =============================================================================
def replace_noise(text):
   text = re.sub(r'(http\S+|www\S+)', 'URLTOKEN', text)
   text = re.sub(r'\S+\.(com|net|org|co|io|info|biz)(/\S*)?', 'URLTOKEN', text)
   text = re.sub(r'\bwatch\?v=[\w\-]+', 'URLTOKEN', text)
   text = re.sub(r'@\w+', 'MENTIONTOKEN', text)
   text = re.sub(r'#\w+', 'HASHTAGTOKEN', text)
   return text

def remove_random_periods(text):
    return re.sub(r"(?<!\.)\.(?!\.)", " ", text)

def replace_emojis(text):
    # Replace all emoji characters with EMOJITOKEN
    return emoji.replace_emoji(text, replace=" EMOJITOKEN ")

def normalize_repeated_letters(text):
    # soo → so, heeey → heey
    return re.sub(r'(.)\1{2,}', r'\1\1', text)

def tokenize_text(text):
    return word_tokenize(text)

def tokens_to_string(t):
    # If it's already a list, just join
    if isinstance(t, list):
        tokens = t
    else:
        # If it's a string like "['word1', 'word2', ...]"
        try:
            tokens = ast.literal_eval(t)
        except (ValueError, SyntaxError):
            # Fallback: just cast to string
            return str(t)
    return " ".join(tokens)

def normalize_label(x):
    if str(x).lower() in ['spam', 'true', '1', 'bot', 'automated']:
        return 1
    return 0

# =============================================================================
# reusable function BUT NEEDS CHANGING PARAMETERS
# =============================================================================
def is_confident(text, language, min_prob=0.90): #replace language with 'en' or 'ar'
    try:
        langs = detect_langs(text) 
        top = langs[0]
        return top.lang == language and top.prob >= min_prob
    except:
        return False

def remove_useless_punctuation(text, language):
    if (language == 'en'):
        return re.sub(r"[,\(\)\[\]\{\}\"\':;]", " ", text)
    else:
        return re.sub(r"[,\(\)\[\]\{\}\"\':;،]", " ", text) #added the comma in arabic

def group_punctuation(text, language):
    if (language == 'en'):
        text = re.sub(r'!{2,}', ' MULTIEXCLAMTOKEN ', text)
        text = re.sub(r'\?{2,}', ' MULTIQUESTTOKEN ', text)
        text = re.sub(r'\.{3,}', ' MULTIPERIOTOKEN ', text)
    else:
        text = re.sub(r'!{2,}', ' MULTIEXCLAMTOKEN ', text)
        #handle both "???" and "???"
        text = re.sub(r'(?:\?{2,}|؟{2,})', ' MULTIQUESTTOKEN ', text)
        text = re.sub(r'\.{3,}', ' MULTIPERIOTOKEN ', text)
    return text

def remove_stopwords(tokens, language):
    if (language == 'en'):
        return [w for w in tokens if w.lower() not in stop_words_en and w not in useless_tokens_en]
    else:
        return [w for w in tokens if w.lower() not in stop_words_ar and w not in useless_tokens_ar]

# =============================================================================
# Different functions for arabic and english
# =============================================================================

def replace_slang(text):
    words = text.split()
    return " ".join([slang_dict.get(w, w) for w in words])

def normalize_text(text):
    text = contractions.fix(text)
    text = replace_slang(text)
    text = normalize_repeated_letters(text)
    return text

# =============================================================================
# Data collection and combining functions
# =============================================================================

def collect_combine_en(raw_data_path_en):
    Path(raw_data_path_en).parent.mkdir(parents=True, exist_ok=True)
    youtube01 = pd.read_csv(DATASET_DIR / "english" / "Youtube01-Psy.csv")
    youtube02 = pd.read_csv(DATASET_DIR / "english" / "Youtube02-KatyPerry.csv")
    youtube03 = pd.read_csv(DATASET_DIR / "english" / "Youtube03-LMFAO.csv")
    youtube04 = pd.read_csv(DATASET_DIR / "english" / "Youtube04-Eminem.csv")
    youtube05 = pd.read_csv(DATASET_DIR / "english" / "Youtube05-Shakira.csv")
    kaggle = pd.read_csv(DATASET_DIR / "english" / "YouTube Comments Dataset with Sentiment Toxicity and Spam Labels (45K Rows).csv")
    splits = {'train': 'data/train-00000-of-00001-daf190ce720b3dbb.parquet', 'test': 'data/test-00000-of-00001-fa9b3e8ade89a333.parquet'}
    hugging_face = pd.read_parquet("hf://datasets/Deysi/spam-detection-dataset/" + splits["train"])
    youtube_all = pd.concat([youtube01, youtube02, youtube03, youtube04, youtube05], ignore_index=True)
    youtube_all.rename(columns={'CONTENT': 'text', 'CLASS': 'label'}, inplace=True)
    kaggle.rename(columns={'comment_text': 'text', 'label_spam': 'label'}, inplace=True)
    #no need to rename the hugging face dataset because it already is (text - label)
    youtube_all = youtube_all[['text', 'label']]
    kaggle = kaggle[['text', 'label']]
    hugging_face = hugging_face[['text', 'label']]
    combined = pd.concat([youtube_all, kaggle, hugging_face], ignore_index=True)
    combined['label'] = combined['label'].apply(normalize_label)
    # Drop duplicates and missing rows
    combined.drop_duplicates(subset='text', inplace=True)
    combined.dropna(subset=['text'], inplace=True)
    combined.to_csv(raw_data_path_en, index=False, mode='w')

def collect_combine_ar(raw_data_path_ar):
    Path(raw_data_path_ar).parent.mkdir(parents=True, exist_ok=True)
    df_xlsx = pd.read_excel(DATASET_DIR / "arabic" / "Augmented_SpamHamTweets.xlsx")
    df_xlsx.to_csv(DATASET_DIR / "arabic" / "Augmented_SpamHamTweets.csv", index=False, encoding="utf-8")
    df_txt = pd.read_csv(DATASET_DIR / "arabic" / "Autotweet-v1.0-Labeled-Tweets-Text.txt",
                        sep="\t",  # change this if the separator is different
                        header=None)
    df_txt.to_csv(DATASET_DIR / "arabic" / "Autotweet-v1.0-Labeled-Tweets-Text.csv",
                index=False,
                encoding="utf-8")
    df_aug = pd.read_csv(DATASET_DIR / "arabic" / "Augmented_SpamHamTweets.csv", encoding="utf-8")
    df_auto = pd.read_csv(DATASET_DIR / "arabic" / "Autotweet-v1.0-Labeled-Tweets-Text.csv", encoding="utf-8")
    df_aug_formatted = df_aug[['Cleaned Text', 'Label']].copy()
    df_aug_formatted = df_aug_formatted.rename(columns={
        'Cleaned Text': 'text',
        'Label': 'label'
    })
    df_auto.columns = ['tweet_id', 'text', 'label']

    # Drop the first row (it's the header inside the file)
    df_auto = df_auto.iloc[1:].reset_index(drop=True)
    df_auto_formatted = df_auto[['text', 'label']]
    df_arabic_all = pd.concat([df_aug_formatted, df_auto_formatted], ignore_index=True)
    df_arabic_all['label'] = df_arabic_all['label'].apply(normalize_label)
    df_arabic_all.to_csv(raw_data_path_ar, index=False, mode='w')

# =============================================================================
# Preprocessing function
# =============================================================================

def preprocessing(language, raw_data_path_en, raw_data_path_ar, cleaned_data_path_en, cleaned_data_path_ar):
    if language == 'en':
        combined = pd.read_csv(raw_data_path_en)
    else:
        combined = pd.read_csv(raw_data_path_ar)

    combined['text'] = combined['text'].fillna('').astype(str)

    # keep only confident language
    combined = combined[combined["text"].apply(lambda t: is_confident(t, language))].reset_index(drop=True)

    # basic normalization
    combined['text'] = combined['text'].apply(lambda x: x.lower().strip().replace('\n', ' '))
    combined["text"] = combined["text"].apply(replace_noise)
    combined["text"] = combined["text"].apply(replace_emojis)
    combined["text"] = combined["text"].apply(lambda t: remove_useless_punctuation(t, language))
    combined["text"] = combined["text"].apply(remove_random_periods)

    # english: contractions + slang, arabic: just letter normalization
    if language == 'en':
        combined["text"] = combined["text"].apply(normalize_text)
    else:
        combined["text"] = combined["text"].apply(normalize_repeated_letters)

    combined["text"] = combined["text"].apply(lambda t: group_punctuation(t, language))
    combined["text"] = combined["text"].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

    # tokenize
    combined["tokens"] = combined["text"].apply(tokenize_text)

    # remove stopwords + useless tokens on the token list
    combined["tokens"] = combined["tokens"].apply(lambda toks: remove_stopwords(toks, language))

    if language == 'en':
        Path(cleaned_data_path_en).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cleaned_data_path_en, index=False)
    else:
        Path(cleaned_data_path_ar).parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(cleaned_data_path_ar, index=False)

    
def preprocess_single_comment(comment, language="en"):
    """
    Apply the SAME preprocessing you used for training,
    but on a single raw comment string.
    Returns a cleaned text string (what goes into TF-IDF / tokenizer).
    """
    text = str(comment)

    # basic normalization
    text = text.lower().strip().replace("\n", " ")
    text = replace_noise(text)
    text = replace_emojis(text)
    text = remove_useless_punctuation(text, language)
    text = remove_random_periods(text)

    # english: contractions + slang, arabic: just repeated-letter normalization
    if language == "en":
        text = normalize_text(text)
    else:
        text = normalize_repeated_letters(text)

    text = group_punctuation(text, language)
    text = re.sub(r"\s+", " ", text).strip()

    # tokenize + remove stopwords
    tokens = tokenize_text(text)
    tokens = remove_stopwords(tokens, language)

    # same representation as "clean_text" column
    clean_text = " ".join(tokens)
    return clean_text


# =============================================================================
# HELPER Modeling functions
# =============================================================================

def prepare_for_modeling_ML_NN(language, cleaned_data_path_en, cleaned_data_path_ar):
    if (language == 'en'):
        combined = pd.read_csv(cleaned_data_path_en)
    else:
        combined = pd.read_csv(cleaned_data_path_ar)
    combined["clean_text"] = combined["tokens"].apply(tokens_to_string)
    X = combined[TEXT_COL].values
    y = combined[LABEL_COL].values

    X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=RANDOM_STATE,)

    X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.5,
    stratify=y_temp,
    random_state=RANDOM_STATE,)

    tfidf = TfidfVectorizer(
    max_features=30000,      # limit vocabulary size
    ngram_range=(1, 2),      # unigrams + bigrams
    min_df=2)         # ignore words that appear in only 1 phrase (optional)

    tfidf.fit(X_train)

    X_train_tfidf = tfidf.transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    X_test_tfidf  = tfidf.transform(X_test)
    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_tfidf, X_val_tfidf, X_test_tfidf,
        tfidf
    )

def prepare_for_modelingNN(X_train, X_val, X_test):
    # Use the global VOCAB_SIZE + OOV_TOKEN defined at the top
    tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token=OOV_TOKEN)

    # Learn the word index from TRAIN ONLY
    tokenizer.fit_on_texts(X_train)

    print("✅ Tokenizer fitted on training data")
    print("Size of full word_index:", len(tokenizer.word_index))

    # Convert texts to sequences of integers
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq   = tokenizer.texts_to_sequences(X_val)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    # Sequence length / padding
    seq_lengths = [len(seq) for seq in X_train_seq]
    if seq_lengths:
        MAX_LEN = int(np.percentile(seq_lengths, 95))
    else:
        MAX_LEN = 50  # some fallback, shouldn't really happen

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
    X_val_pad   = pad_sequences(X_val_seq,   maxlen=MAX_LEN, padding='post', truncating='post')
    X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

    # 🔴 IMPORTANT PART: figure out the maximum index actually used
    max_index = 0
    for seq in X_train_seq:
        if seq:
            mi = max(seq)
            if mi > max_index:
                max_index = mi

    vocab_size_for_embedding = max_index + 1  # +1 because indices start at 1
    print("Using vocab_size for Embedding:", vocab_size_for_embedding)

    return tokenizer, X_train_pad, X_val_pad, X_test_pad, MAX_LEN, vocab_size_for_embedding

    

def prepare_for_modeling_ML_NN_multi(cleaned_data_path_en, cleaned_data_path_ar):
    """
    Prepare a SINGLE modeling dataset by combining the cleaned English
    and Arabic datasets together.

    Returns the same structure as prepare_for_modeling_ML_NN.
    """
    combined_en = pd.read_csv(cleaned_data_path_en)
    combined_ar = pd.read_csv(cleaned_data_path_ar)

    # Optional: keep track of language (not used by models, but useful to inspect)
    combined_en["lang"] = "en"
    combined_ar["lang"] = "ar"

    combined = pd.concat([combined_en, combined_ar], ignore_index=True)

    # make "clean_text" column from tokens, same as before
    combined["clean_text"] = combined["tokens"].apply(tokens_to_string)

    X = combined[TEXT_COL].values
    y = combined[LABEL_COL].values

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.5,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    tfidf = TfidfVectorizer(
        max_features=30000,
        ngram_range=(1, 2),
        min_df=2,
    )

    tfidf.fit(X_train)

    X_train_tfidf = tfidf.transform(X_train)
    X_val_tfidf   = tfidf.transform(X_val)
    X_test_tfidf  = tfidf.transform(X_test)

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        X_train_tfidf, X_val_tfidf, X_test_tfidf,
        tfidf
    )


# =============================================================================
# Training Models functions
# =============================================================================

def train_LSTM(MAX_LEN, y_train, y_val, X_train_pad, X_val_pad, vocab_size):
    EMBED_DIM = 64      # size of word embeddings
    LSTM_UNITS = 64     # number of LSTM units
    DROPOUT_RATE = 0.3  # regularization
    BATCH_SIZE = 128
    EPOCHS = 10         # we'll use early stopping anyway

    model = Sequential([
        Embedding(
            input_dim=vocab_size,   # 🔥 use dynamic vocab size
            output_dim=EMBED_DIM,
            input_length=MAX_LEN
        ),
        Bidirectional(LSTM(LSTM_UNITS)),
        Dropout(DROPOUT_RATE),
        Dense(64, activation="relu"),
        Dropout(DROPOUT_RATE),
        Dense(1, activation="sigmoid")  # binary classification
    ])

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )

    y_train_np = np.array(y_train)
    y_val_np   = np.array(y_val)

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True
    )

    history = model.fit(
        X_train_pad, y_train_np,
        validation_data=(X_val_pad, y_val_np),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1
    )
    return model, history


def train_svm(X_train_tfidf, y_train): #TRAINING SVM
    # Create the SVM model
    svm_clf = LinearSVC(random_state=RANDOM_STATE)

    # Train (fit) on training TF-IDF features
    svm_clf.fit(X_train_tfidf, y_train)
    return (svm_clf)

def train_MultinomialNB(X_train_tfidf, y_train): #TRAINING MultinomialNB
    nb_clf = MultinomialNB(alpha=1) 
    nb_clf.fit(X_train_tfidf, y_train)
    return nb_clf

# =============================================================================
# Evaluation on validation and test data for the ML and NN models
# =============================================================================

def evaluate_ml_models(svm_clf, nb_clf, X_val_tfidf, y_val):
    """
    Use trained SVM and MultinomialNB models to predict on X_val_tfidf
    and print classification reports + confusion matrices.
    """
    # SVM
    print("===== SVM (LinearSVC) – Validation =====")
    y_val_pred_svm = svm_clf.predict(X_val_tfidf)
    print_clfreport_confusion(y_val, y_val_pred_svm)
    print()  # blank line for readability

    # MultinomialNB
    print("===== MultinomialNB – Validation =====")
    y_val_pred_nb = nb_clf.predict(X_val_tfidf)
    print_clfreport_confusion(y_val, y_val_pred_nb)

    return y_val_pred_svm, y_val_pred_nb

def evaluate_nn_model(model, X_val_pad, y_val):
    """
    Use the trained NN (LSTM) model to predict on X_val_pad
    and print the classification report + confusion matrix.
    """
    print("===== LSTM / NN – Validation =====")
    # Predict probabilities then threshold at 0.5
    y_val_pred_proba = model.predict(X_val_pad, verbose=0)
    y_val_pred = (y_val_pred_proba >= 0.5).astype(int).ravel()

    print_clfreport_confusion(y_val, y_val_pred)

    return y_val_pred


# =============================================================================
# HELPER functions
# =============================================================================

def print_clfreport_confusion(y_val, y_val_pred):
    print("Classification report:")
    print(classification_report(y_val, y_val_pred, digits=4))
    print("Confusion matrix (VAL):")
    print(confusion_matrix(y_val, y_val_pred))

def save_pipeline(results):
    """
    Save the trained pipeline (tfidf, SVM, NB, tokenizer, LSTM, MAX_LEN)
    to disk so we don't need to retrain every time.

    Saves under: ./saved_models/<language>/
    """
    language = results.get("language", "en")
    dir_path = os.path.join(BASE_MODEL_DIR, language)
    os.makedirs(dir_path, exist_ok=True)

    print(f"💾 Saving pipeline for language '{language}' to {dir_path}")

    # Save sklearn objects with joblib
    joblib.dump(results["tfidf"], os.path.join(dir_path, "tfidf.pkl"))
    joblib.dump(results["svm"],   os.path.join(dir_path, "svm.pkl"))
    joblib.dump(results["nb"],    os.path.join(dir_path, "nb.pkl"))

    # Save tokenizer
    joblib.dump(results["tokenizer"], os.path.join(dir_path, "tokenizer.pkl"))

    # Save metadata
    meta = {
        "language": language,
        "MAX_LEN": results["MAX_LEN"],
    }
    joblib.dump(meta, os.path.join(dir_path, "meta.pkl"))

    # Save LSTM model
    lstm_path = os.path.join(dir_path, "lstm.keras")
    results["lstm"].save(lstm_path)

    print("✅ Pipeline saved.")


def load_pipeline(language="en"):
    """
    Load a previously saved pipeline (no retraining).
    Looks in: ./saved_models/<language>/

    Returns a dict with:
      - language
      - tfidf
      - tokenizer
      - svm
      - nb
      - lstm
      - MAX_LEN
    """
    if language not in ("en", "ar", "multi"):
        raise ValueError("language must be 'en', 'ar', or 'multi'")
    dir_path = BASE_MODEL_DIR / language
    required_files = ("tfidf.pkl", "svm.pkl", "nb.pkl", "tokenizer.pkl", "meta.pkl", "lstm.keras")
    missing = [name for name in required_files if not (dir_path / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing saved artifacts in {dir_path}: {', '.join(missing)}. "
            "Copy the complete saved_models folder from the original project."
        )
    print(f"📂 Loading pipeline for language '{language}' from {dir_path}")

    tfidf      = joblib.load(os.path.join(dir_path, "tfidf.pkl"))
    svm_clf    = joblib.load(os.path.join(dir_path, "svm.pkl"))
    nb_clf     = joblib.load(os.path.join(dir_path, "nb.pkl"))
    tokenizer  = joblib.load(os.path.join(dir_path, "tokenizer.pkl"))
    meta       = joblib.load(os.path.join(dir_path, "meta.pkl"))
    lstm_model = load_model(os.path.join(dir_path, "lstm.keras"), compile=False)

    results = {
        "language": meta.get("language", language),
        "tfidf": tfidf,
        "tokenizer": tokenizer,
        "svm": svm_clf,
        "nb": nb_clf,
        "lstm": lstm_model,
        "MAX_LEN": meta["MAX_LEN"],
    }
    print("✅ Pipeline loaded.")
    return results



# =============================================================================
# full pipeline
# =============================================================================
def run_pipeline(
    language="en",
    raw_path_en=raw_data_path_en,
    raw_path_ar=raw_data_path_ar,
    cleaned_path_en=cleaned_data_path_en,
    cleaned_path_ar=cleaned_data_path_ar,
):
    """
    Full pipeline for a given language:

      language="en"   → only English
      language="ar"   → only Arabic
      language="multi"→ English + Arabic combined

      1) collect + combine raw datasets
      2) preprocess (cleaning + tokens)
      3) split + build TF-IDF
      4) tokenize + pad for LSTM
      5) train SVM, MultinomialNB, LSTM
      6) print eval for all 3 on val + test

    Returns a dict with models, tokenizer, tfidf, etc.
    """

    if language not in ("en", "ar", "multi"):
        raise ValueError("language must be 'en', 'ar', or 'multi'")

    print(f"\n📝 Running pipeline for language = {language}")

    # --------- STEP 1 + 2 + 3: dataset prep depends on mode ----------

    if language == "en":
        print("🔹 Step 1: collecting & combining EN datasets...")
        collect_combine_en(raw_path_en)

        print("🔹 Step 2: preprocessing EN...")
        preprocessing(
            language="en",
            raw_data_path_en=raw_path_en,
            raw_data_path_ar=raw_path_ar,
            cleaned_data_path_en=cleaned_path_en,
            cleaned_data_path_ar=cleaned_path_ar,
        )

        print("🔹 Step 3: preparing EN splits + TF-IDF...")
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            tfidf
        ) = prepare_for_modeling_ML_NN(
            language="en",
            cleaned_data_path_en=cleaned_path_en,
            cleaned_data_path_ar=cleaned_path_ar,
        )

    elif language == "ar":
        print("🔹 Step 1: collecting & combining AR datasets...")
        collect_combine_ar(raw_path_ar)

        print("🔹 Step 2: preprocessing AR...")
        preprocessing(
            language="ar",
            raw_data_path_en=raw_path_en,
            raw_data_path_ar=raw_path_ar,
            cleaned_data_path_en=cleaned_path_en,
            cleaned_data_path_ar=cleaned_path_ar,
        )

        print("🔹 Step 3: preparing AR splits + TF-IDF...")
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            tfidf
        ) = prepare_for_modeling_ML_NN(
            language="ar",
            cleaned_data_path_en=cleaned_path_en,
            cleaned_data_path_ar=cleaned_path_ar,
        )

    else:  # language == "multi"
        print("🔹 Step 1: collecting & combining EN + AR datasets...")
        collect_combine_en(raw_path_en)
        collect_combine_ar(raw_path_ar)

        print("🔹 Step 2: preprocessing EN + AR separately...")
        preprocessing(
            language="en",
            raw_data_path_en=raw_path_en,
            raw_data_path_ar=raw_path_ar,
            cleaned_data_path_en=cleaned_path_en,
            cleaned_data_path_ar=cleaned_path_ar,
        )
        preprocessing(
            language="ar",
            raw_data_path_en=raw_path_en,
            raw_data_path_ar=raw_path_ar,
            cleaned_data_path_en=cleaned_path_en,
            cleaned_data_path_ar=cleaned_path_ar,
        )

        print("🔹 Step 3: preparing MULTI (EN+AR) splits + TF-IDF...")
        (
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_tfidf, X_val_tfidf, X_test_tfidf,
            tfidf
        ) = prepare_for_modeling_ML_NN_multi(
            cleaned_data_path_en=cleaned_path_en,
            cleaned_data_path_ar=cleaned_path_ar,
        )

    # --------- STEP 4+: shared for all modes ----------

    # 4. Tokenize + pad for LSTM
    print("🔹 Step 4: tokenizing + padding for LSTM...")
    tokenizer, X_train_pad, X_val_pad, X_test_pad, MAX_LEN, vocab_size = prepare_for_modelingNN(
    X_train, X_val, X_test)

    # 5. Train classical ML models
    print("🔹 Step 5: training SVM + MultinomialNB...")
    svm_clf = train_svm(X_train_tfidf, y_train)
    nb_clf  = train_MultinomialNB(X_train_tfidf, y_train)

    # 6. Train LSTM
    print("🔹 Step 6: training LSTM...")
    lstm_model, lstm_history = train_LSTM(
    MAX_LEN,
    y_train, y_val,
    X_train_pad, X_val_pad,
    vocab_size
)

    # 7. Validation evaluation
    print("\n======================")
    print(f"🔸 VALIDATION RESULTS ({language})")
    print("======================\n")
    evaluate_ml_models(svm_clf, nb_clf, X_val_tfidf, y_val)
    evaluate_nn_model(lstm_model, X_val_pad, y_val)

    # 8. Test evaluation
    print("\n=================")
    print(f"🔸 TEST RESULTS ({language})")
    print("=================\n")

    print("===== SVM – Test =====")
    y_test_pred_svm = svm_clf.predict(X_test_tfidf)
    print_clfreport_confusion(y_test, y_test_pred_svm)

    print("\n===== MultinomialNB – Test =====")
    y_test_pred_nb = nb_clf.predict(X_test_tfidf)
    print_clfreport_confusion(y_test, y_test_pred_nb)

    print("\n===== LSTM – Test =====")
    y_test_proba_lstm = lstm_model.predict(X_test_pad, verbose=0)
    y_test_pred_lstm  = (y_test_proba_lstm >= 0.5).astype(int).ravel()
    print_clfreport_confusion(y_test, y_test_pred_lstm)

    # 9. Return everything useful
    return {
        "language": language,
        "tfidf": tfidf,
        "tokenizer": tokenizer,
        "svm": svm_clf,
        "nb": nb_clf,
        "lstm": lstm_model,
        "MAX_LEN": MAX_LEN,
        "vocab_size": vocab_size,
        "splits": {
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
        },
        "lstm_history": lstm_history,
    }



# =============================================================================
# for the demo
# =============================================================================
def predict_comment(comment, pipeline_results, model="svm"):
    """
    Predict if a single comment is bot-written or human-written.

    For language="multi", we auto-detect 'en' vs 'ar' for preprocessing.
    """
    pipeline_lang = pipeline_results.get("language", "en")

    # Decide which preprocessing language to use
    if pipeline_lang == "multi":
        # simple langdetect: choose 'ar' if top lang is Arabic, else 'en'
        effective_lang = "en"
        try:
            langs = detect_langs(str(comment))
            if langs:
                top = langs[0]
                if top.lang.startswith("ar"):
                    effective_lang = "ar"
                elif top.lang.startswith("en"):
                    effective_lang = "en"
        except Exception:
            effective_lang = "en"
    else:
        effective_lang = pipeline_lang

    # 1) Preprocess like training
    clean_text = preprocess_single_comment(comment, language=effective_lang)

    # 2) Choose model
    model = model.lower()
    if model not in ("svm", "nb", "lstm"):
        raise ValueError("model must be 'svm', 'nb', or 'lstm'")

    if model in ("svm", "nb"):
        tfidf = pipeline_results["tfidf"]
        X_feat = tfidf.transform([clean_text])

        if model == "svm":
            clf = pipeline_results["svm"]
        else:
            clf = pipeline_results["nb"]

        numeric_label = int(clf.predict(X_feat)[0])
        proba = None  # not using probabilities here

    else:  # LSTM
        tokenizer = pipeline_results["tokenizer"]
        MAX_LEN   = pipeline_results["MAX_LEN"]
        lstm_model = pipeline_results["lstm"]

        seq = tokenizer.texts_to_sequences([clean_text])
        pad = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

        proba = float(lstm_model.predict(pad, verbose=0)[0][0])
        numeric_label = int(proba >= 0.5)

    label_str = "bot" if numeric_label == 1 else "human"
    return label_str, numeric_label, proba


def run_console_demo():
    """
    Small interactive CLI:

    1) Ask if we want to:
       - run full pipeline (collect + preprocess + train + save)
       - load an already trained model

    2) Ask for language: en / ar / multi

    3) If loading a model:
       - ask for model type (svm / nb / lstm)
       - loop: ask for a comment, predict bot/human
       - ask if user wants to quit or enter another comment
    """
    print("=====================================")
    print("   Bot Comment Detector – Console    ")
    print("=====================================\n")

    while True:
        print("What do you want to do?")
        print("  1) Run full pipeline (train + save)")
        print("  2) Load a trained model and classify comments")
        print("  q) Quit")
        choice = input("Enter choice [1/2/q]: ").strip().lower()

        if choice == "q":
            print("👋 Bye!")
            break

        if choice not in ("1", "2"):
            print("❌ Invalid choice, please type 1, 2, or q.\n")
            continue

        # ----------------- ask for language -----------------
        print("\nWhich language setting?")
        print("  en    → English only")
        print("  ar    → Arabic only")
        print("  multi → English + Arabic combined")
        lang = input("Enter language [en/ar/multi]: ").strip().lower()

        while lang not in ("en", "ar", "multi"):
            lang = input("❌ Invalid language. Please choose [en/ar/multi]: ").strip().lower()

        # ----------------- branch: run full pipeline -----------------
        if choice == "1":
            print(f"\n🚀 Running full pipeline for language = {lang} ...")
            results = run_pipeline(language=lang)
            save_pipeline(results)
            print(f"✅ Training + saving completed for language = {lang}\n")
            # then go back to main menu
            continue

        # ----------------- branch: load model + classify -----------------
        else:
            print(f"\n📂 Loading model for language = {lang} ...")
            try:
                results = load_pipeline(language=lang)
            except Exception as e:
                print(f"❌ Could not load pipeline for language '{lang}': {e}\n")
                continue

            # choose which model to use
            model = input("Choose model [svm/nb/lstm, default=svm]: ").strip().lower()
            if model == "":
                model = "svm"
            while model not in ("svm", "nb", "lstm"):
                model = input("❌ Invalid model. Choose [svm/nb/lstm]: ").strip().lower()

            print("\n✅ Model loaded. Let's classify comments!")
            print("Type an empty line to go back to the main menu.\n")

            while True:
                comment = input("Enter a comment (or press Enter to go back): ").strip()
                if comment == "":
                    print("↩ Back to main menu.\n")
                    break

                label_str, numeric_label, proba = predict_comment(
                    comment,
                    results,
                    model=model
                )

                print("\n--- Prediction ---")
                print("Comment:", comment)
                print("Prediction:", label_str, f"(numeric={numeric_label})")
                if model == "lstm":
                    print(f"LSTM probability (bot=1): {proba:.4f}")
                print("------------------\n")

                # ask if we want another comment or back
                again = input("Press Enter for another comment, or type 'q' to go back: ").strip().lower()
                if again == "q":
                    print("↩ Back to main menu.\n")
                    break



if __name__ == "__main__":
    run_console_demo()
    
