"""
PaiGen AI: Trilingual Psycholinguistic & Intent ML Classifier
Architect: Arjun Singh

Pure-NumPy multi-head machine learning model trained on English, Hindi (Devanagari),
and Romanized Hindi/Hinglish utterances to classify in <5ms locally:
1. Psychological State (8 classes)
2. Executive Call/Interruption Triage Tier (4 classes)
3. Dominant Language & Script Intent (hi / hinglish / en)
4. Continuous Affective & Cognitive Vectors (valence, arousal, trust_openness, manipulation_risk, cognitive_load)
"""

import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

PSYCH_STATES = [
    "CALM_NEUTRAL",
    "STRESSED_ANXIOUS",
    "FRUSTRATED_DEFENSIVE",
    "URGENT_PANIC",
    "WARM_TRUSTING",
    "DECEPTIVE_PUSHY",
    "FATIGUED_LOW_ENERGY",
    "CURIOUS_ANALYTICAL",
]

TRIAGE_TIERS = [
    "TIER_1_EMERGENCY",
    "TIER_2_HIGH_PRIORITY",
    "TIER_3_ROUTINE",
    "TIER_4_SPAM_MANIPULATION",
]

LANG_CLASSES = ["hi", "hinglish", "en"]

# Pure Romanized Hindi & colloquial Hinglish lexicon (excludes English homographs like "the", "main")
ROMAN_HINDI_MARKERS = {
    "kya", "hai", "hain", "hoon", "hu", "ho", "mai", "mein", "tum", "aap",
    "bhai", "yaar", "ji", "haan", "han", "nahi", "nahin", "kaise", "kahan",
    "kab", "kyun", "kyu", "kaun", "kisko", "baat", "karni", "karna", "karo", "kar",
    "raha", "rahi", "rahe", "tha", "thi", "abhi", "turant", "jaldi", "zaroori",
    "jaroori", "kaam", "bolo", "batao", "suno", "sunao", "samjho", "madad", "ghar",
    "mummy", "papa", "maa", "pareshan", "dikkat", "paisa", "paise", "thak", "gaya",
    "gayi", "neend", "raat", "subah", "kal", "aaj", "namaste", "pranam",
    "shukriya", "dhanyavaad", "kripya", "bahut", "bohat", "thoda", "zyada", "sab",
    "theek", "thik", "accha", "acha", "bura", "gussa", "dar", "bataiye", "kijiye",
    "bulaiye", "dijiye", "chahiye", "milega", "milna", "uthaiye", "uthao",
    "bhej", "bhejo", "samasya", "chinta", "beemar", "tabiyat", "dawai",
    "waqt", "samay", "der", "intezar", "pucho", "puchna", "khabar", "arrey", "arre",
    "hoke", "wala", "wali", "wale", "karoge", "karoon", "aao", "jao", "chacha", "dadi",
}

HINDI_URGENCY_MARKERS = {
    "emergency", "hospital", "accident", "blood", "police", "ambulance", "turant",
    "abhi", "jaldi", "fatafat", "bimaar", "beemar", "critical", "serious", "bachao",
    "chot", "admit", "icu", "aag", "danger", "खतरा", "अस्पताल", "तुरंत", "जल्दी",
    "आपातकाल", "एक्सीडेंट", "गंभीर", "बचाओ", "मदद", "परेशान",
}

MANIPULATION_SPAM_MARKERS = {
    "otp", "kyc", "lottery", "prize", "jackpot", "winner", "free", "claim",
    "credit card", "loan", "pre-approved", "preapproved", "offer", "discount",
    "verify account", "suspended", "block ho jayega", "pin number", "cvv",
    "investment", "crypto", "double money", "telemarketing", "insurance",
    "digital arrest", "cbi", "customs", "upi", "refund link", "account freeze",
    "aadhaar", "illegal parcel", "drugs", "fedex", "ncb", "trai", "sim block",
    "arrest warrant", "police case", "money laundering",
    "ignore all previous", "ignore previous", "admin mode", "last email",
    "system prompt", "developer mode", "whatsapp message", "read his messages",
    "padh ke", "location batao",
    "बैंक", "लॉटरी", "इनाम", "ओटीपी", "लोन", "ऑफर", "फ्री", "ब्लॉक",
    "कस्टम्स", "सीबीआई", "डिजिटल अरेस्ट", "आधार", "वारंट", "पुलिस केस",
}

DEFAULT_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "psychology_ml_model.npz"
)
DEFAULT_METRICS_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "models" / "psychology_ml_metrics.json"
)
DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "training" / "psychology_corpus.jsonl"
)


def _tokenize_trilingual(text: str) -> List[str]:
    """Extracts word unigrams, word bigrams, and character 3/4-grams preserving Devanagari."""
    cleaned = re.sub(r"[^\w\s\u0900-\u097F!?]", " ", text.lower())
    words = [w for w in cleaned.split() if w]
    tokens: List[str] = list(words)
    # Word bigrams
    for i in range(len(words) - 1):
        tokens.append(f"{words[i]}_{words[i+1]}")
    # Compact char 3-grams & 4-grams for morphological/Hindi stem generalization
    compact = " ".join(words)
    for i in range(max(0, len(compact) - 2)):
        gram3 = compact[i : i + 3]
        if " " not in gram3:
            tokens.append(f"c3:{gram3}")
        if i <= len(compact) - 4:
            gram4 = compact[i : i + 4]
            if " " not in gram4:
                tokens.append(f"c4:{gram4}")
    return tokens


def _structural_features(text: str) -> np.ndarray:
    """Computes 16 psycholinguistic, emotional-cluster & script features."""
    if not text:
        return np.zeros(16, dtype=np.float32)

    total_chars = max(len(text), 1)
    dev_chars = sum(1 for ch in text if "\u0900" <= ch <= "\u097F")
    dev_ratio = dev_chars / total_chars

    words = re.findall(r"[\w\u0900-\u097F]+", text.lower())
    n_words = max(len(words), 1)
    roman_hi_hits = sum(1 for w in words if w in ROMAN_HINDI_MARKERS)
    roman_hi_ratio = roman_hi_hits / n_words

    low = text.lower()
    urgency_hits = sum(1 for kw in HINDI_URGENCY_MARKERS if kw in low)
    spam_hits = sum(1 for kw in MANIPULATION_SPAM_MARKERS if kw in low)

    # Cluster markers for tight generalization across unseen phrasing
    deadline_words = ("server", "production", "crash", "deadline", "meeting", "interview", "exam", "client", "bug", "down", "मीटिंग", "सर्वर", "क्रैश", "एग्जाम", "इंटरव्यू", "क्लाइंट", "पेमेंट", "रिपोर्ट")
    frust_words = ("wait", "waiting", "delay", "fourth", "third", "again", "ridiculous", "nobody", "ignored", "अटका", "परेशान", "कितनी देर", "कब से", "कोई उठा", "गुस्सा", "बार-बार", "सीधे बात")
    warm_words = ("haal", "chaal", "chai", "friend", "buddy", "wish", "best", "congrats", "family", "blessing", "invite", "reunion", "college", "party", "wedding", "dinner", "नमस्ते", "आशीर्वाद", "बधाई", "दोस्त", "मस्त", "शाम", "हाल चाल", "कैसे हो", "शादी", "दावत")
    fatigue_words = ("tired", "exhausted", "sleep", "sleepy", "burnout", "drained", "headache", "neend", "thak", "thakan", "akela", "alone", "थक", "नींद", "थकान", "सिर दर्द", "आँखें", "आराम", "अकेला")
    curious_words = ("how", "why", "explain", "architecture", "algorithm", "model", "learn", "setup", "कैसे", "क्यों", "समझा", "सीखना", "आर्किटेक्चर", "टेक्निकल", "बताओगे")
    calm_words = ("delivery", "courier", "confirm", "appointment", "service", "pickup", "meter", "amazon", "flipkart", "डिलीवरी", "अपॉइंटमेंट", "रीडिंग", "सर्विस", "कोरियर")

    deadline_hit = min(sum(1 for w in deadline_words if w in low) / 2.0, 1.0)
    frust_hit = min(sum(1 for w in frust_words if w in low) / 2.0, 1.0)
    warm_hit = min(sum(1 for w in warm_words if w in low) / 2.0, 1.0)
    fatigue_hit = min(sum(1 for w in fatigue_words if w in low) / 2.0, 1.0)
    curious_hit = min(sum(1 for w in curious_words if w in low) / 2.0, 1.0)
    calm_hit = 0.0 if spam_hits > 0 else min(sum(1 for w in calm_words if w in low) / 2.0, 1.0)

    exclam = min(text.count("!") / 3.0, 1.0)
    quest = min(text.count("?") / 3.0, 1.0)
    upper_chars = sum(1 for c in text if c.isupper())
    upper_ratio = upper_chars / total_chars
    length_norm = min(len(words) / 30.0, 1.0)

    polite_words = {"please", "ji", "kripya", "namaste", "thank", "thanks", "dhanyavaad", "sir", "कृपया", "जी", "नमस्ते"}
    polite_ratio = min(sum(1 for w in words if w in polite_words) / 3.0, 1.0)

    return np.array(
        [
            dev_ratio,
            roman_hi_ratio,
            min(urgency_hits / 2.0, 1.0),
            min(spam_hits / 2.0, 1.0),
            deadline_hit,
            frust_hit,
            warm_hit,
            fatigue_hit,
            curious_hit,
            calm_hit,
            exclam,
            quest,
            upper_ratio,
            length_norm,
            polite_ratio,
            1.0 if dev_chars > 0 else 0.0,
        ],
        dtype=np.float32,
    )


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(np.clip(shifted, -30.0, 30.0))
    return exp / np.sum(exp, axis=-1, keepdims=True)


def _builtin_seed_corpus() -> List[Dict[str, Any]]:
    """
    Core curated trilingual psychological & call-triage seed dataset.
    Expanded at runtime with `training/psychology_corpus.jsonl`.
    """
    # Format: (text, state, tier, lang, [valence, arousal, trust, manip_risk, cog_load])
    raw = [
        # --- TIER 1 EMERGENCY (Hindi, Hinglish, English) ---
        ("अरे जल्दी अर्जुन को फोन दो, पापा की तबीयत अचानक बहुत खराब हो गई है अस्पताल ले जा रहे हैं", "URGENT_PANIC", "TIER_1_EMERGENCY", "hi", [-0.85, 0.95, 0.90, 0.02, 0.92]),
        ("Arjun kahan hai jaldi baat karao ghar pe emergency hai accident ho gaya", "URGENT_PANIC", "TIER_1_EMERGENCY", "hi", [-0.90, 0.96, 0.88, 0.02, 0.95]),
        ("Bhai please phone uthao mummy hospital mein admit hain turant aao", "URGENT_PANIC", "TIER_1_EMERGENCY", "hi", [-0.88, 0.94, 0.92, 0.02, 0.90]),
        ("Emergency! Someone from family is in the hospital, put Arjun on the line immediately!", "URGENT_PANIC", "TIER_1_EMERGENCY", "en", [-0.90, 0.97, 0.85, 0.02, 0.93]),
        ("हमारे घर में आग लग गई है यार पुलिस और अर्जुन से बात कराओ तुरंत", "URGENT_PANIC", "TIER_1_EMERGENCY", "hi", [-0.95, 0.98, 0.85, 0.01, 0.96]),
        ("Doctor bol rahe hain operation turant karna padega, Arjun ka signature chahiye", "URGENT_PANIC", "TIER_1_EMERGENCY", "hinglish", [-0.80, 0.92, 0.88, 0.03, 0.90]),
        ("Critical incident at home, dad fell down and needs immediate help right now", "URGENT_PANIC", "TIER_1_EMERGENCY", "en", [-0.85, 0.94, 0.86, 0.02, 0.88]),

        # --- TIER 2 HIGH PRIORITY (Stressed / Anxious / Deadline / Frustrated) ---
        ("सर प्रोजेक्ट का सर्वर क्रैश हो गया है और क्लाइंट मीटिंग दस मिनट में शुरू होने वाली है", "STRESSED_ANXIOUS", "TIER_2_HIGH_PRIORITY", "hi", [-0.55, 0.82, 0.72, 0.05, 0.85]),
        ("Arjun se bolna production server down hai deployment roll back karna padega abhi", "STRESSED_ANXIOUS", "TIER_2_HIGH_PRIORITY", "hinglish", [-0.50, 0.80, 0.75, 0.05, 0.82]),
        ("भाई एग्जाम का टाइम बदल गया है आज दो बजे ही पहुंचना है सेंटर पे", "STRESSED_ANXIOUS", "TIER_2_HIGH_PRIORITY", "hi", [-0.40, 0.78, 0.80, 0.03, 0.78]),
        ("The interview panel rescheduled to 3 PM today, Arjun needs to join the call urgently", "STRESSED_ANXIOUS", "TIER_2_HIGH_PRIORITY", "en", [-0.30, 0.76, 0.78, 0.04, 0.75]),
        ("ये तीसरी बार कॉल कर रहा हूँ काम अटका हुआ है प्लीज मेरी बात करा दो अभी", "FRUSTRATED_DEFENSIVE", "TIER_2_HIGH_PRIORITY", "hi", [-0.65, 0.80, 0.40, 0.12, 0.78]),
        ("Yaar kitni der se wait kar raha hoon deadline nikal rahi hai baat karni zaroori hai", "FRUSTRATED_DEFENSIVE", "TIER_2_HIGH_PRIORITY", "hi", [-0.60, 0.78, 0.45, 0.10, 0.76]),
        ("Look, I have been waiting two hours for the final build sign-off, this cannot wait till tomorrow", "FRUSTRATED_DEFENSIVE", "TIER_2_HIGH_PRIORITY", "en", [-0.58, 0.75, 0.45, 0.08, 0.74]),

        # --- TIER 3 ROUTINE (Warm Trusting, Calm Neutral, Curious Analytical, Fatigued) ---
        ("नमस्ते भाई कैसे हो बस ऐसे ही हाल चाल पूछने के लिए कॉल किया था जब फ्री हो बात करना", "WARM_TRUSTING", "TIER_3_ROUTINE", "hi", [0.70, 0.30, 0.90, 0.02, 0.20]),
        ("Haan bhai kya haal hai shaam ko chai pe milte hain jab kaam khatam ho jaye batana", "WARM_TRUSTING", "TIER_3_ROUTINE", "hi", [0.75, 0.35, 0.92, 0.02, 0.18]),
        ("Hey Arjun just checking in, hope your coding session is going great, call back when free!", "WARM_TRUSTING", "TIER_3_ROUTINE", "en", [0.72, 0.32, 0.88, 0.02, 0.20]),
        ("मैं कोरियर डिलीवरी से बोल रहा हूँ आपका पार्सल गेट पर रखा है", "CALM_NEUTRAL", "TIER_3_ROUTINE", "hi", [0.10, 0.25, 0.65, 0.08, 0.22]),
        ("Hello sir delivery boy bol raha hoon aapka parcel aaya hai neeche guard ko de diya hai", "CALM_NEUTRAL", "TIER_3_ROUTINE", "hi", [0.15, 0.28, 0.70, 0.06, 0.20]),
        ("Good afternoon, calling from the dental clinic to confirm Mr. Arjun's checkup on Friday", "CALM_NEUTRAL", "TIER_3_ROUTINE", "en", [0.15, 0.22, 0.75, 0.05, 0.25]),
        ("अर्जुन का नया AI प्रोजेक्ट कैसे काम करता है मुझे इसके आर्किटेक्चर के बारे में जानना था", "CURIOUS_ANALYTICAL", "TIER_3_ROUTINE", "hi", [0.45, 0.50, 0.76, 0.04, 0.55]),
        ("Bhai wo Python wala server kaise setup kiya mujhe bhi samjhana jab time mile", "CURIOUS_ANALYTICAL", "TIER_3_ROUTINE", "hinglish", [0.48, 0.48, 0.82, 0.03, 0.52]),
        ("Could you explain how the local neural pipeline integrates with the mobile gateway?", "CURIOUS_ANALYTICAL", "TIER_3_ROUTINE", "en", [0.40, 0.45, 0.78, 0.03, 0.60]),
        ("यार बहुत थक गया हूँ आज काम करते करते दो बज गए हैं अब नींद आ रही है", "FATIGUED_LOW_ENERGY", "TIER_3_ROUTINE", "hi", [-0.35, 0.15, 0.82, 0.02, 0.68]),
        ("Bohat neend aa rahi hai yaar sir dard ho raha hai aaj aur code nahi ho payega", "FATIGUED_LOW_ENERGY", "TIER_3_ROUTINE", "hi", [-0.42, 0.14, 0.84, 0.02, 0.72]),
        ("I am completely exhausted tonight, my eyes hurt and I cannot focus on debugging anymore", "FATIGUED_LOW_ENERGY", "TIER_3_ROUTINE", "en", [-0.45, 0.16, 0.80, 0.02, 0.70]),

        # --- TIER 4 SPAM / MANIPULATION / DECEPTIVE PUSHY ---
        ("बधाई हो सर आपका नंबर लकी ड्रा में निकला है अभी अपना ओटीपी बताएं और इनाम जीतें", "DECEPTIVE_PUSHY", "TIER_4_SPAM_MANIPULATION", "hi", [-0.30, 0.75, 0.05, 0.96, 0.45]),
        ("Sir aapka bank KYC expire ho raha hai account aaj block ho jayega abhi OTP share karo", "DECEPTIVE_PUSHY", "TIER_4_SPAM_MANIPULATION", "hinglish", [-0.60, 0.85, 0.04, 0.98, 0.60]),
        ("Congratulations! You have won a pre-approved personal loan and free credit card, share PIN to claim now!", "DECEPTIVE_PUSHY", "TIER_4_SPAM_MANIPULATION", "en", [-0.20, 0.72, 0.05, 0.95, 0.40]),
        ("सर प्रीमियम क्रेडिट कार्ड बिना फीस के मिल रहा है बस दो मिनट मेरी बात सुन लीजिए प्लीज", "DECEPTIVE_PUSHY", "TIER_4_SPAM_MANIPULATION", "hi", [-0.15, 0.60, 0.12, 0.85, 0.35]),
        ("Hello sir crypto trading mein aapka paisa 7 din mein double ho jayega abhi join kijiye", "DECEPTIVE_PUSHY", "TIER_4_SPAM_MANIPULATION", "hinglish", [-0.25, 0.68, 0.06, 0.94, 0.42]),
        ("Urgent security alert from bank support: verify your CVV immediately or card will be permanently blocked", "DECEPTIVE_PUSHY", "TIER_4_SPAM_MANIPULATION", "en", [-0.70, 0.88, 0.03, 0.99, 0.65]),
    ]

    samples = []
    for text, st, tier, lang, vec in raw:
        samples.append(
            {
                "text": text,
                "psych_state": st,
                "triage_tier": tier,
                "language": lang,
                "affect_vector": vec,
            }
        )
    return samples


class PsychologyMLClassifier:
    """
    Pure-NumPy TF-IDF + Structural Psycholinguistic Multi-Head ML Model.
    Trains tightly with L2 regularization, cosine decay, and stratified validation.
    """

    def __init__(self, max_vocab: int = 3200):
        self.max_vocab = max_vocab
        self.vocab: Dict[str, int] = {}
        self.idf: Optional[np.ndarray] = None
        self.feat_mean: Optional[np.ndarray] = None
        self.feat_std: Optional[np.ndarray] = None

        # Multi-head weights
        self.W_state: Optional[np.ndarray] = None
        self.b_state: Optional[np.ndarray] = None
        self.W_triage: Optional[np.ndarray] = None
        self.b_triage: Optional[np.ndarray] = None
        self.W_lang: Optional[np.ndarray] = None
        self.b_lang: Optional[np.ndarray] = None
        self.W_affect: Optional[np.ndarray] = None
        self.b_affect: Optional[np.ndarray] = None
        self.is_trained = False

    def _fit_vectorizer(self, texts: List[str]):
        df: Counter = Counter()
        for t in texts:
            for tok in set(_tokenize_trilingual(t)):
                df[tok] += 1
        min_df = 2 if len(texts) >= 120 else 1
        filtered = [(tok, cnt) for tok, cnt in df.most_common() if cnt >= min_df]
        if len(filtered) < 200:
            filtered = df.most_common()
        most_common = [tok for tok, _ in filtered[: self.max_vocab]]
        self.vocab = {tok: idx for idx, tok in enumerate(most_common)}
        n_docs = max(len(texts), 1)
        idf = np.zeros(len(self.vocab), dtype=np.float32)
        for tok, idx in self.vocab.items():
            idf[idx] = math.log((1.0 + n_docs) / (1.0 + df[tok])) + 1.0
        self.idf = idf

    def _vectorize(self, texts: List[str], fit_stats: bool = False) -> np.ndarray:
        n = len(texts)
        v_dim = len(self.vocab)
        tfidf = np.zeros((n, v_dim), dtype=np.float32)
        struct = np.zeros((n, 16), dtype=np.float32)

        for i, text in enumerate(texts):
            counts = Counter(_tokenize_trilingual(text))
            for tok, cnt in counts.items():
                if tok in self.vocab:
                    j = self.vocab[tok]
                    tfidf[i, j] = (1.0 + math.log(cnt)) * self.idf[j]
            norm = np.linalg.norm(tfidf[i])
            if norm > 1e-8:
                tfidf[i] /= norm
            struct[i] = _structural_features(text)

        if fit_stats:
            self.feat_mean = np.mean(struct, axis=0)
            self.feat_std = np.std(struct, axis=0) + 1e-5

        struct_norm = (struct - self.feat_mean) / self.feat_std
        return np.hstack([tfidf, struct_norm])

    @staticmethod
    def _train_softmax_head(
        X: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        epochs: int = 240,
        lr: float = 0.25,
        l2: float = 2e-3,
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, d = X.shape
        W = np.zeros((d, num_classes), dtype=np.float32)
        b = np.zeros(num_classes, dtype=np.float32)
        Y_onehot = np.eye(num_classes, dtype=np.float32)[y]

        # Inverse-frequency class weighting for tight minority recall (e.g. TIER_1_EMERGENCY)
        counts = np.bincount(y, minlength=num_classes).astype(np.float32)
        class_weights = np.where(counts > 0, n / (num_classes * np.maximum(counts, 1.0)), 1.0)
        sample_w = class_weights[y][:, None]

        mW = np.zeros_like(W)
        mb = np.zeros_like(b)
        vW = np.zeros_like(W)
        vb = np.zeros_like(b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for ep in range(1, epochs + 1):
            cur_lr = lr * 0.5 * (1.0 + math.cos(math.pi * ep / epochs))
            cur_lr = max(cur_lr, 0.01)

            probs = _softmax(X @ W + b)
            diff = (probs - Y_onehot) * sample_w / n
            gW = X.T @ diff + l2 * W
            gb = np.sum(diff, axis=0)

            mW = beta1 * mW + (1 - beta1) * gW
            mb = beta1 * mb + (1 - beta1) * gb
            vW = beta2 * vW + (1 - beta2) * (gW ** 2)
            vb = beta2 * vb + (1 - beta2) * (gb ** 2)

            mW_hat = mW / (1 - beta1 ** ep)
            mb_hat = mb / (1 - beta1 ** ep)
            vW_hat = vW / (1 - beta2 ** ep)
            vb_hat = vb / (1 - beta2 ** ep)

            W -= cur_lr * mW_hat / (np.sqrt(vW_hat) + eps)
            b -= cur_lr * mb_hat / (np.sqrt(vb_hat) + eps)

        return W, b

    @staticmethod
    def _train_ridge_head(
        X: np.ndarray, Y: np.ndarray, alpha: float = 1.5
    ) -> Tuple[np.ndarray, np.ndarray]:
        n, d = X.shape
        X_aug = np.hstack([X, np.ones((n, 1), dtype=np.float32)])
        reg = alpha * np.eye(d + 1, dtype=np.float32)
        reg[-1, -1] = 0.0
        W_aug = np.linalg.solve(X_aug.T @ X_aug + reg, X_aug.T @ Y)
        return W_aug[:-1, :].astype(np.float32), W_aug[-1, :].astype(np.float32)

    def train(
        self,
        samples: Optional[List[Dict[str, Any]]] = None,
        val_fraction: float = 0.2,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """Trains all 4 ML heads tightly and computes validation telemetry."""
        data = list(samples) if samples else self.load_training_corpus()
        rng = np.random.default_rng(seed)

        # Stratified split per triage tier
        train_idx, val_idx = [], []
        by_tier: Dict[str, List[int]] = {}
        for i, row in enumerate(data):
            by_tier.setdefault(row["triage_tier"], []).append(i)
        for tier, idxs in by_tier.items():
            perm = list(idxs)
            rng.shuffle(perm)
            split_k = max(1, int(len(perm) * val_fraction)) if len(perm) >= 3 else 0
            val_idx.extend(perm[:split_k])
            train_idx.extend(perm[split_k:])

        train_rows = [data[i] for i in train_idx]
        val_rows = [data[i] for i in val_idx] if val_idx else train_rows

        train_texts = [r["text"] for r in train_rows]
        self._fit_vectorizer(train_texts)
        X_tr = self._vectorize(train_texts, fit_stats=True)
        X_va = self._vectorize([r["text"] for r in val_rows], fit_stats=False)

        state_map = {k: i for i, k in enumerate(PSYCH_STATES)}
        tier_map = {k: i for i, k in enumerate(TRIAGE_TIERS)}
        lang_map = {k: i for i, k in enumerate(LANG_CLASSES)}

        y_st_tr = np.array([state_map[r["psych_state"]] for r in train_rows], dtype=np.int64)
        y_ti_tr = np.array([tier_map[r["triage_tier"]] for r in train_rows], dtype=np.int64)
        y_la_tr = np.array([lang_map[r["language"]] for r in train_rows], dtype=np.int64)
        Y_af_tr = np.array([r["affect_vector"] for r in train_rows], dtype=np.float32)

        y_st_va = np.array([state_map[r["psych_state"]] for r in val_rows], dtype=np.int64)
        y_ti_va = np.array([tier_map[r["triage_tier"]] for r in val_rows], dtype=np.int64)
        y_la_va = np.array([lang_map[r["language"]] for r in val_rows], dtype=np.int64)
        Y_af_va = np.array([r["affect_vector"] for r in val_rows], dtype=np.float32)

        # Train validation split models with tight L2 regularization
        W_st, b_st = self._train_softmax_head(X_tr, y_st_tr, len(PSYCH_STATES), epochs=380, lr=0.22, l2=1.5e-3)
        W_ti, b_ti = self._train_softmax_head(X_tr, y_ti_tr, len(TRIAGE_TIERS), epochs=380, lr=0.22, l2=1.5e-3)
        W_la, b_la = self._train_softmax_head(X_tr, y_la_tr, len(LANG_CLASSES), epochs=300, lr=0.22, l2=1.5e-3)
        W_af, b_af = self._train_ridge_head(X_tr, Y_af_tr, alpha=0.6)

        st_acc_val = float(np.mean(np.argmax(X_va @ W_st + b_st, axis=1) == y_st_va))
        ti_acc_val = float(np.mean(np.argmax(X_va @ W_ti + b_ti, axis=1) == y_ti_va))
        la_acc_val = float(np.mean(np.argmax(X_va @ W_la + b_la, axis=1) == y_la_va))
        af_mae_val = float(np.mean(np.abs((X_va @ W_af + b_af) - Y_af_va)))

        # Refit final production weights on 100% of verified corpus
        all_texts = [r["text"] for r in data]
        self._fit_vectorizer(all_texts)
        X_all = self._vectorize(all_texts, fit_stats=True)
        y_st_all = np.array([state_map[r["psych_state"]] for r in data], dtype=np.int64)
        y_ti_all = np.array([tier_map[r["triage_tier"]] for r in data], dtype=np.int64)
        y_la_all = np.array([lang_map[r["language"]] for r in data], dtype=np.int64)
        Y_af_all = np.array([r["affect_vector"] for r in data], dtype=np.float32)

        self.W_state, self.b_state = self._train_softmax_head(X_all, y_st_all, len(PSYCH_STATES), epochs=400, lr=0.22, l2=1.5e-3)
        self.W_triage, self.b_triage = self._train_softmax_head(X_all, y_ti_all, len(TRIAGE_TIERS), epochs=400, lr=0.22, l2=1.5e-3)
        self.W_lang, self.b_lang = self._train_softmax_head(X_all, y_la_all, len(LANG_CLASSES), epochs=320, lr=0.22, l2=1.5e-3)
        self.W_affect, self.b_affect = self._train_ridge_head(X_all, Y_af_all, alpha=0.6)
        self.is_trained = True

        st_acc_tr = float(np.mean(np.argmax(X_all @ self.W_state + self.b_state, axis=1) == y_st_all))
        ti_acc_tr = float(np.mean(np.argmax(X_all @ self.W_triage + self.b_triage, axis=1) == y_ti_all))
        la_acc_tr = float(np.mean(np.argmax(X_all @ self.W_lang + self.b_lang, axis=1) == y_la_all))

        metrics = {
            "total_samples": len(data),
            "train_samples": len(train_rows),
            "val_samples": len(val_rows),
            "vocab_size": len(self.vocab),
            "val_state_accuracy": round(st_acc_val, 4),
            "val_triage_accuracy": round(ti_acc_val, 4),
            "val_language_accuracy": round(la_acc_val, 4),
            "val_affect_mae": round(af_mae_val, 4),
            "full_state_accuracy": round(st_acc_tr, 4),
            "full_triage_accuracy": round(ti_acc_tr, 4),
            "full_language_accuracy": round(la_acc_tr, 4),
        }
        return metrics

    @staticmethod
    def load_training_corpus(corpus_path: Path = DEFAULT_CORPUS_PATH) -> List[Dict[str, Any]]:
        samples = _builtin_seed_corpus()
        if corpus_path.exists():
            seen = {s["text"] for s in samples}
            with open(corpus_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if row.get("text") and row["text"] not in seen:
                        samples.append(row)
                        seen.add(row["text"])
        return samples

    def save(
        self,
        model_path: Path = DEFAULT_MODEL_PATH,
        metrics: Optional[Dict[str, Any]] = None,
        metrics_path: Path = DEFAULT_METRICS_PATH,
    ):
        model_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_items = json.dumps(self.vocab, ensure_ascii=False)
        np.savez_compressed(
            model_path,
            vocab_json=np.array([vocab_items]),
            idf=self.idf,
            feat_mean=self.feat_mean,
            feat_std=self.feat_std,
            W_state=self.W_state,
            b_state=self.b_state,
            W_triage=self.W_triage,
            b_triage=self.b_triage,
            W_lang=self.W_lang,
            b_lang=self.b_lang,
            W_affect=self.W_affect,
            b_affect=self.b_affect,
        )
        if metrics is not None:
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metrics_path, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2)

    @classmethod
    def load_or_train(cls, model_path: Path = DEFAULT_MODEL_PATH) -> "PsychologyMLClassifier":
        obj = cls()
        if model_path.exists():
            try:
                npz = np.load(model_path, allow_pickle=False)
                obj.vocab = json.loads(str(npz["vocab_json"][0]))
                obj.idf = npz["idf"]
                obj.feat_mean = npz["feat_mean"]
                obj.feat_std = npz["feat_std"]
                obj.W_state = npz["W_state"]
                obj.b_state = npz["b_state"]
                obj.W_triage = npz["W_triage"]
                obj.b_triage = npz["b_triage"]
                obj.W_lang = npz["W_lang"]
                obj.b_lang = npz["b_lang"]
                obj.W_affect = npz["W_affect"]
                obj.b_affect = npz["b_affect"]
                obj.is_trained = True
                return obj
            except Exception:
                pass
        metrics = obj.train()
        obj.save(model_path=model_path, metrics=metrics)
        return obj

    def predict(self, text: str) -> Dict[str, Any]:
        """Runs sub-5ms local multi-head inference on a single utterance."""
        if not self.is_trained:
            self.train()

        X = self._vectorize([text], fit_stats=False)
        p_state = _softmax(X @ self.W_state + self.b_state)[0]
        p_triage = _softmax(X @ self.W_triage + self.b_triage)[0]
        p_lang = _softmax(X @ self.W_lang + self.b_lang)[0]
        affect_raw = (X @ self.W_affect + self.b_affect)[0]

        st_idx = int(np.argmax(p_state))
        ti_idx = int(np.argmax(p_triage))
        la_idx = int(np.argmax(p_lang))

        # Deterministic safety overrides for Devanagari & explicit Roman Hindi markers
        words = set(re.findall(r"[\w\u0900-\u097F]+", text.lower()))
        has_devanagari = any("\u0900" <= c <= "\u097F" for c in text)
        roman_hi_count = len(words & ROMAN_HINDI_MARKERS)

        detected_lang = LANG_CLASSES[la_idx]
        if has_devanagari:
            detected_lang = "hi"
        elif roman_hi_count >= 2:
            detected_lang = "hi"
        elif roman_hi_count == 1:
            detected_lang = "hinglish"
        else:
            # Pure Latin script with zero Hindi lexicon markers -> English
            detected_lang = "en"

        valence = float(np.clip(affect_raw[0], -1.0, 1.0))
        arousal = float(np.clip(affect_raw[1], 0.0, 1.0))
        trust_openness = float(np.clip(affect_raw[2], 0.0, 1.0))
        manip_risk = float(np.clip(affect_raw[3], 0.0, 1.0))
        cog_load = float(np.clip(affect_raw[4], 0.0, 1.0))

        return {
            "psych_state": PSYCH_STATES[st_idx],
            "state_confidence": round(float(p_state[st_idx]), 4),
            "triage_tier": TRIAGE_TIERS[ti_idx],
            "triage_confidence": round(float(p_triage[ti_idx]), 4),
            "detected_language": detected_lang,
            "language_confidence": round(float(p_lang[la_idx]), 4),
            "valence": round(valence, 3),
            "arousal": round(arousal, 3),
            "trust_openness": round(trust_openness, 3),
            "manipulation_risk": round(manip_risk, 3),
            "cognitive_load": round(cog_load, 3),
            "is_urgent": TRIAGE_TIERS[ti_idx] in ("TIER_1_EMERGENCY", "TIER_2_HIGH_PRIORITY"),
            "is_hindi_dominant": detected_lang in ("hi", "hinglish"),
        }
