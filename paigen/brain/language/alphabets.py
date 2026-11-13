"""
PaiGen AI: Alphabets, Phonetics, and Script Foundation (English & Hindi)
Architect: Arjun Singh

Defines the ground-truth linguistic primitives for English (Roman) and
Hindi (Devanagari + Hinglish transliteration), providing syllable tokens,
vowels, consonants, matras, and phonetic pronunciation maps.
"""

from typing import Dict, List, Tuple
from pydantic import BaseModel, Field


class PhoneticUnit(BaseModel):
    char: str
    script: str  # "Roman", "Devanagari", "Hinglish"
    phonetic_ipa: str
    transliterations: List[str]
    category: str  # "vowel", "consonant", "matra", "special"
    example_words: Dict[str, str] = Field(default_factory=dict)  # {"en": "apple", "hi": "सेब"}


class LanguageAlphabetRegistry:
    """Ground-truth repository of English and Hindi alphabets and phonemes."""

    # English Roman Alphabet (A-Z)
    ENGLISH_ALPHABETS: Dict[str, Dict[str, str]] = {
        "A": {"ipa": "/eɪ/", "type": "vowel", "sample": "Apple / Aman"},
        "B": {"ipa": "/biː/", "type": "consonant", "sample": "Book / Bhaav"},
        "C": {"ipa": "/siː/", "type": "consonant", "sample": "Cat / Chain"},
        "D": {"ipa": "/diː/", "type": "consonant", "sample": "Dog / Dost"},
        "E": {"ipa": "/iː/", "type": "vowel", "sample": "Energy / Ek"},
        "F": {"ipa": "/ɛf/", "type": "consonant", "sample": "Friend / Fikar"},
        "G": {"ipa": "/dʒiː/", "type": "consonant", "sample": "Growth / Gyan"},
        "H": {"ipa": "/eɪtʃ/", "type": "consonant", "sample": "Heart / Himmat"},
        "I": {"ipa": "/aɪ/", "type": "vowel", "sample": "Insight / Irada"},
        "J": {"ipa": "/dʒeɪ/", "type": "consonant", "sample": "Journey / Josh"},
        "K": {"ipa": "/keɪ/", "type": "consonant", "sample": "Kind / Kripa"},
        "L": {"ipa": "/ɛl/", "type": "consonant", "sample": "Loyal / Lakshya"},
        "M": {"ipa": "/ɛm/", "type": "consonant", "sample": "Mind / Man"},
        "N": {"ipa": "/ɛn/", "type": "consonant", "sample": "Nature / Niyamo"},
        "O": {"ipa": "/oʊ/", "type": "vowel", "sample": "Objective / Om"},
        "P": {"ipa": "/piː/", "type": "consonant", "sample": "PaiGen / Pyaar"},
        "Q": {"ipa": "/kjuː/", "type": "consonant", "sample": "Quiet / Qadam"},
        "R": {"ipa": "/ɑːr/", "type": "consonant", "sample": "Reason / Rasta"},
        "S": {"ipa": "/ɛs/", "type": "consonant", "sample": "Strategy / Sach"},
        "T": {"ipa": "/tiː/", "type": "consonant", "sample": "Trust / Tapsya"},
        "U": {"ipa": "/juː/", "type": "vowel", "sample": "Unity / Ummeed"},
        "V": {"ipa": "/viː/", "type": "consonant", "sample": "Valence / Vishwas"},
        "W": {"ipa": "/ˈdʌbəl.juː/", "type": "consonant", "sample": "Wisdom / Waqt"},
        "X": {"ipa": "/ɛks/", "type": "consonant", "sample": "X-ray / X-factor"},
        "Y": {"ipa": "/waɪ/", "type": "consonant", "sample": "Yield / Yaari"},
        "Z": {"ipa": "/zɛd/", "type": "consonant", "sample": "Zero / Zindagi"},
    }

    # Hindi Devanagari Vowels (स्वर)
    HINDI_SWAR: Dict[str, Dict[str, str]] = {
        "अ": {"translit": "a", "ipa": "/ə/", "sample": "अमन (Aman / Peace)"},
        "आ": {"translit": "aa", "ipa": "/aː/", "sample": "आशा (Asha / Hope)"},
        "इ": {"translit": "i", "ipa": "/ɪ/", "sample": "इरादा (Irada / Intent)"},
        "ई": {"translit": "ee", "ipa": "/iː/", "sample": "ईमानदारी (Honesty)"},
        "उ": {"translit": "u", "ipa": "/ʊ/", "sample": "उम्मीद (Ummeed / Hope)"},
        "ऊ": {"translit": "oo", "ipa": "/uː/", "sample": "ऊर्जा (Oorja / Energy)"},
        "ऋ": {"translit": "ri", "ipa": "/rɪ/", "sample": "ऋषि (Sage)"},
        "ए": {"translit": "e", "ipa": "/eː/", "sample": "एकता (Ekta / Unity)"},
        "ऐ": {"translit": "ai", "ipa": "/ɛː/", "sample": "ऐनक (Spectacles)"},
        "ओ": {"translit": "o", "ipa": "/oː/", "sample": "ओज (Ojas / Radiance)"},
        "औ": {"translit": "au", "ipa": "/ɔː/", "sample": "औजार (Auzar / Tool)"},
        "अं": {"translit": "am", "ipa": "/əŋ/", "sample": "अंकुर (Sprout)"},
        "अः": {"translit": "ah", "ipa": "/əh/", "sample": "अतः (Therefore)"},
    }

    # Hindi Devanagari Consonants (व्यंजन)
    HINDI_VYANJAN: Dict[str, Dict[str, str]] = {
        # कण्ठ्य (Velar)
        "क": {"translit": "ka", "ipa": "/k/", "sample": "कर्म (Karma / Action)"},
        "ख": {"translit": "kha", "ipa": "/kʰ/", "sample": "ख्याल (Khyaal / Care)"},
        "ग": {"translit": "ga", "ipa": "/ɡ/", "sample": "ज्ञान (Gyan / Wisdom)"},
        "घ": {"translit": "gha", "ipa": "/ɡʱ/", "sample": "घर (Ghar / Home)"},
        "ङ": {"translit": "nga", "ipa": "/ŋ/", "sample": "गंगा (Ganga)"},

        # तालव्य (Palatal)
        "च": {"translit": "cha", "ipa": "/tʃ/", "sample": "चिंता (Chinta / Worry)"},
        "छ": {"translit": "chha", "ipa": "/tʃʰ/", "sample": "छाया (Chhaya / Shadow)"},
        "ज": {"translit": "ja", "ipa": "/dʒ/", "sample": "जोश (Josh / Passion)"},
        "झ": {"translit": "jha", "ipa": "/dʒʱ/", "sample": "झलक (Jhalak / Glimpse)"},
        "ञ": {"translit": "nya", "ipa": "/ɲ/", "sample": "ज्ञान"},

        # मूर्धन्य (Retroflex)
        "ट": {"translit": "ta", "ipa": "/ʈ/", "sample": "टारगेट (Target)"},
        "ठ": {"translit": "tha", "ipa": "/ʈʰ/", "sample": "ठहराव (Calmness/Pause)"},
        "ड": {"translit": "da", "ipa": "/ɖ/", "sample": "डर (Dar / Fear)"},
        "ढ": {"translit": "dha", "ipa": "/ɖʱ/", "sample": "ढाल (Dhaal / Shield)"},
        "ण": {"translit": "na", "ipa": "/ɳ/", "sample": "प्रणाम (Salutation)"},

        # दन्त्य (Dental)
        "त": {"translit": "ta", "ipa": "/t̪/", "sample": "तर्क (Tark / Logic)"},
        "थ": {"translit": "tha", "ipa": "/t̪ʰ/", "sample": "थकान (Thakaan / Fatigue)"},
        "द": {"translit": "da", "ipa": "/d̪/", "sample": "दोस्त (Dost / Friend)"},
        "ध": {"translit": "dha", "ipa": "/d̪ʱ/", "sample": "धैर्य (Dhairya / Patience)"},
        "न": {"translit": "na", "ipa": "/n/", "sample": "निष्ठा (Nishtha / Loyalty)"},

        # ओष्ठ्य (Labial)
        "प": {"translit": "pa", "ipa": "/p/", "sample": "पाइजेन (PaiGen)"},
        "फ": {"translit": "pha", "ipa": "/pʰ/", "sample": "फैसला (Decision)"},
        "ब": {"translit": "ba", "ipa": "/b/", "sample": "बंधु (Bandhu / Brother)"},
        "भ": {"translit": "bha", "ipa": "/bʱ/", "sample": "भरोसा (Bharosa / Trust)"},
        "म": {"translit": "ma", "ipa": "/m/", "sample": "मित्र (Mitra / Friend)"},

        # अन्तःस्थ (Semivowels)
        "य": {"translit": "ya", "ipa": "/j/", "sample": "याद (Yaad / Memory)"},
        "र": {"translit": "ra", "ipa": "/r/", "sample": "रक्षा (Raksha / Protection)"},
        "ल": {"translit": "la", "ipa": "/l/", "sample": "लक्ष्य (Lakshya / Goal)"},
        "व": {"translit": "va", "ipa": "/ʋ/", "sample": "विचार (Vichar / Thought)"},

        # ऊष्म (Sibilants & Fricatives)
        "श": {"translit": "sha", "ipa": "/ʃ/", "sample": "शांति (Shanti / Peace)"},
        "ष": {"translit": "sha", "ipa": "/ʂ/", "sample": "श्रेष्ठ (Shreshtha / Best)"},
        "स": {"translit": "sa", "ipa": "/s/", "sample": "साथी (Saathi / Companion)"},
        "ह": {"translit": "ha", "ipa": "/ɦ/", "sample": "हौसला (Hausla / Courage)"},
    }

    # Common Hinglish Phonetic Pronunciations
    HINGLISH_MAPPINGS: Dict[str, str] = {
        "namaste": "नमस्ते (Formal greeting / Respect)",
        "dost": "दोस्त (Friend / Companion)",
        "bhai": "भाई (Brother / Trusted partner)",
        "kaise ho": "कैसे हो? (How are you?)",
        "sab theek": "सब ठीक (All is well / Grounded)",
        "shabash": "शाबाश (Well done / Positive reinforcement)",
        "chinta mat karo": "चिंता मत करो (Do not worry / Calming)",
        "focus karo": "फोकस करो (Focus / Tactical directive)",
        "vishwas": "विश्वास (Trust / High bond state)",
        "thak gaye ho": "थक गए हो? (Are you fatigued? / Empathy trigger)",
    }

    @classmethod
    def get_summary(cls) -> Dict[str, int]:
        return {
            "english_letters": len(cls.ENGLISH_ALPHABETS),
            "hindi_swar": len(cls.HINDI_SWAR),
            "hindi_vyanjan": len(cls.HINDI_VYANJAN),
            "hinglish_primitives": len(cls.HINGLISH_MAPPINGS),
        }
