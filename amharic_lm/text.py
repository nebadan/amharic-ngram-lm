"""Amharic text preprocessing: normalization, sentence splitting, tokenization."""
import re
import unicodedata

# Characters that sound the same in modern Amharic are written with several
# letters (ሀ/ሐ/ኀ, ሰ/ሠ, አ/ዐ, ጸ/ፀ). Mapping them to one form shrinks the
# vocabulary and merges spelling variants of the same word.
_FAMILIES = [
    ("ሐ", "ሀ"), ("ኀ", "ሀ"), ("ሠ", "ሰ"), ("ዐ", "አ"), ("ፀ", "ጸ"),
]
_CHAR_MAP = {}
for src, dst in _FAMILIES:
    # each Ethiopic consonant has 7 vowel orders laid out consecutively
    for i in range(7):
        _CHAR_MAP[chr(ord(src) + i)] = chr(ord(dst) + i)
# the 1st and 4th orders of the glottal / pharyngeal series are interchangeable
_CHAR_MAP.update({"ኣ": "አ", "ዓ": "አ", "ሃ": "ሀ", "ሓ": "ሀ", "ኃ": "ሀ"})
# labialized variants commonly written as two characters
_CHAR_MAP.update({"ቍ": "ቁ", "ኵ": "ኩ", "ጕ": "ጉ", "ዃ": "ኳ"})
_TRANS = str.maketrans(_CHAR_MAP)

SENT_END = re.compile(r"(?:፡፡|[።፧!?፨])+|\n")
# Ethiopic word-space and punctuation that separate tokens
_PUNCT = re.compile(r"[፡፣፤፥፦፠\"'“”‘’«»()\[\]{}<>.,;:/\\|_*#@&%^~`+=–—\-]+")
_NUM = re.compile(r"[0-9፩-፼]+(?:[.,/:][0-9፩-፼]+)*")
# dots / slashes inside Ethiopic abbreviations: ዓ.ም -> ዓም, ዶ/ር -> ዶር, እ.ኤ.አ -> እኤአ
_ABBR = re.compile(r"(?<=[\u1200-\u137F])[./](?=[\u1200-\u137F])")
# keep Ethiopic letters only (U+1200–U+137F incl. supplements)
_NON_ETHIOPIC = re.compile(r"[^ሀ-ፚᎀ-᎟ⶀ-⷟꬀-꬯\s]+")
_WS = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    return text.translate(_TRANS)


def tokenize(sentence: str):
    """Tokenize one sentence into Amharic words (numbers -> <NUM>, Latin dropped)."""
    s = _ABBR.sub("", sentence)
    s = _NUM.sub(" NUMTOKEN ", s)
    s = _PUNCT.sub(" ", s)
    toks = []
    for t in _WS.split(s):
        if not t:
            continue
        if t == "NUMTOKEN":
            if not toks or toks[-1] != "<NUM>":   # a run of numbers counts once
                toks.append("<NUM>")
            continue
        t = _NON_ETHIOPIC.sub("", t)
        if t:
            toks.append(t)
    return toks


def sentences(text: str, norm: bool = True):
    """Split raw text into tokenized sentences."""
    if norm:
        text = normalize(text)
    for raw in SENT_END.split(text):
        toks = tokenize(raw)
        if toks:
            yield toks


# Curated Amharic function words (pronouns, conjunctions, postpositions,
# copulas, demonstratives, common auxiliaries). Stored normalized.
STOPWORDS = set(normalize(w) for w in """
እና ና ወይም ግን ነገር ግን ስለዚህ ስለሆነም ስለ ደግሞ ም እንጂ ቢሆንም ይሁን እንጂ ሆኖም
ነው ናቸው ነበር ነበሩ ነኝ ነህ ነሽ ነን ናችሁ ናት ሆነ ሆኖ ሆነው ይሆናል ይሆን ሊሆን ነበረች
አለ አሉ አለች አለው አላቸው አለበት አለባቸው ያለ ያሉ የለም ያለው ያላቸው የነበረ የነበሩ የነበረው
ይህ ይህን ይህም ይህንን ይኸው ያ ያን ያንን እነዚህ እነዚያ እዚህ እዚያ ይሄ ይች ይቺ
እሱ እሷ እነሱ እኔ እኛ አንተ አንቺ እናንተ እርሱ እርሷ እርሳቸው እሳቸው እሱን እሷን እነሱን
ላይ ውስጥ ወደ ከ በ ለ የ እስከ ድረስ ጋር ጋራ ዘንድ በኩል አጠገብ ስር በላይ በታች ፊት ኋላ
እንደ እንዲሁም እንዲሁ ብቻ ሁሉ ሁሉም ማንኛውም ሌላ ሌሎች ሌሎችም ብዙ በጣም ጥቂት አንዳንድ
ምን ማን የት መቼ እንዴት ለምን ስንት የትኛው ምንድን ምንድነው
አንድ ሁለት ሦስት ሶስት አራት
ወዘተ ደግሞም ማለት ማለትም ነገሮች ነገር ጊዜ ዓመት አመት
በዚህ በዚያ ከዚህ ከዚያ ለዚህ የዚህ ይህም ይኸውም ስለዚሁ
መሆኑን መሆኑ መሆን መሆናቸውን ሲሆን ሲሆኑ ቢሆን እንዲሆን
እንደሆነ እንደሚገኝ እንደገለጹት ገልጸዋል ተናግረዋል አስታውቀዋል ብለዋል አመልክተዋል
ውስጥም ላይም ጋርም በተጨማሪ ከዚህም ከዚያም በኋላ በፊት ከዚህ በፊት
<NUM>
""".split())
