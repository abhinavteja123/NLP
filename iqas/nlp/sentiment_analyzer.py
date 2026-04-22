"""
IQAS Sentiment & Emotion Analyzer
===================================
Sentence-level sentiment scoring, word-level polarity heatmaps,
6-emotion breakdown, and entity-level sentiment tracking.

NLP Techniques:
    - Sentence tokenization (spaCy)
    - Lemmatization (spaCy)
    - POS-aware weighting (adjectives/adverbs weighted higher)
    - Lexicon-based sentiment scoring (built-in polarity lexicon)
    - Emotion keyword classification (Joy, Anger, Sadness, Fear, Surprise, Trust)
    - Named Entity Recognition for entity-level sentiment
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

from nlp.tokenizer import NLPTokenizer
from nlp.ner import NERExtractor
from utils.logger import get_logger

log = get_logger("sentiment_analyzer")


# ──────────────────────────── Data Models ────────────────────────────


@dataclass
class WordSentiment:
    """Sentiment data for a single word."""
    word: str
    lemma: str
    pos: str
    polarity: float       # -1.0 to 1.0
    is_negated: bool


@dataclass
class SentenceSentiment:
    """Sentiment data for a sentence."""
    text: str
    index: int
    polarity: float       # -1.0 to 1.0
    subjectivity: float   # 0.0 to 1.0
    word_count: int
    emotion: str          # Dominant emotion


@dataclass
class EmotionBreakdown:
    """Emotion category scores."""
    joy: float = 0.0
    anger: float = 0.0
    sadness: float = 0.0
    fear: float = 0.0
    surprise: float = 0.0
    trust: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "Joy": self.joy,
            "Anger": self.anger,
            "Sadness": self.sadness,
            "Fear": self.fear,
            "Surprise": self.surprise,
            "Trust": self.trust,
        }

    @property
    def dominant(self) -> str:
        d = self.to_dict()
        return max(d, key=d.get) if any(v > 0 for v in d.values()) else "Neutral"


@dataclass
class EntitySentiment:
    """Sentiment associated with a named entity."""
    entity: str
    entity_type: str
    avg_polarity: float
    mention_count: int
    label: str             # Positive / Negative / Neutral


@dataclass
class SentimentResult:
    """Complete sentiment analysis result."""
    overall_polarity: float
    overall_subjectivity: float
    overall_label: str
    sentence_sentiments: List[SentenceSentiment]
    word_sentiments: List[WordSentiment]
    emotion_breakdown: EmotionBreakdown
    entity_sentiments: List[EntitySentiment]
    most_positive_sentence: str
    most_negative_sentence: str


# ──────────────────────────── Sentiment Lexicon ────────────────────────────

# Compact polarity lexicon (lemma → polarity score)
POLARITY_LEXICON = {
    # Strong positive (+0.7 to +1.0)
    "excellent": 0.95, "outstanding": 0.95, "amazing": 0.9, "brilliant": 0.9,
    "fantastic": 0.9, "wonderful": 0.9, "superb": 0.9, "exceptional": 0.9,
    "magnificent": 0.85, "perfect": 0.85, "remarkable": 0.85, "extraordinary": 0.85,
    "love": 0.8, "beautiful": 0.8, "incredible": 0.8, "awesome": 0.8,
    "impressive": 0.75, "delightful": 0.75, "terrific": 0.75, "marvelous": 0.75,

    # Moderate positive (+0.3 to +0.7)
    "good": 0.6, "great": 0.65, "nice": 0.55, "fine": 0.45, "happy": 0.65,
    "glad": 0.55, "pleased": 0.6, "enjoy": 0.6, "like": 0.5, "well": 0.45,
    "better": 0.55, "best": 0.7, "useful": 0.5, "helpful": 0.55, "important": 0.45,
    "interesting": 0.5, "effective": 0.55, "efficient": 0.5, "successful": 0.6,
    "powerful": 0.55, "innovative": 0.6, "improve": 0.5, "advantage": 0.55,
    "benefit": 0.55, "positive": 0.5, "strong": 0.45, "clear": 0.4,
    "easy": 0.4, "fast": 0.35, "accurate": 0.5, "reliable": 0.5,
    "popular": 0.4, "progress": 0.45, "advance": 0.5, "achieve": 0.55,
    "enhance": 0.5, "gain": 0.4, "win": 0.55, "elegant": 0.6,
    "smart": 0.5, "capable": 0.45, "robust": 0.45, "valuable": 0.5,

    # Mild positive (+0.1 to +0.3)
    "adequate": 0.2, "okay": 0.15, "acceptable": 0.2, "reasonable": 0.25,
    "satisfactory": 0.25, "sufficient": 0.2, "possible": 0.15, "available": 0.1,
    "new": 0.15, "modern": 0.2, "significant": 0.3, "notable": 0.3,

    # Mild negative (-0.1 to -0.3)
    "difficult": -0.3, "hard": -0.2, "complex": -0.15, "slow": -0.25,
    "lack": -0.3, "miss": -0.25, "issue": -0.2, "problem": -0.3,
    "limit": -0.2, "limitation": -0.25, "challenge": -0.15, "concern": -0.2,
    "confuse": -0.25, "unclear": -0.25, "weak": -0.3, "poor": -0.5,

    # Moderate negative (-0.3 to -0.7)
    "bad": -0.6, "wrong": -0.55, "fail": -0.6, "failure": -0.65,
    "error": -0.5, "mistake": -0.5, "loss": -0.5, "lose": -0.55,
    "damage": -0.6, "harm": -0.6, "risk": -0.35, "danger": -0.55,
    "hate": -0.8, "terrible": -0.8, "horrible": -0.8, "awful": -0.75,
    "worst": -0.7, "worse": -0.55, "negative": -0.45, "ugly": -0.6,
    "boring": -0.5, "annoy": -0.5, "frustrate": -0.55, "disappoint": -0.55,
    "reject": -0.5, "deny": -0.4, "refuse": -0.4, "break": -0.35,
    "destroy": -0.7, "eliminate": -0.35, "remove": -0.15, "crash": -0.6,

    # Strong negative (-0.7 to -1.0)
    "disaster": -0.9, "catastrophe": -0.9, "devastating": -0.85, "dreadful": -0.8,
    "atrocious": -0.9, "abysmal": -0.85, "miserable": -0.75, "pathetic": -0.7,
    "toxic": -0.75, "corrupt": -0.7, "vile": -0.8, "wretched": -0.75,
}

# Emotion keyword lexicon (lemma → emotion category)
EMOTION_LEXICON = {
    # Joy
    "happy": "joy", "joy": "joy", "love": "joy", "delight": "joy",
    "cheerful": "joy", "glad": "joy", "pleased": "joy", "enjoy": "joy",
    "wonderful": "joy", "beautiful": "joy", "fantastic": "joy", "amazing": "joy",
    "brilliant": "joy", "excellent": "joy", "fun": "joy", "celebrate": "joy",
    "laugh": "joy", "smile": "joy", "paradise": "joy", "bliss": "joy",
    "thrill": "joy", "excite": "joy", "enthusiasm": "joy", "optimistic": "joy",

    # Anger
    "angry": "anger", "rage": "anger", "fury": "anger", "hate": "anger",
    "annoy": "anger", "frustrate": "anger", "irritate": "anger", "hostile": "anger",
    "resent": "anger", "outrage": "anger", "bitter": "anger", "aggressive": "anger",
    "violent": "anger", "destroy": "anger", "attack": "anger", "fight": "anger",
    "conflict": "anger", "oppose": "anger", "rebel": "anger", "protest": "anger",

    # Sadness
    "sad": "sadness", "sorrow": "sadness", "grief": "sadness", "mourn": "sadness",
    "depress": "sadness", "miserable": "sadness", "unhappy": "sadness", "lonely": "sadness",
    "despair": "sadness", "regret": "sadness", "disappoint": "sadness", "suffer": "sadness",
    "pain": "sadness", "cry": "sadness", "tragic": "sadness", "loss": "sadness",
    "miss": "sadness", "abandon": "sadness", "hopeless": "sadness", "gloomy": "sadness",

    # Fear
    "fear": "fear", "afraid": "fear", "scare": "fear", "terrify": "fear",
    "panic": "fear", "dread": "fear", "anxiety": "fear", "anxious": "fear",
    "worry": "fear", "nervous": "fear", "horror": "fear", "threat": "fear",
    "danger": "fear", "risk": "fear", "alarm": "fear", "phobia": "fear",
    "frighten": "fear", "terror": "fear", "nightmare": "fear", "haunted": "fear",

    # Surprise
    "surprise": "surprise", "shock": "surprise", "astonish": "surprise", "amaze": "surprise",
    "stun": "surprise", "unexpected": "surprise", "sudden": "surprise", "remarkable": "surprise",
    "incredible": "surprise", "unbelievable": "surprise", "extraordinary": "surprise",
    "wonder": "surprise", "awe": "surprise", "startled": "surprise", "discover": "surprise",

    # Trust
    "trust": "trust", "believe": "trust", "faith": "trust", "reliable": "trust",
    "honest": "trust", "loyal": "trust", "confident": "trust", "safe": "trust",
    "secure": "trust", "stable": "trust", "depend": "trust", "assure": "trust",
    "genuine": "trust", "authentic": "trust", "integrity": "trust", "worthy": "trust",
    "respect": "trust", "responsible": "trust", "consistent": "trust", "accurate": "trust",
}

# Negation words
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
    "nor", "none", "n't", "nt", "cannot", "hardly", "barely", "scarcely",
    "seldom", "rarely", "without", "lack", "absence",
}

# POS weights — adjectives/adverbs carry more sentiment
POS_WEIGHTS = {
    "ADJ": 1.5,    # Adjectives are strong sentiment carriers
    "ADV": 1.3,    # Adverbs modify intensity
    "VERB": 1.0,   # Verbs carry action sentiment
    "NOUN": 0.8,   # Nouns carry less sentiment
}


# ──────────────────────────── Analyzer ────────────────────────────


class SentimentAnalyzer:
    """
    Sentiment and emotion analyzer using lexicon-based scoring with spaCy NLP.

    Features:
        - Overall document polarity and subjectivity
        - Per-sentence sentiment arc
        - Per-word sentiment coloring
        - 6-emotion category breakdown
        - Entity-level sentiment tracking
    """

    def __init__(self):
        """Initialize with shared spaCy model."""
        self._tokenizer = NLPTokenizer()
        self._ner = NERExtractor()
        self.nlp = self._tokenizer.nlp

    def analyze(self, text: str) -> SentimentResult:
        """
        Full sentiment analysis of a text.

        Args:
            text: Input text.

        Returns:
            SentimentResult with all analysis dimensions.
        """
        # Per-sentence analysis
        sentence_sentiments = self.sentence_sentiments(text)

        # Per-word analysis
        word_sentiments = self.word_sentiments(text)

        # Emotion breakdown
        emotion = self.emotion_breakdown(text)

        # Entity sentiments
        entity_sents = self.entity_sentiments(text)

        # Overall scores
        if sentence_sentiments:
            overall_polarity = sum(s.polarity for s in sentence_sentiments) / len(sentence_sentiments)
            subjective_words = sum(1 for w in word_sentiments if abs(w.polarity) > 0.1)
            overall_subjectivity = subjective_words / max(len(word_sentiments), 1)
        else:
            overall_polarity = 0.0
            overall_subjectivity = 0.0

        # Label
        if overall_polarity > 0.15:
            overall_label = "Positive"
        elif overall_polarity < -0.15:
            overall_label = "Negative"
        else:
            overall_label = "Neutral"

        # Most positive/negative sentences
        if sentence_sentiments:
            most_pos = max(sentence_sentiments, key=lambda s: s.polarity)
            most_neg = min(sentence_sentiments, key=lambda s: s.polarity)
        else:
            most_pos = SentenceSentiment("", 0, 0, 0, 0, "Neutral")
            most_neg = SentenceSentiment("", 0, 0, 0, 0, "Neutral")

        return SentimentResult(
            overall_polarity=round(overall_polarity, 4),
            overall_subjectivity=round(overall_subjectivity, 4),
            overall_label=overall_label,
            sentence_sentiments=sentence_sentiments,
            word_sentiments=word_sentiments,
            emotion_breakdown=emotion,
            entity_sentiments=entity_sents,
            most_positive_sentence=most_pos.text,
            most_negative_sentence=most_neg.text,
        )

    def sentence_sentiments(self, text: str) -> List[SentenceSentiment]:
        """
        Compute per-sentence sentiment scores.

        Returns:
            List of SentenceSentiment for each sentence.
        """
        sentences = self._tokenizer.sent_tokenize(text)
        results = []

        for i, sent in enumerate(sentences):
            doc = self.nlp(sent)
            polarity = 0.0
            subj_count = 0
            word_count = 0
            is_negated = False

            for token in doc:
                if token.is_space or token.is_punct:
                    continue
                word_count += 1

                lemma = token.lemma_.lower()

                # Check negation
                if lemma in NEGATION_WORDS or token.dep_ == "neg":
                    is_negated = True
                    continue

                # Look up polarity
                score = POLARITY_LEXICON.get(lemma, 0.0)

                if score != 0.0:
                    # Apply POS weight
                    weight = POS_WEIGHTS.get(token.pos_, 0.7)
                    score *= weight

                    # Apply negation
                    if is_negated:
                        score *= -0.8
                        is_negated = False

                    polarity += score
                    subj_count += 1
                else:
                    # Reset negation after non-sentiment word
                    if is_negated and token.pos_ not in ("ADV", "DET", "ADP"):
                        is_negated = False

            # Normalize
            if word_count > 0:
                avg_polarity = polarity / word_count
            else:
                avg_polarity = 0.0

            subjectivity = subj_count / max(word_count, 1)

            # Determine dominant emotion for this sentence
            sent_emotion = self._sentence_emotion(sent)

            results.append(SentenceSentiment(
                text=sent,
                index=i,
                polarity=round(max(-1.0, min(1.0, avg_polarity)), 4),
                subjectivity=round(subjectivity, 4),
                word_count=word_count,
                emotion=sent_emotion,
            ))

        return results

    def word_sentiments(self, text: str) -> List[WordSentiment]:
        """
        Compute per-word sentiment polarity.

        Returns:
            List of WordSentiment for each non-space token.
        """
        doc = self.nlp(text)
        results = []
        is_negated = False

        for token in doc:
            if token.is_space:
                continue

            lemma = token.lemma_.lower()

            # Check negation
            if lemma in NEGATION_WORDS or token.dep_ == "neg":
                is_negated = True
                results.append(WordSentiment(
                    word=token.text, lemma=lemma, pos=token.pos_,
                    polarity=0.0, is_negated=False,
                ))
                continue

            score = POLARITY_LEXICON.get(lemma, 0.0)
            if score != 0.0:
                weight = POS_WEIGHTS.get(token.pos_, 0.7)
                score *= weight
                if is_negated:
                    score *= -0.8
                    is_negated = False

            results.append(WordSentiment(
                word=token.text, lemma=lemma, pos=token.pos_,
                polarity=round(max(-1.0, min(1.0, score)), 4),
                is_negated=is_negated if score == 0 else False,
            ))

            if token.is_punct or (is_negated and token.pos_ not in ("ADV", "DET", "ADP")):
                is_negated = False

        return results

    def emotion_breakdown(self, text: str) -> EmotionBreakdown:
        """
        Classify text into 6 emotion categories.

        Returns:
            EmotionBreakdown with normalized scores.
        """
        doc = self.nlp(text)
        counts = {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0, "trust": 0}
        total = 0

        for token in doc:
            if token.is_space or token.is_punct:
                continue
            lemma = token.lemma_.lower()
            emotion = EMOTION_LEXICON.get(lemma)
            if emotion:
                weight = POS_WEIGHTS.get(token.pos_, 0.7)
                counts[emotion] += weight
                total += weight

        # Normalize to proportions
        if total > 0:
            for key in counts:
                counts[key] = round(counts[key] / total, 4)

        return EmotionBreakdown(**counts)

    def entity_sentiments(self, text: str) -> List[EntitySentiment]:
        """
        Track sentiment around named entities.

        For each entity, compute average polarity of the sentence(s)
        where the entity appears.

        Returns:
            List of EntitySentiment objects.
        """
        entities = self._ner.extract(text)
        sentence_sents = self.sentence_sentiments(text)
        sentences = self._tokenizer.sent_tokenize(text)

        if not entities or not sentence_sents:
            return []

        # Map entity → sentence polarities
        entity_scores: Dict[str, Dict] = {}

        for ent in entities:
            key = ent.text.lower()
            if key not in entity_scores:
                entity_scores[key] = {
                    "entity": ent.text,
                    "type": ent.label,
                    "polarities": [],
                }

            # Find which sentence contains this entity
            for i, sent in enumerate(sentences):
                if ent.text in sent and i < len(sentence_sents):
                    entity_scores[key]["polarities"].append(sentence_sents[i].polarity)

        # Compute averages
        results = []
        for key, data in entity_scores.items():
            if data["polarities"]:
                avg = sum(data["polarities"]) / len(data["polarities"])
            else:
                avg = 0.0

            if avg > 0.1:
                label = "Positive"
            elif avg < -0.1:
                label = "Negative"
            else:
                label = "Neutral"

            results.append(EntitySentiment(
                entity=data["entity"],
                entity_type=data["type"],
                avg_polarity=round(avg, 4),
                mention_count=len(data["polarities"]),
                label=label,
            ))

        # Sort by absolute polarity
        results.sort(key=lambda e: abs(e.avg_polarity), reverse=True)
        return results

    def _sentence_emotion(self, sentence: str) -> str:
        """Get the dominant emotion for a single sentence."""
        doc = self.nlp(sentence)
        counts = {"joy": 0, "anger": 0, "sadness": 0, "fear": 0, "surprise": 0, "trust": 0}

        for token in doc:
            lemma = token.lemma_.lower()
            emotion = EMOTION_LEXICON.get(lemma)
            if emotion:
                counts[emotion] += 1

        if any(v > 0 for v in counts.values()):
            return max(counts, key=counts.get).capitalize()
        return "Neutral"

    def render_word_heatmap_html(self, text: str) -> str:
        """
        Render text as HTML with each word colored by sentiment.

        Red = negative, Gray = neutral, Green = positive.

        Returns:
            HTML string.
        """
        word_sents = self.word_sentiments(text)
        html_parts = []

        for ws in word_sents:
            if ws.word.strip() == "":
                html_parts.append(" ")
                continue

            polarity = ws.polarity
            if polarity > 0.1:
                # Green shades
                intensity = min(polarity, 1.0)
                r, g, b = 46, int(140 + 64 * intensity), int(80 + 33 * intensity)
                bg_alpha = 0.15 + 0.25 * intensity
            elif polarity < -0.1:
                # Red shades
                intensity = min(abs(polarity), 1.0)
                r, g, b = int(200 + 31 * intensity), int(60 + 16 * intensity), int(60 + 16 * intensity)
                bg_alpha = 0.15 + 0.25 * intensity
            else:
                r, g, b = 200, 200, 200
                bg_alpha = 0.0

            if bg_alpha > 0:
                style = (
                    f"color: rgb({r},{g},{b}); "
                    f"background: rgba({r},{g},{b},{bg_alpha}); "
                    f"padding: 2px 4px; border-radius: 3px; "
                    f"font-weight: {'600' if abs(polarity) > 0.3 else '400'};"
                )
            else:
                style = f"color: rgb({r},{g},{b});"

            html_parts.append(f'<span style="{style}">{ws.word}</span>')

        return (
            '<div style="line-height: 2.0; font-size: 1rem; '
            'background: #1A1D29; padding: 20px; border-radius: 12px;">'
            + " ".join(html_parts)
            + "</div>"
        )
