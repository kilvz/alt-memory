"""AAAK compression dialect for Alt Memory — Agent-Adaptive Abbreviation/Kinesis.

Compresses text to compact symbolic format using entity codes, emotion markers,
pipe-delimited fields, and structured metadata. Decompresses back to readable form.
"""

import re
import string
from pathlib import Path

# ── Constants ────────────────────────────────────────────────────────────────

_STOPWORDS = frozenset({
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'by', 'from', 'as', 'is', 'are', 'was', 'were', 'be',
    'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
    'would', 'could', 'should', 'may', 'might', 'shall', 'can', 'need',
    'it', 'its', 'this', 'that', 'these', 'those', 'i', 'me', 'my', 'we',
    'us', 'our', 'you', 'your', 'he', 'him', 'his', 'she', 'her', 'they',
    'them', 'their', 'what', 'which', 'who', 'whom', 'whose', 'when',
    'where', 'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more',
    'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
    'same', 'so', 'than', 'too', 'very', 'just', 'because', 'until',
    'while', 'about', 'between', 'through', 'during', 'before', 'after',
    'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again',
    'further', 'once', 'here', 'there', 'then', 'else', 'also', 'well',
    'really', 'actually', 'basically', 'literally', 'probably', 'maybe',
    'perhaps', 'quite', 'rather', 'somewhat', 'slightly', 'ago', 'yet',
    'already', 'still', 'ever', 'never', 'always', 'often', 'sometimes',
    'usually', 'generally', 'overall', 'thus', 'hence', 'thereby',
})

_PERSON_TITLES = frozenset({
    'mr', 'ms', 'mrs', 'mx', 'dr', 'prof', 'professor', 'sir', 'dame',
    'lord', 'lady', 'fr', 'st', 'president', 'captain', 'chief', 'sgt',
})

_PERSON_VERBS = frozenset({
    'with', 'and', 'by', 'met', 'meet', 'called', 'call', 'told', 'tell',
    'asked', 'ask', 'saw', 'see', 'helped', 'help', 'joined', 'join',
    'contacted', 'contact', 'spoke', 'speak', 'talked', 'talk',
    'introduced', 'introduce', 'invited', 'invite', 'thanked', 'thank',
})

_TECH_NAMES = frozenset({
    'python', 'javascript', 'typescript', 'rust', 'golang', 'java',
    'c++', 'c#', 'react', 'vue', 'angular', 'svelte', 'django',
    'flask', 'fastapi', 'docker', 'kubernetes', 'aws', 'gcp', 'azure',
    'git', 'linux', 'postgresql', 'mysql', 'mongodb', 'redis',
    'sqlite', 'faiss', 'numpy', 'pytorch', 'tensorflow', 'scipy',
    'pandas', 'opencode', 'claude', 'chatgpt', 'llama',
    'mistral', 'gemini', 'copilot', 'figma', 'jira', 'notion',
    'vscode', 'neovim', 'emacs', 'webpack', 'vite', 'babel',
    'deno', 'bun', 'npm', 'yarn', 'pnpm', 'pip', 'conda',
    'terraform', 'ansible', 'jenkins', 'github',
    'gitlab', 'bitbucket', 'jupyter', 'keras', 'opencv', 'nginx',
    'apache', 'caddy', 'traefik', 'kafka', 'rabbitmq', 'celery',
    'graphql', 'rest', 'grpc', 'websocket', 'oauth', 'jwt',
})

_ORGANIZATION_HINTS = frozenset({
    'inc', 'corp', 'corporation', 'llc', 'ltd', 'limited', 'co',
    'company', 'gmbh', 'sa', 'ag', 'plc', 'llp', 'pte', 'pty',
})

_LOCATION_PREPOSITIONS = frozenset({
    'in', 'at', 'to', 'from', 'via', 'near', 'across', 'into',
    'within', 'outside', 'inside', 'around', 'towards', 'toward',
    'onto', 'upon',
})

_EVENT_KEYWORDS = frozenset({
    'conference', 'summit', 'meetup', 'workshop', 'hackathon',
    'symposium', 'seminar', 'webinar', 'retreat', 'convention',
    'expo', 'forum', 'panel', 'keynote', 'ceremony', 'gala',
    'celebration', 'festival', 'competition', 'tournament',
})

_ACRONYM_ORGS = frozenset({
    'ibm', 'bmw', 'nasa', 'nsa', 'fbi', 'cia', 'un', 'nato', 'who',
    'ios', 'ieee', 'acm', 'mit', 'caltech', 'ucla',
})

_KNOWN_LOCATIONS = frozenset({
    'london', 'paris', 'berlin', 'tokyo', 'beijing', 'moscow',
    'new york', 'san francisco', 'seattle', 'austin', 'boston',
    'chicago', 'los angeles', 'sydney', 'melbourne', 'toronto',
    'vancouver', 'amsterdam', 'zurich', 'singapore', 'hong kong',
    'dubai', 'mumbai', 'bangkok', 'seoul', 'shanghai',
})

# ── AAAK Emotion Signals ──────────────────────────────────────────────────────

EMOTION_SIGNALS: dict[str, str] = {
    'decided': 'determ',
    'prefer': 'convict',
    'worried': 'anx',
    'excited': 'excite',
    'frustrated': 'frust',
    'confused': 'confuse',
    'love': 'love',
    'hate': 'rage',
    'hope': 'hope',
    'fear': 'fear',
    'trust': 'trust',
    'happy': 'joy',
    'sad': 'grief',
    'surprised': 'surprise',
    'grateful': 'grat',
    'curious': 'curious',
    'wonder': 'wonder',
    'anxious': 'anx',
    'relieved': 'relief',
    'satisf': 'satis',
    'disappoint': 'grief',
    'concern': 'anx',
    'gratitude': 'grat',
    'excitement': 'excite',
    'frustration': 'frust',
    'confusion': 'confuse',
    'curiosity': 'curious',
    'determination': 'determ',
    'conviction': 'convict',
}

# ── AAAK Flag Signals ─────────────────────────────────────────────────────────

FLAG_SIGNALS: dict[str, str] = {
    'decided': 'DECISION',
    'chose': 'DECISION',
    'switched': 'DECISION',
    'migrated': 'DECISION',
    'replaced': 'DECISION',
    'instead of': 'DECISION',
    'because': 'DECISION',
    'founded': 'ORIGIN',
    'created': 'ORIGIN',
    'started': 'ORIGIN',
    'born': 'ORIGIN',
    'launched': 'ORIGIN',
    'first time': 'ORIGIN',
    'core': 'CORE',
    'fundamental': 'CORE',
    'essential': 'CORE',
    'principle': 'CORE',
    'belief': 'CORE',
    'always': 'CORE',
    'never forget': 'CORE',
    'turning point': 'PIVOT',
    'changed everything': 'PIVOT',
    'realized': 'PIVOT',
    'breakthrough': 'PIVOT',
    'epiphany': 'PIVOT',
    'api': 'TECHNICAL',
    'database': 'TECHNICAL',
    'architecture': 'TECHNICAL',
    'deploy': 'TECHNICAL',
    'infrastructure': 'TECHNICAL',
    'algorithm': 'TECHNICAL',
    'framework': 'TECHNICAL',
    'server': 'TECHNICAL',
    'config': 'TECHNICAL',
}

_DECISION_WORDS: frozenset = frozenset({
    'decided', 'because', 'instead', 'prefer', 'switched', 'chose',
    'realized', 'important', 'key', 'critical', 'discovered', 'learned',
    'conclusion', 'solution', 'reason', 'why', 'breakthrough', 'insight',
})

# ── Regex Patterns ───────────────────────────────────────────────────────────

_ENTITY_CODE_RE = re.compile(
    r'\$(\$?)(person|project|loc|org|tech|concept|event):'
    r'([^|★\n→]+?)(?=[|★\n→]|$)'
)

_DATE_ISO_RE = re.compile(
    r'\b(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z)?)\b'
)

_EMOTION_RE = re.compile(r'(★{1,3})')

_SESSION_RE = re.compile(r'^SESSION:(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?)\|?')

_DRAWER_RE = re.compile(r'DR:([a-zA-Z0-9_.-]+)')

_REL_ARROW_RE = re.compile(r'([^|→]+?)→([^|→★]+?)(?=\||$|★|\s)')

_REL_DASH_RE = re.compile(r'([A-Za-z0-9_ ]+)--(.+?)--([A-Za-z0-9_ ]+?)(?=\||$|★|\s)')

_CAPITALIZED_PHRASE_RE = re.compile(
    r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b'
)

_CAPITALIZED_WORD_RE = re.compile(r'\b([A-Z][a-z]{2,})\b')

_MIXED_CASE_WORD_RE = re.compile(r'\b([A-Z][a-z]+[A-Z][A-Za-z0-9]*)\b')

_ACRONYM_RE = re.compile(r'\b([A-Z]{2,8})\b')

_SENTENCE_START_WORDS = frozenset({
    'the', 'a', 'an', 'this', 'that', 'these', 'those', 'it', 'its',
    'we', 'they', 'he', 'she', 'i', 'you', 'there', 'here',
    'yesterday', 'today', 'tomorrow', 'now', 'then',
    'when', 'where', 'why', 'how', 'what', 'which', 'who',
    'all', 'every', 'each', 'both', 'some', 'any', 'no',
    'if', 'because', 'although', 'while', 'since', 'after', 'before',
    'in', 'on', 'at', 'for', 'with', 'by', 'from', 'to', 'of',
})

_WHITESPACE_RE = re.compile(r'\s+')

_EXCLAMATION_RE = re.compile(r'!')

_EMOJI_RE = re.compile(
    r'[\U0001F300-\U0001F9FF\u2600-\u26FF\u2700-\u27BF]'
)

_DATE_FIELD_RE = re.compile(r'\bDATE:(\d{4}-\d{2}-\d{2}(?:T\d{2}:\d{2}:\d{2}Z)?)\b')

_MOOD_FIELD_RE = re.compile(r'\bMOOD:(★{1,3})\b')


# ── Helper Functions ─────────────────────────────────────────────────────────

def _is_start_of_sentence(text: str, pos: int) -> bool:
    if pos == 0:
        return True
    before = text[:pos].rstrip()
    if not before:
        return True
    return before[-1] in '.!?'


_HUMAN_VERBS = frozenset({
    'discussed', 'discuss', 'said', 'say', 'told', 'tell', 'asked', 'ask',
    'went', 'go', 'came', 'come', 'thought', 'think', 'believed', 'believe',
    'wanted', 'want', 'decided', 'decide', 'created', 'create', 'built',
    'build', 'wrote', 'write', 'worked', 'work', 'developed', 'develop',
    'designed', 'design', 'implemented', 'implement', 'explained', 'explain',
    'suggested', 'suggest', 'proposed', 'propose', 'mentioned', 'mention',
    'reported', 'report', 'announced', 'announce', 'confirmed', 'confirm',
    'agreed', 'agree', 'promised', 'promise', 'offered', 'offer',
    'responded', 'respond', 'replied', 'reply',
    'met', 'meet', 'ran', 'run', 'walked', 'walk', 'spoke', 'speak',
    'talked', 'talk', 'listened', 'listen', 'showed', 'show',
    'led', 'lead', 'managed', 'manage', 'started', 'start',
})


def _context_before(text: str, pos: int, window: int = 5) -> str:
    before = text[:pos].strip()
    words = before.split()
    if not words:
        return ''
    return ' '.join(words[-window:]).lower()


def _words_set(context: str) -> set:
    return {w.strip(string.punctuation) for w in context.split() if w.strip(string.punctuation)}


def _immediate_prev_word(text: str, pos: int) -> str:
    before = text[:pos].strip()
    if not before:
        return ''
    return before.split()[-1].strip(string.punctuation).lower()


def _immediate_next_word(text: str, pos: int) -> str:
    after = text[pos:].strip()
    words = after.split()
    if len(words) >= 2:
        return words[1].strip(string.punctuation).lower()
    return ''


def _classify_entity(name: str, text: str, pos: int) -> str:
    ctx_before = _context_before(text, pos)
    ctx_before_words = _words_set(ctx_before)
    name_lower = name.lower()
    prev_word = _immediate_prev_word(text, pos)
    next_word = _immediate_next_word(text, pos)

    if name_lower in _TECH_NAMES:
        return 'tech'

    if any(name_lower.endswith(suffix) for suffix in _ORGANIZATION_HINTS):
        return 'org'
    org_patterns = {'corporation', 'incorporated', 'partners', 'associates', 'group'}
    if any(p in name_lower for p in org_patterns):
        return 'org'

    if name_lower in _ACRONYM_ORGS:
        return 'org'

    if name_lower in _KNOWN_LOCATIONS:
        return 'loc'

    if any(kw in name_lower for kw in _EVENT_KEYWORDS):
        return 'event'

    if any(title in ctx_before_words for title in _PERSON_TITLES):
        return 'person'

    if prev_word in _PERSON_VERBS:
        return 'person'

    if next_word in _HUMAN_VERBS:
        return 'person'

    if prev_word == 'project' or next_word == 'project':
        return 'project'
    if re.search(r'\bv?\d+\.\d+\b', ctx_before):
        return 'project'

    if prev_word in _LOCATION_PREPOSITIONS:
        return 'loc'

    if pos == 0 and ' ' not in name:
        return 'person'

    return 'concept'


def _find_entities(text: str):
    entities = []
    seen = set()
    occupied = set()

    def skip_sentence_start(name: str, pos: int) -> bool:
        if pos == 0 or _is_start_of_sentence(text, pos):
            return name.lower() in _SENTENCE_START_WORDS
        return False

    multi_matches = []
    for m in _CAPITALIZED_PHRASE_RE.finditer(text):
        name = m.group(1).strip()
        s, e = m.start(), m.end()
        if not skip_sentence_start(name, s) and name not in seen:
            multi_matches.append((name, s, e))
            seen.add(name)

    for name, s, e in multi_matches:
        for i in range(s, e):
            occupied.add(i)
        entity_type = _classify_entity(name, text, s)
        entities.append({'name': name, 'type': entity_type, 'start': s, 'end': e})

    for pattern in (_CAPITALIZED_WORD_RE, _MIXED_CASE_WORD_RE):
        for m in pattern.finditer(text):
            name = m.group(1)
            s, e = m.start(), m.end()
            if skip_sentence_start(name, s):
                continue
            if any(i in occupied for i in range(s, e)):
                continue
            if name in seen:
                continue
            if len(name) <= 2:
                continue
            if name.lower() in _STOPWORDS:
                continue
            seen.add(name)
            entity_type = _classify_entity(name, text, s)
            entities.append({'name': name, 'type': entity_type, 'start': s, 'end': e})

    seen_lower = {n.lower() for n in seen}
    for m in re.finditer(r'\b([a-z][a-z0-9]+)\b', text):
        name = m.group(1)
        s, e = m.start(), m.end()
        if any(i in occupied for i in range(s, e)):
            continue
        lower = name.lower()
        if lower not in _TECH_NAMES:
            continue
        if lower in seen_lower:
            continue
        if lower in _STOPWORDS:
            continue
        seen_lower.add(lower)
        entities.append({'name': name, 'type': 'tech', 'start': s, 'end': e})

    for m in _ACRONYM_RE.finditer(text):
        name = m.group(1)
        s, e = m.start(), m.end()
        if skip_sentence_start(name, s):
            continue
        if any(i in occupied for i in range(s, e)):
            continue
        if name in seen:
            continue
        seen.add(name)
        entity_type = _classify_entity(name, text, s)
        if entity_type == 'concept' and name.lower() not in _TECH_NAMES:
            entity_type = 'org'
        entities.append({'name': name, 'type': entity_type, 'start': s, 'end': e})

    return entities


def _entity_to_text(code_match: re.Match) -> str:
    return code_match.group(3).strip()


def _strip_stopwords(text: str) -> str:
    words = text.split()
    kept = []
    for w in words:
        clean = w.strip(string.punctuation)
        if not clean:
            kept.append(w)
        elif clean.lower() not in _STOPWORDS:
            kept.append(w)
        elif w.startswith('$') or w.startswith('DR:'):
            kept.append(w)
    return ' '.join(kept)


def _detect_emotion(text: str) -> str | None:
    excl = len(_EXCLAMATION_RE.findall(text))
    emoji = len(_EMOJI_RE.findall(text))
    total = excl + emoji
    if total >= 3:
        return '★★★'
    if total >= 2:
        return '★★'
    if total >= 1:
        return '★'
    return None


def _detect_date(text: str) -> str | None:
    m = _DATE_ISO_RE.search(text)
    return m.group(1) if m else None


# ── Enhanced Emotion/Flag/Topic/Sentence Detection ──────────────────────────


def aaak_detect_emotions(text: str) -> list[str]:
    """Detect emotion codes from plain text using keyword signals."""
    text_lower = text.lower()
    detected = []
    seen = set()
    for keyword, code in EMOTION_SIGNALS.items():
        if keyword in text_lower and code not in seen:
            detected.append(code)
            seen.add(code)
    return detected[:3]


def aaak_detect_flags(text: str) -> list[str]:
    """Detect importance/context flags from plain text using keyword signals."""
    text_lower = text.lower()
    detected = []
    seen = set()
    for keyword, flag in FLAG_SIGNALS.items():
        if keyword in text_lower and flag not in seen:
            detected.append(flag)
            seen.add(flag)
    return detected[:3]


def aaak_extract_topics(text: str, max_topics: int = 3) -> list[str]:
    """Extract key topic words from plain text using frequency analysis."""
    words = re.findall(r"[a-zA-Z][a-zA-Z_-]{2,}", text)
    freq = {}
    for w in words:
        w_lower = w.lower()
        if w_lower in _STOPWORDS or len(w_lower) < 3:
            continue
        freq[w_lower] = freq.get(w_lower, 0) + 1

    for w in words:
        w_lower = w.lower()
        if w_lower in _STOPWORDS:
            continue
        if w[0].isupper() and w_lower in freq:
            freq[w_lower] += 2
        if '_' in w or '-' in w or (any(c.isupper() for c in w[1:])):
            if w_lower in freq:
                freq[w_lower] += 2

    ranked = sorted(freq.items(), key=lambda x: -x[1])
    return [w for w, _ in ranked[:max_topics]]


def aaak_extract_key_sentence(text: str) -> str:
    """Extract the most important sentence fragment from text."""
    sentences = re.split(r"[.!?\n]+", text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if not sentences:
        return ''

    scored = []
    for s in sentences:
        score = 0
        s_lower = s.lower()
        for w in _DECISION_WORDS:
            if w in s_lower:
                score += 2
        if len(s) < 80:
            score += 1
        if len(s) < 40:
            score += 1
        if len(s) > 150:
            score -= 2
        scored.append((score, s))

    scored.sort(key=lambda x: -x[0])
    best = scored[0][1]
    if len(best) > 55:
        best = best[:52] + '...'
    return best


def aaak_count_tokens(text: str) -> int:
    """Estimate token count (~1.3 tokens per word)."""
    words = text.split()
    return max(1, int(len(words) * 1.3))


def aaak_compression_stats(original: str, compressed: str) -> dict:
    """Get size comparison stats for original -> AAAK compression."""
    orig_tokens = aaak_count_tokens(original)
    comp_tokens = aaak_count_tokens(compressed)
    return {
        'original_tokens_est': orig_tokens,
        'summary_tokens_est': comp_tokens,
        'size_ratio': round(orig_tokens / max(comp_tokens, 1), 1),
        'original_chars': len(original),
        'summary_chars': len(compressed),
        'note': 'Estimates only. Use tiktoken for accurate counts. AAAK is lossy.',
    }


# ── AaakDialect Class ──────────────────────────────────────────────────────


class AaakDialect:
    """AAAK encoder with configurable entity mappings, emotion/flag/topic extraction.

    Usage:
        # Basic usage
        dialect = AaakDialect()
        compressed = dialect.compress("We decided to use Python instead of Java...")

        # With custom entity codes
        dialect = AaakDialect(entities={"Alice": "ALC"})

        # From config file
        dialect = AaakDialect.from_config("entities.json")
    """

    def __init__(
        self,
        entities: dict[str, str] | None = None,
        skip_names: list[str] | None = None,
    ):
        self.entity_codes: dict[str, str] = {}
        if entities:
            for name, code in entities.items():
                self.entity_codes[name] = code
                self.entity_codes[name.lower()] = code
        self.skip_names = [n.lower() for n in (skip_names or [])]

    @classmethod
    def from_config(cls, config_path: str) -> 'AaakDialect':
        """Load entity mappings from a JSON config file."""
        import json
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return cls(
            entities=config.get('entities', {}),
            skip_names=config.get('skip_names', []),
        )

    def save_config(self, config_path: str) -> None:
        """Save current entity mappings to a JSON config file."""
        import json
        canonical = {}
        seen_codes = set()
        for name, code in self.entity_codes.items():
            if code not in seen_codes and not name.islower():
                canonical[name] = code
                seen_codes.add(code)
            elif code not in seen_codes:
                canonical[name] = code
                seen_codes.add(code)

        config = {
            'entities': canonical,
            'skip_names': self.skip_names,
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)

    def encode_entity(self, name: str) -> str | None:
        """Convert a person/entity name to its short code."""
        if any(s in name.lower() for s in self.skip_names):
            return None
        if name in self.entity_codes:
            return self.entity_codes[name]
        if name.lower() in self.entity_codes:
            return self.entity_codes[name.lower()]
        for key, code in self.entity_codes.items():
            if key.lower() in name.lower():
                return code
        return name[:3].upper()

    def compress(self, text: str, metadata: dict | None = None) -> str:
        """Compress plain text into AAAK Dialect format (lossy).

        Extracts entities, topics, key sentence, emotions, and flags.
        """
        metadata = metadata or {}

        entities_found = _find_entities(text)
        entity_codes = []
        for ent in entities_found:
            code = self.encode_entity(ent['name'])
            if code and code not in entity_codes:
                entity_codes.append(code)
        entity_str = '+'.join(entity_codes[:3]) if entity_codes else '???'

        topics = aaak_extract_topics(text)
        topic_str = '_'.join(topics[:3]) if topics else 'misc'

        quote = aaak_extract_key_sentence(text)
        quote_part = f'"{quote}"' if quote else ''

        emotions = aaak_detect_emotions(text)
        emotion_str = '+'.join(emotions) if emotions else ''

        flags = aaak_detect_flags(text)
        flag_str = '+'.join(flags) if flags else ''

        source = metadata.get('source_file', '')
        realm = metadata.get('realm', '')
        domain = metadata.get('domain', '')
        date = metadata.get('date', '')

        lines = []

        if source or realm:
            header_parts = [
                realm or '?',
                domain or '?',
                date or '?',
                Path(source).stem if source else '?',
            ]
            lines.append('|'.join(header_parts))

        parts = [f'0:{entity_str}', topic_str]
        if quote_part:
            parts.append(quote_part)
        if emotion_str:
            parts.append(emotion_str)
        if flag_str:
            parts.append(flag_str)

        lines.append('|'.join(parts))
        return '\n'.join(lines)

    def decode(self, dialect_text: str) -> dict:
        """Parse an AAAK Dialect string back into a readable summary."""
        text_lines = dialect_text.strip().split('\n')
        result = {'header': {}, 'zettels': []}

        for line in text_lines:
            if '|' in line and ':' in line.split('|')[0]:
                result['zettels'].append(line)
            elif '|' in line:
                parts = line.split('|')
                result['header'] = {
                    'file': parts[0] if len(parts) > 0 else '',
                    'entities': parts[1] if len(parts) > 1 else '',
                    'date': parts[2] if len(parts) > 2 else '',
                    'title': parts[3] if len(parts) > 3 else '',
                }

        return result

    def compression_stats(self, original: str, compressed: str) -> dict:
        """Get size comparison stats for original -> AAAK compression."""
        return aaak_compression_stats(original, compressed)

    @staticmethod
    def count_tokens(text: str) -> int:
        """Estimate token count (~1.3 tokens per word)."""
        return aaak_count_tokens(text)


# ── Public API ───────────────────────────────────────────────────────────────

def aaak_compress(text: str, max_len: int = 500) -> str:
    if not text or not text.strip():
        return ''

    original = text.strip()

    is_session = False
    session_date = None
    if _SESSION_RE.match(original):
        is_session = True
        m = _SESSION_RE.match(original)
        session_date = m.group(1)
        pipe_pos = original.find('|')
        if pipe_pos != -1:
            original = original[pipe_pos + 1:].strip()
        else:
            original = original[m.end():].strip()

    date_str = session_date or _detect_date(original)
    emotion = _detect_emotion(original)

    if date_str and not session_date:
        main_text = _DATE_ISO_RE.sub('', original, count=1).strip()
    else:
        main_text = original

    entities = _find_entities(main_text)

    clean_text = main_text
    for ent in sorted(entities, key=lambda e: -e['start']):
        clean_text = clean_text[:ent['start']] + clean_text[ent['end']:]

    compressed = _strip_stopwords(clean_text)
    compressed = _WHITESPACE_RE.sub(' ', compressed).strip()

    fields = []

    if is_session:
        fields.append(f'SESSION:{session_date}')
    elif date_str:
        fields.append(f'DATE:{date_str}')

    if emotion:
        fields.append(f'MOOD:{emotion}')

    seen_codes = set()
    for ent in entities:
        etype = ent['type']
        name = ent['name']
        if etype == 'project':
            code = f'$$project:{name}'
        else:
            code = f'${etype}:{name}'
        if code not in seen_codes:
            fields.append(code)
            seen_codes.add(code)

    for m in _REL_ARROW_RE.finditer(original):
        rel = f'{m.group(1).strip()}→{m.group(2).strip()}'
        if rel not in fields:
            fields.append(rel)
    for m in _REL_DASH_RE.finditer(original):
        rel = f'{m.group(1).strip()}--{m.group(2).strip()}--{m.group(3).strip()}'
        if rel not in fields:
            fields.append(rel)

    if compressed:
        fields.append(compressed)

    result = '|'.join(fields)

    if len(result) > max_len:
        if len(fields) > 1 and len('|'.join(fields[:-1])) + 2 < max_len:
            prefix = '|'.join(fields[:-1]) + '|'
            avail = max_len - len(prefix) - 1
            if avail > 10:
                truncated = fields[-1][:avail].rsplit(' ', 1)[0] + '…'
                result = prefix + truncated
            else:
                result = result[:max_len - 1] + '…'
        else:
            result = result[:max_len - 1] + '…'

    return result


def aaak_decompress(text: str) -> str:
    if not text or not text.strip():
        return ''

    result = text.strip()

    result = _ENTITY_CODE_RE.sub(_entity_to_text, result)

    result = result.replace('→', ' → ')

    result = _REL_DASH_RE.sub(r'\1 \2 \3', result)

    result = _MOOD_FIELD_RE.sub(r'\1', result)

    result = _DATE_FIELD_RE.sub(r'\1', result)

    result = result.replace('|', '\n')

    result = _WHITESPACE_RE.sub(' ', result).strip()

    return result


def aaak_validate(text: str) -> bool:
    if not text or not text.strip():
        return False

    checks = [
        bool(_ENTITY_CODE_RE.search(text)),
        '|' in text,
        bool(_EMOTION_RE.search(text)),
        bool(_SESSION_RE.match(text)),
        bool(_DRAWER_RE.search(text)),
        '→' in text,
        bool(_DATE_FIELD_RE.search(text)),
        bool(_MOOD_FIELD_RE.search(text)),
    ]
    return any(checks)


def aaak_parse_entry(text: str) -> dict:
    result: dict = {
        'raw': text,
        'entities': [],
        'emotion': None,
        'date': None,
        'session': False,
        'relationships': [],
    }

    if not text or not text.strip():
        return result

    session_match = _SESSION_RE.match(text)
    if session_match:
        result['session'] = True
        if result['date'] is None:
            result['date'] = session_match.group(1)

    date_match = _DATE_ISO_RE.search(text)
    if date_match:
        result['date'] = date_match.group(1)

    date_field_match = _DATE_FIELD_RE.search(text)
    if date_field_match and result['date'] is None:
        result['date'] = date_field_match.group(1)

    emotion_match = _EMOTION_RE.search(text)
    if emotion_match:
        raw = emotion_match.group(1)
        count = raw.count('★')
        if count >= 3:
            result['emotion'] = '★★★'
        elif count == 2:
            result['emotion'] = '★★'
        elif count == 1:
            result['emotion'] = '★'

    mood_match = _MOOD_FIELD_RE.search(text)
    if mood_match and result['emotion'] is None:
        result['emotion'] = mood_match.group(1)

    for m in _ENTITY_CODE_RE.finditer(text):
        prefix = '$' + m.group(1)
        etype = m.group(2)
        value = m.group(3).strip()
        code = f'{prefix}{etype}:{value}'
        result['entities'].append({'code': code, 'name': value})

    for m in _REL_ARROW_RE.finditer(text):
        result['relationships'].append({
            'source': m.group(1).strip(),
            'target': m.group(2).strip(),
            'label': '',
        })

    for m in _REL_DASH_RE.finditer(text):
        result['relationships'].append({
            'source': m.group(1).strip(),
            'target': m.group(3).strip(),
            'label': m.group(2).strip(),
        })

    return result
