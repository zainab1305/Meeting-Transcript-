
import re
import string
from collections import Counter
from pathlib import Path

import streamlit as st


# ---------- PAGE CONFIGURATION ----------

st.set_page_config(
    page_title="MeetingIQ",
    page_icon="📝",
    layout="wide"
)

st.title("MeetingIQ")
st.caption(
    "NLP-Based Meeting Intelligence and Action Item Extraction"
)


# ---------- TEXT PREPROCESSING ----------

def preprocess_text(text):
    text = text.replace("\r\n", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------- RULE-BASED NLP ----------

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "have", "in", "is", "it", "of", "on", "or", "that", "the",
    "this", "to", "was", "we", "will", "with", "you"
}

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
DEADLINE_PATTERN = re.compile(
    r"\b(?:by|before|until|next)\s+"
    r"(?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday|"
    r"week|month|weekend|week\s+\w+|\d{1,2}(?:st|nd|rd|th)?)\b"
    r"|\b(?:today|tomorrow|tonight)\b"
    r"|\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b",
    re.IGNORECASE
)
SPEAKER_PATTERN = re.compile(
    r"^\s*([A-Z][A-Za-z .'-]{1,30})\s*:\s*(.+)$"
)


def split_sentences(text):
    return [
        sentence.strip()
        for sentence in SENTENCE_PATTERN.split(text)
        if sentence.strip()
    ]


def words_for_scoring(text):
    words = re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())
    return [word.strip(string.punctuation) for word in words if word not in STOP_WORDS]


def decision_topic(text):
    topic = re.sub(
        r"\b(?:we|have|has|decided|decide|agreed|to|use|will|should|"
        r"still|need|select|choose|the|a|an|for|is|not)\b",
        " ",
        text.lower()
    )
    return set(words_for_scoring(topic))


def summarize_text(text, max_sentences=5):
    sentences = [sentence for sentence in split_sentences(text) if len(sentence.split()) >= 4]
    if not sentences:
        return []

    frequencies = Counter(words_for_scoring(text))
    if not frequencies:
        return sentences[:max_sentences]

    highest_frequency = max(frequencies.values())
    normalized = {
        word: count / highest_frequency
        for word, count in frequencies.items()
    }
    confirmed_decisions = [
        sentence for sentence in sentences
        if re.search(r"\b(?:we have decided|we decided|we agreed|agreed to|let's use)\b", sentence, re.IGNORECASE)
    ]
    confirmed_topics = [decision_topic(sentence) for sentence in confirmed_decisions]
    protected = set(confirmed_decisions)
    for sentence in sentences:
        if re.search(r"\b(?:I will|I'll|is responsible for|we will)\b", sentence, re.IGNORECASE):
            protected.add(sentence)
        if re.search(r"\b(?:still need to|need to select|need to choose|open question|unresolved)\b", sentence, re.IGNORECASE):
            protected.add(sentence)

    ranked = []
    for index, sentence in enumerate(sentences):
        sentence_topic = decision_topic(sentence)
        is_repeated_suggestion = (
            bool(re.search(r"\b(?:should|suggest|propose)\b", sentence, re.IGNORECASE))
            and any(sentence_topic and sentence_topic <= topic for topic in confirmed_topics)
        )
        if is_repeated_suggestion and sentence not in protected:
            continue
        sentence_words = words_for_scoring(sentence)
        score = sum(normalized.get(word, 0) for word in sentence_words)
        if sentence in protected:
            score += 2
        ranked.append((score / max(len(sentence_words), 1), -index, index, sentence))

    selected = sorted(ranked, reverse=True)[:max_sentences]
    return [item[3] for item in sorted(selected, key=lambda item: item[2])]


def extract_people(text):
    people = []
    for line in text.splitlines():
        match = SPEAKER_PATTERN.match(line)
        if match:
            person = match.group(1).strip()
            if person not in people:
                people.append(person)
    return people


def extract_dates(text):
    dates = []
    for match in DEADLINE_PATTERN.finditer(text):
        date = match.group(0).strip()
        if date.lower() not in {item.lower() for item in dates}:
            dates.append(date)
    return dates


# ---------- ACTION ITEM EXTRACTION ----------

def extract_action_items(text):
    actions = []
    action_pattern = re.compile(
        r"\b(?:I will|I'll|I can|I should|I need to|"
        r"we will|will|is responsible for|agreed to)\b",
        re.IGNORECASE
    )
    confirmed_pattern = re.compile(
        r"\b(?:I will|I'll|we will|is responsible for|agreed to)\b",
        re.IGNORECASE
    )
    introductory_phrase = re.compile(
        r"^(?:I will|I'll|I can|I should|I need to|we will|will|"
        r"is responsible for|agreed to)\s+",
        re.IGNORECASE
    )

    for line in text.splitlines():
        line = line.strip()

        if not line or not action_pattern.search(line):
            continue

        speaker_match = SPEAKER_PATTERN.match(line)

        speaker = (
            speaker_match.group(1).strip()
            if speaker_match
            else "Not identified"
        )

        sentence = (
            speaker_match.group(2).strip()
            if speaker_match
            else line
        )

        deadline_match = DEADLINE_PATTERN.search(sentence)
        deadline = (
            deadline_match.group(0).removeprefix("by ")
            if deadline_match
            else "Not specified"
        )
        task = introductory_phrase.sub("", sentence).strip()
        if deadline_match:
            task = DEADLINE_PATTERN.sub("", task).strip(" .,;:-")

        actions.append({
            "Person": speaker,
            "Status": "Confirmed" if confirmed_pattern.search(sentence) else "Proposed",
            "Task": task,
            "Deadline": deadline,
            "Original sentence": sentence
        })

    return actions


# ---------- DECISION EXTRACTION ----------

def extract_decisions(text):
    decisions = []

    confirmed_pattern = re.compile(
        r"\b(?:we have decided|we decided|it is decided|we agreed|"
        r"agreed to|let's use|we will use)\b",
        re.IGNORECASE
    )
    unresolved_pattern = re.compile(
        r"\b(?:we still need to|still need to|need to select|"
        r"need to choose|open question|unresolved)\b",
        re.IGNORECASE
    )
    seen_topics = set()

    for content in split_sentences(text):
        status = None
        if confirmed_pattern.search(content):
            status = "Confirmed decision"
        elif unresolved_pattern.search(content):
            status = "Unresolved issue"

        if status:
            topic = " ".join(sorted(decision_topic(content)))
            if topic not in seen_topics:
                decisions.append({
                    "Status": status,
                    "Decision or issue": content,
                    "Original sentence": content
                })
                seen_topics.add(topic)

    return decisions


# ---------- USER INTERFACE ----------

uploaded_file = st.file_uploader(
    "Upload a meeting transcript (.txt)",
    type=["txt"]
)

default_text = ""

if Path("sample_meeting.txt").exists():
    default_text = Path("sample_meeting.txt").read_text(
        encoding="utf-8"
    )

if uploaded_file is not None:
    transcript = uploaded_file.getvalue().decode(
        "utf-8", errors="replace"
    )
else:
    transcript = default_text

transcript = st.text_area(
    "Meeting transcript",
    value=transcript,
    height=260,
    placeholder="Paste your meeting transcript here..."
)

if st.button("Analyze Meeting", type="primary"):
    clean_text = preprocess_text(transcript)

    if not clean_text:
        st.warning("Please provide a transcript first.")
    else:
        with st.spinner("Analyzing meeting..."):
            summary = summarize_text(clean_text)
            people = extract_people(clean_text)
            dates = extract_dates(clean_text)
            actions = extract_action_items(clean_text)
            decisions = extract_decisions(clean_text)

        st.success("Analysis complete")

        st.subheader("Meeting Overview")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Words", len(clean_text.split()))
        c2.metric("Potential Actions", len(actions))
        c3.metric("Decisions Detected", len(decisions))
        c4.metric("People Detected", len(people))

        st.divider()

        left, right = st.columns(2)

        with left:
            st.subheader("Meeting Summary")

            for sentence in summary:
                st.markdown(f"- {sentence}")

            st.subheader("People")

            if people:
                st.write(", ".join(people))
            else:
                st.info("No speaker names detected.")

            st.subheader("Dates and Deadlines")

            if dates:
                for date in dates:
                    st.markdown(f"- {date}")
            else:
                st.info("No dates or deadlines detected.")

        with right:
            st.subheader("Potential Decisions")

            if decisions:
                confirmed_decisions = [
                    decision for decision in decisions
                    if decision["Status"] == "Confirmed decision"
                ]
                unresolved_issues = [
                    decision for decision in decisions
                    if decision["Status"] == "Unresolved issue"
                ]

                if confirmed_decisions:
                    st.markdown("**Confirmed decisions**")
                    st.dataframe(
                        confirmed_decisions,
                        use_container_width=True,
                        hide_index=True
                    )
                if unresolved_issues:
                    st.markdown("**Unresolved issues**")
                    st.dataframe(
                        unresolved_issues,
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("No decisions detected.")

            st.subheader("Potential Action Items")

            if actions:
                confirmed_actions = [
                    action for action in actions
                    if action["Status"] == "Confirmed"
                ]
                proposed_actions = [
                    action for action in actions
                    if action["Status"] == "Proposed"
                ]

                if confirmed_actions:
                    st.markdown("**Confirmed actions**")
                    st.dataframe(confirmed_actions, use_container_width=True, hide_index=True)
                if proposed_actions:
                    st.markdown("**Proposed actions**")
                    st.dataframe(proposed_actions, use_container_width=True, hide_index=True)
            else:
                st.info("No potential action items detected.")

        st.divider()
        st.subheader("Original Transcript")
        st.text(clean_text)
