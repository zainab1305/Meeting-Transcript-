# MeetingIQ — AI-Powered Meeting Intelligence

**A Hybrid NLP-Based Meeting Intelligence System with FLAN-T5 Summarization and Automated Action Item Extraction**

MeetingIQ transforms unstructured meeting transcripts into concise summaries and structured information, helping users identify important discussion points, action items, deadlines, decisions, and unresolved issues.

## Overview

Meetings often contain important information scattered across lengthy conversations. Manually reviewing transcripts to identify responsibilities, decisions, and deadlines can be time-consuming.

MeetingIQ addresses this problem through a hybrid NLP pipeline that combines traditional NLP techniques, rule-based information extraction, and experimental transformer-based abstractive summarization.

## Features

- **Meeting Summarization:** Generates an extractive summary using word-frequency-based sentence scoring.
- **Abstractive Summarization:** Integrates FLAN-T5 for experimental transformer-based summary generation.
- **Action Item Extraction:** Identifies potential tasks and commitments from meeting transcripts.
- **Speaker Identification:** Extracts speaker names from labelled transcript lines.
- **Deadline Detection:** Identifies explicit dates and deadline expressions.
- **Decision Detection:** Identifies statements expressing confirmed decisions.
- **Unresolved Issue Detection:** Detects statements indicating pending decisions or unresolved matters.
- **Structured Output:** Presents extracted information in organized tables.
- **JSON Export:** Allows structured meeting information to be exported for further use.
- **Interactive Interface:** Provides a user-friendly application built with Streamlit.

## System Architecture

```text
Meeting Transcript
        |
        v
Text Preprocessing
        |
        v
NLP Processing Pipeline
        |
        +-------------------------+
        |                         |
        v                         v
Extractive Summarization    Rule-Based Extraction
        |                   - Action Items
        |                   - Speakers
        |                   - Deadlines
        |                   - Decisions
        |                   - Unresolved Issues
        |                         |
        +------------+------------+
                     |
                     v
          Optional FLAN-T5 Summary
          (Experimental Component)
                     |
                     v
           Streamlit Dashboard
                     |
                     v
              JSON Export
```

## Technologies Used

- **Python** — Core application and NLP processing
- **Streamlit** — Interactive web interface
- **Regular Expressions (Regex)** — Pattern-based information extraction
- **NLTK / Text Processing Utilities** — Text processing where configured in the application
- **Hugging Face Transformers** — Transformer-based summarization
- **FLAN-T5** — Experimental abstractive summarization model
- **PyTorch** — Model inference and experimentation
- **JSON** — Structured data export

## Model Development

The project explores FLAN-T5-small for meeting summarization and structured meeting intelligence.

Fine-tuning experiments were conducted using meeting-transcript data to investigate the adaptation of a pretrained sequence-to-sequence language model to meeting-related tasks.

The application also includes an experimental FLAN-T5 summarization component. The rule-based extraction pipeline remains responsible for identifying structured information such as action items, decisions, and deadlines.

**Note:** The FLAN-T5 output is experimental and may occasionally contain unsupported details. Generated summaries should be verified against the original transcript. Model performance and fine-tuning results should be interpreted according to the evaluation conducted.

## Installation and Setup

### Prerequisites

- Python installed on your system
- Git
- A compatible environment for the installed PyTorch and Transformers versions

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd MeetingIQ
```

Replace the placeholder with your actual repository URL.

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

Open the local URL shown in the terminal, usually `http://localhost:8501`.

## How to Use

1. Launch the Streamlit application.
2. Enter or upload a meeting transcript using the supported input method.
3. Run the analysis.
4. Review the generated extractive summary and available experimental abstractive summary.
5. Examine action items, speakers, deadlines, decisions, and unresolved issues.
6. Export structured results as JSON if required.

For best results with the rule-based extraction pipeline, provide transcripts with clearly labelled speakers, such as:

```text
Neha: We decided to use React for the frontend.
Karan: I will complete the login module by Tuesday.
Riya: Could you handle API testing by Friday?
Neha: The launch date is not confirmed yet.
```

## Evaluation

Initial functional tests were performed using multiple sample meeting transcripts. These tests examined the application’s ability to generate summaries and identify action items, deadlines, decisions, and unresolved issues.

Quantitative evaluation should use manually annotated reference outputs and suitable metrics, including precision, recall, and F1-score for information extraction. Summarization quality should be evaluated separately.

No quantitative performance figures are claimed here unless supported by recorded evaluation results.

## Limitations

- Rule-based extraction depends on wording and transcript formatting.
- Implicit commitments and complex references may be missed.
- Deadline detection may not resolve relative expressions into calendar dates.
- Extractive summaries may omit relevant information.
- FLAN-T5 may generate inaccurate or unsupported content.
- Output quality depends on transcript quality and the chosen model configuration.

## Future Scope

- Improve extraction using trained sequence-labelling and information-extraction models.
- Evaluate summarization using suitable automatic metrics and human review.
- Improve deadline normalization and task-owner assignment.
- Support uploaded audio and automatic speech recognition.
- Add meeting search, history, and downloadable reports.
- Explore improved fine-tuning and validation strategies for meeting-specific language.

## Project Significance

MeetingIQ demonstrates how traditional NLP techniques and transformer-based language models can be combined to build a practical meeting intelligence application. It provides a foundation for exploring summarization, information extraction, and the reliability challenges of AI-generated meeting summaries.

## License

Choose and add an appropriate open-source license before distributing the project publicly.
