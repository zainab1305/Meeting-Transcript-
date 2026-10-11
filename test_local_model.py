
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_DIR = Path(__file__).parent / "meetingiq_summary_model"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True)

print("Loading model...")
model = AutoModelForSeq2SeqLM.from_pretrained(
    str(MODEL_DIR),
    local_files_only=True
)
model.eval()

transcript = """
Neha: We need to finalize the mobile application launch plan.
Karan: I will complete the login and authentication module by Tuesday.
Neha: I will prepare the updated UI designs by Wednesday.
Riya: I will write the API testing documentation by Friday.
"""

prompt = (
    "Write a short, factual summary of this meeting transcript. "
    "Mention only facts explicitly stated. Do not repeat sentences "
    "or invent decisions. If details are unclear, omit them.\n\n"
    + transcript
)

inputs = tokenizer(
    prompt,
    return_tensors="pt",
    max_length=512,
    truncation=True
)

print("Generating summary...")
with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=70,
        num_beams=5,
        no_repeat_ngram_size=3,
        repetition_penalty=1.8,
        length_penalty=1.0,
        early_stopping=True,
        do_sample=False
    )

summary = tokenizer.decode(output[0], skip_special_tokens=True)
print("\nMODEL SUMMARY:\n")
print(summary)
