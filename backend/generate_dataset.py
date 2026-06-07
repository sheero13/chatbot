import json
import random

from transformers import pipeline

# =========================================
# LOAD MODEL
# =========================================

generator = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    max_new_tokens=120
)

# =========================================
# LOAD FAQ SEED DATA
# =========================================

with open(
    "faq_seed.json",
    "r",
    encoding="utf-8"
) as f:

    faq_data = json.load(f)

# =========================================
# FINAL DATASET
# =========================================

final_dataset = []

# =========================================
# GENERATE PARAPHRASES
# =========================================

for item in faq_data:

    question = item["question"]

    answer = item["answer"]

    print(f"\nGenerating for: {question}")

    prompt = f"""
Generate 10 different ways students may ask this college FAQ question.

Question:
{question}

Only generate questions.
"""

    try:

        result = generator(
            prompt,
            do_sample=True,
            temperature=0.9
        )

        generated_text = result[0]["generated_text"]

        lines = generated_text.split("\n")

        generated_questions = []

        for line in lines:

            line = line.strip()

            if len(line) > 10:

                if "?" in line:

                    generated_questions.append(line)

        # remove duplicates
        generated_questions = list(
            set(generated_questions)
        )

        # =================================
        # CREATE TRAINING FORMAT
        # =================================

        for q in generated_questions:

            sample = {
                "text":
                f"### Question: {q}\n"
                f"### Answer: {answer}"
            }

            final_dataset.append(sample)

    except Exception as e:

        print("ERROR:", e)

# =========================================
# SAVE DATASET
# =========================================

with open(
    "synthetic_dataset.json",
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        final_dataset,
        f,
        indent=4,
        ensure_ascii=False
    )

print("\n=================================")
print("DATASET GENERATION COMPLETE")
print(f"Total Samples: {len(final_dataset)}")
print("=================================")

