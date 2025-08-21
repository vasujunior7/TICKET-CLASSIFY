# import pandas as pd
# import os
# import time
# import random
# from openai import OpenAI

# # Set up OpenAI API
# client = OpenAI()

# # Define categories and prompts
# categories = {
#     "billing": "Generate a short, realistic customer support ticket related to billing issues such as overcharges, refunds, invoices, or payment failures.",
#     "technical": "Generate a short, realistic customer support ticket related to technical issues such as app crashes, login failures, bugs, or connectivity problems.",
#     "other": "Generate a short, realistic customer support ticket related to general inquiries such as office hours, policies, or account updates."
# }

# def generate_ticket(label, n=1):
#     """Generate synthetic support tickets for a given label"""
#     prompt = f"""
#     You are a support ticket generator. Write {n} short, natural-sounding customer support requests.
#     Category: {label.upper()}
#     Guidelines:
#     - Keep sentences between 8–30 words.
#     - Avoid repeating the same style.
#     - Make them diverse and realistic.
#     - Do not number or bullet the answers, just give raw sentences.
#     """

#     try:
#         response = client.chat.completions.create(
#             model="gpt-3.5-turbo",
#             messages=[
#                 {"role": "system", "content": "You generate synthetic but realistic customer support tickets."},
#                 {"role": "user", "content": prompt}
#             ],
#             max_tokens=500,
#             temperature=0.8
#         )
#         text = response.choices[0].message.content.strip()
#         # Split into lines
#         examples = [line.strip().strip('"') for line in text.split("\n") if line.strip()]
#         return [(ex, label) for ex in examples]
#     except Exception as e:
#         print(f"Error generating {label}: {e}")
#         return []

# def build_dataset(samples_per_class=100, output_file="support_tickets.csv"):
#     """Generate full dataset with 100 examples per class"""
#     all_data = []
#     for label in categories.keys():
#         print(f"Generating {samples_per_class} examples for {label}...")
#         collected = []
#         while len(collected) < samples_per_class:
#             needed = min(10, samples_per_class - len(collected))
#             new_examples = generate_ticket(label, n=needed)
#             collected.extend(new_examples)
#             time.sleep(2)  # avoid rate limits
#         all_data.extend(collected)

#     # Convert to DataFrame
#     df = pd.DataFrame(all_data, columns=["text", "label"])
#     df.to_csv(output_file, index=False, encoding="utf-8")
#     print(f"✅ Dataset created: {output_file} ({len(df)} rows)")

# if __name__ == "__main__":
#     build_dataset(samples_per_class=100, output_file="support_tickets.csv")

import pandas as pd
import os
import time
import random
from tqdm import tqdm
from openai import OpenAI

# Set up OpenAI API
# os.environ[''] = 'xxxxxxxxxxxxxxxxxxxxxxxxx'
client = OpenAI()

# Define categories and prompts
categories = {
    "billing": "Generate a short, realistic customer support ticket related to billing issues such as overcharges, refunds, invoices, or payment failures.",
    "technical": "Generate a short, realistic customer support ticket related to technical issues such as app crashes, login failures, bugs, or connectivity problems.",
    "other": "Generate a short, realistic customer support ticket related to general inquiries such as office hours, policies, or account updates."
}

def generate_ticket(label, n=1):
    """Generate synthetic support tickets for a given label"""
    prompt = f"""
    You are a support ticket generator. Write {n} short, natural-sounding customer support requests.
    Category: {label.upper()}
    Guidelines:
    - Keep sentences between 8–30 words.
    - Avoid repeating the same style.
    - Make them diverse and realistic.
    - Do not number or bullet the answers, just give raw sentences.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You generate synthetic but realistic customer support tickets."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=500,
            temperature=0.8
        )
        text = response.choices[0].message.content.strip()
        # Split into lines
        examples = [line.strip().strip('"') for line in text.split("\n") if line.strip()]
        return [(ex, label) for ex in examples]
    except Exception as e:
        print(f"Error generating {label}: {e}")
        return []

def build_dataset(samples_per_class=100, output_file="datasets/support_tickets.csv"):
    """Generate full dataset with N examples per class"""
    all_data = []
    
    # Ensure folder exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    for label in categories.keys():
        print(f"\n🔹 Generating {samples_per_class} examples for {label}...")
        collected = []

        with tqdm(total=samples_per_class, desc=f"{label}", unit="sample") as pbar:
            while len(collected) < samples_per_class:
                needed = min(10, samples_per_class - len(collected))
                new_examples = generate_ticket(label, n=needed)
                collected.extend(new_examples)
                pbar.update(len(new_examples))
                time.sleep(2)  # avoid rate limits

        all_data.extend(collected)

    # Convert to DataFrame
    df = pd.DataFrame(all_data, columns=["text", "label"])
    df.to_csv(output_file, index=False, encoding="utf-8")
    print(f"\n✅ Dataset created: {output_file} ({len(df)} rows)")

if __name__ == "__main__":
    build_dataset(samples_per_class=100, output_file="./datasets/support_tickets.csv")
