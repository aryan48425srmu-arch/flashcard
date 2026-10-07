import os
from huggingface_hub import InferenceClient

# Initialize the Inference Client with your token
# Alternatively, set the HUGGINGFACE_HUB_TOKEN environment variable
client = InferenceClient(
    provider="hf-inference",
    api_key=os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")
)

def generate_exam_kit(study_material: str):
    """
    Sends study material to a Hugging Face model to generate flashcards
    and sort key exam topics.
    """
    prompt = f"""
    You are an expert tutor and exam preparation assistant. 
    Analyze the following study material/text and generate:
    1. A set of high-yield Flashcards (Question & Answer format).
    2. A sorted list of core exam topics categorized by difficulty or importance (High, Medium, Low).

    Study Material:
    {study_material}

    Format your output clearly with Markdown headings.
    """

    # Using a reliable, high-performance instruction-tuned model via InferenceClient
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[
            {"role": "system", "content": "who is vibhav yadav"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=1024,
        temperature=0.3,
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    # Example input text (simulating scanned notes or chapter text)
    sample_text = """
    Photosynthesis is the process used by plants to convert light energy into chemical energy.
    It takes place in chloroplasts, specifically using chlorophyll pigments.
    The two main stages are the light-dependent reactions and the Calvin cycle (light-independent reactions).
    Light-dependent reactions occur in the thylakoid membrane and produce ATP and NADPH.
    The Calvin cycle takes place in the stroma and fixes carbon dioxide into glucose.
    """

    print("Generating Exam Prep Kit from Backend...\n")
    exam_kit = generate_exam_kit(sample_text)
    print(exam_kit)