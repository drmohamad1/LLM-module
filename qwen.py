import os
import numpy as np
from sentence_transformers import SentenceTransformer, util
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import customtkinter as ctk  # Modern GUI framework

# Configuration
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"
docs_dir = "./translated_text_files"
output_dir = "./responses"
os.makedirs(output_dir, exist_ok=True)
max_new_tokens = 128  # Limit token count for generation
batch_size = 2  # Reduce batch size for memory efficiency

# Load sentence transformer model for embeddings
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Load Qwen model and tokenizer for text generation
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
#     device_map="auto"
# ).to(device)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    "universitytehran/PersianMind-v1.0",
    torch_dtype=torch.int8,
    low_cpu_mem_usage=True,
    device_map={"": device},
)
tokenizer = AutoTokenizer.from_pretrained(
    "universitytehran/PersianMind-v1.0",
)

# Function to encode documents into embeddings
def encode_texts(texts, batch_size=4):
    embeddings = []
    keys = list(texts.keys())
    for i in range(0, len(keys), batch_size):
        batch = keys[i:i + batch_size]
        batch_texts = [texts[k] for k in batch]
        batch_embeddings = embedder.encode(batch_texts, convert_to_tensor=True, batch_size=batch_size)
        embeddings.append(batch_embeddings.cpu().numpy())
    return np.vstack(embeddings), keys

# Function to find the most relevant document using cosine similarity
def find_most_relevant_document(query_embedding, embeddings, keys):
    similarities = util.pytorch_cos_sim(query_embedding, embeddings)[0]
    most_similar_idx = torch.argmax(similarities).item()
    return keys[most_similar_idx]

# Load documents
texts = {}
for filename in os.listdir(docs_dir):
    if filename.endswith(".txt"):
        with open(os.path.join(docs_dir, filename), 'r', encoding='utf-8') as file:
            texts[filename] = file.read()

embeddings, keys = encode_texts(texts, batch_size=batch_size)

# GUI Application
class PersianChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Persian Bot")
        self.geometry("700x600")
        ctk.set_appearance_mode("dark")  # Optional: Use "light" for a bright theme
        ctk.set_default_color_theme("blue")

        # Chat Display
        self.chat_log = ctk.CTkTextbox(self, wrap="word", font=("Arial", 14), state="disabled", height=450)
        self.chat_log.pack(padx=10, pady=10, fill="both", expand=True)

        # Input Box
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(fill="x", padx=10, pady=(0, 10))

        self.input_box = ctk.CTkTextbox(self.input_frame, wrap="word", font=("Arial", 14), height=50)
        self.input_box.pack(side="left", fill="x", expand=True, padx=(0, 10))

        self.send_button = ctk.CTkButton(self.input_frame, text="ارسال", command=self.send_message, width=100)
        self.send_button.pack(side="right")

    def send_message(self):
        user_query = self.input_box.get("1.0", "end").strip()
        if not user_query:
            return

        self.input_box.delete("1.0", "end")
        query_embedding = embedder.encode([user_query], convert_to_tensor=True).cpu().numpy()[0]
        most_relevant_doc = find_most_relevant_document(query_embedding, embeddings, keys)
        context = texts[most_relevant_doc]

        # Build prompt with system and user inputs
        messages = [
            {"role": "system", "content": "تو یک دستیار فارسی برای گیاه شناسی هستی."},
            {"role": "user", "content": f"Context: {context}"},
            {"role": "user", "content": f"Question: {user_query}"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer([prompt], return_tensors="pt").to(device)

        # Generate response
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
        response = tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()

        # Display the bot's response in the chat log
        self.chat_log.configure(state="normal")
        self.chat_log.insert("end", f"پاسخ: {response}\n\n")
        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")  # Scroll to the end

        # Save the response to a file
        output_file = os.path.join(output_dir, f"response_{most_relevant_doc}")
        with open(output_file, 'w', encoding='utf-8') as file:
            file.write(response)


# Run the application
if __name__ == "__main__":
    app = PersianChatbotApp()
    app.mainloop()
