from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

response = pipe("Question: What is hill climbing algorithm?", max_new_tokens=256)
print("Response:", response[0]['generated_text'])
