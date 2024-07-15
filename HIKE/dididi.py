from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/mgenre-wiki")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/mgenre-wiki").eval()

print(model)