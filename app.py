from flask import Flask, render_template, request, jsonify
import spacy
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)

# Spacy modelini yükle
# spacy.cli.download("en_core_web_lg")  # en_core_web_lg modelini indir
nlp = spacy.load("en_core_web_lg")  # en_core_web_lg modelini yükle

# GPT-2 modelini ve tokenizer'ını yükle
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")


def extract_entities(text):
    doc = nlp(text)
    entities = []
    for ent in doc.ents:
        entities.append({
            'text': ent.text,
            'label': ent.label_
        })
    return entities
@app.route('/')
def home():
    return render_template('index.html')


def generate_text(prompt, max_length, num_return_sequences, no_repeat_ngram_size, temperature):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=no_repeat_ngram_size,
        temperature=temperature
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    text = data['text']
    model = data['model']
    max_length = int(data['max_length'])
    num_return_sequences = int(data['num_return_sequences'])
    no_repeat_ngram_size = int(data['no_repeat_ngram_size'])
    temperature = float(data['temperature'])

    entities = extract_entities(text)
    prompt_entities = [entity['text'] for entity in entities]
    prompt = ' '.join(prompt_entities)
    generated_text = generate_text(prompt, max_length, num_return_sequences, no_repeat_ngram_size, temperature)

    return jsonify({'generated_text': generated_text, 'entities': entities})



if __name__ == '__main__':
    app.run(debug=True)
