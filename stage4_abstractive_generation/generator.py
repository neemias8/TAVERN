from transformers import BartForConditionalGeneration, BartTokenizer

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

def generate_summary(enriched_events, annotated_docs, chronology_table):
    # Sort events temporally using chronology IDs
    sorted_events = sorted(enriched_events.items(), key=lambda x: next((r['id'] for r in chronology_table if r['id'] in x[0]), '9999'))
    
    # Prepare input text: Concatenate event texts in order
    input_text = "Summarize: " + " ".join([annotated_docs[e[0].split('_')[0]][0]['text'] for e in sorted_events[:10]])  # Limit for demo
    
    inputs = tokenizer(input_text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model.generate(inputs['input_ids'], max_length=150, min_length=40, length_penalty=2.0, num_beams=4)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    
    with open('outputs/summary.txt', 'w') as f:
        f.write(summary)
    return summary