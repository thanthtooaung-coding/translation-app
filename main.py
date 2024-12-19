from transformers import MarianMTModel, MarianTokenizer
from transformers.utils import logging

logging.set_verbosity_error()


def translate(text, src_lang, tgt_lang):
    """Translate text from source language to target language."""
    try:
        model_name = f'Helsinki-NLP/opus-mt-{src_lang}-{tgt_lang}'

        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)

        inputs = tokenizer(text, return_tensors="pt", padding=True)

        translated = model.generate(**inputs)

        translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
        return translated_text

    except Exception as e:
        return f"Error: Unable to load the model for {src_lang} to {tgt_lang}. Ensure the language pair is valid.\nDetails: {e}"


if __name__ == "__main__":
    # Text to translate
    text = "Hey Buddy! How are you doing?"

    # Source and target languages
    src_lang = "en"  # English
    tgt_lang = "fr"  # Japanese

    translated_text = translate(text, src_lang, tgt_lang)
    print(f"Translated text: {translated_text}")
