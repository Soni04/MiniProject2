from flask import Flask, request, render_template, send_file, url_for
from transformers import BlipProcessor, BlipForConditionalGeneration, MBartForConditionalGeneration, MBart50Tokenizer
from gtts import gTTS
from PIL import Image
import os

app = Flask(__name__, static_folder='static')

# Load BLIP model for image captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load mBART model for translation
mbart_model_name = "facebook/mbart-large-50-many-to-many-mmt"
mbart_tokenizer = MBart50Tokenizer.from_pretrained(mbart_model_name)
mbart_model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

# Language mappings
language_map = {
    "hin": "hi",
    "ben": "bn",
    "guj": "gu",
    "kan": "kn",
    "mal": "ml",
    "mar": "mr",
    "tam": "ta",
    "tel": "te",
    "urd": "ur",
}
lang_code_map = {
    "hin": "hi_IN",
    "ben": "bn_IN",
    "guj": "gu_IN",
    "kan": "kn_IN",
    "mal": "ml_IN",
    "mar": "mr_IN",
    "tam": "ta_IN",
    "tel": "te_IN",
    "urd": "ur_IN",
}

# Function to generate image caption using BLIP
def generate_image_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs, max_length=50)
    caption = blip_processor.decode(output[0], skip_special_tokens=True)
    return caption

# Function to translate text using mBART
def translate_with_mbart(text, target_language):
    lang_code = lang_code_map.get(target_language, "hi_IN")
    tokenized_input = mbart_tokenizer(text, return_tensors="pt")
    generated_tokens = mbart_model.generate(**tokenized_input, forced_bos_token_id=mbart_tokenizer.lang_code_to_id[lang_code])
    return mbart_tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# Function to convert text to speech using gTTS
def text_to_speech(text, language, output_file):
    gtts_lang = language_map.get(language, "en")
    tts = gTTS(text=text, lang=gtts_lang)
    tts.save(output_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process():
    # Get the uploaded file
    image_file = request.files['image']
    target_language = request.form['language']
    
    if image_file and target_language:
        # Save the uploaded image
        upload_folder = os.path.join(app.static_folder, "uploads")
        os.makedirs(upload_folder, exist_ok=True)
        image_path = os.path.join(upload_folder, image_file.filename)
        image_file.save(image_path)

        image_url = url_for('static', filename=f"uploads/{image_file.filename}")
        
        # Process the image
        image = Image.open(image_path).convert('RGB')
        caption = generate_image_caption(image)
        translated_caption = translate_with_mbart(caption, target_language)
        
        # Generate the audio file
        audio_file = os.path.join(upload_folder, "output_description.mp3")
        text_to_speech(translated_caption, target_language, audio_file)
        
        # Pass the audio file to the template
        audio_file_url = url_for('static', filename=f"uploads/output_description.mp3")
        return render_template('index.html', 
        audio_file=audio_file_url,
        caption=caption,
        translated_caption=translated_caption,image_url=image_url)
    
    return "Image and language selection are required!", 400

if __name__ == '__main__':
    app.run(debug=True)
