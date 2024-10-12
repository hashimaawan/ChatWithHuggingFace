import streamlit as st
from PIL import Image
from diffusers import StableDiffusionPipeline
from transformers import pipeline

# Set page configuration before any other Streamlit commands
st.set_page_config(page_title="Chat with Hugging Face", layout="wide")

# Load the custom CSS
with open("hugging.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Initialize model and pipelines
model_id = "CompVis/stable-diffusion-v1-4"
token = ""  # Replace with your Hugging Face token

# Initialize the Stable Diffusion Pipeline with CPU
pipe = StableDiffusionPipeline.from_pretrained(model_id, use_auth_token=token)
pipe.to("cpu")  # Use CPU

# Initialize pipelines for text processing with CPU
text_classifier = pipeline("text-classification", device=-1)  # Run on CPU
summarizer = pipeline("summarization", device=-1)  # Run on CPU
translator = pipeline("translation_en_to_fr", device=-1)  # Run on CPU

# Add a Hugging Face image
st.image("https://huggingface.co/front/assets/huggingface_logo.svg", use_column_width=False, width=100)

# Set page title and subtitle with improved colors
st.markdown("<h1>Chat with Hugging Face</h1>", unsafe_allow_html=True)
st.markdown("<h3>Enter your prompt and select an action:</h3>", unsafe_allow_html=True)

# User input
prompt = st.text_area("Enter your prompt here", height=100)

if st.button("Submit"):
    if not prompt:
        st.error("Please enter a prompt.")
    else:
        prompt_lower = prompt.lower()
        
        # Handling different prompt types
        if "generate image" in prompt_lower:
            prompt_for_image = prompt.replace("generate image", "").strip()
            with st.spinner("Generating image..."):
                image = pipe(prompt_for_image).images[0]
                resized_image = image.resize((256, 256))  # Resize the image to 256x256 pixels
                st.image(resized_image, caption="Generated Image", use_column_width=False)
                st.success("Image generated successfully!")

        elif "classify text" in prompt_lower:
            text_to_classify = prompt.replace("classify text", "").strip()
            with st.spinner("Classifying text..."):
                classification = text_classifier(text_to_classify)
                st.write("Classification:", classification)

        elif "summarize text" in prompt_lower:
            text_to_summarize = prompt.replace("summarize text", "").strip()
            with st.spinner("Summarizing text..."):
                summary = summarizer(text_to_summarize, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
                st.write("Summary:", summary)

        elif "translate text" in prompt_lower:
            text_to_translate = prompt.replace("translate text", "").strip()
            with st.spinner("Translating text..."):
                translation = translator(text_to_translate)[0]['translation_text']
                st.write("Translation:", translation)

        else:
            st.error("Sorry, I didn't understand the prompt.")
