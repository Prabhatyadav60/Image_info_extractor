import streamlit as st
import base64
import time
import json
from openai import OpenAI

# --- OpenRouter API credentials ---
OPENROUTER_API_KEY = st.secrets["OPENROUTER_API_KEY"]

def get_image_info_from_openrouter(data_url):
    """
    Calls the OpenRouter API (using the Qwen2.5 model) to extract image info.
    It sends a multimodal message with a text prompt and the image (as a data URL).
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
    )
    completion = client.chat.completions.create(
        extra_body={},
        model="qwen/qwen2.5-vl-72b-instruct:free",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is in this image?"},
                    {"type": "image_url", "image_url": {"url": data_url}}
                ]
            }
        ]
    )
    return completion.choices[0].message.content

def main():
    st.title("Direct Image Analyzer")
    st.write("Upload an image and let the OpenRouter API analyze it directly (using a data URL).")
    
    uploaded_file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "gif"])
    
    if uploaded_file is not None:
        # Display the uploaded image
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        # Read the image bytes and construct a data URL
        file_bytes = uploaded_file.read()
        mime_type = uploaded_file.type  # e.g., "image/jpeg"
        b64_str = base64.b64encode(file_bytes).decode('utf-8')
        data_url = f"data:{mime_type};base64,{b64_str}"
        
        if st.button("Analyze Image"):
            with st.spinner("Waiting for 5 seconds..."):
                time.sleep(5)  # wait for propagation, if needed
            with st.spinner("Extracting image information..."):
                try:
                    image_info = get_image_info_from_openrouter(data_url)
                    st.success("Image information extracted!")
                    st.subheader("Extracted Information")
                    st.write(image_info)
                    st.image(data_url, caption="Analyzed Image", use_column_width=True)
                except Exception as e:
                    st.error("Error fetching image info: " + str(e))

if __name__ == "__main__":
    main()
