import os
import base64
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image
import io

# Load environment variables
load_dotenv()

def verify_gemini():
    print("Verifying Gemini Multimodal Helper...")
    
    # 1. Check API Key
    if "GOOGLE_API_KEY" not in os.environ:
        print("❌ GOOGLE_API_KEY not found in environment.")
        return
    else:
        print("✅ GOOGLE_API_KEY found.")

    try:
        # 2. Initialize Model
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
        print("✅ Gemini Model initialized.")

        # 3. Create a simple image (Red Square)
        img = Image.new('RGB', (100, 100), color = 'red')
        
        # Save to buffer
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        img_data_url = f"data:image/jpeg;base64,{img_str}"

        # 4. Invoke Model
        print("Sending request to Gemini...")
        message = HumanMessage(
            content=[
                {
                    "type": "text",
                    "text": "What color is this image? Answer in one word.",
                },
                {
                    "type": "image_url",
                    "image_url": {"url": img_data_url},
                },
            ]
        )
        
        response = llm.invoke([message])
        content = response.content.lower()
        print(f"Gemini Response: {content}")

        if "red" in content:
            print("✅ Verification Successful: Model identified the color red.")
        else:
            print(f"⚠️ Verification Inconclusive: Model response was '{content}'.")

    except Exception as e:
        print(f"❌ Verification Failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_gemini()
