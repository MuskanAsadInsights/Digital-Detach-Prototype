from google import genai
from google.genai import types
from PIL import Image
import os
import json
import io
from dotenv import load_dotenv
from google import genai

# This tells Python to look for the .env file in the main folder
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'))

API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

def compress_image(image_path):
    """Resizes image to a smaller width to save API tokens."""
    img = Image.open(image_path)
    # Resize to 800px width while maintaining aspect ratio
    w_percent = (800 / float(img.size[0]))
    h_size = int((float(img.size[1]) * float(w_percent)))
    img = img.resize((800, h_size), Image.LANCZOS)
    
    # Save to a byte buffer
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=70) # Quality 70 is enough for text
    return img_byte_arr.getvalue()

def extract_universal_screentime(image_paths):
    """Handles multiple screenshots by compressing them first."""
    contents = ["Analyze these screenshots and merge data into one JSON object:"]
    
    for path in image_paths:
        try:
            # COMPRESS EACH IMAGE BEFORE SENDING
            compressed_data = compress_image(path)
            contents.append(types.Part.from_bytes(data=compressed_data, mime_type="image/jpeg"))
        except Exception as e:
            print(f"Error processing {path}: {e}")

    prompt_text = """
    Return ONLY this JSON structure:
    {
        "Daily_Usage_Hours": float,
        "Phone_Checks_Per_Day": int,
        "Time_on_Social_Media": float,
        "Device_Type": "iOS" or "Android"
    }
    Rules: Convert time to decimal hours. If data spans multiple images, sum it up or average it logically.
    """
    contents.append(prompt_text)

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=contents,
            config=types.GenerateContentConfig(response_mime_type="application/json")
        )

        # 2. PRIVACY CLEANUP (Delete original files)
        for path in image_paths:
            if os.path.exists(path):
                os.remove(path)
        
        return json.loads(response.text)

    except Exception as e:
        error_msg = str(e)
        if "429" in error_msg:
            return {"error": "API Limit Reached. Even with compression, try only 2-3 images."}
        return {"error": f"Extraction failed: {error_msg[:100]}"}

if __name__ == "__main__":
    test_folder = "uploads"
    valid_extensions = ('.png', '.jpg', '.jpeg', '.webp')
    test_files = [os.path.join(test_folder, f) for f in os.listdir(test_folder) 
                  if f.lower().endswith(valid_extensions)]
    
    if test_files:
        print(f"Compressing and analyzing {len(test_files)} image(s)...")
        result = extract_universal_screentime(test_files)
        print("--- FINAL SYSTEM DATA ---")
        print(json.dumps(result, indent=4))
    else:
        print("Put your screenshots in the 'uploads' folder first.")