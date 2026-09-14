import pyttsx3
import os
import uuid

def create_tts_audio(text, output_dir="static/audio"):
    """
    Generate an offline voice buffer using system TTS and save as a wav file.
    Returns the relative path to the generated file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"story_{uuid.uuid4().hex[:8]}.wav"
    filepath = os.path.join(output_dir, filename)
    
    try:
        engine = pyttsx3.init()
        # Customizing voice speed
        rate = engine.getProperty('rate')
        engine.setProperty('rate', 150) # Slower and more natural for stories
        
        # Save to file silently
        engine.save_to_file(text, filepath)
        engine.runAndWait()
        
        return f"/static/audio/{filename}"
    except Exception as e:
        print(f"[Audio Error] Offline TTS generated an error: {e}")
        return None

if __name__ == "__main__":
    # Test block
    out = create_tts_audio("Once upon a time, there was a tiny ant.")
    print("Generated Audio File:", out)
