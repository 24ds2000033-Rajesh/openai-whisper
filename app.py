import base64
import io
import os
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import whisper

app = FastAPI()

# Load Whisper model globally onto CPU or GPU
# "base" or "tiny" is usually fast and accurate enough for structured tasks
model = whisper.load_model("base")

class AudioRequest(BaseModel):
    audio_id: str
    audio_base64: str

def safe_dict(series):
    """Helper to convert pandas Series to dict, ensuring JSON compatibility."""
    if series is None or series.empty:
        return {}
    # Convert numpy types to python native types
    return {str(k): (v.item() if hasattr(v, "item") else v) for k, v in series.to_dict().items()}

@app.post("/analyze-audio")
async def analyze_audio(payload: AudioRequest):
    try:
        # 1. Decode base64 audio
        audio_bytes = base64.b64decode(payload.audio_base64)
        
        # Save temporarily as WAV (Whisper expects a file path or numpy array)
        temp_filename = f"temp_{payload.audio_id}.wav"
        with open(temp_filename, "wb") as f:
            f.write(audio_bytes)
            
        # 2. Transcribe using Whisper
        # Note: If your audio directly contains tabular data spoken aloud, 
        # you will parse the text here into a pandas DataFrame.
        result = model.transcribe(temp_filename)
        transcribed_text = result.get("text", "")
        
        # Clean up temp file
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
            
        # 3. Simulate/Parse DataFrame from the transcription or audio metadata
        # (Replace this placeholder with your exact audio-to-dataframe logic)
        # For demonstration, we assume a DataFrame is generated:
        df = pd.DataFrame({
            "feature_1": [10, 20, 30, 40, 20],
            "feature_2": [1.5, 2.5, 3.5, 4.5, 2.5]
        })
        
        # 4. Compute requested statistics strictly matching your schema
        numeric_df = df.select_dtypes(include=[np.number])
        
        # Calculations
        mean_stats = numeric_df.mean()
        std_stats = numeric_df.std(ddof=1) # Sample std deviation
        var_stats = numeric_df.var(ddof=1)
        min_stats = numeric_df.min()
        max_stats = numeric_df.max()
        median_stats = numeric_df.median()
        
        # Mode handling (can return multiple values, pick the first)
        mode_dict = {}
        for col in numeric_df.columns:
            modes = numeric_df[col].mode()
            mode_dict[col] = modes.iloc[0] if not modes.empty else np.nan
            
        range_stats = max_stats - min_stats
        
        # Custom placeholders for your format matching rules:
        allowed_values = {col: list(df[col].unique()) for col in df.columns}
        value_range = {col: [float(min_stats[col]), float(max_stats[col])] for col in numeric_df.columns}
        
        # Correlation matrix to list of lists/dicts if required
        corr_matrix = numeric_df.corr().values.tolist()

        # 5. Formulate the exact exact strict JSON response
        response_data = {
            "rows": int(len(df)),
            "columns": list(df.columns),
            "mean": safe_dict(mean_stats),
            "std": safe_dict(std_stats),
            "variance": safe_dict(var_stats),
            "min": safe_dict(min_stats),
            "max": safe_dict(max_stats),
            "median": safe_dict(median_stats),
            "mode": mode_dict,
            "range": safe_dict(range_stats),
            "allowed_values": allowed_values,
            "value_range": value_range,
            "correlation": corr_matrix
        }
        
        return response_data

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
