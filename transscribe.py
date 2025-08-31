import whisper
import glob
import os
import csv

model = whisper.load_model("medium")
transcript_file = "classified/jay_split/metadata.csv"
wav_folder = "classified/jay_split/"


with open(transcript_file, "w", newline="", encoding="utf-8") as csvfile:
    try:     
        writer = csv.writer(csvfile, delimiter="|")

        print(os.path.join(wav_folder, "*.wav"))
        # Process all WAV files
        for i, file in enumerate(glob.glob(os.path.join(wav_folder, "*.wav")), 1):
            print(f"Transcribing {file} ...")
            result = model.transcribe(file)
            text = result["text"].strip()

        # Save filename (without extension) and transcript
            base_name = os.path.splitext(os.path.basename(file))[0]
            writer.writerow([base_name, text])
            print(f"Saved: {base_name} → {text}")
    except Exception as e:
        print(f"Error Processing: {e}")
