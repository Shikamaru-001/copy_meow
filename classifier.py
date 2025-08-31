from speechbrain.inference.speaker import SpeakerRecognition
import glob
import os
import shutil


# Load pre-trained speaker verification model
model = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")



# STEP 1: Build your own voice reference using first sample
reference_file = "classified/jay/134.wav"


os.makedirs("classified/ben", exist_ok=True)
os.makedirs("classified/jay", exist_ok=True)


# STEP 2: Classify other notes using simple verification
files = glob.glob("voice_notes_wav/*.wav")

for f in files:
    try:
        # Use verify_files method which handles audio loading internally
        score, prediction = model.verify_files(reference_file, f)
        score = score.item()
        # Score > 0.5 typically means same speaker
        if score > 0.5:
            target_folder = "classified/jay"
        else:
            target_folder = "classified/ben"

        shutil.move(f, os.path.join(target_folder, os.path.basename(f)))
        print(f"{f} -> {target_folder} score(: {score:.3f})")
    except Exception as e:
        print(f"Error processing {f}: {e}")

