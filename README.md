ASR Model Error Dataset – NVIDIA Parakeet 0.6B
Overview

This dataset contains examples of inputs where the NVIDIA Parakeet TDT 0.6B speech-to-text model made incorrect predictions. The goal is to highlight model blind spots, including errors with accents, background noise, technical vocabulary, and other challenging speech scenarios. This dataset can be used to fine-tune or evaluate ASR models.

Model Tested

Model: nvidia/parakeet-tdt-0.6b-v2

Parameters: 0.6B
Type: Base Automatic Speech Recognition model (not fine-tuned for a specific domain)
Modality: Audio / Speech-to-Text

How the Model Was Loaded

The model was loaded using Hugging Face Transformers in Python:

!pip install transformers datasets soundfile

from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import soundfile as sf

model_id = "nvidia/parakeet-tdt-0.6b-v2"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)

# Example transcription
speech, sr = sf.read("test_audio.wav")
inputs = processor(speech, sampling_rate=sr, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = processor.batch_decode(predicted_ids)
print(transcription)
Dataset Structure
input_audio	model_output	expected_output	error_type
test1.wav	“I go school”	“I go to school”	Missing word
test2.wav	“read the report”	“read the report carefully”	Partial transcription
...	...	...	...

input_audio: Name of the audio file

model_output: Text predicted by the ASR model

expected_output: Correct transcription

error_type: Type of mistake (e.g., accent misrecognition, background noise, technical term error)

Observations

The model struggles in the following scenarios:

Non-standard accents and dialects

Background noise and overlapping speech

Fast or low-volume speech

Technical vocabulary or homophones

Recommendations

To improve model performance, fine-tuning on a diverse, multi-accent, noisy, and domain-rich speech dataset is recommended. A dataset of 10,000+ labeled samples covering these variations would help address blind spots.
