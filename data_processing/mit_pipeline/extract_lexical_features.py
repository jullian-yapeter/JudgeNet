import torch
import pandas as pd
import numpy as np
import os


from judgenet.modules.preprocess import SentenceEncoder

def main():
    folder_path = "data/mit_interview"
    transcripts = pd.read_csv(f"{folder_path}/interview_transcripts_by_turkers.csv")

    for index, row in transcripts.iterrows():
        transcript = row["Transcript"]
        embedding = extract_bert(transcript)
        torch.save(embedding, f"{folder_path}/features/lexical_batched/{index}.pt")
        print(f"Finished file {index}")

def extract_bert(transcript):
    split = transcript.split('Interviewee:')
    # Trim transcript to Interviewee speech
    cleaned = ""
    for row in split[1:]:
        cleaned += row.split('|')[0]

    tokens = se.tokenize(cleaned)
    return se.encode_batched_tokens(tokens)

se = SentenceEncoder()
current_pos = 0

if __name__ == "__main__":
    main()