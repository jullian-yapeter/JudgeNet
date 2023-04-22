import pandas as pd
import torch
import time
import json

import requests
import os
import moviepy.editor as mp
from judgenet.modules.preprocess import SentenceEncoder
from utils import extractVGGish, scrapeVideoURL

positive_ratings = set(['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Informative', 'Fascinating', 'Persuasive', 'Jaw-dropping', 'Inspiring'])
negative_ratings = set(['Longwinded', 'Confusing', 'Unconvincing', 'OK', 'Obnoxious'])
data_processing_root = "data_processing/ted_pipeline"
data_root = "data/ted"

def fetchData():
    start_time = time.time()

    ted_main = pd.read_csv(f"{data_processing_root}/ted_main.csv")
    transcripts = pd.read_csv(f"{data_processing_root}/transcripts.csv")
    dataset_file = open(f"{data_root}/dataset.csv", "a")
    ted_dataset = pd.merge(ted_main, transcripts, on="url", how="inner")

    # Start from highwatermark
    highwater = int(open(f"{data_processing_root}/highwatermark.txt").readline())
    if highwater:
    # Reduce dataset to be from highwater
        ted_dataset = ted_dataset.iloc[highwater+1:]
    se = SentenceEncoder()

    # Loop through rows of the datafile
    for index, row in ted_dataset.iterrows():
        video_path = f"{data_root}/video/{index}.mp4"
        audio_path = f"{data_root}/audio/{index}.wav"
        lexical_feature_path = f"{data_root}/features/lexical/lexical_{index}.pt"
        audio_feature_path = f"{data_root}/features/audio/audio_{index}.pt"

        # Synthesize label from ratings
        ratings = row["ratings"].replace("'", '"')
        ratings_parsed = json.loads(ratings)

        pos_count = 0
        neg_count = 0
        for rating in ratings_parsed:
            if rating["name"] in positive_ratings:
                pos_count += rating["count"]
            elif rating["name"] in negative_ratings:
                neg_count += rating["count"]
            else:
                print("Found undefined rating")

        # Extract BERT embeddings from text
        batched_tokens = se.tokenize([row["transcript"]])
        sentence_embeddings = se.encode_batched_tokens(batched_tokens)
        torch.save(sentence_embeddings, lexical_feature_path)
        print("Got BERT embedding of transcript")

        url = row["url"][:-1]
        video_url = scrapeVideoURL(url)

        print("Fetched URL, downloading video")
        video_response = requests.get(video_url)
        open(video_path, 'wb').write(video_response.content)

        print("Converting to audio")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")

        print("Extracting VGGish features")
        audio_features = torch.from_numpy(extractVGGish(audio_path))
        audio_features_pooled = audio_features.float().mean(0)
        torch.save(audio_features_pooled, audio_feature_path)

        # Add row to dataset file
        print("Adding row to dataset file")
        row = f"\n{url},{audio_feature_path[5:]},{lexical_feature_path[5:]},{pos_count},{neg_count}"
        dataset_file.writelines(row)

        # Remove audio and video files
        os.remove(video_path)
        os.remove(audio_path)

        # Update HWM
        f = open("f{data_processing_root}/highwatermark.txt", "w")
        f.write(str(index))
        f.close()

        print(f"Batch {index} took {time.time() - start_time} seconds to run")
        start_time = time.time()
    dataset_file.close()
        
if __name__ == "__main__":
    fetchData()