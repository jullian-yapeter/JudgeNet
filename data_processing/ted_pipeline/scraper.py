import pandas as pd
import numpy as np
import torch

import requests
import os
import json
from bs4 import BeautifulSoup
import moviepy.editor as mp
from disvoice_local.disvoice.prosody import Prosody



def fetchData():
    ted_main = pd.read_csv("data/ted_main.csv")
    transcripts = pd.read_csv("data/transcripts.csv")

    dataset = pd.merge(ted_main, transcripts, on="url", how="inner")

    positive_ratings = set(['Funny', 'Beautiful', 'Ingenious', 'Courageous', 'Informative', 'Fascinating', 'Persuasive', 'Jaw-dropping', 'Inspiring'])
    negative_ratings = set(['Longwinded', 'Confusing', 'Unconvincing', 'OK', 'Obnoxious'])

    # Loop through rows of the datafile
    for index, row in dataset.iterrows():
        # First synthesize label from ratings
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
        
        rating = pos_count // neg_count
        print(f"Synthesized rating of {rating}")


        # Scrape mp4 file from webpage and convert to wav
        url = row["url"][:-1]
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        text = soup.find(id="__NEXT_DATA__")
        parsed = json.loads(text.string)

        playerData = parsed["props"]["pageProps"]["videoData"]["playerData"]
        parsed2 = json.loads(playerData)

        video_url = parsed2["resources"]["h264"][0]["file"]

        print("Fetched URL, downloading video")
        video_response = requests.get(video_url)
        video_path = "video/video_" + str(index) + ".mp4"
        audio_path = "audio/audio_" + str(index) + ".wav"
        feature_path = "features/prosody_" + str(index) + ".pt"

        open(video_path, 'wb').write(video_response.content)

        print("Converting to audio")
        video = mp.VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path, codec="pcm_s16le")

        print("Extracting Prosody features")
        # Now extract prosody features from audio
        prosody = Prosody()
        prosody_features = prosody.extract_features_file(audio_path, static=False, plots=True, fmt="torch")
        torch.save(prosody_features, feature_path)


        break
        

if __name__ == "__main__":
    fetchData()