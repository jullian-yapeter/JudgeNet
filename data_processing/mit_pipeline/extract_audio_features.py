from data_processing.utils import extractVGGish
import torch
import os

data_processing_root = "data_processing/mit_pipeline"
data_root = "data/mit_interview"

def main():
    print("Extracting VGGish features")
    audio_directory = f"{data_processing_root}/audio"
    i = 1
    for file_name in os.listdir(audio_directory):
        file_number = file_name.split('/')[-1].split('.')[0]
        audio_path = os.path.join(audio_directory, file_name)
        audio_feature_path = f"{data_root}/features/vggish/audio_{file_number}.pt"

        audio_features = torch.from_numpy(extractVGGish(audio_path))
        audio_features_pooled = audio_features.float().mean(0)
        torch.save(audio_features_pooled, audio_feature_path)
        print(f"Finished file {i}/138")
        i += 1

if __name__ == "__main__":
    main()