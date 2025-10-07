# SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets
This repository provides the official implementation of the SD-MVSum method from our paper "SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets", along with two large-scale datasets for script-driven multimodal video summarization: S-MrHiSum and S-VideoXum. It includes the complete training and evaluation framework for SD-MVSum on these two datasets, as well as the supplementary files and notes required to reproduce our experimental results. You can either train SD-MVSum on the provided datasets (S-VideoXum and S-MrHiSum) and then evaluate your trained model, or use the provided already-trained models (checkpoints, trained on S-VideoXum or S-MrHiSum) to directly test their performance. In the following, Section A provides details about their structure and contents of the S-MrHiSum and S-VideoXum datasets, and Section B provides explanations about the SD-MVSum model's implementation and usage.

## A. Datasets: [S-MrHiSum | S-VideoXum] 

The original [MrHiSum](https://proceedings.neurips.cc/paper_files/paper/2023/file/7f880e3a325b06e3601af1384a653038-Paper-Datasets_and_Benchmarks.pdf) dataset (Sul et al., 2024) was constructed from a curated subset of YouTube-8M videos, where highlight annotations were derived from YouTube’s “Most Replayed” statistics. These replay statistics, aggregated from at least 50 independent viewers per video, provide a reliable signal of viewer-preferred highlights. Each video was annotated at the frame level with importance scores, reflecting highlight intensity. Ground-truth video summaries were generated based on a predefined temporal segmentation of the videos and by solving the Knapsack problem for a given time-budget about the summary duration, ensuring that the obtained summaries are concise while covering key highlights. In total, the dataset contains 31,892 videos with aligned visual highlight annotations, supporting the training and evaluation of methods for video highlight detection and summarization.

To make MrHiSum suitable for script-driven multimodal video summarization, we extended it by producing textual descriptions of the human-annotated summaries and extracting audio transcripts, forming the S-MrHiSum dataset. For this, the visual content of each ground-truth video summary (sampled at 1 fps) was described by LLaVA-NeXT-Video-7B which was prompted to "describe the important scenes in this video". Audio transcripts were extracted through a two-step pipeline: the speech was isolated from background noise using a pretrained model of Silero for voice activity detection, and then speech-to-text was performed using a pretrained model of Whisper, which outputs a series of time-stamped transcripts. The created S-MrHiSum dataset contains 29,918 videos, where each video is associated with: a) ground-truth summary, b) a textual description of this summary, and c) a set of time-stamped audio transcripts.

The [S-VideoXum](https://zenodo.org/records/15349075) dataset was presented in [Mylonas et. al, 2025](https://arxiv.org/html/2505.03319v2) as an extension of the [VideoXum](https://videoxum.github.io/) dataset for cross-modal video summarization, that is suitable for training and evaluation of methods for script-driven video summarization. For this, the multiple ground-truth summaries that are available per video of VideoXum, were associated with textual descriptions of their visual content, generated using LLaVA-NeXT-Video-7B. We further extended the S-VideoXum dataset by extracting time-stamped audio transcripts from each full-length video, following the approach described above for the videos of the S-MrHiSum dataset.

The visual, textual, and transcript data of the S-MrHiSum and S-VideoXum datasets were represented using CLIP-based embeddings. In particular, for the S-MrHiSum dataset we employed the CLIP ViT-B/32 model (Radford at. al, 2021) from [HuggingFace](https://huggingface.co/sentence-transformers/clip-ViT-B-32), while for the S-VideoXum dataset we utilized a fine-tuned CLIP model on the data of VideoXum, that was released by the authors of [VideoXum](https://videoxum.github.io/).

### Folder Structure of the Dataset

```
dataset/SS
├── script_videoxum.h5
├── script_videoxum_split.json
├── script_mrhisum.h5
├── script_mrhisum_split.json
```

---
### 1.1 `script_mrhisum.h5`
The core HDF5 file for the S-MrHiSum dataset. Each top-level group corresponds to a different video of the dataset and has been named by the video's name. Each of these groups contains the following information:
 
| Key                   | Description                                                                                                 | Shape / Type                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------|
| `n_frames`            | Number of video frames                                                                                      | Scalar integer                          |
| `change_points`       | Indices of start and end frame of each video shot                                                           | `[num_shots, 2]`                        |
| `gt_summary`          | Binary ground-truth summary derived from gtscores using the Knapsack algorithm with a 15% time budget       | `[num_sampled_frames]` (binary vector)  |
| `video_embeddings`    | Frame-level CLIP embeddings for the sub-sampled video frames (at 1 fps)                                     | `[num_sampled_frames, 512]`             |
| `text_embeddings`     | Sentence-level CLIP embeddings for the textual description of the ground-truth video summary (script)       | `[M, 512]` (M = number of sentences)         |
| `transcript_embeddings` | Chunk-level CLIP embeddings for the extracted audio transcript                                            | `[N, 512]` (N = number of chunks)            |
| `transcript_timestamps` | Start and end time for each chunk of the audio transcript                                                 | `[N, 2]` (N = number of chunks)              |
| `aligned_transcripts` | Transcript embeddings that are time-aligned with the frame-level embeddings; zero-padding when transcripts are not available (there is no spoken content in the video)   | `[num_downsampled_frames, 512]`         |




### 1.2 `script_videoxum.h5`
The core HDF5 file for the S-VideoXum dataset. Each top-level group corresponds to a different video of the dataset and has been named by the video's name. Each of these groups contains the following information:

| Key                    | Description                                                                                                         | Shape / Type                        |
|------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `n_frames`             | Number of sub-sampled frames in the video (at 1 fps)                                                                          | Scalar integer                       |
| `gtscores`             | Ground‐truth frame-level importance scores from 10 human annotators                                                                   | `[10, n_frames]`                    |
| `video_embeddings`     | Frame-level CLIP embeddings for the sub-sampled video frames                                                                                 | `[n_frames, 512]`                   |
| `text_embeddings`      | Sentence-level CLIP embeddings for the textual description of each of the 10 available ground-truth video summaries (scripts); zero padding if a description has less than `M_max`sentences                     | `[10, M_max, 512]`                   |
| `transcript_embeddings`| Chunk-level CLIP embeddings for the extracted audio transcript                                                                 | `[N, 512]` (N = number of transcript chunks)    |
| `transcript_timestamps`| Start and end time for each chunk of the audio transcript                                   | `[N, 2]`                            |
| `aligned_transcripts`  | Transcript embeddings that are time-aligned with the frame-level embeddings; zero-padding when transcripts are not available (there is no spoken content in the video) | `[n_frames, 512]`                   |


### 2. JSON Split Files 
JSON files define train, validation, test splits for the respective datasets.

       `script_mrhisum_split.json`
       `script_videoxum_split.json`
    

### 3. Text Annotations

This folder, available on Zenodo, contains all textual data used in the datasets, including generated scripts and timestamped audio transcripts. The scripts were generated using LLaVA-NeXT-Video-7B, and transcripts were extracted using Silero and Whisper.

#### 3.1 `Scripts/`
  Contains the generated scripts for MrHiSum videos, produced by LLaVA-NeXT-Video-7B.

#### 3.2 `Transcripts/`
  Contains timestamped audio transcripts for both datasets, organized into subfolders produced by Whisper Turbo:
    
      `S-MrHiSum/` — transcripts for MrHiSum videos
        
      `S-VideoXum/` — transcripts for VideoXum videos

## B. SD-MVSum method and models

This section of the repository provides scripts and trained models for training and/or evaluating the SD-MVSum method for script-driven video summarization.

### Installation

Clone the repository
   ```
  git clone https://github.com/IDT-ITI/SD-MVSum.git
  cd SD-MVSum
   ```
Create and activate the Conda environment
   ```
   conda env create -f environment.yml
   conda activate sd_mvsum
   ```
### Dataset Preparation

Download the datasets
   - Retrieve the .h5 files and the split.json files from Zenodo.
   - Place both files under the ```dataset``` directory
      ```
      SD-MVSum
       └── dataset/
            ├── script_mrhisum.h5
            ├── script_mrhisum_split.json
            ├── script_videoxum.h5
            └── script_videoxum_split.json
      ``` 

### Pretrained Models

Download the pretrained SD-MVSum models trained on S-VideoXum and S-MrHiSum from the Zenodo directory, and save them to a local folder.

### Usage
SD-MVSum can be employed for both training on the published datasets and conducting inference using pretrained models.

#### Train model on S-VideoXum and S-MrHiSum
Both S-VideoXum and S-MrHiSum include training, validation, and test splits. After each training epoch, the model is evaluated on the validation set, and once training is complete, the best-performing checkpoint on the validation set is evaluated on the test set. The final model is saved as a .pkl file. To train on either dataset, run one of the following commands:
```
python main.py --exp_num='exp1' --epochs=E --batch_size=B --train=True --dataset='S_VideoXum'
python main.py --exp_num='exp2' --epochs=E --batch_size=B --train=True --dataset='S_MrHisum'
```


#### Inference on Pretrained Models
To run inference using a pretrained checkpoint on either S-VideoXum or S-MrHiSum:

```
python main.py --exp_num='exp1' --ckpt_path='path/to/pkl/file' --train=False --dataset='S_VideoXum'
python main.py --exp_num='exp2' --ckpt_path='path/to/pkl/file' --train=False --dataset='S_MrHisum'
```

## Citation

The S-VideoXum and S-MrHiSum datasets, as well as the SD-MVSum method, were proposed in our paper: M. Mylonas, C. Zerva E. Apostolidis, V. Mezaris, "SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets", Under review.
```bibtex
@misc{sdmvsum2026,
      title={"SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets"}, 
      author={Manolis Mylonas and Charalampia Zerva and Evlampios Apostolidis and Vasileios Mezaris},
      year={2025},
      note={under review}
}
```

## License
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. 

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
