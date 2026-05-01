# SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets
This repository provides the official implementation of the SD-MVSum method from our paper "SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets", along with two large-scale datasets for script-driven multimodal video summarization: SM-MrHiSum and SM-VideoXum. It includes the code for training and evaluating SD-MVSum on each of these two datasets, as well as the supplementary files and notes required to reproduce our experimental results. You can either train SD-MVSum on the provided datasets (SM-MrHiSum and SM-VideoXum) and then evaluate your trained model, or use the provided already-trained models (checkpoints, trained on SM-MrHiSum or SM-VideoXum) to directly test their performance. In the following, Section A provides details about the structure and contents of the SM-MrHiSum and SM-VideoXum datasets, and Section B provides information about the SD-MVSum model's implementation and usage.

## A. Datasets: [SM-MrHiSum | SM-VideoXum] 

The original [MrHiSum](https://proceedings.neurips.cc/paper_files/paper/2023/file/7f880e3a325b06e3601af1384a653038-Paper-Datasets_and_Benchmarks.pdf) dataset (Sul et al., 2024) was constructed from a curated subset of YouTube-8M videos, where highlight annotations were derived from YouTube’s “Most Replayed” statistics. These video replay statistics, aggregated from at least 50 independent viewers per video, serve as a reliable indicator of audience engagement. Each video was annotated at the frame level with importance scores, representing highlight intensity. Ground-truth video summaries were generated based on a predefined temporal segmentation of the videos and by solving the Knapsack problem for a given time-budget about the summary duration, ensuring that the obtained summaries are concise while covering key highlights. In total, the dataset contains 31,892 videos and the associated ground-truth annotations, supporting the training and evaluation of methods for video highlight detection and summarization.

To make MrHiSum suitable for script-driven multimodal video summarization, we extended it by producing textual descriptions of the human-annotated summaries and extracting audio transcripts, forming the SM-MrHiSum dataset. For this, the visual content of each ground-truth video summary (sampled at 1 fps) was described by Qwen3-VL-8B-Instruct which was prompted to "describe the scenery and the main persons and activities shown in the video". Audio transcripts were extracted through a two-step pipeline: the speech was isolated from background noise using a pretrained model of Silero for voice activity detection, and then speech-to-text was performed using a pretrained model of Whisper, which outputs a series of time-stamped transcripts. The created SM-MrHiSum dataset contains 29,918 videos, where each video is associated with: a) ground-truth summary, b) a textual description of this summary (the so-called script), and c) a set of time-stamped audio transcripts.

The [S-VideoXum](https://zenodo.org/records/15349075) dataset was presented in [Mylonas et. al, 2025](https://arxiv.org/html/2505.03319v2) as an extension of the [VideoXum](https://videoxum.github.io/) dataset for cross-modal video summarization, that is suitable for training and evaluation of methods for script-driven video summarization. For this, the multiple ground-truth summaries that are available per video of VideoXum, were associated with textual descriptions of their visual content, generated using LLaVA-NeXT-Video-7B. We further extended the S-VideoXum dataset by extracting time-stamped audio transcripts from each full-length video, following the approach described above for the videos of the SM-MrHiSum dataset, forming the SM-VideoXum dataset that is suitable for training and evaluation of methods for script-driven multimodal video summarization.

In our implementations and experiments, all the visual, textual, and transcript data of the SM-MrHiSum and SM-VideoXum datasets have been represented using CLIP-based embeddings. In particular, for the data of the SM-MrHiSum dataset we employed the CLIP ViT-B/32 model (Radford at. al, 2021) from [HuggingFace](https://huggingface.co/sentence-transformers/clip-ViT-B-32), while for the data of the SM-VideoXum dataset we utilized a fine-tuned CLIP model on the data of VideoXum, that was released by the authors of [VideoXum](https://videoxum.github.io/).

### Folder Structure for each Dataset

```
<dataset_name>/   with <dataset_name> = [sm_mrhisum | sm_videoxum]
├── dataset/
├──── <dataset_name>.h5
├──── <dataset_name>_split.json
├── model/
├──── layers/
├──── utils/
├──── configs.py
├──── data_loader.py
├──── solver.py
├── main.py
```

---
### 1.1 `sm_mrhisum.h5`
The core HDF5 file for the S-MrHiSum dataset. Each top-level group corresponds to a different video of the dataset and has been named by the video's name. Each of these groups contains the following information:
 
| Key                   | Description                                                                                                 | Shape / Type                       |
|-----------------------|-------------------------------------------------------------------------------------------------------------|------------------------------------|
| `n_frames`            | Number of sub-sampled frames in the video (at 1 fps)                                                        | Scalar integer                          |
| `change_points`       | Indices of start and end frame of each video shot                                                           | `[num_shots, 2]`                        |
| `gt_summary`          | Binary ground-truth summary derived from gtscores using the Knapsack algorithm with a 15% time budget       | `[n_frames]` (binary vector)  |
| `video_embeddings`    | Frame-level CLIP embeddings for the sub-sampled video frames (at 1 fps)                                     | `[n_frames, 512]`             |
| `qwen_text_embeddings`  | Sentence-level CLIP embeddings for the textual description of the ground-truth video summary (script)       | `[M, 512]` (M = number of sentences)         |
| `transcript_embeddings` | Chunk-level CLIP embeddings for the extracted audio transcript                                            | `[N, 512]` (N = number of chunks)            |
| `transcript_timestamps` | Start and end time for each chunk of the audio transcript                                                 | `[N, 2]` (N = number of chunks)              |
| `aligned_transcripts` | Transcript embeddings that are time-aligned with the frame-level embeddings; zero-padding when transcripts are not available (there is no spoken content in the video)   | `[n_frames, 512]`         |


### 1.2 `script_videoxum.h5`
The core HDF5 file for the S-VideoXum dataset. Each top-level group corresponds to a different video of the dataset and has been named by the video's name. Each of these groups contains the following information:

| Key                    | Description                                                                                                         | Shape / Type                        |
|------------------------|---------------------------------------------------------------------------------------------------------------------|-------------------------------------|
| `n_frames`             | Number of frames in the video                                                                          | Scalar integer                       |
| `gtscores`             | Ground‐truth frame-level importance scores from 10 human annotators                                    | `[10, n_sampled_frames]`                    |
| `video_embeddings`     | Frame-level CLIP embeddings for the sub-sampled video frames (at 1 fps)                                | `[n_sampled_frames, 512]`                   |
| `qwen2_embeddings`     | Sentence-level CLIP embeddings for the textual description of each of the 10 available ground-truth video summaries (scripts); zero padding if a description has less than `M_max`sentences                     | `[10, M_max, 512]`                   |
| `transcript_embeddings`| Chunk-level CLIP embeddings for the extracted audio transcript                                         | `[N, 512]` (N = number of transcript chunks)    |
| `transcript_timestamps`| Start and end time for each chunk of the audio transcript                                   | `[N, 2]`                            |
| `aligned_transcripts`  | Transcript embeddings that are time-aligned with the frame-level embeddings; zero-padding when transcripts are not available (there is no spoken content in the video) | `[n_sampled_frames, 512]`                   |

### 2. JSON Split Files 
JSON files with the video names in the train, validation, and test set of each dataset.

       `sm_mrhisum_split.json`
       `sm_videoxum_split.json`
    
### 3. Text Annotations

The generated text annotations, i.e., the textual descriptions of the ground-truth video summaries (the so-called scripts) and the time-stamped audio transcripts of the full-length videos, are publicly available on [Zenodo](https://zenodo.org/records/17294445).

#### 3.1 `Scripts/`
  Contains the generated scripts for the ground-truth summaries of the MrHiSum and VideoXum videos, organized into the following subfolders:.

      `SM-MrHiSum/` — scripts for the ground-truth summaries of the MrHiSum videos
      `SM-VideoXum/` — scripts for the ground-truth summaries of the VideoXum videos

#### 3.2 `Transcripts/`
  Contains the extracted time-stamped audio transcripts for the full-length videos of both datasets, organized into the following subfolders:
    
      `SM-MrHiSum/` — audio transcripts for the MrHiSum videos
      `SM-VideoXum/` — audio transcripts for the VideoXum videos

## B. SD-MVSum method and models

This section provides details about the training and evaluation of the developed SD-MVSum method, and access to pretrained models of SD-MVSum on the SM-MrHiSum and SM-VideoXum datasets.

The files for implementing SD-MVSum and preparing it for training and evaluation on each dataset are shown below: 

```
<dataset_name>/   with <dataset_name> = [sm_mrhisum | sm_videoxum]
├── model/
├──── layers/
├──── utils/
├──── configs.py
├──── data_loader.py
├──── solver.py
├── main.py
```

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
### Dataset preparation

Download the datasets
   - Download the .h5 files and the split.json files from [Zenodo](https://zenodo.org/records/17294445).
   - Place each files under the ```dataset``` directories as shown below.
      ```
      SD-MVSum
      └── sm-mrhisum/
          └── dataset/
              ├── sm_mrhisum.h5
              └── sm_mrhisum_split.json
      └── sm-videoxum/
          └── dataset/
              ├── sm_videoxum.h5
              └── sm_videoxum_split.json
      ``` 

#### Training on S-VideoXum and S-MrHiSum
To train a model on the SM-MrHiSum and SM-VideoXum datasets, please use the relevant 'main.py' script and run the following command:
```
python main.py --exp_num='exp1' --epochs=50 --batch_size=64 --train=True --dataset='SM_MrHisum'
python main.py --exp_num='exp1' --epochs=50 --batch_size=4 --train=True --dataset='SM_VideoXum'
```
After each training epoch, the trained model is evaluated on the samples of the validation set. When training is completed, the best-performing model on the validation set is selected and evaluated on the test set. Moreover, its checkpoint is saved as a .pkl file (see the generated folder "best_f1score_model").

#### Inference using pretrained models
Download the pretrained SD-MVSum models (.pkl files) on SM-MrHiSum and SM-VideoXum from [Zenodo](https://zenodo.org/records/17294445).
To run them at inference mode on the SM-MrHiSum and SM-VideoXum datasets, please run the following commmands:

```
python main.py --exp_num='exp2' --ckpt_path='path/to/pkl/file' --train=False --dataset='SM_MrHisum'
python main.py --exp_num='exp2' --ckpt_path='path/to/pkl/file' --train=False --dataset='SM_VideoXum'
```
After the completion of the inference stage, the performance of these models is shown on the terminal.

## Citation

The SM-MrHiSum and SM-VideoXum datasets, as well as the SD-MVSum method for script-driven multimodal video summarization, were proposed in our paper: M. Mylonas, C. Zerva E. Apostolidis, V. Mezaris, "SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets", Under review.
```bibtex
@misc{sdmvsum2026,
      title={"SD-MVSum: Script-Driven Multimodal Video Summarization Method and Datasets"}, 
      author={Manolis Mylonas and Charalampia Zerva and Evlampios Apostolidis and Vasileios Mezaris},
      year={2026},
      note={under review}
}
```

## License
This code is provided for academic, non-commercial use only. Please also check for any restrictions applied in the code parts and datasets used here from other sources. For the materials not covered by any such restrictions, redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution. 

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
