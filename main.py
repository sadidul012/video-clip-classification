import av
import torch
import numpy as np
import tqdm

from transformers import AutoProcessor, AutoModel
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")
model.to(device)
labels = ["EMD pressing", "EMD contesting", "EMD in possession", "NJ pressing", "NJ contesting", "NJ in possession"]


def show(frame, wait=1000):
    # cv2.imshow('Frame', frame)
    # key = cv2.waitKey(wait)
    # # Press Q on keyboard to exit
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     return True
    #
    # return False
    plt.imshow(frame)
    plt.show(block=False)


def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)

    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)

    # end_idx = np.random.randint(converted_len, seg_len)
    indices_list = []
    for i in range(0, seg_len, converted_len):
        end_idx = converted_len + i
        start_idx = end_idx - converted_len
        indices = np.linspace(start_idx, end_idx, num=clip_len)
        indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
        indices_list.append(indices)
    return indices_list


if __name__ == '__main__':
    # video clip consists of 300 frames (10 seconds at 30 FPS)
    # file_path = hf_hub_download(
    #     repo_id="nielsr/video-demo", filename="eating_spaghetti.mp4", repo_type="dataset"
    # )
    file_path = "/mnt/Cache/downloads/Puck Possession Example Video.MOV"
    # file_path = "/mnt/Workspace/PycharmProjects/ice-hockey-insights/videos/12_31_21_EDM-NJD_P1.mp4"
    print(file_path)
    container = av.open(file_path)
    video = container.streams.video[0]

    print("frames", video.frames)
    print("frames", float(video.duration * video.time_base))
    print("framerate", video.frames / float(video.duration * video.time_base))

    # sample 8 frames
    indices_list = sample_frame_indices(clip_len=8, frame_sample_rate=5, seg_len=container.streams.video[0].frames)
    results = []
    for i in tqdm.tqdm(indices_list[:-2]):
        video = read_video_pyav(container, i)
        inputs = processor(text=labels, videos=list(video), return_tensors="pt", padding=True, )
        with torch.no_grad():
            for key in inputs.keys():
                inputs[key] = inputs[key].to(device)
            outputs = model(**inputs)
            logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
            probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
            label_id = probs.argmax().item()
            results.append([labels[label_id], probs[0][label_id].item(), i[0]])
            # logits_per_video = outputs.logits_per_video  # this is the video-text similarity score
            # probs = logits_per_video.softmax(dim=1)  # we can take the softmax to get the label probabilities
            # print("logits_per_video", probs)
            # print("label", labels[probs.argmax().item()])
            # print("logits_per_text", outputs.logits_per_text)

    container.close()

    results = pd.DataFrame(results, columns=["label", "confidence", "start_frame"])
    print(results)
    results.to_csv("output.csv", index=False)
