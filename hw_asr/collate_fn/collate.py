import logging
import torch
from typing import List

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {}

    batch_audios = torch.zeros(len(dataset_items), max([dataset_items[i]['audio'].shape[1] for i in range(len(dataset_items))]))
    for i in range(len(dataset_items)):
        batch_audios[i,:dataset_items[i]['audio'].shape[1]] = dataset_items[i]['audio'].squeeze()

    batch_spectrograms = torch.zeros(
        len(dataset_items), 
        dataset_items[0]['spectrogram'].shape[1], 
        max([dataset_items[i]['spectrogram'].shape[-1] for i in range(len(dataset_items))])
    )
    for i in range(len(dataset_items)):
        batch_spectrograms[i,:,:dataset_items[i]['spectrogram'].shape[-1]] = dataset_items[i]['spectrogram'].squeeze()
    
    batch_text_encoded = torch.zeros(len(dataset_items), max([dataset_items[i]['text_encoded'].shape[1] for i in range(len(dataset_items))]))
    for i in range(len(dataset_items)):
        batch_text_encoded[i,:dataset_items[i]['text_encoded'].shape[1]] = dataset_items[i]['text_encoded'].squeeze()
    batch_text_encoded = batch_text_encoded.to(dtype=torch.int32)

    batch_text_encoded_length = torch.tensor([dataset_items[i]['text_encoded'].shape[1] for i in range(len(dataset_items))], dtype=torch.int32)

    batch_spectrogram_length = torch.tensor([dataset_items[i]['spectrogram'].shape[-1] for i in range(len(dataset_items))], dtype=torch.int32)

    batch_text = [dataset_items[i]['text'] for i in range(len(dataset_items))]
    batch_duration = [dataset_items[i]['duration'] for i in range(len(dataset_items))]
    batch_audio_path = [dataset_items[i]['audio_path'] for i in range(len(dataset_items))]

    result_batch = {
        "audio": batch_audios,
        "spectrogram": batch_spectrograms,
        "text_encoded": batch_text_encoded,
        "text_encoded_length": batch_text_encoded_length,
        "spectrogram_length": batch_spectrogram_length,
        "text": batch_text,
        "duration": batch_duration,
        "audio_path": batch_audio_path
    }
    
    return result_batch