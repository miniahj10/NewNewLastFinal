from pyannote.audio import Pipeline
from transformers import Wav2Vec2ForCTC, Wav2Vec2Tokenizer
import torch
import librosa
import numpy as np
import os


def speaker_output(dlist):
    filtered = []
    i = 0
    dlist = remove_spaces(dlist)

    while i < len(dlist):
        speaker, text = dlist[i]
        combined_text = text

        # Keep combining while the next speaker is the same
        while i + 1 < len(dlist) and dlist[i + 1][0] == speaker:
            combined_text += " " + dlist[i + 1][1]
            i += 1

        sentence = speaker + " : " + combined_text
        filtered.append(sentence)
        i += 1  # Move to the next entry

    return filtered


def remove_spaces(dlist):
    filtered = []
    for i in range(len(dlist)):
        ele = dlist[i]
        if ele[1]:
            filtered.append(ele)

    return filtered


class Speaker_Diarisation:
    def __init__(self, audio_file):
        self.pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
                                                 use_auth_token="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")
        self.audio_file = audio_file
    def extract(self):
        diarization = self.pipeline(self.audio_file)
        # Load speech-to-text model and processor (e.g., wav2vec2)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")  # Wav2Vec2CTCTokenizer
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")  # 16000 Hz

        audio, sr = librosa.load(self.audio_file, sr=16000)
        min_length = 16000  # Adjust based on the sampling rate (sr)
        d = []
        # For each diarized segment, run speech-to-text
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start = int(turn.start * sr)  # Convert time to sample index
            end = int(turn.end * sr)

            # Extract segment of audio
            segment = audio[start:end]
            segment = np.array(segment).astype(np.float32)
            # segment = repr(segment)

            # Pad the segment if it's too short
            if len(segment) < min_length:
                pad_length = min_length - len(segment)
                # Pad with zeros (silence) at the end
                segment = np.pad(segment, (0, pad_length), 'constant')

            input_values = tokenizer(segment, return_tensors="pt", padding="longest").input_values
            logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.decode(predicted_ids[0])

            d.append([speaker, transcription])
        return d

    def conversation(self):
        combined = self.extract()
        return speaker_output(combined)

    def extract_with_speaker_id(self):
        diarization = self.pipeline(self.audio_file)
        # Load speech-to-text model and processor (e.g., wav2vec2)
        tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")  # Wav2Vec2CTCTokenizer
        model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")  # 16000 Hz

        audio, sr = librosa.load(self.audio_file, sr=16000)
        min_length = 16000  # Adjust based on the sampling rate (sr)
        d = []
        # For each diarized segment, run speech-to-text
        # for turn, _, speaker in diarization.itertracks(yield_label=True):
        for i, (turn, _, speaker) in enumerate(diarization.itertracks(yield_label=True)):
            start = int(turn.start * sr)  # Convert time to sample index
            end = int(turn.end * sr)
            segment = audio[start:end]
            segment_int16 = (segment * 32767).astype('int16')

            os.chdir('/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/user_wav_temp')
            # Save as WAV file
            filename = f"/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/user_wav_temp/speaker_segment_temp_{i}.wav"
            write(filename, sr, segment_int16)

            possible_speaker = Speaker_identification().check_given_voice(filename)
            os.remove(filename)

            segment = np.array(segment).astype(np.float32)

            # Pad the segment if it's too short
            if len(segment) < min_length:
                pad_length = min_length - len(segment)
                # Pad with zeros (silence) at the end
                segment = np.pad(segment, (0, pad_length), 'constant')

            input_values = tokenizer(segment, return_tensors="pt", padding="longest").input_values
            logits = model(input_values).logits

            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = tokenizer.decode(predicted_ids[0])

            # print([possible_speaker, transcription])
            d.append([possible_speaker, transcription])
        return d

    def conversation_with_speaker_id(self):
        combined = self.extract_with_speaker_id()
        return speaker_output(combined)


from resemblyzer import VoiceEncoder, preprocess_wav
from pathlib import Path
import pickle
from scipy.io.wavfile import write

class Speaker_identification:
    def __init__(self):
        self.encoder = VoiceEncoder()

    def store_user_voice(self, init_audio_file, user_name):
        reference_wav = preprocess_wav(Path(init_audio_file))
        reference_embed = self.encoder.embed_utterance(reference_wav)
        os.chdir('/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/user_voice_storage')
        with open(f"{user_name}_ref_embedding.pkl", "wb") as f:
            pickle.dump(reference_embed, f)
            return f'Embedding stored for {user_name}'

    def check_given_voice(self, diarised_output):
        segment_wav = preprocess_wav(Path(diarised_output))
        segment_embed = self.encoder.embed_utterance(segment_wav)

        main_dict = {}
        for file in os.listdir('/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/user_voice_storage'):
            os.chdir('/Users/jaiharishsatheshkumar/PycharmProjects/lastPhase1/user_voice_storage')

            user_name = file.split('_')[0]
            with open(file, "rb") as f:
                reference_embed = pickle.load(f)

            similarity = np.dot(reference_embed, segment_embed) / (
                    np.linalg.norm(reference_embed) * np.linalg.norm(segment_embed))
            main_dict[user_name] = similarity

        pred_user = max(main_dict, key=main_dict.get)
        return pred_user
