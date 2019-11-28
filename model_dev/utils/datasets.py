import os, sys
sys.path.append(os.path.abspath('..'))

from settings import Settings
from utils.google_drive import download_file_from_google_drive
from utils.midi import midi_to_numpy
from utils.util import search

import csv
import json
from tqdm import tqdm
import numpy as np
import shutil
import zipfile
from torch.utils.data import Dataset


class Dataset_Backbone(Dataset):
    def __init__(self, settings: Settings, dataset_id: str, force_preprocess=False):
        self.settings = settings
        self.dataset_path = os.path.join(self.settings.dataset_root, self.settings.dataset_info[dataset_id]['path'])

        # Validate Dataset
        if not os.path.exists(self.settings.dataset_root):
            os.mkdir(self.settings.dataset_root)

        if not os.path.exists(self.dataset_path):
            link = self.settings.dataset_info[dataset_id]['link']
            print("Downloading '{dataset_id}' dataset from cloud... id:[{link}]".format(dataset_id=dataset_id, link=link))
            comp_file_name = self.download_dataset(dataset_path=self.settings.dataset_root, link=link)

            print("Unzipping...".format(dataset_id=dataset_id))
            with zipfile.ZipFile(comp_file_name, 'r') as zip_ref:
                zip_ref.extractall(self.settings.dataset_root)

            if os.path.exists(self.dataset_path):
                print("Successfully downloaded '{dataset_id}' dataset @ [{dataset_path}]".format(dataset_id=dataset_id, dataset_path=self.dataset_path))
                os.remove(comp_file_name)
            else:
                raise Exception('dataset_path does not match downloaded dataset.. please check settings')

        # Preprocess Midi
        if self.need_preprocess() or force_preprocess:
            self.preprocess_midi(quantization_period=self.settings.quantization_period)

    def __getitem__(self, item: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def preprocess_midi(self, quantization_period: float):
        """
        1. Split Midi file based on sample frequency
        2. Generate Train/Validation/Test index file
        :param frequency: Sample Frequency in ms (ex: 1000 -> each vector is 1second)
        """
        raise NotImplementedError

    def need_preprocess(self):
        raise NotImplementedError

    def download_dataset(self, dataset_path: str, link: str):
        file_name = os.path.join(dataset_path, 'compressed.zip')
        download_file_from_google_drive(id=link, destination=file_name)
        return file_name


class MaestroDataset(Dataset_Backbone):
    def __init__(self, split_mode, settings: Settings):
        dataset_id = 'Maestro'
        self.split_mode = split_mode # train / validation / test
        super().__init__(settings=settings, dataset_id=dataset_id, force_preprocess=False)
        
        self.meta_chunk = self.ticks2chunk(meta_ticks=self.get_meta_ticks())[split_mode]

    def __getitem__(self, global_index):
        assert global_index>=0 and global_index<self.__len__(), "invalid index"
        file_index, file_name, local_index = self.get_local_index(global_index)
        piano_roll_file_name = os.path.join(self.get_root(), self.split_mode, "piano_roll", file_name)
        velocity_file_name = os.path.join(self.get_root(), self.split_mode, "velocity", file_name)
        
        # Load file
        piano_roll = np.load(piano_roll_file_name)
        velocity = np.load(velocity_file_name)

        # Slice np_array based in local_index
        start_index = local_index * self.settings.length
        end_index = start_index + self.settings.length # exclusive indexing

        piano_roll = piano_roll[start_index: end_index]
        velocity = velocity[start_index: end_index]

        return piano_roll, velocity

    def __len__(self):
        return self.meta_chunk[-1]["length"]
        
    def get_root(self):
        root = os.path.join(self.dataset_path, "period_{}".format(self.settings.quantization_period))
        return root

    def get_local_index(self, global_index):
        file_index = search(self.meta_chunk, global_index)
        file_name = self.meta_chunk[file_index]["name"]
        if file_index !=0:
            local_index = global_index - self.meta_chunk[file_index-1]["length"] 
        else:
            local_index = global_index
        return file_index, file_name, local_index

    def ticks2chunk(self, meta_ticks):
        # meta data in ticks to chunks based on settings.length
        meta_chunk = {}
        for key in meta_ticks.keys():
            meta_chunk[key] = []

        for key in meta_ticks.keys():
            chunk_sum = 0
            for obj in meta_ticks[key]:
                chunk_sum += int(obj["length"] / self.settings.length)
                temp = {}
                temp["name"] = obj["name"]
                temp["length"] = chunk_sum
                meta_chunk[key].append(temp)

        return meta_chunk

    def get_meta_data_path(self):
        path = os.path.join(self.get_root(), 'meta.json')
        return path

    def get_meta_ticks(self):
        # Read meta data in ticks for meta data path
        with open(self.get_meta_data_path()) as meta:
            data = json.load(meta)
        return data

    def need_preprocess(self):
        root = self.get_root()
        if os.path.exists(root):
            return False
        else:
            return True

    def preprocess_midi(self, quantization_period: float):
        # make train/valid dir to store augmented midi
        root = self.get_root()
        if os.path.exists(root):
            shutil.rmtree(root) 
        os.mkdir(root)
        train_path = os.path.join(root, 'train')
        valid_path = os.path.join(root, 'validation')
        test_path = os.path.join(root, 'test')
        paths = [train_path, valid_path, test_path]
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        if os.path.exists(valid_path):
            shutil.rmtree(valid_path)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        for path in paths:
            os.mkdir(path)
            os.mkdir(os.path.join(path, 'piano_roll'))
            os.mkdir(os.path.join(path, 'velocity'))

        print('Start Data Preprocessing...')
        print('Save_path: {}'.format(root))

        # parse csv meta data
        csv_file_name = '{}.csv'.format(self.dataset_path.split('/')[-1])
        csv_file_path = os.path.join(self.dataset_path, csv_file_name)

        meta_data_label = []
        meta_data = {}

        with open(csv_file_path) as csv_file:
            reader = csv.reader(csv_file, delimiter=',')
            for idx, row in enumerate(reader):
                if idx == 0:
                    meta_data_label = row
                else:
                    data = {}
                    for idx, column in enumerate(row):
                        data[meta_data_label[idx]] = column
                    meta_data[data['midi_filename']] = {'split': data['split']}
        
        # augment midi
        processed_meta_data = {"train": [], "validation": [],  "test": []}
        
        #TODO Update meta_data.keys() to full after development
        for midi_path in tqdm(meta_data.keys()):
            file_name = midi_path.split('/')[1]
            split_type = meta_data[midi_path]['split']  # train/validation/test
            if split_type == 'train':
                save_root = train_path
            elif split_type == 'validation':
                save_root = valid_path
            elif split_type == 'test':
                save_root = test_path
            
            quantized_piano_roll, quantized_velocity = midi_to_numpy(midi_path=os.path.join(self.dataset_path, midi_path), quantization_period=quantization_period)  # quantized midi in list(numpy_array)
            
            # Update Meta Data
            processed_meta_data[split_type].append({'name': "{}.npy".format(file_name), 'length': int(quantized_piano_roll.shape[0])})

            # Save Numpy Array
            piano_save_path = os.path.join(save_root, "piano_roll", "{file_name}.npy".format(file_name=file_name))
            velocity_save_path = os.path.join(save_root, "velocity", "{file_name}.npy".format(file_name=file_name))
            np.save(piano_save_path, quantized_piano_roll)
            np.save(velocity_save_path, quantized_velocity)
       
        # Save Meta Data
        with open(self.get_meta_data_path(), 'w') as meta_data_file:
            json.dump(processed_meta_data, meta_data_file)
        print('finished pre-processing...')


if __name__ == '__main__':
    settings = Settings()
    dataset = MaestroDataset(split_mode="train", settings=settings)
    piano_roll, velocity = dataset[0]
