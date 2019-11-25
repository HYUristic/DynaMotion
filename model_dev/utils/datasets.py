from settings import Settings
from utils.google_drive import download_file_from_google_drive
from utils.midi import midi_to_numpy

import os
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
    def __init__(self, settings: Settings):
        dataset_id = 'Maestro'
        super().__init__(settings=settings, dataset_id=dataset_id, force_preprocess=False)
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass
        
    def get_root(self):
        root = os.path.join(self.dataset_path, "period_{}".format(self.settings.quantization_period))
        return root

    def get_meta_data_path(self):
        path = os.path.join(self.get_root(), 'meta.json')
        return path

    def get_meta_data(self):
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
        valid_path = os.path.join(root, 'valid')
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
        processed_meta_data = {"train": {}, "validation": {}, "test": {}}
        
        #TODO Update meta_data.keys() to full after development
        for midi_path in tqdm(list(meta_data.keys())[:20]):
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
            processed_meta_data[split_type][file_name] = int(quantized_piano_roll.shape[0])

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
    dataset = MaestroDataset(settings=settings)

