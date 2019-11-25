from settings import Settings
from utils.google_drive import download_file_from_google_drive
from utils.midi import midi_to_numpy

import os
import csv
import shutil
import zipfile
from torch.utils.data import Dataset


class Dataset_Backbone(Dataset):
    def __init__(self, settings: Settings, dataset_id: str):
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
        if self.need_preprocess():
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
        super().__init__(settings=settings, dataset_id=dataset_id)
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

    def need_preprocess(self):
        # TODO Check if Preprocessing is needed
        return True

    def preprocess_midi(self, quantization_period: float):
        # TODO Generate dataset_info.json and store current quantization period

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
                    meta_data[data['midi_filename']] = {'split': data['split'], 'duration': data['duration']}

        # make train/valid dir to store augmented midi
        train_path = os.path.join(self.dataset_path, 'train')
        valid_path = os.path.join(self.dataset_path, 'valid')
        test_path = os.path.join(self.dataset_path, 'test')
        if os.path.exists(train_path):
            shutil.rmtree(train_path)
        if os.path.exists(valid_path):
            shutil.rmtree(valid_path)
        if os.path.exists(test_path):
            shutil.rmtree(test_path)
        os.mkdir(os.path.join(self.dataset_path, 'train'))
        os.mkdir(os.path.join(self.dataset_path, 'valid'))
        os.mkdir(os.path.join(self.dataset_path, 'test'))

        # augment midi
        for midi_path in meta_data.keys():
            file_name = midi_path.split('/')[1]
            split_type = meta_data[midi_path]['split']  # train/validation/test
            if split_type == 'train':
                save_root = os.path.join(train_path, file_name)
            elif split_type == 'valid':
                save_root = os.path.join(valid_path, file_name)
            elif split_type == 'test':
                save_root = os.path.join(test_path, file_name)
            
            quantized_piano_roll, quantized_velocity = midi_to_numpy(midi_path=os.path.join(self.dataset_path, midi_path), quantization_period=quantization_period)  # quantized midi in list(numpy_array)
            length_tick = quantized_piano_roll.shape[0]  # number of quantized samples generated

            # TODO Save Numpy Array and save information
            pass



if __name__ == '__main__':
    settings = Settings()
    dataset = MaestroDataset(settings=settings)
