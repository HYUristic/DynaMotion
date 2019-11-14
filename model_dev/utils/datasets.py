from model_dev.settings import Settings
from model_dev.utils.google_drive import download_file_from_google_drive

import os
import zipfile
from torch.utils.data import Dataset

class Dataset_Backbone(Dataset):
    def __init__(self, settings: Settings, dataset_id: str):
        self.settings = settings
        dataset_path = os.path.join(self.settings.dataset_root, self.settings.dataset_info[dataset_id]['path'])

        # Validate Dataset
        if not os.path.exists(self.settings.dataset_root):
            os.mkdir(self.settings.dataset_root)

        if not os.path.exists(dataset_path):
            link = self.settings.dataset_info[dataset_id]['link']
            print("Downloading '{dataset_id}' dataset from cloud... id:[{link}]".format(dataset_id=dataset_id, link=link))
            comp_file_name = self.download_dataset(dataset_path=self.settings.dataset_root, link=link)

            print("Unzipping...".format(dataset_id=dataset_id))
            with zipfile.ZipFile(comp_file_name, 'r') as zip_ref:
                zip_ref.extractall(self.settings.dataset_root)

            if os.path.exists(dataset_path):
                print("Successfully downloaded '{dataset_id}' dataset @ [{dataset_path}]".format(dataset_id=dataset_id, dataset_path=dataset_path))
                os.remove(comp_file_name)
            else:
                raise Exception('dataset_path does not match downloaded dataset.. please check settings')

    def __getitem__(self, item: int):
        raise NotImplementedError

    def __len__(self):
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

if __name__ == '__main__':
    settings = Settings()
    dataset = MaestroDataset(settings=settings)
