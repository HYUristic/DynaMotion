from model_dev.settings import Settings
from model_dev.utils.google_drive import download_file_from_google_drive

import os
from torch.utils.data import Dataset

class Dataset_Backbone(Dataset):
    def __init__(self, settings: Settings, dataset_id: str):
        self.settings = Settings
        dataset_path = os.path.join(self.settings.dataset_root, self.settings.dataset_info[dataset_id]['path'])

        # Validate Dataset
        if not os.path.exists(dataset_path):
            print("Downloading '{dataset_id}'dataset from cloud...".format(dataset_id=dataset_id))
            link = self.settings.dataset_info[dataset_id]['link']
            self.download_dataset(dataset_path=dataset_path, link=link)

    def __getitem__(self, item: int):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def download_dataset(self, dataset_path: str, link: str):
        #TODO Download via download_file_from_google_drive
        pass


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
