import os


class Settings:
    root = os.path.dirname(os.path.realpath(__file__))

    # Dataset Settings
    dataset_root = os.path.join(root, '../../DataHub')
    dataset_info = {
        'Maestro': {
            'path': './maestro-v2.0.0',
            'link': '1BKfsI61Y9kZj1RETksJDUhr4UqAuh9VQ'
        }
    }
    quantization_period = 1/16
    length = 64 # 4 bars in 1/16 period
