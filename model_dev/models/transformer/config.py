
class Config:
    def __init__(self):
        self.input_length = 128  # Length of Sequence
        self.input_dim = 128  # Number of Piano Roll
        self.d_model = 512
        self.d_inner = 2048

        # Multi Head Attention
        self.n_layer = 6
        self.n_head = 8
        self.d_k = 64
        self.d_v = 64
