import torch


class ModelConfig:
    def __init__(
        self,
        Omega_max: float,
        Delta_0: float,
        singal_window: float,
        K: int,
        device: torch.device
        out_dir: str
    ):
        self.Omega_max = Omega_max
        self.Delta_0 = Delta_0
        self.K = K
        self.device = device
        self.out_dir = out_dir



class ControlModel(torch.nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        


    def generate_control_sequence(self, delta_vals: torch.Tensor, alpha_vals: torch.Tensor) -> torch.Tensor:
        x = self.build_feature_vector(delta_vals, alpha_vals)



    def build_feature_vector(self, delta_vals: torch.Tensor, alpha_vals: torch.Tensor) -> torch.Tensor:
        # delta_vals and alpha_vals are both shape (N,)
        assert delta_vals.shape == alpha_vals.shape
        pass

    def forward(x):
        pass


class 



def output_control_sequence
pass
