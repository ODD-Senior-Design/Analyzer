import os
import torch
import sys
sys.path.append( './src' )
from model import CNN

# ==== GLOBAL DEFAULTS ====
DEFAULT_INPUT_HEIGHT: int = 224
DEFAULT_INPUT_WIDTH: int = 224
DEFAULT_DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==== GLOBAL PATHS ====
NORMAL_MODEL_PATH: str = "./base_model/trial-1_calibrated_80_20_split_96_acc.pt"
TRACED_MODEL_PATH: str = "./base_model/trial-1_calibrated_80_20_split_96_acc_traced.pt"

def convert_to_traced_model(
    normal_model_path: str,
    traced_model_path: str,
    input_shape: tuple = (1, 3, DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH),
    device: str = DEFAULT_DEVICE
) -> None:
    model = CNN( device = torch.device( device ) )
    model.load_model( normal_model_path )

    example_input: torch.Tensor = torch.randn( *input_shape ).to( model.get_device() )
    model.eval()
    traced_model: torch.jit.ScriptModule = torch.jit.trace( model, example_input )

    os.makedirs( os.path.dirname( traced_model_path ), exist_ok = True )
    traced_model.save( traced_model_path )
    print( f"Traced model saved to: { traced_model_path }" )


def main() -> None:
    convert_to_traced_model(
        normal_model_path = NORMAL_MODEL_PATH,
        traced_model_path = TRACED_MODEL_PATH,
        input_shape = (1, 3, DEFAULT_INPUT_HEIGHT, DEFAULT_INPUT_WIDTH)
    )


if __name__ == "__main__":
    main()
