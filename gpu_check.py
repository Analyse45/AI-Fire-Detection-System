import torch

print("PyTorch Version:", torch.__version__)

# Check CUDA availability
print("CUDA Available:", torch.cuda.is_available())

# GPU count
print("GPU Count:", torch.cuda.device_count())

# GPU name
if torch.cuda.is_available():

    print("GPU Name:", torch.cuda.get_device_name(0))

    print("Current Device:", torch.cuda.current_device())

else:

    print("Running on CPU ")