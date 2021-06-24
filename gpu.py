import torch
device = "cuda" if torch.cuda.is_available()  else "cpu"
print( torch.cuda.is_available() )

# 현재 Setup 되어있는 device 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

