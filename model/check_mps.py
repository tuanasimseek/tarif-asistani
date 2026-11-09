import torch

# 1) PyTorch sürümü
print("PyTorch sürümü:", torch.__version__)

# 2) MPS (Apple GPU) kullanılabilir mi?
print("MPS kullanılabilir mi:", torch.backends.mps.is_available())

# 3) Cihaz seçimi: MPS varsa onu, yoksa CPU'yu kullan.
#    Projedeki her model kodunda bu satır olacak.
device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Seçilen cihaz:", device)

# 4) Seçilen cihazda iki rastgele 1000x1000 matris oluştur ve çarp
a = torch.randn(1000, 1000, device=device)
b = torch.randn(1000, 1000, device=device)
c = a @ b          # @ = matris çarpımı (torch.matmul ile aynı)

print("Sonuç boyutu:", c.shape)
print("Sonuç hangi cihazda:", c.device)