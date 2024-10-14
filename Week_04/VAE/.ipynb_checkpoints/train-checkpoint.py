import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from model.VAE import VAE, loss_function
from model.my_VAE import my_VAE, my_loss_function

# GPU 사용 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"사용 중인 디바이스: {device}")

# 데이터셋 준비 (Fashion MNIST)
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

model_version = 'my_vae' # Or, my_vae

if model_version == 'vae':
    # 모델 초기화 및 GPU로 이동
    model = VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 학습 루프
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)  # 데이터를 GPU로 이동
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')

    # 모델 저장
    torch.save(model.state_dict(), 'vae.pth')
    print("모델 저장 완료!")

elif model_version == 'my_vae':
    model = my_VAE().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 50
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, logvar = model(data)
            loss = my_loss_function(recon_batch, data, mu, logvar)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}')

    # 모델 저장
    torch.save(model.state_dict(), 'my_vae.pth')
    print("모델 저장 완료!")
        
else:
    raise ValueError
