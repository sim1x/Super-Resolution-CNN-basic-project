import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from skimage.transform import resize


class HighResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        
        self.upconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 32-64
        self.upconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 64-128

        self.final_conv = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # (B, 64, 32, 32)
        x = torch.relu(self.conv2(x))  # (B, 64, 32, 32)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.upconv1(x))  # (B, 32, 64, 64)
        x = torch.relu(self.upconv2(x))  # (B, 16, 128, 128)
        x = self.final_conv(x)  # (B, 3, 128, 128)
        return torch.relu(x) #perchè le immagini sono già normalizzate tra 0 e 1


def load_stl10_dataset(num_images=100):
    #Carico dataset STL-10
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    stl10 = datasets.STL10(root='./data', split='unlabeled', download=True, transform=transform)

    subset_images = []
    count = 0

    for img, _ in stl10:
        if count >= num_images:
            break

        img_np = img.permute(1, 2, 0).numpy()
        subset_images.append(img_np)
        count += 1

    print(f"Caricate {len(subset_images)} immagini da STL-10")
    return np.array(subset_images)


def create_dataset():

    images = load_stl10_dataset(100)

    HiRes = []
    LowRes = []

    for img in images:
        # HiRes: 128x128
        hi_res = img
        # LowRes: 32x32 (4x downscale)
        low_res = resize(img, (32, 32), anti_aliasing=True)

        HiRes.append(hi_res)
        LowRes.append(low_res)

    HiRes = np.array(HiRes)
    LowRes = np.array(LowRes)

    # SPLIT: 60% Train / 20% Validation / 20% Test
    train_split = 0.6
    val_split = 0.8

    train_size = int(len(HiRes) * train_split)  # 60 immagini
    val_size = int(len(HiRes) * val_split)  # 80 immagini totali

    # TRAINING SET: 0-59 (60 immagini)
    HiRes_train = HiRes[:train_size]
    LowRes_train = LowRes[:train_size]

    # VALIDATION SET: 60-79 (20 immagini)
    HiRes_val = HiRes[train_size:val_size]
    LowRes_val = LowRes[train_size:val_size]

    # TEST SET: 80-99 (20 immagini)
    HiRes_test = HiRes[val_size:]
    LowRes_test = LowRes[val_size:]

    return (HiRes_train, LowRes_train, HiRes_val, LowRes_val, HiRes_test, LowRes_test)


def convert_to_tensors(HiRes_train, LowRes_train, HiRes_val, LowRes_val, HiRes_test, LowRes_test):
    # Converto arrays numpy in tensori PyTorch
    HiRes_train = torch.tensor(HiRes_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    LowRes_train = torch.tensor(LowRes_train.transpose(0, 3, 1, 2), dtype=torch.float32)
    HiRes_val = torch.tensor(HiRes_val.transpose(0, 3, 1, 2), dtype=torch.float32)
    LowRes_val = torch.tensor(LowRes_val.transpose(0, 3, 1, 2), dtype=torch.float32)
    HiRes_test = torch.tensor(HiRes_test.transpose(0, 3, 1, 2), dtype=torch.float32)
    LowRes_test = torch.tensor(LowRes_test.transpose(0, 3, 1, 2), dtype=torch.float32)

    return HiRes_train, LowRes_train, HiRes_val, LowRes_val, HiRes_test, LowRes_test


def train_model(net, criterion, optimizer, HiRes_train, LowRes_train, HiRes_val, LowRes_val, num_epochs=100):

    train_losses = []
    val_losses = []

    print(f"\nInizio training per {num_epochs} epoche:")

    for epoch in range(num_epochs):
        #TRAINING: Solo su training set (immagini 0-59)
        net.train()
        train_loss = 0

        for i in range(len(HiRes_train)):
            optimizer.zero_grad()

            input_img = LowRes_train[i:i + 1]
            target_img = HiRes_train[i:i + 1]

            output = net(input_img)
            loss = criterion(output, target_img)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        #VALIDATION: Solo su validation set (immagini 60-79)
        net.eval()
        val_loss = 0
        with torch.no_grad():
            for i in range(len(HiRes_val)):
                input_img = LowRes_val[i:i + 1]
                target_img = HiRes_val[i:i + 1]
                output = net(input_img)
                loss = criterion(output, target_img)
                val_loss += loss.item()

        train_loss /= len(HiRes_train)
        val_loss /= len(HiRes_val)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if (epoch + 1) % 10 == 0: #stampo loss e val_loss ogni 10 epoche
            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')

    return train_losses, val_losses


def plot_training_progress(train_losses, val_losses):
    #Plot delle curve di training
    plt.figure(figsize=(8, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Validation Loss', color='orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_training_results(net, HiRes_val, LowRes_val):
    #Plot dei risultati su validation set """
    net.eval()
    with torch.no_grad():
        val_outputs = net(LowRes_val[:3])  # Prime 3 immagini di validation

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.suptitle('Risultati su Validation Set (durante training)\n[Immagini 61-63]',
                 fontsize=16, fontweight='bold')

    for i in range(3):
        # Input LowRes
        lowres_display = torch.nn.functional.interpolate(LowRes_val[i:i + 1], size=(128, 128), mode='nearest')[0]
        #potrei anche non interpolare e visualizzare direttamente l'immagine
        axes[i, 0].imshow(lowres_display.permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f'Input LowRes #{i + 61} (32x32)')
        axes[i, 0].axis('off')

        # Network Output
        axes[i, 1].imshow(torch.clamp(val_outputs[i].permute(1, 2, 0), 0, 1).numpy())
        axes[i, 1].set_title(f'Network Output #{i + 61} (128x128)')
        axes[i, 1].axis('off')

        # Ground Truth
        axes[i, 2].imshow(HiRes_val[i].permute(1, 2, 0).numpy())
        axes[i, 2].set_title(f'Ground Truth #{i + 61} (128x128)')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def final_test_evaluation(net, criterion, HiRes_test, LowRes_test):
    #Valutazione finale
    print(f"\n Valutazione finale su test set (MAI visto durante training/validation)...")

    net.eval()
    test_loss = 0
    with torch.no_grad():
        for i in range(len(HiRes_test)):
            input_img = LowRes_test[i:i + 1]
            target_img = HiRes_test[i:i + 1]
            output = net(input_img)
            loss = criterion(output, target_img)
            test_loss += loss.item()

    test_loss /= len(HiRes_test)
    print(f"Test Loss finale (valutazione onesta): {test_loss:.6f}")
    return test_loss


def save_model_and_data(net, optimizer, train_losses, val_losses, test_loss, HiRes_test, LowRes_test):
    #Salva modello e dati
    print("\n Salvataggio modello e dati...")

    # Salva modello
    torch.save({
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
        'test_loss': test_loss,
        'epoch': len(train_losses)
    }, 'super_resolution_model.pth')

    # Salva dati di test
    torch.save({
        'HiRes_test': HiRes_test,
        'LowRes_test': LowRes_test
    }, 'test_data.pth')

    print("Modello salvato come 'super_resolution_model.pth'")
    print("Dati test salvati come 'test_data.pth'")


# MAIN SCRIPT
if __name__ == "__main__":
    print("Training di una rete CNN 'Super-Resolution' ")

    # Crea dataset con split corretto
    HiRes_train, LowRes_train, HiRes_val, LowRes_val, HiRes_test, LowRes_test = create_dataset()
    #fig,axes = plt.subplots(1,2)
    #axes[0].imshow(LowRes_train[0])
    #axes[1].imshow(HiRes_train[0])
    #plt.show()

    # Converti in tensori
    HiRes_train, LowRes_train, HiRes_val, LowRes_val, HiRes_test, LowRes_test = convert_to_tensors(
        HiRes_train, LowRes_train, HiRes_val, LowRes_val, HiRes_test, LowRes_test)

    # Setup modello
    net = HighResNet()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training
    train_losses, val_losses = train_model(
        net, criterion, optimizer, HiRes_train, LowRes_train, HiRes_val, LowRes_val, num_epochs=100
    )

    # Plot training progress
    print("\nVisualizzazione progresso training...")
    plot_training_progress(train_losses, val_losses)

    # Plot risultati su validation set
    print("\nVisualizzazione risultati su validation set...")
    plot_training_results(net, HiRes_val, LowRes_val)

    # Valutazione finale
    test_loss = final_test_evaluation(net, criterion, HiRes_test, LowRes_test)

    # Salvo modello e dati di test
    save_model_and_data(net, optimizer, train_losses, val_losses, test_loss, HiRes_test, LowRes_test)

    print(f"\n Training completato!")
    print(f"Risultati finali:")
    print(f"   - Train Loss finale: {train_losses[-1]:.6f}")
    print(f"   - Validation Loss finale: {val_losses[-1]:.6f}")
    print(f"   - Test Loss finale (onesto): {test_loss:.6f}")

