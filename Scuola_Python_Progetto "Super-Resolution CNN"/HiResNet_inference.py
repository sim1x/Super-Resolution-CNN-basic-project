import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


class HighResNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, padding=1)

        # 32x32 → 64x64 → 128x128
        self.upconv1 = nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)  # 32→64
        self.upconv2 = nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1)  # 64→128

        self.final_conv = nn.Conv2d(16, 3, 3, padding=1)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # (B, 64, 32, 32)
        x = torch.relu(self.conv2(x))  # (B, 64, 32, 32)
        x = torch.relu(self.conv3(x))
        x = torch.relu(self.upconv1(x))  # (B, 32, 64, 64)
        x = torch.relu(self.upconv2(x))  # (B, 16, 128, 128)
        x = self.final_conv(x)  # (B, 3, 128, 128)
        return torch.relu(x)


def load_model():
    #Carica il modello pre-addestrato
    net = HighResNet()

    checkpoint = torch.load('super_resolution_model.pth')
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    print("Modello caricato")
    print(f"Addestrato per {checkpoint['epoch']} epoche")
    print(f"Train Loss finale: {checkpoint['train_losses'][-1]:.6f}")
    print(f"Validation Loss finale: {checkpoint['val_losses'][-1]:.6f}")
    print(f"Test Loss finale (onesto): {checkpoint['test_loss']:.6f}")

    return net


def load_test_data():
    #Carica i dati di test (MAI visti durante training/validation)
    test_data = torch.load('test_data.pth')
    return test_data['HiRes_test'], test_data['LowRes_test']


def super_resolve_images(net, lowres_images):
    #Applica super-resolution alle immagini di test
    with torch.no_grad():
        outputs = net(lowres_images)
    return outputs


def plot_results_figure1(lowres_images, outputs, hires_images, start_idx=0, num_images=5):
    #Figura 1: Prime 5 immagini di test
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    fig.suptitle('Figura 1: Super-Resolution su Test Set (Immagini 1-5)\n[MAI viste durante training/validation]',
                 fontsize=16, fontweight='bold')

    for i in range(num_images):
        idx = start_idx + i

        # Input LowRes (32x32) - ingrandito per visualizzazione
        lowres_display = torch.nn.functional.interpolate(lowres_images[idx:idx + 1], size=(128, 128), mode='nearest' )[0]
        axes[i, 0].imshow(lowres_display.permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f'Input LowRes #{idx + 81} (32x32)')  # +81 perché test inizia da indice 80
        axes[i, 0].axis('off')

        # Output della rete (128x128)
        axes[i, 1].imshow(torch.clamp(outputs[idx].permute(1, 2, 0), 0, 1).numpy())
        axes[i, 1].set_title(f'Network Output #{idx + 81} (128x128)')
        axes[i, 1].axis('off')

        # Target HiRes (128x128)
        axes[i, 2].imshow(hires_images[idx].permute(1, 2, 0).numpy())
        axes[i, 2].set_title(f'Ground Truth #{idx + 81} (128x128)')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


def plot_results_figure2(lowres_images, outputs, hires_images, start_idx=5, num_images=5):
    #Figura 2: Seconde 5 immagini di test
    fig, axes = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))
    fig.suptitle('Figura 2: Super-Resolution su Test Set (Immagini 6-10)\n[MAI viste durante training/validation]',
                 fontsize=16, fontweight='bold')

    for i in range(num_images):
        idx = start_idx + i

        # Input LowRes (32x32) - ingrandito per visualizzazione
        lowres_display = torch.nn.functional.interpolate(lowres_images[idx:idx + 1], size=(128, 128), mode='nearest')[0]
        axes[i, 0].imshow(lowres_display.permute(1, 2, 0).numpy())
        axes[i, 0].set_title(f'Input LowRes #{idx + 81} (32x32)')  # +81 perché test inizia da indice 80
        axes[i, 0].axis('off')

        # Output della rete (128x128)
        axes[i, 1].imshow(torch.clamp(outputs[idx].permute(1, 2, 0), 0, 1).numpy())
        axes[i, 1].set_title(f'Network Output #{idx + 81} (128x128)')
        axes[i, 1].axis('off')

        # Target HiRes (128x128)
        axes[i, 2].imshow(hires_images[idx].permute(1, 2, 0).numpy())
        axes[i, 2].set_title(f'Ground Truth #{idx + 81} (128x128)')
        axes[i, 2].axis('off')

    plt.tight_layout()
    plt.show()


# ✅ MAIN SCRIPT
if __name__ == "__main__":
    print("Inferenza Super-Resolution CNN:")
    print("Test su immagini MAI viste durante training/validation\n")

    # Carico modello
    net = load_model()

    # Carico dati di test
    print(f"\nCaricamento dati di test...")
    HiRes_test, LowRes_test = load_test_data()
    print(f"Dati caricati: {len(HiRes_test)} immagini di test")

    # Applico super-resolution
    print(f"\nApplicazione super-resolution su test set...")
    outputs = super_resolve_images(net, LowRes_test)

    # Mostra risultati
    print(f"\nVisualizzazione risultati...")

    # Figura 1: Prime 5 immagini di test
    print("Figura 1: Immagini di test 1-5 (indici originali 80-84)")
    plot_results_figure1(LowRes_test, outputs, HiRes_test, start_idx=0, num_images=5)

    # Figura 2: Seconde 5 immagini di test
    print("Figura 2: Immagini di test 6-10 (indici originali 85-89)")
    plot_results_figure2(LowRes_test, outputs, HiRes_test, start_idx=5, num_images=5)

    print("Inferenza completata!")

