import torch
import torch.nn as nn
from torchvision.models import resnet18

class NeuralPlanner(nn.Module):
    def __init__(self, in_channels=3, num_waypoints=10):
        """
        Inizializza il modello.
        Args:
            in_channels: Numero di canali dell'immagine BEV (Mappa, Ego, Predizione).
            num_waypoints: Quanti punti futuri vogliamo predire.
        """
        super(NeuralPlanner, self).__init__()
        
        # 1. Carichiamo la Backbone
        # Usiamo ResNet18. weights=None (pesi casuali)
        # imparerà tutto da zero guardando le mappe.
        self.backbone = resnet18(weights=None)
        
        # 2. Modifica dell'Input
        # La ResNet originale si aspetta foto RGB (3 canali). 
        # riscriviamo questo layer per renderlo flessibile.
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 
            64,            
            kernel_size=7,  
            stride=2, 
            padding=3, 
            bias=False
        )
        
        # 3. Modifica dell'Output
        # Dobbiamo arrivare ad un livello che restituisce 10 coordinate.
        
        num_features = self.backbone.fc.in_features
        
        # Sostituiamo il layer 'fc' (Fully Connected) originale
        self.backbone.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),                    
            nn.Dropout(0.3),              
            nn.Linear(512, num_waypoints * 2)
        )
        
        self.num_waypoints = num_waypoints

    def forward(self, x):
        """
        Forward Pass.
        l'input attraversa i layer ed esce la predizione.
        """
        # x shape in ingresso: [Batch, Canali, H, W]
        x = self.backbone(x)
        
        # x ha shape: [Batch, num_waypoints * 2]
        
        # Reshape:
        # -1 non modifica batch size
        # self.num_waypoints è il numero di punti (10)
        # 2 sono le coordinate (x, y)
        return x.view(-1, self.num_waypoints, 2)

# --- BLOCCO DI TEST ---
if __name__ == "__main__":
    # 1. Creiamo il modello
    model = NeuralPlanner(in_channels=3, num_waypoints=10)
    print("Modello creato")
    
    # 2. Creiamo un dato finto (Batch=2)
    dummy_input = torch.randn(2, 3, 200, 200)
    
    # 3. Proviamo a fargli fare una predizione
    output = model(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    # Verifica dimensionale
    if output.shape == (2, 10, 2):
        print("Le dimensioni sono corrette.")
    else:
        print("Errore: Le dimensioni non sono corrette!")