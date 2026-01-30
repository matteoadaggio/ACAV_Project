import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

from model import NeuralPlanner
from mock_data_gen import get_mock_batch

def train():
    
    # Configurazione dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    #Creazione del modello
    model = NeuralPlanner(in_channels=3, num_waypoints=10).to(device)
    
    model.train()
    
    #Parametri di training
    LR = 1e-4
    EPOCHS = 100
    BATCH_SIZE = 8
    
    # Definizione dell'ottimizzatore
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Generiamo UN SOLO batch e usiamo sempre quello.
    # Se la rete funziona, deve riuscire ad arrivare a Loss -> 0.
    inputs, targets = get_mock_batch(batch_size=BATCH_SIZE)
    
    # Spostiamo i dati sul dispositivo
    inputs = inputs.to(device)
    targets = targets.to(device)

    print("Inizio Training...")
    
    loss_history = []

    # 5. IL LOOP DI TRAINING
    for epoch in range(EPOCHS):
        
        # A. Zero Gradients
        # Puliamo i gradienti vecchi prima di calcolare quelli nuovi
        optimizer.zero_grad()
        
        # B. Forward Pass
        # Chiediamo al modello di predire
        predictions = model(inputs)
        
        # C. Calcolo Loss
        loss = criterion(predictions, targets)
        
        # D. Backward Pass
        # Calcola i gradienti
        loss.backward()
        
        # E. Optimizer Step
        # Aggiorna effettivamente i pesi
        optimizer.step()
        
        # Salviamo la loss per il grafico
        loss_history.append(loss.item())
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{EPOCHS}] | Loss: {loss.item():.6f}")

    print("Training completato!")

    # 6. VISUALIZZAZIONE DELLA LOSS
    plt.plot(loss_history)
    plt.title("Curva di Apprendimento (Loss)")
    plt.xlabel("Epoche")
    plt.ylabel("Errore (MSE)")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train()