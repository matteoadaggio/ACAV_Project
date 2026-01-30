import torch
import matplotlib.pyplot as plt

def get_mock_batch(batch_size=4, image_size=200, num_waypoints=10):
    """
    Simula un batch di dati proveniente dal DataLoader.
    
    Args:
        batch_size: Quanti esempi processiamo insieme.
        image_size: Risoluzione della BEV.
        num_waypoints: Quanti punti futuri vogliamo predire.
        
    Returns:
        inputs: Tensore [Batch, 3, H, W] -> La BEV a 3 canali.
        targets: Tensore [Batch, num_waypoints, 2] -> Le coordinate (x, y) vere.
    """
    
    # --- 1. CREAZIONE INPUT SIMULATO (BEV) ---
    # Canali: 0=Mappa, 1=Ego, 2=Ostacoli Futuri
    # Usiamo randn per generare rumore, ma nella realtà saranno 0 e 1.
    inputs = torch.randn(batch_size, 3, image_size, image_size)
    
    # --- 2. CREAZIONE TARGET SIMULATO (Traiettoria) ---
    # Simuliamo 10 punti (x, y). 
    # Supponiamo che l'auto sia al centro dell'immagine o in basso.
    # Per ora generiamo numeri casuali, ma idealmente sono coordinate in metri.
    targets = torch.randn(batch_size, num_waypoints, 2)
    
    return inputs, targets

# --- TEST VISIVO ---
if __name__ == "__main__":

    BATCH_SIZE = 2
    fake_inputs, fake_targets = get_mock_batch(batch_size=BATCH_SIZE)
    
    print(f"Input Shape: {fake_inputs.shape}")  
    print(f"Target Shape: {fake_targets.shape}")
    
    # Visualizziamo il primo esempio del batch
    first_image = fake_inputs[0] 
    
    # Per visualizzare con Matplotlib, dobbiamo cambiare l'ordine delle dimensioni:
    # Da [Canali, H, W] a [H, W, Canali]
    img_to_show = first_image.permute(1, 2, 0) 
    
    # Normalizziamo tra 0 e 1 per visualizzare (solo per il grafico)
    img_to_show = (img_to_show - img_to_show.min()) / (img_to_show.max() - img_to_show.min())

    plt.figure(figsize=(5,5))
    plt.title("Visualizzazione Mock BEV (Rumore Casuale)")
    plt.imshow(img_to_show.numpy())
    plt.axis('off')
    plt.show()
    