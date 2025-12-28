import torch
from pytorch_lightning import LightningModule
from matcha.models.components.text_encoder import TextEncoder
from matcha.models.components.decoder import Decoder

class MatchaTTS(LightningModule):
    def __init__(self, n_vocab, out_channels, hidden_channels):
        super().__init__()

        # Enregistre les hyperparamètres pour les logs
        self.save_hyperparameters()
        
        # Initialisation des composants définis dans votre protocole
        self.encoder = TextEncoder(
            n_vocab=n_vocab,
            out_channels=out_channels,
            hidden_channels=hidden_channels,
            filter_channels=768,
            n_heads=2,
            n_layers=6,
            kernel_size=3,
            p_dropout=0.1,
            duration_filter_channels=256,
            duration_kernel_size=3,
            duration_p_dropout=0.1
        )
        
        self.decoder = Decoder(
            in_channels=out_channels,      # 80
            hidden_channels=hidden_channels, # 192
            out_channels=out_channels      # 80
        )
        
    def generate_path(dw, mask):
        """
        Génère une matrice d'alignement monotone à partir de durées.

        Args:
            dw: Durées en échantillons. Forme [batch, text_len].
            mask: Masque pour le texte. Forme [batch, text_len].
        Returns:
            attn: Matrice d'alignement. Forme [batch, text_len, mel_len].
        """
        batch_size, text_len = dw.shape
        mel_len = torch.sum(dw, dim=1).long()  # Longueur totale du mel pour chaque exemple
        max_mel_len = mel_len.max().item()

        # Initialise la matrice d'alignement
        attn = torch.zeros(batch_size, text_len, max_mel_len, device=dw.device)

        # Pour chaque exemple du batch
        for b in range(batch_size):
            # Récupère les durées pour cet exemple
            d = dw[b].long()
            # Crée des indices pour chaque phonème
            for t in range(text_len):
                if d[t] > 0:  # Ignore les phonèmes masqués (durée = 0)
                    start = torch.sum(d[:t]).item()
                    end = start + d[t].item()
                    attn[b, t, start:end] = 1.0  # Alignement binaire

        return attn

    def forward(self, x, x_lengths, y=None, y_lengths=None):
        # 1. Encodage du texte -> h
        # 2. Prédiction des durées -> upsampling
        # 3. Calcul de mu (condition acoustique)
        pass

    def training_step(self, batch, batch_idx):
        x, x_lengths = batch["x"], batch["x_lengths"]
        y, y_lengths = batch["y"], batch["y_lengths"]

        # 1. Passe dans l'encodeur pour obtenir mu_x, x_mask et les log-durées
        mu_x, x_mask, durations = self.encoder(x, x_lengths)

        # 2. Convertit les log-durées en durées (en échantillons)
        w = durations * x_mask  # [batch, 1, text_len]
        w_ceil = w.squeeze(1)     # [batch, text_len]

        # 3. Génère le masque pour l'audio (y)
        y_max_length = y.shape[-1]
        y_mask = torch.arange(y_max_length, device=y.device)[None, :] < y_lengths[:, None]  # [batch, mel_len]
        y_mask = y_mask.unsqueeze(1)  # [batch, 1, mel_len]

        # 4. Génère l'alignement monotone
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)  # [batch, text_len, mel_len]
        attn = generate_path(w_ceil, attn_mask.squeeze(1))  # [batch, text_len, mel_len]

        # 5. Aligne mu_x avec les durées pour obtenir mu_y
        mu_y = torch.matmul(attn.transpose(1, 2), mu_x.transpose(1, 2))  # [batch, mel_len, n_feats]
        mu_y = mu_y.transpose(1, 2)  # [batch, n_feats, mel_len]

        # 2. Flow Matching : Temps t aléatoire
        t = torch.rand(y.shape[0], device=y.device)
        
        # 2.5 [CORRECTION CRITIQUE] Alignement Naïf (Upsampling)
        # On étire mu pour qu'il fasse la même taille que y (l'audio)
        # mu: [Batch, 80, Text_Len] -> [Batch, 80, Audio_Len]
        mu = torch.nn.functional.interpolate(mu, size=y.shape[-1], mode='nearest')
        
        # On doit aussi créer un masque pour l'audio (pour ignorer le padding audio)
        # On recrée un masque basé sur y_lengths au lieu d'utiliser x_mask (qui est pour le texte)
        y_mask = torch.arange(y.size(2), device=y.device)[None, :] < y_lengths[:, None]
        
        # 3. On crée le chemin (z0 -> y)
        z0 = torch.randn_like(y)
        zt = (1 - t.view(-1, 1, 1)) * z0 + t.view(-1, 1, 1) * y
        target = y - z0

        # 4. Le Décodeur
        # ATTENTION : on passe y_mask maintenant, car on travaille sur la dimension audio
        v_pred = self.decoder(zt, t, mu, y_mask)

        # 5. Loss
        loss = torch.mean((v_pred - target)**2)

        self.log("train_loss", loss, prog_bar=True, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)