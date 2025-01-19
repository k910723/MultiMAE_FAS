import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ModalityDisentangleAdapter(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=64, num_modalities=3, filter_size=2):
        """
        Modality-Disentangle Adapter implementation.
        
        Args:
            input_dim (int): Dimension of input features
            hidden_dim (int): Dimension of adapter hidden layer (K in paper)
            num_modalities (int): Number of modalities (default 3 for RGB, Depth, IR)
            filter_size (int): Size of frequency filter kernel (f in paper)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_modalities = num_modalities
        self.filter_size = filter_size


        # Downsampling layer
        self.down = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, 1),
            nn.GELU()
        )

        # High frequency convolution branches (one per modality)
        self.hf_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
            for _ in range(num_modalities)
        ])

        # Low frequency convolution branches (one per modality)
        self.lf_convs = nn.ModuleList([
            nn.Conv2d(hidden_dim, hidden_dim, 5, padding=2)
            for _ in range(num_modalities)
        ])

        # Cross-modal fusion convolution
        self.fusion_conv = nn.Conv2d(hidden_dim * num_modalities, hidden_dim * num_modalities, 1)

        # Upsampling layer
        self.up = nn.Sequential(
            nn.Conv1d(hidden_dim, input_dim, 1),
            nn.GELU()
        )

    def get_frequency_masks(self, h, w):
        """Generate low and high frequency masks."""
        center_h = h // 2
        center_w = w // 2
        
        # Create meshgrid
        y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
        y = y.float().to(next(self.parameters()).device)
        x = x.float().to(next(self.parameters()).device)
        
        # Calculate distance from center
        y = y - center_h
        x = x - center_w
        distance = torch.sqrt(x**2 + y**2)
        
        # Create masks
        low_freq_mask = (distance <= self.filter_size).float()
        high_freq_mask = (distance > self.filter_size).float()
        
        return low_freq_mask, high_freq_mask

    def frequency_decomposition(self, x):
        """Decompose input into low and high frequency components using FFT."""
        # Apply FFT2D
        fft = torch.fft.fft2(x)
        fft_shifted = torch.fft.fftshift(fft)
        
        # Get frequency masks
        h, w = x.shape[-2:]
        low_mask, high_mask = self.get_frequency_masks(h, w)
        low_mask = low_mask.unsqueeze(0).unsqueeze(0)
        high_mask = high_mask.unsqueeze(0).unsqueeze(0)
        
        # Apply masks and inverse FFT
        low_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_shifted * low_mask))
        high_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_shifted * high_mask))
        
        return low_freq.real, high_freq.real

    def forward(self, x):
        """
        Forward pass through the adapter.
        
        Args:
            x (torch.Tensor): Input features [batch_size, seq_len, input_dim]
        
        Returns:
            torch.Tensor: Adapted features [batch_size, seq_len, input_dim]
        """
        B, N, C = x.shape
        
        # Downsample
        #print(x.shape)
        x = x.transpose(1, 2)  # [B, C, N]
        x = self.down(x)  # [B, K, N]
        
        # Reshape for 2D operations
        #print(x.shape)
        h = w = int(math.sqrt((N-1)/3)) # -1 global token and then /3 modalities
        global_token = x[:, :, -1]  # Extract global tokens
        # Extract modality-specific tokens
        modality_tokens = {}
        for i, modality in enumerate(['rgb', 'depth', 'ir']):
            start_idx = i * h * w
            end_idx = (i + 1) * h * w
            modality_tokens[modality] = x[:, :, start_idx:end_idx]

        x_rgb = modality_tokens['rgb'].view(B, self.hidden_dim, h, w)
        x_depth = modality_tokens['depth'].view(B, self.hidden_dim, h, w)
        x_ir = modality_tokens['ir'].view(B, self.hidden_dim, h, w)
        #print(x_rgb.shape)
        
        # Apply frequency decomposition
        low_freq, high_freq = {}, {}
        low_freq_rgb, high_freq_rgb = self.frequency_decomposition(x_rgb)
        low_freq_depth, high_freq_depth = self.frequency_decomposition(x_depth)
        low_freq_ir, high_freq_ir = self.frequency_decomposition(x_ir)
        low_freq['rgb'], high_freq['rgb'] = low_freq_rgb, high_freq_rgb
        low_freq['depth'], high_freq['depth'] = low_freq_depth, high_freq_depth
        low_freq['ir'], high_freq['ir'] = low_freq_ir, high_freq_ir
        
        # Process each modality
        modality_features = []
        for i, modality in enumerate(['rgb', 'depth', 'ir']):
            # Apply modality-specific convolutions
            high_freq_features = self.hf_convs[i](high_freq[modality])
            low_freq_features = self.lf_convs[i](low_freq[modality])
            
            # Combine frequency components
            modality_feature = high_freq_features + low_freq_features
            modality_features.append(modality_feature)
        #print(modality_features[0].shape)
        
        # Cross-modal fusion
        fused = self.fusion_conv(torch.cat(modality_features, dim=1)) # Concatenate along channel dimension
        fused_rgb = fused[:, :self.hidden_dim, :, :]
        fused_depth = fused[:, self.hidden_dim:2*self.hidden_dim, :, :]
        fused_ir = fused[:, 2*self.hidden_dim:, :, :]
        #print(fused.shape)
        
        # Reshape and upsample
        fused_rgb = fused_rgb.view(B, self.hidden_dim, h * w)
        fused_depth = fused_depth.view(B, self.hidden_dim, h * w)
        fused_ir = fused_ir.view(B, self.hidden_dim, h * w)
        fused = torch.cat([fused_rgb, fused_depth, fused_ir], dim=2)
        fused = torch.cat([fused, global_token.unsqueeze(2)], dim=2) # add back global token
        #print(fused.shape)
        output = self.up(fused)  # [B, C, N]
        
        return output.transpose(1, 2)  # [B, N, C]