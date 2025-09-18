import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import os

def matplotlib_to_opencv(fig_or_plt):
    """
    Converte uma figura ou plot do Matplotlib para uma imagem compatível com OpenCV.

    :param fig_or_plt: Um objeto `matplotlib.figure.Figure` ou `matplotlib.pyplot`.
    :return: Imagem no formato BGR (compatível com OpenCV).
    """
    # Verifica se o parâmetro é um objeto Figure
    if hasattr(fig_or_plt, "canvas"):
        fig = fig_or_plt  # É uma figura (Figure)
    else:
        fig = plt.gcf()  # Usa a figura atual do pyplot

    # Renderiza a figura em uma imagem NumPy
    fig.canvas.draw()  # Renderiza o gráfico
    img_rgb = np.array(fig.canvas.renderer.buffer_rgba())  # Converte para array RGBA

    # Fecha a figura para liberar memória (opcional, depende do uso)
    plt.close(fig)

    # Converte de RGBA para BGR (formato do OpenCV)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGBA2BGR)

    return img_bgr




class VideoDataset:
    def __init__(self, video_path, device="cpu"):
        """
        Inicializa o dataset com o caminho do vídeo.
        
        :param video_path: Caminho para o arquivo de vídeo.
        :param device: Dispositivo PyTorch ("cpu" ou "cuda").
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Não foi possível abrir o vídeo: {video_path}")
        
        # Obtém informações sobre o vídeo
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        
        # Estado interno para rastrear o frame atual
        self.current_frame_idx = 0

        # Configurações adicionais
        self.device = torch.device(device)

    def __len__(self):
        """
        Retorna o número total de frames no vídeo.
        """
        return self.total_frames

    def rgb_to_yiq(self, rgb_tensor):
        """
        Converts an RGB image or batch of images to YIQ color space.

        :param rgb_tensor: Tensor of shape [3, H, W] or [B, 3, H, W], float in [0, 1]
        :return: Tensor in same shape with YIQ values
        """
        if not isinstance(rgb_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")
        if rgb_tensor.dim() not in [3, 4]:
            raise ValueError("Input must have shape [3,H,W] or [B,3,H,W].")

        is_batched = rgb_tensor.dim() == 4
        device = rgb_tensor.device

        # RGB → YIQ conversion matrix
        yiq_from_rgb = torch.tensor(
            [[0.299,  0.587,  0.114],
            [0.596, -0.274, -0.322],
            [0.211, -0.523,  0.312]],
            dtype=torch.float32,
            device=device
        )

        if not is_batched:
            rgb_tensor = rgb_tensor.unsqueeze(0)  # Add batch dimension

        # Correct einsum: apply linear transformation across channels
        yiq_tensor = torch.einsum('ij,bjhw->bihw', yiq_from_rgb, rgb_tensor)

        return yiq_tensor if is_batched else yiq_tensor.squeeze(0)

    def yiq_to_rgb(self, yiq_tensor):
        """
        Converts a YIQ tensor [3, H, W] or [B, 3, H, W] to RGB.
        
        :param yiq_tensor: PyTorch tensor of shape [3,H,W] or [B,3,H,W], float in [0,1]
        :return: Tensor in the same shape as input with RGB values, float in [0,1]
        """
        if not isinstance(yiq_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")

        if yiq_tensor.dim() not in [3, 4]:
            raise ValueError("Input must have 3 or 4 dimensions: [3,H,W] or [B,3,H,W].")

        is_batched = yiq_tensor.dim() == 4
        device = yiq_tensor.device

        # Conversion matrix: YIQ → RGB
        rgb_from_yiq = torch.tensor(
            [[1.0,  0.956,  0.621],
            [1.0, -0.272, -0.647],
            [1.0, -1.106,  1.703]],
            dtype=torch.float32,
            device=device
        )

        if not is_batched:
            yiq_tensor = yiq_tensor.unsqueeze(0)  # Add batch dimension → [1, 3, H, W]

        B, C, H, W = yiq_tensor.shape

        # Reshape tensor to [B, H, W, C] for matmul
        yiq_tensor = yiq_tensor.permute(0, 2, 3, 1)  # → [B, H, W, 3]

        # Apply matrix multiplication: [B, H, W, 3] @ [3,3] → [B, H, W, 3]
        rgb_tensor = torch.matmul(yiq_tensor, rgb_from_yiq.T)

        # Restore original shape: [B, 3, H, W]
        rgb_tensor = rgb_tensor.permute(0, 3, 1, 2)

        # Clamp values to valid range [0,1]
        rgb_tensor = rgb_tensor.clamp(0.0, 1.0)

        return rgb_tensor if is_batched else rgb_tensor.squeeze(0)
        
    def get_frame(self, frame_idx, scale_factor=1.0, to_torch=True):
        """
        Obtém um frame específico pelo índice.
        
        :param frame_idx: Índice do frame desejado.
        :param scale_factor: Fator de escala para redimensionar os frames.
        :param only_y: Se True, retorna apenas o canal Y em YIQ.
        :param to_torch: Se True, retorna o frame como um tensor PyTorch.
        :return: Frame processado.
        """
        if frame_idx < 0 or frame_idx >= self.total_frames:
            raise IndexError(f"Índice de frame inválido: {frame_idx}. O vídeo tem {self.total_frames} frames.")
        
        # Move o cursor para o frame desejado
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError(f"Falha ao ler o frame {frame_idx}.")

        # Redimensiona o frame se necessário
        if scale_factor != 1.0:
            new_width = int(self.frame_width * scale_factor)
            new_height = int(self.frame_height * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height))

        # Processa o frame conforme os parâmetros
        if to_torch:
            frame = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)  # [C, H, W]

        return frame

    def get_next_batch(self, batch_size, scale_factor=1.0, only_y=False, to_torch=True):
        """
        Obtém o próximo batch de frames sequenciais.
        
        :param batch_size: Número de frames no batch.
        :param scale_factor: Fator de escala para redimensionar os frames.
        :param only_y: Se True, retorna apenas o canal Y em YIQ.
        :param to_torch: Se True, retorna o batch como um tensor PyTorch.
        :return: Batch processado.
        """
        batch = []
        for _ in range(batch_size):
            if self.current_frame_idx >= self.total_frames:
                break  # Fim do vídeo
            frame = self.get_frame(
                self.current_frame_idx,
                scale_factor=scale_factor,
                only_y=only_y,
                to_torch=to_torch
            )
            batch.append(frame)
        if not batch:
            return None
        
        # Concatena os frames no batch
        if to_torch:
            batch_tensor = torch.stack(batch, dim=0)  # [batch_size, C, H, W]
            if only_y:
                batch_tensor = batch_tensor.unsqueeze(1)  # [batch_size, 1, H, W]
            return batch_tensor
        else:
            return batch
    def get_frames_range(self, i, j, scale_factor=1.0, only_y=True, to_torch=True):
        """
        Obtém frames do índice i até j-1 (inclusive).

        :param i: Índice inicial do frame.
        :param j: Índice final do frame (não incluído).
        :param scale_factor: Fator de escala para redimensionar os frames.
        :param only_y: Se True, retorna apenas o canal Y em YIQ.
        :param to_torch: Se True, retorna os frames como tensores PyTorch.
        :return: Lista de frames processados.
        """
        if i < 0 or i >= j:
            raise ValueError("Índices inválidos.")

        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, i)  # Move o ponteiro para o frame i

        for frame_idx in range(i, j):
            ret, frame = self.cap.read()
            if not ret:
                break

            # Redimensiona o frame se necessário
            if scale_factor != 1.0:
                new_width = int(self.frame_width * scale_factor)
                new_height = int(self.frame_height * scale_factor)
                frame = cv2.resize(frame, (new_width, new_height))

            # Processa o frame conforme os parâmetros
            if only_y:
                frame = self.rgb_to_yiq(frame)  # Converte para YIQ e extrai o canal Y
            elif to_torch:
                frame = torch.from_numpy(frame).to(self.device).permute(2, 0, 1)  # [C, H, W]

            frames.append(frame)
        while len(frames)<j-i:
            frames.append(frames[-1])

        # Concatena os frames no batch se necessário
        if to_torch:
            frames_tensor = torch.stack(frames, dim=0)  # [batch_size, C, H, W]
            if only_y:
                frames_tensor = frames_tensor.unsqueeze(1)  # [batch_size, 1, H, W]
            return frames_tensor
        else:
            return frames
    def rgb_to_bgr(self, rgb_tensor):
        """
        Converts an RGB tensor [3, H, W] or [B, 3, H, W] to BGR format (OpenCV style).
        
        :param rgb_tensor: Tensor in RGB format, shape [3, H, W] or [B, 3, H, W]
        :return: Tensor in BGR format with the same shape
        """
        if not isinstance(rgb_tensor, torch.Tensor):
            raise TypeError("Input must be a PyTorch tensor.")

        if rgb_tensor.dim() not in [3, 4]:
            raise ValueError("Input must have 3 or 4 dimensions: [3,H,W] or [B,3,H,W].")

        # Inverte a ordem dos canais: RGB → BGR ([0,1,2] → [2,1,0])
        bgr_tensor = rgb_tensor.flip(dims=(-3,))  # Flip no canal (dim=-3)

        return bgr_tensor       
    def reset(self):
        """
        Reinicia o vídeo para o início.
        """
        self.current_frame_idx = 0
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def close(self):
        """
        Libera os recursos do OpenCV.
        """
        self.cap.release()

def frequency_bandpass_mask(T, fs, low, high):
                        freqs = torch.fft.fftfreq(T, d=1/fs)  # shape: (T,)
                        mask = (freqs >= low) & (freqs <= high) | (freqs <= -low) & (freqs >= -high)
                        return mask.float()


class VideoWriterAuto:
    def __init__(self, path, fps=30):
        self.path = path
        self.fps = fps
        self.writer = None
        self.closed = False

    def add(self, frame):
        if self.closed:
            print("[VideoWriterAuto] Warning: Tried to write to a closed writer.")
            return

        if self.writer is None:
            h, w = frame.shape[:2]
            is_color = frame.ndim == 3 and frame.shape[2] == 3
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h), isColor=is_color)

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        self.writer.write(np.uint8(frame))

    def close(self):
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        self.closed = True  # Mark as closed

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def restart(self):
        self.close()  # close current writer if open
        self.closed = False
        self.writer = None
        



def view_pyr(coeffs):
    for i in range(1,coeffs['last_level']+1):
        for j in range(coeffs['level_{i}']):
            pass

def dilate_mask(mask: torch.Tensor, kernel_size: int = 3, iterations: int = 1) -> torch.Tensor:
    """
    Dilate a 2D binary mask using max pooling.
    
    Args:
        mask (torch.Tensor): Binary mask (H, W) or (1, 1, H, W), dtype=bool or float.
        kernel_size (int): Size of the dilation kernel (must be odd).
        iterations (int): How many times to apply dilation.

    Returns:
        torch.Tensor: Dilated mask (same shape as input).
    """
    if mask.ndim == 2:
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
    
    mask = mask.float()
    padding = kernel_size // 2

    for _ in range(iterations):
        mask = F.max_pool2d(mask, kernel_size, stride=1, padding=padding)
    
    return mask.squeeze(0).squeeze(0) > 0.5  # Return as boolean mask

def resize_complex_image(image: torch.Tensor, size: tuple) -> torch.Tensor:
    """
    Resize a complex image to the given size.

    Args:
        image (torch.Tensor): Complex tensor of shape [H, W] or [B, H, W].
        size (tuple): Desired (new_height, new_width).

    Returns:
        torch.Tensor: Resized complex tensor of the same shape format.
    """
    assert torch.is_complex(image), "Input must be a complex tensor"
    dim = image.ndim
    assert dim in (2, 3), "Image must be [H, W] or [B, H, W]"

    # Separate real and imaginary parts
    real = image.real
    imag = image.imag

    if dim == 2:
        real = real.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
        imag = imag.unsqueeze(0).unsqueeze(0)
    else:  # dim == 3
        real = real.unsqueeze(1)  # [B, 1, H, W]
        imag = imag.unsqueeze(1)

    # Resize using bilinear or bicubic interpolation
    real_resized = F.interpolate(real, size=size, mode='bilinear', align_corners=False)
    imag_resized = F.interpolate(imag, size=size, mode='bilinear', align_corners=False)

    # Combine back to complex
    complex_resized = torch.complex(real_resized.squeeze(1), imag_resized.squeeze(1))  # [B, H, W] or [1, H, W]

    if dim == 2:
        return complex_resized.squeeze(0)  # [H, W]
    return complex_resized  # [B, H, W]



import torch
import torch.nn.functional as F

def warp_image_with_complex_field(image: torch.Tensor, flow_complex: torch.Tensor) -> torch.Tensor:
    """
    Warps a grayscale or color image [C, H, W] using a complex-valued motion field [H, W].

    Args:
        image (torch.Tensor): Input image, shape [1, H_img, W_img] or [3, H_img, W_img]
        flow_complex (torch.Tensor): Complex motion field, shape [H, W] (real=dx, imag=dy)

    Returns:
        torch.Tensor: Warped image, same shape as input image
    """
    H_flow, W_flow = flow_complex.shape
    C, H_img, W_img = image.shape

    # Resize if needed
    if (H_img, W_img) != (H_flow, W_flow):
        image = image.unsqueeze(0)  # [1, C, H_img, W_img]
        image = F.interpolate(image, size=(H_flow, W_flow), mode='bilinear', align_corners=True)
        image = image[0]  # [C, H, W]

    # Coordinate grid normalized to [-1, 1]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, H_flow, device=image.device),
        torch.linspace(-1, 1, W_flow, device=image.device),
        indexing='ij'
    )
    grid = torch.stack((x, y), dim=-1)  # [H, W, 2]

    # Normalize flow
    dx = flow_complex.real / (W_flow / 2)
    dy = flow_complex.imag / (H_flow / 2)
    flow = torch.stack((dx, dy), dim=-1)  # [H, W, 2]

    sample_grid = (grid - flow).unsqueeze(0)  # [1, H, W, 2]
    image = image.unsqueeze(0)  # [1, C, H, W]

    # Warp
    warped = F.grid_sample(image, sample_grid, mode='bilinear', padding_mode='border', align_corners=True)

    return warped[0]  # [C, H, W]

import torch
import numpy as np
import cv2


def draw_complex_flow_on_image_bkp(image: torch.Tensor, flow: torch.Tensor, step: int = 8, color=(0, 0, 0)) -> torch.Tensor:
    """
    Draw complex flow as arrows over a grayscale or RGB image.

    Args:
        image: torch.Tensor, shape [H, W], [1, H, W], or [3, H, W], with values in [0, 1].
        flow: torch.complex64 or complex128 tensor, shape [H, W], representing flow vectors.
        step: int, spacing between arrows.
        color: tuple of 3 ints (B, G, R), color of the arrows (default: black).
    
    Returns:
        torch.Tensor, shape [3, H, W], dtype=torch.uint8, RGB image with arrows.
    """
    if image.ndim == 3 and image.shape[0] == 1:
        image = image.squeeze(0)  # shape [H, W]

    H, W = flow.shape

    # Resize image if needed
    if image.shape[-2:] != (H, W):
        image = torch.nn.functional.interpolate(
            image.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        ).squeeze(0)

    # Convert image to uint8 numpy
    if image.ndim == 2:
        img_np = (image.clamp(0, 1).cpu().numpy() * 255).astype(np.uint8)  # shape [H, W]
        vis = cv2.cvtColor(img_np, cv2.COLOR_GRAY2BGR)
    elif image.ndim == 3 and image.shape[0] == 3:
        img_np = (image.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)  # [H, W, 3]
        vis = img_np.copy()
    else:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Prepare flow vectors
    flow_np = flow.cpu().numpy()
    y, x = np.mgrid[step//2:H:step, step//2:W:step].reshape(2, -1).astype(int)
    fx = np.real(flow_np[y, x])
    fy = np.imag(flow_np[y, x])

    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw lines on image
    cv2.polylines(vis, lines, isClosed=False, color=color, thickness=1)

    # Return as torch.Tensor in [3, H, W], uint8
    return torch.from_numpy(vis).permute(2, 0, 1).to(torch.uint8)


def draw_complex_flow_on_image(img: torch.Tensor, flow: torch.Tensor, scale = 1, step: int = 8) -> np.ndarray:
    """
    Resize image to flow shape and draw complex flow as arrows using OpenCV.

    Args:
        img (torch.Tensor): Image tensor, shape [1,H,W], [3,H,W], or [H,W], values in [0,1] or [0,255].
        flow (torch.Tensor): Complex-valued [H,W] torch tensor representing flow.
        step (int): Step size between drawn arrows.

    Returns:
        np.ndarray: BGR image with flow arrows drawn (dtype=uint8, shape [H, W, 3]).
    """
    H, W = flow.shape

    # Resize image to match flow shape
    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img_resized = torch.nn.functional.interpolate(
            img.unsqueeze(0), size=(H, W), mode='bilinear', align_corners=False
        )[0]
    elif img.ndim == 2:
        img_resized = img
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # Convert to NumPy
    img_np = img_resized.detach().cpu().numpy()
    flow_np = flow.detach().cpu().numpy()

    # Format image for drawing (BGR)
    if img_np.ndim == 2:  # [H, W] grayscale
        img_vis = cv2.cvtColor((img_np * scale).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img_np.ndim == 3 and img_np.shape[0] == 1:  # [1, H, W] grayscale
        img_vis = cv2.cvtColor((img_np[0] * scale).clip(0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    elif img_np.ndim == 3 and img_np.shape[0] == 3:  # [3, H, W] color
        img_np = np.transpose(img_np, (1, 2, 0))  # to HWC
        img_vis = (img_np * scale).clip(0, 255).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported image format after resize: {img_np.shape}")

    # Ensure contiguous layout for OpenCV
    img_vis = np.ascontiguousarray(img_vis)

    # Generate line coordinates from flow
    y, x = np.mgrid[step//2:H:step, step//2:W:step].reshape(2, -1).astype(int)
    fx = np.real(flow_np[y, x])
    fy = np.imag(flow_np[y, x])
    lines = np.vstack([x, y, x + fx, y + fy]).T.reshape(-1, 2, 2)
    lines = np.int32(lines + 0.5)

    # Draw flow lines
    cv2.polylines(img_vis, lines, isClosed=True, color=(0, 0, 0), thickness=2)

    return img_vis


def flow_to_color(flow: torch.Tensor, max_magnitude: float = None) -> np.ndarray:
    """
    Convert a complex-valued flow field into a BGR image using HSV encoding.
    
    Args:
        flow (torch.Tensor): Complex tensor of shape [H, W].
        max_magnitude (float or None): Optional clip value for normalization.

    Returns:
        np.ndarray: BGR image [H, W, 3] in uint8 format.
    """
    assert flow.ndim == 2 and torch.is_complex(flow)
    flow_np = flow.cpu().numpy()
    fx, fy = flow_np.real, flow_np.imag

    magnitude = np.sqrt(fx**2 + fy**2)
    angle = np.arctan2(fy, fx)  # radians

    if max_magnitude is None:
        max_magnitude = np.percentile(magnitude, 99)  # robust normalization

    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle + np.pi) * 180 / np.pi / 2).astype(np.uint8)  # Hue: [0, 180]
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip((magnitude / max_magnitude) * 255, 0, 255).astype(np.uint8)

    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def get_highest_magnitude_complex(complex_tensor):
    """
    Finds the single complex value with the highest magnitude in a
    multidimensional PyTorch tensor.

    Args:
        complex_tensor (torch.Tensor): A multidimensional PyTorch tensor
                                       with a complex data type.

    Returns:
        torch.Tensor: A single-element complex tensor containing the value
                      with the highest magnitude.
    """
    magnitudes = torch.abs(complex_tensor)
    max_magnitude = torch.max(magnitudes)
    # Find the index (or indices if there are ties) of the maximum magnitude
    max_magnitude_indices = (magnitudes == max_magnitude).nonzero(as_tuple=True)

    # Since we want a single value, we'll take the first occurrence
    # You might want to handle ties differently based on your needs.
    index_tuple = tuple(idx[0].item() for idx in max_magnitude_indices)

    return complex_tensor[index_tuple].unsqueeze(0) # Keep it as a tensor


from typing import Union

def pack_images(images: list[Union[np.ndarray, torch.Tensor]], first_row: int, flip = True) -> np.ndarray:
    """
    Arrange images (NumPy or PyTorch, grayscale or color) into a grid with optional padding.
    Accepts tensors in (H, W), (H, W, 3), or (C, H, W) format.

    Args:
        images: List of images as np.ndarray or torch.Tensor.
        first_row: Number of images in the first row.

    Returns:
        np.ndarray grid of all images with padding if needed.
    """
    assert len(images) > 0 and first_row > 0

    converted = []
    for img in images:
        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()
        if img.ndim == 3 and img.shape[0] in {1, 3}:  # (C, H, W) -> (H, W, C)
            img = np.transpose(img, (1, 2, 0))
        elif img.ndim == 2:
            img = np.expand_dims(img, axis=-1)  # (H, W) -> (H, W, 1)
        converted.append(img)

    images = converted
    is_color = images[0].shape[2] == 3
    dtype = images[0].dtype

    rows = (len(images) + first_row - 1) // first_row
    cols = first_row

    max_h = np.zeros(rows, dtype=int)
    max_w = np.zeros(cols, dtype=int)

    for idx, img in enumerate(images):
        r, c = divmod(idx, cols)
        h, w = img.shape[:2]
        max_h[r] = max(max_h[r], h)
        max_w[c] = max(max_w[c], w)

    total_h, total_w = sum(max_h), sum(max_w)
    canvas_shape = (total_h, total_w, 3) if is_color else (total_h, total_w)
    canvas = np.zeros(canvas_shape, dtype=dtype)

    y = 0
    for r in range(rows):
        x = 0
        for c in range(cols):
            idx = r * cols + c
            if idx >= len(images):
                continue
            img = images[idx]
            if flip:
                img = np.flip(img, axis=(0,1))
            h, w = img.shape[:2]
            if not is_color and img.shape[2] == 1:
                img = img[:, :, 0]
            if is_color:
                canvas[y:y+h, x:x+w, :] = img
            else:
                canvas[y:y+h, x:x+w] = img
            x += max_w[c]
        y += max_h[r]

    return canvas



def get_fps(video_path):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Arquivo '{video_path}' não encontrado.")
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    return fps

def frequencies(fps):

    f_lo = get('freq_low')/10
    f_hi = get('freq_high')/10
    if f_hi <= f_lo:
        f_hi = f_lo +0.1
    return f_lo, f_hi

def resize_tensor(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Resize a real or complex tensor of shape [C, H, W] to [C, H2, W2].

    Args:
        t (torch.Tensor): Real or complex tensor of shape [C, H, W]
        size (tuple): Target size (H2, W2)

    Returns:
        torch.Tensor: Resized tensor of shape [C, H2, W2], same dtype as input
    """
    is_complex = torch.is_complex(t)
    C, H, W = t.shape

    if is_complex:
        real = t.real.unsqueeze(0)  # [1, C, H, W]
        imag = t.imag.unsqueeze(0)
        real_resized = F.interpolate(real, size=size, mode='bilinear', align_corners=False)
        imag_resized = F.interpolate(imag, size=size, mode='bilinear', align_corners=False)
        return torch.complex(real_resized.squeeze(0), imag_resized.squeeze(0))  # [C, H2, W2]
    else:
        t_exp = t.unsqueeze(0)  # [1, C, H, W]
        t_resized = F.interpolate(t_exp, size=size, mode='bilinear', align_corners=False)
        return t_resized.squeeze(0)  # [C, H2, W2]

def add_text_line(image, text, margin=10, font_scale=1.0, thickness=2,
                  xpos=None, ypos=None, crop=None):
    """
    Add outlined text to an image, with optional lines and cropping.

    Drawing order:
        1. Draw lines (on full image)
        2. Crop if specified
        3. Add outlined text on cropped image

    Args:
        image (np.ndarray): Input image.
        text (str): Text to draw.
        margin (int): Margin from bottom-left for text.
        font_scale (float): Font scale.
        thickness (int): Text thickness.
        xpos (int or None): X position for vertical red line.
        ypos (int or None): Y position for horizontal green line.
        crop (tuple or None): (ymin, ymax, xmin, xmax) crop region.

    Returns:
        np.ndarray: Final annotated (and possibly cropped) image.
    """
    img_copy = image.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    line_type = cv2.LINE_AA

    H, W = img_copy.shape[:2]

    # --- Draw vertical red line at xpos ---
    if xpos is not None and 0 <= xpos < W:
        cv2.line(img_copy, (xpos, 0), (xpos, H), color=(0, 0, 255), thickness=1, lineType=line_type)

    # --- Draw horizontal green line at ypos ---
    if ypos is not None and 0 <= ypos < H:
        cv2.line(img_copy, (0, ypos), (W, ypos), color=(0, 255, 0), thickness=1, lineType=line_type)

    # --- Crop if requested ---
    if crop is not None:
        ymin, ymax, xmin, xmax = crop
        img_copy = img_copy[ymin:ymax, xmin:xmax]

    # --- Draw outlined white text near bottom-left ---
    Hc, Wc = img_copy.shape[:2]
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = margin
    y = Hc - margin - baseline

    outline_thickness = thickness + 2
    cv2.putText(img_copy, text, (x, y), font, font_scale, (0, 0, 0), outline_thickness, line_type)
    cv2.putText(img_copy, text, (x, y), font, font_scale, (255, 255, 255), thickness, line_type)

    return img_copy


def to_rgb_image(image, scale = False):
    """
    Converts an image to a 3-channel NumPy array with shape [H, W, 3].
    If the values are in the 0–255 range, it is cast to uint8.

    Parameters:
        image: np.ndarray or torch.Tensor
            Input image. Can be grayscale or color, with shape [H,W], [1,H,W], [3,H,W], [1,1,H,W], etc.

    Returns:
        np.ndarray: Image with shape [H, W, 3], possibly uint8.
    """
    # Convert torch.Tensor to np.ndarray
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    # Remove leading singleton dimensions (e.g. [1,1,H,W] -> [H,W])
    while image.ndim > 3:
        image = image.squeeze(0)

    # From [1, H, W] → [H, W]
    if image.ndim == 3 and image.shape[0] == 1:
        image = image[0]

    # From [3, H, W] → [H, W, 3]
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))

    # From [H, W] → [H, W, 3]
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)

    # From [H, W, 1] → [H, W, 3]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.concatenate([image] * 3, axis=-1)

    if scale and image.max()-image.min()>0:
        image = image-image.min()
        image = image/image.max()
        image*=255
    image = np.round(image).astype(np.uint8)

    return image