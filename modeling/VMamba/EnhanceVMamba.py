import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


class DWConvBranch(nn.Module):
    """
    Depthwise Separable Conv branch để bắt local texture.
    Chi phí tính toán rất nhỏ vì dùng depthwise (groups=channels).
    Phù hợp để phát hiện lesion nhỏ (microaneurysm, hard exudate) trong ảnh fundus.
    """

    def __init__(self, dim: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2

        # Depthwise conv: mỗi channel xử lý độc lập → rất nhẹ
        self.dw_conv = nn.Conv2d(
            dim, dim,
            kernel_size=kernel_size,
            padding=padding,
            groups=dim,  # depthwise
            bias=False
        )
        # Pointwise conv: mix thông tin cross-channel
        self.pw_conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.norm = nn.BatchNorm2d(dim)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C) → cần transpose sang (B, C, H, W) cho Conv2d
        x = x.permute(0, 3, 1, 2).contiguous()  # (B, C, H, W)
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        x = self.norm(x)
        x = self.act(x)
        x = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        return x


class SpatialGate(nn.Module):
    """
    Learned spatial gate: học trọng số alpha(x,y) ∈ [0,1] cho từng vị trí.
    output = alpha * ssm_feat + (1 - alpha) * conv_feat

    Với ảnh fundus DR:
      - Vùng mạch máu, optic disc → SSM (global context) được ưu tiên
      - Vùng lesion nhỏ, texture  → Conv (local detail) được ưu tiên
    """

    def __init__(self, dim: int):
        super().__init__()
        # Gate nhẹ: chỉ 2 conv để tính alpha map
        self.gate = nn.Sequential(
            nn.Conv2d(dim * 2, dim // 4, kernel_size=1, bias=False),
            nn.GELU(),
            nn.Conv2d(dim // 4, 1, kernel_size=1, bias=False),
            nn.Sigmoid()  # alpha ∈ [0, 1]
        )

    def forward(
            self,
            ssm_feat: torch.Tensor,  # (B, H, W, C)
            conv_feat: torch.Tensor,  # (B, H, W, C)
    ) -> torch.Tensor:
        # Transpose sang (B, C, H, W) để chạy Conv2d
        s = ssm_feat.permute(0, 3, 1, 2).contiguous()
        c = conv_feat.permute(0, 3, 1, 2).contiguous()

        # Concat → tính alpha map
        concat = torch.cat([s, c], dim=1)  # (B, 2C, H, W)
        alpha = self.gate(concat)  # (B, 1, H, W)

        # Weighted sum
        fused = alpha * s + (1.0 - alpha) * c  # (B, C, H, W)
        fused = fused.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        return fused


class ImprovedSS2D(nn.Module):
    """
    SS2D cải tiến: DWConv Branch + Gated Fusion

    Kiến trúc:
        Input ──┬── [SS2D gốc] ──────────────────────┐
                │                                      ├── SpatialGate ──► Output
                └── [DWConv Branch (local texture)] ──┘

    Tham số:
        d_model   : số channels (= dim trong VMamba block)
        d_state   : state dimension của SSM (mặc định 16)
        ssm_ratio : expansion ratio của SSM (mặc định 2.0)
        dt_rank   : rank của delta_t projection ("auto" = d_model // 16)
        use_dwconv: bật/tắt DWConv branch (default True)
        conv_kernel: kernel size của DWConv (default 3)
    """

    def __init__(
            self,
            d_model: int,
            d_state: int = 16,
            ssm_ratio: float = 2.0,
            dt_rank: str = "auto",
            use_dwconv: bool = True,
            conv_kernel: int = 3,
            # ── Các tham số gốc của SS2D ──
            act_layer=nn.SiLU,
            d_conv: int = 3,
            dropout: float = 0.0,
            bias: bool = False,
            dt_min: float = 0.001,
            dt_max: float = 0.1,
            dt_init: str = "random",
            dt_scale: float = 1.0,
            dt_init_floor: float = 1e-4,
            **kwargs,
    ):
        super().__init__()
        self.d_model = d_model
        self.use_dwconv = use_dwconv

        # ── 1. SS2D gốc (giữ nguyên) ─────────────────────────────────────────
        # Thay thế bằng import SS2D gốc từ VMamba:
        # self.ssm = SS2D_Original(d_model=d_model, d_state=d_state, ...)
        #
        # Ở đây dùng placeholder để demo cấu trúc:
        self.ssm = _SS2DPlaceholder(d_model, d_state, ssm_ratio)

        # ── 2. DWConv branch (local texture) ─────────────────────────────────
        if self.use_dwconv:
            self.conv_branch = DWConvBranch(d_model, kernel_size=conv_kernel)
            self.gate = SpatialGate(d_model)

        # ── 3. Dropout ────────────────────────────────────────────────────────
        self.drop = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, H, W, C)  — layout của VMamba
        Returns:
            out: (B, H, W, C)
        """
        # Nhánh SSM (global context, long-range)
        ssm_out = self.ssm(x)  # (B, H, W, C)

        if self.use_dwconv:
            # Nhánh Conv (local texture, lesion detection)
            conv_out = self.conv_branch(x)  # (B, H, W, C)

            # Gated fusion: học trọng số spatial theo từng vị trí
            out = self.gate(ssm_out, conv_out)
        else:
            out = ssm_out

        return self.drop(out)


# ─────────────────────────────────────────────────────────────────────────────
# Placeholder để demo khi chưa có VMamba source
# Thay bằng SS2D thật từ VMamba khi tích hợp thực tế
# ─────────────────────────────────────────────────────────────────────────────
class _SS2DPlaceholder(nn.Module):
    """
    Placeholder cho SS2D gốc của VMamba.
    THAY BẰNG SS2D THẬT khi tích hợp:
        from vmamba.models.vmamba import SS2D
        self.ssm = SS2D(d_model=d_model, d_state=d_state, ...)
    """

    def __init__(self, d_model, d_state, ssm_ratio):
        super().__init__()
        d_inner = int(d_model * ssm_ratio)
        self.in_proj = nn.Linear(d_model, d_inner, bias=False)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)
        self.norm = nn.LayerNorm(d_inner)

    def forward(self, x):
        return self.out_proj(self.norm(F.gelu(self.in_proj(x))))


# ─────────────────────────────────────────────────────────────────────────────
# Cách tích hợp vào VMambaBlock gốc
# ─────────────────────────────────────────────────────────────────────────────
class ImprovedVMambaBlock(nn.Module):
    """
    VMamba Block với SS2D được thay bằng ImprovedSS2D.

    Thay thế VSSBlock / VMambaBlock gốc trong:
        vmamba/models/vmamba.py → class VSSBlock

    Giữ nguyên:
        - LayerNorm trước SSM
        - MLP (FFN) sau SSM
        - Skip connection

    Chỉ thay:
        self.op = SS2D(...)  →  self.op = ImprovedSS2D(...)
    """

    def __init__(
            self,
            hidden_dim: int,
            mlp_ratio: float = 4.0,
            drop: float = 0.0,
            use_dwconv: bool = True,
            **ssm_kwargs,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # ← Đây là chỗ thay SS2D gốc
        self.op = ImprovedSS2D(
            d_model=hidden_dim,
            use_dwconv=use_dwconv,
            dropout=drop,
            **ssm_kwargs,
        )

        # MLP (FFN) giữ nguyên như VMamba gốc
        mlp_hidden = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden, hidden_dim),
            nn.Dropout(drop),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, H, W, C)
        # SSM branch với skip connection
        x = x + self.op(self.norm1(x))
        # MLP branch với skip connection
        x = x + self.mlp(self.norm2(x))
        return x
