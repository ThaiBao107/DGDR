import cv2
import numpy as np
import torch
from torchvision.transforms import Compose, Normalize, ToTensor
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from modeling.VMamba.OCAB_VMamba import *

# 1. TẢI MÔ HÌNH CỦA BẠN
# Thay thế dòng này bằng mô hình classification bạn đã train
# Ví dụ: model = MyCustomResNet(num_classes=5)
# model.load_state_dict(torch.load('model_weights.pth'))


model = vmamba_small_m2()

# ==========================================
# 2. NẠP PHẦN THÂN (TỪ FILE best_model_APTOS.pth)
# ==========================================
body_path = "/kaggle/input/models/iampeter107/ocab-vmamba-4khoi-chaix/pytorch/default/1/best_model_APTOS.pth"
checkpoint_body = torch.load(body_path, map_location='cpu')

if 'model' in checkpoint_body:
    checkpoint_model = checkpoint_body['model']
else:
    checkpoint_model = checkpoint_body

model_dict = model.state_dict()
new_checkpoint = {}

print("BẮT ĐẦU NẠP PHẦN THÂN (BACKBONE)...")
for k, v in checkpoint_model.items():
    k = k.replace("module.", "")

    # Chủ động bỏ qua các key cũ của classifier trong file này
    # (vì lát nữa mình sẽ nạp bản xịn từ file best_classifier)
    if "classifier" in k:
        continue

    # Load dữ liệu vào model
    if k in model_dict:
        if v.shape == model_dict[k].shape:
            new_checkpoint[k] = v
        else:
            print(f"⚠️ Bỏ qua {k}: Sai kích thước")
    else:
        print(f"ℹ️ Bỏ qua {k}: Không tồn tại")

# Nạp các trọng số đã lọc vào model
model.load_state_dict(new_checkpoint, strict=False)
print("✅ NẠP XONG PHẦN THÂN!")

# ==========================================
# 3. NẠP PHẦN NÃO BỘ (TỪ FILE best_classifier.pth)
# ==========================================
print("\nBẮT ĐẦU NẠP NÃO BỘ (CLASSIFIER)...")
# LƯU Ý: Đổi lại đường dẫn này trỏ tới đúng file best_classifier.pth của bạn trên Kaggle
classifier_path = "/kaggle/input/models/iampeter107/ocab-vmamba-classification-aptos/pytorch/default/1/best_classifier.pth"
checkpoint_head = torch.load(classifier_path, map_location='cpu')

# Bơm thẳng 2 tensor (weight và bias) vào lớp classifier của bạn.
# Vì lớp nn.Linear tự động nhận dạng 'weight' và 'bias', lệnh này khớp hoàn hảo 100%!
model.classifier.load_state_dict(checkpoint_head)
print("✅ NẠP XONG NÃO BỘ!")

# ==========================================
# 4. CHUẨN BỊ CHẠY GRAD-CAM
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

# 2. CHỌN LỚP MỤC TIÊU (TARGET LAYER)
# Grad-CAM cần lấy gradient từ một lớp Convolutional.
# Thường chúng ta sẽ chọn lớp Conv CUỐI CÙNG trước lớp Linear/Dense
# Ví dụ nếu dùng ResNet: target_layers = [model.layer4[-1]]
# Ví dụ nếu dùng VGG16: target_layers = [model.features[-1]]
target_layers = [model.layers[-1]]


def reshape_transform(tensor):
    # tensor là kết quả trích xuất từ self.layers[-1]

    # Kiểm tra: Nếu kích thước cuối cùng (Channels) lớn hơn kích thước chiều cao (thường là 7)
    # Tức là nó đang ở dạng (B, H, W, C) -> Bắt buộc phải lật về (B, C, H, W)
    if len(tensor.shape) == 4 and tensor.shape[-1] > tensor.shape[1]:
        return tensor.permute(0, 3, 1, 2)

    # Nếu nó đã ở dạng (B, C, H, W) sẵn rồi thì trả về nguyên bản
    return tensor


# 3. KHỞI TẠO GRAD-CAM
cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

# 4. CHUẨN BỊ ẢNH ĐẦU VÀO
# Đọc ảnh gốc bằng OpenCV
img_path = "/kaggle/working/DGDR/data/FundusDG/images/APTOS/severe_npdr/070f67572d03.png"
rgb_img = cv2.imread(img_path, 1)[:, :, ::-1]
rgb_img = np.float32(rgb_img) / 255.0  # Chuẩn hóa ảnh về khoảng [0, 1] cho hàm show_cam_on_image

# Tiền xử lý ảnh thành Tensor để đưa vào model (cần giống hệt lúc bạn train)
# Ví dụ chuẩn hóa theo ImageNet:
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

# 5. CHỌN CLASS (LỚP) ĐỂ VẼ HEATMAP
# Ví dụ: Bài toán có 5 lớp (0, 1, 2, 3, 4). Bạn muốn xem heatmap của lớp 3.
# Nếu bạn để targets = None, thư viện sẽ tự động lấy class có điểm dự đoán cao nhất.
targets = None

# 6. TẠO HEATMAP
# Trả về một mảng 2D (grayscale_cam) có giá trị từ 0 đến 1
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
grayscale_cam = grayscale_cam[0, :]  # Lấy ảnh đầu tiên trong batch

# ==========================================
# 6.5. CHẾ TÁC LẠI HEATMAP (PHONG CÁCH BÀI BÁO)
# ==========================================
# a. Lọc bỏ nền đen ngoài nhãn cầu
retina_mask = np.sum(rgb_img, axis=-1) > 0.05
grayscale_cam = grayscale_cam * retina_mask

# b. Tạo hiệu ứng phân lớp màu (Contour / Quantization)
grayscale_cam[grayscale_cam < 0.2] = 0  # Xóa nhiễu nhẹ
num_levels = 5
grayscale_cam = np.ceil(grayscale_cam * num_levels) / num_levels

# ==========================================
# 7. PHỦ HEATMAP LÊN ẢNH GỐC (OVERLAY)
# Hàm này sẽ chuyển grayscale_cam thành dạng nhiệt (đỏ/vàng/xanh) và trộn với ảnh gốc
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True, image_weight=0.5)

# 7.5 TẠO ẢNH CHỈ CÓ HEATMAP (HEATMAP ONLY)
# Chuyển grayscale_cam (từ 0-1) sang dạng 0-255 và ép kiểu uint8
cam_uint8 = np.uint8(255 * grayscale_cam)
# Áp dụng dải màu JET (Xanh đậm -> Đỏ) của OpenCV
heatmap_only = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)

# 8. LƯU KẾT QUẢ
# Lưu ảnh Overlaid (Nhớ đảo hệ màu RGB -> BGR cho OpenCV)
cv2.imwrite('/kaggle/working/DGDR/figures/overlaid_aptos_nor.jpg', visualization[:, :, ::-1])

# Lưu ảnh Heatmap Only (OpenCV mặc định lưu BGR nên không cần đảo)
cv2.imwrite('/kaggle/working/DGDR/figures/heapmaponly_aptos_nor.jpg', heatmap_only)

print("✅ Đã lưu thành công cả ảnh Overlaid và Heatmap Only!")