import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
import time
import copy
import os

def train_model(model, criterion, optimizer, num_epochs=20):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    time_elapsed = time.time() - since
    print(f"การฝึกเสร็จใน {time_elapsed // 60:.0f} นาที {time_elapsed % 60:.0f} วินาที")
    print(f"ความแม่นยำสูงสุด: {best_acc:.4f}")

    model.load_state_dict(best_model_wts)
    return model

if __name__ == '__main__':
    # ⚙️ การตั้งค่าไฮเปอร์พารามิเตอร์
    data_dir = 'top10_thai_food_dataset'  # โฟลเดอร์ข้อมูล
    num_classes = 10      # จำนวนคลาส (10 ประเภทอาหาร)
    batch_size = 32       # ขนาด batch
    num_epochs = 20       # จำนวนรอบการฝึก
    learning_rate = 0.001 # อัตราการเรียนรู้

    # 🎉 **1. เตรียมข้อมูล**
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

    # โหลดข้อมูล train, val
    image_datasets = {
        x: datasets.ImageFolder(os.path.join(data_dir, x), data_transforms[x])
        for x in ['train', 'val']
    }
    dataloaders = {
        x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=2)
        for x in ['train', 'val']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 🎉 **2. สร้างโมเดล MobileNetV2**
    model = models.mobilenet_v2(weights="MobileNet_V2_Weights.DEFAULT")
    # แช่ค่าทั้งหมด (ไม่ให้ปรับปรุงค่าใน convolution layers)
    for param in model.features.parameters():
        param.requires_grad = False
    # ปรับส่วน classifier
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)

    # 🎉 **3. กำหนด Loss และ Optimizer**
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 🎉 **5. ฝึกโมเดล**
    best_model = train_model(model, criterion, optimizer, num_epochs=num_epochs)

    # 🎉 **6. บันทึกโมเดล**
    torch.save(best_model.state_dict(), 'thai_food_modelv8.pth')
    print("✅ บันทึกโมเดล thai_food_model.pth สำเร็จ!")