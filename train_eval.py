import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import random
import json

# ---------------------------
# 0️⃣ Seed for reproducibility
# ---------------------------
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ---------------------------
# 1️⃣ Load H5 dataset
# ---------------------------
H5_FILE = "train_pneumonia_subset_correct.h5"
with h5py.File(H5_FILE, "r") as h5:
    X = h5["images"][:].astype("float32") / 255.0
    y = h5["labels"][:]

print("Loaded X:", X.shape, "y:", y.shape)
class_counts = dict(zip([0,1], np.bincount(y)))
print("Class distribution:", class_counts)

results = {
    "class_counts": class_counts,
    "sample_tracking": {
        "original": {
            "class_0": int(class_counts[0]),
            "class_1": int(class_counts[1]),
            "total": int(len(y))
        }
    },
    "metrics": {}
}

# ---------------------------
# 2️⃣ Train/Val split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED
)

print(f"\n📊 Train set class distribution:")
train_class_counts = dict(zip([0,1], np.bincount(y_train)))
print(f"   Class 0: {train_class_counts[0]}, Class 1: {train_class_counts[1]}")

# ---------------------------
# 3️⃣ PyTorch Datasets
# ---------------------------
class PneumoniaDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype("int64")
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        img = np.transpose(self.X[idx], (2,0,1))
        return torch.tensor(img), torch.tensor(self.y[idx])

# CombinedDataset for synthetic data
class CombinedDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y.astype("int64")
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

train_ds = PneumoniaDataset(X_train, y_train)
val_ds   = PneumoniaDataset(X_val, y_val)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

# ---------------------------
# 4️⃣ Simple CNN
# ---------------------------
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,8,3,padding=1)
        self.conv2 = nn.Conv2d(8,16,3,padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(16*56*56, 32)
        self.fc2 = nn.Linear(32,1)
    def forward(self,x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

def train_model(model, train_loader, val_loader, epochs=5):
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.float().view(-1,1).to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {total_loss/len(train_loader):.4f}")

    # Validation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = (outputs>0.5).long().view(-1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return model, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "confusion_matrix": cm.tolist()}

# ---------------------------
# 5️⃣ CNN on original data
# ---------------------------
print("\n" + "="*50)
print("🔹 Training on ORIGINAL dataset")
print("="*50)
cnn_orig = SimpleCNN()
cnn_orig, metrics_orig = train_model(cnn_orig, train_loader, val_loader)
print("✅ Original dataset metrics:", metrics_orig)
results["metrics"]["original"] = metrics_orig

# ===========================
# 6️⃣ AE + SMOTE
# ===========================
print("\n" + "="*50)
print("🔹 Training AUTOENCODER + SMOTE")
print("="*50)

# Encoder / Decoder classes
class Encoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(16,32,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32,64,3,stride=2,padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*28*28, latent_dim)
        )
    def forward(self,x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.fc = nn.Linear(latent_dim,64*28*28)
        self.decode = nn.Sequential(
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )
    def forward(self,z):
        x = self.fc(z)
        x = x.view(-1,64,28,28)
        return self.decode(x)

# Minority class for AE
minority_idx = np.where(y_train==0)[0]
X_minority = X_train[minority_idx]
X_minority_t = torch.tensor(X_minority).permute(0,3,1,2).float().to(device)

latent_dim = 128
encoder = Encoder(latent_dim).to(device)
decoder = Decoder(latent_dim).to(device)
ae_params = list(encoder.parameters()) + list(decoder.parameters())
optimizer_ae = torch.optim.Adam(ae_params, lr=1e-3)
criterion_ae = nn.MSELoss()

for epoch in range(5):
    encoder.train(); decoder.train()
    optimizer_ae.zero_grad()
    z = encoder(X_minority_t)
    X_rec = decoder(z)
    loss = criterion_ae(X_rec, X_minority_t)
    loss.backward()
    optimizer_ae.step()
    print(f"AE Epoch {epoch+1} Loss: {loss.item():.4f}")

# Latent space + SMOTE
encoder.eval()
with torch.no_grad():
    latent_minority = encoder(X_minority_t).cpu().numpy()

majority_idx = np.where(y_train==1)[0]
num_to_sample = len(minority_idx)
X_majority_sample = X_train[majority_idx[:num_to_sample]]
X_majority_sample_t = torch.tensor(X_majority_sample).permute(0,3,1,2).float().to(device)

with torch.no_grad():
    latent_majority = encoder(X_majority_sample_t).cpu().numpy()

X_combined = np.vstack([latent_minority, latent_majority])
y_combined = np.hstack([np.zeros(len(latent_minority)), np.ones(len(latent_majority))])

sm = SMOTE(random_state=SEED)
X_res, y_res = sm.fit_resample(X_combined, y_combined)

num_new = X_res.shape[0] - X_combined.shape[0]
synthetic_latent = X_res[-num_new:]

# Decode
decoder.eval()
synthetic_imgs = []
with torch.no_grad():
    for z in synthetic_latent:
        z_tensor = torch.tensor(z).float().to(device)
        img_syn = decoder(z_tensor.unsqueeze(0)).cpu().numpy()[0]
        synthetic_imgs.append(img_syn)

synthetic_imgs = np.stack(synthetic_imgs, axis=0)
synthetic_labels = np.zeros(len(synthetic_imgs))

print(f"\n📊 AE+SMOTE Sample Counts:")
print(f"   Original Class 0: {len(minority_idx)}")
print(f"   Original Class 1: {len(y_train) - len(minority_idx)}")
print(f"   Synthetic samples created: {len(synthetic_imgs)}")
print(f"   Final Class 0: {len(minority_idx) + len(synthetic_imgs)}")
print(f"   Final Class 1: {len(y_train) - len(minority_idx)}")
print(f"   Total training samples: {len(y_train) + len(synthetic_imgs)}")

X_train_ae = np.concatenate([X_train.transpose(0,3,1,2), synthetic_imgs], axis=0)
y_train_ae = np.concatenate([y_train, synthetic_labels], axis=0)

# Track AE+SMOTE samples
results["sample_tracking"]["ae_smote"] = {
    "original_class_0": int(len(minority_idx)),
    "original_class_1": int(len(y_train) - len(minority_idx)),
    "synthetic_class_0": int(len(synthetic_imgs)),
    "synthetic_class_1": 0,
    "final_class_0": int(len(minority_idx) + len(synthetic_imgs)),
    "final_class_1": int(len(y_train) - len(minority_idx)),
    "total": int(len(y_train) + len(synthetic_imgs))
}

train_ds_ae = CombinedDataset(X_train_ae, y_train_ae)
train_loader_ae = DataLoader(train_ds_ae, batch_size=32, shuffle=True)

cnn_ae = SimpleCNN()
cnn_ae, metrics_ae = train_model(cnn_ae, train_loader_ae, val_loader)
print("✅ AE+SMOTE dataset metrics:", metrics_ae)
results["metrics"]["ae_smote"] = metrics_ae

# ===========================
# 7️⃣ GAN-based augmentation
# ===========================
print("\n" + "="*50)
print("🔹 Training GAN")
print("="*50)

class GAN_Generator(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 64*28*28),
            nn.ReLU(),
            nn.Unflatten(1,(64,28,28)),
            nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32,16,3,stride=2,padding=1,output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,3,stride=2,padding=1,output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, z):
        return self.model(z)

class GAN_Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,16,3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16,32,3,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(32*56*56,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.model(x)

latent_dim = 128
G = GAN_Generator(latent_dim).to(device)
D = GAN_Discriminator().to(device)

criterion_gan = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=2e-4)
optimizer_D = torch.optim.Adam(D.parameters(), lr=2e-4)

minority_ds = PneumoniaDataset(X_train[minority_idx], y_train[minority_idx])
minority_loader = DataLoader(minority_ds, batch_size=32, shuffle=True)

for epoch in range(5):
    for imgs, _ in minority_loader:
        imgs = imgs.to(device)
        real_labels = torch.ones(imgs.size(0),1).to(device)
        fake_labels = torch.zeros(imgs.size(0),1).to(device)

        # D
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = G(z)
        D_real = D(imgs)
        D_fake = D(fake_imgs.detach())
        loss_D = criterion_gan(D_real, real_labels) + criterion_gan(D_fake, fake_labels)
        optimizer_D.zero_grad()
        loss_D.backward()
        optimizer_D.step()

        # G
        z = torch.randn(imgs.size(0), latent_dim).to(device)
        fake_imgs = G(z)
        D_fake = D(fake_imgs)
        loss_G = criterion_gan(D_fake, real_labels)
        optimizer_G.zero_grad()
        loss_G.backward()
        optimizer_G.step()

# Generate synthetic GAN images
num_synth = len(minority_idx)*2
z = torch.randn(num_synth, latent_dim).to(device)
with torch.no_grad():
    synthetic_gan = G(z).cpu().numpy()
synthetic_labels_gan = np.zeros(len(synthetic_gan))

print(f"\n📊 GAN Sample Counts:")
print(f"   Original Class 0: {len(minority_idx)}")
print(f"   Original Class 1: {len(y_train) - len(minority_idx)}")
print(f"   Synthetic samples created: {len(synthetic_gan)}")
print(f"   Final Class 0: {len(minority_idx) + len(synthetic_gan)}")
print(f"   Final Class 1: {len(y_train) - len(minority_idx)}")
print(f"   Total training samples: {len(y_train) + len(synthetic_gan)}")

X_train_gan = np.concatenate([X_train.transpose(0,3,1,2), synthetic_gan], axis=0)
y_train_gan = np.concatenate([y_train, synthetic_labels_gan], axis=0)

# Track GAN samples
results["sample_tracking"]["gan"] = {
    "original_class_0": int(len(minority_idx)),
    "original_class_1": int(len(y_train) - len(minority_idx)),
    "synthetic_class_0": int(len(synthetic_gan)),
    "synthetic_class_1": 0,
    "final_class_0": int(len(minority_idx) + len(synthetic_gan)),
    "final_class_1": int(len(y_train) - len(minority_idx)),
    "total": int(len(y_train) + len(synthetic_gan))
}

train_ds_gan = CombinedDataset(X_train_gan, y_train_gan)
train_loader_gan = DataLoader(train_ds_gan, batch_size=32, shuffle=True)

cnn_gan = SimpleCNN()
cnn_gan, metrics_gan = train_model(cnn_gan, train_loader_gan, val_loader)
print("✅ GAN-augmented dataset metrics:", metrics_gan)
results["metrics"]["gan"] = metrics_gan

# ===========================
# 8️⃣ Save metrics to JSON
# ===========================
def convert_to_native(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    else:
        return obj

# Convert results dict before saving
results_native = convert_to_native(results)

with open("results_metrics.json", "w") as f:
    json.dump(results_native, f, indent=4)

print("\n" + "="*50)
print("✅ Metrics saved to results_metrics.json")
print("="*50)

# Print summary
print("\n📋 SUMMARY:")
print(f"Original dataset - Class 0: {results['sample_tracking']['original']['class_0']}, Class 1: {results['sample_tracking']['original']['class_1']}")
print(f"AE+SMOTE - Synthetic Class 0: {results['sample_tracking']['ae_smote']['synthetic_class_0']}, Final Class 0: {results['sample_tracking']['ae_smote']['final_class_0']}")
print(f"GAN - Synthetic Class 0: {results['sample_tracking']['gan']['synthetic_class_0']}, Final Class 0: {results['sample_tracking']['gan']['final_class_0']}")