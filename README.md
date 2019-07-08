# ProjectHW1
## Task
* Given 5000 images of car, detect the license block and unwrap it.

* It is guaranteed that each image with only one license block.

* How to donwload and uncompress these images(dataset)?

  Input the following in your code:
  ```
  !rm -rf ./ccpd5000/ && rm ccpd5000.tar.gz
  !wget https://github.com/amoshyc/cvlab-2019w-project/releases/download/v0.1/ccpd5000.tar.gz
  !tar zxf ccpd5000.tar.gz
  !ls ccpd5000/**/*.jpg | wc -l
  ```
 * Colab
 # Environment
 * What we need:
    * Python 3.6
    * Pytorch 1.0.0
    * torchvision 0.2.1
    * scikit-image 0.14.1
    * tqdm 4.29.0
 * How to setup those environment?
    * Input the followings in your code:
    ```
    !pip3 install torch torchvision tqdm matplotlib scikit-image
    ```
    ```
    !pip install tqdm matplotlib scikit-image
    ```
 # Code
 * Setup environment
    ```
    !pip install tqdm matplotlib scikit-image
    ```
    ```
    !pip install tqdm matplotlib scikit-image
    ```
* Download images
  ```
  !rm -rf ./ccpd5000/ && rm ccpd5000.tar.gz
  !wget https://github.com/amoshyc/cvlab-2019w-project/releases/download/v0.1/ccpd5000.tar.gz
  !tar zxf ccpd5000.tar.gz
  !ls ccpd5000/**/*.jpg | wc -l
  ```
  ```py
  from pathlib import Path

  img_dir = Path('./ccpd5000/train/')
  img_paths = img_dir.glob('*.jpg')
  img_paths = sorted(list(img_paths))
  ```
* Util
  ```py
  import warnings
  
  import torch
  import numpy as np
  from PIL import Image, ImageDraw
  from skimage import util
  from skimage.transform import ProjectiveTransform, warp
  
  def draw_kpts(img, kpts, c='red', r=2.0):
  
      draw = ImageDraw.Draw(img)
      kpts = kpts.view(4, 2)
      kpts = kpts * torch.FloatTensor(img.size)
      kpts = kpts.numpy().tolist()
      for (x, y) in kpts:
          draw.ellipse([x - r, y - r, x + r, y + r], fill=c)
      return img
  
  
  def draw_plate(img, kpts):
    
      src = np.float32([[96, 30], [0, 30], [0, 0], [96, 0]])
      dst = kpts.view(4, 2).numpy()
      dst = dst * np.float32(img.size)

      transform = ProjectiveTransform()
      transform.estimate(src, dst)
      with warnings.catch_warnings():
          warnings.simplefilter("ignore")
          warped = warp(np.array(img), transform, output_shape=(30, 96))
          warped = util.img_as_ubyte(warped)
      plate = Image.fromarray(warped)
      img.paste(plate)
      return img
  ```
 * Data
   ```py
   from PIL import Image
   from tqdm import tqdm
   from pathlib import Path
   
   import torch
   from torch.utils.data import DataLoader
   from torchvision.transforms import functional as tf
   
   class CCPD5000:
     def __init__(self, img_dir):
       self.img_dir = Path(img_dir)
       self.img_paths = self.img_dir.glob('*.jpg')
       self.img_paths = sorted(list(self.img_paths))
       
     def __len__(self):
       return len(self.img_paths)
     
     def __getitem__(self, idx):
       img_path = self.img_paths[idx]
       
       img = Image.open(img_path)
       W, H = img.size
       img = img.convert('RGB')
       img = img.resize((192, 320))
       img = tf.to_tensor(img)
       
       name = img_path.name
       token = name.split('-')[3]
       token = token.replace('&', '_')
       kpt = [float(val) for val in token.split('_')]
       kpt = torch.tensor(kpt)
       kpt = kpt.view(4, 2)
       kpt = kpt / torch.FloatTensor([W, H])
       kpt = kpt.view(-1)
       
       return img, kpt
     
   
   train_set = CCPD5000('./ccpd5000/train')
   ```
   ```py
   img = tf.to_pil_image(img)
   vis = draw_kpts(img, kpt, c='orange')
   vis = draw_plate(vis, kpt)
   vis.save('./check.jpg')
   
   from IPython import display
   display.Image('./check.jpg')
   ```
   ```py
   import torch
   from torch import nn
   from torch.nn import functional as F
   
   
   class ConvBlock(nn.Module):
       def __init__(self, cin, cout):
           super().__init__()
           self.conv1 = nn.Conv2d(cin, cout, (3, 3), padding=1)
           self.conv2 = nn.Conv2d(cout, cout, (3, 3), padding=1)
           self.bn1 = nn.BatchNorm2d(cout)
           self.bn2 = nn.BatchNorm2d(cout)
           self.act1 = nn.LeakyReLU()
           self.act2 = nn.LeakyReLU()
       
       def forward(self, x):
           x = self.act1(self.bn1(self.conv1(x)))
           x = self.act1(self.bn2(self.conv2(x)))
           return x
   
   
   class CCPDRegressor(nn.Module):
       def __init__(self):
           super().__init__()
           self.features = nn.Sequential(
               ConvBlock(3, 32),
               nn.MaxPool2d((8, 8)),
               ConvBlock(32, 32),
               nn.MaxPool2d((4, 4)),
               ConvBlock(32, 64),
               nn.MaxPool2d((2, 2)),
               ConvBlock(64, 64),
               nn.MaxPool2d((2, 2)),
           )
           self.regressor = nn.Sequential(
               nn.Linear(128, 32),
               nn.LeakyReLU(),
               nn.Linear(32, 8),
               nn.Sigmoid(),
           )
   
       def forward(self, x):
           N = x.size(0)
           x = self.features(x)
           x = x.view(N, -1)
           x = self.regressor(x)
           return x
   
        
   device = 'cuda'
   model = CCPDRegressor().to(device)
   img_b = torch.rand(16, 3, 192, 320).to(device)
   out_b = model(img_b)
   ```

 * Train
   ```py
   import json
   import random
   import numpy as np
   from tqdm import tqdm
   from pathlib import Path
   from datetime import datetime

   import pandas as pd
   import matplotlib.pyplot as plt
   plt.style.use('seaborn')
   
   import torch
   from torch import nn
   from torch.nn import functional as F
   from torchvision.utils import save_image
   from torch.utils.data import Subset, ConcatDataset, DataLoader
   from torchvision.transforms import functional as tf
   
   seed = 999
   random.seed(seed)
   np.random.seed(seed)
   torch.manual_seed(seed)
   torch.backends.cudnn.deterministic = True
   
   train_set = CCPD5000('./ccpd5000/train/')
   valid_set = CCPD5000('./ccpd5000/valid/')
   visul_set = ConcatDataset([
       Subset(train_set, random.sample(range(len(train_set)), 32)),
       Subset(valid_set, random.sample(range(len(valid_set)), 32)),
   ])
   train_loader = DataLoader(train_set, 32, shuffle=True, num_workers=3)
   valid_loader = DataLoader(valid_set, 32, shuffle=False, num_workers=1)
   visul_loader = DataLoader(visul_set, 32, shuffle=False, num_workers=1)
   
   device = 'cuda'
   model = CCPDRegressor().to(device)
   criterion = nn.L1Loss().to(device)
   optimizer = torch.optim.SGD(model.parameters(), lr=0.05)
   
   log_dir = Path('./log/') / f'{datetime.now():%Y.%m.%d-%H:%M:%S}'
   log_dir.mkdir(parents=True)
   print(log_dir)
   history = {
       'train_mae': [],
       'valid_mae': [],
       'train_mse': [],
       'valid_mse': [],
   }
   
   
   def train(pbar):
       model.train()
       mae_steps = []
       mse_steps = []
   
       for img_b, kpt_b in iter(train_loader):
           img_b = img_b.to(device)
           kpt_b = kpt_b.to(device)
   
           optimizer.zero_grad()
           pred_b = model(img_b)
           loss = criterion(pred_b, kpt_b)
           loss.backward()
           optimizer.step()
   
           mae = loss.detach().item()
           mse = F.mse_loss(pred_b.detach(), kpt_b.detach()).item()
           mae_steps.append(mae)
           mse_steps.append(mse)
   
           pbar.set_postfix(mae=mae, mse=mse)
           pbar.update(img_b.size(0))
   
       avg_mae = sum(mae_steps) / len(mae_steps)
       avg_mse = sum(mse_steps) / len(mse_steps)
       pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
       history['train_mae'].append(avg_mae)
       history['train_mse'].append(avg_mse)
   
   
   def valid(pbar):
       model.eval()
       mae_steps = []
       mse_steps = []
   
       for img_b, kpt_b in iter(valid_loader):
           img_b = img_b.to(device)
           kpt_b = kpt_b.to(device)
           pred_b = model(img_b)
           loss = criterion(pred_b, kpt_b)
           mae = loss.detach().item()
   
           mse = F.mse_loss(pred_b.detach(), kpt_b.detach()).item()
           mae_steps.append(mae)
           mse_steps.append(mse)
   
           pbar.set_postfix(mae=mae, mse=mse)
           pbar.update(img_b.size(0))
   
       avg_mae = sum(mae_steps) / len(mae_steps)
       avg_mse = sum(mse_steps) / len(mse_steps)
       pbar.set_postfix(avg_mae=f'{avg_mae:.5f}', avg_mse=f'{avg_mse:.5f}')
       history['valid_mae'].append(avg_mae)
       history['valid_mse'].append(avg_mse)
   
   
   def visul(pbar, epoch):
       model.eval()
       epoch_dir = log_dir / f'{epoch:03d}'
       epoch_dir.mkdir()
       for img_b, kpt_b in iter(visul_loader):
           pred_b = model(img_b.to(device)).cpu()
           for img, pred_kpt, true_kpt in zip(img_b, pred_b, kpt_b):
               img = tf.to_pil_image(img)
               vis = draw_plate(img, pred_kpt)
               vis = draw_kpts(vis, true_kpt, c='orange')
               vis = draw_kpts(vis, pred_kpt, c='red')
               vis.save(epoch_dir / f'{pbar.n:03d}.jpg')
               pbar.update()
   
   
   def log(epoch):
       with (log_dir / 'metrics.json').open('w') as f:
           json.dump(history, f)
   
       fig, ax = plt.subplots(2, 1, figsize=(6, 6), dpi=100)
       ax[0].set_title('MAE')
       ax[0].plot(range(epoch + 1), history['train_mae'], label='Train')
       ax[0].plot(range(epoch + 1), history['valid_mae'], label='Valid')
       ax[0].legend()
       ax[1].set_title('MSE')
       ax[1].plot(range(epoch + 1), history['train_mse'], label='Train')
       ax[1].plot(range(epoch + 1), history['valid_mse'], label='Valid')
       ax[1].legend()
       fig.savefig(str(log_dir / 'metrics.jpg'))
       plt.close()
   
   
   for epoch in range(10):
       print('Epoch', epoch, flush=True)
       with tqdm(total=len(train_set), desc='  Train') as pbar:
           train(pbar)
   
       with torch.no_grad():
           with tqdm(total=len(valid_set), desc='  Valid') as pbar:
               valid(pbar)
           with tqdm(total=len(visul_set), desc='  Visul') as pbar:
               visul(pbar, epoch)
           log(epoch)
   ```
   ```py
   display.Image(str(log_dir / 'metrics.jpg'))
   ```
   ```py
   display.Image(str(log_dir / '009' / '000.jpg'))
   ```
   ```py
   display.Image(str(log_dir / '009' / '032.jpg'))
   ```
# My full code
https://colab.research.google.com/drive/1_I698504kjOPTwuVTwMqMYEMRhjJnRQX#scrollTo=Id4_SmMo9EuZ
