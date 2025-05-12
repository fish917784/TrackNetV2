Input (9, 360, 640)
  ↓
ConvBlock(9, 64) → ConvBlock(64, 64) → MaxPool2d
  ↓
ConvBlock(64, 128) → ConvBlock(128, 128) → MaxPool2d
  ↓
ConvBlock(128, 256) → ConvBlock(256, 256) → ConvBlock(256, 256) → MaxPool2d
  ↓
ConvBlock(256, 512) → ConvBlock(512, 512) → ConvBlock(512, 512)
  ↓
Upsample → ConvBlock(512, 256) → ConvBlock(256, 256) → ConvBlock(256, 256)
  ↓
Upsample → ConvBlock(256, 128) → ConvBlock(128, 128)
  ↓
Upsample → ConvBlock(128, 64) → ConvBlock(64, 64) → ConvBlock(64, 256)
  ↓
Reshape & (測試時) Softmax
'
### TrackNet 架構說明
1. 輸入層
   
   - 輸入資料為 9 個通道（通常是連續 3 幅 RGB 影像堆疊），尺寸為 360x640。
2. 特徵提取（下採樣）
   
   - 連續多層 ConvBlock（卷積、ReLU、BatchNorm）與 MaxPool2d 組合，逐步提取影像特徵並降低空間解析度，同時增加通道數。
   - 通道數依序為 64 → 128 → 256 → 512。
3. 特徵融合與上採樣
   
   - 在特徵提取後，透過 Upsample（上採樣）將特徵圖尺寸逐步放大，並經過多層 ConvBlock 進行特徵融合與細節恢復。
   - 通道數依序為 256 → 128 → 64 → 256。
4. 輸出層
   
   - 最終輸出經過 reshape，於測試時會加上 softmax，得到每個像素點屬於球體的機率分布。

   前半段多層 ConvBlock（卷積、ReLU、BatchNorm）與 MaxPool2d 的堆疊方式，與 VGG 網路非常相似。這種設計有助於逐步提取影像的高階特徵，並且是許多影像分割與定位任務常用的 backbone。

總結：TrackNet 的 backbone 架構本質上就是參考 VGG 的設計理念。



### Decoder 結構說明
1. Upsampling（上採樣）
   
   - 將特徵圖的空間尺寸逐步放大，恢復到原始輸入的解析度。
   - 每次上採樣後都會接一個卷積區塊（Conv+ReLU+BatchNorm）。
2. 卷積區塊（Conv+ReLU+BatchNorm）
   
   - 每次上採樣後，通道數會逐步減少（512→256→128→64→256→64），同時空間解析度增加。
3. 最後一層
   
   - 使用 1x1 卷積將通道數降為 1，對應單通道的熱力圖。
   - 最後經過 Softmax（或 Sigmoid）得到每個像素的預測機率。
### 圖中對應流程
- 輸入 ：來自 Encoder 的特徵圖 (batch, 512, 80, 45)
- 上採樣與卷積 ：
  - Upsample → Conv(512, 256) → Upsample → Conv(256, 128) → Upsample → Conv(128, 64) → Conv(64, 256) → Conv(256, 64)
- 輸出 ：最後一層 Conv(64, 1) + Softmax，得到 (batch, 1, 360, 640) 的熱力圖