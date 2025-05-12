### EfficientNetV2 架構特點

1. **Fused-MBConv 模組**  
   - 除了原本的 MBConv（Mobile Inverted Bottleneck Conv）外，EfficientNetV2 引入了 Fused-MBConv，將部分深度可分離卷積與標準卷積融合，提升訓練速度與早期特徵提取能力。

2. **更快的訓練速度**  
   - 架構設計針對訓練效率優化，能在更短時間內達到更高準確率，適合大規模資料集。

3. **更靈活的複合縮放**  
   - EfficientNetV2 進一步優化了複合縮放策略，根據不同模型規模（S/M/L/XL）自動調整深度、寬度與解析度。

4. **正則化與訓練技巧**  
   - 採用更先進的正則化方法（如 progressive learning），提升泛化能力。

5. **更高效能**  
   - 在 ImageNet 等資料集上，EfficientNetV2 在相同或更少參數下，準確率超越 EfficientNetV1、ResNet、VGG 等架構。

---

### EfficientNetV2 架構簡圖

```
Input
  ↓
Stem (Conv3x3)
  ↓
多層 Fused-MBConv & MBConv Block（根據複合縮放調整）
  ↓
Head (Conv1x1, Pooling, FC)
  ↓
Output
```
