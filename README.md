# pairwise-hair-aniso-highlight

## 领域

游戏、图形与多媒体

## 主题

毛发各向异性高光

## 说明

切线连续、LOD 近似与毛发阴影通道。

初始状态下 `graphics/hair_aniso_highlight.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
