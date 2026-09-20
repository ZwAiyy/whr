import jieba
import jieba.posseg as pseg

text = "这款葡萄酒呈现出迷人的深宝石红色泽，边缘泛着幽幽紫光，彰显出浓郁的活力。轻嗅之下，黑樱桃的馥郁果香与紫罗兰的优雅花香层层交织，其间还点缀着一丝若隐若现的烟熏气息，层次丰富。入口品味，单宁紧致且富有结构感，酒体饱满，悠长的回味在唇齿间久久萦绕。整体风格极为平衡，既有力量感又不失细腻优雅，无论是搭配红肉还是独自品鉴，都是一款极具魅力的佳酿。"

print("精确模式:", "/".join(jieba.cut(text, cut_all=False)))
print("全模式  :", "/".join(jieba.cut(text, cut_all=True)))
print("搜索模式:", "/".join(jieba.cut_for_search(text)))
print("词性标注:", " ".join(f"{w}/{p}" for w, p in pseg.cut(text)))