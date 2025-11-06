# SPERLCL
scientific paper recommendation with LLMs and contrastive learning

实验运行步骤：

1.通过explanation_generation模块生成推荐正负向解释和embedding。

1.1 分别通过user和item文件夹里的程序生成用户画像和论文画像。

1.2 通过explanation里的程序生成正负向推荐解释。

1.3 通过emb里的程序生成正负向解释的embedding。

2. 通过运行main.py文件实现推荐分类。
   
命令：python main.py -c configs/triplet_config.json
