"""
這段程式碼是用來計算動態知識蒸餾損失（dynamic knowledge distillation loss）的。知識蒸餾是一種模型壓縮的技術，通過將複雜模型（教師模型）的知識轉移到簡化模型（學生模型）上來提高學生模型的性能。

在這個函數中，接受學生模型的預測結果 student_logits 和教師模型的預測結果 teacher_logits 作為輸入，還有一個溫度參數 temperature。

函數首先使用 softmax 函數計算學生模型預測結果的概率分佈，然後計算學生模型的熵（entropy），並通過對熵的正規化來獲得每個實例的權重 instance_weight。

接下來，對學生模型和教師模型的預測結果應用 log_softmax 函數，然後計算它們之間的 KL 散度（Kullback-Leibler divergence），並乘以溫度的平方以得到 batch_loss。

最後，將 batch_loss 乘以 instance_weight 後取平均值，得到加權的 KL 散度 weighted_kld 作為最終的損失值返回。

這個函數可以用於訓練過程中，通過最小化動態知識蒸餾損失來使學生模型學習教師模型的知識。

"""

import torch
import torch.nn as nn
from torch.nn import functional as F

def dynamic_kd_loss(student_logits, teacher_logits, temperature=1.0):

  with torch.no_grad():
    student_probs = F.softmax(student_logits, dim=-1)
    student_entropy = - torch.sum(student_probs * torch.log(student_probs + 1e-6), dim=1) # student entropy, (bsz, )
    # normalized entropy score by student uncertainty:
    # i.e.,  entropy / entropy_upper_bound
    # higher uncertainty indicates the student is more confusing about this instance
    instance_weight = student_entropy / torch.log(torch.ones_like(student_entropy) * student_logits.size(1))

  input = F.log_softmax(student_logits / temperature, dim=-1)
  target = F.softmax(teacher_logits / temperature, dim=-1)
  batch_loss = F.kl_div(input, target, reduction="none").sum(-1) * temperature ** 2  # bsz
  weighted_kld = torch.mean(batch_loss * instance_weight)

return weighted_kld
