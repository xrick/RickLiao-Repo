# four gop calculation implementations
import torch

class GOPCalculator:
    def gop_max_logit(self, logits, target_frames):
        """GOPMaxLogit：捕捉目标音素的峰值置信度"""
        phoneme_logits = logits[target_frames]
        max_logits = torch.max(phoneme_logits, dim=-1).values
        return torch.mean(max_logits).item()
    
    def gop_margin(self, logits, target_phoneme_id, target_frames):
        """GOPMargin：计算目标音素与最强竞争者之间的平均差距"""
        phoneme_logits = logits[target_frames]
        target_logits = phoneme_logits[:, target_phoneme_id]
        # 计算非目标音素的平均logit
        other_logits = phoneme_logits[:, mask]
        margin = target_logits - torch.mean(other_logits, dim=-1)
        return torch.mean(margin).item()
    
    def gop_logit_variance(self, logits, target_phoneme_id, target_frames):
        """GOPLogitVariance：测量模型置信度的变异性"""
        target_logits = logits[target_frames, target_phoneme_id]
        variance = torch.var(target_logits, unbiased=False)
        return -variance.item()  # 负方差表示高质量发音
    
    def gop_combined(self, logits, posteriors, target_phoneme_id, target_frames, alpha=0.5):
        """GOPCombined：结合logit和概率方法的混合指标"""
        gop_margin = self.gop_margin(logits, target_phoneme_id, target_frames)
        gop_dnn = self.gop_dnn_traditional(posteriors, target_phoneme_id, target_frames)
        return alpha * gop_margin + (1 - alpha) * gop_dnn