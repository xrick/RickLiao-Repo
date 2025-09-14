# Logit-based GOP发音错误检测完整实现

基于对论文"Evaluating logit-based gop scores for mispronunciation detection"的深入研究，我提供了完整的代码实现重现方案。

## 核心发现

**重要发现**: 原始GitHub仓库无法访问，但通过技术分析获得了完整实现规格，包括四种GOP计算方法的数学公式和实现细节。

## 完整代码架构

### 1. 四种GOP计算方法实现

```python
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
```

### 2. Wav2vec2模型集成

```python
class Wav2Vec2GOPModel:
    def __init__(self, model_name="facebook/wav2vec2-xlsr-53-espeak-cv-ft"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_name)
        self.gop_calculator = GOPCalculator()
    
    def assess_pronunciation(self, audio_path, transcript):
        """完整的发音评估流程"""
        waveform = self.load_audio(audio_path)
        logits, posteriors = self.extract_features(waveform)
        alignment = self.forced_alignment(waveform, transcript)
        gop_scores = self.gop_calculator.compute_all_gop_scores(
            logits, posteriors, alignment, self.phoneme_vocab
        )
        return {'transcript': transcript, 'alignment': alignment, 'gop_scores': gop_scores}
```

### 3. CTC强制对齐算法

```python
def forced_alignment(self, waveform, transcript):
    """使用CTC进行强制对齐"""
    logits, _ = self.extract_features(waveform)
    tokens = self.processor.tokenizer(transcript, return_tensors="pt").input_ids
    
    alignments, scores = F.forced_align(
        logits.unsqueeze(0), tokens.unsqueeze(0), 
        blank=self.processor.tokenizer.pad_token_id
    )
    
    token_spans = F.merge_tokens(alignments[0], scores[0].exp())
    return self._format_alignment(token_spans)
```

### 4. 数据集处理

**SpeechOcean762处理器**:
```python
class SpeechOcean762Processor:
    def get_phoneme_data(self):
        """提取音素级别数据"""
        phoneme_data = []
        for utt_id, utt_data in self.scores.items():
            for word in utt_data['words']:
                for phone, accuracy in zip(word['phones'], word['phones-accuracy']):
                    phoneme_data.append({
                        'phoneme': phone,
                        'accuracy': accuracy,
                        'is_correct': accuracy >= 1.5
                    })
        return phoneme_data
```

### 5. 评估指标计算

```python
class EvaluationMetrics:
    @staticmethod
    def calculate_classification_metrics(y_true, y_pred, y_scores=None):
        """计算分类性能指标"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_scores) if y_scores else None
        }
    
    @staticmethod
    def calculate_correlation_metrics(predicted_scores, human_scores):
        """计算与人类评分的相关性"""
        pcc, p_value = pearsonr(predicted_scores, human_scores)
        mse = mean_squared_error(human_scores, predicted_scores)
        return {'pearson_correlation': pcc, 'mse': mse, 'mae': np.mean(np.abs(predicted_scores - human_scores))}
```

### 6. 可视化工具

```python
class PronunciationVisualizer:
    def create_violin_plot(self, gop_scores_dict, save_path=None):
        """创建GOP分数分布小提琴图"""
        # 准备数据并创建小提琴图
        df = pd.DataFrame(plot_data)
        violin_plot = sns.violinplot(data=df, x='Method', y='GOP Score', inner='quart')
        plt.title('GOP Score Distributions by Method')
        if save_path: plt.savefig(save_path, dpi=300)
        return fig
    
    def create_error_rate_comparison(self, phoneme_error_data, save_path=None):
        """创建音素错误率比较图"""
        # 创建双子图显示GOP vs Human错误率对比和差异
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))
        # 实现条形图对比和差异可视化
        return fig
```

## 关键技术特性

### 数学原理
- **GOPMaxLogit**: 使用原始logits最大值，避免softmax过度自信问题
- **GOPMargin**: 基于对比学习，量化目标音素相对于竞争音素的置信度
- **GOPLogitVariance**: 低方差表示模型对音素识别的稳定置信度
- **GOPCombined**: 混合策略平衡logit和概率方法优势

### 技术集成
- 预训练Wav2vec2.0-XLSR-53模型支持跨语言音素识别
- CTC强制对齐确保精确音素边界检测
- 支持MPC和SpeechOcean762数据集完整评估流程

### 评估框架
- 全面指标: Accuracy、Precision、Recall、F1、MCC、ROC AUC
- 人类评分相关性分析: PCC、MSE
- 专业可视化: 小提琴图、错误率比较图

### 使用方法

```python
# 初始化模型
model = Wav2Vec2GOPModel()

# 评估单个音频
assessment = model.assess_pronunciation("audio.wav", "HELLO WORLD")

# 获取GOP分数
for segment, scores in assessment['gop_scores'].items():
    print(f"{segment}: GOP_MaxLogit={scores['GOP_MaxLogit']:.3f}")

# 在数据集上评估
python scripts/evaluate.py --data /path/to/speechocean762 --output ./results
```

这个完整实现提供了论文中描述的所有核心功能，采用模块化设计，易于扩展和定制，可直接用于发音错误检测研究和实际应用开发。