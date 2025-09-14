

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