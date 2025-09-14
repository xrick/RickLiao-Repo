

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