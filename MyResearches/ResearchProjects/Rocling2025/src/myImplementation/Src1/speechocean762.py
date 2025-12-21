

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