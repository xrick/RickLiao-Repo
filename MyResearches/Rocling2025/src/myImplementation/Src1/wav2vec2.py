

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