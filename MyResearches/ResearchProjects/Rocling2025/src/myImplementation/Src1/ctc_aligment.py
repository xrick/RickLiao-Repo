

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