import random
import numpy as np
from mir.nn.data_decorator import CQTPitchShifter

class CQTSpecAugmentPitchShifter(CQTPitchShifter):
    """
    Extends CQTPitchShifter to apply Time and Frequency Masking (SpecAugment)
    during synthetic pretraining to simulate acoustic masking, bleed, and noise.
    """
    def __init__(self, spec_dim, shift_low, shift_high, shift_step=3,
                 max_freq_mask=24, max_time_mask=60, num_freq_masks=2, num_time_masks=2,
                 mask_value=-80.0, enabled=True):
        super().__init__(spec_dim, shift_low, shift_high, shift_step)
        self.max_freq_mask = max_freq_mask
        self.max_time_mask = max_time_mask
        self.num_freq_masks = num_freq_masks
        self.num_time_masks = num_time_masks
        self.mask_value = mask_value
        self.enabled = enabled

    def pitch_shift(self, data, shift):
        # 1. Standard pitch shift slice
        spec = super().pitch_shift(data, shift).copy() # shape: (T, F=spec_dim)

        if not self.enabled:
            return spec

        t_len, f_len = spec.shape

        # 2. Frequency Masking (zero/silence out consecutive frequency bands)
        for _ in range(self.num_freq_masks):
            f_width = random.randint(0, self.max_freq_mask)
            f_start = random.randint(0, max(0, f_len - f_width))
            spec[:, f_start:f_start + f_width] = self.mask_value

        # 3. Time Masking (zero/silence out consecutive time frames)
        for _ in range(self.num_time_masks):
            t_width = random.randint(0, min(self.max_time_mask, t_len))
            t_start = random.randint(0, max(0, t_len - t_width))
            spec[t_start:t_start + t_width, :] = self.mask_value

        return spec
