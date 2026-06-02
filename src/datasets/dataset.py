class Dataset:
    def __init__(
            self,
            name,
            dataset,
            n_classes,
            fmin,  # Cutoff frequency of the highpass filter
            fmax,  # Cutoff frequency of the lowpass filter
            tmax,  # Length of the trial periods
            sample_rate,
            use_channels_names,
    ):
        self.name = name
        self.dataset = dataset
        self.n_classes = n_classes
        self.fmin = fmin
        self.fmax = fmax
        self.tmin = 0
        self.tmax = tmax
        self.use_channels_names = use_channels_names

        # Target samples the transformer expects
        TARGET_SAMPLES = 1024
        duration = tmax

        # Find the largest clean divisor of 1024 that doesn't exceed the original Hz
        self.resample = self._calc_resample(TARGET_SAMPLES, duration, sample_rate)
        actual_samples = round(self.resample * duration)

        if actual_samples != TARGET_SAMPLES:
            raise ValueError(
                f"Could not find a clean resample rate for {duration}s window at "
                f"<={sample_rate} Hz that yields exactly {TARGET_SAMPLES} samples. "
                f"Best found: {self.resample} Hz → {actual_samples} samples. "
                f"Try adjusting tmin/tmax to a duration that divides evenly into {TARGET_SAMPLES} "
                f"(e.g. 1, 2, 4, 8 seconds)."
            )
        else:
            print(f"Resample rate of {self.resample}")

        # Batch size: largest power of 2 up to 128 that fits the resample rate
        self.batch_size = self._calc_batch_size(self.resample)

    @staticmethod
    def _calc_resample(target_samples, duration, max_hz):
        """Find the highest clean resample rate <= max_hz that gives exactly target_samples."""

        # All divisors of target_samples, descending
        divisors = sorted(
            [d for d in range(1, target_samples + 1) if target_samples % d == 0],
            reverse=True
        )

        rate = target_samples / duration
        if rate == int(rate) and int(rate) <= max_hz:
            return int(rate)

        for candidate in divisors:
            if candidate <= max_hz and (candidate * duration) == target_samples:
                return candidate

        return target_samples / duration  # may be non-integer, will fail the check above

    @staticmethod
    def _calc_batch_size(resample):
        """Largest power of 2 <= 128 scaled sensibly to resample rate."""
        if resample >= 256:
            return 64
        elif resample >= 128:
            return 32
        else:
            return 16

    def get_name(self):
        return self.name

    def get_dataset(self):
        return self.dataset

    def get_n_classes(self):
        return self.n_classes

    def get_fmin(self):
        return self.fmin

    def get_fmax(self):
        return self.fmax

    def get_tmin(self):
        return self.tmin

    def get_tmax(self):
        return self.tmax

    def get_resample(self):
        return self.resample

    def get_batch_size(self):
        return self.batch_size

    def get_use_channels_names(self):
        return self.use_channels_names