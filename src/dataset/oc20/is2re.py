from .common import OC20LmdbDataset, normalize_common_oc20_sample


class OC20IS2REDataset(OC20LmdbDataset):
    lmdb_norm_factor = (-1.525913953781128, 2.279365062713623)

    @staticmethod
    def normalize_raw_sample(raw_sample):
        return normalize_common_oc20_sample(
            raw_sample,
            include_force=False,
            energy_keys=("y_relaxed", "y", "target", "energy"),
        )

    def get_norm_factor(self):
        return self.lmdb_norm_factor
