import numpy as np

from .common import DEFAULT_OC22_PBC, OC22LmdbDataset, normalize_common_oc22_sample

OC22_LINREF_COEFF = np.array(
    [
        815220.7256646516,
        -3.434026360178362,
        -58304191664.18995,
        -3.803193647027048,
        -6.503872350843496,
        1245825730.325898,
        -7.358191058569556,
        -6.69350349363733,
        -6.664511258268206,
        -240389727.58167014,
        260765952.72144407,
        -2.2783517837524414,
        -4.125508785247803,
        -7.924036026000977,
        -9.409330576658249,
        31365899.601307064,
        -17328159.04369854,
        -8465758.535224315,
        20903228.18922116,
        -2.06325626373291,
        -5.28334903717041,
        -11.26516342163086,
        -11.802779197692871,
        -9.260477066040039,
        -8.626986503601074,
        -8.304924488067627,
        -6.075632095336914,
        -4.482022762298584,
        -1.7982172966003418,
        -2.7231812477111816,
        -1.5106525421142578,
        -4.461834669113159,
        -5.460780143737793,
        -3.5337538719177246,
        -2.3410415649414062,
        -23343.96598444777,
        -4690.488743327472,
        -1.8217530250549316,
        -4.899622917175293,
        -11.949428915977478,
        -12.438602209091187,
        -13.265165776014328,
        -8.212218284606934,
        156.37240938152786,
        -8.139511108398438,
        -5.7271928787231445,
        -3.3899335861206055,
        -1.2708024978637695,
        -0.5846166610717773,
        -3.2793075144290924,
        -4.573459148406982,
        -4.397623062133789,
        -1.428633689880371,
        -0.004567831754684448,
        0.0003960132598876953,
        -1.711404800415039,
        -4.954904556274414,
        0.0,
        -11.65596890449524,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        -10.306586265563965,
        -16.033945441246033,
        -16.202502012252808,
        -9.161677598953247,
        -12.682242393493652,
        -9.945820331573486,
        -6.414948463439941,
        -4.079216837882996,
        -1.0848636627197266,
        1.2394871711730957,
        -1.6953181028366089,
        -3.429639458656311,
        -3.8894996643066406,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float64,
)


def _apply_lin_ref(sample, lin_ref):
    z = sample["z"].long().numpy()
    correction = float(lin_ref[z].sum())
    sample["energy"] = sample["energy"] - correction
    sample["y"] = sample["energy"]
    return sample


def _load_lin_ref(lin_ref):
    if lin_ref is None:
        return None
    lin_ref = np.asarray(lin_ref, dtype=np.float64)
    if lin_ref.shape != OC22_LINREF_COEFF.shape:
        raise ValueError(
            f"Unexpected OC22 lin_ref shape: {lin_ref.shape}, expected {OC22_LINREF_COEFF.shape}"
        )
    return lin_ref.copy()


class OC22S2EFDataset(OC22LmdbDataset):
    lmdb_norm_factor = (0.0, 25.119809935106424)
    force_norm_factor = (0.0, 0.14759646356105804)

    def __init__(self, root, lmdb_path, lin_ref=OC22_LINREF_COEFF, cache_in_memory=False):
        self._lin_ref = _load_lin_ref(lin_ref)
        super().__init__(root=root, lmdb_path=lmdb_path, cache_in_memory=cache_in_memory)

    @staticmethod
    def normalize_raw_sample(raw_sample):
        sample = normalize_common_oc22_sample(
            raw_sample, include_force=True, energy_keys=("y", "energy", "target")
        )
        sample["pbc"] = DEFAULT_OC22_PBC.clone()
        return sample

    def postprocess_sample(self, sample):
        if self._lin_ref is not None:
            sample = _apply_lin_ref(sample, self._lin_ref)
        return sample
