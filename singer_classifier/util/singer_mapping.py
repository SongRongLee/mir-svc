from pathlib import Path
from bidict import bidict


class SingerMapping:
    def __init__(self, data_dir=None, singer_ids=None):
        """
        Either data_dir or singer_ids should be provided.
        """
        if data_dir is not None:
            data_dir = Path(data_dir)
            self.singer_ids = self.data_dir_to_singer_ids(data_dir)
        elif singer_ids is not None:
            self.singer_ids = singer_ids
        else:
            raise ValueError('Either data_dir or singer_ids should be provided to SingerMapping.')

        self.singer_mapping_dict = bidict({singer_id: idx for idx, singer_id in enumerate(self.singer_ids)})

    def data_dir_to_singer_ids(self, data_dir):
        singer_ids = [singer_path.stem for singer_path in data_dir.iterdir()]
        return sorted(singer_ids)

    def singer_id_to_idx(self, singer_id):
        return self.singer_mapping_dict[singer_id]

    def singer_idx_to_id(self, singer_idx):
        return self.singer_mapping_dict.inverse[singer_idx]

    def get_singer_ids(self):
        return self.singer_ids.copy()

    def get_singer_idxs(self):
        return self.singer_mapping_dict.inverse.keys()
