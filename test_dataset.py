from . import dataset

data_features_dir = '/mnt/c/datasets/xlsa17/data/AWA2'

def test_dataset():
    d = dataset.Dataset(data_features_dir)
    seen = set()
    for i, l in enumerate(d.labels):
        if l not in seen:
            print(f'class: {d.class_idx_to_name[l]}, example: {d.image_files[i]}')
            seen.add(l)
        assert d.class_idx_to_name[l] in d.image_files[i]
    assert len(seen) == 50
