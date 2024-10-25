import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import open3d as o3d
import copy

from scnn_tm import models
from scnn_tm import utils



# parser = argparse.ArgumentParser()
CONFIG_FILE = Path(__file__).parent / "config" / "default_test_configs.yaml"
with open(CONFIG_FILE, "r") as f:
    config = yaml.safe_load(f)

args = utils.scnn_io.Dict2Class(config)


def test_cv_dataset_factory_k_fold_split():
    ''' Test if the global_map_id patches are a) unique and b) complete for a k fold cross validation '''
    test_args = copy.deepcopy(args)

    # Test case arguments
    data_sets = utils.sweep_utils.train_data_set_by_key(args.data_set_key, args.voxel_size)
    test_args.num_splits = len(data_sets)
    feature_set = utils.scnn_io.generate_feature_set_from_key("occ")

    test_patches = []
    total_number_of_patches = 0
    for fold_i in range(test_args.num_splits):

        test_args.current_fold = fold_i
        data_set_factory = models.ForestTravDataSetFactory.ForestTravDataSetFactory(data_set=data_sets,
                                                                             feature_set=feature_set,
                                                                             params=test_args)
        test_patches.extend(
            [patch_i.global_patch_id for patch_i in data_set_factory.test_data_set.patches])
        total_number_of_patches = data_set_factory.total_number_of_patches

    assert len(test_patches) == len(set(test_patches))  # Uniqueness
    assert (total_number_of_patches) == len(set(test_patches))  # Completeness


def test_patch_data_loader_no_data_mutation():
    test_args = copy.deepcopy(args)
    sample_number = 10000

    data_set_coords = np.random.uniform(low=-20.0, high=20.0, size=(sample_number, 3))
    data_set_features = np.random.uniform(size=(sample_number, 15))
    data_set_labels = np.random.uniform(size=(sample_number, 1))

    # Copy original data points
    original_data_set_coords = copy.deepcopy(data_set_coords)
    original_data_set_features = copy.deepcopy(data_set_features)
    original_data_set_labels = copy.deepcopy(data_set_labels)

    # We make one large patch so the order of coordinates stays the same (in theory)
    data_patches = [models.ForestTravDataSet.MapPatch(map_id=0, bounds=np.array(
        [-20.0, -20.0, -20.0, 20.0, 20.0, 20.0]), global_patch_id=0)]

    # Modify the test args
    test_args.use_data_augmentation = False

    torch_data_set = models.ForestTravDataSet.ForestTravDataSet(
        data_sets_coords=[data_set_coords],
        data_sets_feature_data_scaled=[data_set_features],
        data_sets_labels=[data_set_labels],
        patches=data_patches,
        voxel_size=test_args.voxel_size,
        nvoxel_leaf=test_args.nvoxel_leaf,
        use_data_augmentation=test_args.use_data_augmentation,
        data_augmentor=None,
    )

    # Test that data has not been modified ()
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)

    assert not (np.isclose(batch_data["coordinates"], original_data_set_coords)).all()
    assert (np.isclose(batch_data["coordinates"] *
                       test_args.voxel_size, original_data_set_coords)).all()
    assert (np.isclose(batch_data["features"], original_data_set_features)).all()
    assert (np.isclose(batch_data["labels"], original_data_set_labels)).all()


def test_patch_data_loader_feature_mutation():
    test_args = copy.deepcopy(args)
    sample_number = 10000

    data_set_coords = np.random.uniform(low=-20.0, high=20.0, size=(sample_number, 3))
    data_set_features = np.random.uniform(size=(sample_number, 15))
    data_set_labels = np.random.uniform(size=(sample_number, 1))

    # Copy original data points
    original_data_set_coords = copy.deepcopy(data_set_coords)
    original_data_set_features = copy.deepcopy(data_set_features)
    original_data_set_labels = copy.deepcopy(data_set_labels)

    # We make one large patch so the order of coordinates stays the same (in theory)
    data_patches = [models.ForestTravDataSet.MapPatch(map_id=0, bounds=np.array(
        [-20.0, -20.0, -20.0, 20.0, 20.0, 20.0]), global_patch_id=0)]

    # Modify the test args
    test_args.use_data_augmentation = True

    torch_data_set = models.ForestTravDataSet.ForestTravDataSet(
        data_sets_coords=[data_set_coords],
        data_sets_feature_data_scaled=[data_set_features],
        data_sets_labels=[data_set_labels],
        patches=data_patches,
        voxel_size=test_args.voxel_size,
        nvoxel_leaf=test_args.nvoxel_leaf,
        use_data_augmentation=test_args.use_data_augmentation,
        data_augmentor=models.DataAugmenter.DataPatchAugmenter(
            test_args.voxel_size, noise_chance=1.0, noise_std=0.05),
    )

    # Test that data has not been modified ()
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)

    #
    assert not (np.isclose(batch_data["features"], original_data_set_features)).all()

    assert not (np.isclose(batch_data["coordinates"], original_data_set_coords)).all()
    assert (np.isclose(batch_data["coordinates"] *
                       test_args.voxel_size, original_data_set_coords)).all()
    assert (np.isclose(batch_data["labels"], original_data_set_labels)).all()


def test_patch_data_loader_all_mutation():
    " We test all mutiation except pruning"
    test_args = copy.deepcopy(args)
    sample_number = 10000

    data_set_coords = np.random.uniform(low=-20.0, high=20.0, size=(sample_number, 3))
    data_set_features = np.random.uniform(size=(sample_number, 15))
    data_set_labels = np.random.uniform(size=(sample_number, 1))

    # Copy original data points
    original_data_set_coords = copy.deepcopy(data_set_coords)
    original_data_set_features = copy.deepcopy(data_set_features)
    original_data_set_labels = copy.deepcopy(data_set_labels)

    # We make one large patch so the order of coordinates stays the same (in theory)
    data_patches = [models.ForestTravDataSet.MapPatch(map_id=0, bounds=np.array(
        [-20.0, -20.0, -20.0, 20.0, 20.0, 20.0]), global_patch_id=0)]

    # Modify the test args
    test_args.use_data_augmentation = True

    torch_data_set = models.ForestTravDataSet.ForestTravDataSet(
        data_sets_coords=[data_set_coords],
        data_sets_feature_data_scaled=[data_set_features],
        data_sets_labels=[data_set_labels],
        patches=data_patches,
        voxel_size=test_args.voxel_size,
        nvoxel_leaf=test_args.nvoxel_leaf,
        use_data_augmentation=test_args.use_data_augmentation,
        data_augmentor=models.DataAugmenter.DataPatchAugmenter(
            test_args.voxel_size, noise_chance=1.0, noise_std=0.05, batch_nvoxel_displacement=10, batch_translation_chance=1.0, mirror_chance=1.0),
    )

    # Test that data has not been modified ()
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)
    batch_data = torch_data_set.__getitem__(0)

    #
    assert not (np.isclose(batch_data["features"], original_data_set_features)).all()

    assert not (np.isclose(batch_data["coordinates"], original_data_set_coords)).all()
    assert not (np.isclose(batch_data["coordinates"] *
                           test_args.voxel_size, original_data_set_coords)).all()
    assert (np.isclose(batch_data["labels"], original_data_set_labels)).all()


################### PATCH SAMPLER #################
