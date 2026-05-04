from ViTclass.data.datasets import build_dtd_transforms, get_dataset_builder


def test_get_dataset_builder_returns_dtd_builder():
    builder = get_dataset_builder("dtd")
    assert callable(builder)


def test_build_dtd_transforms_returns_train_and_eval():
    train_tf, eval_tf = build_dtd_transforms(image_size=224)
    assert train_tf is not None
    assert eval_tf is not None
