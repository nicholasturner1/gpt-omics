"""Utility functions for reading *parts* of a pytorch_model.bin file.

This mostly copies code from torch.serialiation, but cuts and pastes things
together to give flexibility in which tensors we actually read. It also uses
some monkey patch shenanigans at the bottom.
"""
import io
import pickle

import torch
from torch import serialization as ser


def read_tensor_metadata(filename: str) -> list[str]:
    """Reads the metadata in pytorch_model.bin."""
    ser._check_dill_version(pickle)
    with ser._open_file_like(filename, "rb") as f:
        assert ser._is_zipfile(f), "These tools do not support PyTorch legacy storage."
        with ser._open_zipfile_reader(f) as zf:
            assert not ser._is_torchscript_zip(
                zf
            ), "These tools do not support TorchScript archives."
            return _read_tensor_metadata(zf)


def read_tensor_names(filename: str) -> list[str]:
    """Reads the metadata in pytorch_model.bin and returns the tensor names."""
    ser._check_dill_version(pickle)
    with ser._open_file_like(filename, "rb") as f:
        assert ser._is_zipfile(f), "These tools do not support PyTorch legacy storage."
        with ser._open_zipfile_reader(f) as zf:
            assert not ser._is_torchscript_zip(
                zf
            ), "These tools do not support TorchScript archives."
            tensor_meta = _read_tensor_metadata(zf)

            return list(tensor_meta.keys())


def read_tensors(filename: str, tensor_names: list[str]) -> dict[str, torch.tensor]:
    """Reads a named set of tensors from pytorch_model.bin."""
    ser._check_dill_version(pickle)
    with ser._open_file_like(filename, "rb") as f:
        assert ser._is_zipfile(f), "These tools do not support PyTorch legacy storage."
        with ser._open_zipfile_reader(f) as zf:
            assert not ser._is_torchscript_zip(
                zf
            ), "These tools do not support TorchScript archives."
            tensor_meta = _read_tensor_metadata(zf)

            return _read_tensors(zf, tensor_meta, tensor_names)


def read_tensor(filename: str, tensor_name: str) -> torch.tensor:
    """Reads a single tensor from pytorch_model.bin."""
    return read_tensors(filename, [tensor_name])[tensor_name]


def _read_tensor_metadata(zf) -> dict[str, tuple]:
    """Reads the stored metadata about each tensor within the zipfile."""
    loaded_keys = dict()
    metadata = list()

    # overrides an Unpickler method, just returning the tensor metadata
    def _persistent_load(saved_id):
        assert isinstance(saved_id, tuple)
        typename = ser._maybe_decode_ascii(saved_id[0])
        assert typename == "storage", f"unexpected typename: {typename}"

        data_type, key, location, size = saved_id[1:]

        if key not in loaded_keys:
            temp_key = len(loaded_keys)
            loaded_keys[key] = temp_key
            # nbytes = numel * torch._utils.element_size(dtype)
            metadata.append((data_type, key, location, size))

        # we have to return SOME kind of file location for this to work,
        # but maybe we can return some dummy instead
        return loaded_keys[key]

    data_file = io.BytesIO(zf.get_record("data.pkl"))

    with _monkey_patched_rebuild_tensor():
        # unpickling the data file
        unpickler = UnpicklerWrapper(data_file)
        unpickler.persistent_load = _persistent_load
        tensor_temp_id_lookup = unpickler.load()

    filled_metadata = dict()
    for (k, dummy_tensor) in tensor_temp_id_lookup.items():
        # unpack the dummy tensor
        temp_key = dummy_tensor[0].item()
        storage_offset = dummy_tensor[1].item()

        shape_ = list()
        i = 2
        while dummy_tensor[i] != -1:
            shape_.append(dummy_tensor[i].item())
            i += 1
        shape = tuple(shape_)

        stride_ = list()
        i += 1
        while i < len(dummy_tensor) and dummy_tensor[i] != -1:
            stride_.append(dummy_tensor[i].item())
            i += 1
        stride = tuple(stride_)

        filled_metadata[k] = metadata[temp_key] + (storage_offset, shape, stride)

    return filled_metadata


def _read_tensors(
    zf, metadata: dict[str, tuple], tensor_names: list[str]
) -> dict[str, torch.tensor]:
    """Reads the desired tensors from the zipfile."""
    assert all(name in metadata for name in tensor_names)

    tensors = dict()
    for name in tensor_names:
        data_type, key, location_, size, storage_offset, shape, stride = metadata[name]

        recordname = f"data/{key}"
        dtype = data_type(0).dtype

        storage = zf.get_storage_from_record(recordname, size, dtype).storage()
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, shape, stride)

        tensors[name] = tensor

    return tensors


def monkeypatch_rebuild_tensor(key, storage_offset, size, stride):
    return torch.tensor(
        # separating variable-length fields with -1s
        [key, storage_offset, *size, -1, *stride],
        dtype=torch.int,
        device="cpu",
    )


class _monkey_patched_rebuild_tensor:
    def __init__(self):
        self.orig_method = torch._utils._rebuild_tensor

    def __enter__(self):
        torch._utils._rebuild_tensor = monkeypatch_rebuild_tensor

    def __exit__(self, *args):
        torch._utils._rebuild_tensor = self.orig_method


# https://github.com/pytorch/pytorch/blob/1f1d0b30ce869c02c372d735c5856307e5edb4d6/torch/serialization.py#L1028
class UnpicklerWrapper(pickle.Unpickler):
    def find_class(self, mod_name, name):
        # if type(name) and "Storage" in name:
        #    try:
        #        return ser.StorageType(name)
        #    except KeyError:
        #        pass
        mod_name = {"torch.tensor": "torch._tensor"}.get(mod_name, mod_name)
        return super().find_class(mod_name, name)
