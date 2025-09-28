import numpy as np
import torch
import json
import struct
from typing import Dict, Any, Union, Optional

class SafetensorFileInfo:
    def __init__(self, filename:  str):
        self.filename = filename
        with open(filename, "rb") as f:
            self.header, self.header_size = self._read_header(f)
    
    def _read_header(self, file):
        """Read and parse the header from the safetensors file.

        Returns:
            tuple: (header_dict, header_size) containing parsed header and its size.
        """
        # Read header size (8 bytes, little-endian unsigned long long)
        header_size = struct.unpack("<Q", file.read(8))[0]
        # Read and decode header JSON
        header_json = file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size
    
    def keys(self):
        """Get all tensor keys in the file.

        Returns:
            list: List of tensor names (excludes metadata).
        """
        return [k for k in self.header.keys() if k != "__metadata__"]


class MemoryEfficientSafeOpen:
    """Memory-efficient reader for safetensors files.

    This class provides a memory-efficient way to read tensors from safetensors files
    by using memory mapping for large tensors and avoiding unnecessary copies.
    """

    def __init__(self, filename):
        """Initialize the SafeTensor reader.

        Args:
            filename (str): Path to the safetensors file to read.
        """
        self.filename = filename
        self.file = open(filename, "rb")
        self.header, self.header_size = self._read_header()

    def __enter__(self):
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager and close file."""
        self.file.close()

    def keys(self):
        """Get all tensor keys in the file.

        Returns:
            list: List of tensor names (excludes metadata).
        """
        return [k for k in self.header.keys() if k != "__metadata__"]

    def metadata(self) -> Dict[str, str]:
        """Get metadata from the file.

        Returns:
            Dict[str, str]: Metadata dictionary.
        """
        return self.header.get("__metadata__", {})

    def _read_header(self):
        """Read and parse the header from the safetensors file.

        Returns:
            tuple: (header_dict, header_size) containing parsed header and its size.
        """
        # Read header size (8 bytes, little-endian unsigned long long)
        header_size = struct.unpack("<Q", self.file.read(8))[0]
        # Read and decode header JSON
        header_json = self.file.read(header_size).decode("utf-8")
        return json.loads(header_json), header_size

    def get_tensor(self, key: str, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None):
        """Load a tensor from the file with memory-efficient strategies.

        **Note:**
        If device is 'cuda' , the transfer to GPU is done efficiently using pinned memory and non-blocking transfer.
        So you must ensure that the transfer is completed before using the tensor (e.g., by `torch.cuda.synchronize()`).

        If the tensor is large (>10MB) and the target device is CUDA, memory mapping with numpy.memmap is used to avoid intermediate copies.

        Args:
            key (str): Name of the tensor to load.
            device (Optional[torch.device]): Target device for the tensor.
            dtype (Optional[torch.dtype]): Target dtype for the tensor.

        Returns:
            torch.Tensor: The loaded tensor.

        Raises:
            KeyError: If the tensor key is not found in the file.
        """
        if key not in self.header:
            raise KeyError(f"Tensor '{key}' not found in the file")

        metadata = self.header[key]
        offset_start, offset_end = metadata["data_offsets"]
        num_bytes = offset_end - offset_start

        original_dtype = self._get_torch_dtype(metadata["dtype"])
        target_dtype = dtype if dtype is not None else original_dtype

        # Handle empty tensors
        if num_bytes == 0:
            return torch.empty(metadata["shape"], dtype=target_dtype, device=device)

        # Determine if we should use pinned memory for GPU transfer
        non_blocking = device is not None and device.type == "cuda"

        # Calculate absolute file offset
        tensor_offset = self.header_size + 8 + offset_start  # adjust offset by header size

        # Memory mapping strategy for large tensors to GPU
        # Use memmap for large tensors to avoid intermediate copies.
        # If device is cpu, tensor is not copied to gpu, so using memmap locks the file, which is not desired.
        # So we only use memmap if device is not cpu.
        if num_bytes > 10 * 1024 * 1024 and device is not None and device.type != "cpu":
            # Create memory map for zero-copy reading
            mm = np.memmap(self.filename, mode="c", dtype=np.uint8, offset=tensor_offset, shape=(num_bytes,))
            byte_tensor = torch.from_numpy(mm)  # zero copy
            del mm

            # Deserialize tensor (view and reshape)
            cpu_tensor = self._deserialize_tensor(byte_tensor, metadata)  # view and reshape
            del byte_tensor

            # Transfer to target device and dtype
            gpu_tensor = cpu_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)
            del cpu_tensor
            return gpu_tensor

        # Standard file reading strategy for smaller tensors or CPU target
        # seek to the specified position
        self.file.seek(tensor_offset)

        # read directly into a numpy array by numpy.fromfile without intermediate copy
        numpy_array = np.fromfile(self.file, dtype=np.uint8, count=num_bytes)
        byte_tensor = torch.from_numpy(numpy_array)
        del numpy_array

        # deserialize (view and reshape)
        deserialized_tensor = self._deserialize_tensor(byte_tensor, metadata)
        del byte_tensor

        # cast to target dtype and move to device
        return deserialized_tensor.to(device=device, dtype=target_dtype, non_blocking=non_blocking)

    def _deserialize_tensor(self, byte_tensor: torch.Tensor, metadata: Dict):
        """Deserialize byte tensor to the correct shape and dtype.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor from file.
            metadata (Dict): Tensor metadata containing dtype and shape info.

        Returns:
            torch.Tensor: Deserialized tensor with correct shape and dtype.
        """
        dtype = self._get_torch_dtype(metadata["dtype"])
        shape = metadata["shape"]

        # Handle special float8 types
        if metadata["dtype"] in ["F8_E5M2", "F8_E4M3"]:
            return self._convert_float8(byte_tensor, metadata["dtype"], shape)

        # Standard conversion: view as target dtype and reshape
        return byte_tensor.view(dtype).reshape(shape)

    @staticmethod
    def _get_torch_dtype(dtype_str):
        """Convert string dtype to PyTorch dtype.

        Args:
            dtype_str (str): String representation of the dtype.

        Returns:
            torch.dtype: Corresponding PyTorch dtype.
        """
        # Standard dtype mappings
        dtype_map = {
            "F64": torch.float64,
            "F32": torch.float32,
            "F16": torch.float16,
            "BF16": torch.bfloat16,
            "I64": torch.int64,
            "I32": torch.int32,
            "I16": torch.int16,
            "I8": torch.int8,
            "U8": torch.uint8,
            "BOOL": torch.bool,
        }
        # Add float8 types if available in PyTorch version
        if hasattr(torch, "float8_e5m2"):
            dtype_map["F8_E5M2"] = torch.float8_e5m2
        if hasattr(torch, "float8_e4m3fn"):
            dtype_map["F8_E4M3"] = torch.float8_e4m3fn
        return dtype_map.get(dtype_str)

    @staticmethod
    def _convert_float8(byte_tensor, dtype_str, shape):
        """Convert byte tensor to float8 format if supported.

        Args:
            byte_tensor (torch.Tensor): Raw byte tensor.
            dtype_str (str): Float8 dtype string ("F8_E5M2" or "F8_E4M3").
            shape (tuple): Target tensor shape.

        Returns:
            torch.Tensor: Tensor with float8 dtype.

        Raises:
            ValueError: If float8 type is not supported in current PyTorch version.
        """
        # Convert to specific float8 types if available
        if dtype_str == "F8_E5M2" and hasattr(torch, "float8_e5m2"):
            return byte_tensor.view(torch.float8_e5m2).reshape(shape)
        elif dtype_str == "F8_E4M3" and hasattr(torch, "float8_e4m3fn"):
            return byte_tensor.view(torch.float8_e4m3fn).reshape(shape)
        else:
            # Float8 not supported in this PyTorch version
            raise ValueError(f"Unsupported float8 type: {dtype_str} (upgrade PyTorch to support float8 types)")

class MultipleSafetensorLoader:

    def __init__(self, model_index_file: str):
        """Initialize the MultipleSafetensorLoader.

        Args:
            model_index_file (str): Path to the model index JSON file.
        """
        f = open(model_index_file, "r") 
        self.index = json.load(f)
        f.close()
        import pathlib
        path = pathlib.Path(model_index_file)
        self.path = path.parent
        
    
    def load_dict(self) -> Dict:
        """Load all tensors from the multiple safetensors files as specified in the index.

        Returns:
            Dict: Dictionary of all tensors loaded from the files.
        """
        import pathlib
        all_tensors = {}
        files = set(self.index.get('weight_map', {}).values())
        files = sorted(files)
        for file in files:
            file = self.path / pathlib.Path(file)
            print(f"load {file}")
            with MemoryEfficientSafeOpen(file) as safe:
                for key in safe.keys():
                    all_tensors[key] = safe.get_tensor(key)
        return all_tensors
        


if __name__ == "__main__":
    # Example usage
    import argparse
    parser = argparse.ArgumentParser(description="Memory-efficient safetensors reader example")
    parser.add_argument("--filename", type=str, help="Path to the safetensors file", default="./models/vibevoice/model.safetensors.index.json")
    args = parser.parse_args()
    models = MultipleSafetensorLoader(args.filename).load_dict()
    print(models.keys())