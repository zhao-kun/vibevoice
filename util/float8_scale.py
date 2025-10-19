import torch
import torch.nn as nn

from typing import Union, List

from util.vibevoice_norm import QwenRMSNorm

# These codes are referenced from ComfyUI (https://github.com/comfyanonymous/ComfyUI) for 
# handling the torch native float8_e4m3fn operations. The main idea is to cast weights 
# and biases to dtype of input before the forward pass. 

def cast_bias_weight(module, input :torch.Tensor=None, dtype: torch.dtype=None, device: torch.device=None, bias_dtype: torch.dtype=None):
    if input is None:
        return module.weight, module.bias if hasattr(module, 'bias') else None

    if dtype is None:
        dtype = input.dtype

    if bias_dtype is None:
        bias_dtype = dtype

    if device is None:
        device = input.device
    bias = None
    if hasattr(module, 'bias') and module.bias is not None:
        bias = module.bias.to(device=device, dtype=bias_dtype)

    weight = module.weight.to(device=device, dtype=dtype)
    return weight, bias

class ResetParametersMixin:
    def reset_parameters(self):
        return None

class AutoCast:
    class Linear(nn.Linear, ResetParametersMixin):

        def fp8_linear(self, input):
            dtype = self.weight.dtype
            if dtype not in [torch.float8_e4m3fn]:
                return None

            tensor_2d = False
            if len(input.shape) == 2:
                tensor_2d = True
                input = input.unsqueeze(1)

            input_shape = input.shape
            input_dtype = input.dtype
            if len(input.shape) == 3:
                w, bias = cast_bias_weight(self, input, dtype=dtype, bias_dtype=input_dtype)
                w = w.t()

                scale_weight = self.scale_weight if hasattr(self, 'scale_weight') else None
                scale_input = self.scale_input if hasattr(self, 'scale_input') else None
                if scale_weight is None:
                    scale_weight = torch.ones((), device=input.device, dtype=torch.float32)
                else:
                    scale_weight = scale_weight.to(input.device)

                if scale_input is None:
                    scale_input = torch.ones((), device=input.device, dtype=torch.float32)
                    input = torch.clamp(input, min=-448, max=448, out=input)
                    input = input.reshape(-1, input_shape[2]).to(dtype).contiguous()
                else:
                    scale_input = scale_input.to(input.device)
                    input = (input * (1.0 / scale_input).to(input_dtype)).reshape(-1, input_shape[2]).to(dtype).contiguous()

                if bias is not None:
                    o = torch._scaled_mm(input, w, out_dtype=input_dtype, bias=bias, scale_a=scale_input, scale_b=scale_weight)
                else:
                    o = torch._scaled_mm(input, w, out_dtype=input_dtype, scale_a=scale_input, scale_b=scale_weight)

                if isinstance(o, tuple):
                    o = o[0]

                if tensor_2d:
                    return o.reshape(input_shape[0], -1)
        
                return o.reshape((-1, input_shape[1], self.weight.shape[0]))
            return None

        def forward_comfy_cast_weights(self, input):
            fp8_out = self.fp8_linear(input)
            if fp8_out is not None:
                return fp8_out

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.linear(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn: 
                return self.forward_comfy_cast_weights(*args, **kwargs)
            return super().forward(*args, **kwargs)

    class Conv1d(torch.nn.Conv1d, ResetParametersMixin):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Conv2d(torch.nn.Conv2d, ResetParametersMixin):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return self._conv_forward(input, weight, bias)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class GroupNorm(torch.nn.GroupNorm, ResetParametersMixin):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.group_norm(input, self.num_groups, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class LayerNorm(torch.nn.LayerNorm, ResetParametersMixin):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
                bias = None
            return torch.nn.functional.layer_norm(input, self.normalized_shape, weight, bias, self.eps)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class QwenRMSNorm(QwenRMSNorm, ResetParametersMixin):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input):
            if self.weight is not None:
                weight, bias = cast_bias_weight(self, input)
            else:
                weight = None
            return super().forward(input, weight)  # TODO: switch to commented out line when old torch is deprecated
            # return torch.nn.functional.rms_norm(input, self.normalized_shape, weight, self.eps)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose2d(torch.nn.ConvTranspose2d, ResetParametersMixin):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 2
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose2d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class ConvTranspose1d(torch.nn.ConvTranspose1d, ResetParametersMixin):
        def reset_parameters(self):
            return None

        def forward_comfy_cast_weights(self, input, output_size=None):
            num_spatial_dims = 1
            output_padding = self._output_padding(
                input, output_size, self.stride, self.padding, self.kernel_size,
                num_spatial_dims, self.dilation)

            weight, bias = cast_bias_weight(self, input)
            return torch.nn.functional.conv_transpose1d(
                input, weight, bias, self.stride, self.padding,
                output_padding, self.groups, self.dilation)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                return super().forward(*args, **kwargs)

    class Embedding(torch.nn.Embedding, ResetParametersMixin):
        def reset_parameters(self):
            self.bias = None
            return None

        def forward_comfy_cast_weights(self, input, out_dtype=None):
            output_dtype = torch.bfloat16
            weight, bias = cast_bias_weight(self, device=input.device, dtype=out_dtype)
            return torch.nn.functional.embedding(input, weight, self.padding_idx, self.max_norm, self.norm_type, self.scale_grad_by_freq, self.sparse).to(dtype=output_dtype)

        def forward(self, *args, **kwargs):
            if self.weight.dtype == torch.float8_e4m3fn:
                return self.forward_comfy_cast_weights(*args, **kwargs)
            else:
                if "out_dtype" in kwargs:
                    kwargs.pop("out_dtype")
                return super().forward(*args, **kwargs)
