# Conversion policy

The native converter plans a target qtype for each tensor before writing the
GGUF file. Planning is deterministic and depends on the requested file type,
tensor name and shape, detected architecture, selected policy, and explicit
overrides.

## Policies

- `uniform`: quantize eligible two-dimensional weight tensors with the requested
  base qtype.
- `comfy`: apply the converter's architecture and tensor-role rules.
- `dynamic`: start from `comfy` and promote selected tensor roles or layer
  positions to higher-precision qtypes.

The default is `comfy`.

## Overrides

- `--tensor-type PATTERN=QTYPE` assigns a target type to matching tensors.
- `--include PATTERN` forces matching two-dimensional tensors into quantization
  when their shape and qtype permit it.
- `--exclude PATTERN` keeps matching tensors unquantized.

Patterns use shell-style matching. Shape and qtype constraints can still force
a storage fallback.
