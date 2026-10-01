"""List a GGUF's tensors under a prefix, with type and shape.

usage: list_tensors.py <gguf> <name-prefix>
"""
import sys

sys.path.insert(0, "/home/peb/llama.cpp-glm5/gguf-py")
from gguf import GGUFReader


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    r = GGUFReader(sys.argv[1])
    prefix = sys.argv[2]
    n = 0
    for t in r.tensors:
        if t.name.startswith(prefix):
            print("  %-44s type %-4s shape %s" % (t.name, t.tensor_type, list(t.shape)))
            n += 1
    print("  (%d tensors under %r)" % (n, prefix))
    return 0


if __name__ == "__main__":
    sys.exit(main())
