from pathlib import Path

from PIL import Image
import imagehash


def compute_phash(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image))


def hamming_distance(hash_a: str, hash_b: str) -> int:
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")

