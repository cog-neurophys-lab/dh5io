import dh5io
import argparse
from dh5io.dh5file import DH5File


def display_tree(file_path: str):
    with DH5File(file_path, mode="r") as dh5_file:
        print(dh5_file)


def main():
    parser = argparse.ArgumentParser(
        description="Display the contents of a .dh5 file as a tree."
    )
    parser.add_argument("file", type=str, help="Path to the .dh5 file")
    args = parser.parse_args()

    try:
        display_tree(args.file)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
