import argparse

from ViTclass.engine.trainer import run_training


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()

