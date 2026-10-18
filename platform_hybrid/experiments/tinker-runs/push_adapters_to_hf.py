"""Serial entrypoint for the shared receipt-verified adapter exporter."""

from parallel_push_hf import main

if __name__ == "__main__":
    main(max_workers=1)
