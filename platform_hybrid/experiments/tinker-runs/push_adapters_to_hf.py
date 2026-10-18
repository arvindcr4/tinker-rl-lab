"""Serial entrypoint for the shared receipt-verified adapter exporter."""
from parallel_push_hf import branch_of, load_cells, main, push

if __name__ == "__main__":
    main(max_workers=1)
