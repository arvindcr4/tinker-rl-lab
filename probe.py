from collections import defaultdict
per_rate = defaultdict(list)
with open('experiments/results/p5p8/p8_iter152_acd_5seed_per_cell.tsv') as f:
    next(f)
    for line in f:
        parts = line.rstrip().split('\t')
        seed = int(parts[0])
        rate = float(parts[1])
        tier = parts[2]  # corrected: tier is column 3
        acd = float(parts[14])
        if tier != 'frontier_gpt4':
            continue
        per_rate[(seed, rate)].append(acd)
print(f"per_rate keys: {len(per_rate)}")
for (seed, rate), vals in sorted(per_rate.items()):
    print(f'seed={seed} rate={rate} mean={sum(vals)/len(vals):.4f} vals={vals}')
