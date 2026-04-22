import glob

def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Replacement 1: the equation
    old_eq = r"P(\text{usable group}) = 1 - p_0^G - (1-p_0)^G ."
    new_eq = r"P(\text{usable}) \approx \frac{1}{N}\sum_x \left[1 - (1-p_x)^G - p_x^G\right]."
    content = content.replace(old_eq, new_eq)

    old_eq_inline = r"P(\text{usable})\!=\!1-(1-p_0)^{G}-p_0^{G}"
    new_eq_inline = r"P(\text{usable}) \approx \frac{1}{N}\sum_x [1 - (1-p_x)^G - p_x^G]"
    content = content.replace(old_eq_inline, new_eq_inline)

    old_eq_inline_2 = r"P(\text{usable})\!\approx\!0"
    new_eq_inline_2 = r"P(\text{usable}) \approx 0"
    content = content.replace(old_eq_inline_2, new_eq_inline_2)

    old_eq_inline_3 = r"P(\text{usable})\!\approx\!1"
    new_eq_inline_3 = r"P(\text{usable}) \approx 1"
    content = content.replace(old_eq_inline_3, new_eq_inline_3)
    
    old_heur = r"p_0 > 1/G"
    new_heur = r"p_x > 1/G"
    content = content.replace(old_heur, new_heur)
    
    old_heur2 = r"p_0\!>\!1/G"
    new_heur2 = r"p_x\!>\!1/G"
    content = content.replace(old_heur2, new_heur2)

    # Some manual fixes for p_0 -> p_x in text
    content = content.replace("small-$p_0$", "small-$p_x$")
    content = content.replace("probability $p_0$ and the group size", "success rate $p_x$ for prompt $x$ and group size")

    with open(filepath, 'w') as f:
        f.write(content)

for file in glob.glob("paper/sections/*.tex"):
    fix_file(file)

print("Fixed paper sections")
