import os

def build_context(output_file="context.txt"):
    py_files = [f for f in os.listdir('.') if f.endswith('.py') and os.path.isfile(f)]
    with open(output_file, 'w', encoding='utf-8') as out:
        for fname in py_files:
            out.write(f"# BEGIN {fname}\n")
            with open(fname, 'r', encoding='utf-8') as src:
                out.write(src.read())
            out.write(f"\n# END {fname}\n\n")

if __name__ == "__main__":
    build_context()
