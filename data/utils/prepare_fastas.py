import pandas as pd

INPUT_CSV = "s1a_vs_background_dataset.csv"
POS_FASTA = "s1a_positive.fasta"
BG_FASTA = "background.fasta"

def write_fasta(df, path):
    with open(path, "w") as f:
        for _, row in df.iterrows():
            seq_id = str(row["accession"])
            seq = str(row["sequence"])
            f.write(f">{seq_id}\n{seq}\n")

def main():
    df = pd.read_csv(INPUT_CSV)

    pos = df[df["class_label"] == 1].copy()
    bg = df[df["class_label"] == 0].copy()

    write_fasta(pos, POS_FASTA)
    write_fasta(bg, BG_FASTA)

    print(f"Wrote {len(pos):,} positive sequences to {POS_FASTA}")
    print(f"Wrote {len(bg):,} background sequences to {BG_FASTA}")

if __name__ == "__main__":
    main()