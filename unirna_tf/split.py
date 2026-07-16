import os
import sys

from Bio import SeqIO
from tqdm import tqdm


def split_fasta(input_file, output_dir, num_files):
    """Split a FASTA file into ``num_files`` shards with roughly even record counts."""
    num_files = int(num_files)
    if num_files <= 0:
        raise ValueError("num_files must be positive")

    total_records = sum(1 for _ in tqdm(SeqIO.parse(input_file, "fasta"), desc="Counting sequences"))

    records_per_file = total_records // num_files
    if total_records % num_files != 0:
        records_per_file += 1

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(input_file) as input_handle:
        record_iter = SeqIO.parse(input_handle, "fasta")
        for i in tqdm(range(num_files), desc="Writing splits"):
            filename = os.path.join(output_dir, "split_{}.fasta".format(i))
            with open(filename, "w") as out_handle:
                count = 0
                while count < records_per_file:
                    try:
                        seq_record = next(record_iter)
                        SeqIO.write(seq_record, out_handle, "fasta")
                        count += 1
                    except StopIteration:
                        break  # Reached the end of the source FASTA


if __name__ == "__main__":
    split_fasta(sys.argv[1], sys.argv[2], sys.argv[3])
