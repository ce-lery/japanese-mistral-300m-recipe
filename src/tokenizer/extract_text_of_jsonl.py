import json
import time
import argparse
from argparse import Namespace

def extract_text_of_jsonl(args:Namespace)->None:
    error_count = 0
    print(f"Start extract_of_jsonl {args.input_path}.")

    # Load jsonl.
    with open(args.input_path, "r", encoding="utf-8") as file, open(args.output_path, "w",encoding="utf-8") as output:
        while True:
            try:
                line = file.readline()
                # If end of file, break while loop.
                if not line:
                    break 

                data = json.loads(line)
                output.write(data["text"])
                output.write("\n")

            except Exception as e:
                error_count += 1
                print(f"exception: line: ({e})")
                continue

    print("Completed.")
    print(f"Failed reading line number:{error_count}")

if __name__ == "__main__": 
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_path",help="",default="")
    parser.add_argument("--output_path",help="",default="")
    
    args = parser.parse_args()
    extract_text_of_jsonl(args)

