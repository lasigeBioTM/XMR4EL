from datasets import load_dataset

def main():
    nlm_chem = load_dataset("bigbio/nlmchem")
    
    print(nlm_chem)

# Here is code
if __name__ == "__main__":
    main()