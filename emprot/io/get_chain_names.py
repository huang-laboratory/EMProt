import pickle

if __name__ == '__main__':
    chain_names = []
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    # One-letter chain id
    for i in range(len(letters)):
        chain_names.append(letters[i])
    # Two-letter chain id
    for i in range(len(letters)):
        for k in range(len(letters)):
            chain_names.append(letters[i]+letters[k])
    # In total we have 62 + 62 * 62 ~= 4000 chain ids
    # Three or more-letter chain ids
    for l in range(len(letters)):
        for m in range(len(letters)):
            for n in range(len(letters)):
                chain_names.append(letters[l]+letters[m]+letters[n])
    # 62 * 62 * 62 = 238328 chains

    chain_names = "|".join(chain_names)

    with open("chain_names.txt", "w") as f:
        f.write(chain_names + "\n")
