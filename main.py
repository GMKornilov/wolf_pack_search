from parse_wpa import read_from_file


def main():
    wpa = read_from_file()
    result_file = open("output.txt", "w")
    optimal_y, optimal_wolf = wpa.run()
    print(optimal_y, file=result_file)
    print(*optimal_wolf, file=result_file)
    result_file.close()


if __name__ == "__main__":
    main()
