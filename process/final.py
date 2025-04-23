import json
import argparse


def calculate_max_f1_scores(data):
    summe_data = data["summe"]
    tvsum_data = data["tvsum"]

    common_keys = set(summe_data.keys()) & set(tvsum_data.keys())

    max_combined = -float('inf')
    best_key = None
    summe_f1 = None
    tvsum_f1 = None

    for key in common_keys:
        # 计算summe的均值
        summe_scores = summe_data[key]
        avg_summe = sum(summe_scores) / len(summe_scores)

        # 计算tvsum的均值
        tvsum_scores = tvsum_data[key]
        avg_tvsum = sum(tvsum_scores) / len(tvsum_scores)

        combined = avg_summe + avg_tvsum

        if combined > max_combined:
            max_combined = combined
            best_key = key
            summe_f1 = avg_summe
            tvsum_f1 = avg_tvsum

    return best_key, summe_f1, tvsum_f1


def main():
    parser = argparse.ArgumentParser(
        description='Calculate maximum combined F1 scores from JSON data.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input JSON file')
    args = parser.parse_args()

    try:
        with open(args.input, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file {args.input} does not exist.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file {args.input} is not a valid JSON.")
        return

    best_key, summe_score, tvsum_score = calculate_max_f1_scores(data)

    print(f"Best Key: {best_key}")
    print(f"Summe F1-score: {summe_score:.2f}")
    print(f"TVSum F1-score: {tvsum_score:.2f}")
    print(f"Combined F1-score: {(summe_score + tvsum_score):.2f}")


if __name__ == "__main__":
    main()
