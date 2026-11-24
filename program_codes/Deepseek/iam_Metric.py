
import os
import csv
import json
import re


# ============================================================
# CONFIGURATION
# ============================================================

IGNORE_EMPTY_VALUES = True

CSV_OUTPUT_FILE = "csv_txt_value_level_metrics.csv"

# Optional: exclude CSV columns that contain metadata rather
# than ground-truth content.
# Example: ["image_width", "image_height"]
IGNORE_CSV_COLUMNS = []


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_value(value):
    """
    Ignore capitalization and all whitespace.
    Preserve punctuation and symbols.
    """
    if value is None:
        return ""

    value = str(value)
    value = re.sub(r"\s+", "", value)
    value = value.lower()

    return value


def normalize_header(header):
    return re.sub(r"[^a-z0-9]", "", str(header).lower())


def get_file_stem(filename):
    """
    0000.png -> 0000
    0000.txt -> 0000
    0000.json -> 0000
    """
    filename = str(filename).strip().replace("\\", "/")
    filename = os.path.basename(filename)

    return os.path.splitext(filename)[0].strip().lower()


# ============================================================
# EXTRACT GROUND-TRUTH VALUES FROM CSV CELLS
# ============================================================

def extract_csv_cell_values(cell):
    if cell is None:
        return []

    text = str(cell).strip()

    if text == "":
        return []

    # Support JSON objects/lists stored in CSV cells.
    if text.startswith("{") or text.startswith("["):
        try:
            parsed = json.loads(text)
            return extract_json_values(parsed)
        except json.JSONDecodeError:
            pass

    normalized = normalize_value(text)

    if IGNORE_EMPTY_VALUES and normalized == "":
        return []

    return [normalized]


def extract_json_values(data):
    """
    Recursively extract all leaf values from JSON content
    embedded in a ground-truth CSV cell.
    """
    values = []

    if isinstance(data, dict):
        for value in data.values():
            values.extend(extract_json_values(value))

    elif isinstance(data, list):
        for item in data:
            values.extend(extract_json_values(item))

    else:
        normalized = normalize_value(data)

        if not IGNORE_EMPTY_VALUES or normalized != "":
            values.append(normalized)

    return values


# ============================================================
# DETECT CSV FILENAME COLUMN
# ============================================================

def detect_filename_column(headers, requested_column=""):

    if requested_column:
        for header in headers:
            if header.strip().lower() == requested_column.strip().lower():
                return header

        raise ValueError(
            f"Filename column '{requested_column}' was not found.\n"
            f"Available columns: {headers}"
        )

    aliases = {
        "filename",
        "file",
        "filepath",
        "image",
        "imagename",
        "imagefile",
        "imagefilename",
        "imagepath",
        "img",
        "imgname",
        "imgfilename",
        "documentname",
        "documentfile",
    }

    for header in headers:
        if normalize_header(header) in aliases:
            return header

    print(
        "\nWARNING: Could not automatically identify the filename column."
    )
    print(f"Using the first column: {headers[0]}")

    return headers[0]


# ============================================================
# READ GROUND TRUTH CSV
# ============================================================

def read_ground_truth_csv(csv_path, requested_column=""):

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"Ground-truth CSV not found:\n{csv_path}"
        )

    ground_truth = {}

    with open(
        csv_path,
        "r",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        reader = csv.DictReader(file)
        headers = reader.fieldnames

        if not headers:
            raise ValueError(
                "The CSV is empty or has no header row."
            )

        filename_column = detect_filename_column(
            headers,
            requested_column
        )

        print(
            f"\nDetected filename column: {filename_column}"
        )

        ignored_columns = {
            normalize_header(column)
            for column in IGNORE_CSV_COLUMNS
        }

        for row_number, row in enumerate(reader, start=2):

            raw_filename = row.get(filename_column)

            if raw_filename is None or not str(raw_filename).strip():
                print(
                    f"WARNING: CSV row {row_number} has no filename."
                )
                continue

            stem = get_file_stem(raw_filename)

            if not stem:
                continue

            if stem not in ground_truth:
                ground_truth[stem] = {
                    "filename": str(raw_filename).strip(),
                    "values": [],
                }

            # All columns other than the filename are considered
            # ground-truth values unless explicitly ignored.
            for header in headers:

                if header == filename_column:
                    continue

                if normalize_header(header) in ignored_columns:
                    continue

                cell = row.get(header)

                ground_truth[stem]["values"].extend(
                    extract_csv_cell_values(cell)
                )

    return ground_truth


# ============================================================
# READ PREDICTION TXT FILE
# ============================================================

def read_prediction_txt(filepath, gt_value_count):

    try:
        with open(
            filepath,
            "r",
            encoding="utf-8-sig"
        ) as file:
            text = file.read()

    except UnicodeDecodeError:
        with open(
            filepath,
            "r",
            encoding="latin-1"
        ) as file:
            text = file.read()

    except OSError as error:
        print(f"ERROR reading {filepath}: {error}")
        return []

    if not text.strip():
        return []

    # When a sample has one GT value, treat the entire TXT
    # as one prediction. This is suitable for transcription tasks.
    if gt_value_count <= 1:
        normalized = normalize_value(text)

        if IGNORE_EMPTY_VALUES and normalized == "":
            return []

        return [normalized]

    # When a sample has multiple GT values, treat each non-empty
    # line as one predicted value.
    #
    # Lines like "Name: John Smith" become "John Smith".
    # Lines like "Amount = 2500" become "2500".
    pred_values = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        # Remove simple field labels if present.
        match = re.match(
            r"^\s*[^:=\t]{1,80}\s*[:=\t]\s*(.*?)\s*$",
            line
        )

        if match:
            line = match.group(1).strip()

        normalized = normalize_value(line)

        if IGNORE_EMPTY_VALUES and normalized == "":
            continue

        pred_values.append(normalized)

    return pred_values


# ============================================================
# FIND RESULT TXT FILES
# ============================================================

def get_txt_lookup(results_folder):

    lookup = {}

    for filename in sorted(os.listdir(results_folder)):

        filepath = os.path.join(results_folder, filename)

        if not os.path.isfile(filepath):
            continue

        if not filename.lower().endswith(".txt"):
            continue

        stem = get_file_stem(filename)

        if stem in lookup:
            print(
                f"WARNING: Multiple TXT files have the same "
                f"filename stem: {stem}"
            )
            continue

        lookup[stem] = filename

    return lookup


# ============================================================
# ORDER-INDEPENDENT MATCHING
# ============================================================

def match_values(gt_values, pred_values):
    """
    Match values in any order.
    Each prediction is used at most once.
    Duplicate values are handled one-to-one.
    """

    used_prediction = [False] * len(pred_values)

    matched = 0
    unmatched_gt = []

    for gt_value in gt_values:

        found_match = False

        for index, pred_value in enumerate(pred_values):

            if used_prediction[index]:
                continue

            if gt_value == pred_value:

                used_prediction[index] = True
                matched += 1
                found_match = True
                break

        if not found_match:
            unmatched_gt.append(gt_value)

    unmatched_prediction = [
        pred_values[index]
        for index in range(len(pred_values))
        if not used_prediction[index]
    ]

    return matched, unmatched_gt, unmatched_prediction


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(gt_count, pred_count, matched_count):

    # Proportion of ground-truth values that were matched.
    accuracy = (
        matched_count / gt_count * 100
        if gt_count > 0 else 0.0
    )

    precision = (
        matched_count / pred_count * 100
        if pred_count > 0 else 0.0
    )

    recall = (
        matched_count / gt_count * 100
        if gt_count > 0 else 0.0
    )

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0 else 0.0
    )

    return accuracy, precision, recall, f1


# ============================================================
# MAIN EVALUATION
# ============================================================

def evaluate(csv_path, results_folder, requested_column=""):

    if not os.path.isdir(results_folder):
        raise NotADirectoryError(
            f"Results folder does not exist:\n{results_folder}"
        )

    ground_truth = read_ground_truth_csv(
        csv_path,
        requested_column
    )

    if not ground_truth:
        print("No valid ground-truth records found.")
        return

    result_lookup = get_txt_lookup(results_folder)

    gt_stems = set(ground_truth)
    result_stems = set(result_lookup)

    missing_txt = sorted(gt_stems - result_stems)
    extra_txt = sorted(result_stems - gt_stems)

    csv_rows = []

    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_unmatched_gt = 0
    total_extra_pred = 0

    files_evaluated = 0

    print("\n" + "=" * 80)
    print("CSV GROUND TRUTH VS TXT RESULTS")
    print("=" * 80)

    print(f"Ground-truth images : {len(ground_truth)}")
    print(f"Results TXT files   : {len(result_lookup)}")
    print(f"Matching filenames  : {len(gt_stems & result_stems)}")
    print(f"Missing TXT files   : {len(missing_txt)}")
    print(f"Extra TXT files     : {len(extra_txt)}")

    # Evaluate every GT sample, including samples with no TXT file.
    for stem in sorted(gt_stems):

        record = ground_truth[stem]
        gt_values = record["values"]
        gt_filename = record["filename"]

        txt_filename = result_lookup.get(stem)

        pred_values = []

        if txt_filename is not None:

            txt_path = os.path.join(
                results_folder,
                txt_filename
            )

            pred_values = read_prediction_txt(
                txt_path,
                len(gt_values)
            )

        matched, unmatched_gt, unmatched_pred = match_values(
            gt_values,
            pred_values
        )

        gt_count = len(gt_values)
        pred_count = len(pred_values)

        unmatched_gt_count = len(unmatched_gt)
        extra_pred_count = len(unmatched_pred)

        accuracy, precision, recall, f1 = calculate_metrics(
            gt_count,
            pred_count,
            matched
        )

        total_gt += gt_count
        total_pred += pred_count
        total_matched += matched
        total_unmatched_gt += unmatched_gt_count
        total_extra_pred += extra_pred_count

        files_evaluated += 1

        print("\n" + "-" * 80)
        print(f"Image             : {gt_filename}")
        print(f"Prediction TXT    : {txt_filename or 'MISSING'}")
        print(f"GT values         : {gt_count}")
        print(f"Prediction values : {pred_count}")
        print(f"Matched values    : {matched}")
        print(f"Unmatched GT      : {unmatched_gt_count}")
        print(f"Extra predictions : {extra_pred_count}")
        print(f"Accuracy          : {accuracy:.2f}%")
        print(f"Precision         : {precision:.2f}%")
        print(f"Recall            : {recall:.2f}%")
        print(f"F1 score          : {f1:.2f}%")

        if unmatched_gt:
            print("Unmatched GT values:")
            for value in unmatched_gt:
                print(f"  - {value}")

        if unmatched_pred:
            print("Extra prediction values:")
            for value in unmatched_pred:
                print(f"  - {value}")

        csv_rows.append({
            "Image/File": gt_filename,
            "Prediction TXT File": txt_filename or "MISSING",
            "Ground Truth Values": gt_count,
            "Prediction Values": pred_count,
            "Matched Values": matched,
            "Unmatched GT Count": unmatched_gt_count,
            "Extra Prediction Count": extra_pred_count,
            "Accuracy (%)": round(accuracy, 2),
            "Precision (%)": round(precision, 2),
            "Recall (%)": round(recall, 2),
            "F1 Score (%)": round(f1, 2),
            "Unmatched GT Values": " | ".join(unmatched_gt),
            "Extra Prediction Values": " | ".join(unmatched_pred),
            "Ground Truth Images": "",
            "Results TXT Files": "",
            "Images Evaluated": "",
        })

    # Overall metrics across all images.
    overall_accuracy, overall_precision, overall_recall, overall_f1 = (
        calculate_metrics(
            total_gt,
            total_pred,
            total_matched
        )
    )

    print("\n\n" + "=" * 80)
    print("OVERALL RESULTS")
    print("=" * 80)

    print(f"Ground-truth images     : {len(ground_truth)}")
    print(f"Results TXT files       : {len(result_lookup)}")
    print(f"Images evaluated        : {files_evaluated}")
    print(f"Total GT values         : {total_gt}")
    print(f"Total prediction values : {total_pred}")
    print(f"Matched values          : {total_matched}")
    print(f"Unmatched GT values     : {total_unmatched_gt}")
    print(f"Extra predicted values  : {total_extra_pred}")
    print(f"\nOVERALL ACCURACY        : {overall_accuracy:.2f}%")
    print(f"VALUE PRECISION         : {overall_precision:.2f}%")
    print(f"VALUE RECALL            : {overall_recall:.2f}%")
    print(f"VALUE F1 SCORE          : {overall_f1:.2f}%")

    # Final summary row.
    csv_rows.append({
        "Image/File": "OVERALL",
        "Prediction TXT File": "",
        "Ground Truth Values": total_gt,
        "Prediction Values": total_pred,
        "Matched Values": total_matched,
        "Unmatched GT Count": total_unmatched_gt,
        "Extra Prediction Count": total_extra_pred,
        "Accuracy (%)": round(overall_accuracy, 2),
        "Precision (%)": round(overall_precision, 2),
        "Recall (%)": round(overall_recall, 2),
        "F1 Score (%)": round(overall_f1, 2),
        "Unmatched GT Values": "",
        "Extra Prediction Values": "",
        "Ground Truth Images": len(ground_truth),
        "Results TXT Files": len(result_lookup),
        "Images Evaluated": files_evaluated,
    })

    # Save metrics CSV INSIDE the results folder.
    output_path = os.path.join(
        results_folder,
        CSV_OUTPUT_FILE
    )

    fieldnames = [
        "Image/File",
        "Prediction TXT File",
        "Ground Truth Values",
        "Prediction Values",
        "Matched Values",
        "Unmatched GT Count",
        "Extra Prediction Count",
        "Accuracy (%)",
        "Precision (%)",
        "Recall (%)",
        "F1 Score (%)",
        "Unmatched GT Values",
        "Extra Prediction Values",
        "Ground Truth Images",
        "Results TXT Files",
        "Images Evaluated",
    ]

    with open(
        output_path,
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(csv_rows)

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"Metrics CSV saved to:\n{output_path}")

    if missing_txt:
        print("\nGround-truth images without TXT results:")
        for stem in missing_txt:
            print(f"  - {ground_truth[stem]['filename']}")

    if extra_txt:
        print("\nTXT results without ground truth:")
        for stem in extra_txt:
            print(f"  - {result_lookup[stem]}")


# ============================================================
# ENTRY POINT
# ============================================================

def main():

    print("=" * 80)
    print("CSV GROUND TRUTH / TXT RESULTS EVALUATOR")
    print("=" * 80)

    csv_path = input(
        "\nEnter the ground-truth CSV file path: "
    ).strip().strip('"')

    results_folder = input(
        "Enter the results folder path: "
    ).strip().strip('"')

    requested_column = input(
        "Enter the CSV filename column name "
        "(press Enter for auto-detection): "
    ).strip()

    try:
        evaluate(
            csv_path,
            results_folder,
            requested_column
        )

    except Exception as error:
        print(f"\nERROR: {error}")


if __name__ == "__main__":
    main()
