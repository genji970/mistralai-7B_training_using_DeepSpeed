import argparse
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Dataset Preprocessing Script")

    parser.add_argument(
        "--save_path",
        type=str,
        required=True,
        help="전처리된 데이터를 저장할 경로"
    )

    parser.add_argument(
        "--file_name",
        type=str,
        required=True,
        help="전처리된 데이터 파일 이름"
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        required=True,
        help="Hugging Face에서 불러올 데이터셋 이름 (예: nvidia/Nemotron-Post-Training-Dataset-v1)"
    )

    parser.add_argument(
        "--sample_ratio",
        type=float,
        help="load한 데이터셋에서 몇프로만 훈련에 사용할지"
    )

    parser.add_argument(
        "--rlhf_mode",
        action="store_true",
        help="rlhf 데이터셋 만들지 말지 여부"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="debugging_mode"
    )

    return parser.parse_args()



