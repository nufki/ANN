import pandas as pd
import shutil
from pathlib import Path
import re

df = pd.read_csv('AdienceBenchmarkGenderAndAgeClassification/fold_0_data.txt', sep='\t')

female_dir = Path('src/main/resources/faces_dataset/female')
male_dir = Path('src/main/resources/faces_dataset/male')

for idx, row in df.iterrows():
    user_id = row['user_id']
    original_image = row['original_image']
    image_name = original_image

    gender_raw = row.get('gender')
    if pd.isna(gender_raw):
        print(f"Skipping NaN gender for {original_image}")
        continue

    gender = str(gender_raw).lower()

    # NEW: Age filter - skip if min OR max < 20
    age_raw = row.get('age')
    if pd.isna(age_raw):
        print(f"Skipping NaN age for {original_image}")
        continue

    age_str = str(age_raw)
    age_match = re.search(r'\((\d+),\s*(\d+)\)', age_str)
    if age_match:
        age_min = int(age_match.group(1))
        age_max = int(age_match.group(2))
        if age_min < 20 or age_max < 20:
            print(f"Skipping young age ({age_min}-{age_max}) for {original_image}")
            continue
    else:
        print(f"Skipping invalid age format '{age_str}' for {original_image}")
        continue

    print(f"user_id: {user_id}")
    print(f"original_image: {original_image}")

    # FIXED: Search user_id folder for ANY image containing original_image string
    user_folder = Path(
        'AdienceBenchmarkGenderAndAgeClassification/faces') / user_id

    print(f"user_folder exists: {user_folder.exists()}")
    print(f"user_folder absolute: {user_folder.absolute()}")

    matches = list(user_folder.glob(f"*{original_image}*"))
    print(f"Matches found: {len(matches)}")

    if not matches:
        print(f"Source not found containing '{original_image}' in {user_folder}")
        continue

    # NEW: Skip if multiple matches
    if len(matches) > 1:
        print(f"Multiple matches found for '{original_image}' in {user_folder}, skipping record")
        continue

    source_path = matches[0]  # First match

    if gender == 'f':
        target_dir = female_dir
    elif gender == 'm':
        target_dir = male_dir
    else:
        print(f"Skipping unknown gender '{gender}' for {image_name}")
        continue

    target_path = target_dir / image_name

    shutil.copy2(source_path, target_path)
    print(f"Copied {source_path.name} -> {target_dir.name}/{image_name}")
