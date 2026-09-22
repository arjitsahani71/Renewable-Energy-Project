import os
import requests


FILE_ID = "1GfNrdwD_7ooXBzaBIduuADTulZm9IKCB"
OUTPUT_PATH = "models/solar_power_model.pkl"


def download_google_drive_file(file_id, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    session = requests.Session()

    url = "https://drive.usercontent.google.com/download"

    params = {
        "id": file_id,
        "export": "download",
        "confirm": "t"
    }

    print("Downloading model from Google Drive...")

    response = session.get(
        url,
        params=params,
        stream=True
    )

    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    content_length = response.headers.get("Content-Length")

    print("Content-Type:", content_type)
    print("Content-Length:", content_length)

    if "text/html" in content_type.lower():
        raise RuntimeError(
            "Google Drive returned an HTML page instead of the model file."
        )

    total_downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                total_downloaded += len(chunk)

                mb = total_downloaded / (1024 * 1024)
                print(f"\rDownloaded: {mb:.2f} MB", end="")

    print()

    if total_downloaded < 1024 * 1024:
        raise RuntimeError(
            f"Downloaded file is too small: {total_downloaded} bytes."
        )

    print(f"Model downloaded successfully: {output_path}")
    print(f"Size: {total_downloaded / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    download_google_drive_file(FILE_ID, OUTPUT_PATH)