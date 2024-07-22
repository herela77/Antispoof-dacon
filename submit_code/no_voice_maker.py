import os
import subprocess
import tensorflow as tf
import shutil

def check_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("GPU is available.")
    else:
        print("GPU is not available.")

def separate_vocal(input_path, output_dir):
    command = f'spleeter separate -p spleeter:2stems -o "{output_dir}" -c ogg "{input_path}"'
    try:
        result = subprocess.run(command, check=True, shell=True, capture_output=True, text=True)
        print(result.stdout)
        print(f"Processed {input_path} successfully")

        # Move accompaniment file to the main output directory
        input_filename = os.path.splitext(os.path.basename(input_path))[0]
        accompaniment_file = os.path.join(output_dir, input_filename, 'accompaniment.ogg')
        new_accompaniment_file = os.path.join(output_dir, f'{input_filename}_accompaniment.ogg')

        if os.path.exists(accompaniment_file):
            shutil.move(accompaniment_file, new_accompaniment_file)
            print(f"Moved accompaniment file to: {new_accompaniment_file}")

        # Clean up: Remove the vocals file and the now-empty directory
        vocals_file = os.path.join(output_dir, input_filename, 'vocals.ogg')
        if os.path.exists(vocals_file):
            os.remove(vocals_file)
            print(f"Removed vocals file: {vocals_file}")

        # Remove the subdirectory
        subdirectory = os.path.join(output_dir, input_filename)
        if os.path.exists(subdirectory):
            os.rmdir(subdirectory)
            print(f"Removed subdirectory: {subdirectory}")
        
    except subprocess.CalledProcessError as e:
        print(f"Failed to process {input_path}: {e.stderr}")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.ogg') or file.endswith('.wav'):
                input_path = os.path.join(root, file)
                separate_vocal(input_path, output_dir)

if __name__ == "__main__":
    input_dir = "./unlabeled_data"
    output_dir = "./novoice"
    check_gpu()
    process_directory(input_dir, output_dir)
