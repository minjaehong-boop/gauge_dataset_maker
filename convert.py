import ffmpeg
import os


def convert_webm_to_mp4_ffmpeg_python(input_path, output_path):
    print(f"Converting file '{input_path}'...")
    try:
        (
            ffmpeg.input(input_path)
            .output(output_path, vcodec="libx264", acodec="aac")
            .run(overwrite_output=True, quiet=True)
        )
        print(f"✅ Successfully created '{output_path}'.")
    except ffmpeg.Error as e:
        print("❌ An FFmpeg error occurred:")
    except FileNotFoundError:
        print(
            "❌ Error: 'ffmpeg' not found. Make sure FFmpeg is installed and registered in the system's PATH."
        )


if __name__ == "__main__":
    input_file = "/e/gauge/inputs/부하변동_영상_0903.webm"
    output_file = os.path.splitext(input_file)[0] + ".mp4"

    if os.path.exists(input_file):
        convert_webm_to_mp4_ffmpeg_python(input_file, output_file)
    else:
        print(f"'{input_file}' not found.")
