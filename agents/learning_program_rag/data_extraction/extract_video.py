import os
import sys
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up ffmpeg path for Windows
try:
    import imageio_ffmpeg as iio
    ffmpeg_exe = iio.get_ffmpeg_exe()
    ffmpeg_path = os.path.dirname(ffmpeg_exe)

    # Add ffmpeg to PATH
    if ffmpeg_path not in os.environ.get('PATH', ''):
        os.environ['PATH'] = os.environ.get('PATH', '') + os.pathsep + ffmpeg_path
        print(f"Added ffmpeg to PATH: {ffmpeg_path}")

    # Set IMAGEIO_FFMPEG_EXE environment variable for MoviePy
    os.environ['IMAGEIO_FFMPEG_EXE'] = ffmpeg_exe
    print(f"Set IMAGEIO_FFMPEG_EXE to: {ffmpeg_exe}")

except ImportError:
    print("Warning: imageio-ffmpeg not found, ffmpeg may not work")

from moviepy import VideoFileClip
import whisper
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import uuid

import config
from util.embeddings import get_embedding_model


class VideoProcessor:
    """Process video files: extract audio, transcribe, and store in vector DB."""

    def __init__(self):
        """Initialize video processor with Whisper model and Qdrant client."""
        print("Initializing Video Processor...")

        # Initialize Whisper model
        print(f"Loading Whisper model: {config.WHISPER_MODEL}")
        self.whisper_model = whisper.load_model(config.WHISPER_MODEL)

        # Initialize text splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            length_function=len,
        )

        # Initialize embeddings using modular architecture
        self.embeddings = get_embedding_model("huggingface")

        # Initialize Qdrant client
        print(f"Connecting to Qdrant at {config.QDRANT_HOST}:{config.QDRANT_PORT}")
        self.qdrant_client = QdrantClient(
            host=config.QDRANT_HOST,
            port=config.QDRANT_PORT
        )

        # Create collection if it doesn't exist
        self._ensure_collection()

    def _ensure_collection(self):
        """Ensure Qdrant collection exists."""
        try:
            collections = self.qdrant_client.get_collections().collections
            collection_names = [c.name for c in collections]

            if config.COLLECTION_NAME not in collection_names:
                print(f"Creating collection: {config.COLLECTION_NAME}")
                self.qdrant_client.create_collection(
                    collection_name=config.COLLECTION_NAME,
                    vectors_config=VectorParams(
                        size=config.EMBEDDING_DIMENSION,
                        distance=Distance.COSINE
                    )
                )
            else:
                print(f"Collection {config.COLLECTION_NAME} already exists")
        except Exception as e:
            print(f"Error with Qdrant collection: {e}")
            print("Make sure Qdrant is running (docker-compose up -d)")
            sys.exit(1)

    def extract_audio(self, video_path: str, audio_path: str) -> bool:
        """Extract audio from video file."""
        try:
            print(f"Extracting audio from: {video_path}")
            video = VideoFileClip(video_path)

            if video.audio is None:
                print(f"Warning: No audio track found in {video_path}")
                video.close()
                return False

            video.audio.write_audiofile(
                audio_path,
                codec='libmp3lame',
                bitrate='192k',
                logger=None  # Suppress moviepy output
            )
            video.close()
            print(f"Audio saved to: {audio_path}")
            return True
        except Exception as e:
            print(f"Error extracting audio: {e}")
            return False

    def transcribe_audio(self, audio_path: str) -> List[Dict]:
        """Transcribe audio file using Whisper."""
        try:
            print(f"Transcribing audio: {audio_path}")

            # Ensure ffmpeg is available for Whisper on Windows
            # This must be done before whisper.transcribe() spawns ffmpeg subprocess
            try:
                import imageio_ffmpeg as iio
                import shutil

                ffmpeg_exe = iio.get_ffmpeg_exe()
                ffmpeg_dir = os.path.dirname(ffmpeg_exe)

                # Prepend to PATH to ensure ffmpeg is found first
                current_path = os.environ.get('PATH', '')
                if ffmpeg_dir not in current_path:
                    os.environ['PATH'] = ffmpeg_dir + os.pathsep + current_path
                    print(f"Added ffmpeg to PATH: {ffmpeg_dir}")

                # On Windows, also create a symlink/copy named 'ffmpeg.exe' if needed
                # because imageio-ffmpeg has a versioned filename
                ffmpeg_standard = os.path.join(ffmpeg_dir, 'ffmpeg.exe')
                if not os.path.exists(ffmpeg_standard) and os.path.exists(ffmpeg_exe):
                    try:
                        import shutil
                        shutil.copy2(ffmpeg_exe, ffmpeg_standard)
                        print(f"Created ffmpeg.exe copy at: {ffmpeg_standard}")
                    except Exception as copy_err:
                        print(f"Warning: Could not create ffmpeg.exe copy: {copy_err}")

                # Verify ffmpeg is now accessible
                which_ffmpeg = shutil.which('ffmpeg')
                if which_ffmpeg:
                    print(f"ffmpeg found via PATH: {which_ffmpeg}")
                else:
                    print(f"Warning: ffmpeg not found in PATH, using direct path")
                    # Monkey-patch whisper to use full path
                    import whisper.audio
                    original_load_audio = whisper.audio.load_audio
                    def patched_load_audio(file, sr=16000):
                        import subprocess
                        cmd = [
                            ffmpeg_exe,
                            "-nostdin",
                            "-threads", "0",
                            "-i", file,
                            "-f", "s16le",
                            "-ac", "1",
                            "-acodec", "pcm_s16le",
                            "-ar", str(sr),
                            "-"
                        ]
                        import numpy as np
                        out = subprocess.run(cmd, capture_output=True, check=True).stdout
                        return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0
                    whisper.audio.load_audio = patched_load_audio
                    print(f"Patched whisper.audio.load_audio to use: {ffmpeg_exe}")

            except ImportError:
                print("Warning: imageio-ffmpeg not installed, whisper may fail")

            # Verify audio file exists
            if not os.path.exists(audio_path):
                print(f"Error: Audio file not found: {audio_path}")
                return []

            result = self.whisper_model.transcribe(
                audio_path,
                language="en",
                fp16=False  # Disable fp16 for CPU compatibility
            )

            detected_lang = result.get("language", "unknown")
            print(f"Detected language: {detected_lang}")

            transcription_segments = []
            full_text = []

            for segment in result.get("segments", []):
                transcription_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip()
                })
                full_text.append(segment['text'].strip())

            print(f"Transcription complete: {len(transcription_segments)} segments")
            return transcription_segments
        except Exception as e:
            print(f"Error transcribing audio: {e}")
            return []

    def save_transcript(self, transcript_segments: List[Dict], output_path: str):
        """Save transcript to text file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                for seg in transcript_segments:
                    timestamp = f"[{seg['start']:.2f} - {seg['end']:.2f}]"
                    f.write(f"{timestamp} {seg['text']}\n")
            print(f"Transcript saved to: {output_path}")
        except Exception as e:
            print(f"Error saving transcript: {e}")

    def process_and_store(self, video_name: str, transcript_segments: List[Dict]):
        """Process transcript and store in Qdrant."""
        try:
            # Combine all segments into full text
            full_text = " ".join([seg['text'] for seg in transcript_segments])

            # Split text into chunks
            chunks = self.text_splitter.split_text(full_text)
            print(f"Split transcript into {len(chunks)} chunks")

            # Generate embeddings and store in Qdrant
            points = []
            for idx, chunk in enumerate(chunks):
                # Generate embedding
                embedding = self.embeddings.embed_query(chunk)

                # Create point
                point = PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={
                        "source": video_name,
                        "type": "video_transcript",
                        "chunk_index": idx,
                        "text": chunk
                    }
                )
                points.append(point)

            # Upload to Qdrant
            self.qdrant_client.upsert(
                collection_name=config.COLLECTION_NAME,
                points=points
            )
            print(f"Stored {len(points)} chunks in Qdrant")

        except Exception as e:
            print(f"Error processing and storing transcript: {e}")

    def process_video(self, video_path: str):
        """Complete pipeline: extract audio, transcribe, and store."""
        video_path = Path(video_path)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return

        print(f"\n{'='*60}")
        print(f"Processing video: {video_path.name}")
        print(f"{'='*60}")

        # Prepare output paths
        audio_filename = video_path.stem + ".mp3"
        audio_path = os.path.join(config.AUDIO_DIR, audio_filename)

        transcript_filename = video_path.stem + ".txt"
        transcript_path = os.path.join(config.TRANSCRIPTS_DIR, transcript_filename)

        # Step 1: Extract audio
        if not self.extract_audio(str(video_path), audio_path):
            print("Failed to extract audio. Skipping video.")
            return

        # Step 2: Transcribe audio
        transcript_segments = self.transcribe_audio(audio_path)
        if not transcript_segments:
            print("Failed to transcribe audio. Skipping video.")
            return

        # Step 3: Save transcript
        self.save_transcript(transcript_segments, transcript_path)

        # Step 4: Process and store in vector DB
        self.process_and_store(video_path.name, transcript_segments)

        print(f"\n✓ Successfully processed: {video_path.name}\n")


def main():
    """Main function to process all videos in data directory."""
    # Get all video files
    data_path = Path(config.DATA_DIR)
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    video_files = [
        f for f in data_path.iterdir()
        if f.is_file() and f.suffix.lower() in video_extensions
    ]

    if not video_files:
        print(f"No video files found in {config.DATA_DIR}")
        return

    print(f"Found {len(video_files)} video file(s) to process")

    # Initialize processor
    processor = VideoProcessor()

    # Process each video
    for video_file in video_files:
        try:
            processor.process_video(str(video_file))
        except Exception as e:
            print(f"Error processing {video_file.name}: {e}")
            continue

    print("\n" + "="*60)
    print("All videos processed!")
    print("="*60)


if __name__ == "__main__":
    main()

