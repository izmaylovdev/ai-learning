import os
import sys
from pathlib import Path
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

from moviepy import VideoFileClip
from faster_whisper import WhisperModel
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
        self.whisper_model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE
        )

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
            segments, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=5,
                language="en"
            )

            print(f"Detected language: {info.language} (probability: {info.language_probability:.2f})")

            transcription_segments = []
            full_text = []

            for segment in segments:
                transcription_segments.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text.strip()
                })
                full_text.append(segment.text.strip())

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

